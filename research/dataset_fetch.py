#!/usr/bin/env python3
"""Selective downloading of dataset pairs: download ONLY the needed tracks.

For each row of `corpus/dataset_plan.jsonl` with status `matched`:
  1. .torrent of the chosen original → torrents/{topic}.torrent (cache);
  2. add to Transmission paused → the torrent's file list;
  3. match files against the master's track list (number + title + pieces);
  4. files-unwanted on everything extra (lives/bonuses/scans, other albums of
     the discography) → start;
  5. journal entry in dataset/state.jsonl.

Modes:
  --dry-run   download nothing: show what would be picked (files/size)
  --go        actually start the downloads (the --max-gb limit is respected)
  --status    summary over state.jsonl

Examples:
  python3 scripts/dataset_fetch.py --dry-run --limit 5
  python3 scripts/dataset_fetch.py --go --max-gb 20
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import planner, workspace  # noqa: E402
from rnext.rtclient import FORUM, RTClient  # noqa: E402
from rnext.transm import Transm  # noqa: E402

AUDIO_EXT = (".flac", ".ape", ".wv", ".wav", ".alac", ".m4a", ".aiff", ".aif")
_NUM = re.compile(r"(?:^|[/_\-\s])(\d{1,3})\s*[-._)\s]\s*(?=\D)", re.UNICODE)
_DISC = re.compile(r"(?:\bcd|\bdisc|\bdisk)[\s_-]?(\d)", re.IGNORECASE)


# --- paths -------------------------------------------------------------------
def corpus() -> Path:
    return workspace.dir_of("corpus")


def dataset_dir() -> Path:
    p = workspace.data_root() / "dataset"
    for sub in ("raw", "pairs"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def torrent_path(tid: int) -> Path:
    return workspace.dir_of("torrents") / f"{tid}.torrent"


def state_file() -> Path:
    return dataset_dir() / "state.jsonl"


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def append_state(rec: dict) -> None:
    with open(state_file(), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def state_map() -> dict[str, dict]:
    from rnext import sides as S
    return S.merged_state(state_file())


# --- track matching ---------------------------------------------------
def file_track_no(name: str) -> int | None:
    base = name.rsplit("/", 1)[-1]
    m = re.match(r"^(\d{1,3})[\s._\-)]", base)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{2})\s*[-._]\s*[^\d]", base)
    return int(m.group(1)) if m else None


def disc_no(name: str) -> int | None:
    m = _DISC.search(name)
    if m:
        return int(m.group(1))
    m = re.match(r"^(?:cd|disc|disk)[\s_-]?(\d)", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def album_dir_filter(files: list[dict], album: str | None) -> tuple[list[dict], str | None]:
    """For a discography torrent keep only the files of the needed album (by folder).

    Compare only the LAST path component (the album folder name): otherwise a
    "Covers" subfolder inside "…/Silver Side Up/" would win via its parent.
    Folders without audio (scans) are not candidates.
    """
    if not album:
        return files, None
    album_t = planner.tokens(album)
    dirs: dict[str, list[dict]] = {}
    for f in files:
        d = f["name"].rsplit("/", 1)[0] if "/" in f["name"] else ""
        dirs.setdefault(d, []).append(f)
    if len(dirs) <= 1:
        return files, None
    best = None
    for d, fs in dirs.items():
        if not any(f["name"].lower().endswith(AUDIO_EXT) for f in fs):
            continue
        last = d.rsplit("/", 1)[-1]
        cont = planner.containment(album_t, planner.tokens(last))
        if best is None or cont > best[0]:
            best = (cont, d, fs)
    if best and best[0] >= 0.5:
        return best[2], best[1]
    # Multi-folder torrent (compilation/discography) without the album in it:
    # return EMPTY, otherwise the whole collection would be downloaded (for a
    # JP compilation that is 3.2 GB instead of the needed 0.5 GB) and the pair
    # would honestly end up as no_match.
    if len(dirs) > 1:
        return [], None
    return files, None


def match_tracks(master_tracks: list[dict], files: list[dict],
                 album: str | None = None) -> dict:
    """Greedy match "master track → torrent file". Returns wanted/skipped."""
    all_files = files                      # file indexes are numbered over the FULL list
    sel, album_dir = album_dir_filter(all_files, album)
    audio = [f for f in sel if f["name"].lower().endswith(AUDIO_EXT)]
    cues = [f for f in sel if f["name"].lower().endswith(".cue")]
    images = [f for f in audio if (f.get("length") or 0) > 120_000_000]
    if images and len(audio) <= max(2, len(cues) + 1):
        # image + cue: take it whole, cut tracks by cue after the download
        wanted = [f["index"] for f in audio] + [f["index"] for f in cues]
        return {"mode": "image", "wanted": wanted, "matched": len(master_tracks),
                "skipped": [], "album_dir": album_dir,
                "size": sum(all_files[i].get("length") or 0 for i in wanted)}

    scored = []
    for mt in master_tracks:
        tt = planner.tokens(mt.get("s") or mt.get("title") or "")
        best, best_s = None, 0.0
        for f in audio:
            ft = planner.tokens(f["name"].rsplit("/", 1)[-1])
            cont = planner.containment(tt, ft) if tt else 0.0
            s = cont * 3.0
            fn, mn = file_track_no(f["name"]), mt.get("n")
            if fn is not None and mn is not None:
                if fn == mn:
                    s += 1.0
                elif fn % 100 == mn % 100:
                    s += 0.5
            dn = disc_no(f["name"])
            if dn:
                s += 0.1
            if s > best_s:
                best, best_s = f, s
        scored.append((best_s, mt, best))

    used: set[int] = set()
    wanted: list[int] = []
    skipped: list[str] = []
    for s, mt, f in sorted(scored, key=lambda x: -x[0]):
        if f is None or s < 1.2 or f["index"] in used:
            skipped.append(f"{mt.get('n')}. {mt.get('s')}")
            continue
        used.add(f["index"])
        wanted.append(f["index"])
    return {
        "mode": "tracks",
        "wanted": sorted(wanted),
        "matched": len(wanted),
        "skipped": skipped,
        "album_dir": album_dir,
        "size": sum(all_files[i].get("length") or 0 for i in wanted),
    }


# --- transmission -----------------------------------------------------------
def ensure_torrent(client: RTClient, tid: int) -> str:
    p = torrent_path(tid)
    if p.is_file() and p.stat().st_size > 100:
        return str(p)
    client.download_torrent(FORUM + f"dl.php?t={tid}", str(p))
    return str(p)


def add_paused(t: Transm, path: str, dl_dir: str) -> dict:
    return t.add_torrent_file(path, dl_dir, stopped=True)


def torrent_details(t: Transm, tid: int) -> dict | None:
    for tor in t.torrents(["id", "hashString", "name", "files", "wanted", "fileStats",
                           "totalSize", "percentDone", "status", "downloadDir",
                           "leftUntilDone"]):
        if tor["id"] == tid:
            tor["files"] = [{**f, "index": i} for i, f in enumerate(tor.get("files") or [])]
            return tor
    return None


def set_unwanted(t: Transm, tid: int, indices: list[int]) -> None:
    t._call("torrent-set", {"ids": [tid], "files-unwanted": indices})


def start(t: Transm, tid: int) -> None:
    t._call("torrent-start", {"ids": [tid]})


# --- main pass --------------------------------------------------------
def plan_rows(args=None) -> list[dict]:
    p = getattr(args, "plan", None)
    p = Path(p) if p else (corpus() / "dataset_plan.jsonl")
    return [r for r in read_jsonl(p) if r["status"].startswith("matched")]


def sort_key(r: dict) -> tuple:
    """Priority: low master DR (more crushed) → smaller size → seeders."""
    md = ((r["release"].get("master_dr") or {}).get("official_dr") or 99)
    size = r["chosen"]["size_bytes"] or 0
    return (md, size, -(r["chosen"]["seeders"] or 0))


def side_setup(t: Transm, client: RTClient, topic_id: int, dl_dir: str,
               master_tracks: list[dict], album: str | None,
               min_matched: float) -> dict:
    """Add the torrent paused, pick the needed files, start it.

    Returns {ok, torrent_tid, mode, wanted_bytes, matched, of, album_dir, error}.
    """
    tp = ensure_torrent(client, topic_id)
    rec = add_paused(t, tp, dl_dir)
    duplicate = "torrent-duplicate" in rec
    added = rec.get("torrent-added") or rec.get("torrent-duplicate") or {}
    ttid = added.get("id")
    if ttid is None:
        return {"ok": False, "error": "add failed"}
    time.sleep(1.0)
    det = torrent_details(t, ttid)
    if not det:
        return {"ok": False, "torrent_tid": ttid, "error": "no details"}
    files = det["files"]
    mt = match_tracks(master_tracks, files, album=album)
    frac = mt["matched"] / max(1, len(master_tracks))
    if frac < min_matched:
        t._call("torrent-remove", {"ids": [ttid], "delete-local-data": True})
        return {"ok": False, "torrent_tid": ttid, "mode": mt["mode"],
                "matched": mt["matched"], "of": len(master_tracks),
                "album_dir": mt.get("album_dir"), "error": "files_mismatch"}
    want = set(mt["wanted"])
    if duplicate:
        # the same topic is already running for another pair — union the wanted files
        prev = det.get("wanted") or []
        want |= {i for i, w in enumerate(prev) if w and i < len(files)}
    unwanted = [f["index"] for f in files if f["index"] not in want]
    t._call("torrent-set", {"ids": [ttid], "files-wanted": sorted(want),
                            "files-unwanted": unwanted})
    start(t, ttid)
    stats = det.get("fileStats") or []
    new_bytes = 0
    for i in sorted(want):
        ln = files[i].get("length") or 0
        done = (stats[i].get("bytesCompleted") if i < len(stats) else 0) or 0
        new_bytes += max(0, ln - done)
    return {"ok": True, "torrent_tid": ttid, "mode": mt["mode"],
            "wanted": sorted(want),
            "wanted_names": [files[i]["name"] for i in sorted(want)],
            "wanted_bytes": sum((files[i].get("length") or 0) for i in want),
            "new_bytes": new_bytes,
            "matched": mt["matched"], "of": len(master_tracks),
            "album_dir": mt.get("album_dir"), "all_bytes": det.get("totalSize"),
            "name": det.get("name")}


def backfill_masters(args) -> int:
    """Backfill master sides for pairs where it did not stick the first time."""
    t = Transm()
    client = RTClient()
    st = read_jsonl(state_file())
    seen: dict[str, dict] = {}
    for r in st:
        seen[r["key"]] = r
    todo = [r for r in seen.values() if r.get("state") == "downloading"
            and not (isinstance(r.get("master"), dict) and r["master"].get("ok"))]
    plan = {r["key"]: r for r in plan_rows(args)}
    if args.limit:
        todo = todo[:args.limit]
    print(f"pairs without a master side: {len(todo)}")
    added = 0
    for r in todo:
        src = plan.get(r["key"])
        if not src:
            print(f"  [skip] {r['key']}: not in the plan")
            continue
        tracks = (src["release"].get("tracks") or [])
        if not tracks:
            tracks = (src["release"].get("master_dr") or {}).get("tracks") or []
        base = r.get("dir") or str(dataset_dir() / "raw" / r["key"].replace("/", "__"))
        try:
            m = side_setup(t, client, src["restored_topic_id"], base + "/master",
                           tracks, src["release"].get("album"), args.min_matched)
        except Exception as e:                        # noqa: BLE001
            m = {"ok": False, "error": f"{type(e).__name__}: {e}"[:120]}
        if m.get("ok"):
            added += 1
            print(f"  [ok] {r['key']}: master {m['wanted_bytes'] / 1e6:.0f}MB "
                  f"({m['matched']}/{m['of']})")
        else:
            print(f"  [fail] {r['key']}: {m.get('error')}")
        # update the journal (the last line per key wins)
        append_state({"key": r["key"], "state": "downloading", "dir": r.get("dir"),
                      "orig_topic_id": r.get("orig_topic_id"),
                      "master_topic_id": r.get("master_topic_id"),
                      "artist": r.get("artist"), "album": r.get("album"),
                      "year": r.get("year"), "master_dr": r.get("master_dr"),
                      "artist": r.get("artist"), "album": r.get("album"),
                      "orig": r.get("orig"), "master": m})
    print(f"master sides added: {added}")
    return 0




def repair_masters(args) -> int:
    """Recompute torrent wanted-sets: UNION over live pairs + removal of "zombies".

    One discography topic serves several pairs while a torrent has a single
    wanted-set — so we compute the union over all LIVE (downloading) pairs.
    Torrents referenced by no live pair (pairs postponed for space or dropped)
    are removed from the session: otherwise they silently download into deleted
    directories, raw comes back and disk space leaks.
    """
    t = Transm()
    plan = {r["key"]: r for r in plan_rows(args)}
    st = state_map()
    # downloadDir is required: it tells OUR torrents from the user's (the user
    # has their own torrents in Transmission, which must not be touched)
    cur = {tor["id"]: tor for tor in t.torrents(
        ["id", "name", "files", "wanted", "totalSize", "status", "downloadDir",
         "peersConnected", "leftUntilDone"])}

    want_by_torrent: dict[int, set[int]] = {}
    label: dict[int, str] = {}
    for key, r in st.items():
        src = plan.get(key)
        if not src or r.get("state") != "downloading":
            continue
        tracks = src["release"].get("tracks") or []
        if not tracks:
            tracks = (src["release"].get("master_dr") or {}).get("tracks") or []
        for side in ("orig", "master"):
            rec = r.get(side) if isinstance(r.get(side), dict) else None
            tid = (rec or {}).get("torrent_tid")
            if tid is None or tid not in cur:
                continue
            files = [{**f, "index": i} for i, f in enumerate(cur[tid].get("files") or [])]
            mt = match_tracks(tracks, files, album=src["release"].get("album"))
            if not mt["wanted"]:
                continue
            want_by_torrent.setdefault(tid, set()).update(mt["wanted"])
            label.setdefault(tid, f"{cur[tid]['name'][:34]} ({side} {key})")

    fixed = 0
    for tid, want in want_by_torrent.items():
        files = [{**f, "index": i} for i, f in enumerate(cur[tid].get("files") or [])]
        unw = [f["index"] for f in files if f["index"] not in want]
        t._call("torrent-set", {"ids": [tid], "files-wanted": sorted(want),
                                "files-unwanted": unw})
        t._call("torrent-verify", {"ids": [tid]})
        start(t, tid)
        size = sum(files[i].get("length") or 0 for i in want)
        fixed += 1
        print(f"  [fix] t={tid}: wanted {len(want)} of {len(files)} ({size / 1e6:.0f} MB) "
              f"— {label.get(tid, '')}")

    # "zombies": our torrents referenced by no live pair
    zombies = [tid for tid, x in cur.items()
               if "/dataset/raw/" in (x.get("downloadDir") or "")
               and tid not in want_by_torrent and any(x.get("wanted") or [])]
    for tid in zombies:
        name = (cur[tid].get("name") or "")[:40]
        nw = sum(cur[tid].get("wanted") or [])
        # no data (raw dropped), no live pairs — no point keeping it in the session:
        # the .torrent stays cached and is re-added if the pair comes back
        t._call("torrent-remove", {"ids": [tid], "delete-local-data": False})
        print(f"  [zombie] t={tid} {name}: removed from the session ({nw} files wanted)")
    print(f"torrents fixed: {fixed}, zombies removed: {len(zombies)}")
    return 0


def resolve_dirs(args) -> int:
    """Write the ACTUAL side directories into the journal.

    A shared discography topic downloads into the directory of the pair that
    added it first, so another pair's "base/orig" may not exist. Here we ask
    Transmission for each torrent's downloadDir and store it in the pair record.
    """
    t = Transm()
    dirs = {tor["id"]: tor.get("downloadDir") for tor in t.torrents(["id", "downloadDir"])}
    st = state_map()
    fixed = 0
    for key, r in st.items():
        if r.get("state") != "downloading":
            continue
        upd = dict(r)
        changed = False
        for side in ("orig", "master"):
            rec = r.get(side) if isinstance(r.get(side), dict) else None
            tid = (rec or {}).get("torrent_tid")
            d = dirs.get(tid) if tid is not None else None
            if d and rec.get("dir") != d:
                rec = {**rec, "dir": d}
                upd[side] = rec
                changed = True
        if changed:
            append_state(upd)
            fixed += 1
    print(f"directories updated for {fixed} pairs")
    return 0


def files_present(srec: dict | None, root: Path | None) -> bool:
    """Whether all needed files of the side are on disk (directory self-heal)."""
    from rnext import sides as S
    if not isinstance(srec, dict):
        return False
    base = (root.parent if root else Path("."))
    pres = S.side_presence(srec, "orig" if root and root.name == "orig" else "master", base)
    return bool(pres["wanted"]) and pres["complete"]


def refetch_missing(args) -> int:
    """Re-download sides whose raw was deleted after extraction (for re-evaluation)."""
    t = Transm()
    client = RTClient()
    st = state_map()
    plan = {r["key"]: r for r in plan_rows(args)}
    todo = [r for r in st.values() if r.get("state") == "downloading" and r["key"] in plan]
    if args.limit:
        todo = todo[:args.limit]
    print(f"pairs to check raw for: {len(todo)}")
    n = 0
    for r in todo:
        src = plan[r["key"]]
        tracks = src["release"].get("tracks") or []
        base = Path(r.get("dir") or (dataset_dir() / "raw" / r["key"].replace("/", "__")))
        for side, topic in (("orig", r.get("orig_topic_id")), ("master", r.get("master_topic_id"))):
            srec = r.get(side) if isinstance(r.get(side), dict) else None
            root = Path((srec or {}).get("dir") or (base / side))
            if files_present(srec, root):
                continue
            if topic is None:
                continue
            dl = str(root)
            try:
                m = side_setup(t, client, topic, dl, tracks, src["release"].get("album"),
                               args.min_matched)
            except Exception as e:                     # noqa: BLE001
                m = {"ok": False, "error": f"{type(e).__name__}: {e}"[:100]}
            print(f"  {r['key']}/{side}: "
                  + (f"re-downloading {m['wanted_bytes'] / 1e6:.0f}MB" if m.get("ok")
                     else f"error {m.get('error')}"))
            if m.get("ok"):
                srec = {**(srec or {}), **{k: m[k] for k in
                                           ("torrent_tid", "mode", "wanted", "wanted_names",
                                            "wanted_bytes", "matched", "of", "album_dir")},
                        "dir": dl}
                r = {**r, side: srec}
                append_state(r)
                n += 1
    print(f"sides re-downloaded: {n}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-gb", type=float, default=15.0)
    ap.add_argument("--min-matched", type=float, default=0.6,
                    help="fraction of matched tracks below which the pair is skipped")
    ap.add_argument("--plan", default=None, help="alternative plan file")
    ap.add_argument("--no-master", action="store_true",
                    help="do not download the master restoration (original only)")
    ap.add_argument("--masters-only", action="store_true",
                    help="backfill master sides for already started pairs")
    ap.add_argument("--repair", action="store_true",
                    help="recompute wanted-sets for all our torrents")
    ap.add_argument("--resolve-dirs", action="store_true",
                    help="write the actual side directories into the journal")
    ap.add_argument("--refetch", action="store_true",
                    help="re-download sides whose raw was deleted after extraction")
    args = ap.parse_args()

    if args.status:
        st = read_jsonl(state_file())
        from collections import Counter
        print("pairs in journal:", len(st), dict(Counter(r.get("state") for r in st)))
        got = sum((r.get("wanted_bytes") or 0) + (r.get("master", {}).get("wanted_bytes") or 0)
                  for r in st)
        print(f"total selected for download: {got / 1e9:.2f} GB")
        return 0

    if not (args.dry_run or args.go or args.repair or args.status or args.masters_only
            or args.resolve_dirs or args.refetch):
        ap.error("need --dry-run, --go, --repair, --masters-only, --resolve-dirs or --status")

    if args.repair:
        return repair_masters(args)
    if args.resolve_dirs:
        return resolve_dirs(args)
    if args.refetch:
        return refetch_missing(args)
    if args.masters_only:
        return backfill_masters(args)

    if args.masters_only:
        return backfill_masters(args)

    rows = sorted(plan_rows(args), key=sort_key)
    done = state_map()
    rows = [r for r in rows if done.get(r["key"], {}).get("state") not in
            ("downloading", "done", "extracting")]
    print(f"download candidates: {len(rows)}")

    t = None if args.dry_run else Transm()
    client = RTClient()
    budget_left = workspace.remaining_bytes() if not args.dry_run else int(args.max_gb * 1e9)
    reserve = 2 * 1024 ** 3          # do not eat the flash drive down to zero
    spent = 0
    processed = 0
    for r in rows:
        if processed >= args.limit:
            break
        tid = r["chosen"]["topic_id"]
        key = r["key"]
        rel = r["release"]
        md = rel.get("master_dr") or {}

        if args.dry_run:
            try:
                tp = ensure_torrent(client, tid)
            except Exception as e:                    # noqa: BLE001
                print(f"[torrent-fail] {key} t={tid}: {type(e).__name__} {str(e)[:60]}")
                append_state({"key": key, "state": "error", "reason": f"torrent: {e}"[:200]})
                continue
            sz = r["chosen"]["size_bytes"] or 0
            print(f"{key} orig t={tid} {sz / 1e6:.0f}MB | master t={r['restored_topic_id']} "
                  f"DR{md.get('official_dr')} | {r['artist'][:22]} | {rel['album'][:34]}")
            append_state({"key": key, "state": "dry", "torrent": tp, "size_upper": sz})
            processed += 1
            continue

        # 1) original
        base = str(dataset_dir() / "raw" / key.replace("/", "__"))
        try:
            o = side_setup(t, client, tid, base + "/orig", rel["tracks"], rel["album"],
                           args.min_matched)
        except Exception as e:                        # noqa: BLE001
            print(f"[fail] {key} orig t={tid}: {type(e).__name__} {str(e)[:80]}")
            append_state({"key": key, "state": "error", "reason": f"orig: {e}"[:200]})
            continue
        if not o["ok"]:
            append_state({"key": key, "state": "no_match", **{k: o[k] for k in
                          ("torrent_tid", "matched", "of", "error") if k in o}})
            print(f"[skip] {key}: original did not match ({o.get('matched')}/{o.get('of')}) "
                  f"{rel['album'][:40]}")
            processed += 1
            continue
        need_new = o.get("new_bytes", o["wanted_bytes"])
        if need_new + spent > int(args.max_gb * 1e9) or \
                need_new + reserve > budget_left:
            print(f"[budget] {key}: new {need_new / 1e6:.0f}MB "
                  f"(whole topic {o['wanted_bytes'] / 1e6:.0f}MB) — stopping on budget")
            t._call("torrent-remove", {"ids": [o["torrent_tid"]], "delete-local-data": True})
            break

        # 2) master restoration (needed for features/band deltas and listening checks)
        mrec = None
        if not args.no_master:
            try:
                mrec = side_setup(t, client, r["restored_topic_id"], base + "/master",
                                  rel["tracks"], rel["album"], args.min_matched)
            except Exception as e:                    # noqa: BLE001
                mrec = {"ok": False, "error": f"{type(e).__name__}: {e}"[:120]}
            if mrec and not mrec["ok"]:
                print(f"[warn] {key}: master side did not stick ({mrec.get('error')}) — "
                      f"keeping a single side")

        spent += o.get("new_bytes", o["wanted_bytes"]) + \
            ((mrec or {}).get("new_bytes", (mrec or {}).get("wanted_bytes")) or 0)
        processed += 1
        append_state({
            "key": key, "state": "downloading", "dir": base,
            "orig_topic_id": tid, "master_topic_id": r["restored_topic_id"],
            "artist": r["artist"], "album": rel["album"], "year": rel["year"],
            "master_dr": md or None,
            "orig": {**{k: o.get(k) for k in ("torrent_tid", "mode", "wanted", "wanted_names",
                                              "wanted_bytes", "matched", "of", "album_dir",
                                              "name")}, "dir": base + "/orig"},
            "master": ({**{k: mrec.get(k) for k in ("torrent_tid", "mode", "wanted",
                         "wanted_names", "wanted_bytes", "matched", "of", "album_dir",
                         "name")}, "dir": base + "/master"}
                       if mrec and mrec.get("ok") else mrec),
        })
        mtxt = (f"+ master {mrec['wanted_bytes'] / 1e6:.0f}MB"
                if mrec and mrec.get("ok") else "(no master)")
        print(f"[start] {key} '{o.get('name', '')[:38]}' {o['mode']} "
              f"{o['matched']}/{o['of']} tracks, {o['wanted_bytes'] / 1e6:.0f}MB {mtxt}")
    print(f"total: processed {processed}, selected {spent / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
