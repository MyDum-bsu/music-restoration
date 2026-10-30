#!/usr/bin/env python3
"""Building the dataset plan "original → master restoration" for two catalogs [RM].

Stages (idempotent, everything is cached on the flash drive):
  topics  — master topic pages: releases[] + the master's DR table (post-hoc),
            cache in corpus/restored/{tid}.json
  match   — a pool of original candidates per artist (1 search instead of 1 per
            release), local matching by title/year/catalog number
  plan    — corpus/dataset_plan.jsonl (rewritten) + report

Examples:
  python3 scripts/dataset_plan.py topics --author 5618698,3048819
  python3 scripts/dataset_plan.py match
  python3 scripts/dataset_plan.py report
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import planner  # noqa: E402
from rnext import workspace  # noqa: E402
from rnext.rtclient import FORUM, RTClient  # noqa: E402
from rnext.rttopic import fetch_topic  # noqa: E402

AUTHORS = {"5618698": "Lisnoy Yozhyg", "3048819": "Alias Divas"}


# --- paths -------------------------------------------------------------------
def corpus() -> Path:
    p = workspace.dir_of("corpus")
    (p / "restored").mkdir(exist_ok=True)
    (p / "search").mkdir(exist_ok=True)
    return p


def authors_file(pid: str) -> Path:
    return corpus() / "authors" / f"{pid}.jsonl"


def restored_file(tid: int) -> Path:
    return corpus() / "restored" / f"{tid}.json"


def search_cache(query: str) -> Path:
    h = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
    return corpus() / "search" / f"{h}.jsonl"


# --- cache read/write ------------------------------------------------------
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


def write_json(p: Path, obj) -> None:
    # tmp name carries the pid: parallel workers must not clobber each other's tmp
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def write_jsonl(p: Path, rows: list[dict]) -> None:
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


# --- stage 1: topic pages ---------------------------------------------------
def is_restoration_topic(row: dict) -> bool:
    """Restoration topics only: [RM] tag or restored/declipped, and not a lossy torrent."""
    title = row.get("title") or ""
    if "MP3" in title.upper() or "lossy" in (row.get("forum") or "").lower():
        return False
    return bool(planner.OWN_MARK.search(title))


def step_topics(args) -> int:
    pids = [p.strip() for p in args.author.split(",") if p.strip()]
    topics = []
    for pid in pids:
        rows = read_jsonl(authors_file(pid))
        if not rows:
            print(f"[skip] no author catalog {pid} — run the poster_topics scrape first")
            continue
        kept = [r for r in rows if is_restoration_topic(r)]
        print(f"author {pid}: {len(rows)} topics, {len(kept)} restorations")
        topics.extend(kept)
    print(f"topics to parse: {len(topics)}")
    client = RTClient()
    done = fresh = failed = 0
    for i, row in enumerate(topics):
        tid = row["topic_id"]
        dst = restored_file(tid)
        if dst.is_file() and dst.stat().st_size > 200 and not args.refresh:
            done += 1
            continue
        if args.limit and fresh >= args.limit:
            break
        try:
            d = fetch_topic(client, str(tid))
        except Exception as e:                       # noqa: BLE001 — the network is moody
            failed += 1
            print(f"[fail] {tid}: {type(e).__name__} {str(e)[:90]}")
            time.sleep(2.0)
            continue
        d["author_pid"] = row.get("poster_pid")
        d["author"] = row.get("poster")
        d["topic_size_bytes"] = row.get("size_bytes")
        d["topic_seeders"] = row.get("seeders")
        for rel in d.get("releases", []):
            rel.pop("tracks_raw", None)
        try:
            write_json(dst, d)
        except OSError as e:                         # ntfs3 sometimes yields ENOENT
            failed += 1
            print(f"[write-fail] {tid}: {e}")
            continue
        fresh += 1
        if i % 10 == 0:
            print(f"  [{i}/{len(topics)}] {tid} releases {len(d.get('releases', []))} "
                  f"dr-tables {sum(1 for r in d.get('releases', []) if r.get('master_dr'))}")
        time.sleep(args.pause)
    print(f"topics: parsed {done + fresh} ({fresh} new), errors {failed}")


# --- stage 2: original search ----------------------------------------------
def search_pool(client: RTClient, query: str, pages: int = 2, pause: float = 1.5,
                refresh: bool = False) -> list[dict]:
    """rutracker search by query string with an on-disk cache."""
    p = search_cache(query)
    if p.is_file() and not refresh:
        return read_jsonl(p)
    rows: dict[int, dict] = {}
    for page in range(pages):
        url = FORUM + f"tracker.php?nm={__import__('requests').utils.quote(query)}&start={50 * page}"
        r = client.get(url)
        from bs4 import BeautifulSoup
        from rnext.rtscrape import ROW_SEL, _row_dict
        trs = BeautifulSoup(r.text, "html.parser").select(ROW_SEL)
        if not trs:
            break
        for tr in trs:
            d = _row_dict(tr)
            if d["topic_id"] and d["topic_id"] not in rows:
                rows[d["topic_id"]] = d
        time.sleep(pause)
    out = list(rows.values())
    write_jsonl(p, out)
    return out


def pool_query(d: dict, artist: str) -> str:
    """Pool query: the source link from the master topic ("source for processing"),
    otherwise the artist name."""
    from urllib.parse import parse_qs, urlparse
    for u in d.get("source_search_urls") or []:
        q = parse_qs(urlparse(u).query)
        nm = (q.get("nm") or [""])[0].replace("+", " ").strip()
        if not nm or len(nm) < 3:
            continue
        if re.search(r"restored|declipped|\[RM\]|RM\b", nm, re.IGNORECASE):
            continue        # a link to the master's own tags, not to the original
        return nm
    return artist


def match_release(client: RTClient, d: dict, rel: dict, artist_tokens: list[str],
                  artist: str, author_pid, pool: list[dict], q: str,
                  pause: float) -> tuple[dict | None, str, list[dict]]:
    """Picking an original for one release: pool → targeted search → discography."""
    album_tokens = planner.tokens(rel["album"])
    codes = planner.edition_codes(rel.get("source_edition"))
    cands: list[dict] = []
    for row in pool:
        s = planner.score_candidate(row, artist_tokens, album_tokens, rel.get("year"),
                                    codes, author_pid)
        if s["score"] > -900:
            cands.append({**row, **s})

    chosen = planner.choose(cands, kind="album")
    src = "pool"
    if chosen is None:
        # targeted "artist + album" search — only for pool misses
        try:
            tpool = search_pool(client, f"{artist} {rel['album']}", pages=1, pause=pause)
        except Exception:                             # noqa: BLE001
            tpool = []
        known = {c["topic_id"] for c in cands}
        for row in tpool:
            if row["topic_id"] in known:
                continue
            s = planner.score_candidate(row, artist_tokens, album_tokens, rel.get("year"),
                                        codes, author_pid)
            if s["score"] > -900:
                cands.append({**row, **s})
        chosen = planner.choose(cands, kind="album")
        src = "targeted" if chosen else "pool"

    if chosen is None:
        # the artist's CD discography: the needed disc is picked via the torrent's file list
        g = planner.choose(cands, kind="graphy", threshold=planner.GRAPHY_THRESHOLD)
        if g is not None:
            chosen, src = g, "discography"
    return chosen, src, cands


def step_match(args) -> int:
    restored = sorted(corpus().glob("restored/*.json"))
    out = corpus() / (args.out or "dataset_plan.jsonl")
    if args.rewrite and out.is_file():
        out.unlink()          # otherwise append mode would add a second copy of the plan
    prev = read_jsonl(out)
    done_topics = {r["restored_topic_id"] for r in prev}
    if args.rewrite:
        prev, done_topics = [], set()
    counts = {"matched": 0, "matched_graphy": 0, "weak": 0, "no_original": 0}
    for r in prev:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"topics parsed: {len(restored)}, already in plan: {len(done_topics)} ({len(prev)} rows)")

    client = RTClient()
    pool_cache: dict[str, list[dict]] = {}
    fresh_topics = 0
    with open(out, "a", encoding="utf-8") as fh:
        for k, path in enumerate(restored):
            d = json.loads(path.read_text(encoding="utf-8"))
            if not planner.OWN_MARK.search(d.get("title") or ""):
                continue
            if d["topic_id"] in done_topics:
                continue
            rt = planner.parse_restore_title(d["title"])
            author_pid = d.get("author_pid")
            q = pool_query(d, rt["artist"])
            if q not in pool_cache:
                try:
                    pool_cache[q] = search_pool(client, q, pages=args.pages,
                                                pause=args.pause,
                                                refresh=args.refresh_search)
                except Exception as e:                # noqa: BLE001
                    print(f"[fail-search] {q}: {type(e).__name__} {str(e)[:80]}")
                    pool_cache[q] = []
            pool = pool_cache[q]
            for idx, rel in enumerate(d.get("releases", [])):
                album = (rel.get("album") or "").strip()
                if not album or album.lower() in {"mediainfo", "info"}:
                    continue
                if not rel.get("tracks") and not rel.get("master_dr"):
                    continue       # the master did not parse the release — no pair
                chosen, src, cands = match_release(client, d, rel, rt["artist_tokens"],
                                                   rt["artist"], author_pid, pool, q,
                                                   args.pause)
                status = "matched" if chosen else ("weak" if cands else "no_original")
                if chosen is not None and chosen.get("kind") == "graphy":
                    status = "matched_graphy"
                counts[status] = counts.get(status, 0) + 1
                row = {
                    "key": f"{d['topic_id']}/rel{idx}",
                    "author_pid": author_pid,
                    "author": d.get("author"),
                    "restored_topic_id": d["topic_id"],
                    "restore_title": d["title"],
                    "artist": rt["artist"],
                    "genre": rt["genre"],
                    "release": {
                        "year": rel.get("year"),
                        "album": album,
                        "n_tracks": len(rel.get("tracks") or []),
                        "tracks": rel.get("tracks") or [],
                        "source_edition": rel.get("source_edition"),
                        "processing": rel.get("processing"),
                        "master_dr": rel.get("master_dr"),
                    },
                    "status": status,
                    "pool_query": q,
                    "match_source": src,
                    "chosen": ({"topic_id": chosen["topic_id"], "title": chosen["title"],
                                "size_bytes": chosen.get("size_bytes"),
                                "seeders": chosen.get("seeders"),
                                "score": chosen["score"], "flags": chosen["flags"],
                                "kind": chosen.get("kind"),
                                "code_hit": chosen.get("code_hit")} if chosen else None),
                    "candidates": [{"topic_id": c["topic_id"], "title": c["title"][:160],
                                    "size_bytes": c.get("size_bytes"),
                                    "seeders": c.get("seeders"), "score": c["score"],
                                    "flags": c["flags"]}
                                   for c in sorted(cands, key=lambda x: -x["score"])[:5]],
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            done_topics.add(d["topic_id"])
            fresh_topics += 1
            if fresh_topics % 10 == 0:
                print(f"  topics processed {fresh_topics}: {counts}")
    print(f"plan: {out}")
    print("  ", counts)


# --- stage 3: report ----------------------------------------------------------
def step_report(args) -> int:
    rows = read_jsonl(corpus() / "dataset_plan.jsonl")
    if not rows:
        print("no plan")
        return 1
    from collections import Counter, defaultdict
    by_status = Counter(r["status"] for r in rows)
    by_author = defaultdict(Counter)
    for r in rows:
        by_author[r.get("author") or r.get("author_pid")][r["status"]] += 1
    print(f"releases in plan: {len(rows)}")
    print("by status:", dict(by_status))
    for a, c in by_author.items():
        print(f"  {a}: {dict(c)}")
    m = [r for r in rows if r["status"] == "matched"]
    size = sum((r["chosen"]["size_bytes"] or 0) for r in m)
    print(f"total download size (all matched): {size / 1e9:.1f} GB "
          f"({size / len(m) / 1e6:.0f} MB on average)")
    years = Counter((r["release"]["year"] or 0) // 10 * 10 for r in m)
    print("by decade:", dict(sorted(years.items())))
    drs = [r["release"]["master_dr"]["official_dr"] for r in m
           if r["release"].get("master_dr") and r["release"]["master_dr"].get("official_dr")]
    if drs:
        import statistics as S
        print(f"master DR (matched): median {S.median(drs):.0f}, "
              f"spread {min(drs)}..{max(drs)}, n={len(drs)}")
    frag = sum(1 for r in m if "graphy" in (r["chosen"]["flags"] or []))
    print(f"discography originals (file-based selection required): {frag}/{len(m)}")
    print("\ntop-10 by \"interestingness\" (low master DR ⇒ heavily crushed source):")
    def rank(r):
        md = (r["release"].get("master_dr") or {}).get("official_dr") or 99
        return (md, -(r["chosen"]["seeders"] or 0))
    for r in sorted(m, key=rank)[:10]:
        print(f"  DR{r['release']['master_dr']['official_dr']:>2} {r['release']['year']} "
              f"{r['artist'][:28]:<28} | {r['release']['album'][:42]:<42} | "
              f"t={r['chosen']['topic_id']} {r['chosen']['seeders']}se "
              f"{(r['chosen']['size_bytes'] or 0) / 1e6:.0f}MB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["topics", "match", "report"])
    ap.add_argument("--author", default=",".join(AUTHORS))
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--pause", type=float, default=1.3)
    ap.add_argument("--pages", type=int, default=2, help="search pages per artist")
    ap.add_argument("--refresh", action="store_true", help="re-fetch topic parses")
    ap.add_argument("--refresh-search", action="store_true", help="reset the search cache")
    ap.add_argument("--rewrite", action="store_true", help="rewrite the plan from scratch")
    ap.add_argument("--out", default=None, help="plan file (default dataset_plan.jsonl)")
    args = ap.parse_args()
    if args.step == "topics":
        return step_topics(args) or 0
    if args.step == "match":
        return step_match(args) or 0
    return step_report(args)


if __name__ == "__main__":
    sys.exit(main())
