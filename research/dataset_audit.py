#!/usr/bin/env python3
"""Dataset audit: whether anything downloaded was lost along the way.

Cross-checks three sources of truth:
  1. the download journal `dataset/state.jsonl` — what was ordered and how much was wanted;
  2. Transmission — how much traffic was actually spent and what sits on disk;
  3. `dataset/pairs/*/meta.json` + `features.json` — what was extracted into the dataset.

Prints anomalies:
  * `downloaded_not_extracted` — the torrent ran, but the pair is not in the dataset;
  * `present_no_state`        — a directory with audio not linked to any pair;
  * `partial`                 — not all needed files are in place;
  * `extracted_no_raw`        — normal (raw deleted after extraction);
  * `no_tracks`/`edition_mismatch`/`no_match` — rejected pairs (expected).

  python3 scripts/dataset_audit.py
  python3 scripts/dataset_audit.py --json out.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import sides as S  # noqa: E402
from rnext import sides as S  # noqa: E402
from rnext import workspace  # noqa: E402
from rnext.transm import Transm  # noqa: E402

AUDIO = (".flac", ".ape", ".wv", ".wav", ".m4a", ".alac", ".aiff", ".aif")


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_audio(root: Path) -> tuple[int, float]:
    n = 0
    b = 0
    if not root.is_dir():
        return 0, 0.0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO:
            n += 1
            b += p.stat().st_size
    return n, b


def integrity() -> int:
    """Dataset integrity: feature parsing, meta presence, excerpt readability.

    Needed after an abrupt termination: checks that files are not truncated
    and the row structure has not shifted.
    """
    import subprocess
    import numpy as np
    pairs = dataset_dir() / "pairs"
    bad: list[str] = []
    n_pairs = n_tracks = n_ex = 0
    for d in sorted(pairs.glob("*/")):
        fp, mp = d / "features.json", d / "meta.json"
        if not fp.is_file():
            if not mp.is_file():
                continue          # extraction in progress — not a problem
            bad.append(f"{d.name}: meta present but features.json missing")
            continue
        try:
            rows = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:                             # noqa: BLE001
            bad.append(f"{d.name}: features.json does not parse ({e})")
            continue
        if not mp.is_file():
            bad.append(f"{d.name}: meta.json missing")
        n_pairs += 1
        n_tracks += len(rows)
        for r in rows:
            for side in ("original", "restored"):
                if side not in r:
                    bad.append(f"{d.name} t{r.get('track')}: missing {side} block")
                    break
            else:
                for side in ("original", "restored"):
                    b = r[side].get("bands_db") or []
                    if len(b) != 30:
                        bad.append(f"{d.name} t{r.get('track')}: bands {len(b)} (expected 30)")
                        break
                if r.get("delta") is None or "crest" not in r["delta"]:
                    bad.append(f"{d.name} t{r.get('track')}: delta.crest missing")
        ex = sorted((d / "excerpts").glob("*.flac")) if (d / "excerpts").is_dir() else []
        for f in ex:
            n_ex += 1
            try:
                out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                      "format=duration", "-of", "csv=p=0", str(f)],
                                     capture_output=True, text=True, timeout=30)
                if out.returncode != 0 or not out.stdout.strip():
                    bad.append(f"{d.name}: corrupt excerpt {f.name}")
            except Exception as e:                          # noqa: BLE001
                bad.append(f"{d.name}: excerpt {f.name} — {type(e).__name__}")
    print(f"integrity: {n_pairs} pairs, {n_tracks} tracks, {n_ex} excerpts")
    if bad:
        print(f"PROBLEMS ({len(bad)}):")
        for b in bad[:20]:
            print("  ", b)
    else:
        print("no problems")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--integrity", action="store_true",
                    help="check data integrity after a crash (files/fields/excerpts)")
    args = ap.parse_args()

    if args.integrity:
        return integrity()

    uniq = S.merged_state(dataset_dir() / "state.jsonl")

    # what actually arrived in Transmission
    tors = S.torrents_by_id()

    ours = {tid: x for tid, x in tors.items()
            if "/restore/dataset/raw/" in (x.get("downloadDir") or "")}
    traffic = sum(x.get("downloadedEver") or 0 for x in ours.values())

    pairs_root = dataset_dir() / "pairs"
    extracted = {}
    for d in sorted(pairs_root.glob("*/")):
        meta = d / "meta.json"
        feats = d / "features.json"
        if not feats.is_file():
            continue
        rows = json.loads(feats.read_text(encoding="utf-8"))
        m = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        extracted[d.name.replace("__", "/")] = {"tracks": len(rows), "meta": m}

    # per-pair breakdown
    rows = []
    used_dirs: set[str] = set()
    for key, r in sorted(uniq.items()):
        sides = {}
        want = 0
        have = 0
        for side in ("orig", "master"):
            s = r.get(side) if isinstance(r.get(side), dict) else None
            if not s:
                continue
            base = Path(r.get("dir") or "")
            pres = S.side_presence(s, side, base, tmap=tors)
            used_dirs.add(pres["dir"])
            w = s.get("wanted_bytes") or 0
            want += w
            have += pres["bytes"]
            sides[side] = {"dir": pres["dir"], "wanted": pres["wanted"],
                           "present": pres["present"], "bytes_wanted": w,
                           "bytes": pres["bytes"]}
        st = r.get("state")
        ex = extracted.get(key)
        if ex:
            status = "extracted"
        elif st in ("no_match", "error"):
            status = "rejected" if st == "no_match" else "error"
        elif want and have < want * 0.98:
            status = "partial"
        elif want:
            status = "downloaded_not_extracted"
        else:
            status = "no_files"
        rows.append({"key": key, "status": status, "artist": r.get("artist"),
                     "album": r.get("album"), "sides": sides,
                     "bytes_wanted": want, "bytes_present": have,
                     "extracted_tracks": ex["tracks"] if ex else 0,
                     "match_fraction": (ex["meta"].get("match_fraction") if ex else None)})

    # directories with audio not present in any journal entry
    present_dirs = defaultdict(lambda: [0, 0.0])
    raw = dataset_dir() / "raw"
    if raw.is_dir():
        for d in raw.iterdir():
            if d.is_dir():
                n, b = dir_audio(d)
                present_dirs[str(d)] = [n, b]

    orphans = {d: v for d, v in present_dirs.items()
               if not any(str(d).startswith(u) or u.startswith(str(d)) for u in used_dirs if u)}

    # --- report
    by_status = Counter(r["status"] for r in rows)
    total_want = sum(r["bytes_wanted"] for r in rows)
    total_present = sum(r["bytes_present"] for r in rows)
    tracks = sum(v["tracks"] for v in extracted.values())
    print(f"pairs in journal: {len(rows)}   states: {dict(by_status)}")
    print(f"extracted into dataset: {len(extracted)} pairs / {tracks} tracks")
    print(f"ordered (wanted): {human(total_want)}   on disk now: {human(total_present)}")
    print(f"traffic on our torrents (downloadedEver): {human(traffic)}")
    print(f"dataset torrents in Transmission: {len(ours)}; "
          f"unfinished: {sum(1 for x in ours.values() if (x.get('leftUntilDone') or 0) > 0)}")

    anom = [r for r in rows if r["status"] in ("downloaded_not_extracted", "partial", "error")]
    if anom:
        print("\nANOMALIES:")
        for r in anom:
            print(f"  {r['status']:<26} {r['key']:<16} "
                  f"{str(r['artist'])[:20]:<20} {str(r['album'])[:30]:<30} "
                  f"got {human(r['bytes_present'])} of {human(r['bytes_wanted'])}")
    if orphans:
        print("\nDIRECTORIES WITH AUDIO OUTSIDE THE JOURNAL:")
        for d, (n, b) in orphans.items():
            print(f"  {d} — {n} files, {human(b)}")

    rejected = [r for r in rows if r["status"] == "rejected"]
    if rejected:
        print(f"\nRejected at file cross-check (expected, data not needed): {len(rejected)}")
        for r in rejected[:8]:
            print(f"  {r['key']:<16} {str(r['artist'])[:22]:<22} {str(r['album'])[:30]}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"pairs": rows, "by_status": dict(by_status), "traffic_bytes": traffic,
             "extracted_pairs": len(extracted), "extracted_tracks": tracks,
             "orphan_dirs": {d: v for d, v in orphans.items()}},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\njson → {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
