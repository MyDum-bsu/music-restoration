#!/usr/bin/env python3
"""inventory.py — audit a phone music library: what every file has and lacks.

Checks per audio file: lyrics, embedded cover, tracknumber, artist/title/album/date.
Writes STATE_DIR/inventory.tsv (STATUS\trel) and prints a per-top-level-dir summary.
Run this before any bulk job and to spot albums worth enriching.

Usage:
  python3 inventory.py                                  # whole library (~5 min / 2800 files)
  python3 inventory.py --dir "Place Vendome"            # one group
  python3 inventory.py --missing cover --missing lyrics # filter the report
"""
import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = Path(__file__).resolve().parents[3]
if not (_REPO / "rnext").is_dir():
    import os
    _REPO = Path(os.environ.get("MUSIC_RESTORATION_HOME", _REPO))
sys.path.insert(0, str(_REPO))
from rnext.phone import mtp
from rnext.phone.config import ensure_dirs
from rnext.phone.tags import audio_flags

ap = argparse.ArgumentParser()
ap.add_argument("--dir", help="audit only this top-level dir (relative to the music root)")
ap.add_argument("--root", help="explicit music root (default: auto-detected phone)")
ap.add_argument("--missing", action="append", choices=["lyrics", "cover", "tracknumber", "tags"],
                help="only report files missing this (repeatable)")
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()

ensure_dirs()
base = Path(a.root) if a.root else mtp.phone_dir()
if base is None or not base.exists():
    sys.exit("no music root found: connect the phone (MTP) or pass --root / set MR_PHONE_DIR")
root = base / a.dir if a.dir else base
if not root.exists():
    sys.exit(f"--dir '{a.dir}' does not exist under {base}")
files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in mtp.AUDIO_EXTS)
print(f"auditing {len(files)} files under {root}", flush=True)

out_lines = []
stat = Counter()
t0 = time.time()


def check(p):
    flags = audio_flags(p)
    rel = str(p.relative_to(base)) if p.is_relative_to(base) else str(p)
    if flags is None:
        return rel, "UNREADABLE", None
    miss = []
    if not flags["lyrics"]: miss.append("lyrics")
    if not flags["cover"]: miss.append("cover")
    if not flags["tracknumber"]: miss.append("tracknumber")
    if not (flags["artist"] and flags["title"] and flags["album"]): miss.append("tags")
    return rel, ("OK" if not miss else "MISS:" + ",".join(miss)), flags


with ThreadPoolExecutor(max_workers=a.workers) as ex:
    for i, (rel, status, flags) in enumerate(ex.map(check, files), 1):
        out_lines.append(f"{status}\t{rel}")
        stat[status.split(":")[0] if status.startswith("MISS") else status] += 1
        if i % 200 == 0:
            print(f"...{i} ({time.time()-t0:.0f}s)", flush=True)

from rnext.phone.config import state_dir
tsv = state_dir() / "inventory.tsv"
tsv.write_text("\n".join(out_lines) + "\n")

print(f"\n=== summary ({time.time()-t0:.0f}s) — TSV: {tsv}")
for k, n in stat.most_common():
    print(f"  {n:5d}  {k}")
if a.missing:
    want = set(a.missing)
    print("\n=== files missing " + "/".join(sorted(want)) + ":")
    for line in out_lines:
        status, rel = line.split("\t", 1)
        miss = set(status[5:].split(",")) if status.startswith("MISS:") else set()
        if miss & want:
            print(f"  {rel}")
