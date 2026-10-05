#!/usr/bin/env python3
"""scan_dr.py — TT DR of every track in a library, with resume and [restored] skipping.

Strategy: bulk-copy album groups to local disk (per-file fuse reads are ~100x
slower than bulk copies on MTP), decode locally in parallel, delete the stage.
Result: STATE_DIR/dr_scan.tsv + per-album STATE_DIR/dr_albums.tsv sorted by
average DR (lowest first = most crushed = best restoration candidates).

Usage:
  scan_dr.py                       # phone library (auto-detected)
  scan_dr.py /path/to/library      # any local or mounted library
  scan_dr.py --include-restored    # do not skip groups with 'restored' in the path
  scan_dr.py --workers 4           # local decode parallelism (default 4)

Resume: rows already present in dr_scan.tsv are kept; their groups are skipped.
"""
import argparse
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drlib.config import ensure_dirs, state_dir, work_dir
from drlib.mtp import phone_dir
from measure import decode, tt_dr

EXTS = (".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav")


def one(args):
    local, album, rel, fmt = args
    try:
        x, fs = decode(str(local), "s16le")
        dr, drx, _, pk, _ = tt_dr(x / 32768.0, fs)
        return f"{album}\t{rel}\t{dr}\t{drx:.1f}\t{pk:.1f}\t{fmt}"
    except Exception:
        return f"{album}\t{rel}\tERR\t0\t0\t{fmt}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", help="library root (default: phone Music dir)")
    p.add_argument("--out", help="output TSV (default STATE_DIR/dr_scan.tsv)")
    p.add_argument("--include-restored", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    ensure_dirs()
    root = Path(a.root) if a.root else phone_dir()
    if root is None or not root.exists():
        sys.exit("no library root: connect the phone (MTP) or pass a path")
    out = Path(a.out) if a.out else state_dir() / "dr_scan.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = work_dir() / "dr_scan_stage"

    keep = (lambda al: True) if a.include_restored else (lambda al: "restored" not in al.lower())

    groups = {}
    n_files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(EXTS):
                f = Path(dirpath) / fn
                parts = f.relative_to(root).parts[:-1]
                album = " / ".join(parts) if parts else "(root)"
                if keep(album):
                    groups.setdefault(album, []).append((f, f.relative_to(root)))
                    n_files += 1
    print(f"to measure: {n_files} tracks in {len(groups)} groups"
          + ("" if a.include_restored else " ('restored' groups skipped)"), flush=True)

    old_lines, measured = [], set()
    if out.exists():
        for line in out.read_text().splitlines()[1:]:
            if line.strip():
                al = line.split("\t")[0]
                if keep(al):
                    old_lines.append(line)
                    measured.add(al)
    skip = [g for g in groups if g in measured]
    print(f"resume: {len(old_lines)} rows already present; skipping {len(skip)} measured groups", flush=True)

    new_lines = []
    done = 0
    with open(out, "w") as t:
        t.write("album\tfile\tdr\tdrx\tpeak_db\tfmt\n")
        for l in old_lines:
            t.write(l + "\n")
        for gi, (album, items) in enumerate(sorted(groups.items()), 1):
            if album in measured:
                continue
            gstage = stage / f"g{gi}"
            gstage.mkdir(parents=True, exist_ok=True)
            jobs = []
            for j, (f, rel) in enumerate(items):
                local = gstage / f"{j:04d}{f.suffix.lower()}"
                try:
                    shutil.copyfile(f, local)
                    jobs.append((local, album, str(rel), f.suffix.lower().lstrip(".")))
                except Exception as e:
                    print(f"COPY ERR {rel}: {e}", flush=True)
            with Pool(a.workers) as pool:
                for line in pool.imap_unordered(one, jobs):
                    new_lines.append(line)
                    t.write(line + "\n")
            t.flush()
            shutil.rmtree(gstage)
            done += len(jobs)
            print(f"[{gi}/{len(groups)}] {album[:60]}: {len(jobs)} tracks (new {done}/{n_files})", flush=True)

    rows = {}
    for line in old_lines + new_lines:
        al, fn, dr, drx, pk, fmt = line.split("\t")
        if dr != "ERR":
            rows.setdefault(al, []).append((int(dr), float(drx)))
    with open(out.parent / "dr_albums.tsv", "w") as t:
        t.write("album\tn\tavg_dr\tavg_drx\tmin_dr\n")
        for al, v in sorted(rows.items(), key=lambda kv: sum(x[0] for x in kv[1]) / len(kv[1])):
            n = len(v)
            t.write(f"{al}\t{n}\t{sum(x[0] for x in v)/n:.1f}\t{sum(x[1] for x in v)/n:.2f}\t{min(x[0] for x in v)}\n")
    shutil.rmtree(stage, ignore_errors=True)
    print(f"DONE -> {out} + {out.parent / 'dr_albums.tsv'}")


if __name__ == "__main__":
    main()
