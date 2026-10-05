#!/usr/bin/env python3
"""embed_covers.py — enrich albums: embedded front covers + track numbers.

Deterministic album fixer:
- for every FLAC missing an embedded cover, embeds one picked from the album
  folder (Cover.jpg > Front.jpg > cover.jpg/front.jpg/folder.jpg > Cover|Scans/...)
  — big scans (> --max-kb) are downscaled with ffmpeg to --scale px first;
- for every FLAC missing TRACKNUMBER, sets it from the filename's leading
  number (zero-padded), plus TRACKTOTAL for the album folder.

Files without problems are not touched. Copy-back only for changed files,
each verified on the phone afterwards. Works on the phone (auto-detected) or
any local library root (--root).

Usage:
  python3 embed_covers.py --dir "Place Vendome" --dry-run   # one group, preview
  python3 embed_covers.py --album "path/to/album/dir"
  python3 embed_covers.py --all                             # whole library
"""
import argparse
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "restore-dr" / "scripts"))
from drlib import mtp
from drlib.config import ensure_dirs, state_dir
from drlib.util import run

ap = argparse.ArgumentParser()
ap.add_argument("--dir", help="top-level group under the music root (its subdirs are albums)")
ap.add_argument("--album", help="a single album dir (relative to the music root)")
ap.add_argument("--all", action="store_true", help="whole library (slow but safe)")
ap.add_argument("--root", help="explicit music root (default: auto-detected phone)")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--force", action="store_true", help="re-embed cover even if one exists")
ap.add_argument("--max-kb", type=int, default=600, help="downscale covers bigger than this (default 600)")
ap.add_argument("--scale", type=int, default=1200, help="downscale width in px (default 1200)")
a = ap.parse_args()

ensure_dirs()
root = Path(a.root) if a.root else mtp.phone_dir()
if root is None or not root.exists():
    sys.exit("no music root found: connect the phone (MTP) or pass --root / set MR_PHONE_DIR")

if a.album:
    album_dirs = [root / a.album]
elif a.dir:
    album_dirs = sorted(d for d in (root / a.dir).iterdir() if d.is_dir())
elif a.all:
    album_dirs = [d for d in root.rglob("*") if d.is_dir() and list(d.glob("*.flac"))]
else:
    sys.exit("need --dir, --album or --all")

COVER_PRIORITY = (("Cover.jpg", "Cover/01.jpg", "Front.jpg", "cover.jpg", "front.jpg", "folder.jpg"),
                  ("Cover", "Scans", "Covers"), ("front.jpg", "Front.jpg", "01.jpg"))


def pick_cover(album_dir: Path):
    for name in COVER_PRIORITY[0]:
        p = album_dir / name
        if p.exists(): return p
    for sub in COVER_PRIORITY[1]:
        d = album_dir / sub
        if d.is_dir():
            for name in COVER_PRIORITY[2]:
                p = d / name
                if p.exists(): return p
    return None


def downscale_if_needed(src: Path, stage: Path):
    """Return a local cover file <= max-kb, downscaling with ffmpeg if needed."""
    size = src.stat().st_size
    target = stage / ("cover_" + src.name)
    shutil.copy2(src, target)
    if size <= a.max_kb * 1024:
        return target, size
    r = run(["ffmpeg", "-v", "error", "-y", "-i", str(target),
             "-vf", f"scale={a.scale}:-2", "-q:v", "4", str(target) + ".small.jpg"])
    if r.returncode == 0 and (target.parent / (target.name + ".small.jpg")).exists():
        small = target.parent / (target.name + ".small.jpg")
        target.unlink()
        small.rename(target)
    return target, target.stat().st_size


def flags_of(p: Path):
    from mutagen.flac import FLAC
    f = FLAC(str(p))
    return f, bool(f.pictures), bool(f.get("TRACKNUMBER"))


total_fixed = total_ok = 0
for alb in album_dirs:
    flacs = sorted(alb.glob("*.flac"))
    if not flacs:
        continue
    cover_src = pick_cover(alb)
    need = []
    for p in flacs:
        try:
            _, has_cover, has_tn = flags_of(p)
        except Exception as e:
            print(f"READ-ERR {p.name}: {e}", flush=True); continue
        if a.force or not has_cover or not has_tn:
            need.append((p, has_cover, has_tn))
    if not need:
        total_ok += len(flacs)
        print(f"OK  {alb.name} — {len(flacs)} files already complete", flush=True)
        continue
    if not cover_src:
        print(f"NO-COVER {alb.name} — {len(need)} files need one, no candidate image in folder", flush=True)
        continue

    stage = state_dir() / "stage_covers"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    cover_local, cover_bytes = downscale_if_needed(cover_src, stage)
    note = f"(downscaled {cover_bytes // 1024} KB)" if cover_bytes != cover_src.stat().st_size else f"({cover_bytes // 1024} KB)"
    print(f"FIX  {alb.name} — {len(need)}/{len(flacs)} files, cover {cover_src.name} {note}", flush=True)

    if a.dry_run:
        continue

    from mutagen.flac import FLAC, Picture
    staged = []
    for p, _, _ in need:
        t = stage / p.name
        ok, err = mtp.copy_from_phone(root, p, t)
        if ok: staged.append((t, p))
        else: print(f"COPY-ERR {p.name}: {err}", flush=True)

    tracktotal = str(len(flacs))
    fixed = 0
    for t, orig in staged:
        m = re.match(r"(\d+)", t.name)
        f = FLAC(str(t))
        if not f.get("TRACKNUMBER") and m:
            f["TRACKNUMBER"] = m.group(1).zfill(2)
            f["TRACKTOTAL"] = tracktotal
        if a.force or not f.pictures:
            f.clear_pictures()
            pic = Picture()
            pic.type = 3; pic.mime = "image/jpeg"; pic.desc = "Cover"
            pic.data = cover_local.read_bytes()
            f.add_picture(pic)
        f.save()
        fixed += 1
    d = run(["flac", "-t", "--totally-silent", str(staged[0][0])], timeout=900)
    print(f"     tagged {fixed}, decode sample: {'OK' if d.returncode == 0 else 'FAIL ' + d.stderr[:200]}", flush=True)

    back = verified = 0
    t0 = time.time()
    for t, orig in staged:
        rel = orig.relative_to(root)
        ok, err = mtp.copy_to_phone(root, t, rel)
        if not ok:
            print(f"BACK-ERR {rel}: {err}", flush=True); continue
        back += 1
        _, has_cover, has_tn = flags_of(orig)
        if has_cover and has_tn: verified += 1
        else: print(f"VERIFY-FAIL {rel}", flush=True)
    print(f"     copied back {back}/{len(staged)} verified={verified} in {time.time()-t0:.0f}s", flush=True)
    total_fixed += fixed
    shutil.rmtree(stage, ignore_errors=True)

print(f"\nDONE: fixed {total_fixed} files, {total_ok} files already complete")
