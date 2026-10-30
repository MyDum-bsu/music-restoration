#!/usr/bin/env python3
"""Cover art for a release when the source has none.

Source order:
  1. MusicBrainz → Cover Art Archive (by artist and album).
  2. A cover already embedded in a donor file (`--donor file.flac`) — for
     example, in the old version on the phone.

  python3 scripts/fetch_cover.py --artist Subsignal --album "La Muerta" --out cover.jpg
  python3 scripts/fetch_cover.py --donor /path/old.flac --out cover.jpg

WEB releases often come without a cover file (like Subsignal "La Muerta"), so
without this step the album lands on the phone "naked".
"""
import argparse
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

UA = "music-restoration/2.0 ( audio-restoration )"


def curl(url: str, out: Path, timeout: int = 90) -> int:
    r = subprocess.run(["curl", "-s", "-L", "-o", str(out), "-w", "%{http_code}",
                        "-H", f"User-Agent: {UA}", "--max-time", str(timeout), url],
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def from_musicbrainz(artist: str, album: str, out: Path) -> bool:
    q = f'release:"{album}" AND artist:"{artist}"'
    url = ("https://musicbrainz.org/ws/2/release?query="
           + urllib.parse.quote(q) + "&fmt=json&limit=25")
    tmp = out.with_suffix(".mb.json")
    if curl(url, tmp) != 200:
        return False
    try:
        data = json.loads(tmp.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return False
    finally:
        tmp.unlink(missing_ok=True)
    # One album in MusicBrainz has dozens of releases, and "front" can be a
    # photo of the box or non-square. Iterate and pick the best one: square
    # and as large as possible.
    from PIL import Image
    best = None
    tmp_im = out.with_name("_cand.jpg")
    for rel in data.get("releases", []):
        mbid = rel.get("id")
        if not mbid:
            continue
        for size in ("front-1200", "front-500", "front"):
            if curl(f"https://coverartarchive.org/release/{mbid}/{size}", tmp_im) != 200:
                continue
            if tmp_im.stat().st_size <= 5000:
                continue
            try:
                w, h = Image.open(tmp_im).size
            except Exception:  # noqa: BLE001
                break
            ratio = min(w, h) / max(w, h)
            score = (round(ratio, 2), w)
            print(f"  candidate: {rel.get('title')} ({rel.get('date') or '?'}) "
                  f"{w}x{h} squareness {ratio:.2f}")
            if best is None or score > best[0]:
                best = (score, tmp_im.read_bytes(), w, h)
            break
    tmp_im.unlink(missing_ok=True)
    if best:
        out.write_bytes(best[1])
        print(f"  cover chosen {best[2]}x{best[3]} (squareness {best[0][0]:.2f})")
        return True
    return False


def from_donor(path: Path, out: Path) -> bool:
    if not path.is_file():
        return False
    r = subprocess.run(["metaflac", "--export-picture-to=" + str(out), str(path)],
                       capture_output=True, text=True)
    if r.returncode == 0 and out.is_file() and out.stat().st_size > 5000:
        print(f"  cover taken from the donor file: {path.name}")
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="")
    ap.add_argument("--album", default="")
    ap.add_argument("--donor", default="", help="FLAC with an embedded cover")
    ap.add_argument("--out", default="cover.jpg")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    ok = False
    if args.donor:
        ok = from_donor(Path(args.donor).expanduser(), out)
    if not ok and args.artist and args.album:
        for attempt in (1, 2):          # MusicBrainz network/rate limits — retry
            ok = from_musicbrainz(args.artist, args.album, out)
            if ok:
                break
            if attempt == 1:
                print("  retrying the cover request...")
                import time
                time.sleep(3)
    if not ok:
        print("  cover not found in MusicBrainz or the donor")
        return 1
    from PIL import Image
    im = Image.open(out)
    if max(im.size) > 1200:
        im.thumbnail((1200, 1200))
        im.convert("RGB").save(out, quality=92)
    print(f"  cover: {out} ({out.stat().st_size // 1024} KB, {im.size[0]}x{im.size[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
