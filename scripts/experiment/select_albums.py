#!/usr/bin/env python3
"""Pick a seeded album sample for the independent-restore experiment.

  python3 scripts/experiment/select_albums.py --src <root> --per-artist 5 \
          --seed <int> --out <manifest.json>

An album is a top-level directory of <root> named "<artist> - <album>";
the artist is everything before the first " - ", the album is the rest of the
name.  Per artist the script keeps min(--per-artist, available) albums sampled
with random.Random(seed) from the sorted list, so the same seed always writes
the same bytes.  Directories that are not album-shaped are reported and
skipped; an album-shaped directory without audio, or a root without a single
album, is a hard error rather than an empty manifest.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

AUDIO_EXT = {".flac", ".wav", ".ape", ".m4a", ".mp3", ".ogg", ".opus", ".aiff",
             ".alac"}

SEPARATOR = " - "


def audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)


def scan(src: Path) -> tuple[dict[str, list[dict]], list[str]]:
    """Return {artist: [album entries]} plus the names that are not albums."""
    albums: dict[str, list[dict]] = {}
    ignored: list[str] = []
    for path in sorted(src.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if SEPARATOR not in name:
            ignored.append(name)
            continue
        artist, album = name.split(SEPARATOR, 1)
        artist, album = artist.strip(), album.strip()
        if not artist or not album:
            ignored.append(name)
            continue
        n_tracks = len(audio_files(path))
        if n_tracks == 0:
            raise SystemExit(f"select_albums: album directory without audio files: {path}")
        albums.setdefault(artist, []).append(
            {"artist": artist, "album": name, "rel_path": name, "n_tracks": n_tracks}
        )
    return albums, ignored


def main() -> int:
    ap = argparse.ArgumentParser(description="Seeded album sample for the experiment.")
    ap.add_argument("--src", required=True, help="root with one directory per album")
    ap.add_argument("--per-artist", type=int, default=5, help="albums per artist")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--out", required=True, help="manifest.json to write")
    args = ap.parse_args()

    if args.per_artist < 1:
        print(f"select_albums: --per-artist must be >= 1, got {args.per_artist}",
              file=sys.stderr)
        return 2

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        print(f"select_albums: not a readable directory: {src}", file=sys.stderr)
        return 2
    try:
        albums, ignored = scan(src)
    except OSError as exc:
        print(f"select_albums: cannot read {src}: {exc}", file=sys.stderr)
        return 2
    if not albums:
        print(f"select_albums: no album directories with audio files under {src}",
              file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    chosen: list[dict] = []
    for artist in sorted(albums):
        pool = sorted(albums[artist], key=lambda entry: entry["album"])
        k = min(args.per_artist, len(pool))
        chosen.extend(sorted(rng.sample(pool, k), key=lambda entry: entry["album"]))

    payload = {
        "src": str(src),
        "per_artist": args.per_artist,
        "seed": args.seed,
        "albums": sorted(chosen, key=lambda entry: (entry["artist"], entry["album"])),
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for name in ignored:
        print(f"select_albums: skipped non-album directory {name!r}")
    print(f"select_albums: {len(chosen)} albums from {len(albums)} artists "
          f"(seed {args.seed}, per-artist {args.per_artist}) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
