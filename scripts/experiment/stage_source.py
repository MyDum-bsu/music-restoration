#!/usr/bin/env python3
"""Line a release library up with the canonical track list, without copying bytes.

  python3 scripts/experiment/stage_source.py --src <release_src> --ours <canon root> \
          --manifest <manifest.json> --out <stage dir> [--report <json>]

Why this exists: a release download is laid out for humans - a nested release
folder, "<NN> - <Title>.flac", sometimes a whole-album image next to the single
tracks, sometimes a bonus disc - while the canon is flat and named
"<NN>. [<Artist> - ]<Title>.flac".  The comparator matches tracks by exact key,
so the source side has to be normalised once, visibly, before it is measured.

For every album of the manifest the canon directory is located, and for every
canonical track exactly one source file is linked into
<stage>/<album>/<canonical key>.<source suffix>: same track title, and when
several source files share that title, the same track number.  Source files
that the canon does not cover (whole-album images, bonus discs, pregaps) are
never staged - they are listed per album in the report and therefore in the
analysis, not dropped silently.  Sources are only read; the stage holds
symlinks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

AUDIO_EXT = {".flac", ".wav", ".ape", ".m4a", ".mp3", ".ogg", ".opus", ".aiff",
             ".alac"}
CANON_SUFFIX = " [restored per-track]"
VERSION_TAGS = (" TR24", " JP", " FLAC", " 24bit")


def audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)


def album_base(album: str) -> str:
    name = album
    changed = True
    while changed:
        changed = False
        for tag in VERSION_TAGS:
            if name.endswith(tag):
                name = name[: -len(tag)].rstrip()
                changed = True
    return name


def canon_dir(root: Path, album: str) -> Path:
    candidates = [root / album, root / (album_base(album) + CANON_SUFFIX),
                  root / (album + CANON_SUFFIX), root / album_base(album)]
    existing = [path for path in dict.fromkeys(candidates) if path.is_dir()]
    if not existing:
        raise SystemExit(f"stage_source: no canon directory for album {album!r} under {root}")
    if len(existing) > 1:
        raise SystemExit(f"stage_source: ambiguous canon directories for {album!r}: "
                         f"{', '.join(str(p) for p in existing)}")
    return existing[0]


def track_number(stem: str) -> int | None:
    match = re.match(r"^\s*(\d{1,3})(?!\d)", stem)
    return int(match.group(1)) if match else None


def norm_title(stem: str, artist: str) -> str:
    text = unicodedata.normalize("NFKD", stem).casefold()
    text = re.sub(r"^\s*\d{1,3}\s*[-._)\]]\s+", "", text)
    text = re.sub(r"^\s*\d{1,3}\s+", "", text)
    prefix = unicodedata.normalize("NFKD", artist).casefold() + " - "
    if text.startswith(prefix):
        text = text[len(prefix):]
    return re.sub(r"[\W_]+", "", text)


def key_of(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix(""))


def stage_album(src: Path, rel_path: str, artist: str, canon_root: Path,
                out: Path) -> tuple[list[dict], list[str]]:
    src_dir = src / rel_path
    if not src_dir.is_dir():
        raise SystemExit(f"stage_source: source album directory is missing: {src_dir}")
    target = canon_dir(canon_root, rel_path)
    canon_files = audio_files(target)
    if not canon_files:
        raise SystemExit(f"stage_source: canon directory without audio: {target}")

    sources = audio_files(src_dir)
    if not sources:
        raise SystemExit(f"stage_source: source directory without audio: {src_dir}")
    pool: dict[str, list[Path]] = {}
    for path in sources:
        pool.setdefault(norm_title(path.stem, artist), []).append(path)

    staged: list[dict] = []
    used: set[Path] = set()
    problems: list[str] = []
    for canon_file in canon_files:
        canon_key = key_of(canon_file, target)
        title = norm_title(canon_file.stem, artist)
        number = track_number(canon_file.stem)
        hits = pool.get(title, [])
        if len(hits) > 1:
            hits = [p for p in hits if track_number(p.stem) == number] or hits
        if len(hits) != 1:
            problems.append(f"{canon_key!r} (title {title!r}, number {number}): "
                            f"{len(hits)} candidates"
                            + (": " + ", ".join(str(p.relative_to(src_dir)) for p in hits)
                               if hits else ""))
            continue
        source = hits[0]
        link = out / rel_path / (canon_key + source.suffix.lower())
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(source.resolve(), link)
        used.add(source)
        staged.append({"track": canon_key, "source": str(source.relative_to(src_dir))})
    if problems:
        raise SystemExit(f"stage_source: album {rel_path!r} cannot be staged:\n  "
                         + "\n  ".join(problems))

    dropped = [str(p.relative_to(src_dir)) for p in sources if p not in used]
    return staged, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalise a release library onto the canon keys.")
    ap.add_argument("--src", required=True, help="release library root (read only)")
    ap.add_argument("--ours", required=True, help="root holding the canonical album directories")
    ap.add_argument("--manifest", required=True, help="manifest.json from select_albums.py")
    ap.add_argument("--out", required=True, help="stage directory to create")
    ap.add_argument("--report", default="", help="JSON report of staged/dropped files")
    ap.add_argument("--append", action="store_true",
                    help="add these albums to an existing report")
    ap.add_argument("--only", action="append", default=[],
                    help="stage only this album (rel_path); may be repeated")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    canon_root = Path(args.ours).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expected = {"src", "per_artist", "seed", "albums"}
    if set(manifest) != expected:
        raise SystemExit(f"stage_source: manifest keys {sorted(manifest)} != {sorted(expected)}")
    if not manifest["albums"]:
        raise SystemExit("stage_source: manifest lists no albums")
    if src in out.parents or out == src:
        raise SystemExit(f"stage_source: --out must not be inside --src: {out}")

    report = {"src": str(src), "ours": str(canon_root), "albums": []}
    report_path = Path(args.report).expanduser() if args.report else None
    if report_path and args.append and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = [entry for entry in manifest["albums"]
                if not args.only or entry["rel_path"] in args.only]
    if args.only and len(selected) != len(set(args.only)):
        known = {entry["rel_path"] for entry in selected}
        raise SystemExit(f"stage_source: --only albums not in the manifest: "
                         f"{sorted(set(args.only) - known)}")
    if args.append and args.only:
        report["albums"] = [item for item in report["albums"]
                            if item["album"] not in set(args.only)]
    for entry in selected:
        rel_path = entry["rel_path"]
        staged, dropped = stage_album(src, rel_path, entry["artist"], canon_root, out)
        report["albums"].append({
            "album": rel_path,
            "artist": entry["artist"],
            "staged": len(staged),
            "dropped": dropped,
            "tracks": staged,
        })
        note = f" ({len(dropped)} source files outside the canon)" if dropped else ""
        print(f"stage_source: {rel_path}: {len(staged)} tracks{note}")
        for name in dropped:
            print(f"    not compared: {name}")

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    total = sum(len(item["dropped"]) for item in report["albums"])
    print(f"stage_source: {sum(a['staged'] for a in report['albums'])} tracks staged, "
          f"{total} source files left out → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
