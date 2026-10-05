"""Metadata enrichment: MusicBrainz tracklists, Cover Art Archive, filename parsing.

Fallback philosophy: MusicBrainz matched by track count (±2) because release
editions differ; a title from the filename beats no title; a folder image beats
no cover. Everything is a suggestion layer on top of tags kept from the source.
"""
import json
import re
from pathlib import Path

from .util import run

UA = "music-restoration/1.0 (https://github.com/MyDum-bsu/music-restoration)"


def mb_search_release_ids(artist: str, album: str, limit: int = 5) -> list[str]:
    q = f'release:"{album}" AND artist:"{artist}"'
    d = run(["curl", "-s", "-A", UA,
             "https://musicbrainz.org/ws/2/release?query=" + _enc(q) + f"&fmt=json&limit={limit}"])
    try:
        return [r["id"] for r in json.loads(d.stdout).get("releases", [])]
    except Exception:
        return []


def mb_release(mbid: str) -> dict:
    d = run(["curl", "-s", "-A", UA,
             f"https://musicbrainz.org/ws/2/release/{mbid}?inc=recordings+media&fmt=json"])
    try:
        return json.loads(d.stdout)
    except Exception:
        return {}


def mb_trackmap(artist: str, album: str, n_files: int):
    """(title-by-position dict, release date) from the first MB release whose
    track count matches n_files within ±2, or ({}, None)."""
    for mbid in mb_search_release_ids(artist, album):
        r = mb_release(mbid)
        tracks = [t for m in r.get("media", []) for t in m.get("tracks", [])]
        if tracks and abs(len(tracks) - n_files) <= 2:
            return {t["position"]: t["title"] for t in tracks}, (r.get("date") or "")[:4]
    return {}, None


def caa_cover(mbid: str, dest: Path) -> bool:
    run(["curl", "-sL", "-A", UA, "-o", str(dest), f"https://coverartarchive.org/release/{mbid}/front-500"])
    ok = dest.exists() and dest.stat().st_size > 5000 and _looks_like_image(dest)
    if not ok and dest.exists():
        dest.unlink()
    return ok


def _looks_like_image(path: Path) -> bool:
    """Content sniff — no external `file` utility (absent on Windows)."""
    head = path.open("rb").read(512).lstrip()
    if head.startswith((b"\xff\xd8", b"\x89PNG", b"GIF8", b"BM")):  # jpeg/png/gif/bmp
        return True
    return not head[:64].lower().lstrip(b"\xef\xbb\xbf").startswith((b"<", b"<!doctype"))


def parse_trackno(name: str) -> int:
    m = re.match(r"^(\d+)", name)
    return int(m.group(1)) if m else 0


def title_from_filename(name: str) -> str:
    t = re.sub(r"\.flac$", "", name, flags=re.I)
    t = re.sub(r"^\d+\s*[-–._)]?\s*", "", t)
    return t.strip()


COVER_PRIORITY = (("Cover.jpg", "Front.jpg", "cover.jpg", "front.jpg", "folder.jpg", "Cover/01.jpg"),
                  ("Cover", "Scans", "Covers"), ("front.jpg", "Front.jpg", "01.jpg"))


def pick_folder_cover(album_dir: Path) -> Path | None:
    """Likeliest front-cover image in an album folder; None if nothing plausible."""
    for name in COVER_PRIORITY[0]:
        p = album_dir / name
        if p.exists():
            return p
    for sub in COVER_PRIORITY[1]:
        d = album_dir / sub
        if d.is_dir():
            for name in COVER_PRIORITY[2]:
                p = d / name
                if p.exists():
                    return p
    return None


def _enc(s: str) -> str:
    import urllib.parse
    return urllib.parse.quote(s, safe="")
