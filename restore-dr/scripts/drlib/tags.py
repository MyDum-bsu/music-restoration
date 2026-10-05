"""FLAC tag/cover helpers built on mutagen + metaflac.

copy_tags() is mutagen-based and multiline-safe: the earlier metaflac
text-export version silently dropped every LYRICS continuation line.
It does NOT copy PICTURE blocks — covers are handled explicitly, because
restored files must never silently inherit (or lose) artwork.
"""
import subprocess
from pathlib import Path

import numpy as np

from .audio import decode
from .util import run

RG_KEYS = ("replaygain_track_gain", "replaygain_track_peak",
           "replaygain_album_gain", "replaygain_album_peak")


def lufs_of(file) -> float:
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", str(file), "-filter_complex",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    import re
    m = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", p.stderr.split("Summary:")[-1])
    return float(m.group(1))


def copy_tags(src_flac, dst_flac) -> None:
    """ALL tags of the original (incl. multiline LYRICS) + ReplayGain (RG2.0, ref -18 LUFS)."""
    from mutagen.flac import FLAC
    o, r = FLAC(str(src_flac)), FLAC(str(dst_flac))
    i = lufs_of(dst_flac)
    x, _ = decode(str(dst_flac), "s16le")
    peak = float(np.abs(x).max())
    r.tags.clear()
    for k, v in (o.tags or {}).items():
        if k.lower() not in RG_KEYS:
            r.tags[k] = list(v)
    r.tags["REPLAYGAIN_TRACK_GAIN"] = [f"{-18.0 - i:+.2f} dB"]
    r.tags["REPLAYGAIN_TRACK_PEAK"] = [f"{peak:.6f}"]
    r.save()


def has_cover(flac) -> bool:
    from mutagen.flac import FLAC
    return bool(FLAC(str(flac)).pictures)


def export_cover(src_flac, dest_img) -> bool:
    """Export the first PICTURE block of src to dest. mutagen has no export; metaflac does."""
    dest_img = Path(dest_img)
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    p = run(["metaflac", f"--export-picture-to={dest_img}", str(src_flac)])
    return p.returncode == 0 and dest_img.exists() and dest_img.stat().st_size > 0


def import_cover(flacs, img) -> None:
    """Replace artwork on all files with img (metaflac --import-picture-from)."""
    for f in flacs:
        run(["metaflac", "--remove", "--block-type=PICTURE", str(f)])
        run(["metaflac", f"--import-picture-from={img}", str(f)], check=True)


def set_core_tags(flac, artist=None, album=None, date=None, genre=None,
                  tracknumber=None, tracktotal=None, title=None,
                  albumartist=None, discnumber=1, disctotal=1) -> None:
    """Set the fields that are not None. Always normalizes discnumber/disctotal —
    MIXED discnumber presence splits an album into 'Disk 0'/'Disk 1' in players."""
    from mutagen.flac import FLAC
    f = FLAC(str(flac))
    vals = {"artist": artist, "album": album, "date": date, "genre": genre,
            "tracknumber": tracknumber, "totaltracks": tracktotal, "title": title,
            "albumartist": albumartist or artist,
            "discnumber": str(discnumber), "disctotal": str(disctotal)}
    for k, v in vals.items():
        if v is not None and str(v).strip():
            f.tags[k] = [str(v)]
    f.save()


def audio_flags(path):
    """One mutagen read per file -> dict of flags, or None if unreadable.

    lyrics/cover/tracknumber are bools; artist/title/album/date hold the values.
    """
    from mutagen import File as MF
    from mutagen.mp4 import MP4
    try:
        f = MF(str(path), easy=True)
        if not f or f.tags is None:
            return None

        def g(k):
            v = f.get(k)
            return (v[0].strip() if v and v[0].strip() else "")

        lyrics = bool(g("lyrics") or g("unsyncedlyrics"))
        cover = False
        if path.suffix.lower() in (".m4a", ".m4b", ".mp4"):
            mp4 = MP4(str(path))
            cover = bool(mp4.get("covr"))
        elif hasattr(f.tags, "getall"):  # ID3
            cover = bool(f.tags.getall("APIC"))
        else:  # FLAC / vorbis comment containers
            cover = bool(getattr(f, "pictures", []))
        return {
            "lyrics": lyrics,
            "cover": cover,
            "tracknumber": bool(g("tracknumber")),
            "artist": g("artist"),
            "title": g("title"),
            "album": g("album"),
            "date": g("date"),
        }
    except Exception:
        return None
