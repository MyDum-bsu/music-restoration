"""Reading tags/covers via mutagen.

The full tag and ReplayGain copy stayed in v1; here is only what the phone
library inventory needs — no numpy/ffmpeg dependencies.
"""
from pathlib import Path


def has_cover(flac) -> bool:
    from mutagen.flac import FLAC
    return bool(FLAC(str(flac)).pictures)


def audio_flags(path):
    """One mutagen read per file -> dict of flags, or None if unreadable.

    lyrics/cover/tracknumber are bools; artist/title/album/date are values.
    """
    from mutagen import File as MF
    from mutagen.mp4 import MP4
    path = Path(path)
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
