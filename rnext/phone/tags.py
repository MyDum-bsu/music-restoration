"""Чтение тегов/обложек через mutagen.

Полная копия тегов и ReplayGain осталась в v1; здесь только то, что нужно
инвентарю телефонной библиотеки, — без numpy/ffmpeg-зависимостей.
"""
from pathlib import Path


def has_cover(flac) -> bool:
    from mutagen.flac import FLAC
    return bool(FLAC(str(flac)).pictures)


def audio_flags(path):
    """Одно чтение mutagen на файл -> словарь флагов, или None если не читается.

    lyrics/cover/tracknumber — bool; artist/title/album/date — значения.
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
