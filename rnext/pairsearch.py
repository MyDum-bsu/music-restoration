"""Original-release search on rutracker: title normalization, candidate scoring."""
import re

from bs4 import BeautifulSoup

from rnext.rtclient import FORUM, RTClient
from rnext.rtscrape import ROW_SEL, _row_dict

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\[\(].*?[\]\)]", " ", s)          # bracketed annotations
    # keep Cyrillic letters (rutracker titles are often Russian): \u0430-\u044F = a-ya, \u0451 = yo
    s = re.sub(r"[^\w\u0430-\u044F\u0451\s]", " ", s, flags=re.IGNORECASE)
    return _WS.sub(" ", s).strip()


def title_tags(title: str) -> dict:
    """Tail of the rutracker title format: genre, artist, album/discography, year, format."""
    t = title
    genre = None
    m = re.match(r"\(([^)]+)\)\s*", t)
    if m:
        genre = m.group(1).strip()
        t = t[m.end():]
    year = None
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", t)
    if m:
        year = int(m.group(1))
    fmt = None
    if re.search(r"\bFLAC\b", t):
        fmt = "FLAC"
    elif re.search(r"\bMP3\b", t):
        fmt = "MP3"
    # ru title words: "Discography" / "Compilations"
    is_graphy = bool(re.search(
        r"\u0414\u0438\u0441\u043A\u043E\u0433\u0440\u0430\u0444\u0438\u044F|releases|\u0421\u0431\u043E\u0440\u043D\u0438\u043A\u0438",
        t, re.IGNORECASE))
    return {"genre": genre, "year": year, "fmt": fmt, "discography": is_graphy}


def split_artist_album(title_tail: str) -> tuple[str, str | None]:
    """'Accept - Metal Heart - 1985' -> ('accept', 'metal heart')."""
    parts = [p.strip() for p in title_tail.split(" - ")]
    if len(parts) == 1:
        return _norm(parts[0]), None
    artist = _norm(parts[0])
    album = _norm(parts[1]) if len(parts) > 1 else None
    return artist, album


def search_topics(nm: str, client: RTClient | None = None, pages: int = 1) -> list[dict]:
    """Full-text tracker search (all topics under `nm`), rows from the results table."""
    c = client or RTClient()
    out: dict[int, dict] = {}
    for page in range(pages):
        r = c.get(FORUM + f"tracker.php?nm={nm}&start={50 * page}")
        for tr in BeautifulSoup(r.text, "html.parser").select(ROW_SEL):
            d = _row_dict(tr)
            if d["topic_id"] and d["topic_id"] not in out:
                out[d["topic_id"]] = d
    return list(out.values())


def score_candidate(title: str, artist_n: str, album_n: str | None, year: int | None, must: dict) -> dict:
    """Scoring of an original candidate: artist/album in the title, year, format, 'not a restoration'."""
    tags = title_tags(title)
    t_n = _norm(title)
    score = 0.0
    if artist_n and artist_n in t_n:
        score += 40
    if album_n:
        if album_n in t_n:
            score += 40
    if year and tags["year"]:
        if year == tags["year"]:
            score += 10
        elif abs(year - tags["year"]) <= 3:
            score += 6
        else:
            score -= 4
    if tags["fmt"] == "FLAC":
        score += 15
    if tags["discography"]:
        score -= 25
    # \u0440\u0435\u043C\u0430\u0441\u0442\u0435\u0440 = ru "remaster"
    if not must.get("allow_rm") and re.search(
            r"\bRM\b|\brestored\b|remaster|\u0440\u0435\u043C\u0430\u0441\u0442\u0435\u0440", title, re.IGNORECASE):
        score -= 15
    return {"score": score, "tags": tags}


def find_originals(artist: str, album: str | None, year: int | None, client: RTClient | None = None,
                   limit: int = 25, pages: int = 2) -> list[dict]:
    """Full search scan for "Artist Album Year" -> sorted candidates."""
    artist_n, _ = split_artist_album(artist)
    album_n = _norm(album) if album else None
    q = " ".join(x for x in (artist, album) if x)
    rows = search_topics(q, client=client, pages=pages)
    out = []
    for r in rows:
        if not r.get("title"):
            continue
        s = score_candidate(r["title"], artist_n, album_n, year, {})
        out.append({**r,
                    "score": s["score"],
                    "tags": s["tags"]})
    out.sort(key=lambda d: -d["score"])
    return out[:limit]
