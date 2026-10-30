"""Dataset planning: master release -> original candidates -> decision.

Unlike `pairsearch` (one search per release): here the candidate pool is
fetched ONCE per artist (rutracker search by name), and releases are matched
locally by normalized title/year/catalog number. That gives ~1 request per
artist instead of a request per release.

An original candidate must be a lossless edition, must not be someone else's
restoration, must contain the album title and (ideally) match by year and by
the catalog number from "Source edition".
"""
import re

_STOP = {
    "the", "a", "an", "and", "of", "in", "on", "to", "for", "with",
    "pt", "part", "cd", "disc", "disk", "vol", "volume", "ep",
    "deluxe", "remaster", "remastered", "edition", "expanded", "anniversary",
    "bonus", "japanese", "japan", "us", "uk", "eu", "reissue", "special",
    "ost", "soundtrack", "flac", "lossless", "ape", "image", "cue", "tracks",
}

_BRACKETS = re.compile(r"[\[\(].*?[\]\)]")
_NONWORD = re.compile(r"[^\w\s]+", re.UNICODE)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

# \u0440\u0435\u0441\u0442\u0430\u0432\u0440 = ru "restoration" stem
OWN_MARK = re.compile(r"\[RM\]|restored|declipped|\u0440\u0435\u0441\u0442\u0430\u0432\u0440|Declipped", re.IGNORECASE)
LOSSY = re.compile(r"\bMP3\b|\bAAC\b|\bM4A\b|\bOGG\b|V0|V2|~?320", re.IGNORECASE)
HIRES = re.compile(r"24/\d|24-?bit|\[TR24\]|\[Hi-?Res\]|\[\d{2}/\d{3}\]|DSD|MQA", re.IGNORECASE)
# ru title words: "Discography" / "Compilation"
GRAPH = re.compile(r"\u0414\u0438\u0441\u043A\u043E\u0433\u0440\u0430\u0444\u0438\u044F|Discography|Studio Albums|\(releases\)|Collection|\u0421\u0431\u043E\u0440\u043D\u0438\u043A",
                   re.IGNORECASE)
# \u0412\u0438\u043D\u0438\u043B = ru "Vinyl"
BAD_MEDIA = re.compile(r"\[\d{0,2}LP\]|\bLPCM\b|\[SACD\]|\[DVD|\[VHS|\bVinyl\b|\u0412\u0438\u043D\u0438\u043B", re.IGNORECASE)
# ru: "live concert" / "bootleg"
BOOTLEG = re.compile(r"CAMRip|Bootleg|\[Unofficial\]|\u043A\u043E\u043D\u0446\u0435\u0440\u0442|\u0441\u0430\u043C\u043E\u043F\u0430\u043B", re.IGNORECASE)
EDITION = re.compile(r"Deluxe|Expanded|Remaster|Anniversary|Edition", re.IGNORECASE)
# Editions that change the CONTENT: acapellas, instrumentals, cover versions —
# not a pair for a studio album even if the title and year match.
CONTENT_ALT = re.compile(
    r"acapella|a cappella|\u0438\u043D\u0441\u0442\u0440\u0443\u043C\u0435\u043D\u0442\u0430\u043B|instrumental|karaoke|\u043A\u0430\u0440\u0430\u043E\u043A\u0435|remix|\u0440\u0435\u043C\u0438\u043A\u0441\u0435\u0440|"
    r"cover|\u043A\u0430\u0432\u0435\u0440|tribute|\u0442\u0440\u0438\u0431\u044C\u044E\u0442|demo|\u0434\u0435\u043C\u043E|backing track|playback", re.IGNORECASE)
CONTENT_ALT_SOFT = re.compile(r"radio edit|single|promo|sampler|split|hits|best of|compilation",
                              re.IGNORECASE)
_TAIL_SEP = re.compile(r"\s+[-–—]\s+")


def _title_parts(title: str) -> tuple[str, str]:
    """('artist', 'album') from a torrent title: split on the first " - "."""
    t = _BRACKETS.sub(" ", title or "")
    parts = _TAIL_SEP.split(t)
    if len(parts) > 1:
        return parts[0], " ".join(parts[1:])
    return t, ""



def tokens(s: str) -> list[str]:
    # \u0451 -> \u0435: fold ru "yo" into "e" for normalization
    s = (s or "").lower().replace("\u0451", "\u0435")
    s = _BRACKETS.sub(" ", s)
    s = _YEAR.sub(" ", s)
    s = _NONWORD.sub(" ", s)
    return [w for w in s.split() if len(w) > 1 and w not in _STOP]


def containment(need: list[str], have: list[str]) -> float:
    """What fraction of `need` tokens are present in `have` (0..1)."""
    A, B = set(need), set(have)
    if not A:
        return 0.0
    return len(A & B) / len(A)


def parse_restore_title(title: str) -> dict:
    """'[RM] [restored] (Heavy Metal) Accept - Discography (24 releases) - 1985-2022, FLAC'."""
    t = re.sub(r"^\s*(?:\[[^\]]*\]\s*)+", "", title or "")
    genre = None
    m = re.match(r"\(([^)]*)\)\s*", t)
    if m:
        genre = m.group(1).strip()
        t = t[m.end():]
    parts = [p.strip() for p in t.split(" - ")]
    artist = parts[0] if parts else t
    artist = re.sub(r"\s*\([^)]*\)\s*$", "", artist).strip()
    years = _YEAR.findall(title or "")
    return {
        "genre": genre,
        "artist": artist,
        "artist_tokens": tokens(artist),
        "title_years": [int(y) for y in years],
        "is_graphy": bool(GRAPH.search(t)),
    }


def edition_codes(source_edition: str | None) -> list[str]:
    """Catalog numbers from "Source edition: Noise – N 0061"."""
    if not source_edition:
        return []
    # "Music Theories Recordings – MTR 7562 2" -> 'MTR 7562 2', 'MTR75622', 'MTR'
    raw = source_edition.replace("\u200e", " ")
    raw = re.sub(r"^\s*[^-–]+?[-–]\s*", "", raw)      # cut the label before the dash
    raw = raw.strip()
    out = {raw}
    out.add(raw.replace(" ", ""))
    for m in re.finditer(r"[A-Z]{1,6}\s?-?\s?\d{2,6}(?:[-~/]\d{1,6})?", raw):
        out.add(m.group(0).strip())
        out.add(m.group(0).replace(" ", ""))
    return [c for c in out if len(c) >= 4]


def title_year(title: str) -> int | None:
    ys = _YEAR.findall(title or "")
    return int(ys[-1]) if ys else None


def score_candidate(row: dict, artist_tokens: list[str], album_tokens: list[str],
                    year: int | None, codes: list[str], own_poster: int | None) -> dict:
    """Score one candidate. Returns score + diagnostic flags."""
    title = row.get("title") or ""
    t_tokens = tokens(title)
    flags = []
    score = 0.0

    if OWN_MARK.search(title) and row.get("poster_pid") == own_poster:
        return {"score": -1000.0, "flags": ["own_restoration"], "contain": 0.0}
    if LOSSY.search(title):
        return {"score": -1000.0, "flags": ["lossy"], "contain": 0.0}
    if not re.search(r"FLAC|APE|WV|WavPack|lossless|ALAC", title, re.IGNORECASE):
        # not a lossless music release (movies, software, MP3 torrents)
        return {"score": -1000.0, "flags": ["not_lossless"], "contain": 0.0}

    head, tail = _title_parts(title)
    a_cont = containment(artist_tokens, tokens(head)) if artist_tokens else 0.0
    kind = "album"
    if not tail:
        # no " - " separator: look for both artist and album in the whole title
        a_cont = max(a_cont, containment(artist_tokens, t_tokens) if artist_tokens else 0.0)
        cont = containment(album_tokens, t_tokens)
        # ...but without signs of a lossless release such a title is noise
        if not re.search(r"\b(19|20)\d{2}\b|FLAC|APE|\bCD\b|lossless", title, re.IGNORECASE):
            return {"score": -1000.0, "flags": ["no_evidence"], "contain": cont}
    else:
        cont = containment(album_tokens, tokens(tail))
        if cont < 0.5:
            # no album in the title, but it is the artist's CD discography —
            # a valid (and often the only) source of CD editions:
            # the wanted disc is picked later from the torrent file list.
            if GRAPH.search(title) and a_cont >= 0.5:
                kind = "graphy"
                cont = 0.5
                flags.append("graphy_artist")
            else:
                return {"score": -1000.0, "flags": ["album_mismatch"], "contain": cont}
        elif a_cont < 0.5:
            return {"score": -1000.0, "flags": ["artist_part_mismatch"], "contain": cont}
    if cont < 0.5:
        return {"score": -1000.0, "flags": ["album_mismatch"], "contain": cont}

    if cont >= 0.999:
        score += 60
    elif cont >= 0.75:
        score += 45
    else:
        score += 20

    if a_cont >= 0.999:
        score += 25
    elif a_cont >= 0.5:
        score += 25 if kind == "graphy" else 12
    else:
        flags.append("artist_partial")

    ty = title_year(title)
    if len(album_tokens) == 1 and album_tokens == artist_tokens and cont >= 0.999:
        flags.append("self_titled")

    a_cont_legacy = containment(artist_tokens, t_tokens) if artist_tokens else 0.0

    if kind == "graphy":
        flags.append("year_na")            # a discography spans all years
    elif year and ty:
        if year == ty:
            score += 12
        elif abs(year - ty) <= 1:
            score += 6
        elif abs(year - ty) <= 3:
            pass
        else:
            score -= 8
            flags.append("year_far")
    elif year and not ty:
        flags.append("year_unknown")

    # vinyl/SACD/DVD editions are a different mastering — no good at all for a
    # pair of "the same edition the master restored".
    if BAD_MEDIA.search(title):
        return {"score": -1000.0, "flags": ["media_mismatch"], "contain": cont}

    # vinyl/hi-res editions are a DIFFERENT mastering — not suitable for a pair
    if re.search(r"\bFLAC\b|\bAPE\b|\bWV\b|WavPack", title, re.IGNORECASE):
        score += 10
    if re.search(r"image\s*\+?\s*\.?cue|cue", title, re.IGNORECASE):
        score += 8
        flags.append("image_cue")
    # an edition with different content (acapella/instrumental/cover) is not the original
    alt = CONTENT_ALT.search(title)
    if alt and kind != "graphy" and \
            not CONTENT_ALT.search(album_tokens and " ".join(album_tokens) or ""):
        # discographies often carry "covers/bonus/demo" in the title — that is
        # about the box contents; the specific album is checked at download
        # (folder + durations)
        return {"score": -1000.0, "flags": [f"content_alt:{alt.group(0).lower()}"],
                "contain": cont}
    if CONTENT_ALT_SOFT.search(title) and kind != "graphy":
        # "singles/hits/best of" in a discography title is about the box
        # contents, not about an alternative edition of the specific album
        score -= 30
        flags.append("alt_edition")
    if re.search(r"\[CD\]", title):
        score += 10
    if re.search(r"\(tracks", title, re.IGNORECASE):
        score += 4
    if HIRES.search(title):
        score -= 25
        flags.append("hires")
    if GRAPH.search(title) and kind != "graphy":
        score -= 20
        flags.append("graphy")
    if EDITION.search(title) and kind != "graphy":
        # "Editions" in a discography title is about box contents, not an edition
        score -= 25
        flags.append("edition")
    if BOOTLEG.search(title):
        score -= 40
        flags.append("bootleg")
    # a studio album cannot be a live recording
    if re.search(r"\bLive\b|Live At|Live In|\u043A\u043E\u043D\u0446\u0435\u0440\u0442", title, re.IGNORECASE) and \
            "live" not in album_tokens:
        score -= 30
        flags.append("live_mismatch")

    code_hit = [c for c in codes if c and c.lower() in title.lower()]
    if code_hit:
        score += 35
        flags.append("edition_code")

    seed = row.get("seeders") or 0
    if seed <= 0:
        score -= 60
        flags.append("dead")
    else:
        score += min(seed, 20) * 0.5

    return {"score": round(score, 1), "flags": flags, "kind": kind,
            "contain": round(cont, 3), "code_hit": code_hit[:2]}


MATCH_THRESHOLD = 80.0
GRAPHY_THRESHOLD = 68.0     # compilations/discographies: the wanted disc is checked at download


def choose(cands: list[dict], threshold: float = MATCH_THRESHOLD,
           kind: str | None = None) -> dict | None:
    alive = [c for c in cands if c["score"] > -900 and (kind is None or c.get("kind") == kind)]
    alive.sort(key=lambda c: (-c["score"], -(c.get("seeders") or 0)))
    if alive and alive[0]["score"] >= threshold:
        return alive[0]
    return None
