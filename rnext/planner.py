"""Планирование датасета: релиз мастера → кандидаты-оригиналы → решение.

Отличие от `pairsearch` (один поиск на один релиз): здесь пул кандидатов
берётся ОДИН раз на артиста (поиск rutracker по имени), а релизы матчатся
локально по нормализованному названию/году/каталожному номеру. Это даёт
~1 запрос на артиста вместо запроса на каждый релиз.

Кандидат-оригинал обязан быть lossless-изданием, не быть чужой
реставрацией, содержать название альбома и (в идеале) совпадать по году и
каталожному номеру из «Исходное издание».
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

OWN_MARK = re.compile(r"\[RM\]|restored|declipped|реставр|Declipped", re.IGNORECASE)
LOSSY = re.compile(r"\bMP3\b|\bAAC\b|\bM4A\b|\bOGG\b|V0|V2|~?320", re.IGNORECASE)
HIRES = re.compile(r"24/\d|24-?bit|\[TR24\]|\[Hi-?Res\]|\[\d{2}/\d{3}\]|DSD|MQA", re.IGNORECASE)
GRAPH = re.compile(r"Дискография|Discography|Studio Albums|\(releases\)|Collection|Сборник",
                   re.IGNORECASE)
BAD_MEDIA = re.compile(r"\[\d{0,2}LP\]|\bLPCM\b|\[SACD\]|\[DVD|\[VHS|\bVinyl\b|Винил", re.IGNORECASE)
BOOTLEG = re.compile(r"CAMRip|Bootleg|\[Unofficial\]|концерт|самопал", re.IGNORECASE)
EDITION = re.compile(r"Deluxe|Expanded|Remaster|Anniversary|Edition", re.IGNORECASE)
# Издания, меняющие СОДЕРЖАНИЕ: акапеллы, инструменталы, кавер-версии — не пара
# для студийного альбома, даже если название и год совпадают.
CONTENT_ALT = re.compile(
    r"acapella|a cappella|инструментал|instrumental|karaoke|караоке|remix|ремиксер|"
    r"cover|кавер|tribute|трибьют|demo|демо|backing track|playback", re.IGNORECASE)
CONTENT_ALT_SOFT = re.compile(r"radio edit|single|promo|sampler|split|hits|best of|compilation",
                              re.IGNORECASE)
_TAIL_SEP = re.compile(r"\s+[-–—]\s+")


def _title_parts(title: str) -> tuple[str, str]:
    """('артист', 'альбом') из заголовка раздачи: режем по первому « - »."""
    t = _BRACKETS.sub(" ", title or "")
    parts = _TAIL_SEP.split(t)
    if len(parts) > 1:
        return parts[0], " ".join(parts[1:])
    return t, ""



def tokens(s: str) -> list[str]:
    s = (s or "").lower().replace("ё", "е")
    s = _BRACKETS.sub(" ", s)
    s = _YEAR.sub(" ", s)
    s = _NONWORD.sub(" ", s)
    return [w for w in s.split() if len(w) > 1 and w not in _STOP]


def containment(need: list[str], have: list[str]) -> float:
    """Какая доля токенов `need` присутствует в `have` (0..1)."""
    A, B = set(need), set(have)
    if not A:
        return 0.0
    return len(A & B) / len(A)


def parse_restore_title(title: str) -> dict:
    """'[RM] [restored] (Heavy Metal) Accept - Дискография (24 releases) - 1985-2022, FLAC'."""
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
    """Каталожные номера из «Исходное издание: Noise – N 0061»."""
    if not source_edition:
        return []
    # «Music Theories Recordings – MTR 7562 2» → 'MTR 7562 2', 'MTR75622', 'MTR'
    raw = source_edition.replace("\u200e", " ")
    raw = re.sub(r"^\s*[^-–]+?[-–]\s*", "", raw)      # отрезаем лейбл до тире
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
    """Скоринг одного кандидата. Возвращает score + диагностические флаги."""
    title = row.get("title") or ""
    t_tokens = tokens(title)
    flags = []
    score = 0.0

    if OWN_MARK.search(title) and row.get("poster_pid") == own_poster:
        return {"score": -1000.0, "flags": ["own_restoration"], "contain": 0.0}
    if LOSSY.search(title):
        return {"score": -1000.0, "flags": ["lossy"], "contain": 0.0}
    if not re.search(r"FLAC|APE|WV|WavPack|lossless|ALAC", title, re.IGNORECASE):
        # не музыкальный lossless-релиз (фильмы, софт, MP3-раздачи)
        return {"score": -1000.0, "flags": ["not_lossless"], "contain": 0.0}

    head, tail = _title_parts(title)
    a_cont = containment(artist_tokens, tokens(head)) if artist_tokens else 0.0
    kind = "album"
    if not tail:
        # без разделителя « - »: и артиста, и альбом ищем в целом заголовке
        a_cont = max(a_cont, containment(artist_tokens, t_tokens) if artist_tokens else 0.0)
        cont = containment(album_tokens, t_tokens)
        # ...но без признаков lossless-релиза такой заголовок — шум
        if not re.search(r"\b(19|20)\d{2}\b|FLAC|APE|\bCD\b|lossless", title, re.IGNORECASE):
            return {"score": -1000.0, "flags": ["no_evidence"], "contain": cont}
    else:
        cont = containment(album_tokens, tokens(tail))
        if cont < 0.5:
            # нет альбома в заголовке, но это CD-дискография артиста —
            # валидный (и часто единственный) источник CD-изданий:
            # нужный диск выбирается уже по списку файлов раздачи.
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
        flags.append("year_na")            # дискография охватывает все годы
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

    # винил/SACD/DVD-издания — другой мастеринг, для пары «то же издание, что
    # реставрировал мастер» не годятся вовсе.
    if BAD_MEDIA.search(title):
        return {"score": -1000.0, "flags": ["media_mismatch"], "contain": cont}

    # виниловые/хайрез-издания — это ДРУГОЙ мастеринг, для пары не годятся
    if re.search(r"\bFLAC\b|\bAPE\b|\bWV\b|WavPack", title, re.IGNORECASE):
        score += 10
    if re.search(r"image\s*\+?\s*\.?cue|cue", title, re.IGNORECASE):
        score += 8
        flags.append("image_cue")
    # издание с другим содержимым (акапеллы/инструментал/кавер) — не оригинал
    alt = CONTENT_ALT.search(title)
    if alt and kind != "graphy" and \
            not CONTENT_ALT.search(album_tokens and " ".join(album_tokens) or ""):
        # дискографии часто содержат в названии «covers/bonus/demo» — это про
        # состав бокса; конкретный альбом проверяется при загрузке (папка + длительности)
        return {"score": -1000.0, "flags": [f"content_alt:{alt.group(0).lower()}"],
                "contain": cont}
    if CONTENT_ALT_SOFT.search(title) and kind != "graphy":
        # «singles/hits/best of» в названии дискографии — это про состав бокса,
        # а не про альтернативное издание конкретного альбома
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
        # у дискографии «Editions» в названии — это про состав бокса, не про издание
        score -= 25
        flags.append("edition")
    if BOOTLEG.search(title):
        score -= 40
        flags.append("bootleg")
    # студийный альбом не может быть концертником
    if re.search(r"\bLive\b|Live At|Live In|концерт", title, re.IGNORECASE) and \
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
GRAPHY_THRESHOLD = 68.0     # сборники/дискографии: нужный диск проверяется при загрузке


def choose(cands: list[dict], threshold: float = MATCH_THRESHOLD,
           kind: str | None = None) -> dict | None:
    alive = [c for c in cands if c["score"] > -900 and (kind is None or c.get("kind") == kind)]
    alive.sort(key=lambda c: (-c["score"], -(c.get("seeders") or 0)))
    if alive and alive[0]["score"] >= threshold:
        return alive[0]
    return None
