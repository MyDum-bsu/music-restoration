#!/usr/bin/env python3
"""Поиск по трекеру, разбор версий и выбор лучшей — не только для музыки.

  python3 scripts/tracker_hunt.py search "Jelusick"
  python3 scripts/tracker_hunt.py info 6415449
  python3 scripts/tracker_hunt.py compare 6415449 6414919 6415449
  python3 scripts/tracker_hunt.py fetch 6415449 --dir "Jelusick/Album [TR24]"

Профиль разбора подбирается по разделу темы: музыка, видео, книги, софт,
иначе — общий. Сравнение даёт Парето-набор и объяснение, а не «одна лучшая»:
у версий разные компромиссы (источник мастеринга, полнота, вес).

Проверка подлинности (фейковый hi-res, lossy-транскод) делается ПОСЛЕ закачки
скриптом audio_measure.py — по раздаче этого не видно.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bs4 import BeautifulSoup  # noqa: E402

from rnext.rtclient import RTClient  # noqa: E402

SIZE_RE = re.compile(r"([\d.,]+)\s*(GB|MB|KB|ТБ|ГБ|МБ|КБ)", re.I)
UNIT = {"GB": 1e9, "ГБ": 1e9, "MB": 1e6, "МБ": 1e6, "KB": 1e3, "КБ": 1e3, "ТБ": 1e12}


# ---------------------------------------------------------------- профили
def parse_music(text: str, title: str) -> dict:
    t = f"{title}\n{text}"
    low = t.lower()
    d: dict = {"kind": "music"}
    # источник
    if re.search(r"\[tr24\]|24bit|24-bit|hi-res|hi res", low):
        d["source"] = "hi-res (24 бита)"
    elif re.search(r"\[web\]|web[- ]?release|digital", low):
        d["source"] = "WEB (цифровой релиз)"
    elif re.search(r"\[cd\]|\bcd\b", low):
        d["source"] = "CD"
    elif re.search(r"\[lp\]|vinyl|винил|dsd|sacd", low):
        d["source"] = "винил/DSD"
    else:
        d["source"] = "?"
    d["vinyl"] = bool(re.search(r"\[lp\]|vinyl|винил|dsd|sacd|1/5[.,]64", low))
    # формат и лосслес
    if re.search(r"\bflac\b|\bape\b|\bwav\b|\balac\b", low):
        d["lossless"] = True
        d["format"] = (re.search(r"\b(flac|ape|wav|alac)\b", low) or [None, "?"])[1].upper()
    elif re.search(r"\bmp3\b|\baac\b|\bm4a\b|\bopus\b", low):
        d["lossless"] = False
        d["format"] = (re.search(r"\b(mp3|aac|m4a|opus)\b", low) or [None, "?"])[1].upper()
    else:
        d["lossless"] = bool(re.search(r"lossless|без потерь", low))
        d["format"] = "?"
    m = re.search(r"(\d{2,3})\s*(?:кбит|kbps|kbit)", low)
    if m:
        d["kbps"] = int(m.group(1))
    m = re.search(r"(\d{2})\s*[- ]?bit", low)
    if m and int(m.group(1)) in (16, 20, 24, 32):
        d["bits"] = int(m.group(1))
    m = re.search(r"(\d{2,3}(?:[.,]\d)?)\s*(?:кгц|khz|khz)", low)
    if m:
        d["khz"] = float(m.group(1).replace(",", "."))
    m = re.search(r"1/5[.,]64|dsd\s*(\d+)", low)
    if m:
        d["khz"] = 5644.8
    # структура
    d["image_cue"] = bool(re.search(r"image\+?\.?cue|image\+cue|образ", low))
    d["tracks"] = bool(re.search(r":\s*tracks|\btracks\b", low))
    d["has_log"] = bool(re.search(r"\blog\b|eac|accurip|xld", low))
    d["has_scans"] = bool(re.search(r"сканы|scans|booklet|буклет", low))
    # DR из отчёта (foobar DR / TT DR), если он в теме
    drs = [int(x) for x in re.findall(r"^\s*DR(\d{1,2})\s", text, re.M)]
    if drs:
        d["dr_published"] = sorted(drs)[len(drs) // 2]
    # альтернативы
    d["remaster"] = bool(re.search(r"ремастер|remaster|\[rm\]|restored", low))
    return d


def parse_video(text: str, title: str) -> dict:
    t = f"{title}\n{text}"
    low = t.lower()
    d: dict = {"kind": "video"}
    m = re.search(r"\b(2160p|1080p|720p|480p|4k|uhd)\b", low)
    d["resolution"] = (m.group(1) if m else "?").upper()
    if re.search(r"\bbdremux\b|bd-remux|remux", low):
        d["source"] = "BDRemux"
    elif re.search(r"bdrip|brrip|bd-rip", low):
        d["source"] = "BDRip"
    elif re.search(r"web-?dl|webdl", low):
        d["source"] = "WEB-DL"
    elif re.search(r"web-?rip|webrip", low):
        d["source"] = "WEBRip"
    elif re.search(r"hdtv|dvb", low):
        d["source"] = "HDTV"
    elif re.search(r"dvdrip|dvd-rip", low):
        d["source"] = "DVDRip"
    else:
        d["source"] = "?"
    for codec in ("x265", "hevc", "h.265", "x264", "h.264", "av1", "xvid"):
        if codec in low:
            d["codec"] = codec
            break
    else:
        d["codec"] = "?"
    d["hdr"] = "DV" if re.search(r"\bdv\b|dolby vision", low) else (
        "HDR" if re.search(r"hdr10\+?|hdr", low) else None)
    d["audio"] = [a for a in ("TrueHD", "DTS-HD MA", "DTS-HD", "DTS", "AC3", "E-AC3",
                              "DD+", "AAC", "FLAC", "LPCM", "Atmos")
                  if a.lower() in low]
    d["subs"] = [s for s in ("rus", "eng", "ukr") if re.search(rf"\b{s}\b", low)]
    d["stereo3d"] = bool(re.search(r"\b3d\b", low))
    return d


def parse_book(text: str, title: str) -> dict:
    t = f"{title}\n{text}"
    low = t.lower()
    d: dict = {"kind": "book"}
    for f in ("pdf", "epub", "djvu", "fb2", "mobi", "azw3", "txt"):
        if re.search(rf"\b{f}\b", low):
            d["format"] = f.upper()
            break
    else:
        d["format"] = "?"
    d["scanned"] = bool(re.search(r"скан|scan|ocr|изображен", low))
    d["text_layer"] = bool(re.search(r"текстов\w+ слой|ocr|text layer", low))
    return d


def parse_generic(text: str, title: str) -> dict:
    return {"kind": "generic"}


def guess_kind(forum: str, title: str) -> str:
    low = f"{forum} {title}".lower()
    # Сначала видео: у фильмов в названии есть разрешение/источник, а раздел
    # может называться «Музыка» из-за слова Score — иначе фильм уедет в музыку.
    # Маркеры фильма: разрешение/источник в заголовке ИЛИ кинополя в теле темы
    # (перевод/субтитры/продолжительность) — заголовок у rutracker часто обрезан.
    if re.search(r"\b(2160p|1080p|720p|480p|4k|uhd)\b|bdremux|bdrip|brrip|web-?dl|"
                 r"web-?rip|hdtv|dvdrip|blu-?ray|фильм|кино|сериал|документал|аниме|мульт|"
                 r"перевод|субтитры|продолжительность", low):
        return "video"
    if re.search(r"книг|литератур|book|учебник|аудиокниг|журнал", low):
        return "book"
    if re.search(r"музык|music|rock|metal|jazz|classical|поп|хип|electro|soundtrack|саундтрек", low):
        return "music"
    if re.search(r"программ|софт|software|игр|game|android|linux|windows", low):
        return "generic"
    return "music"


PARSERS = {"music": parse_music, "video": parse_video, "book": parse_book,
           "generic": parse_generic}


# ---------------------------------------------------------------- поиск
def search_rows(c: RTClient, query: str, pages: int = 1) -> list[dict]:
    out = []
    for pg in range(1, pages + 1):
        r = c.search(query, page=pg)
        s = BeautifulSoup(r.text, "html.parser")
        n = 0
        for tr in s.select("table#tor-tbl tr"):
            tds = tr.find_all("td")
            if len(tds) < 8:
                continue
            a = tr.find("a", class_="tLink") or tr.find("a", href=re.compile(r"viewtopic"))
            if not a:
                continue
            n += 1
            cells = [td.get_text(" ", strip=True) for td in tds]
            tid = re.search(r"t=(\d+)", a.get("href", "") or "")
            size_b = None
            m = SIZE_RE.search(cells[5].replace("\xa0", " "))
            if m:
                size_b = float(m.group(1).replace(",", ".")) * UNIT.get(m.group(2).upper(), 1)
            out.append({
                "topic": tid.group(1) if tid else "?",
                "title": a.get_text(" ", strip=True),
                "forum": cells[2],
                "size": cells[5].replace("\xa0", " "),
                "size_bytes": size_b,
                "seeders": int(cells[6]) if cells[6].isdigit() else None,
                "leechers": int(cells[7]) if cells[7].isdigit() else None,
                "date": cells[-1],
            })
        if n == 0:
            break
    return out


def topic_facts(c: RTClient, tid: str) -> dict:
    soup = BeautifulSoup(c.view_topic(tid).text, "html.parser")
    ttl = soup.select_one("#topic-title")
    title = ttl.get_text(" ", strip=True) if ttl else f"topic-{tid}"
    post = soup.select_one("div.post_body")
    text = post.get_text("\n", strip=True) if post else ""
    forum = ""
    fa = soup.find("a", href=re.compile(r"f=\d+"))
    if fa:
        forum = fa.get_text(" ", strip=True)
    kind = guess_kind(forum, title + " " + text[:2500])
    facts = PARSERS[kind](text, title)
    facts.update(topic=tid, title=title, forum=forum, text_len=len(text))
    # Размер из текста НЕ берём: в описании попадаются мелкие числа (размер
    # обложки, куски треклиста) и они врут. Точный размер даёт поиск — enrich_size.
    # .torrent-ссылка
    dl = soup.find("a", href=re.compile(r"dl\.php"))
    facts["has_torrent"] = bool(dl)
    return facts


# ---------------------------------------------------------------- оценка
def score(f: dict) -> tuple[float, list[str]]:
    """Оценка версии и человеческое объяснение. Больше — лучше."""
    s = 0.0
    why: list[str] = []
    if f["kind"] == "music":
        if f.get("lossless"):
            s += 100
            why.append("lossless")
        else:
            s -= 100
            why.append("lossy")
        src = f.get("source", "?")
        s += {"hi-res (24 бита)": 30, "WEB (цифровой релиз)": 25, "CD": 25,
              "винил/DSD": 5}.get(src, 0)
        if src != "?":
            why.append(f"источник: {src}")
        if f.get("vinyl"):
            why.append("винил — другой мастеринг, шум/рокот")
        if f.get("bits"):
            s += 3 if f["bits"] >= 24 else 1
        if f.get("dr_published"):
            s += min(f["dr_published"], 16) * 0.5
            why.append(f"DR в теме {f['dr_published']}")
        if f.get("has_log"):
            s += 3
            why.append("есть log")
        if f.get("has_scans"):
            s += 2
            why.append("есть сканы")
        if f.get("image_cue"):
            s -= 1
            why.append("image+cue (нужно разбивать)")
    elif f["kind"] == "video":
        res = {"2160P": 40, "4K": 40, "UHD": 40, "1080P": 25, "720P": 10}.get(
            f.get("resolution", "?"), 0)
        s += res
        why.append(f"разрешение {f.get('resolution')}")
        s += {"BDRemux": 30, "BDRip": 20, "WEB-DL": 22, "WEBRip": 12, "HDTV": 5}.get(
            f.get("source", "?"), 0)
        why.append(f"источник {f.get('source')}")
        if f.get("codec") in ("x265", "hevc", "h.265", "av1"):
            s += 5
        if f.get("hdr"):
            s += 8
            why.append(f.get("hdr"))
        if any(a in ("DTS-HD MA", "TrueHD", "LPCM", "Atmos") for a in f.get("audio", [])):
            s += 8
            why.append("lossless-звук")
        if "rus" in f.get("subs", []):
            s += 3
    else:
        s += 10 if f.get("format") not in ("?", None) else 0
        if f.get("text_layer"):
            s += 5
            why.append("текстовый слой")
    return s, why


def fmt_facts(f: dict) -> str:
    keys = [k for k in ("format", "source", "bits", "khz", "kbps", "resolution", "codec",
                        "hdr", "dr_published", "image_cue", "has_log", "has_scans", "vinyl")
            if f.get(k) not in (None, "", False)]
    return ", ".join(f"{k}={f[k]}" for k in keys)


def cmd_search(args) -> int:
    c = RTClient()
    rows = search_rows(c, args.query, pages=args.pages)
    if not rows:
        print("ничего не найдено")
        return 1
    print(f"{'t':<10}{'размер':>10}{'сид':>5}  заголовок")
    for r in rows:
        print(f"{r['topic']:<10}{r['size']:>10}{(r['seeders'] if r['seeders'] is not None else '?'):>5}  "
              f"{r['title'][:88]}")
    if args.facts:
        print("\nразбор версий:")
        for r in rows:
            try:
                f = topic_facts(c, r["topic"])
            except Exception as e:  # noqa: BLE001
                print(f"  t={r['topic']}: не разобрать ({type(e).__name__})")
                continue
            sc, why = score(f)
            print(f"  t={r['topic']:<9} {sc:>6.1f}  {fmt_facts(f)}")
    return 0


def cmd_info(args) -> int:
    c = RTClient()
    for tid in args.topic:
        f = topic_facts(c, tid)
        sc, why = score(f)
        print(f"\n=== t={tid} [{f['kind']}] {f['title'][:90]}")
        print(f"    раздел: {f['forum']}")
        print(f"    факты:  {fmt_facts(f) or '—'}")
        print(f"    оценка: {sc:.1f}  ({'; '.join(why) or '—'})")
    return 0


def enrich_size(c: RTClient, f: dict) -> dict:
    """Размер и сиды по теме: на странице темы их нет, берём из поиска."""
    if f.get("size_bytes") and f["size_bytes"] >= 5e6:
        return f
    # ключ для поиска — первые значимые слова заголовка без префиксов
    words = re.sub(r"[\[\]()]", " ", f["title"])
    words = re.sub(r"\b(tr24|of|ldr|cd|web|lp|rm|flac|mp3|lossless)\b", " ", words, flags=re.I)
    key = " ".join([w for w in words.split() if len(w) > 2][:4])
    if not key:
        return f
    try:
        for r in search_rows(c, key, pages=1):
            if r["topic"] == f["topic"]:
                f["size_bytes"] = r.get("size_bytes")
                f["size"] = r.get("size")
                f["seeders"] = r.get("seeders")
                break
    except Exception:  # noqa: BLE001
        pass
    return f


def cmd_compare(args) -> int:
    c = RTClient()
    facts = []
    for tid in args.topic:
        try:
            facts.append(enrich_size(c, topic_facts(c, tid)))
        except Exception as e:  # noqa: BLE001
            print(f"t={tid}: не разобрать ({e})")
    if not facts:
        return 1
    scored = sorted(((score(f)[0], f) for f in facts), key=lambda x: -x[0])
    print(f"{'t':<10}{'оценка':>7}  {'размер':>10}  факты")
    for sc, f in scored:
        size = f"{f['size_bytes'] / 1e9:.2f} GB" if f.get("size_bytes") else "?"
        seed = f.get("seeders")
        print(f"{f['topic']:<10}{sc:>7.1f}  {size:>10}  сид {seed if seed is not None else '?':>3}  {fmt_facts(f)}")
    best = scored[0][1]
    print(f"\nлучший по совокупности: t={best['topic']} — {best['title'][:80]}")
    print(f"  причина: {'; '.join(score(best)[1])}")
    if len(scored) > 1:
        second = scored[1][1]
        print(f"  альтернатива: t={second['topic']} ({'; '.join(score(second)[1])})")
    print("\nНапоминание: подлинность (фейковый hi-res, lossy-транскод) по раздаче "
          "не видна — проверять после закачки: scripts/audio_measure.py <каталог>")
    return 0


def cmd_fetch(args) -> int:
    script = Path(__file__).resolve().parent / "fetch_release.py"
    for tid in args.topic:
        cmd = [sys.executable, str(script), "--topic", tid, "--priority", args.priority]
        if args.dir:
            cmd += ["--dir", args.dir]
        print(f"$ {' '.join(cmd[1:])}")
        subprocess.run(cmd, check=False)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--pages", type=int, default=1)
    s.add_argument("--facts", action="store_true", help="разобрать каждую версию")
    s.set_defaults(func=cmd_search)

    i = sub.add_parser("info")
    i.add_argument("topic", nargs="+")
    i.set_defaults(func=cmd_info)

    cp = sub.add_parser("compare")
    cp.add_argument("topic", nargs="+")
    cp.set_defaults(func=cmd_compare)

    f = sub.add_parser("fetch")
    f.add_argument("topic", nargs="+")
    f.add_argument("--dir", default="")
    f.add_argument("--priority", choices=("low", "normal", "high"), default="high")
    f.set_defaults(func=cmd_fetch)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
