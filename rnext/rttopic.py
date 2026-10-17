"""Парсер темы руТрекера: мета + раскрывание per-release блоков ([RM] дискографии)."""
import re

from bs4 import BeautifulSoup

from rnext.rtclient import RTClient

_WS = re.compile(r"\s+")

META_KEYS = ["Жанр", "Год выпуска", "Аудио кодек", "Тип рипа", "Формат раздачи",
             "Продолжительность", "Наличие сканов", "Битрейт"]


def _txt(el) -> str:
    return _WS.sub(" ", el.get_text(" ", strip=True))


def _txt_lines(el) -> str:
    """Текст с сохранением переводов строк (для блоков «Код:» с таблицами)."""
    raw = el.get_text("\n")
    raw = raw.replace("\xa0", " ")
    return "\n".join(line.strip() for line in raw.splitlines())


# строка таблицы DR-метра: «DR11      -1.39 dB   -13.72 dB     2:08 01-Carpathia»
_DR_ROW = re.compile(
    r"^DR\s*(\d+)\s+(-?\d+(?:[.,]\d+)?)\s*dB\s+(-?\d+(?:[.,]\d+)?)\s*dB\s+(\d+:\d+)\s+(.+)$"
)
_DR_HEAD = re.compile(r"Динамический диапазон|DR-?метр|АЧХ|Спектр", re.IGNORECASE)


def parse_dr_table(text: str) -> dict | None:
    """Разбор вставки DR-метра из темы мастера.

    Мастер публикует анализ СВОЕЙ (уже реставрированной) версии: DR/Peak/RMS
    по каждому треку. Это даёт цель без скачивания мастер-релиза:
    crest_tt = peak − rms.
    """
    rows = []
    analyzed = None
    for line in text.splitlines():
        line = _WS.sub(" ", line).strip()
        m = re.match(r"^Analyzed:\s*(.+)$", line)
        if m:
            analyzed = m.group(1).strip()
            continue
        m = _DR_ROW.match(line)
        if not m:
            continue
        num = re.match(r"^(\d{1,3})[-.\s]", m.group(5).strip())
        rows.append({
            "n": int(num.group(1)) if num else None,
            "dr": int(m.group(1)),
            "peak_db": float(m.group(2).replace(",", ".")),
            "rms_db": float(m.group(3).replace(",", ".")),
            "dur": m.group(4),
            "title": re.sub(r"^\d{1,3}[-.\s]+", "", m.group(5).strip()),
        })
    if not rows:
        return None
    official = None
    m = re.search(r"Official DR value:\s*DR\s*(\d+)", text)
    if m:
        official = int(m.group(1))
    return {"analyzed": analyzed, "official_dr": official, "tracks": rows}


def parse_topic_page(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.title.get_text() if soup.title else ""
    title = title_tag.split("::")[0].strip()
    body = soup.select_one(".post_body")
    out = {
        "title": title,
        "meta": {},
        "source_search_urls": [],
        "releases": [],
    }
    if not body:
        return out

    raw_first = str(body)

    # --- мета (до первого sp-wrap): span.post-b -> следующий текст
    text = _txt(body)
    for key in META_KEYS:
        m = re.search(rf"{re.escape(key)}\s*:\s*([^|​:]+?)(?=\s+(?:[А-ЯA-Z][a-zа-яё-]+\s*:|\Z)|\s+\Z)", text)
        if m:
            out["meta"][key] = m.group(1).strip(" :")

    # --- ссылки-исходники (вне спо́йлеров): 'Исходник для обработки : здесь'
    for a in body.select("a.postLink"):
        if a.get_text(strip=True).lower() in ("здесь", "here"):
            out["source_search_urls"].append(a.get("href"))

    # --- per-release: спо́йлеры sp-wrap
    out["info_blocks"] = []
    wraps = body.select(".sp-wrap")
    for w in wraps:
        head = w.select_one(".sp-head")
        b = w.select_one(".sp-body")
        if not head or not b:
            continue
        head_text = _txt(head)
        if _DR_HEAD.search(head_text):
            # Блок анализа идёт сразу за своим релизом → цепляем таблицу к нему:
            # это цель мастера по DR/Peak/RMS без скачивания мастер-релиза.
            table = parse_dr_table(_txt_lines(b))
            if table and out["releases"]:
                out["releases"][-1]["master_dr"] = table
            elif table:
                out["info_blocks"].append({"head": head_text, "table": table})
            continue
        if re.search(r"Дополнительная информация", head_text):
            out["info_blocks"].append({"head": head_text, "text": _txt(b)[:800]})
            continue
        m = re.match(r"(\d{4})\s*-\s*(.+)", head_text)
        rel = {
            "year": int(m.group(1)) if m else None,
            "album": m.group(2).strip() if m else head_text,
            "source_edition": None,
            "processing": None,
            "tracks": [],
            "master_dr": None,
        }
        btxt = _txt(b)
        m = re.search(r"Исходное издание\s*:\s*([^И]+?)(?=\s+Обработка|$)", btxt)
        if m:
            rel["source_edition"] = m.group(1).strip(" :")
        m = re.search(r"Обработка\s*:\s*(.+?)(?=\s+Спектр|$)", btxt)
        if m:
            rel["processing"] = m.group(1).strip(" :")
        for mm in re.finditer(r"\b(\d{1,2})\.\s*([^.|​:]*?)\s*\((\d{1,2}:\d{2})\)", btxt):
            rel["tracks"].append({"n": int(mm.group(1)), "s": mm.group(2).strip(), "dur": mm.group(3)})
        out["releases"].append(rel)

    # --- без спо́йлеров: одиночный релиз из заголовка темы
    if not wraps:
        rel = {"year": None, "album": title, "source_edition": None, "processing": None,
               "tracks": [], "master_dr": None}
        m = re.search(r"Исходное издание\s*([^И]+?)(?=\s+Обработка|$)", text)
        if m:
            rel["source_edition"] = m.group(1).strip(" -:").strip()
        out["releases"].append(rel)
        m = re.match(r".*?(?:(19|20)\d{2}).*?", title)
        ym = re.findall(r"(?<!\d)(\d{4})(?!\d)", title)
        if ym:
            rel["year"] = int(ym[-1])
    return out


def fetch_topic(client: RTClient, topic_id: str) -> dict:
    r = client.view_topic(topic_id)
    d = parse_topic_page(r.text)
    d["topic_id"] = int(topic_id)
    return d
