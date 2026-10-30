"""rutracker topic parser: metadata + per-release blocks ([RM] discographies)."""
import re

from bs4 import BeautifulSoup

from rnext.rtclient import RTClient

_WS = re.compile(r"\s+")

# Exact site labels (ru: Genre, Release year, Audio codec, Rip type, Release
# format, Duration, Scans present, Bitrate) — kept byte-identical via \u escapes.
META_KEYS = ["\u0416\u0430\u043D\u0440", "\u0413\u043E\u0434 \u0432\u044B\u043F\u0443\u0441\u043A\u0430",
             "\u0410\u0443\u0434\u0438\u043E \u043A\u043E\u0434\u0435\u043A", "\u0422\u0438\u043F \u0440\u0438\u043F\u0430",
             "\u0424\u043E\u0440\u043C\u0430\u0442 \u0440\u0430\u0437\u0434\u0430\u0447\u0438",
             "\u041F\u0440\u043E\u0434\u043E\u043B\u0436\u0438\u0442\u0435\u043B\u044C\u043D\u043E\u0441\u0442\u044C",
             "\u041D\u0430\u043B\u0438\u0447\u0438\u0435 \u0441\u043A\u0430\u043D\u043E\u0432",
             "\u0411\u0438\u0442\u0440\u0435\u0439\u0442"]


def _txt(el) -> str:
    return _WS.sub(" ", el.get_text(" ", strip=True))


def _txt_lines(el) -> str:
    """Text with line breaks preserved (for the "Code:" blocks with tables)."""
    raw = el.get_text("\n")
    raw = raw.replace("\xa0", " ")
    return "\n".join(line.strip() for line in raw.splitlines())


# DR-meter table row: "DR11      -1.39 dB   -13.72 dB     2:08 01-Carpathia"
_DR_ROW = re.compile(
    r"^DR\s*(\d+)\s+(-?\d+(?:[.,]\d+)?)\s*dB\s+(-?\d+(?:[.,]\d+)?)\s*dB\s+(\d+:\d+)\s+(.+)$"
)
# head words (ru): "Dynamic range" / "DR-meter" / "Freq response" / "Spectrum"
_DR_HEAD = re.compile(
    r"\u0414\u0438\u043D\u0430\u043C\u0438\u0447\u0435\u0441\u043A\u0438\u0439 \u0434\u0438\u0430\u043F\u0430\u0437\u043E\u043D"
    r"|DR-?\u043C\u0435\u0442\u0440|\u0410\u0427\u0425|\u0421\u043F\u0435\u043A\u0442\u0440", re.IGNORECASE)


def parse_dr_table(text: str) -> dict | None:
    """Parse the DR-meter insert from a master topic.

    The master publishes an analysis of THEIR OWN (already restored) version:
    DR/Peak/RMS per track. This gives the target without downloading the
    master release: crest_tt = peak - rms.
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

    # --- metadata (before the first sp-wrap): span.post-b -> following text
    text = _txt(body)
    for key in META_KEYS:
        # the lookahead matches the next "Capitalized label:" (ru/en ranges) or EOL
        m = re.search(rf"{re.escape(key)}\s*:\s*([^|​:]+?)(?=\s+(?:[\u0410-\u042FA-Z][a-z\u0430-\u044F\u0451-]+\s*:|\Z)|\s+\Z)", text)
        if m:
            out["meta"][key] = m.group(1).strip(" :")

    # --- source links (outside spoilers): 'source for processing : here'
    for a in body.select("a.postLink"):
        # \u0437\u0434\u0435\u0441\u044C = ru "here" (exact link text on the site)
        if a.get_text(strip=True).lower() in ("\u0437\u0434\u0435\u0441\u044C", "here"):
            out["source_search_urls"].append(a.get("href"))

    # --- per-release: sp-wrap spoilers
    out["info_blocks"] = []
    wraps = body.select(".sp-wrap")
    for w in wraps:
        head = w.select_one(".sp-head")
        b = w.select_one(".sp-body")
        if not head or not b:
            continue
        head_text = _txt(head)
        if _DR_HEAD.search(head_text):
            # The analysis block follows its release -> attach the table to it:
            # the master's DR/Peak/RMS target without downloading the release.
            table = parse_dr_table(_txt_lines(b))
            if table and out["releases"]:
                out["releases"][-1]["master_dr"] = table
            elif table:
                out["info_blocks"].append({"head": head_text, "table": table})
            continue
        # ru: "Additional information"
        if re.search(r"\u0414\u043E\u043F\u043E\u043B\u043D\u0438\u0442\u0435\u043B\u044C\u043D\u0430\u044F \u0438\u043D\u0444\u043E\u0440\u043C\u0430\u0446\u0438\u044F", head_text):
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
        # field labels (ru): "Source edition" / "Processing" / "Spectrum"
        m = re.search(
            r"\u0418\u0441\u0445\u043E\u0434\u043D\u043E\u0435 \u0438\u0437\u0434\u0430\u043D\u0438\u0435\s*:\s*([^\u0418]+?)(?=\s+\u041E\u0431\u0440\u0430\u0431\u043E\u0442\u043A\u0430|$)",
            btxt)
        if m:
            rel["source_edition"] = m.group(1).strip(" :")
        m = re.search(r"\u041E\u0431\u0440\u0430\u0431\u043E\u0442\u043A\u0430\s*:\s*(.+?)(?=\s+\u0421\u043F\u0435\u043A\u0442\u0440|$)", btxt)
        if m:
            rel["processing"] = m.group(1).strip(" :")
        for mm in re.finditer(r"\b(\d{1,2})\.\s*([^.|​:]*?)\s*\((\d{1,2}:\d{2})\)", btxt):
            rel["tracks"].append({"n": int(mm.group(1)), "s": mm.group(2).strip(), "dur": mm.group(3)})
        out["releases"].append(rel)

    # --- no spoilers: a single release taken from the topic title
    if not wraps:
        rel = {"year": None, "album": title, "source_edition": None, "processing": None,
               "tracks": [], "master_dr": None}
        # ru: "Source edition" / "Processing"
        m = re.search(
            r"\u0418\u0441\u0445\u043E\u0434\u043D\u043E\u0435 \u0438\u0437\u0434\u0430\u043D\u0438\u0435\s*([^\u0418]+?)(?=\s+\u041E\u0431\u0440\u0430\u0431\u043E\u0442\u043A\u0430|$)",
            text)
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
