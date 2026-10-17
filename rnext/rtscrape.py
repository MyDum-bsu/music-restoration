"""Скрапер rutracker: реестр тем по мастер-профилям (статическая разметка tr[data-topic_id])."""
import re
import time

from bs4 import BeautifulSoup

from rnext.rtclient import RTClient

ROW_SEL = 'tr[data-topic_id]'


def _pid_of(a_user) -> int | None:
    if not a_user:
        return None
    m = re.search(r"pid=(\d+)", a_user.get("href", "") or "")
    return int(m.group(1)) if m else None


def _fid_of(a_forum) -> int | None:
    if not a_forum:
        return None
    m = re.search(r"f=(\d+)", a_forum.get("href", "") or "")
    return int(m.group(1)) if m else None


def _row_dict(tr) -> dict:
    topic_id = tr.get("data-topic_id")
    a_title = tr.select_one("a.tLink")
    a_forum = tr.select_one(".f-name a")
    a_user = tr.select_one(".u-name a")
    size_cell = tr.select_one(".tor-size")
    seed = tr.select_one(".seedmed")
    leech = tr.select_one(".leechmed")
    dl = tr.select_one("a.tr-dl")
    approved = bool(tr.select_one(".tor-approved"))
    return {
        "topic_id": int(topic_id),
        "title": a_title.get_text(strip=True) if a_title else None,
        "forum_id": _fid_of(a_forum),
        "forum": a_forum.get_text(strip=True) if a_forum else None,
        "poster_pid": _pid_of(a_user),
        "poster": a_user.get_text(strip=True) if a_user else None,
        "size_bytes": int(size_cell["data-ts_text"]) if (size_cell and size_cell.has_attr("data-ts_text")) else None,
        "size_label": size_cell.get_text(strip=True) if size_cell else None,
        "seeders": int(seed.get_text(strip=True)) if seed else None,
        "leechers": int(leech.get_text(strip=True)) if leech else None,
        "dl": dl["href"] if dl else None,
        "approved": approved,
    }


def poster_topics(pid: str, max_pages: int = 40, pause: float = 1.2, client: RTClient | None = None) -> list[dict]:
    """Все темы профиля (по 50 на страницу, пагинация start=50k)."""
    c = client or RTClient()
    out: dict[int, dict] = {}
    for page in range(max_pages):
        r = c.get(c.FORUM + f"tracker.php?pid={pid}&start={50 * page}")
        trs = BeautifulSoup(r.text, "html.parser").select(ROW_SEL)
        if not trs:
            break
        new = 0
        for tr in trs:
            d = _row_dict(tr)
            if d["topic_id"] and d["topic_id"] not in out:
                out[d["topic_id"]] = d
                new += 1
        if new == 0:
            break
        time.sleep(pause)
    return list(out.values())
