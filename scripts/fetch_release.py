#!/usr/bin/env python3
"""Закачка релиза с трекера в отдельный каталог (мимо датасетного сырья).

Зачем отдельный путь: цикл обогащения датасета чистит раздачи только под
`dataset/raw/`, поэтому тестовые релизы в `release_src/` он не тронет.

  python3 scripts/fetch_release.py --topic 6415449 --dir "Jelusick/Follow The Blind Man (2023)"
  python3 scripts/fetch_release.py --topic 6499624 --priority high --wait
  python3 scripts/fetch_release.py --list
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bs4 import BeautifulSoup  # noqa: E402

from rnext import workspace  # noqa: E402
from rnext.rtclient import RTClient  # noqa: E402
from rnext.transm import Transm  # noqa: E402

PRIO = {"low": -1, "normal": 0, "high": 1}


def release_root() -> Path:
    p = workspace.data_root() / "release_src"
    p.mkdir(parents=True, exist_ok=True)
    return p


def torrent_dir() -> Path:
    p = release_root() / "_torrents"
    p.mkdir(parents=True, exist_ok=True)
    return p


def topic_title(c: RTClient, tid: str) -> str:
    s = BeautifulSoup(c.view_topic(tid).text, "html.parser")
    t = s.select_one("#topic-title")
    return (t.get_text(" ", strip=True) if t else f"topic-{tid}")


def torrent_url(c: RTClient, tid: str) -> str:
    """dl.php-ссылка со страницы темы."""
    s = BeautifulSoup(c.view_topic(tid).text, "html.parser")
    a = s.find("a", href=re.compile(r"dl\.php"))
    if not a:
        raise RuntimeError(f"в теме {tid} нет ссылки dl.php")
    href = a["href"]
    if href.startswith("http"):
        return href
    # rutracker отдаёт относительные ссылки: и "dl.php?...", и "/forum/dl.php?..."
    return "https://rutracker.org/forum/" + href.lstrip("/")


def fetch_one(tid: str, subdir: str, priority: str, start: bool = True) -> dict:
    c = RTClient()
    title = topic_title(c, tid)
    tp = torrent_dir() / f"rutracker-{tid}.torrent"
    if not tp.is_file():
        c.download_torrent(torrent_url(c, tid), str(tp))
    dest = release_root() / subdir
    dest.mkdir(parents=True, exist_ok=True)
    t = Transm()
    # Дубли определяет САМ Transmission по infohash. Проверка «по имени» была
    # багом: имя «Q5» совпадало с другим релизом, и раздача не добавлялась.
    import hashlib
    r = t.add_torrent_file(str(tp), str(dest), stopped=not start)
    added = r.get("torrent-added") or {}
    dup = r.get("torrent-duplicate") or {}
    tid_ = (added or dup).get("id")
    if dup and not added:
        cur = next((x for x in t.torrents(["id", "downloadDir"]) if x["id"] == tid_), {})
        print(f"  уже был (t={tid_}, {cur.get('downloadDir', '?')}) — приоритет обновлён")
    else:
        print(f"  добавлено: t={tid_} → {dest}")
    t._call("torrent-set", {"ids": [tid_], "bandwidthPriority": PRIO[priority]})
    if start:
        t._call("torrent-start", {"ids": [tid_]})
    return {"topic": tid, "title": title, "transmission_id": tid_, "dir": str(dest),
            "priority": priority, "_hash": hashlib.sha1(tp.read_bytes()).hexdigest()[:12]}


def add_local(src: Path, subdir: str, priority: str) -> dict:
    """Добавить уже скачанный .torrent (трекер недоступен)."""
    if not src.is_file():
        raise FileNotFoundError(src)
    import shutil
    tp = torrent_dir() / src.name
    shutil.copy2(src, tp)
    dest = release_root() / (subdir or src.stem[:60])
    dest.mkdir(parents=True, exist_ok=True)
    t = Transm()
    r = t.add_torrent_file(str(tp), str(dest), stopped=False)
    tid = (r.get("torrent-added") or r.get("torrent-duplicate") or {}).get("id")
    t._call("torrent-set", {"ids": [tid], "bandwidthPriority": PRIO[priority]})
    t._call("torrent-start", {"ids": [tid]})
    name = next((x["name"] for x in t.torrents(["id", "name"]) if x["id"] == tid), "")
    print(f"  добавлено из файла: t={tid} «{name[:56]}» → {dest}")
    return {"topic": None, "title": name, "transmission_id": tid, "dir": str(dest),
            "priority": priority, "source": str(src)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", action="append", default=[], help="id темы (можно несколько)")
    ap.add_argument("--file", action="append", default=[],
                    help="локальный .torrent (когда трекер на профилактике)")
    ap.add_argument("--dir", default="", help="подкаталог в release_src")
    ap.add_argument("--priority", choices=sorted(PRIO), default="high")
    ap.add_argument("--wait", action="store_true", help="ждать завершения")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        t = Transm()
        for x in t.torrents(["id", "name", "downloadDir", "percentDone", "leftUntilDone",
                             "peersConnected", "rateDownload", "status", "bandwidthPriority"]):
            d = x.get("downloadDir") or ""
            if "release_src" not in d:
                continue
            st = {0: "stop", 3: "queue", 4: "dl", 6: "seed"}.get(x["status"], x["status"])
            print(f"  t{x['id']:<5} {x['percentDone'] * 100:5.1f}% left {x['leftUntilDone'] / 1e6:7.1f}MB "
                  f"пиры {x['peersConnected']:>2} {x['rateDownload'] / 1e6:5.2f}MB/s {st:<5} "
                  f"prio {x['bandwidthPriority']:>2} | {x['name'][:46]}")
        return 0

    if not args.topic and not args.file:
        ap.error("нужен --topic, --file или --list")
    added = []
    for tid in args.topic:
        sub = args.dir or tid
        added.append(fetch_one(tid, sub, args.priority))
    for f in args.file:
        added.append(add_local(Path(f).expanduser(), args.dir, args.priority))
    if args.wait:
        t = Transm()
        ids = [a["transmission_id"] for a in added]
        while True:
            cur = {x["id"]: x for x in t.torrents(["id", "leftUntilDone", "percentDone",
                                                   "rateDownload", "peersConnected", "status"])}
            left = [cur.get(i, {}).get("leftUntilDone", 0) for i in ids]
            if all(not l for l in left):
                break
            done = sum(cur.get(i, {}).get("percentDone", 0) for i in ids) / max(len(ids), 1)
            rate = sum(cur.get(i, {}).get("rateDownload", 0) for i in ids)
            print(f"  {time.strftime('%H:%M:%S')} готовность {done * 100:5.1f}%  "
                  f"{rate / 1e6:5.2f} MB/s  осталось {sum(left) / 1e6:.0f} MB")
            time.sleep(30)
        print("  готово")
    print(json.dumps(added, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
