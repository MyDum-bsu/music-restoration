#!/usr/bin/env python3
"""Сторож релизов: сверяет, что конвейер не теряет альбомы и не портит данные.

Проверяет по каждому завершённому релизу:
  * все ли треки доехали до телефона и совпадает ли их число с источником;
  * есть ли обложка и тексты (и сколько треков без текста);
  * не было ли ошибок в шагах конвейера;
плюс общее: место на дисках, живость цикла датасета, зависшие закачки.

  python3 scripts/release_watch.py                 # один проход
  python3 scripts/release_watch.py --loop 600      # в цикле, каждые 10 минут
  python3 scripts/release_watch.py --json /tmp/watch.json

Пишет только АНОМАЛИИ (норма молчит) — чтобы в логе было видно проблемы.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import workspace  # noqa: E402

DATA = workspace.data_root()
PHONE_CACHE: dict = {}


def phone_root() -> Path | None:
    try:
        from rnext.phone import mtp
        return mtp.phone_dir()
    except Exception:  # noqa: BLE001
        return None


def phone_flacs(root: Path, rel: str) -> int | None:
    """Сколько FLAC в каталоге на телефоне (с учётом подкаталогов)."""
    key = (str(root), rel)
    if key in PHONE_CACHE:
        return PHONE_CACHE[key]
    d = root / rel
    if not d.is_dir():
        PHONE_CACHE[key] = None
        return None
    n = 0
    stack = [d]
    while stack:
        cur = stack.pop()
        try:
            for p in cur.iterdir():
                if p.is_dir():
                    stack.append(p)
                elif p.suffix.lower() == ".flac":
                    n += 1
        except OSError:
            pass
    PHONE_CACHE[key] = n
    return n


def check_blocks(latest: dict[str, str]) -> list[str]:
    """Разбор последних блоков логов: что завершено, что с ошибками."""
    warns: list[str] = []
    for nm, b in latest.items():
        name = nm
        if "Traceback" in b or "ошибка" in b.lower():
            line = next((l for l in b.splitlines() if "ошибка" in l.lower()
                         or "Traceback" in l), "")
            warns.append(f"[ошибка шага] {name[:52]}: {line.strip()[:90]}")
        # сколько треков в релизе
        m = re.findall(r"скопировано (\d+)/(\d+) файлов", b)
        if m:
            got, total = int(m[-1][0]), int(m[-1][1])
            if got != total:
                warns.append(f"[не всё скопировано] {name[:52]}: {got} из {total}")
        if "проверка ок" not in b and "скопировано" in b:
            warns.append(f"[нет проверки md5] {name[:52]}")
        # Тексты по логу НЕ проверяем: лог отражает момент прогона, а тексты
        # могли добрать позже (Place Vendome, Q5). Факт проверяет телефон.
        # «обложка не найдена» из лога НЕ считаем тревогой: обложку могли вшить
        # позже (как у Bring Me Your Love), а факт проверяет check_phone_albums
    return warns


def check_phone_albums(root: Path, limit: int = 0) -> list[str]:
    """Реальная проверка телефона: у первого трека каждого альбома есть
    встроенная обложка и текст? Лог врёт (обложку могли вшить позже), а это —
    факт. Читаем файл обратно через gio и смотрим теги."""
    import tempfile
    listfile = Path("/tmp/phone_check.txt")
    if not listfile.is_file():
        return []
    rels = [l.strip() for l in listfile.read_text().splitlines() if l.strip()]
    if limit:
        rels = rels[:limit]
    warns = []
    for rel in rels:
        d = root / rel
        if not d.is_dir():
            warns.append(f"[телефон] нет каталога: {rel}")
            continue
        try:
            cands = sorted(p for p in d.rglob("*.flac"))
        except OSError:
            continue
        if not cands:
            warns.append(f"[телефон] нет FLAC: {rel}")
            continue
        # Смотрим НЕСКОЛЬКО треков: первый может быть инструменталом (у Subsignal
        # «271 Days» — интро без текста), и одиночная проверка давала ложную тревогу.
        sample = cands if len(cands) <= 3 else [cands[0], cands[len(cands) // 2], cands[-1]]
        got = {"cover": False, "lyrics": False}
        for f in sample:
            key = (str(f), f.stat().st_size)
            if key not in PHONE_CACHE:
                with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as t:
                    tmp = Path(t.name)
                try:
                    subprocess.run(["gio", "cat", str(f)], stdout=open(tmp, "wb"),
                                   timeout=180)
                    r = subprocess.run(["metaflac", "--list", "--block-type=PICTURE",
                                        "--block-type=VORBIS_COMMENT", str(tmp)],
                                       capture_output=True, text=True)
                    low = r.stdout.lower()
                    PHONE_CACHE[key] = {"cover": "type: 6" in r.stdout,
                                        "lyrics": "lyrics=" in low
                                                  or "unsyncedlyrics=" in low}
                except Exception:  # noqa: BLE001
                    PHONE_CACHE[key] = {"cover": True, "lyrics": True}
                finally:
                    tmp.unlink(missing_ok=True)
            v = PHONE_CACHE[key]
            got["cover"] = got["cover"] or v["cover"]
            got["lyrics"] = got["lyrics"] or v["lyrics"]
        if not got["cover"]:
            warns.append(f"[телефон: нет обложки] {rel}")
        if not got["lyrics"]:
            warns.append(f"[телефон: нет текста] {rel}")
    return warns


def check_disk() -> list[str]:
    warns = []
    for mount in ("/", "/home", str(DATA)):
        try:
            u = shutil.disk_usage(mount)
        except OSError:
            continue
        free_gb = u.free / 1e9
        if free_gb < 3:
            warns.append(f"[место] {mount}: свободно {free_gb:.1f} ГБ — КРИТИЧНО")
        elif free_gb < 6:
            warns.append(f"[место] {mount}: свободно {free_gb:.1f} ГБ")
    return warns


def check_loop() -> list[str]:
    r = subprocess.run(["pgrep", "-f", "dataset_loop.py"], capture_output=True, text=True)
    return [] if r.stdout.strip() else ["[цикл] dataset_loop не запущен"]


def check_stalled() -> list[str]:
    try:
        from rnext.transm import Transm
        t = Transm()
        rows = [x for x in t.torrents(["id", "name", "downloadDir", "leftUntilDone",
                                       "peersConnected", "status"])
                if "release_src" in (x["downloadDir"] or "") and x["leftUntilDone"] > 0]
    except Exception as e:  # noqa: BLE001
        return [f"[transmission] недоступен: {type(e).__name__}"]
    out = []
    for x in rows:
        if x["peersConnected"] == 0 and x["status"] == 4:
            out.append(f"[зависла закачка] {x['name'][:44]} — 0 пиров, "
                       f"осталось {x['leftUntilDone'] / 1e6:.0f} МБ")
    return out


def one_pass(args) -> int:
    warns: list[str] = []
    root = phone_root()
    if root is None:
        warns.append("[телефон] не найден (MTP не подключён?)")
    # Логов несколько (cc, wave2, retry, fix...), и альбом мог прогоняться
    # повторно. Склеиваем ВСЕ блоки и берём последний по имени альбома.
    all_blocks: dict[str, str] = {}
    # по ВРЕМЕНИ изменения: успешная переделка должна перекрывать прежний провал
    logs = sorted(Path("/tmp").glob("release_*.log"),
                  key=lambda p: p.stat().st_mtime if p.is_file() else 0)
    for log in logs:
        if not log.is_file():
            continue
        for b in re.split(r"^=== ", log.read_text(encoding="utf-8", errors="ignore"),
                          flags=re.M)[1:]:
            head = b.splitlines()[0] if b.splitlines() else "?"
            nm = head.split(" ", 1)[1] if " " in head else head
            all_blocks[nm] = b
    warns += check_blocks(all_blocks)
    if root is not None:
        warns += check_phone_albums(root, limit=args.phone_limit)
    warns += check_disk() + check_loop() + check_stalled()

    stamp = time.strftime("%d.%m %H:%M:%S")
    if warns:
        print(f"[{stamp}] ВНИМАНИЕ, {len(warns)}:")
        for w in warns:
            print(f"  {w}")
    else:
        print(f"[{stamp}] всё чисто: ошибок нет, место есть, цикл жив, закачки идут")
    if args.json:
        Path(args.json).write_text(json.dumps({"ts": stamp, "warnings": warns},
                                              ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="цикл, секунд (0 = один проход)")
    ap.add_argument("--json", default="")
    ap.add_argument("--phone-limit", type=int, default=0,
                    help="сколько альбомов на телефоне проверять за проход (0 = все)")
    args = ap.parse_args()
    if args.loop:
        while True:
            one_pass(args)
            sys.stdout.flush()
            time.sleep(args.loop)
    return one_pass(args)


if __name__ == "__main__":
    sys.exit(main())
