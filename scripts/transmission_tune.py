#!/usr/bin/env python3
"""Настройка Transmission для датасета: и в сессии, и в файле настроек.

Зачем отдельный скрипт: настройки, выставленные через RPC (`session-set`),
живут только до перезапуска клиента — после ребута transmission-gtk возвращает
свои дефолты (очередь 2, DHT выкл), и скорость качки падает в ~20 раз. Этот
скрипт правит ещё и `settings.json`, поэтому значения сохраняются навсегда.

  python3 scripts/transmission_tune.py            # показать текущие
  python3 scripts/transmission_tune.py --apply    # применить (RPC + файл)
  python3 scripts/transmission_tune.py --apply --download-dir /home/mydum/Music
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext.transm import Transm  # noqa: E402

SETTINGS = Path.home() / ".config" / "transmission" / "settings.json"

# Целевые значения: очередь побольше, DHT/LPD включены, лимиты выключены.
# Обоснование — в README: с дефолтами (очередь 2, DHT выкл) датасет рос
# ~1.9 ГБ/ч, с этими — ~40 ГБ/ч.
TUNE = {
    "download-queue-size": 24,
    "download-queue-enabled": True,
    "queue-stalled-enabled": True,
    "queue-stalled-minutes": 20,
    "peer-limit-global": 500,
    "dht-enabled": True,
    "pex-enabled": True,
    "lpd-enabled": True,
    "speed-limit-down-enabled": False,
    "speed-limit-up-enabled": False,
    "alt-speed-enabled": False,
    "ratio-limit-enabled": False,
    "seed-queue-enabled": False,
}

WATCH = ["download-queue-size", "peer-limit-global", "dht-enabled", "lpd-enabled",
         "queue-stalled-minutes", "speed-limit-down-enabled", "alt-speed-enabled"]


def show() -> int:
    t = Transm()
    s = t.session()
    print("сессия (RPC):", {k: s.get(k) for k in WATCH})
    if SETTINGS.is_file():
        d = json.loads(SETTINGS.read_text(encoding="utf-8"))
        print("файл settings.json:", {k: d.get(k) for k in WATCH})
    return 0


def apply(extra: dict) -> int:
    tune = {**TUNE, **extra}
    # 1) сессия
    t = Transm()
    t._call("session-set", tune)
    s = t.session()
    ok_session = all(s.get(k) == v for k, v in tune.items())
    print("сессия обновлена:", "ок" if ok_session else "частично",
          {k: s.get(k) for k in WATCH})

    # 2) файл настроек (чтобы пережить перезапуск клиента)
    if not SETTINGS.is_file():
        print(f"settings.json не найден: {SETTINGS}")
        return 1 if not ok_session else 0
    shutil.copy2(SETTINGS, SETTINGS.with_suffix(".json.bak"))
    d = json.loads(SETTINGS.read_text(encoding="utf-8"))
    before = {k: d.get(k) for k in WATCH}
    d.update(tune)
    SETTINGS.write_text(json.dumps(d, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    after = {k: d.get(k) for k in WATCH}
    print("файл обновлён (бэкап settings.json.bak)")
    print("  было: ", before)
    print("  стало:", after)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--download-dir", default=None, help="заодно выставить каталог загрузок")
    args = ap.parse_args()
    extra = {"download-dir": args.download_dir} if args.download_dir else {}
    return apply(extra) if args.apply else show()


if __name__ == "__main__":
    sys.exit(main())
