#!/usr/bin/env python3
"""Tune Transmission for the dataset: both the session and the settings file.

Why a separate script: settings applied via RPC (`session-set`) only live until
the client restarts — after a reboot transmission-gtk returns to its defaults
(queue 2, DHT off) and download speed drops ~20x. This script also edits
`settings.json`, so the values persist for good.

  python3 scripts/transmission_tune.py            # show current values
  python3 scripts/transmission_tune.py --apply    # apply (RPC + file)
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

# Target values: bigger queue, DHT/LPD on, limits off.
# Rationale in the README: with defaults (queue 2, DHT off) the dataset grew
# ~1.9 GB/h, with these — ~40 GB/h.
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
    print("session (RPC):", {k: s.get(k) for k in WATCH})
    if SETTINGS.is_file():
        d = json.loads(SETTINGS.read_text(encoding="utf-8"))
        print("settings.json file:", {k: d.get(k) for k in WATCH})
    return 0


def apply(extra: dict) -> int:
    tune = {**TUNE, **extra}
    # 1) session
    t = Transm()
    t._call("session-set", tune)
    s = t.session()
    ok_session = all(s.get(k) == v for k, v in tune.items())
    print("session updated:", "ok" if ok_session else "partial",
          {k: s.get(k) for k in WATCH})

    # 2) settings file (so it survives a client restart)
    if not SETTINGS.is_file():
        print(f"settings.json not found: {SETTINGS}")
        return 1 if not ok_session else 0
    shutil.copy2(SETTINGS, SETTINGS.with_suffix(".json.bak"))
    d = json.loads(SETTINGS.read_text(encoding="utf-8"))
    before = {k: d.get(k) for k in WATCH}
    d.update(tune)
    SETTINGS.write_text(json.dumps(d, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    after = {k: d.get(k) for k in WATCH}
    print("file updated (backup settings.json.bak)")
    print("  before:", before)
    print("  after: ", after)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--download-dir", default=None, help="also set the download directory")
    args = ap.parse_args()
    extra = {"download-dir": args.download_dir} if args.download_dir else {}
    return apply(extra) if args.apply else show()


if __name__ == "__main__":
    sys.exit(main())
