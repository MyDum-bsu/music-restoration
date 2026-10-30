#!/usr/bin/env python3
"""Workplace status: flash drive budget, cookie, transmission RPC."""
import sys
import json

import requests

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import workspace  # noqa: E402
from rnext import secrets  # noqa: E402


def human(n: int) -> str:
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return f"{n:,.1f} {u}".replace(",", " ")
        n /= 1024
    return f"{n} GiB"


def main() -> int:
    try:
        used = workspace.usage_bytes()
    except workspace.WorkspaceError as e:
        print(f"[FAIL] workspace: {e}")
        return 1
    budget = workspace.budget_bytes()
    bar = int(20 * used / budget)
    print(f"flash drive {workspace.data_root()}")
    print(f"  [{('#' * bar).ljust(20)}] {human(used)} / {human(budget)}  (free {human(budget - used)})")

    cookies = secrets.rutracker_cookies()
    if cookies:
        print(f"cookie: ok ({len(cookies)} pcs: {', '.join(list(cookies)[:6])})")
    else:
        print("cookie: NONE")

    # RPC requires the X-Transmission-Session-Id handshake — use the client
    from rnext import transm as transm_mod
    rpc = transm_mod.check()
    if not rpc.get("ok"):
        print(f"transmission rpc: down ({rpc.get('err')})")
        return 0
    print(f"transmission rpc: ok {rpc['version']}")
    # Verify the download tuning is still in place (otherwise speed drops severalfold)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.transmission_tune import WATCH, TUNE  # type: ignore
    except Exception:                                      # noqa: BLE001
        WATCH, TUNE = ["download-queue-size", "dht-enabled"], {}
    s = transm_mod.Transm().session()
    drift = {k: (s.get(k), TUNE.get(k)) for k in WATCH if k in TUNE and s.get(k) != TUNE[k]}
    if drift:
        print("transmission tuning: DRIFTED " + ", ".join(f"{k}: {v[0]}≠{v[1]}" for k, v in drift.items()))
        print("   → run: python3 scripts/transmission_tune.py --apply")
    else:
        print(f"transmission tuning: ok (queue {s.get('download-queue-size')}, "
              f"DHT {'on' if s.get('dht-enabled') else 'off'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
