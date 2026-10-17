#!/usr/bin/env python3
"""Статус рабочего места: лимит флешки, cookie, transmission RPC."""
import sys
import json

import requests

sys.path.insert(0, "/home/mydum/Desktop/projects/restore-next")

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
    print(f"флешка {workspace.data_root()}")
    print(f"  [{('#' * bar).ljust(20)}] {human(used)} / {human(budget)}  (свободно {human(budget - used)})")

    cookies = secrets.rutracker_cookies()
    if cookies:
        print(f"cookie: ок ({len(cookies)} шт: {', '.join(list(cookies)[:6])})")
    else:
        print("cookie: НЕТО")

    # RPC требует рукопожатие X-Transmission-Session-Id — используем клиент
    from rnext import transm as transm_mod
    rpc = transm_mod.check()
    if rpc.get("ok"):
        print(f"transmission rpc: ok {rpc['version']}")
    else:
        print(f"transmission rpc: оффа ({rpc.get('err')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
