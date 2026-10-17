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

    cookie = secrets.rutracker_bb_data()
    print(f"cookie bb_data: {'ок' if cookie else 'НЕТО'}")

    try:
        rpc = secrets.transmission_rpc()
        r = requests.post(rpc["url"], json={"method": "session-get"}, auth=(rpc["username"], rpc["password"]), timeout=4)
        print(f"transmission rpc: {r.status_code}")
    except Exception as e:
        print(f"transmission rpc: оффа ({e.__class__.__name__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
