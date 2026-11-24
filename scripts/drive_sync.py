#!/usr/bin/env python3
"""Thin CLI for safe non-destructive Google Drive sync via rclone.

Defaults to ``--dry-run`` on write operations (pass ``--execute`` to transfer).
Local files are never deleted or overwritten by default.

Usage:
  python3 scripts/drive_sync.py upload ./out smoke/run1               # dry-run
  python3 scripts/drive_sync.py upload ./out smoke/run1 --execute     # copy + check --one-way
  python3 scripts/drive_sync.py verify ./out smoke/run1               # rclone check --one-way
  python3 scripts/drive_sync.py download smoke/run1 ./local_dl        # dry-run
  python3 scripts/drive_sync.py ls smoke                              # list remote entries
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext.cloud.drive import (
    DEFAULT_REMOTE_BASE,
    DriveError,
    download_path,
    list_remote,
    upload_path,
    verify_one_way,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="drive_sync.py",
        description=f"Safe non-destructive Google Drive sync (default remote: {DEFAULT_REMOTE_BASE}).",
    )
    p.add_argument("--remote", default=None, help="Override rclone remote base.")
    p.add_argument("--rclone-bin", default=None, help="Optional path to rclone binary.")
    p.add_argument("--json", action="store_true", help="Emit JSON output.")

    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Copy local file/dir to Drive (dry-run by default).")
    up.add_argument("local_path", type=Path)
    up.add_argument("remote_subpath")
    up.add_argument("--execute", action="store_true", help="Actually execute rclone copy (default is dry-run).")
    up.add_argument("--dry-run", action="store_true", help="Explicitly force dry-run (overrides --execute).")
    up.add_argument("--no-verify", action="store_true", help="Skip post-upload rclone check --one-way.")

    ver = sub.add_parser("verify", help="Verify local file/dir against Drive via rclone check --one-way.")
    ver.add_argument("local_path", type=Path)
    ver.add_argument("remote_subpath")

    dl = sub.add_parser("download", help="Copy remote subpath to local dir (dry-run by default).")
    dl.add_argument("remote_subpath")
    dl.add_argument("local_dest", type=Path)
    dl.add_argument("--execute", action="store_true", help="Actually execute rclone copy (default is dry-run).")
    dl.add_argument("--dry-run", action="store_true", help="Explicitly force dry-run (overrides --execute).")
    dl.add_argument("--overwrite", action="store_true", help="Allow overwriting existing local files (default: --ignore-existing).")

    ls = sub.add_parser("ls", help="List entries under remote subpath.")
    ls.add_argument("remote_subpath", nargs="?", default="")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "upload":
            dry_run = args.dry_run or (not args.execute)
            res = upload_path(
                args.local_path,
                args.remote_subpath,
                remote_base=args.remote,
                dry_run=dry_run,
                verify=not args.no_verify,
                rclone_bin=args.rclone_bin,
            )
            payload = res.to_dict()
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if args.command == "verify":
            res = verify_one_way(
                args.local_path,
                args.remote_subpath,
                remote_base=args.remote,
                rclone_bin=args.rclone_bin,
            )
            payload = {
                "ok": res.ok,
                "remote_uri": res.remote_uri,
                "returncode": res.returncode,
                "stderr": res.stderr.strip(),
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0 if res.ok else 1

        if args.command == "download":
            dry_run = args.dry_run or (not args.execute)
            res = download_path(
                args.remote_subpath,
                args.local_dest,
                remote_base=args.remote,
                dry_run=dry_run,
                overwrite=args.overwrite,
                rclone_bin=args.rclone_bin,
            )
            print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
            return 0

        if args.command == "ls":
            entries = list_remote(
                args.remote_subpath,
                remote_base=args.remote,
                rclone_bin=args.rclone_bin,
            )
            if args.json:
                print(json.dumps(entries, indent=2, ensure_ascii=False))
            else:
                for item in entries:
                    kind = "DIR " if item.get("IsDir") else "FILE"
                    print(f"{kind} {item.get('Size', 0):>10}  {item.get('Path', item.get('Name', ''))}")
            return 0
    except DriveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
