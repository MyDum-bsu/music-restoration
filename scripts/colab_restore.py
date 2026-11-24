#!/usr/bin/env python3
"""Thin CLI for running heavy audio restoration tasks on Colab Pro+ GPUs.

Usage:
  python3 scripts/colab_restore.py --input track.wav --out-dir ./out --task gpu_smoke --gpu T4 --dry-run
  python3 scripts/colab_restore.py --input track.wav --out-dir ./out --task neural_declip --gpu L4 --drive-subpath smoke/run1
  python3 scripts/colab_restore.py --check-sessions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext.cloud.colab_runner import (
    DEFAULT_AUTH,
    DEFAULT_GPU,
    VALID_AUTHS,
    VALID_GPUS,
    VALID_TASKS,
    ColabError,
    ColabJobConfig,
    check_active_sessions,
    run_colab_job,
)
from rnext.cloud.drive import DriveError
from rnext.cloud.metrics import MetricsError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="colab_restore.py",
        description="Run GPU audio restoration on Google Colab with guaranteed auto-stop and SHA-256 manifests.",
    )
    p.add_argument("--input", type=Path, help="Local input audio file (WAV/FLAC/etc.).")
    p.add_argument("--out-dir", type=Path, help="Local output directory for restored audio and manifest.json.")
    p.add_argument("--task", choices=VALID_TASKS, default="gpu_smoke", help="Restoration task (default: gpu_smoke).")
    p.add_argument("--gpu", choices=VALID_GPUS, default=DEFAULT_GPU, help=f"Colab GPU tier (default: {DEFAULT_GPU}).")
    p.add_argument("--auth", choices=VALID_AUTHS, default=DEFAULT_AUTH, help=f"Colab CLI auth strategy (default: {DEFAULT_AUTH}).")
    p.add_argument("--session", default=None, help="Optional Colab session name.")
    p.add_argument("--timeout", type=float, default=600.0, help="Inactivity timeout in seconds (default: 600).")
    p.add_argument("--high-mem", action="store_true", help="Request high-RAM Colab machine.")
    p.add_argument("--keep-session", action="store_true", help="Keep Colab VM running after completion (default: auto-stop).")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned commands without starting a VM.")
    p.add_argument("--drive-subpath", default=None, help="Optional Google Drive subpath under remote base for post-run upload + check.")
    p.add_argument("--drive-remote", default=None, help="Override rclone remote base (default: MR_RCLONE_REMOTE or gdrive:darkfactory-storage/music-restoration).")
    p.add_argument("--colab-bin", default=None, help="Optional path to colab CLI executable.")
    p.add_argument("--rclone-bin", default=None, help="Optional path to rclone executable.")
    p.add_argument("--check-sessions", action="store_true", help="Check active Colab sessions and exit (0 if 0 active, 1 otherwise).")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.check_sessions:
            no_active, text = check_active_sessions(auth=args.auth, colab_bin=args.colab_bin)
            if args.json:
                print(json.dumps({"no_active_sessions": no_active, "output": text}, indent=2))
            else:
                print(text or ("No active sessions." if no_active else "Active sessions detected."))
            return 0 if no_active else 1

        if not args.input or not args.out_dir:
            parser.error("--input and --out-dir are required unless --check-sessions is set")

        cfg = ColabJobConfig(
            input_path=args.input,
            out_dir=args.out_dir,
            task=args.task,
            gpu=args.gpu,
            auth=args.auth,
            session=args.session,
            timeout_s=args.timeout,
            high_mem=args.high_mem,
            keep_session=args.keep_session,
            dry_run=args.dry_run,
            drive_subpath=args.drive_subpath,
            drive_remote_base=args.drive_remote,
            colab_bin=args.colab_bin,
            rclone_bin=args.rclone_bin,
        )
        res = run_colab_job(cfg)
    except (ColabError, DriveError, MetricsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json or args.dry_run:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"task={res.plan.task} gpu={res.plan.gpu} session={res.plan.session} sessions_clean={res.sessions_clean_after}")
        if res.output_audio_path:
            print(f"output: {res.output_audio_path}")
        if res.manifest_path:
            print(f"manifest: {res.manifest_path}")
        if res.drive_transfers:
            print(f"drive: uploaded {len(res.drive_transfers)} verified artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
