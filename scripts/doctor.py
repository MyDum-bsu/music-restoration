#!/usr/bin/env python3
"""Thin CLI over rnext.doctor: argument parsing, printing, exit codes.

Usage:
    python3 scripts/doctor.py [--json] [--config PATH]

Exit codes: 0 all green; 1 a HARD check failed; 2 only WARN failures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext.doctor import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
