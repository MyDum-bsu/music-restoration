"""Repo-root on sys.path: tests import `scripts.*` and `rnext.*` from source.

No installation step is required — the same imports the CLI scripts use.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Test-environment repair: sandboxed runners may exec the interpreter under a
# foreign argv[0] (here: a desktop AppImage wrapper). CPython derives
# sys.executable from argv[0], so tests that spawn subprocesses via
# sys.executable would launch a non-python binary that ignores the script.
# sys.base_prefix survives such execs, so once per session point
# sys.executable back at the real interpreter when it clearly isn't one.
if not Path(sys.executable).name.startswith("python"):
    for name in ("python3", "python"):
        real = Path(sys.base_prefix) / "bin" / name
        if real.is_file():
            sys.executable = str(real)
            break
