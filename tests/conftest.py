"""Repo-root on sys.path: tests import `scripts.*` and `rnext.*` from source.

No installation step is required — the same imports the CLI scripts use.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
