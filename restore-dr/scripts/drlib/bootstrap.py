"""Locate sibling skill scripts across symlinked/copied installs.

Skills in this repo reference each other (phone-music uses add-lyrics;
everything can use restore-dr/scripts/drlib). install.sh symlinks whole skill
dirs from the repo, so repo-relative lookups resolve. A copied-out skill still
works if MUSIC_RESTORATION_HOME or ~/.agents/skills has the siblings.
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]  # repo root (restore-dr/scripts/drlib -> up 3)


def skill_scripts_dir(name: str) -> Path:
    candidates = [
        Path(os.environ.get("MUSIC_RESTORATION_HOME", "")) / name / "scripts"
        if os.environ.get("MUSIC_RESTORATION_HOME") else None,
        _REPO / name / "scripts",
        Path.home() / ".agents" / "skills" / name / "scripts",
    ]
    for c in candidates:
        if c and c.is_dir():
            return c
    raise ImportError(
        f"cannot locate the '{name}' skill scripts. Install the whole "
        f"music-restoration repo (install.sh) or set MUSIC_RESTORATION_HOME to its root.")


def add_skill_scripts(name: str) -> Path:
    p = skill_scripts_dir(name)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def add_drlib() -> Path:
    """Put restore-dr/scripts on sys.path so `from drlib import ...` works."""
    return add_skill_scripts("restore-dr")
