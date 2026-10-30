"""Locating scripts of neighboring skills under symlink/copy installs.

Skills reference each other (phone-music uses add-lyrics).
install.sh symlinks the skill directories from the repo by default, so the
repo-relative search works. A copied skill also works if MUSIC_RESTORATION_HOME
is set or the skills live in ~/.agents/skills.
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # rnext/phone/bootstrap.py -> repo root


def skill_scripts_dir(name: str) -> Path:
    home = os.environ.get("MUSIC_RESTORATION_HOME")
    candidates = [
        Path(home) / "skills" / name / "scripts" if home else None,
        Path(home) / name / "scripts" if home else None,  # v1 layout
        _REPO / "skills" / name / "scripts",
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
