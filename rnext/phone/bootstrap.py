"""Поиск скриптов соседних скиллов при symlink/copy-установке.

Скиллы ссылаются друг на друга (phone-music использует add-lyrics).
install.sh по умолчанию симлинкует каталоги скиллов из репозитория, поэтому
repo-relative поиск работает. Скопированный скилл тоже работает, если задан
MUSIC_RESTORATION_HOME или скиллы лежат в ~/.agents/skills.
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # rnext/phone/bootstrap.py -> корень репо


def skill_scripts_dir(name: str) -> Path:
    home = os.environ.get("MUSIC_RESTORATION_HOME")
    candidates = [
        Path(home) / "skills" / name / "scripts" if home else None,
        Path(home) / name / "scripts" if home else None,  # раскладка v1
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
