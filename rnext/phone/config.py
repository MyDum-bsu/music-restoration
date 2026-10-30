"""Конфигурация: переменная окружения > config-файл > автодетект/дефолт.

Файл: ~/.config/music-restoration/config.env (или $MUSIC_RESTORATION_CONFIG),
строки KEY=VALUE, комментарии с '#'. Образец — config.env.example в корне репо.

Правила, выученные на практике:
- Рабочие каталоги по умолчанию под домашним каталогом, не в /tmp: корневая
  файловая система часто маленькая, и полный /tmp посреди пакетного прогона
  убивает процесс.
- Телефон определяется автоматически по gvfs MTP-монтированию; серийник
  устройства нигде не зашит.
"""
import os
import shlex
import tempfile
from pathlib import Path

APP = "music-restoration"
ENV_PREFIX = "MR_"

_file: dict[str, str] = {}


def _config_file() -> Path:
    p = os.environ.get("MUSIC_RESTORATION_CONFIG")
    return Path(p).expanduser() if p else Path.home() / ".config" / APP / "config.env"


def _load_file() -> None:
    if _file:
        return
    f = _config_file()
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        try:
            _file[k.strip()] = " ".join(shlex.split(v.strip())) or v.strip()
        except ValueError:
            _file[k.strip()] = v.strip()


def get(key: str, default: str | None = None) -> str | None:
    _load_file()
    return os.environ.get(ENV_PREFIX + key) or os.environ.get(key) or _file.get(key) or default


def get_path(key: str, default: Path | None = None) -> Path | None:
    v = get(key)
    return Path(v).expanduser() if v else default


def music_dir() -> Path:
    return get_path("MUSIC_DIR", Path.home() / "Music")


def state_dir() -> Path:
    return get_path("STATE_DIR", Path.home() / ".local" / "state" / APP)


def work_dir() -> Path:
    return get_path("WORK_DIR", state_dir() / "work")


def ensure_dirs() -> None:
    """Создать state/work. Вызывать один раз в начале CLI."""
    state_dir().mkdir(parents=True, exist_ok=True)
    work_dir().mkdir(parents=True, exist_ok=True)


def work_tmpdir(prefix: str = "mr_") -> Path:
    """mkdtemp под WORK_DIR, не в /tmp (корневая ФС бывает крошечной)."""
    ensure_dirs()
    return Path(tempfile.mkdtemp(prefix=prefix, dir=work_dir()))
