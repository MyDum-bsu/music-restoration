"""Central configuration: environment variable > config file > auto-detect/default.

Config file: ~/.config/music-restoration/config.env (or $MUSIC_RESTORATION_CONFIG),
KEY=VALUE lines, '#' comments. See config.env.example in the repo root.

Design rules baked in here (learned the hard way, see docs/pipeline.md):
- Work/staging dirs live under the user's home by default, never /tmp — the
  root filesystem is often tiny and a full /tmp mid-batch kills the run.
- The phone is auto-detected from gvfs MTP mounts; no device serial anywhere.
"""
import os
import shlex
import tempfile
from pathlib import Path

APP = "music-restoration"
ENV_PREFIX = "MR_"

_SKILL_ROOT = Path(__file__).resolve().parents[2]  # .../restore-dr

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
    """Create state/work dirs. Call once at CLI start; pipeline code may rely on them."""
    state_dir().mkdir(parents=True, exist_ok=True)
    work_dir().mkdir(parents=True, exist_ok=True)


def work_tmpdir(prefix: str = "mr_") -> Path:
    """mkdtemp under WORK_DIR, not /tmp (root fs can be tiny; a full /tmp kills the run)."""
    ensure_dirs()
    return Path(tempfile.mkdtemp(prefix=prefix, dir=work_dir()))


def stereo_tool_bin() -> Path:
    v = get_path("STEREO_TOOL_BIN")
    if v:
        return v
    for name in ("stereo_tool_cmd_64.exe", "stereo_tool_cmd_64"):  # Windows build first if present
        p = _SKILL_ROOT / "bin" / name
        if p.exists():
            return p
    return _SKILL_ROOT / "bin" / "stereo_tool_cmd_64"


def sts_dir() -> Path:
    return _SKILL_ROOT / "sts"
