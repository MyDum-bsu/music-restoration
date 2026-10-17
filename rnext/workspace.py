"""Рабочая зона на флешке: единые пути и строгий лимит 40 GiB (NTFS, только файлы)."""
import subprocess
from pathlib import Path

import json


class WorkspaceError(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg = repo_root() / "config" / "workspace.json"
    conf = json.loads(cfg.read_text(encoding="utf-8"))
    root = Path(conf["data_root"])
    if not root.is_dir():
        raise WorkspaceError(f"data_root is missing: {root}")
    return conf


def data_root() -> Path:
    return Path(load_config()["data_root"])


def dir_of(kind: str) -> Path:
    kind = (kind or "").strip().lower()
    dirs = load_config()["dirs"]
    if kind not in dirs:
        raise WorkspaceError(f"unknown workspace dir: {kind}")
    p = data_root() / dirs[kind]
    p.mkdir(parents=True, exist_ok=True)
    return p


def usage_bytes() -> int:
    """Реальный занятый байтовый объём data_root (du -sb)."""
    out = subprocess.run(
        ["du", "-sb", "--apparent-size", str(data_root())],
        capture_output=True, text=True, check=True,
    ).stdout.split()[0]
    return int(out)


def budget_bytes() -> int:
    return int(load_config()["budget_bytes"])


def remaining_bytes() -> int:
    return budget_bytes() - usage_bytes()


def assert_budget(need_bytes: int = 0) -> None:
    rem = remaining_bytes()
    if rem <= 0 or rem < need_bytes:
        raise WorkspaceError(
            f"budget exceeded: need {need_bytes}, remaining {rem}"
        )
