"""Доступ к телефону по MTP через gvfs + gio. Без констант конкретного устройства.

Выученные правила (подробности — docs/mtp-notes.md):
- fuse-монтирование gvfs на практике read-only: запись через него падает с
  'Operation not supported'. Любая запись идёт через
  `gio copy -p <local> <mtp://...>` с URL-encoded сегментами пути.
- gio умеет только перезаписывать существующие файлы, но не создавать новые —
  каталоги заранее создаются `gio mkdir -p`.
- Большие пачки читать через copy_from_phone(use_gio=True) — примерно вдвое
  быстрее, чем fuse cp.
- Запись проверять чтением обратно (gio cat) и сверкой md5: fuse-метаданные
  сразу после перезаписи врут.
"""
import os
import time
import urllib.parse
from pathlib import Path

from .config import get
from .util import md5f, run

AUDIO_EXTS = {".flac", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".m4b", ".mp4", ".wav"}
MUSIC_DIR_NAMES = {"music", "музыка"}


def gvfs_mtp_mounts() -> list[Path]:
    """MTP-монтирования gvfs (Linux-десктопы). Пусто в остальных местах —
    телефонные функции деградируют в понятную ошибку, а не в падение."""
    if not hasattr(os, "getuid"):
        return []
    base = Path("/run/user") / str(os.getuid()) / "gvfs"
    return sorted(p for p in base.glob("mtp:host=*") if p.exists())


def _music_dirs_below(mount: Path, depth: int = 3) -> list[Path]:
    """Поиск Music-каталога в MTP-монтировании с ограничением глубины (fuse медленный)."""
    found = []

    def rec(d: Path, lev: int):
        try:
            for child in d.iterdir():
                if child.is_dir() and child.name.lower() in MUSIC_DIR_NAMES:
                    found.append(child)
                elif lev < depth and child.is_dir() and not child.name.startswith("."):
                    rec(child, lev + 1)
        except (PermissionError, OSError):
            pass

    rec(mount, 1)
    return found


def phone_dir() -> Path | None:
    """Корень музыки на телефоне: $MR_PHONE_DIR / config / автодетект gvfs."""
    v = get("PHONE_DIR")
    if v:
        return Path(v).expanduser()
    candidates: list[Path] = []
    for m in gvfs_mtp_mounts():
        candidates += _music_dirs_below(m)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print("multiple Music dirs found on MTP mounts, using the first; set MR_PHONE_DIR to override:")
        for c in candidates:
            print(f"  {c}")
    return candidates[0] if candidates else None


def mount_of(music_root: Path) -> Path:
    for m in gvfs_mtp_mounts():
        try:
            music_root.relative_to(m)
            return m
        except ValueError:
            continue
    raise ValueError(f"{music_root} is not under a gvfs mtp mount; set MR_MTP_BASE explicitly")


def mtp_base(music_root: Path | None = None) -> str:
    """Базовый gio URI для корня музыки, например mtp://DEV/Storage/Music."""
    root = music_root or phone_dir()
    v = get("MTP_BASE")
    if v:
        return v
    m = mount_of(root)
    rel = root.relative_to(m)
    tail = "/".join(urllib.parse.quote(seg) for seg in rel.parts)
    return "mtp://" + m.name[len("mtp:host="):] + ("/" + tail if tail else "")


def is_mtp(path) -> bool:
    return "mtp:host=" in str(path)


def uri_for(music_root: Path, rel: Path) -> str:
    """mtp:// URI для пути относительно корня музыки. Каждый сегмент кодируется
    отдельно — кодирование всего пути превратило бы '/' в %2F и сломало gio."""
    return mtp_base(music_root) + "/" + "/".join(urllib.parse.quote(seg) for seg in Path(rel).parts)


def phone_alive(music_root: Path | None = None) -> bool:
    root = music_root or phone_dir()
    return root is not None and root.exists()


def copy_to_phone(music_root: Path, src: Path, rel: Path, retries: int = 3, wait: int = 5):
    """Скопировать локальный файл поверх СУЩЕСТВУЮЩЕГО файла на телефоне."""
    if is_mtp(music_root):
        r = None
        for _ in range(retries):
            r = run(["gio", "copy", "-p", str(src), uri_for(music_root, rel)])
            if r.returncode == 0:
                return True, None
            time.sleep(wait)
        return False, ((r.stderr if isinstance(r.stderr, str) else "") or "")[:200]
    import shutil
    target = music_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return True, None


def copy_from_phone(music_root: Path, src: Path, dst: Path, use_gio: bool = False):
    """Прочитать файл с телефона в локальный путь. fuse cp работает; gio ~2x быстрее."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_gio and is_mtp(src):
        r = run(["gio", "copy", "-p", uri_for(music_root, src.relative_to(music_root)), str(dst)])
        return r.returncode == 0, r.stderr[:200]
    import shutil
    shutil.copy2(src, dst)
    return True, None


def mkdir_phone(music_root: Path, rel: Path) -> bool:
    if not is_mtp(music_root):
        (music_root / rel).mkdir(parents=True, exist_ok=True)
        return True
    return run(["gio", "mkdir", "-p", uri_for(music_root, rel)]).returncode == 0


def cat_md5(music_root: Path, rel: Path, local_path: Path) -> str | None:
    """Прочитать файл с телефона через `gio cat` и вернуть md5 (None при ошибке).
    Не доверять fuse-метаданным сразу после записи — читать байты."""
    import subprocess
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as fh:
        subprocess.run(["gio", "cat", uri_for(music_root, rel)], stdout=fh)
    if local_path.stat().st_size == 0:
        return None
    return md5f(local_path)
