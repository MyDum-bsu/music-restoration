"""Phone access over MTP via gvfs + gio. No device-specific constants.

Rules learned the hard way (details in docs/mtp-notes.md):
- gvfs fuse mounts are read-only in practice: writing through them fails with
  'Operation not supported'. All writes go through
  `gio copy -p <local> <mtp://...>` with URL-encoded path segments.
- gio can only overwrite existing files, not create new ones — directories are
  created in advance with `gio mkdir -p`.
- Read large batches via copy_from_phone(use_gio=True) — roughly twice as fast
  as fuse cp.
- Verify writes by reading back (gio cat) and comparing md5: fuse metadata
  lies right after an overwrite.
"""
import os
import time
import urllib.parse
from pathlib import Path

from .config import get
from .util import md5f, run

AUDIO_EXTS = {".flac", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".m4b", ".mp4", ".wav"}
# "music" / ru "music" — exact on-device directory names
MUSIC_DIR_NAMES = {"music", "\u043C\u0443\u0437\u044B\u043A\u0430"}


def gvfs_mtp_mounts() -> list[Path]:
    """gvfs MTP mounts (Linux desktops). Empty elsewhere — phone functions
    degrade to a clear error instead of crashing."""
    if not hasattr(os, "getuid"):
        return []
    base = Path("/run/user") / str(os.getuid()) / "gvfs"
    return sorted(p for p in base.glob("mtp:host=*") if p.exists())


def _music_dirs_below(mount: Path, depth: int = 3) -> list[Path]:
    """Search for a Music directory in an MTP mount with a depth limit (fuse is slow)."""
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
    """Music root on the phone: $MR_PHONE_DIR / config / gvfs autodetect."""
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
    """Base gio URI for the music root, e.g. mtp://DEV/Storage/Music."""
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
    """mtp:// URI for a path relative to the music root. Each segment is encoded
    separately — encoding the whole path would turn '/' into %2F and break gio."""
    return mtp_base(music_root) + "/" + "/".join(urllib.parse.quote(seg) for seg in Path(rel).parts)


def phone_alive(music_root: Path | None = None) -> bool:
    root = music_root or phone_dir()
    return root is not None and root.exists()


def copy_to_phone(music_root: Path, src: Path, rel: Path, retries: int = 3, wait: int = 5):
    """Copy a local file over an EXISTING file on the phone."""
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
    """Read a file from the phone into a local path. fuse cp works; gio is ~2x faster."""
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
    """Read a file from the phone via `gio cat` and return its md5 (None on failure).
    Do not trust fuse metadata right after a write — read the bytes."""
    import subprocess
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as fh:
        subprocess.run(["gio", "cat", uri_for(music_root, rel)], stdout=fh)
    if local_path.stat().st_size == 0:
        return None
    return md5f(local_path)
