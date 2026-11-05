"""Phone transfer backend: ADB when online (device-side md5, no fuse cache),
gvfs MTP as the fallback. One small surface for the sync scripts."""
from __future__ import annotations

from pathlib import Path

from . import adb, mtp


def backend_name() -> str:
    if adb.adb_bin() is not None and adb.ensure_connected():
        return "adb"
    return "mtp"


class Phone:
    """Album sync over the best available transport."""

    def __init__(self) -> None:
        self.kind = backend_name()
        if self.kind == "adb":
            self.root = adb.music_root()
        else:
            root = mtp.phone_dir()
            if root is None:
                raise RuntimeError("no phone backend available (no adb device, no MTP mount)")
            self.root = root

    # --- queries ---------------------------------------------------------
    def is_dir(self, rel: str) -> bool:
        if self.kind == "adb":
            return adb._shell(f"[ -d {adb.shlex.quote(self.root + '/' + rel)} ] && echo yes").stdout.strip() == "yes"
        return (Path(self.root) / rel).is_dir()

    def list_flacs(self, rel: str) -> list[str]:
        if self.kind == "adb":
            return adb.list_flacs(f"{self.root}/{rel}")
        d = Path(self.root) / rel
        if not d.is_dir():
            return []
        return [p.name for p in d.iterdir() if p.suffix.lower() == ".flac"]

    def md5(self, rel: str) -> str | None:
        if self.kind == "adb":
            return adb.md5(f"{self.root}/{rel}")
        import os
        import tempfile

        tmp = Path(tempfile.mktemp(prefix="mr_md5_"))
        try:
            return mtp.cat_md5(self.root, Path(rel), tmp)
        finally:
            if tmp.exists():
                os.unlink(tmp)

    # --- writes ----------------------------------------------------------
    def push(self, local: Path, rel: str) -> tuple[bool, str]:
        if self.kind == "adb":
            return adb.push(str(local), f"{self.root}/{rel}")
        return mtp.copy_to_phone(self.root, local, Path(rel))
