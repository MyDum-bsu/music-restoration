"""Small shared helpers: subprocess with safe decoding, md5, file hashing.

run() decodes stdout/stderr with errors="replace" — plain subprocess.run(text=True)
crashes with UnicodeDecodeError on binary output (e.g. `gio cat` of a FLAC),
and bytes-only helpers crash on .stdout.lower(). This bit four different
scripts before it became the single canonical helper.
"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


def python_exe() -> str:
    """sys.executable — unless it does not look like a python interpreter
    (sandboxed AppImage launchers shadow it), then fall back to python3 on PATH."""
    exe = Path(sys.executable or "")
    if exe.name.lower().startswith("python"):
        return str(exe)
    return shutil.which("python3") or "python3"


def run(cmd, timeout: int = 1800, check: bool = False) -> subprocess.CompletedProcess:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, timeout=timeout)
    p.stdout = p.stdout.decode("utf-8", "replace")
    p.stderr = p.stderr.decode("utf-8", "replace")
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd[:3]))}... failed:\n{(p.stderr or p.stdout)[-400:]}")
    return p


def md5f(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
