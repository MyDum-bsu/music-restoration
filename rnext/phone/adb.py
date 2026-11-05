"""Phone access over ADB (USB or Wi-Fi). Wireless: `adb pair` once, then `adb connect`.

Advantages over gvfs MTP: real device-side md5 (`adb shell md5sum`), no fuse
write-cache lies, no stale mounts. Falls back cleanly when no device is online.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

STATE_FILE = Path.home() / ".config" / "music-restoration" / "phone_adb.json"
DEFAULT_MUSIC = "/sdcard/Music"
_CANDIDATES = (
    Path.home() / "opt" / "platform-tools" / "adb",
    Path("/usr/bin/adb"),
    Path("/usr/local/bin/adb"),
)


def adb_bin() -> str | None:
    for p in _CANDIDATES:
        if p.is_file():
            return str(p)
    from shutil import which

    return which("adb")


def _run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    adb = adb_bin()
    if adb is None:
        raise RuntimeError("adb not found: install platform-tools or put adb on PATH")
    return subprocess.run([adb, *args], capture_output=True, text=True, timeout=timeout)


def save_endpoint(hostport: str) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"endpoint": hostport}), encoding="utf-8")


def saved_endpoint() -> str | None:
    try:
        return json.loads(STATE_FILE.read_text()).get("endpoint")
    except (OSError, ValueError):
        return None


def pair(hostport: str, code: str) -> tuple[bool, str]:
    r = _run(["pair", hostport, code], timeout=120)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def connect(hostport: str) -> tuple[bool, str]:
    r = _run(["connect", hostport], timeout=30)
    ok = r.returncode == 0 and "connected" in (r.stdout + r.stderr)
    if ok:
        save_endpoint(hostport)
    return ok, (r.stdout + r.stderr).strip()


def devices() -> list[str]:
    r = _run(["devices"])
    out = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            out.append(parts[0])
    return out


def connected() -> bool:
    return bool(devices())


def ensure_connected() -> bool:
    """True when a device is online; try the saved endpoint once if not."""
    if connected():
        return True
    ep = saved_endpoint()
    if ep:
        connect(ep)
        return connected()
    return False


def _shell(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return _run(["shell", cmd], timeout=timeout)


def music_root() -> str:
    for cand in (DEFAULT_MUSIC, "/sdcard/music", "/storage/emulated/0/Music"):
        r = _shell(f"[ -d {shlex.quote(cand)} ] && echo yes")
        if r.stdout.strip() == "yes":
            return cand
    return DEFAULT_MUSIC


def md5(remote: str) -> str | None:
    """Device-side md5 hash — no host-side cache involved."""
    r = _shell(f"md5sum {shlex.quote(remote)}")
    line = r.stdout.strip()
    if not line or "No such file" in (r.stderr + line):
        return None
    parts = line.split()
    return parts[0] if parts else None


def push(local: str, remote: str) -> tuple[bool, str]:
    if not Path(local).is_file():
        return False, f"no such file: {local}"
    parent = str(Path(remote).parent)
    mk = _shell(f"mkdir -p {shlex.quote(parent)}")
    if mk.returncode != 0:
        return False, (mk.stdout + mk.stderr).strip()
    r = _run(["push", local, remote], timeout=1800)
    ok = r.returncode == 0
    return ok, (r.stdout + r.stderr).strip()


def pull(remote: str, local: str) -> tuple[bool, str]:
    r = _run(["pull", remote, local], timeout=1800)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def list_flacs(remote_dir: str) -> list[str]:
    r = _shell(f"ls {shlex.quote(remote_dir)}/*.flac 2>/dev/null")
    return [Path(line.strip()).name for line in r.stdout.splitlines() if line.strip()]


def _main(argv: list[str]) -> int:
    import sys

    cmd = argv[0] if argv else "status"
    if cmd == "pair":
        if len(argv) < 3:
            print("usage: adb.py pair <ip:pair_port> <6-digit-code>")
            return 2
        ok, msg = pair(argv[1], argv[2])
        print(("paired" if ok else "pair failed") + f": {msg}")
        return 0 if ok else 1
    if cmd == "connect":
        if len(argv) < 2:
            print("usage: adb.py connect <ip:port>")
            return 2
        ok, msg = connect(argv[1])
        print(("connected" if ok else "connect failed") + f": {msg}")
        return 0 if ok else 1
    if cmd == "status":
        print("adb:", adb_bin() or "MISSING")
        print("devices:", devices() or "none")
        print("saved endpoint:", saved_endpoint() or "none")
        if connected():
            print("music root:", music_root())
        return 0
    if cmd == "ls":
        print("\n".join(list_flacs(argv[1] if len(argv) > 1 else music_root())))
        return 0
    if cmd == "md5":
        print(md5(argv[1]))
        return 0
    print("commands: pair | connect | status | ls [dir] | md5 <file>")
    return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
