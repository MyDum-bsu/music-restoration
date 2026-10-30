"""Secrets from ~/.config/music-restoration/secrets — never stored in the repo.

The old ~/.config/restore-next/secrets directory is read as a fallback so the
project rename does not require manually migrating cookies.
"""
import json
from pathlib import Path

_NEW = Path.home() / ".config" / "music-restoration" / "secrets"
_OLD = Path.home() / ".config" / "restore-next" / "secrets"
SECRETS = _NEW if _NEW.exists() or not _OLD.exists() else _OLD


def rutracker_cookies() -> dict[str, str]:
    """File rutracker_cookies.txt: 'name=value' lines (empty/# lines are ignored).

    Expected at minimum: bb_session, bb_ssl, bb_t, cf_clearance.
    bb_data (old scheme) is also picked up if present.
    """
    f = SECRETS / "rutracker_cookies.txt"
    if not f.is_file():
        return {}
    out: dict[str, str] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        if name and value:
            out[name] = value
    return out


def rutracker_ua() -> str | None:
    """User-Agent from exactly the browser the cookies came from (matters for cf_clearance)."""
    f = SECRETS / "rutracker_ua.txt"
    if f.is_file():
        v = f.read_text(encoding="utf-8").strip()
        if v.startswith("Mozilla/"):
            return v
    return None


def transmission_rpc() -> dict:
    f = SECRETS / "transmission_rpc.json"
    return json.loads(f.read_text(encoding="utf-8"))
