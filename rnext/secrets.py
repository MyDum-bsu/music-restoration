"""Секреты из ~/.config/restore-next/secrets — никогда не хранятся в репо."""
import json
from pathlib import Path

SECRETS = Path.home() / ".config" / "restore-next" / "secrets"


def rutracker_cookies() -> dict[str, str]:
    """Файл rutracker_cookies.txt: строки 'name=value' (пустые/с # игнорируются).

    Ожидаем минимум: bb_session, bb_ssl, bb_t, cf_clearance.
    bb_data (старая схема) тоже подхватится, если есть.
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
    """User-Agent точно из того браузера, откуда взяты cookies (важно для cf_clearance)."""
    f = SECRETS / "rutracker_ua.txt"
    if f.is_file():
        v = f.read_text(encoding="utf-8").strip()
        if v.startswith("Mozilla/"):
            return v
    return None


def transmission_rpc() -> dict:
    f = SECRETS / "transmission_rpc.json"
    return json.loads(f.read_text(encoding="utf-8"))
