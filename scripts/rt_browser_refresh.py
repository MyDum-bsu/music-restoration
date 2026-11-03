#!/usr/bin/env python3
"""Refresh the rutracker cookies (cf_clearance included) from a Firefox profile.

Why this exists: `cf_clearance` is minted by Cloudflare for one IP + one
User-Agent and expires. When it goes stale every request fails and the tracker
flow stops — until somebody re-exports the cookies by hand. This script does
that step, offline, from a browser profile.

  python3 scripts/rt_browser_refresh.py --check
  python3 scripts/rt_browser_refresh.py --profile ~/.mozilla/firefox/abcd.default-release
  python3 scripts/rt_browser_refresh.py --profile <dir> --user-agent "<UA>" --dry-run

What it does:
  1. reads the rutracker.org cookies out of a COPY of the profile's
     cookies.sqlite (the browser may be running; the live DB is not touched);
  2. writes `rutracker_cookies.txt` (`name=value` per line, the format
     rnext.secrets parses) and `rutracker_ua.txt` into the secrets directory;
  3. reads the result back and verifies that the required cookies are there.

Cookie values are never printed: only names and value lengths.

Not supported on purpose: Chromium-family profiles. Their cookie values are
encrypted with an OS-keyring key (v10/v11) and decrypting that is not
something this script pretends to do — export those cookies from DevTools
instead (Application → Cookies → rutracker.org).
"""
import argparse
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import secrets  # noqa: E402

# rnext.rtclient needs a logged-in session; cf_clearance expires fastest
# (Cloudflare), the bb_* triple is the login itself.
REQUIRED = ("cf_clearance", "bb_session", "bb_ssl", "bb_t")
HOSTS = ("rutracker.org", ".rutracker.org", "www.rutracker.org")
HOST_ORDER = {host: i for i, host in enumerate(HOSTS)}
COOKIE_FILE = "rutracker_cookies.txt"
UA_FILE = "rutracker_ua.txt"


class RefreshError(RuntimeError):
    """An explicit, actionable failure — never a silent fallback."""


def firefox_profiles(home: Path) -> list[Path]:
    root = home / ".mozilla" / "firefox"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "cookies.sqlite").is_file())


def pick_profile(spec: str | None, home: Path) -> Path:
    """The cookies.sqlite to read: explicit, or the only Firefox profile there is."""
    if spec:
        path = Path(spec).expanduser()
        if path.is_file():                     # direct cookies.sqlite
            return path
        if (path / "cookies.sqlite").is_file():
            return path / "cookies.sqlite"
        raise RefreshError(
            f"{path}: no cookies.sqlite — pass the Firefox profile directory "
            f"or the file itself"
        )
    found = firefox_profiles(home)
    if not found:
        raise RefreshError(
            "no Firefox profile with cookies.sqlite found; pass --profile <dir> "
            "(Chromium cookies are encrypted and are not read by this script)"
        )
    if len(found) > 1:
        listing = "\n  ".join(str(p) for p in found)
        raise RefreshError(
            f"several Firefox profiles found, pick one with --profile:\n  {listing}"
        )
    return found[0] / "cookies.sqlite"


def read_firefox_cookies(db_path: Path) -> dict[str, str]:
    """rutracker.org cookies from a private copy of the profile database."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "cookies.sqlite"
        shutil.copy2(db_path, copy)            # the browser keeps its DB locked
        con = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
        try:
            placeholders = ",".join("?" * len(HOSTS))
            rows = con.execute(
                f"SELECT host, name, value FROM moz_cookies "
                f"WHERE host IN ({placeholders}) ORDER BY name",
                HOSTS,
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise RefreshError(f"{db_path}: not a readable cookies.sqlite ({exc})") from exc
        finally:
            con.close()

    out: dict[str, str] = {}
    best: dict[str, int] = {}
    for host, name, value in rows:
        if not value:
            continue
        rank = HOST_ORDER.get(host, len(HOSTS))
        if rank <= best.get(name, len(HOSTS) + 1):
            out[name] = value
            best[name] = rank
    return out


def read_cookie_file(path: Path) -> dict[str, str]:
    """Same `name=value` format rnext.secrets reads: comments and blanks skipped."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() and value.strip():
            out[name.strip()] = value.strip()
    return out


def missing_required(cookies: dict[str, str]) -> list[str]:
    return [name for name in REQUIRED if not cookies.get(name)]


def require_all(cookies: dict[str, str], where: str) -> None:
    absent = missing_required(cookies)
    if absent:
        raise RefreshError(
            f"{where}: no " + ", ".join(absent) +
            " for rutracker.org — log in to the tracker in that browser first"
        )


def resolve_user_agent(explicit: str | None) -> str:
    if explicit:
        if not explicit.startswith("Mozilla/"):
            raise RefreshError(
                f"--user-agent does not look like a UA string: {explicit[:20]}..."
            )
        return explicit
    stored = secrets.rutracker_ua()
    if stored:
        return stored
    raise RefreshError(
        "no User-Agent: cf_clearance is bound to the UA of the browser that "
        "minted it. Pass it explicitly: --user-agent \"<UA of that browser>\""
    )


def write_secrets(cookies: dict[str, str], ua: str, out_dir: Path, dry_run: bool) -> tuple[Path, Path]:
    cookie_path = out_dir / COOKIE_FILE
    ua_path = out_dir / UA_FILE
    if dry_run:
        print(f"would write {cookie_path} ({len(cookies)} cookies) and {ua_path}")
        return cookie_path, ua_path
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# rutracker.org cookies, refreshed from a browser profile",
             f"# names: {', '.join(sorted(cookies))}"]
    lines += [f"{name}={value}" for name, value in sorted(cookies.items())]
    cookie_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cookie_path.chmod(0o600)
    ua_path.write_text(ua + "\n", encoding="utf-8")
    ua_path.chmod(0o600)
    return cookie_path, ua_path


def describe(cookies: dict[str, str]) -> str:
    return ", ".join(f"{name} ({len(value)} chars)" for name, value in sorted(cookies.items()))


def cmd_refresh(args) -> int:
    profile = pick_profile(args.profile, Path.home())
    cookies = read_firefox_cookies(profile)
    print(f"profile: {profile}")
    print(f"cookies: {describe(cookies) or '—'}")
    require_all(cookies, str(profile))
    ua = resolve_user_agent(args.user_agent)
    print(f"user-agent: {ua}")
    cookie_path, ua_path = write_secrets(cookies, ua, Path(args.out), args.dry_run)
    if args.dry_run:
        return 0
    stored = read_cookie_file(cookie_path)
    require_all(stored, str(cookie_path))
    if not ua_path.is_file() or not ua_path.read_text(encoding="utf-8").strip():
        raise RefreshError(f"{ua_path}: user-agent file was not written")
    print(f"ok: {cookie_path} holds {len(stored)} cookies; rnext.rtclient can use them now")
    return 0


def cmd_check(args) -> int:
    out_dir = Path(args.out)
    cookie_path = out_dir / COOKIE_FILE
    if not cookie_path.is_file():
        print(f"FAIL: {cookie_path} is missing — run this script without --check")
        return 1
    cookies = read_cookie_file(cookie_path)
    absent = missing_required(cookies)
    if absent:
        print(f"FAIL: {cookie_path} has no " + ", ".join(absent))
        return 1
    ua_path = out_dir / UA_FILE
    if not ua_path.is_file() or not ua_path.read_text(encoding="utf-8").strip():
        print(f"FAIL: {ua_path} is missing — cf_clearance needs the browser UA")
        return 1
    print(f"ok: {cookie_path} holds {len(cookies)} cookies, UA in {ua_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Refresh rutracker cookies (cf_clearance) from a Firefox profile")
    ap.add_argument("--profile", help="Firefox profile directory or its cookies.sqlite "
                                      "(default: the only profile found)")
    ap.add_argument("--out", default=str(secrets.SECRETS),
                    help="secrets directory (default: %(default)s)")
    ap.add_argument("--user-agent", help="UA of the browser that minted cf_clearance "
                                         "(default: reuse rutracker_ua.txt)")
    ap.add_argument("--check", action="store_true",
                    help="only verify the stored cookies are complete")
    ap.add_argument("--dry-run", action="store_true", help="read and verify, write nothing")
    args = ap.parse_args()
    try:
        return cmd_check(args) if args.check else cmd_refresh(args)
    except RefreshError as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
