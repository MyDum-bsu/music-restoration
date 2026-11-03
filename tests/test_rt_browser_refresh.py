"""Cookie refresh script: reads a profile DB copy, writes the secrets files.

Everything runs in tmp_path — no browser profile, no secrets directory of the
machine is read or written.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.rt_browser_refresh import (
    RefreshError,
    missing_required,
    pick_profile,
    read_cookie_file,
    read_firefox_cookies,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rt_browser_refresh.py"
UA = "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/131.0"

SCHEMA = """
CREATE TABLE moz_cookies (
    id INTEGER PRIMARY KEY,
    host TEXT NOT NULL,
    name TEXT NOT NULL,
    value TEXT NOT NULL,
    expiry INTEGER,
    path TEXT
)
"""


def firefox_db(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    db = tmp_path / "cookies.sqlite"
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    con.executemany("INSERT INTO moz_cookies (host, name, value) VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()
    return db


def test_reads_rutracker_cookies_and_ignores_other_hosts(tmp_path: Path):
    db = firefox_db(tmp_path, [
        (".rutracker.org", "bb_session", "session-value"),
        (".rutracker.org", "cf_clearance", "cf-value"),
        (".rutracker.org", "bb_t", "t-value"),
        (".rutracker.org", "bb_ssl", "ssl-value"),
        (".example.com", "other", "ignored"),
    ])

    cookies = read_firefox_cookies(db)

    assert cookies["cf_clearance"] == "cf-value"
    assert cookies["bb_session"] == "session-value"
    assert "other" not in cookies
    assert missing_required(cookies) == []


def test_exact_host_wins_over_the_dotted_one(tmp_path: Path):
    db = firefox_db(tmp_path, [
        (".rutracker.org", "cf_clearance", "dotted"),
        ("rutracker.org", "cf_clearance", "exact"),
    ])

    assert read_firefox_cookies(db)["cf_clearance"] == "exact"


def test_missing_cookie_is_reported_by_name(tmp_path: Path):
    db = firefox_db(tmp_path, [(".rutracker.org", "bb_session", "v")])

    absent = missing_required(read_firefox_cookies(db))

    assert absent == ["cf_clearance", "bb_ssl", "bb_t"]


def test_profile_is_required_when_several_exist(tmp_path: Path):
    home = tmp_path / "home"
    for name in ("aaa.default", "bbb.default-release"):
        (home / ".mozilla" / "firefox" / name).mkdir(parents=True)
        (home / ".mozilla" / "firefox" / name / "cookies.sqlite").write_bytes(b"")

    with pytest.raises(RefreshError, match="several Firefox profiles"):
        pick_profile(None, home)
    chosen = home / ".mozilla" / "firefox" / "bbb.default-release"
    assert pick_profile(str(chosen), home) == chosen / "cookies.sqlite"


def test_missing_profile_is_an_explicit_error(tmp_path: Path):
    with pytest.raises(RefreshError, match="no Firefox profile"):
        pick_profile(None, tmp_path / "home")
    with pytest.raises(RefreshError, match="no cookies.sqlite"):
        pick_profile(str(tmp_path / "nowhere"), tmp_path / "home")


def test_cli_writes_a_parseable_cookie_file(tmp_path: Path):
    profile = tmp_path / "profile"
    profile.mkdir()
    firefox_db(profile, [
        (".rutracker.org", "bb_session", "session-value"),
        (".rutracker.org", "cf_clearance", "cf-value"),
        (".rutracker.org", "bb_t", "t-value"),
        (".rutracker.org", "bb_ssl", "ssl-value"),
    ])
    out = tmp_path / "secrets"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--profile", str(profile), "--out", str(out),
         "--user-agent", UA],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cf-value" not in proc.stdout, "cookie values must not be printed"
    stored = read_cookie_file(out / "rutracker_cookies.txt")
    assert stored["cf_clearance"] == "cf-value"
    assert (out / "rutracker_ua.txt").read_text(encoding="utf-8").strip() == UA

    check = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--check"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )
    assert check.returncode == 0, check.stdout + check.stderr


def test_cli_fails_when_the_profile_has_no_login(tmp_path: Path):
    profile = tmp_path / "profile"
    profile.mkdir()
    firefox_db(profile, [(".rutracker.org", "bb_session", "only-this")])

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--profile", str(profile),
         "--out", str(tmp_path / "secrets"), "--user-agent", UA],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )

    assert proc.returncode == 1
    assert "cf_clearance" in proc.stdout
    assert not (tmp_path / "secrets" / "rutracker_cookies.txt").exists()


def test_help_documents_the_flags():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )

    assert proc.returncode == 0
    for flag in ("--profile", "--out", "--user-agent", "--check", "--dry-run"):
        assert flag in proc.stdout
