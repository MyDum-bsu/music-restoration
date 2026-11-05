"""ADB phone backend: endpoint persistence and output parsing, no device needed."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext.phone import adb  # noqa: E402


def test_saved_endpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(adb, "STATE_FILE", tmp_path / "phone_adb.json")
    assert adb.saved_endpoint() is None
    adb.save_endpoint("192.168.1.50:43210")
    assert adb.saved_endpoint() == "192.168.1.50:43210"


def test_list_flacs_parses_ls_output(monkeypatch):
    monkeypatch.setattr(
        adb, "_shell", lambda cmd, timeout=60: adb.subprocess.CompletedProcess(cmd, 0, "/sdcard/Music/a/x.flac\n/sdcard/Music/a/y.flac\n", "")
    )
    assert adb.list_flacs("/sdcard/Music/a") == ["x.flac", "y.flac"]


def test_md5_reads_device_hash(monkeypatch):
    monkeypatch.setattr(
        adb, "_shell", lambda cmd, timeout=60: adb.subprocess.CompletedProcess(cmd, 0, "0123abcd  /sdcard/Music/a/x.flac\n", "")
    )
    assert adb.md5("/sdcard/Music/a/x.flac") == "0123abcd"


def test_md5_missing_file_returns_none(monkeypatch):
    monkeypatch.setattr(
        adb, "_shell", lambda cmd, timeout=60: adb.subprocess.CompletedProcess(cmd, 1, "", "No such file or directory")
    )
    assert adb.md5("/sdcard/Music/nope.flac") is None


def test_adb_bin_missing_reports_none(tmp_path, monkeypatch):
    monkeypatch.setattr(adb, "_CANDIDATES", ())
    monkeypatch.setenv("PATH", str(tmp_path))
    assert adb.adb_bin() is None
