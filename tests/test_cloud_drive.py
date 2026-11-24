"""Offline unit tests for rnext.cloud.drive (safe rclone adapter, dry-run default, check --one-way)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rnext.cloud.drive import (
    DEFAULT_REMOTE_BASE,
    DriveError,
    build_remote_uri,
    download_path,
    list_remote,
    upload_path,
    validate_remote_subpath,
    verify_one_way,
)


def test_validate_remote_subpath_rejects_traversal_and_colons() -> None:
    assert validate_remote_subpath("/smoke/run1/") == "smoke/run1"
    for bad in ("", "   ", "../escape", "a/../b", "gdrive:other", "a\\b"):
        with pytest.raises(DriveError):
            validate_remote_subpath(bad)


def test_build_remote_uri_uses_default_base() -> None:
    uri = build_remote_uri("smoke/test1")
    assert uri == f"{DEFAULT_REMOTE_BASE}/smoke/test1"


def test_upload_defaults_to_dry_run_and_explicit_bounded_retries(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"audio-data")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry_run=True must not invoke rclone")

    res = upload_path(wav, "smoke/run1", rclone_bin="/bin/true", runner=fail_if_called)
    assert res.dry_run is True
    assert res.executed is False
    assert res.commands[0][1] == "copy"
    assert "--retries" in res.commands[0] and "--low-level-retries" in res.commands[0]
    assert res.commands[1][1:3] == ["check", "--one-way"]
    assert wav.is_file()


def test_upload_execute_runs_copy_and_one_way_check_without_deleting_local(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    wav = out_dir / "restored.wav"
    wav.write_bytes(b"restored-audio")
    calls: list[list[str]] = []

    def fake_runner(cmd, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="1 matching files")

    res = upload_path(
        out_dir,
        "smoke/run1",
        dry_run=False,
        verify=True,
        rclone_bin="/bin/true",
        runner=fake_runner,
    )
    assert res.executed is True
    assert res.dry_run is False
    assert res.verified is True
    assert len(calls) == 2
    assert calls[0][1] == "copy"
    assert "move" not in calls[0] and "sync" not in calls[0] and "delete" not in calls[0]
    assert calls[1][1:3] == ["check", "--one-way"]
    assert wav.is_file() and wav.read_bytes() == b"restored-audio"


def test_upload_blocks_dotfiles_netrc_secrets_and_sensitive_parent_dirs(tmp_path: Path) -> None:
    for name in ("rutracker_cookies.txt", ".netrc", "_netrc", ".env", "id_rsa", "sessions.json", "_colab_sessions.json"):
        f = tmp_path / name
        f.write_text("sensitive", encoding="utf-8")
        with pytest.raises(DriveError, match="sensitive"):
            upload_path(f, "smoke/leak", dry_run=True)

    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir()
    innocent_name = secrets_dir / "config.json"
    innocent_name.write_text("{}", encoding="utf-8")
    with pytest.raises(DriveError, match="sensitive"):
        upload_path(innocent_name, "smoke/leak", dry_run=True)

    d = tmp_path / "bundle"
    (d / ".secrets").mkdir(parents=True)
    (d / ".secrets" / "api.key").write_text("secret", encoding="utf-8")
    with pytest.raises(DriveError, match="sensitive"):
        upload_path(d, "smoke/bundle", dry_run=True)


def test_upload_requires_positive_matching_file_count_from_rclone_check(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"audio")

    # Real rclone v1.76.0 prints only "0 differences found" when a filter matches 0 files.
    def empty_filter_runner(cmd, **kwargs):
        if cmd[1] == "copy":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="0 differences found")

    with pytest.raises(DriveError, match="check --one-way failed"):
        upload_path(wav, "smoke/run1", dry_run=False, verify=True, rclone_bin="/bin/true", runner=empty_filter_runner)


def test_single_file_upload_and_verify_agree_on_remote_dir_and_escape_globs(tmp_path: Path) -> None:
    wav = tmp_path / "track[1].wav"
    wav.write_bytes(b"wav")
    seen: list[list[str]] = []

    def fake_runner(cmd, **kwargs):
        seen.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="1 matching files")

    res_dir = verify_one_way(wav, "smoke/run1", rclone_bin="/bin/true", runner=fake_runner)
    assert res_dir.ok is True
    assert seen[-1][-1].endswith("/smoke/run1")
    assert r"+ /track\[1\].wav" in seen[-1]
    assert "- *" in seen[-1]

    res_exact = verify_one_way(wav, "smoke/run1/track[1].wav", rclone_bin="/bin/true", runner=fake_runner)
    assert res_exact.ok is True
    assert seen[-1][-1].endswith("/smoke/run1")


def test_download_defaults_to_dry_run_and_ignore_existing(tmp_path: Path) -> None:
    dest = tmp_path / "dl"
    res = download_path("smoke/run1", dest, rclone_bin="/bin/true")
    assert res.dry_run is True
    assert res.executed is False
    assert "--ignore-existing" in res.commands[0]
    assert "--retries" in res.commands[0]
    assert res.commands[0][1] == "copy"


def test_list_remote_parses_lsjson() -> None:
    payload = [{"Name": "smoke", "Path": "smoke", "Size": -1, "IsDir": True}]

    def fake_runner(cmd, **kwargs):
        assert cmd[1] == "lsjson"
        assert "--retries" in cmd
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    entries = list_remote("", rclone_bin="/bin/true", runner=fake_runner)
    assert entries == payload
