"""Portable workspace config: fail-fast without workspace.json, RESTORE_DATA_ROOT priority, example validity."""
import json

import pytest

from rnext.workspace import WorkspaceError, load_config, repo_root

MISSING_MSG = (
    "copy config/workspace.example.json to config/workspace.json and set data_root"
)


def _fake_repo(tmp_path, cfg=None):
    (tmp_path / "config").mkdir(exist_ok=True)
    if cfg is not None:
        (tmp_path / "config" / "workspace.json").write_text(
            json.dumps(cfg), encoding="utf-8"
        )
    return tmp_path


def _live_there(monkeypatch, root):
    # load_config() resolves the repo root from the module (parents[1] of the
    # source file), not from cwd; pin it to the tmp dir and make tmp the cwd.
    monkeypatch.chdir(root)
    monkeypatch.setattr("rnext.workspace.repo_root", lambda: root)


def test_missing_config_fails_fast(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)  # config dir exists, workspace.json does not
    _live_there(monkeypatch, root)
    with pytest.raises(WorkspaceError) as exc:
        load_config()
    assert MISSING_MSG in str(exc.value)


def test_restore_data_root_env_wins_over_config(tmp_path, monkeypatch):
    real = tmp_path / "real-data"
    real.mkdir()
    root = _fake_repo(
        tmp_path, {"data_root": str(tmp_path / "from-config"), "dirs": {}}
    )
    # "from-config" does not exist: if the env did not win, load_config()
    # would raise "data_root is missing" instead of returning the env path.
    monkeypatch.setenv("RESTORE_DATA_ROOT", str(real))
    _live_there(monkeypatch, root)
    cfg = load_config()
    assert cfg["data_root"] == str(real)


def test_example_config_valid_json_with_required_keys():
    example = repo_root() / "config" / "workspace.example.json"
    cfg = json.loads(example.read_text(encoding="utf-8"))
    assert isinstance(cfg.get("data_root"), str) and cfg["data_root"]
    dirs = cfg.get("dirs")
    assert isinstance(dirs, dict) and dirs
    assert all(isinstance(v, str) and v for v in dirs.values())
