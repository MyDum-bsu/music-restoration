"""doctor: exit codes 0/1/2 and the JSON report schema.

shutil.which and the workspace config are mocked (tmp dir); the RPC probe and
the cookies path are pinned, so the tests are hermetic and stay local.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import rnext.doctor as doctor

REPO_ROOT = Path(__file__).resolve().parents[1]

# sys.executable may point at the app host binary (e.g. an AppImage) that
# launched this session rather than at a python interpreter; the wrapper test
# needs the real one from PATH. Resolved at import time, before the autouse
# fixture patches shutil.which for the in-process checks.
REAL_PYTHON3 = shutil.which("python3") or sys.executable

CHECK_IDS = [
    "python", "modules",
    "bin:ffmpeg", "bin:ffprobe", "bin:flac", "bin:metaflac",
    "bin:shntool", "bin:cuetag",
    "config", "data_root", "disk",
    "secrets", "transmission", "gio", "adb",
]


def make_config(tmp_path: Path) -> str:
    """A valid workspace.json pointing at a fresh writable tmp dir."""
    root = tmp_path / "data"
    root.mkdir()
    cfg = tmp_path / "workspace.json"
    cfg.write_text(
        json.dumps({
            "data_root": str(root),
            "budget_bytes": 40 * 1024 ** 3,
            "dirs": {"torrents": "torrents", "corpus": "corpus",
                     "abx": "abx", "cache": "cache"},
        }),
        encoding="utf-8",
    )
    return str(cfg)


@pytest.fixture(autouse=True)
def green_machine(tmp_path, monkeypatch):
    """Baseline: every which() hits, RPC answers, cookies exist, no env override."""
    monkeypatch.delenv("RESTORE_DATA_ROOT", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(doctor, "port_open", lambda host, port, timeout=0.5: True)
    cookies = tmp_path / "rutracker_cookies.txt"
    cookies.write_text("ru_tracker=1", encoding="utf-8")
    monkeypatch.setattr(doctor, "secrets_path", lambda: cookies)


def run_doctor(capsys, *args):
    code = doctor.main(list(args))
    out = capsys.readouterr().out
    return code, json.loads(out) if "--json" in args else out


def assert_schema(doc):
    assert {"python", "platform", "exit_code", "checks"} <= set(doc)
    assert isinstance(doc["python"], str)
    assert doc["exit_code"] in (0, 1, 2)
    assert [c["id"] for c in doc["checks"]] == CHECK_IDS
    for check in doc["checks"]:
        assert {"id", "name", "severity", "status", "detail", "fix"} <= set(check)
        assert check["severity"] in ("hard", "warn")
        assert check["name"] and check["detail"]
        assert isinstance(check["fix"], str)
        if check["severity"] == "hard":
            assert check["status"] in ("ok", "fail")
        else:
            assert check["status"] in ("ok", "warn")
        if check["status"] == "fail":
            # every hard failure carries an install command per platform
            assert set(check["fix_by_platform"]) == {"apt", "dnf", "pacman", "brew"}
            assert all(check["fix_by_platform"].values())


def test_all_green_exit_0(tmp_path, capsys):
    code, doc = run_doctor(capsys, "--config", make_config(tmp_path), "--json")
    assert code == 0
    assert doc["exit_code"] == 0
    assert all(c["status"] == "ok" for c in doc["checks"])
    assert_schema(doc)


def test_missing_binary_exit_1(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        shutil, "which", lambda n: None if n == "flac" else f"/usr/bin/{n}")
    code, doc = run_doctor(capsys, "--config", make_config(tmp_path), "--json")
    assert code == 1
    failed = [c for c in doc["checks"] if c["status"] == "fail"]
    assert [c["id"] for c in failed] == ["bin:flac"]
    per = failed[0]["fix_by_platform"]
    assert per == {
        "apt": "sudo apt install flac",
        "dnf": "sudo dnf install flac",
        "pacman": "sudo pacman -S flac",
        "brew": "brew install flac",
    }
    assert failed[0]["fix"] in per.values()
    assert_schema(doc)


def test_missing_config_exit_1(tmp_path, capsys):
    code, doc = run_doctor(
        capsys, "--config", str(tmp_path / "nope.json"), "--json")
    assert code == 1
    by_id = {c["id"]: c for c in doc["checks"]}
    assert by_id["config"]["status"] == "fail"
    assert "не найден" in by_id["config"]["detail"]
    assert by_id["data_root"]["status"] == "fail"
    assert by_id["disk"]["status"] == "fail"
    assert_schema(doc)


def test_invalid_config_json_exit_1(tmp_path, capsys):
    cfg = tmp_path / "workspace.json"
    cfg.write_text("{not json", encoding="utf-8")
    code, doc = run_doctor(capsys, "--config", str(cfg), "--json")
    assert code == 1
    by_id = {c["id"]: c for c in doc["checks"]}
    assert by_id["config"]["status"] == "fail"
    assert "JSON" in by_id["config"]["detail"]


def test_config_without_data_root_exit_1(tmp_path, capsys):
    cfg = tmp_path / "workspace.json"
    cfg.write_text(json.dumps({"budget_bytes": 1, "dirs": {}}), encoding="utf-8")
    code, doc = run_doctor(capsys, "--config", str(cfg), "--json")
    assert code == 1
    by_id = {c["id"]: c for c in doc["checks"]}
    assert by_id["config"]["status"] == "fail"
    assert "data_root" in by_id["config"]["detail"]


def test_only_warns_exit_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        shutil, "which",
        lambda n: None if n in ("gio", "adb") else f"/usr/bin/{n}")
    monkeypatch.setattr(doctor, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(doctor, "secrets_path",
                        lambda: tmp_path / "absent.txt")
    code, doc = run_doctor(capsys, "--config", make_config(tmp_path), "--json")
    assert code == 2
    assert not any(c["status"] == "fail" for c in doc["checks"])
    warns = {c["id"]: c for c in doc["checks"] if c["status"] == "warn"}
    assert set(warns) == {"secrets", "transmission", "gio", "adb"}
    assert "tracker-сценарий недоступен" in warns["secrets"]["detail"]
    assert "скачивание недоступно" in warns["transmission"]["detail"]
    assert "MTP-доставка недоступна" in warns["gio"]["detail"]
    assert "доставка по ADB недоступна" in warns["adb"]["detail"]
    assert_schema(doc)


def test_table_mode_headers(tmp_path, capsys):
    code, out = run_doctor(capsys, "--config", make_config(tmp_path))
    assert code == 0
    assert "СТАТУС" in out and "ПРОВЕРКА" in out and "КАК ИСПРАВИТЬ" in out
    assert "[OK]" in out


def test_scripts_wrapper_json(tmp_path):
    """End-to-end: scripts/doctor.py prints JSON and its returncode matches the report."""
    proc = subprocess.run(
        [REAL_PYTHON3, str(REPO_ROOT / "scripts" / "doctor.py"),
         "--json", "--config", make_config(tmp_path)],
        capture_output=True, text=True, timeout=300,
    )
    try:
        doc = json.loads(proc.stdout)
    except ValueError:
        raise AssertionError(
            f"stdout is not JSON: head={proc.stdout[:200]!r} "
            f"tail={proc.stdout[-200:]!r} stderr={proc.stderr[:200]!r}"
        )
    assert proc.returncode == doc["exit_code"]
    assert doc["exit_code"] in (0, 1, 2)
    assert [c["id"] for c in doc["checks"]] == CHECK_IDS
