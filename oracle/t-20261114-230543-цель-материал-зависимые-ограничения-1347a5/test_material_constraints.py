"""Oracle: material-dependent ceilings, release-level targets, loudness pass.

Contract of t-20261114-230543-цель-материал-зависимые-ограничения-1347a5.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def load(name: str, rel: str):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def sel():
    return load("rnext_selection_oracle", "rnext/dsp/selection.py")


@pytest.fixture(scope="session")
def safe():
    return load("restore_safe_oracle", "scripts/restore_safe.py")


@pytest.fixture(scope="session")
def am():
    return load("audio_measure_oracle", "scripts/audio_measure.py")


def write_wav(path: Path, amplitude: float = 0.5, clipped: bool = False,
              seconds: float = 2.0, sr: int = 48000) -> None:
    t = np.arange(int(sr * seconds)) / sr
    x = np.clip(np.sin(2 * np.pi * 1000.0 * t) * 4.0, -1, 1) if clipped \
        else np.sin(2 * np.pi * 1000.0 * t) * amplitude
    pcm = np.stack([np.round(x * 32767), np.round(x * 32767)], axis=1).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def stereo_sine(sr: int = 4000, seconds: float = 24.0, amplitude: float = 0.5,
                burst: bool = False) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    x = np.sin(2 * np.pi * 220.0 * t) * amplitude
    if burst:
        env = ((np.sin(2 * np.pi * 0.5 * t) > 0).astype(float) * 0.95 + 0.05)
        x = x * env
    return np.stack([x, x], axis=1).astype(np.float32)


# --- material-dependent ceiling ----------------------------------------------


def test_tp_ceiling_follows_material_class(sel):
    crushed = sel.material_plan({"dr": 5, "rms_tt": -9.0}, 30.0)
    mid = sel.material_plan({"dr": 9, "rms_tt": -9.0}, 30.0)
    sparse = sel.material_plan({"dr": 5, "rms_tt": -20.0}, 30.0, allow_sparse=True)
    assert crushed["tp_ceiling"] == -0.3
    assert mid["tp_ceiling"] == -0.5
    assert sparse["tp_ceiling"] == -1.5
    assert sel.material_plan({"dr": 12, "rms_tt": -9.0}, 30.0)["tp_ceiling"] is None
    policies = sel.candidate_policies(crushed)
    assert policies["gentle"].target_tp_dbfs == -0.3
    assert sel.candidate_policies(sparse)["adaptive"].target_tp_dbfs == -1.5


def test_output_violations_use_the_plan_ceiling(sel):
    sr = 48000
    x = stereo_sine(sr=48000, seconds=1.0, amplitude=0.5)
    y = stereo_sine(sr=48000, seconds=1.0, amplitude=10 ** (-0.4 / 20))
    orig, meas = sel.measure(x, sr), sel.measure(y, sr)
    assert "true_peak_above_ceiling" in sel.output_violations(x, y, sr, orig, meas)
    assert "true_peak_above_ceiling" not in sel.output_violations(
        x, y, sr, orig, meas, tp_ceiling=-0.3)


# --- release-level dynamics target -------------------------------------------


def test_release_target_from_least_compressed_third(sel):
    uniform = [{"dr": 6} for _ in range(6)]
    assert sel.release_target(uniform) is None
    spread = [{"dr": d} for d in (4, 5, 5, 6, 9, 10)]
    assert sel.release_target(spread) == 4.0  # median [9,10] - median [4..10]
    huge = [{"dr": d} for d in (1, 1, 1, 20, 20, 20)]
    assert sel.release_target(huge) == 5.3
    assert sel.release_target([{"dr": 3}, {"dr": 9}]) is None


def test_material_plan_uses_release_target(sel):
    crushed = {"dr": 5, "rms_tt": -9.0}
    assert sel.material_plan(crushed, 30.0)["target"] == 5.3
    assert sel.material_plan(crushed, 30.0, release_target=2.0)["target"] == 2.0
    sparse = {"dr": 5, "rms_tt": -20.0}
    assert sel.material_plan(sparse, 30.0, release_target=2.0,
                             allow_sparse=True)["target"] == 0.8


def test_select_track_threads_release_target(sel):
    x = stereo_sine()
    report = sel.select_track(x.copy(), 4000, release_target=1.0)[1]
    assert report["plan"]["class"] == "crushed"
    assert report["plan"]["target"] == 1.0
    assert report["plan"]["tp_ceiling"] == -0.3


# --- opt-in loudness pass -----------------------------------------------------


def make_flac(tmp_path: Path, name: str, clipped: bool) -> Path:
    wav = tmp_path / f"{name}.wav"
    write_wav(wav, clipped=clipped)
    flac = tmp_path / f"{name}.flac"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(flac)],
                   check=True, capture_output=True)
    return flac


def test_loudness_pass_hits_target_under_ceiling(safe, am, tmp_path):
    """A declipped loud candidate gained to the target must land there."""
    src = make_flac(tmp_path, "loud", clipped=True)
    dest = tmp_path / "out.flac"
    x, sr = safe.F.decode_f32(str(src), strict=True, channels=None)
    y, report = safe.select_restoration(x, sr, mode="declip", bits=16)
    assert report["status"] == "processed_candidate", report.get("violations")
    safe.write_24(y, sr, dest, 0)
    report["loudness_target"] = -20.0
    y2 = safe._loudness_pass(x, y, dest, report, sr)
    assert report["loudness_plan"]["applied"] is True
    assert report["status"] == "processed_candidate"
    assert not report["violations"]
    lufs, tp = am.lufs_tp(dest)
    assert abs(lufs - (-20.0)) <= 0.5
    assert tp <= -0.95


def test_loudness_pass_rejects_on_violation(safe, tmp_path):
    """A pass over an invalid candidate cannot ship: refuse, never deliver."""
    sr = 48000
    x = stereo_sine(sr=sr, seconds=1.0, amplitude=0.05)
    y = stereo_sine(sr=sr, seconds=2.0, amplitude=0.5)
    dest = tmp_path / "cand.flac"
    safe.write_24(y, sr, dest, 0)
    report = {"loudness_target": -20.0, "orig": safe.measure(x, sr),
              "mode": "dynamics", "plan": {"class": "crushed", "target": 5.3},
              "policy": {"target_tp_dbfs": -1.0}, "violations": []}
    with pytest.raises(ValueError, match="violated output constraints"):
        safe._loudness_pass(x, y, dest, report, sr)


# --- qc_gates LUFS window -----------------------------------------------------


def run_gates(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "qc_gates.py"), *args],
        capture_output=True, text=True, cwd=str(REPO))


def test_qc_gates_lufs_window(tmp_path):
    wav = tmp_path / "quiet.wav"
    write_wav(wav, amplitude=0.02)
    ok = run_gates(str(wav))
    assert ok.returncode == 0, ok.stdout
    fail = run_gates("--lufs-window=-14.0:-11.0", str(wav))
    assert fail.returncode == 1
    rec = json.loads(fail.stdout.strip().splitlines()[-1])
    assert rec["lufs_window"] == [-14.0, -11.0]
    assert rec["lufs_fail"] == "outside_window"
    bad = run_gates("--lufs-window", "not-a-window", str(wav))
    assert bad.returncode == 2
