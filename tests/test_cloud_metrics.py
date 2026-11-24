"""Offline unit tests for rnext.cloud.metrics (DR, LUFS/TP, spectrum, artifacts, SHA-256 manifests)."""
from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rnext.cloud.metrics import (
    MANIFEST_SCHEMA_VERSION,
    MetricsError,
    artifact_check,
    build_manifest,
    compare_metrics,
    measure_audio_arrays,
    measure_audio_file,
    measure_lufs_tp,
    sha256_file,
    spectral_profile,
    write_manifest,
)


def _write_wav(path: Path, x: np.ndarray, sr: int = 44100) -> Path:
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def test_spectral_profile_distinguishes_tone_from_noise() -> None:
    sr = 44100
    t = np.arange(sr * 2, dtype=np.float64) / sr
    tone = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)
    rng = np.random.default_rng(42)
    noise = 0.25 * rng.standard_normal(sr * 2)

    prof_tone = spectral_profile(tone, sr)
    prof_noise = spectral_profile(noise, sr)

    assert 0.0 <= prof_tone["spectral_flatness"] < 0.05
    assert prof_noise["spectral_flatness"] > 0.3
    assert prof_tone["nyquist_hz"] == 22050.0


def test_artifact_check_detects_clipping_and_spikes() -> None:
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    clean = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    res_clean = artifact_check(clean, tp_dbfs=-3.0)
    assert res_clean["clipped_samples"] == 0
    assert res_clean["spike_count"] == 0
    assert res_clean["passed"] is True

    clipped = np.clip(clean * 5.0, -1.0, 1.0)
    clipped[1000, 0] = -1.0
    clipped[1001, 0] = 1.0
    res_bad = artifact_check(clipped, tp_dbfs=0.2)
    assert res_bad["clipped_samples"] > 0
    assert res_bad["spike_count"] >= 1
    assert res_bad["passed"] is False


def test_measure_lufs_tp_parses_ffmpeg_stderr() -> None:
    sample_stderr = (
        "  Integrated loudness:\n"
        "    I:         -14.2 LUFS\n"
        "    Threshold: -24.5 LUFS\n"
        "  True peak:\n"
        "    Peak:      -0.8 dBFS\n"
    )

    def fake_runner(cmd, **kwargs):
        assert "ebur128=peak=true" in cmd
        return SimpleNamespace(returncode=0, stdout="", stderr=sample_stderr)

    lufs, tp = measure_lufs_tp(Path("dummy.wav"), runner=fake_runner)
    assert lufs == -14.2
    assert tp == -0.8


def test_measure_audio_file_and_compare_metrics(tmp_path: Path) -> None:
    sr = 44100
    t = np.arange(sr * 4, dtype=np.float32) / sr
    # Clipped signal vs restored softer signal
    base = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1320 * t)
    before_arr = np.clip(np.stack([base * 2.5, base * 2.5], axis=1), -1.0, 1.0)
    after_arr = np.stack([base * 0.7, base * 0.7], axis=1)

    p_before = _write_wav(tmp_path / "before.wav", before_arr, sr)
    p_after = _write_wav(tmp_path / "after.wav", after_arr, sr)

    m_before = measure_audio_file(p_before, include_lufs=True)
    m_after = measure_audio_file(p_after, include_lufs=True)

    assert m_before["sample_rate"] == sr
    assert m_before["channels"] == 2
    assert len(m_before["sha256"]) == 64
    assert m_before["artifacts"]["clipped_samples"] > 0
    assert m_after["artifacts"]["clipped_samples"] == 0
    assert m_before["lufs_i"] is not None
    assert m_after["tp_dbfs"] is not None

    delta = compare_metrics(m_before, m_after)
    assert delta["clipped_samples_delta"] < 0
    assert delta["lufs_i_delta"] < 0


def test_build_and_write_manifest_deterministic_and_portable(tmp_path: Path) -> None:
    inp = tmp_path / "input.wav"
    out = tmp_path / "input_gpu_smoke.wav"
    inp.write_bytes(b"RIFF-input-bytes")
    out.write_bytes(b"RIFF-output-bytes")

    m_before = measure_audio_arrays(np.zeros((44100 * 4, 2), dtype=np.float32) + 0.1, 44100, lufs_i=-16.0, tp_dbfs=-1.0)
    m_after = measure_audio_arrays(np.zeros((44100 * 4, 2), dtype=np.float32) + 0.05, 44100, lufs_i=-18.0, tp_dbfs=-2.0)

    manifest = build_manifest(
        task="gpu_smoke",
        gpu="T4",
        session="mr-smoke-test",
        input_path=inp,
        output_paths=[out],
        metrics_before=m_before,
        metrics_after=m_after,
        colab_metadata={"gpu_name": "Tesla T4", "cuda_available": True},
        drive_uri="gdrive:darkfactory-storage/music-restoration/smoke/run1",
    )
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["input"]["name"] == "input.wav"
    assert manifest["input"]["sha256"] == sha256_file(inp)
    assert manifest["outputs"][0]["name"] == "input_gpu_smoke.wav"
    assert manifest["outputs"][0]["sha256"] == sha256_file(out)
    assert manifest["metrics_delta"]["lufs_i_delta"] == -2.0

    # Manifest must not leak absolute machine paths.
    m_path = write_manifest(tmp_path / "manifest.json", manifest)
    text = m_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert json.loads(text) == manifest


def test_sha256_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MetricsError, match="missing file"):
        sha256_file(tmp_path / "does_not_exist.wav")
