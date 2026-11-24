"""Reproducible audio quality metrics and SHA-256 artifact manifests for cloud GPU runs.

Delegates decoding and TT DR computation to ``rnext.features`` (one-owner rule)
and adds integrated LUFS, True Peak, spectral profile (cutoff, rolloff, HF floor,
spectral flatness), artifact checks (clipping, click-spikes, DC offset, NaN/Inf),
before/after deltas, and deterministic SHA-256 manifests.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from rnext import features as F

MANIFEST_SCHEMA_VERSION = 1
_SHA256_CHUNK_BYTES = 1024 * 1024


class MetricsError(RuntimeError):
    """Raised when an audio file cannot be measured or hashed."""


def sha256_file(path: Path) -> str:
    """Return lowercase hex SHA-256 digest of a file."""
    target = Path(path)
    if not target.is_file():
        raise MetricsError(f"cannot hash missing file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        while True:
            chunk = fh.read(_SHA256_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _loudest_window(x: np.ndarray, sr: int, win_s: float = 2.0) -> np.ndarray:
    """Return the loudest window of duration ``win_s`` seconds."""
    n = int(win_s * sr)
    if len(x) <= n:
        return x
    mono = np.abs(x).mean(axis=1) if x.ndim > 1 else np.abs(x)
    step = max(1, n // 4)
    best_i, best_e = 0, -1.0
    for i in range(0, len(mono) - n + 1, step):
        energy = float(np.mean(mono[i : i + n]))
        if energy > best_e:
            best_e, best_i = energy, i
    return x[best_i : best_i + n]


def spectral_profile(x: np.ndarray, sr: int, drop_db: float = 55.0) -> dict[str, float]:
    """Compute upper cutoff, rolloff, HF floor, and Wiener spectral flatness."""
    seg = _loudest_window(x, sr)
    if seg.ndim > 1:
        seg = seg.mean(axis=1)
    if len(seg) < 64:
        return {
            "cutoff_hz": 0.0,
            "rolloff_db_per_khz": 0.0,
            "hf_floor_db": 0.0,
            "spectral_flatness": 0.0,
            "nyquist_hz": round(sr / 2.0, 2),
        }

    n = 1 << math.floor(math.log2(len(seg)))
    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(seg[:n].astype(np.float64) * win))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # Wiener entropy (spectral flatness) on power spectrum, bounded [0, 1].
    psd = mag ** 2
    eps = 1e-20
    arith_mean = float(np.mean(psd))
    if arith_mean <= eps:
        flatness = 0.0
    else:
        geom_mean = float(math.exp(float(np.mean(np.log(psd + eps)))))
        flatness = max(0.0, min(1.0, geom_mean / (arith_mean + eps)))

    if n < 4096:
        return {
            "cutoff_hz": 0.0,
            "rolloff_db_per_khz": 0.0,
            "hf_floor_db": 0.0,
            "spectral_flatness": round(flatness, 6),
            "nyquist_hz": round(sr / 2.0, 2),
        }

    k = 64
    spec = np.convolve(mag, np.ones(k) / k, mode="same")
    peak = float(spec.max())
    if peak <= 0.0:
        return {
            "cutoff_hz": 0.0,
            "rolloff_db_per_khz": 0.0,
            "hf_floor_db": 0.0,
            "spectral_flatness": round(flatness, 6),
            "nyquist_hz": round(sr / 2.0, 2),
        }

    thresh = peak * (10.0 ** (-drop_db / 20.0))
    idx = np.where(spec > thresh)[0]
    if len(idx) == 0:
        cut = 0.0
        roll = 0.0
        floor_db = 0.0
    else:
        hi = int(idx[-1])
        cut = float(freqs[hi])
        lo = int(np.searchsorted(freqs, max(0.0, cut - 2000.0)))
        if hi > lo + 1 and spec[lo] > 0.0:
            span_khz = max((cut - float(freqs[lo])) / 1000.0, 1e-6)
            roll = 20.0 * math.log10(max(float(spec[hi]), 1e-12) / float(spec[lo])) / span_khz
        else:
            roll = 0.0
        passband = spec[(freqs > 1000.0) & (freqs < max(2000.0, cut - 1000.0))]
        above = spec[(freqs > cut + 500.0) & (freqs < freqs[-1])]
        pb = float(np.median(passband)) if passband.size else 0.0
        hf = float(np.median(above)) if above.size else 0.0
        floor_db = 20.0 * math.log10(max(hf, 1e-12) / max(pb, 1e-12)) if pb > 0.0 else 0.0

    return {
        "cutoff_hz": round(cut, 2),
        "rolloff_db_per_khz": round(float(roll), 2),
        "hf_floor_db": round(float(floor_db), 2),
        "spectral_flatness": round(flatness, 6),
        "nyquist_hz": round(sr / 2.0, 2),
    }


def artifact_check(x: np.ndarray, tp_dbfs: float | None = None) -> dict[str, Any]:
    """Detect clipping, isolated click-spikes, DC offset, and non-finite samples.

    True Peak verification is required for ``passed=True``; when ``tp_dbfs`` is
    omitted, ``tp_checked`` is ``False`` and ``passed`` is ``False``.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]

    nan_or_inf = int(np.sum(~np.isfinite(arr)))
    finite = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

    x16 = np.clip(np.round(finite * 32768.0), -32768, 32767).astype(np.int16)
    clipped_samples = int(np.sum(np.abs(x16.astype(np.int32)) >= 32767))
    clipped_pct = round(100.0 * clipped_samples / max(x16.size, 1), 4)

    dc_offset = round(float(np.max(np.abs(np.mean(finite, axis=0)))) if finite.size else 0.0, 6)

    # Isolated click-spike detection via second-order finite difference.
    spike_count = 0
    if finite.shape[0] >= 4:
        d2 = np.abs(np.diff(finite, n=2, axis=0))
        med = float(np.median(d2))
        mad = float(np.median(np.abs(d2 - med)))
        thresh = max(0.35, med + 12.0 * max(mad, 1e-4))
        spike_count = int(np.sum(d2 > thresh))

    tp_checked = tp_dbfs is not None
    tp_ok = tp_checked and (float(tp_dbfs) <= -0.1)
    passed = (nan_or_inf == 0) and (clipped_pct <= 0.05) and tp_ok

    return {
        "clipped_samples": clipped_samples,
        "clipped_pct": clipped_pct,
        "spike_count": spike_count,
        "dc_offset": dc_offset,
        "nan_or_inf_samples": nan_or_inf,
        "tp_checked": tp_checked,
        "passed": bool(passed),
    }


def measure_lufs_tp(path: Path, *, runner: Any = subprocess.run) -> tuple[float, float]:
    """Measure integrated LUFS and True Peak (dBFS) using ffmpeg ebur128.

    Raises ``MetricsError`` if ffmpeg is missing, fails, times out, or omits
    either integrated loudness or true peak.
    """
    try:
        proc = runner(
            [
                "ffmpeg",
                "-v",
                "info",
                "-i",
                str(path),
                "-filter:a",
                "ebur128=peak=true",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise MetricsError(f"ffmpeg ebur128 failed for {path}: {exc}") from exc

    if int(getattr(proc, "returncode", 1)) != 0:
        err = (getattr(proc, "stderr", "") or "").strip()[-400:]
        raise MetricsError(f"ffmpeg ebur128 exited {proc.returncode} for {path}: {err}")

    out = getattr(proc, "stderr", "") or ""
    lufs: float | None = None
    tp: float | None = None
    i = out.rfind("Integrated loudness")
    if i >= 0:
        for line in out[i:].splitlines()[:4]:
            if "I:" in line and "LUFS" in line:
                try:
                    lufs = round(float(line.split("I:")[1].split("LUFS")[0].strip()), 2)
                except ValueError:
                    pass
    j = out.rfind("True peak")
    if j >= 0:
        for line in out[j:].splitlines()[:4]:
            if "Peak:" in line and "dBFS" in line:
                try:
                    tp = round(float(line.split("Peak:")[1].split("dBFS")[0].strip()), 2)
                except ValueError:
                    pass
    if lufs is None or tp is None:
        raise MetricsError(f"ffmpeg ebur128 did not report LUFS and True Peak for {path}")
    return lufs, tp


def measure_audio_arrays(
    x: np.ndarray,
    sr: int,
    *,
    lufs_i: float | None = None,
    tp_dbfs: float | None = None,
) -> dict[str, Any]:
    """Compute objective audio quality metrics on a decoded float32/float64 array."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.size == 0 or sr <= 0:
        raise MetricsError("audio array is empty or sample rate is non-positive")

    dr, dr_exact, _ch_dr, peak_dbfs, rms_db = F.tt_dr(arr, sr)
    spec = spectral_profile(arr, sr)
    artifacts = artifact_check(arr, tp_dbfs=tp_dbfs)

    return {
        "sample_rate": int(sr),
        "channels": int(arr.shape[1]),
        "num_frames": int(arr.shape[0]),
        "duration_s": round(float(arr.shape[0]) / float(sr), 3),
        "dr": int(dr),
        "dr_exact": round(float(dr_exact), 2),
        "crest_db": round(float(peak_dbfs - rms_db), 2),
        "peak_dbfs": round(float(peak_dbfs), 2),
        "rms_db": round(float(rms_db), 2),
        "lufs_i": lufs_i,
        "tp_dbfs": tp_dbfs,
        "spectral": spec,
        "artifacts": artifacts,
    }


def measure_audio_file(path: Path, *, include_lufs: bool = True) -> dict[str, Any]:
    """Decode an audio file preserving native channels and return objective metrics."""
    target = Path(path)
    if not target.is_file():
        raise MetricsError(f"audio file not found: {target}")
    try:
        x, sr = F.decode_f32(str(target), channels=None)
    except Exception as exc:
        raise MetricsError(f"failed to decode audio file {target}: {exc}") from exc

    lufs_i, tp_dbfs = measure_lufs_tp(target) if include_lufs else (None, None)
    metrics = measure_audio_arrays(x, sr, lufs_i=lufs_i, tp_dbfs=tp_dbfs)
    metrics["file_name"] = target.name
    metrics["size_bytes"] = target.stat().st_size
    metrics["sha256"] = sha256_file(target)
    return metrics


def _delta(after_val: float | None, before_val: float | None, ndigits: int = 4) -> float | None:
    if after_val is None or before_val is None:
        return None
    return round(float(after_val) - float(before_val), ndigits)


def compare_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Return reproducible before/after deltas across loudness, DR, spectrum, and artifacts."""
    spec_b = before.get("spectral", {})
    spec_a = after.get("spectral", {})
    art_b = before.get("artifacts", {})
    art_a = after.get("artifacts", {})

    return {
        "dr_delta": _delta(after.get("dr_exact"), before.get("dr_exact"), 2),
        "crest_db_delta": _delta(after.get("crest_db"), before.get("crest_db"), 2),
        "lufs_i_delta": _delta(after.get("lufs_i"), before.get("lufs_i"), 2),
        "tp_dbfs_delta": _delta(after.get("tp_dbfs"), before.get("tp_dbfs"), 2),
        "cutoff_hz_delta": _delta(spec_a.get("cutoff_hz"), spec_b.get("cutoff_hz"), 2),
        "spectral_flatness_delta": _delta(
            spec_a.get("spectral_flatness"), spec_b.get("spectral_flatness"), 6
        ),
        "clipped_samples_delta": (
            int(art_a["clipped_samples"]) - int(art_b["clipped_samples"])
            if "clipped_samples" in art_a and "clipped_samples" in art_b
            else None
        ),
        "spike_count_delta": (
            int(art_a["spike_count"]) - int(art_b["spike_count"])
            if "spike_count" in art_a and "spike_count" in art_b
            else None
        ),
    }


def build_manifest(
    *,
    task: str,
    gpu: str,
    session: str,
    input_path: Path,
    output_paths: list[Path],
    metrics_before: dict[str, Any] | None = None,
    metrics_after: dict[str, Any] | None = None,
    colab_metadata: dict[str, Any] | None = None,
    drive_uri: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, machine-portable artifact manifest with SHA-256 digests."""
    inp = Path(input_path)
    input_entry = {
        "name": inp.name,
        "size_bytes": inp.stat().st_size,
        "sha256": sha256_file(inp),
    }

    outputs_list = []
    for out_p in sorted((Path(p) for p in output_paths), key=lambda p: p.name):
        if out_p.is_file():
            outputs_list.append(
                {
                    "name": out_p.name,
                    "size_bytes": out_p.stat().st_size,
                    "sha256": sha256_file(out_p),
                }
            )

    delta = (
        compare_metrics(metrics_before, metrics_after)
        if (metrics_before is not None and metrics_after is not None)
        else None
    )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task": task,
        "gpu": gpu,
        "session": session,
        "input": input_entry,
        "outputs": outputs_list,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "metrics_delta": delta,
        "colab_metadata": colab_metadata or {},
        "drive_uri": drive_uri,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    """Write a deterministic JSON manifest (sorted keys, trailing newline)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    target.write_text(payload, encoding="utf-8")
    return target
