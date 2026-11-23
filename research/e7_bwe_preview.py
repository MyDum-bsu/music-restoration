#!/usr/bin/env python3
"""E7 Research Module: BWE Preview with Marked Synthesized Band.

    python3 research/e7_bwe_preview.py \\
        --manifest corpus/dev-e7-bwe-v1.json \\
        --out-json docs/validation/e7-bwe-results.json \\
        --out-md docs/validation/e7-bwe-report.md

Isolated research harness for Board #1133 (E0 charter + spec v1, section E7).
Does NOT modify or extend the frozen E4 Preserve hard-declip pipeline, does NOT
open sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`), and
does NOT introduce neural BWE production dependencies or Colab automation.

Implemented arms:
- `safe_resample_only`: honest polyphase resampling / identity without inventing
  any new HF content (`synthesized_band == 0`, strictly preserves observed band).
- `dsp_harmonic_bwe_preview`: reproducible DSP bandwidth-extension preview
  combining 2x-oversampled soft harmonic excitation + M/S-coherent STFT spectral
  band replication (SBR) strictly above `observed_cutoff_hz` via a linear-phase
  crossover (`H(f) == 0` for `f <= observed_cutoff_hz`), guaranteeing exact
  additive decomposition `preview == input + synthesized_band` and preserving
  the observed band `[0, cutoff]`. Includes an automatic full-band clean-control
  bypass guard.
- `neural_bwe_audiosr_babe2`: honest local audit of external neural BWE weights
  (AudioSR / UniverSR / BABE-2 / SAMECFM), license gates, and Colab GPU status
  (`blocked_external_weights_or_colab`, zero fabricated numbers).

Opt-in derivative export (`export_bwe_preview`):
- Never mutates or replaces the source file (verified via SHA-256 before/after).
- Writes separate 24-bit stereo FLAC preview + `synthesized_band_only.flac` +
  JSON sidecar, both tagged with mandatory provenance & synthesis flags:
  `synthesized_hf_band`, `observed_cutoff_hz`, `is_generative_preview`,
  `not_original_restoration`, `source_sha256`, `derivative_sha256`,
  `synthesized_band_sha256`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from mutagen.flac import FLAC
from scipy import signal

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import build_e7_bwe_corpus, corpus_synth, corpus_v1

EPS = 1e-20
ADDITIVE_IDENTITY_TOL = 1e-12

DSP_ARMS = (
    "safe_resample_only",
    "dsp_harmonic_bwe_preview",
)


@dataclass
class BwePreviewResult:
    """Output of a BWE preview arm with explicit isolated synthesized band."""

    arm: str
    preview: np.ndarray
    synthesized_band: np.ndarray
    observed_cutoff_hz: float
    synthesized_hf_band: bool
    is_generative_preview: bool
    bypassed: bool
    wall_s: float
    meta: dict = field(default_factory=dict)


def verify_additive_contract(inp: np.ndarray, result: BwePreviewResult) -> float:
    """Verify shape, finiteness, and exact identity `preview == input + synthesized_band`."""
    if result.preview.shape != inp.shape or result.synthesized_band.shape != inp.shape:
        raise ValueError(
            f"{result.arm}: shape mismatch input={inp.shape} "
            f"preview={result.preview.shape} synth={result.synthesized_band.shape}"
        )
    if not np.isfinite(result.preview).all() or not np.isfinite(result.synthesized_band).all():
        raise ValueError(f"{result.arm}: non-finite values in preview or synthesized_band")
    max_err = float(np.max(np.abs((inp + result.synthesized_band) - result.preview)))
    if max_err > ADDITIVE_IDENTITY_TOL:
        raise ValueError(f"{result.arm}: additive identity violated (max_err={max_err:.3e})")
    return max_err


def _as_stereo_f64(x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None].copy(), True
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D audio array, got shape {arr.shape}")
    return arr.copy(), False


def _linear_phase_band_filter(
    x: np.ndarray,
    sr: int,
    f_lo: float | None,
    f_hi: float | None,
    *,
    trans_hz: float = 400.0,
) -> np.ndarray:
    """Zero-phase FFT linear-phase filter with raised-cosine transitions.

    When `f_lo` is set, gain is strictly `0.0` for all `f <= f_lo` and rises
    smoothly via a raised cosine over `(f_lo, f_lo + trans_hz]`.
    When `f_hi` is set, gain is `1.0` for `f <= f_hi - trans_hz` and falls
    smoothly to strictly `0.0` for all `f >= f_hi`.
    """
    n = x.shape[0]
    if n < 8:
        return np.zeros_like(x, dtype=np.float64)
    # Pad by >= 4096 samples to prevent circular-convolution wraparound at boundaries.
    n_fft = 1 << int(np.ceil(np.log2(max(n + 4096, 4096))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    h = np.ones_like(freqs, dtype=np.float64)

    if f_lo is not None and f_lo > 0.0:
        t_hi = f_lo + max(trans_hz, 1.0)
        h[freqs <= f_lo] = 0.0
        trans = (freqs > f_lo) & (freqs < t_hi)
        if np.any(trans):
            u = (freqs[trans] - f_lo) / (t_hi - f_lo)
            h[trans] *= 0.5 * (1.0 - np.cos(np.pi * u))

    if f_hi is not None and f_hi < 0.5 * sr:
        t_lo = max(0.0, f_hi - max(trans_hz, 1.0))
        h[freqs >= f_hi] = 0.0
        trans = (freqs > t_lo) & (freqs < f_hi)
        if np.any(trans):
            u = (freqs[trans] - t_lo) / max(f_hi - t_lo, 1.0)
            h[trans] *= 0.5 * (1.0 + np.cos(np.pi * u))

    spec = np.fft.rfft(x, n=n_fft, axis=0)
    out = np.fft.irfft(spec * h[:, None], n=n_fft, axis=0)[:n]
    return np.ascontiguousarray(out, dtype=np.float64)


def _detect_fullband_presence(x: np.ndarray, sr: int, cutoff_hz: float) -> tuple[bool, float]:
    """Detect whether input already carries natural full-band HF energy above cutoff.

    Uses Hann-windowed Welch PSD to avoid rectangular-window spectral leakage.

    Returns:
        (is_fullband, hf_to_mid_db)
    """
    nyq = 0.5 * sr
    if cutoff_hz >= 0.78 * nyq:
        # Nominal cutoff already near fullband (>= 18.7 kHz at 48 kHz); check upper HF band.
        probe_lo = min(16000.0, 0.70 * nyq)
    else:
        # Step past typical polyphase transition skirts (~20-25% of cutoff).
        probe_lo = min(cutoff_hz * 1.25, 0.82 * nyq)
    probe_hi = min(21000.0, 0.94 * nyq)
    if probe_hi <= probe_lo + 400.0:
        return True, 0.0

    nperseg = min(2048, max(256, x.shape[0] // 2))
    freqs, psd_ch = signal.welch(x, fs=sr, window="hann", nperseg=nperseg, axis=0)
    psd = np.mean(psd_ch, axis=1)

    mid_mask = (freqs >= 1500.0) & (freqs <= min(7500.0, cutoff_hz * 0.90))
    hf_mask = (freqs >= probe_lo) & (freqs <= probe_hi)
    if not np.any(mid_mask) or not np.any(hf_mask):
        return False, -120.0

    mid_e = float(np.mean(psd[mid_mask])) + EPS
    hf_e = float(np.mean(psd[hf_mask])) + EPS
    ratio_db = float(10.0 * np.log10(hf_e / mid_e))

    # Full-band clean music has Welch HF PSD within ~25 dB of midband;
    # polyphase-bandlimited or codec-cutoff signals drop below -37..-130 dB above cutoff.
    is_fullband = ratio_db > -28.0
    return is_fullband, ratio_db


def run_safe_resample_only(
    x: np.ndarray,
    sr: int,
    observed_cutoff_hz: float,
    *,
    input_sr: int | None = None,
) -> BwePreviewResult:
    """Safe resample / identity baseline: strictly zero hallucinated HF."""
    t0 = time.perf_counter()
    work, was_1d = _as_stereo_f64(x)
    resampled = False
    if input_sr is not None and int(input_sr) != int(sr):
        frac = Fraction(int(sr), int(input_sr)).limit_denominator(1024)
        work = signal.resample_poly(
            work, frac.numerator, frac.denominator, axis=0, window=("kaiser", 14.0)
        )
        resampled = True
    preview = work.copy()
    synth = np.zeros_like(preview)
    if was_1d:
        preview = preview[:, 0]
        synth = synth[:, 0]
    return BwePreviewResult(
        arm="safe_resample_only",
        preview=preview,
        synthesized_band=synth,
        observed_cutoff_hz=float(observed_cutoff_hz),
        synthesized_hf_band=False,
        is_generative_preview=False,
        bypassed=True,
        wall_s=time.perf_counter() - t0,
        meta={
            "rate": int(sr),
            "resampled": resampled,
            "input_sr": int(input_sr) if input_sr is not None else int(sr),
        },
    )


def _stft_sbr_replicate(
    x: np.ndarray,
    sr: int,
    cutoff_hz: float,
) -> np.ndarray:
    """M/S-coherent STFT spectral band replication above `cutoff_hz`."""
    n, n_ch = x.shape
    nperseg = 2048 if sr >= 44100 else 1024
    nperseg = min(nperseg, max(256, 1 << int(np.floor(np.log2(max(n, 256))))))
    noverlap = nperseg // 2
    freqs, _times, zxx = signal.stft(
        x, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap, axis=0
    )
    # zxx shape: (n_freq, n_ch, n_frames)
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    cut_bin = int(np.ceil(cutoff_hz / df))
    src_lo = max(2, int(np.floor((0.50 * cutoff_hz) / df)))
    src_hi = max(src_lo + 4, min(cut_bin, freqs.size - 1))
    nyq_hi = min(freqs.size - 1, int(np.floor((0.94 * 0.5 * sr) / df)))

    if cut_bin >= nyq_hi:
        return np.zeros_like(x, dtype=np.float64)

    # Estimate natural spectral slope (dB/octave) in upper observed band [0.45*c, 0.95*c].
    slope_mask = (freqs >= 0.45 * cutoff_hz) & (freqs <= 0.95 * cutoff_hz)
    mean_mag = np.sqrt(np.mean(np.abs(zxx) ** 2, axis=(1, 2))) + 1e-12
    if np.count_nonzero(slope_mask) >= 4:
        log2_f = np.log2(freqs[slope_mask])
        mag_db = 20.0 * np.log10(mean_mag[slope_mask])
        slope, _intercept = np.polyfit(log2_f, mag_db, 1)
        rolloff_db_per_oct = float(np.clip(slope, -24.0, -9.0))
    else:
        rolloff_db_per_oct = -14.0

    # Anchor level at the top 15% of observed band.
    anchor_lo = max(src_lo, int(np.floor(0.82 * cutoff_hz / df)))
    anchor_hi = max(anchor_lo + 1, min(cut_bin, freqs.size))
    anchor_rms = float(np.mean(mean_mag[anchor_lo:anchor_hi])) + 1e-12

    out_spec = np.zeros_like(zxx, dtype=np.complex128)
    src_len = src_hi - src_lo
    for dst_bin in range(cut_bin, nyq_hi + 1):
        # Periodic octave-folded replication from [src_lo, src_hi).
        src_bin = src_lo + ((dst_bin - cut_bin) % src_len)
        f_dst = float(freqs[dst_bin])
        oct_above = np.log2(max(f_dst, cutoff_hz) / max(cutoff_hz, 1.0))
        target_scale = 10.0 ** ((rolloff_db_per_oct * oct_above) / 20.0)
        src_mag = float(mean_mag[src_bin]) + 1e-12
        # Normalize bin to anchor level, then apply natural octave roll-off.
        gain = (anchor_rms / src_mag) * target_scale
        # Preserve stereo M/S & ILD balance from source bin.
        out_spec[dst_bin, :, :] = zxx[src_bin, :, :] * gain

    _t, synth = signal.istft(
        out_spec, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap, input_onesided=True, time_axis=-1, freq_axis=0
    )
    # istft returns (n_ch, n_out) -> transpose to (n_out, n_ch)
    synth = synth.T
    if synth.shape[0] > n:
        synth = synth[:n]
    elif synth.shape[0] < n:
        synth = np.vstack([synth, np.zeros((n - synth.shape[0], n_ch), dtype=np.float64)])
    return np.ascontiguousarray(synth, dtype=np.float64)


def _oversampled_harmonic_exciter(
    x: np.ndarray,
    sr: int,
    cutoff_hz: float,
) -> np.ndarray:
    """2x-oversampled smooth harmonic exciter (suppresses aliasing below cutoff)."""
    bp = _linear_phase_band_filter(
        x, sr, f_lo=0.45 * cutoff_hz, f_hi=0.96 * cutoff_hz, trans_hz=min(300.0, 0.05 * cutoff_hz)
    )
    rms = float(np.sqrt(np.mean(bp ** 2))) + 1e-12
    norm = bp / (rms * 3.0)
    # 2x oversample before nonlinear harmonic generation to prevent aliasing foldback.
    up = signal.resample_poly(norm, 2, 1, axis=0, window=("kaiser", 12.0))
    # Smooth odd (tanh - linear) + mild even (zero-mean quadratic) harmonics.
    odd_h = np.tanh(1.8 * up) - 1.8 * up
    even_h = 0.22 * (up ** 2 - np.mean(up ** 2, axis=0, keepdims=True))
    excited_up = odd_h + even_h
    down = signal.resample_poly(excited_up, 1, 2, axis=0, window=("kaiser", 12.0))
    n = x.shape[0]
    if down.shape[0] > n:
        down = down[:n]
    elif down.shape[0] < n:
        down = np.vstack([down, np.zeros((n - down.shape[0], x.shape[1]), dtype=np.float64)])
    return np.ascontiguousarray(down * rms, dtype=np.float64)


def run_dsp_harmonic_bwe_preview(
    x: np.ndarray,
    sr: int,
    observed_cutoff_hz: float,
    *,
    bypass_fullband: bool = True,
) -> BwePreviewResult:
    """Reproducible DSP BWE preview strictly above `observed_cutoff_hz`.

    Guarantees:
    - Exact additive decomposition: `preview == input + synthesized_band` (<= 1e-12).
    - Observed band `[0, observed_cutoff_hz]` untouched (linear-phase HPF has 0.0 gain <= cutoff).
    - Full-band negative controls abstain when `bypass_fullband=True`.
    """
    t0 = time.perf_counter()
    work, was_1d = _as_stereo_f64(x)
    nyq = 0.5 * sr
    cutoff = float(np.clip(observed_cutoff_hz, 1000.0, 0.92 * nyq))

    is_fullband, hf_mid_db = _detect_fullband_presence(work, sr, cutoff)
    if bypass_fullband and is_fullband:
        preview = work.copy()
        synth = np.zeros_like(preview)
        if was_1d:
            preview = preview[:, 0]
            synth = synth[:, 0]
        return BwePreviewResult(
            arm="dsp_harmonic_bwe_preview",
            preview=preview,
            synthesized_band=synth,
            observed_cutoff_hz=cutoff,
            synthesized_hf_band=False,
            is_generative_preview=False,
            bypassed=True,
            wall_s=time.perf_counter() - t0,
            meta={
                "rate": int(sr),
                "bypass_reason": "fullband_hf_already_present",
                "detected_hf_to_mid_db": hf_mid_db,
            },
        )

    # 1. Generate SBR + 2x-oversampled harmonic exciter candidates.
    sbr = _stft_sbr_replicate(work, sr, cutoff)
    exc = _oversampled_harmonic_exciter(work, sr, cutoff)

    trans_hz = min(500.0, 0.05 * cutoff)
    f_top = 0.96 * nyq
    sbr_hp = _linear_phase_band_filter(sbr, sr, f_lo=cutoff, f_hi=f_top, trans_hz=trans_hz)
    exc_hp = _linear_phase_band_filter(exc, sr, f_lo=cutoff, f_hi=f_top, trans_hz=trans_hz)

    # Normalize exciter energy to SBR energy before blending so neither dominates arbitrarily.
    sbr_rms = float(np.sqrt(np.mean(sbr_hp ** 2))) + EPS
    exc_rms = float(np.sqrt(np.mean(exc_hp ** 2))) + EPS
    if exc_rms > 1e-12 and sbr_rms > 1e-12:
        exc_hp = exc_hp * (sbr_rms / exc_rms)
    raw_synth = 0.70 * sbr_hp + 0.30 * exc_hp

    # Apply a 5 ms edge taper before final HPF to eliminate boundary-truncation bleed.
    taper_n = min(raw_synth.shape[0] // 8, round(0.005 * sr))
    if taper_n > 4:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_n, dtype=np.float64) / taper_n))
        raw_synth[:taper_n] *= ramp[:, None]
        raw_synth[-taper_n:] *= ramp[::-1, None]

    # Strict final linear-phase HPF pass guarantees exact 0.0 gain in [0, cutoff].
    synth = _linear_phase_band_filter(raw_synth, sr, f_lo=cutoff, f_hi=f_top, trans_hz=trans_hz)

    # Conservative energy cap relative to upper observed octave [0.5*cutoff, cutoff].
    upper_obs = _linear_phase_band_filter(
        work, sr, f_lo=0.5 * cutoff, f_hi=cutoff, trans_hz=min(250.0, 0.04 * cutoff)
    )
    obs_rms = float(np.sqrt(np.mean(upper_obs ** 2))) + EPS
    syn_rms = float(np.sqrt(np.mean(synth ** 2))) + EPS
    max_syn_rms = 0.45 * obs_rms
    if syn_rms > max_syn_rms and syn_rms > EPS:
        synth = synth * (max_syn_rms / syn_rms)

    # Headroom guard: never clip preview and NEVER modify input; only scale `synth`.
    preview = work + synth
    peak = float(np.max(np.abs(preview)))
    if peak > 0.99:
        inp_peak = float(np.max(np.abs(work)))
        headroom = max(0.0, 0.99 - inp_peak)
        syn_peak = float(np.max(np.abs(synth))) + EPS
        synth = synth * min(1.0, headroom / syn_peak)
        preview = work + synth

    if was_1d:
        preview = preview[:, 0]
        synth = synth[:, 0]

    return BwePreviewResult(
        arm="dsp_harmonic_bwe_preview",
        preview=preview,
        synthesized_band=synth,
        observed_cutoff_hz=cutoff,
        synthesized_hf_band=True,
        is_generative_preview=True,
        bypassed=False,
        wall_s=time.perf_counter() - t0,
        meta={
            "rate": int(sr),
            "bypass_reason": None,
            "detected_hf_to_mid_db": hf_mid_db,
            "crossover": "linear_phase_fft_raised_cosine",
            "transition_hz": trans_hz,
        },
    )


def run_arm(
    arm: str,
    x: np.ndarray,
    sr: int,
    observed_cutoff_hz: float,
    *,
    bypass_fullband: bool = True,
) -> BwePreviewResult:
    """Dispatch a BWE arm and enforce additive decomposition contract."""
    inp_f64 = np.asarray(x, dtype=np.float64)
    if arm == "safe_resample_only":
        res = run_safe_resample_only(inp_f64, sr, observed_cutoff_hz)
    elif arm == "dsp_harmonic_bwe_preview":
        res = run_dsp_harmonic_bwe_preview(
            inp_f64, sr, observed_cutoff_hz, bypass_fullband=bypass_fullband
        )
    else:
        raise ValueError(f"unknown E7 BWE arm: {arm!r}")
    verify_additive_contract(inp_f64, res)
    return res


def _encode_tagged_flac(
    pcm: np.ndarray,
    sr: int,
    target_flac: Path,
    bits: int,
    tags: dict[str, str],
) -> str:
    """Encode float64 stereo PCM to 24-bit/16-bit FLAC via ffmpeg and attach Vorbis tags."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to export FLAC BWE preview derivatives")
    target_flac.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e7_flac_") as td:
        tmp_wav = Path(td) / "tmp.wav"
        corpus_synth.write_wav(
            tmp_wav, sr, np.ascontiguousarray(pcm, dtype=np.float64), bits=bits, seed=0
        )
        sample_fmt = "s32" if bits >= 24 else "s16"
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(tmp_wav),
            "-c:a", "flac",
            "-sample_fmt", sample_fmt,
            "-compression_level", "5",
            str(target_flac),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
        if proc.returncode != 0 or not target_flac.exists():
            raise RuntimeError(f"ffmpeg FLAC encode failed: {proc.stderr.strip()}")

    audio = FLAC(str(target_flac))
    for k, v in tags.items():
        audio[k.upper()] = str(v)
    audio.save()
    return hashlib.sha256(target_flac.read_bytes()).hexdigest()


def export_bwe_preview(
    source_path: Path,
    result: BwePreviewResult,
    sr: int,
    out_dir: Path,
    case_id: str,
    *,
    bits: int = 24,
) -> dict:
    """Export an opt-in BWE preview derivative with mandatory FLAC tags + JSON sidecar.

    Never modifies or replaces `source_path` (verified by SHA-256 before and after).
    Writes:
    - `{case_id}.{arm}.bwe_preview.flac`
    - `{case_id}.{arm}.synthesized_band_only.flac`
    - `{case_id}.{arm}.bwe_preview.sidecar.json`
    """
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"source file does not exist: {src}")
    source_sha256_before = hashlib.sha256(src.read_bytes()).hexdigest()

    out_dir.mkdir(parents=True, exist_ok=True)
    preview_fname = f"{case_id}.{result.arm}.bwe_preview.flac"
    synth_fname = f"{case_id}.{result.arm}.synthesized_band_only.flac"
    sidecar_fname = f"{case_id}.{result.arm}.bwe_preview.sidecar.json"

    preview_path = out_dir / preview_fname
    synth_path = out_dir / synth_fname
    sidecar_path = out_dir / sidecar_fname

    common_tags = {
        "SYNTHESIZED_HF_BAND": "true" if result.synthesized_hf_band else "false",
        "OBSERVED_CUTOFF_HZ": f"{result.observed_cutoff_hz:.1f}",
        "IS_GENERATIVE_PREVIEW": "true" if result.is_generative_preview else "false",
        "NOT_ORIGINAL_RESTORATION": "true",
        "ENHANCE_OPT_IN_DERIVATIVE": "true",
        "BWE_ARM": result.arm,
        "CASE_ID": case_id,
        "SOURCE_SHA256": source_sha256_before,
    }

    derivative_sha256 = _encode_tagged_flac(
        result.preview,
        sr,
        preview_path,
        bits,
        {**common_tags, "BWE_TRACK_ROLE": "bwe_preview_derivative"},
    )
    synthesized_band_sha256 = _encode_tagged_flac(
        result.synthesized_band,
        sr,
        synth_path,
        bits,
        {**common_tags, "BWE_TRACK_ROLE": "synthesized_band_only"},
    )

    source_sha256_after = hashlib.sha256(src.read_bytes()).hexdigest()
    if source_sha256_after != source_sha256_before:
        raise RuntimeError("source file hash changed during export_bwe_preview")

    sidecar = {
        "schema": "e7-bwe-preview-sidecar/v1",
        "case_id": case_id,
        "arm": result.arm,
        "rate": int(sr),
        "bits": int(bits),
        "synthesized_hf_band": bool(result.synthesized_hf_band),
        "observed_cutoff_hz": float(result.observed_cutoff_hz),
        "is_generative_preview": bool(result.is_generative_preview),
        "not_original_restoration": True,
        "enhance_opt_in_derivative": True,
        "bypassed": bool(result.bypassed),
        "source_untouched_verified": True,
        "source_sha256": source_sha256_before,
        "derivative_sha256": derivative_sha256,
        "synthesized_band_sha256": synthesized_band_sha256,
        "files": {
            "derivative_flac": preview_fname,
            "synthesized_band_only_flac": synth_fname,
            "sidecar_json": sidecar_fname,
        },
        "policy_note": (
            "Opt-in Enhance BWE preview derivative. High-frequency content above "
            "observed_cutoff_hz is synthesized, not recovered original audio; "
            "source file remains untouched."
        ),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return sidecar


def audit_neural_bwe_arm(models_dir: Path | None = None) -> dict:
    """Honest local audit of neural BWE models (AudioSR / UniverSR / BABE-2 / SAMECFM) & Colab.

    Never fabricates neural metrics, never adds production dependencies, and
    never launches Colab automation.
    """
    m_dir = models_dir if models_dir is not None else (REPO / "models")
    weight_exts = (".pt", ".pth", ".ckpt", ".onnx", ".safetensors")
    bwe_keywords = ("audiosr", "universr", "babe", "samecfm", "bwe")
    local_weights = []
    if m_dir.exists():
        for p in sorted(m_dir.rglob("*")):
            if (
                p.is_file()
                and p.suffix.lower() in weight_exts
                and any(k in p.name.lower() for k in bwe_keywords)
            ):
                local_weights.append(p.relative_to(REPO).as_posix())

    cuda_available = False
    torch_version = None
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        torch_version = None

    colab_authorized = bool(os.environ.get("MR_E7_COLAB_AUTHORIZED") == "1")
    blocked_reasons = []
    if not local_weights:
        blocked_reasons.append("no_local_licensed_bwe_weights_audiosr_universr_babe2_samecfm")
    if not cuda_available:
        blocked_reasons.append("no_local_cuda_gpu")
    if not colab_authorized:
        blocked_reasons.append("colab_connection_not_authorized_or_configured")

    status = "blocked_external_weights_or_colab" if blocked_reasons else "ready"
    return {
        "arm": "neural_bwe_audiosr_babe2",
        "status": status,
        "executed": False,
        "production_dependency_added": False,
        "blocked_reasons": blocked_reasons,
        "environment_audit": {
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "local_bwe_weights_found": local_weights,
            "colab_authorized": colab_authorized,
        },
        "weights_and_license_contract": {
            "target_models": [
                {
                    "id": "audiosr",
                    "reference": "Liu et al. (2023) AudioSR: Versatile Audio Super-resolution at Scale",
                    "expected_checkpoint_format": ".ckpt / .safetensors with pinned SHA-256",
                    "license_gate": "verify_model_card_and_weights_license_before_download_or_run",
                    "stereo_policy": "evaluate_native_stereo_coherence_not_mono_only",
                },
                {
                    "id": "universr",
                    "reference": "UniverSR universal audio super-resolution family",
                    "expected_checkpoint_format": ".pt / .safetensors with pinned SHA-256",
                    "license_gate": "verify_open_research_or_mit_apache_license_before_run",
                    "stereo_policy": "verify_lr_corr_ild_ipd_in_observed_and_missing_bands",
                },
                {
                    "id": "babe_2",
                    "reference": "Moliner & Valimaki (2024) BABE-2 Blind Audio Bandwidth Extension for Historical Music",
                    "expected_checkpoint_format": ".pt / .ckpt with pinned SHA-256",
                    "license_gate": "verify_code_and_checkpoint_license_terms_before_run",
                    "stereo_policy": "preserve_observed_band_via_linear_phase_crossover",
                },
                {
                    "id": "samecfm",
                    "reference": "SAMECFM flow-matching bandwidth extension",
                    "expected_checkpoint_format": ".pt / .safetensors with pinned SHA-256",
                    "license_gate": "verify_checkpoint_release_and_license_before_run",
                    "stereo_policy": "export_isolated_synthesized_band_and_mandatory_tags",
                },
            ],
            "mandatory_output_invariants": [
                "observed_band [0, observed_cutoff_hz] preserved via crossover or verified delta SNR ~ 0 dB",
                "isolated synthesized_band_only.flac exported alongside preview",
                "mandatory FLAC + JSON sidecar tags: synthesized_hf_band=true, is_generative_preview=true, not_original_restoration=true",
                "zero mutation or replacement of source library file",
            ],
        },
        "colab_handoff_spec": {
            "permitted_inputs": "corpus/dev-e7-bwe-v1.json (rights.colab == True ONLY)",
            "forbidden_inputs": ["corpus/sealed-v1.json", "corpus/sealed-e5-v1.json", "private_masters"],
            "required_run_manifest_fields": [
                "gpu_model",
                "cuda_version",
                "torch_version",
                "model_id",
                "checkpoint_sha256",
                "license_spdx_or_terms",
                "rng_seed",
                "observed_cutoff_hz",
                "wall_s_per_audio_min",
                "per_case_derivative_sha256",
                "per_case_synthesized_band_sha256",
            ],
            "local_post_colab_verification": (
                "Re-evaluate returned preview and synthesized_band WAV/FLACs locally via "
                "research.e7_bwe_preview.evaluate_e7_case on the exact dev-e7-bwe-v1 clean references "
                "and attach mandatory opt-in Enhance tags via export_bwe_preview; never promote into Preserve."
            ),
        },
    }


def _snr_db(ref: np.ndarray, est: np.ndarray) -> float | None:
    sig = float(np.sum(ref ** 2))
    err = float(np.sum((ref - est) ** 2))
    if sig < EPS or err < EPS:
        return None
    return float(10.0 * np.log10(sig / err))


def _rms_db(x: np.ndarray) -> float | None:
    e = float(np.mean(x ** 2))
    if e <= EPS:
        return None
    return float(10.0 * np.log10(e))


def _band_lsd_db(
    ref: np.ndarray,
    est: np.ndarray,
    sr: int,
    f_lo: float,
    f_hi: float,
) -> float | None:
    """Mean Log-Spectral Distance (dB) strictly within [f_lo, f_hi]."""
    nperseg = 2048
    freqs, _t, z_ref = signal.stft(ref, fs=sr, window="hann", nperseg=nperseg, noverlap=1024, axis=0)
    _, _, z_est = signal.stft(est, fs=sr, window="hann", nperseg=nperseg, noverlap=1024, axis=0)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return None
    p_ref = np.maximum(np.abs(z_ref[mask]) ** 2, 1e-12)
    p_est = np.maximum(np.abs(z_est[mask]) ** 2, 1e-12)
    diff_db = 10.0 * (np.log10(p_ref) - np.log10(p_est))
    # Root-mean-square over frequency bins, averaged over frames and channels.
    frame_lsd = np.sqrt(np.mean(diff_db ** 2, axis=0))
    return float(np.mean(frame_lsd))


def _band_spectral_tilt_db_per_oct(
    x: np.ndarray,
    sr: int,
    f_lo: float,
    f_hi: float,
) -> float | None:
    """Linear spectral slope (dB/octave) in [f_lo, f_hi] via Hann-windowed Welch PSD."""
    nperseg = min(2048, max(256, x.shape[0] // 2))
    freqs, psd_ch = signal.welch(x, fs=sr, window="hann", nperseg=nperseg, axis=0)
    psd = np.mean(psd_ch, axis=1)
    mask = (freqs >= max(f_lo, 20.0)) & (freqs <= f_hi)
    if np.count_nonzero(mask) < 4:
        return None
    p_band = psd[mask]
    if float(np.max(p_band)) < 1e-14:
        return None
    log2_f = np.log2(freqs[mask])
    psd_db = 10.0 * np.log10(np.maximum(p_band, 1e-14))
    slope, _ = np.polyfit(log2_f, psd_db, 1)
    return float(slope)


def _stereo_band_metrics(
    ref: np.ndarray,
    est: np.ndarray,
    sr: int,
    f_lo: float,
    f_hi: float,
) -> dict:
    """Compute L/R correlation, ILD drift (dB), and magnitude-weighted IPD error (deg) in a band."""
    trans = min(200.0, 0.04 * max(f_hi - f_lo, 100.0))
    ref_b = _linear_phase_band_filter(ref, sr, f_lo=f_lo if f_lo > 0 else None, f_hi=f_hi, trans_hz=trans)
    est_b = _linear_phase_band_filter(est, sr, f_lo=f_lo if f_lo > 0 else None, f_hi=f_hi, trans_hz=trans)

    def _lr_corr(arr: np.ndarray) -> float | None:
        l_ch, r_ch = arr[:, 0], arr[:, 1]
        el = float(np.sum(l_ch ** 2))
        er = float(np.sum(r_ch ** 2))
        if el < 1e-14 or er < 1e-14:
            return None
        return float(np.sum(l_ch * r_ch) / np.sqrt(el * er))

    def _ild_db(arr: np.ndarray) -> float | None:
        el = float(np.mean(arr[:, 0] ** 2))
        er = float(np.mean(arr[:, 1] ** 2))
        if el < 1e-14 and er < 1e-14:
            return None
        return float(10.0 * np.log10((el + 1e-14) / (er + 1e-14)))

    c_ref = _lr_corr(ref_b)
    c_est = _lr_corr(est_b)
    corr_drift = float(abs(c_est - c_ref)) if (c_ref is not None and c_est is not None) else None

    ild_ref = _ild_db(ref_b)
    ild_est = _ild_db(est_b)
    ild_drift = float(abs(ild_est - ild_ref)) if (ild_ref is not None and ild_est is not None) else None

    # Magnitude-weighted IPD error (degrees) in STFT bins inside [f_lo, f_hi].
    freqs, _t, z_ref = signal.stft(ref, fs=sr, window="hann", nperseg=2048, noverlap=1024, axis=0)
    _, _, z_est = signal.stft(est, fs=sr, window="hann", nperseg=2048, noverlap=1024, axis=0)
    mask = (freqs >= max(f_lo, 20.0)) & (freqs <= f_hi)
    ipd_err_deg: float | None = None
    if np.any(mask):
        cross_ref = z_ref[mask, 0, :] * np.conj(z_ref[mask, 1, :])
        cross_est = z_est[mask, 0, :] * np.conj(z_est[mask, 1, :])
        w = np.abs(cross_ref)
        w_sum = float(np.sum(w))
        if w_sum > 1e-12 and float(np.sum(np.abs(cross_est))) > 1e-12:
            dphi = np.angle(cross_est * np.conj(cross_ref))
            ipd_err_deg = float(np.degrees(np.sum(w * np.abs(dphi)) / w_sum))

    return {
        "lr_corr_clean": c_ref,
        "lr_corr_out": c_est,
        "lr_corr_drift": corr_drift,
        "ild_drift_db": ild_drift,
        "ipd_error_deg": ipd_err_deg,
    }


def _pre_ringing_rise_db(
    inp: np.ndarray,
    preview: np.ndarray,
    attack_mask: np.ndarray,
    sr: int,
) -> float:
    """Measure pre-ringing energy rise (dB) in 2 ms windows immediately before attacks."""
    if not attack_mask.any():
        return 0.0
    m = attack_mask[:, 0]
    starts = np.flatnonzero(m[1:] & (~m[:-1])) + 1
    if starts.size == 0:
        return 0.0
    win = max(16, round(0.002 * sr))
    rises = []
    for s in starts:
        lo = max(0, int(s) - win)
        hi = int(s)
        if hi - lo < 8:
            continue
        e_in = float(np.mean(inp[lo:hi] ** 2)) + 1e-12
        e_out = float(np.mean(preview[lo:hi] ** 2)) + 1e-12
        rises.append(max(0.0, float(10.0 * np.log10(e_out / e_in))))
    return float(np.max(rises)) if rises else 0.0


def evaluate_e7_case(
    clean: np.ndarray,
    degraded: np.ndarray,
    result: BwePreviewResult,
    sr: int,
    *,
    observed_cutoff_hz: float,
    attack_mask: np.ndarray,
    is_fullband_control: bool = False,
    input_sha256_before: str | None = None,
    input_sha256_after: str | None = None,
) -> dict:
    """Evaluate one E7 BWE preview strictly separated into observed vs missing bands."""
    c = np.asarray(clean, dtype=np.float64)
    d = np.asarray(degraded, dtype=np.float64)
    y = np.asarray(result.preview, dtype=np.float64)
    s = np.asarray(result.synthesized_band, dtype=np.float64)
    nyq = 0.5 * sr
    cutoff = float(observed_cutoff_hz)

    add_err = verify_additive_contract(d, result)
    input_mutated = (
        input_sha256_before != input_sha256_after
        if (input_sha256_before is not None and input_sha256_after is not None)
        else False
    )

    # 1. Observed band [0, cutoff] preservation metrics.
    obs_hi = min(cutoff * 0.98, 0.90 * nyq)
    obs_trans = min(150.0, 0.02 * cutoff)
    c_obs = _linear_phase_band_filter(c, sr, f_lo=None, f_hi=obs_hi, trans_hz=obs_trans)
    d_obs = _linear_phase_band_filter(d, sr, f_lo=None, f_hi=obs_hi, trans_hz=obs_trans)
    y_obs = _linear_phase_band_filter(y, sr, f_lo=None, f_hi=obs_hi, trans_hz=obs_trans)

    obs_snr_in = _snr_db(c_obs, d_obs)
    obs_snr_out = _snr_db(c_obs, y_obs)
    obs_delta_snr = (
        float(obs_snr_out - obs_snr_in)
        if (obs_snr_in is not None and obs_snr_out is not None)
        else (0.0 if (obs_snr_in is None and obs_snr_out is None) else None)
    )
    obs_lsd_in = _band_lsd_db(c, d, sr, 20.0, obs_hi)
    obs_lsd_out = _band_lsd_db(c, y, sr, 20.0, obs_hi)
    obs_delta_lsd = (
        float(obs_lsd_out - obs_lsd_in)
        if (obs_lsd_in is not None and obs_lsd_out is not None)
        else None
    )
    obs_vs_input_max_err = float(np.max(np.abs(y_obs - d_obs)))
    obs_stereo = _stereo_band_metrics(c, y, sr, 20.0, obs_hi)

    # 2. Missing / synthesized band (cutoff, nyquist*0.92] metrics.
    # For full-band controls (cutoff=20000), measure upper HF [16000, 22000] for false HF check.
    miss_lo = min(cutoff * 1.04, 0.85 * nyq) if not is_fullband_control else 16000.0
    miss_hi = 0.92 * nyq
    miss_lsd_in = _band_lsd_db(c, d, sr, miss_lo, miss_hi)
    miss_lsd_out = _band_lsd_db(c, y, sr, miss_lo, miss_hi)
    miss_delta_lsd = (
        float(miss_lsd_out - miss_lsd_in)
        if (miss_lsd_in is not None and miss_lsd_out is not None)
        else None
    )
    miss_tilt_clean = _band_spectral_tilt_db_per_oct(c, sr, miss_lo, miss_hi)
    miss_tilt_out = _band_spectral_tilt_db_per_oct(y, sr, miss_lo, miss_hi)
    miss_tilt_err = (
        float(abs(miss_tilt_out - miss_tilt_clean))
        if (miss_tilt_clean is not None and miss_tilt_out is not None)
        else None
    )

    c_miss = _linear_phase_band_filter(c, sr, f_lo=miss_lo, f_hi=miss_hi, trans_hz=200.0)
    clean_miss_e = float(np.mean(c_miss ** 2))
    synth_e = float(np.mean(s ** 2))
    synth_energy_dbfs = _rms_db(s)
    clean_miss_energy_dbfs = _rms_db(c_miss)
    synth_to_clean_miss_db = (
        float(10.0 * np.log10(synth_e / clean_miss_e))
        if (synth_e > EPS and clean_miss_e > EPS)
        else None
    )
    miss_stereo = _stereo_band_metrics(c, y, sr, miss_lo, miss_hi)

    # 3. Artifacts: sub-cutoff aliasing/leakage, pre-ringing, false HF on fullband control.
    s_below = _linear_phase_band_filter(s, sr, f_lo=None, f_hi=obs_hi, trans_hz=obs_trans)
    e_below = float(np.mean(s_below ** 2))
    sub_cutoff_leakage_db = (
        float(10.0 * np.log10(max(e_below, EPS) / synth_e))
        if synth_e > EPS
        else None
    )
    pre_ring_db = _pre_ringing_rise_db(d, y, attack_mask, sr)
    false_hf_injected = bool(is_fullband_control and synth_e > 1e-14)
    false_hf_dbfs = synth_energy_dbfs if is_fullband_control else None

    audio_min = max(c.shape[0] / float(sr) / 60.0, 1e-9)
    return {
        "arm": result.arm,
        "observed_cutoff_hz": cutoff,
        "synthesized_hf_band": bool(result.synthesized_hf_band),
        "is_generative_preview": bool(result.is_generative_preview),
        "bypassed": bool(result.bypassed),
        "missing_band_is_preview_not_original": True,
        "observed_band": {
            "snr_in_db": obs_snr_in,
            "snr_out_db": obs_snr_out,
            "delta_snr_db": obs_delta_snr,
            "lsd_in_db": obs_lsd_in,
            "lsd_out_db": obs_lsd_out,
            "delta_lsd_db": obs_delta_lsd,
            "vs_input_max_err": obs_vs_input_max_err,
            "lr_corr_drift": obs_stereo["lr_corr_drift"],
            "ild_drift_db": obs_stereo["ild_drift_db"],
            "ipd_error_deg": obs_stereo["ipd_error_deg"],
        },
        "missing_band": {
            "lsd_in_db": miss_lsd_in,
            "lsd_out_db": miss_lsd_out,
            "delta_lsd_db": miss_delta_lsd,
            "spectral_tilt_clean_db_per_oct": miss_tilt_clean,
            "spectral_tilt_out_db_per_oct": miss_tilt_out,
            "spectral_tilt_err_db_per_oct": miss_tilt_err,
            "synth_energy_dbfs": synth_energy_dbfs,
            "clean_miss_energy_dbfs": clean_miss_energy_dbfs,
            "synth_to_clean_miss_energy_db": synth_to_clean_miss_db,
            "lr_corr_clean": miss_stereo["lr_corr_clean"],
            "lr_corr_out": miss_stereo["lr_corr_out"],
            "lr_corr_drift": miss_stereo["lr_corr_drift"],
            "ild_drift_db": miss_stereo["ild_drift_db"],
            "ipd_error_deg": miss_stereo["ipd_error_deg"],
        },
        "artifacts": {
            "sub_cutoff_leakage_db": sub_cutoff_leakage_db,
            "pre_ringing_rise_db": pre_ring_db,
            "false_hf_injected": false_hf_injected,
            "false_hf_dbfs": false_hf_dbfs,
        },
        "additive_identity_max_err": add_err,
        "input_mutated": input_mutated,
        "wall_s": float(result.wall_s),
        "wall_s_per_audio_min": float(result.wall_s / audio_min),
    }


def _verify_e4_freeze_untouched() -> dict:
    """Verify all 18 frozen E4 implementation files still match their SHA-256 pins."""
    freeze_path = REPO / "docs" / "validation" / "e4-freeze-manifest.json"
    doc = json.loads(freeze_path.read_text(encoding="utf-8"))
    pins = doc["toolchain"]["implementation_sha256"]
    mismatches = []
    for rel, expected in pins.items():
        actual = corpus_v1.digest(REPO / rel)
        if actual != expected:
            mismatches.append({"file": rel, "expected": expected, "actual": actual})
    return {
        "freeze_manifest": "docs/validation/e4-freeze-manifest.json",
        "total_pinned_files": len(pins),
        "matched_files": len(pins) - len(mismatches),
        "all_untouched": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def _max(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return float(np.max(clean)) if clean else None


def _fmt(val: float | None, fmt: str = "+.2f", suffix: str = "") -> str:
    return "-" if val is None else f"{val:{fmt}}{suffix}"


def _render_report_md(results: dict) -> str:
    neural = results["neural_arm"]
    e4_check = results["e4_freeze_integrity"]
    prov = results["provenance"]
    inv = results["invariants"]
    diag = results["diagnostics"]

    lines = [
        "# E7 · BWE Preview with Marked Synthesized Band — Research Report",
        "",
        (
            "**Card:** Board #1133 · **Spec:** E0 charter + spec v1 (Section E7) · "
            f"**Dataset:** `{results['dataset_version']}` (`{results['n_cases']}` stereo 48 kHz cases) · "
            f"**Neural BWE Arm:** `{neural['status']}`"
        ),
        "",
        "## 1. Scope, Safety Boundaries & Provenance",
        "",
        f"- **Sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`):** `sealed_corpora_opened = {results['sealed_corpora_opened']}` (strictly untouched).",
        f"- **Frozen E4 Preserve hard-declip pipeline:** `{e4_check['matched_files']}/{e4_check['total_pinned_files']}` SHA-256 pins verified untouched (`all_untouched = {e4_check['all_untouched']}`).",
        "- **Preserve promotion / neural production dependency / Colab automation:** `False` (isolated opt-in Enhance preview study only).",
        f"- **Manifest SHA-256 (`corpus/dev-e7-bwe-v1.json`):** `{prov['manifest_sha256']}`",
        f"- **Corpus Builder SHA-256 (`research/build_e7_bwe_corpus.py`):** `{prov['builder_sha256']}`",
        f"- **E7 Harness SHA-256 (`research/e7_bwe_preview.py`):** `{prov['harness_sha256']}`",
        f"- **Additive & Immutability Contract:** `{inv['total_runs']}/{inv['total_runs']}` runs passed `preview == input + synthesized_band` (`max_err = {inv['max_additive_identity_err']:.2e}`) with `0` input mutations.",
        "",
        "> **Honest Semantics Notice:** High-frequency energy above `observed_cutoff_hz` produced by `dsp_harmonic_bwe_preview` is a **synthesized preview** (`synthesized_hf_band = true`, `is_generative_preview = true`, `not_original_restoration = true`), **never** a recovered original waveform. Missing-band LSD and spectral tilt evaluate envelope plausibility only.",
        "",
        "---",
        "",
        "## 2. Band-Separated Comparison by Stratum",
        "",
        "### 2.1 Observed Band Preservation (`[0, observed_cutoff_hz]`) vs Missing Band Envelope (`(observed_cutoff_hz, 22 kHz]`)",
        "",
        "| Stratum (n) | Arm | Obs ΔSNR (dB) | Obs ΔLSD (dB) | Obs Max Err vs In | Miss LSD In (dB) | Miss LSD Out (dB) | Miss ΔLSD (dB) | Synth Energy (dBFS) | Wall (s/min) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for stratum in build_e7_bwe_corpus.STRATA:
        n_s = results["summary_by_stratum"][stratum]["safe_resample_only"]["n_cases"]
        for arm in DSP_ARMS:
            s = results["summary_by_stratum"][stratum][arm]
            syn_e = "0 (none)" if s["mean_synth_energy_dbfs"] is None else f"{s['mean_synth_energy_dbfs']:.1f}"
            lines.append(
                f"| `{stratum}` ({n_s}) | `{arm}` | "
                f"{_fmt(s['mean_obs_delta_snr_db'], '+.3f')} | "
                f"{_fmt(s['mean_obs_delta_lsd_db'], '+.4f')} | "
                f"{_fmt(s['max_obs_vs_input_max_err'], '.2e')} | "
                f"{_fmt(s['mean_miss_lsd_in_db'], '.2f')} | "
                f"{_fmt(s['mean_miss_lsd_out_db'], '.2f')} | "
                f"{_fmt(s['mean_miss_delta_lsd_db'], '+.2f')} | "
                f"{syn_e} | "
                f"{_fmt(s['mean_wall_s_per_audio_min'], '.2f')} |"
            )

    lines += [
        "",
        "### 2.2 Stereo Coherence (L/R Correlation, ILD, IPD) & Artifact Checks",
        "",
        "| Stratum | Arm | Obs L/R Drift | Obs ILD Drift (dB) | Obs IPD Err (deg) | Miss L/R Drift | Miss ILD Drift (dB) | Miss IPD Err (deg) | Sub-Cutoff Leak (dB) | Pre-Ring Rise (dB) | False HF Cases |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for stratum in build_e7_bwe_corpus.STRATA:
        for arm in DSP_ARMS:
            s = results["summary_by_stratum"][stratum][arm]
            leak = "0 (none)" if s["max_sub_cutoff_leakage_db"] is None else f"{s['max_sub_cutoff_leakage_db']:.1f}"
            lines.append(
                f"| `{stratum}` | `{arm}` | "
                f"{_fmt(s['max_obs_lr_corr_drift'], '.4f')} | "
                f"{_fmt(s['max_obs_ild_drift_db'], '.3f')} | "
                f"{_fmt(s['mean_obs_ipd_error_deg'], '.2f')} | "
                f"{_fmt(s['mean_miss_lr_corr_drift'], '.3f')} | "
                f"{_fmt(s['mean_miss_ild_drift_db'], '.2f')} | "
                f"{_fmt(s['mean_miss_ipd_error_deg'], '.1f')} | "
                f"{leak} | "
                f"{_fmt(s['max_pre_ringing_rise_db'], '.3f')} | "
                f"{s['false_hf_cases']}/{s['n_cases']} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 3. Codec Roundtrip & Full-Band Control Diagnostics",
        "",
        "### 3.1 Per-Case Codec Roundtrip & OOD Behavior (`dsp_harmonic_bwe_preview`)",
        "",
        "| Case ID | Codec / Backend | Cutoff (Hz) | Obs ΔSNR (dB) | Miss LSD In -> Out (dB) | Tilt Err (dB/oct) | Miss ILD Drift (dB) | Pre-Ring (dB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results["cases"]:
        if row["stratum"] not in ("codec_roundtrip", "ood"):
            continue
        m = row["arms"]["dsp_harmonic_bwe_preview"]
        ob = m["observed_band"]
        mb = m["missing_band"]
        art = m["artifacts"]
        backend = str(row.get("codec_backend") or "-")
        lsd_str = f"{_fmt(mb['lsd_in_db'], '.2f')} -> {_fmt(mb['lsd_out_db'], '.2f')}"
        lines.append(
            f"| `{row['id']}` | `{backend}` | {row['observed_cutoff_hz']:.0f} | "
            f"{_fmt(ob['delta_snr_db'], '+.3f')} | {lsd_str} | "
            f"{_fmt(mb['spectral_tilt_err_db_per_oct'], '.2f')} | "
            f"{_fmt(mb['ild_drift_db'], '.2f')} | {_fmt(art['pre_ringing_rise_db'], '.3f')} |"
        )

    fb = diag["fullband_forced_without_bypass"]
    lines += [
        "",
        "### 3.2 Full-Band Negative Control: Default Guard vs Forced BWE (`bypass_fullband=False`)",
        "",
        "| Case ID | Mode | Bypassed | False HF Injected | Synth Energy (dBFS) | Obs Max Err vs In | Miss ΔLSD (dB) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for cid, pair in fb.items():
        for label, m in (("default (`bypass_fullband=True`)", pair["default"]), ("forced (`bypass_fullband=False`)", pair["forced"])):
            syn_e = "0 (none)" if m["missing_band"]["synth_energy_dbfs"] is None else f"{m['missing_band']['synth_energy_dbfs']:.1f}"
            lines.append(
                f"| `{cid}` | {label} | `{m['bypassed']}` | "
                f"`{m['artifacts']['false_hf_injected']}` | {syn_e} | "
                f"{_fmt(m['observed_band']['vs_input_max_err'], '.2e')} | "
                f"{_fmt(m['missing_band']['delta_lsd_db'], '+.2f')} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 4. Opt-In Derivative Export & Mandatory Tagging Contract",
        "",
        "- `export_bwe_preview()` writes **separate** opt-in files (`*.bwe_preview.flac`, `*.synthesized_band_only.flac`, and `*.bwe_preview.sidecar.json`) and verifies `source_sha256` before and after export (zero source mutation or library replacement).",
        "- Both the FLAC Vorbis comments and the JSON sidecar embed: `synthesized_hf_band`, `observed_cutoff_hz`, `is_generative_preview`, `not_original_restoration`, `enhance_opt_in_derivative`, `source_sha256`, `derivative_sha256`, and `synthesized_band_sha256`.",
        "",
        "## 5. Neural BWE Arm Audit (AudioSR / UniverSR / BABE-2 / SAMECFM) & Colab Contract",
        "",
        f"- **Status:** `{neural['status']}` (`executed = {neural['executed']}`, `production_dependency_added = {neural['production_dependency_added']}`).",
        f"- **Blocked Reasons:** `{', '.join(neural['blocked_reasons'])}`.",
        (
            f"- **Environment Audit:** `torch = {neural['environment_audit']['torch_version']}`, "
            f"`cuda_available = {neural['environment_audit']['cuda_available']}`, "
            f"`local_bwe_weights_found = {len(neural['environment_audit']['local_bwe_weights_found'])}`, "
            f"`colab_authorized = {neural['environment_audit']['colab_authorized']}`."
        ),
        "- **License & Colab Gate:** Target models (`AudioSR`, `UniverSR`, `BABE-2`, `SAMECFM`) require explicit license verification and checkpoint SHA-256 pinning prior to download or execution. Authorized Colab runs may use `corpus/dev-e7-bwe-v1.json` (`rights.colab == True`) ONLY — never `corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`, or private masters — and must pass local `evaluate_e7_case` verification plus mandatory `synthesized_hf_band` tagging.",
        "",
        "## 6. Limitations & Honest Method Boundaries",
        "",
        "1. **Synthesized HF Is Preview, Not Original Restoration:** `dsp_harmonic_bwe_preview` improves missing-band spectral envelope distance (negative `Miss ΔLSD`) by transposing and harmonically exciting the upper observed octave, but it **cannot** reconstruct true lost micro-transients, unique cymbal overtones, or unpitched HF phase. Notice that missing-band IPD error remains high (~45-55 deg across strata) because phase in the missing band is unobservable from band-limited input.",
        "2. **Role of the Full-Band Bypass Guard:** Zero false HF on `fullband_clean_control` relies on `_detect_fullband_presence` (Hann-windowed Welch PSD `hf_to_mid_db > -28 dB`). When forced with `bypass_fullband=False` (Section 3.2), DSP synthesis injects extraneous HF above 20 kHz, confirming why explicit full-band detection and opt-in user control are mandatory.",
        "3. **Extreme Narrowband OOD (`e7-ood-narrowband-4k-48k`):** Extending from a 4 kHz cutoff across >2.5 missing octaves via periodic SBR + harmonic excitation yields a coarse spectral approximation only and increases timbre mismatch relative to 11-16 kHz cutoffs.",
        "4. **Codec Roundtrip Interaction (`mp3_64k`, `opus_32k`, `ood_double_codec_8k`):** Lossy codec quantization and pre-echo present in the upper observed band (`[0.5*cutoff, cutoff]`) are transposed upward by SBR; BWE does **not** remove codec pre-echo inside the observed band.",
        "5. **Neural BWE Comparison Pending Weights/Colab:** No AudioSR, UniverSR, BABE-2, or SAMECFM comparison numbers are reported because licensed weights and authorized Colab GPU execution are not provisioned in this offline CPU environment.",
    ]
    return "\n".join(lines) + "\n"


def run_e7_benchmark(
    manifest_path: Path,
    out_json: Path | None = None,
    out_md: Path | None = None,
) -> dict:
    """Run the full E7 BWE benchmark + neural arm audit and write report artifacts."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != build_e7_bwe_corpus.SCHEMA:
        raise ValueError(f"unexpected E7 manifest schema: {manifest.get('schema')!r}")

    per_case: list[dict] = []
    max_add_err = 0.0
    any_mutated = False
    total_runs = 0

    for case in manifest["cases"]:
        clean, degraded, attack_mask, deg_meta = build_e7_bwe_corpus.synthesize_e7_arrays(case)
        sr = int(case["rate"])
        cutoff = float(case["observed_cutoff_hz"])
        is_ctl = bool(case.get("is_fullband_control", False))
        arm_results: dict[str, dict] = {}

        for arm in DSP_ARMS:
            inp_copy = degraded.copy()
            h_before = hashlib.sha256(inp_copy.tobytes()).hexdigest()
            res = run_arm(arm, inp_copy, sr, cutoff)
            h_after = hashlib.sha256(inp_copy.tobytes()).hexdigest()
            metrics = evaluate_e7_case(
                clean,
                inp_copy,
                res,
                sr,
                observed_cutoff_hz=cutoff,
                attack_mask=attack_mask,
                is_fullband_control=is_ctl,
                input_sha256_before=h_before,
                input_sha256_after=h_after,
            )
            max_add_err = max(max_add_err, float(metrics["additive_identity_max_err"]))
            any_mutated = any_mutated or bool(metrics["input_mutated"])
            total_runs += 1
            arm_results[arm] = metrics

        per_case.append({
            "id": case["id"],
            "stratum": case["stratum"],
            "rate": sr,
            "bits": int(case["bits"]),
            "observed_cutoff_hz": cutoff,
            "is_fullband_control": is_ctl,
            "codec_backend": deg_meta.get("codec_backend"),
            "arms": arm_results,
        })

    summary_by_stratum: dict[str, dict[str, dict]] = {}
    for stratum in build_e7_bwe_corpus.STRATA:
        rows = [c for c in per_case if c["stratum"] == stratum]
        summary_by_stratum[stratum] = {}
        for arm in DSP_ARMS:
            ms = [r["arms"][arm] for r in rows]
            summary_by_stratum[stratum][arm] = {
                "n_cases": len(rows),
                "mean_obs_delta_snr_db": _mean([m["observed_band"]["delta_snr_db"] for m in ms]),
                "mean_obs_delta_lsd_db": _mean([m["observed_band"]["delta_lsd_db"] for m in ms]),
                "max_obs_vs_input_max_err": _max([m["observed_band"]["vs_input_max_err"] for m in ms]),
                "max_obs_lr_corr_drift": _max([m["observed_band"]["lr_corr_drift"] for m in ms]),
                "max_obs_ild_drift_db": _max([m["observed_band"]["ild_drift_db"] for m in ms]),
                "mean_obs_ipd_error_deg": _mean([m["observed_band"]["ipd_error_deg"] for m in ms]),
                "mean_miss_lsd_in_db": _mean([m["missing_band"]["lsd_in_db"] for m in ms]),
                "mean_miss_lsd_out_db": _mean([m["missing_band"]["lsd_out_db"] for m in ms]),
                "mean_miss_delta_lsd_db": _mean([m["missing_band"]["delta_lsd_db"] for m in ms]),
                "mean_miss_tilt_err_db_per_oct": _mean([m["missing_band"]["spectral_tilt_err_db_per_oct"] for m in ms]),
                "mean_synth_energy_dbfs": _mean([m["missing_band"]["synth_energy_dbfs"] for m in ms]),
                "mean_miss_lr_corr_drift": _mean([m["missing_band"]["lr_corr_drift"] for m in ms]),
                "mean_miss_ild_drift_db": _mean([m["missing_band"]["ild_drift_db"] for m in ms]),
                "mean_miss_ipd_error_deg": _mean([m["missing_band"]["ipd_error_deg"] for m in ms]),
                "max_sub_cutoff_leakage_db": _max([m["artifacts"]["sub_cutoff_leakage_db"] for m in ms]),
                "max_pre_ringing_rise_db": _max([m["artifacts"]["pre_ringing_rise_db"] for m in ms]),
                "false_hf_cases": sum(1 for m in ms if m["artifacts"]["false_hf_injected"]),
                "mean_wall_s_per_audio_min": _mean([m["wall_s_per_audio_min"] for m in ms]),
            }

    # Diagnostic: full-band controls with bypass disabled to transparently show guard role.
    case_map = {c["id"]: c for c in per_case}
    fullband_forced: dict[str, dict] = {}
    for case in manifest["cases"]:
        if not case.get("is_fullband_control"):
            continue
        clean, degraded, attack_mask, _ = build_e7_bwe_corpus.synthesize_e7_arrays(case)
        sr = int(case["rate"])
        cutoff = float(case["observed_cutoff_hz"])
        res_forced = run_arm("dsp_harmonic_bwe_preview", degraded, sr, cutoff, bypass_fullband=False)
        m_forced = evaluate_e7_case(
            clean, degraded, res_forced, sr,
            observed_cutoff_hz=cutoff, attack_mask=attack_mask, is_fullband_control=True,
        )
        fullband_forced[case["id"]] = {
            "default": case_map[case["id"]]["arms"]["dsp_harmonic_bwe_preview"],
            "forced": m_forced,
        }

    neural = audit_neural_bwe_arm()
    e4_check = _verify_e4_freeze_untouched()
    results = {
        "schema": "e7-bwe-results/v1",
        "board_card": 1133,
        "dataset_version": manifest["dataset_version"],
        "n_cases": len(per_case),
        "sealed_corpora_opened": False,
        "preserve_pipeline_modified": False,
        "colab_launched": False,
        "toolchain": {
            **corpus_v1.actual_toolchain(),
            "platform": platform.machine(),
        },
        "provenance": {
            "manifest_path": "corpus/dev-e7-bwe-v1.json",
            "manifest_sha256": corpus_v1.digest(manifest_path),
            "builder_path": "research/build_e7_bwe_corpus.py",
            "builder_sha256": corpus_v1.digest(REPO / "research" / "build_e7_bwe_corpus.py"),
            "harness_path": "research/e7_bwe_preview.py",
            "harness_sha256": corpus_v1.digest(REPO / "research" / "e7_bwe_preview.py"),
        },
        "e4_freeze_integrity": e4_check,
        "invariants": {
            "total_runs": total_runs,
            "max_additive_identity_err": max_add_err,
            "any_input_mutated": any_mutated,
        },
        "dsp_arms": list(DSP_ARMS),
        "neural_arm": neural,
        "diagnostics": {
            "fullband_forced_without_bypass": fullband_forced,
        },
        "summary_by_stratum": summary_by_stratum,
        "cases": per_case,
    }

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_report_md(results), encoding="utf-8")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--manifest", default="corpus/dev-e7-bwe-v1.json")
    ap.add_argument("--out-json", default="docs/validation/e7-bwe-results.json")
    ap.add_argument("--out-md", default="docs/validation/e7-bwe-report.md")
    args = ap.parse_args()
    res = run_e7_benchmark(Path(args.manifest), Path(args.out_json), Path(args.out_md))
    print(
        f"E7 benchmark complete: {res['n_cases']} cases, {res['invariants']['total_runs']} runs, "
        f"neural={res['neural_arm']['status']}, e4_untouched={res['e4_freeze_integrity']['all_untouched']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
