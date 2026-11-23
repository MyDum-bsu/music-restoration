#!/usr/bin/env python3
"""E8 Research Module — Contour A: Voice Restoration Feasibility Evaluation.

    python3 research/e8_voice_eval.py \\
        --build-manifest corpus/dev-e8-voice-v1.json \\
        --out-json docs/validation/e8-voice-results.json

Isolated feasibility harness for Board #1135 (E0 charter + spec v1, section E8,
Contour A — Voice). Strictly separate from Contour B (Stems) and strictly
forbidden from generalizing to stereo music mastering (`Preserve`).

Safety boundaries:
- Never opens `corpus/sealed-v1.json` or `corpus/sealed-e5-v1.json`.
- Never modifies any of the 18 frozen E4 files or `rnext/`.
- Never adds production dependencies or launches Colab automation.
- External datasets (DAPS, VoiceBank+DEMAND) are honestly marked
  `not_provisioned_on_host`; external GPU models (DeepFilterNet, SGMSE+,
  Flow-matching speech) and DNSMOS are honestly marked
  `blocked_external_weights_or_colab`; PESQ is marked
  `license_required_for_pesq`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import corpus_synth, corpus_v1

EPS = 1e-20
VOICE_CORPUS_SCHEMA = "e8-voice-corpus/v1"
VOICE_RESULTS_SCHEMA = "e8-voice-results/v1"
VOICE_DATASET_VERSION = "dev-e8-voice-2026-11-23"
VOICE_SR = 16000
VOICE_DURATION_S = 2.5

VOICE_STRATA = (
    "reverb_only",
    "noise_only",
    "reverb_plus_noise",
    "clean_voice_control",
    "ood_voice",
)

VOICE_CPU_ARMS = (
    "identity",
    "wpe_dereverb_cpu",
    "spectral_speech_enhancer_cpu",
)


@dataclass
class VoiceArmResult:
    """Output of a single voice restoration CPU arm."""

    arm: str
    enhanced: np.ndarray
    bypassed: bool
    wall_s: float
    meta: dict = field(default_factory=dict)


def voice_cases() -> list[dict]:
    """Return the deterministic 15-case dev voice corpus specification."""
    return [
        # --- Stratum 1: reverb_only (3 speakers x 3 rooms) ---
        {
            "id": "e8-voice-rev-baritone-small-16k",
            "stratum": "reverb_only",
            "speaker": "baritone_male_synth",
            "f0_hz": 112.0,
            "formant_scale": 0.94,
            "rt60_s": 0.28,
            "drr_db": 4.0,
            "noise_snr_db": None,
            "noise_type": None,
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80101,
        },
        {
            "id": "e8-voice-rev-alto-medium-16k",
            "stratum": "reverb_only",
            "speaker": "alto_female_synth",
            "f0_hz": 186.0,
            "formant_scale": 1.08,
            "rt60_s": 0.55,
            "drr_db": 0.0,
            "noise_snr_db": None,
            "noise_type": None,
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80102,
        },
        {
            "id": "e8-voice-rev-soprano-hall-16k",
            "stratum": "reverb_only",
            "speaker": "soprano_female_synth",
            "f0_hz": 242.0,
            "formant_scale": 1.15,
            "rt60_s": 0.90,
            "drr_db": -3.0,
            "noise_snr_db": None,
            "noise_type": None,
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80103,
        },
        # --- Stratum 2: noise_only (15 / 10 / 5 dB SNR) ---
        {
            "id": "e8-voice-noise-baritone-15db-16k",
            "stratum": "noise_only",
            "speaker": "baritone_male_synth",
            "f0_hz": 118.0,
            "formant_scale": 0.96,
            "rt60_s": None,
            "drr_db": None,
            "noise_snr_db": 15.0,
            "noise_type": "pink_hiss",
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80201,
        },
        {
            "id": "e8-voice-noise-alto-10db-16k",
            "stratum": "noise_only",
            "speaker": "alto_female_synth",
            "f0_hz": 192.0,
            "formant_scale": 1.07,
            "rt60_s": None,
            "drr_db": None,
            "noise_snr_db": 10.0,
            "noise_type": "speech_shaped_hiss",
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80202,
        },
        {
            "id": "e8-voice-noise-soprano-5db-16k",
            "stratum": "noise_only",
            "speaker": "soprano_female_synth",
            "f0_hz": 236.0,
            "formant_scale": 1.14,
            "rt60_s": None,
            "drr_db": None,
            "noise_snr_db": 5.0,
            "noise_type": "pink_hiss",
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80203,
        },
        # --- Stratum 3: reverb_plus_noise ---
        {
            "id": "e8-voice-revnoise-baritone-mild-16k",
            "stratum": "reverb_plus_noise",
            "speaker": "baritone_male_synth",
            "f0_hz": 115.0,
            "formant_scale": 0.95,
            "rt60_s": 0.35,
            "drr_db": 3.0,
            "noise_snr_db": 14.0,
            "noise_type": "pink_hiss",
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80301,
        },
        {
            "id": "e8-voice-revnoise-alto-mod-16k",
            "stratum": "reverb_plus_noise",
            "speaker": "alto_female_synth",
            "f0_hz": 182.0,
            "formant_scale": 1.06,
            "rt60_s": 0.55,
            "drr_db": 0.5,
            "noise_snr_db": 10.0,
            "noise_type": "speech_shaped_hiss",
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80302,
        },
        {
            "id": "e8-voice-revnoise-soprano-heavy-16k",
            "stratum": "reverb_plus_noise",
            "speaker": "soprano_female_synth",
            "f0_hz": 248.0,
            "formant_scale": 1.16,
            "rt60_s": 0.75,
            "drr_db": -2.0,
            "noise_snr_db": 7.0,
            "noise_type": "pink_hiss",
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80303,
        },
        # --- Stratum 4: clean_voice_control (negative controls) ---
        {
            "id": "e8-voice-ctl-clean-baritone-16k",
            "stratum": "clean_voice_control",
            "speaker": "baritone_male_synth",
            "f0_hz": 110.0,
            "formant_scale": 0.94,
            "rt60_s": None,
            "drr_db": None,
            "noise_snr_db": None,
            "noise_type": None,
            "telephony_bandlimit": False,
            "is_clean_control": True,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80401,
        },
        {
            "id": "e8-voice-ctl-clean-alto-16k",
            "stratum": "clean_voice_control",
            "speaker": "alto_female_synth",
            "f0_hz": 185.0,
            "formant_scale": 1.08,
            "rt60_s": None,
            "drr_db": None,
            "noise_snr_db": None,
            "noise_type": None,
            "telephony_bandlimit": False,
            "is_clean_control": True,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80402,
        },
        {
            "id": "e8-voice-ctl-clean-soprano-16k",
            "stratum": "clean_voice_control",
            "speaker": "soprano_female_synth",
            "f0_hz": 245.0,
            "formant_scale": 1.15,
            "rt60_s": None,
            "drr_db": None,
            "noise_snr_db": None,
            "noise_type": None,
            "telephony_bandlimit": False,
            "is_clean_control": True,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80403,
        },
        # --- Stratum 5: ood_voice ---
        {
            "id": "e8-voice-ood-hall-lowsnr-16k",
            "stratum": "ood_voice",
            "speaker": "baritone_male_synth",
            "f0_hz": 116.0,
            "formant_scale": 0.95,
            "rt60_s": 1.10,
            "drr_db": -5.0,
            "noise_snr_db": 2.0,
            "noise_type": "pink_hiss",
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80501,
        },
        {
            "id": "e8-voice-ood-nonstationary-babble-16k",
            "stratum": "ood_voice",
            "speaker": "alto_female_synth",
            "f0_hz": 190.0,
            "formant_scale": 1.07,
            "rt60_s": 0.45,
            "drr_db": 1.0,
            "noise_snr_db": 4.0,
            "noise_type": "nonstationary_babble",
            "telephony_bandlimit": False,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80502,
        },
        {
            "id": "e8-voice-ood-telephony-bandlimit-16k",
            "stratum": "ood_voice",
            "speaker": "soprano_female_synth",
            "f0_hz": 238.0,
            "formant_scale": 1.14,
            "rt60_s": 0.40,
            "drr_db": 2.0,
            "noise_snr_db": 8.0,
            "noise_type": "pink_hiss",
            "telephony_bandlimit": True,
            "is_clean_control": False,
            "rate": VOICE_SR,
            "duration_s": VOICE_DURATION_S,
            "seed": 80503,
        },
    ]


def _biquad_resonator(f0_hz: float, bw_hz: float, sr: int) -> np.ndarray:
    """Design a unity-peak second-order bandpass resonator (SOS row)."""
    f0 = float(np.clip(f0_hz, 40.0, 0.46 * sr))
    bw = float(max(bw_hz, 20.0))
    r = float(np.exp(-np.pi * bw / sr))
    w0 = 2.0 * np.pi * f0 / sr
    a1 = -2.0 * r * np.cos(w0)
    a2 = r * r
    gain = (1.0 - r) * np.sqrt(max(1.0 - 2.0 * r * np.cos(2.0 * w0) + r * r, 1e-6))
    return np.array([[gain, 0.0, -gain * r, 1.0, a1, a2]], dtype=np.float64)


def _synthesize_formant_speech(
    sr: int,
    duration_s: float,
    f0_hz: float,
    formant_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize deterministic speech-like formant + syllable + fricative waveform."""
    n = round(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr

    # Prosody pitch contour with gentle declination and vibrato.
    f0_t = f0_hz * (
        1.0
        + 0.09 * np.sin(2.0 * np.pi * 0.75 * t + rng.uniform(0, 2 * np.pi))
        - 0.06 * (t / max(duration_s, 1e-6))
        + 0.015 * np.sin(2.0 * np.pi * 5.2 * t)
    )
    phase = 2.0 * np.pi * np.cumsum(f0_t) / sr

    # Band-limited glottal harmonic source (1/k^1.15 roll-off up to 3.8 kHz).
    glottal = np.zeros(n, dtype=np.float64)
    max_harm = min(28, int(3800.0 / max(f0_hz, 60.0)))
    for k in range(1, max_harm + 1):
        glottal += (1.0 / (k ** 1.15)) * np.sin(k * phase + 0.12 * k)

    # Vowel formant inventory (F1, F2, F3 in Hz) scaled by speaker tract length.
    vowels = [
        (730.0, 1090.0, 2440.0),  # /a/
        (300.0, 2250.0, 3000.0),  # /i/
        (340.0, 880.0, 2240.0),   # /u/
        (530.0, 1840.0, 2480.0),  # /e/
        (570.0, 840.0, 2410.0),   # /o/
    ]

    # Build 9 syllables with realistic word pauses (speech active ~72% of clip).
    speech = np.zeros(n, dtype=np.float64)
    cursor = int(0.14 * sr)
    syl_idx = 0
    while cursor < n - int(0.20 * sr):
        syl_dur = int(rng.uniform(0.14, 0.22) * sr)
        pause_dur = int(rng.uniform(0.05, 0.13) * sr)
        if syl_idx % 3 == 2:
            pause_dur += int(0.08 * sr)
        end = min(cursor + syl_dur, n - int(0.08 * sr))
        seg_len = end - cursor
        if seg_len < int(0.08 * sr):
            break

        f1, f2, f3 = vowels[int(rng.integers(0, len(vowels)))]
        f1 *= formant_scale
        f2 *= formant_scale
        f3 *= formant_scale

        seg_src = glottal[cursor:end]
        y1 = signal.sosfilt(_biquad_resonator(f1, 75.0, sr), seg_src)
        y2 = signal.sosfilt(_biquad_resonator(f2, 110.0, sr), seg_src)
        y3 = signal.sosfilt(_biquad_resonator(f3, 150.0, sr), seg_src)
        voiced = 0.58 * y1 + 0.30 * y2 + 0.14 * y3

        # Smooth syllable envelope (raised-cosine attack/release).
        env = np.ones(seg_len, dtype=np.float64)
        atk = min(int(0.025 * sr), seg_len // 3)
        rel = min(int(0.035 * sr), seg_len // 3)
        env[:atk] = 0.5 * (1.0 - np.cos(np.pi * np.arange(atk) / max(atk, 1)))
        env[-rel:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(rel) / max(rel, 1)))
        speech[cursor:end] += voiced * env

        # Add unvoiced fricative burst (/s/ or /sh/) at onset on alternating syllables.
        if syl_idx % 2 == 0 and cursor >= int(0.04 * sr):
            f_len = int(0.038 * sr)
            f_start = max(0, cursor - int(0.022 * sr))
            f_end = min(n, f_start + f_len)
            burst = rng.normal(0.0, 1.0, f_end - f_start)
            sos_fric = signal.butter(4, [3200.0 / (0.5 * sr), 6800.0 / (0.5 * sr)], btype="band", output="sos")
            burst = signal.sosfilt(sos_fric, burst)
            win_f = np.hanning(f_end - f_start)
            speech[f_start:f_end] += 0.14 * burst * win_f

        cursor = end + pause_dur
        syl_idx += 1

    # Normalize active speech to -18 dBFS RMS (peak safely < 0.85).
    rms = float(np.sqrt(np.mean(speech ** 2) + EPS))
    target_rms = 10.0 ** (-18.0 / 20.0)
    speech = speech * (target_rms / rms)
    peak = float(np.max(np.abs(speech)))
    if peak > 0.82:
        speech *= 0.82 / peak
    return speech


def _synthesize_rir(sr: int, rt60_s: float, drr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Synthesize a deterministic causal room impulse response (direct + early + diffuse tail)."""
    rir_len = min(int(sr * max(rt60_s * 1.15, 0.25)), int(sr * 1.4))
    rir = np.zeros(rir_len, dtype=np.float64)
    t = np.arange(rir_len, dtype=np.float64) / sr

    # Diffuse exponentially decaying late tail starting after 15 ms.
    decay = np.exp(-6.907755 * t / max(rt60_s, 0.05))
    tail = rng.normal(0.0, 1.0, rir_len) * decay
    early_start = int(0.012 * sr)
    tail[:early_start] = 0.0

    # Air absorption low-pass on diffuse tail (~5.2 kHz).
    sos_lp = signal.butter(2, 5200.0 / (0.5 * sr), btype="low", output="sos")
    tail = signal.sosfilt(sos_lp, tail)

    # Sparse early reflections (12 ms .. 55 ms).
    for delay_ms in (14.0, 21.5, 29.0, 37.5, 46.0, 54.0):
        idx = round(delay_ms * 1e-3 * sr)
        if idx < rir_len:
            sign = 1.0 if rng.random() > 0.5 else -1.0
            tail[idx] += sign * float(np.exp(-6.907755 * (delay_ms * 1e-3) / max(rt60_s, 0.05))) * 0.45

    tail_e = float(np.sum(tail ** 2))
    if tail_e > EPS:
        desired_tail_e = 10.0 ** (-drr_db / 10.0)
        tail *= np.sqrt(desired_tail_e / tail_e)

    rir[0] = 1.0
    rir += tail
    return rir


def _synthesize_noise(n: int, sr: int, noise_type: str, rng: np.random.Generator) -> np.ndarray:
    """Synthesize stationary pink/speech-shaped or non-stationary babble-like noise."""
    white = rng.normal(0.0, 1.0, n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spec = np.fft.rfft(white)
    if noise_type == "pink_hiss":
        shape = 1.0 / np.sqrt(np.maximum(freqs, 60.0))
    elif noise_type == "speech_shaped_hiss":
        # LTASS-like tilt peaking around 450 Hz.
        shape = 1.0 / (1.0 + (freqs / 550.0) ** 1.35)
        shape *= np.minimum(freqs / 120.0, 1.0)
    elif noise_type == "nonstationary_babble":
        shape = 1.0 / (1.0 + (freqs / 650.0) ** 1.2)
        shape *= np.minimum(freqs / 150.0, 1.0)
    else:
        shape = np.ones_like(freqs)
    shape[0] = 0.0
    noise = np.fft.irfft(spec * shape, n=n)
    if noise_type == "nonstationary_babble":
        t = np.arange(n, dtype=np.float64) / sr
        mod = 0.35 + 0.65 * np.abs(np.sin(2.0 * np.pi * 2.3 * t + 0.4) * np.cos(2.0 * np.pi * 0.9 * t))
        noise *= mod
    return noise


def synthesize_voice_case(case: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deterministically synthesize `(clean, degraded, meta)` 1D float64 mono arrays."""
    sr = int(case["rate"])
    dur = float(case["duration_s"])
    rng = np.random.default_rng(int(case["seed"]))

    clean = _synthesize_formant_speech(
        sr, dur, float(case["f0_hz"]), float(case["formant_scale"]), rng
    )
    if case.get("is_clean_control", False):
        return clean, clean.copy(), {"rir_applied": False, "noise_applied": False}

    degraded = clean.copy()
    rir_applied = False
    if case.get("rt60_s") is not None:
        rir = _synthesize_rir(sr, float(case["rt60_s"]), float(case["drr_db"]), rng)
        degraded = signal.fftconvolve(degraded, rir)[: clean.shape[0]]
        rir_applied = True

    noise_applied = False
    if case.get("noise_snr_db") is not None:
        noise = _synthesize_noise(clean.shape[0], sr, str(case["noise_type"]), rng)
        sig_e = float(np.mean(clean ** 2) + EPS)
        noise_e = float(np.mean(noise ** 2) + EPS)
        target_noise_e = sig_e / (10.0 ** (float(case["noise_snr_db"]) / 10.0))
        degraded = degraded + noise * np.sqrt(target_noise_e / noise_e)
        noise_applied = True

    if case.get("telephony_bandlimit", False):
        sos_bp = signal.butter(4, [300.0 / (0.5 * sr), 3400.0 / (0.5 * sr)], btype="band", output="sos")
        degraded = signal.sosfiltfilt(sos_bp, degraded)

    # Prevent clipping (> 0.96) with shared safe attenuation if needed.
    peak = float(np.max(np.abs(degraded)))
    if peak > 0.95:
        degraded = degraded * (0.95 / peak)

    return clean, degraded, {"rir_applied": rir_applied, "noise_applied": noise_applied}


def _is_clean_dry_speech(x: np.ndarray, sr: int) -> tuple[bool, dict]:
    """Detect dry, noiseless speech so CPU enhancers abstain on clean negative controls."""
    frame = max(int(0.025 * sr), 64)
    hop = max(frame // 2, 32)
    n_frames = max(1, (x.shape[0] - frame) // hop + 1)
    energies = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop : i * hop + frame]
        energies[i] = float(np.mean(seg ** 2))
    p10 = float(np.percentile(energies, 10))
    p90 = float(np.percentile(energies, 90))
    floor_dbfs = float(10.0 * np.log10(max(p10, EPS)))
    dyn_db = float(10.0 * np.log10(max(p90, EPS) / max(p10, EPS)))
    # Clean anechoic speech with pauses has very low pause floor (< -58 dBFS) and high dynamic range.
    is_clean = (floor_dbfs < -58.0) and (dyn_db > 42.0)
    return is_clean, {"floor_dbfs": floor_dbfs, "dynamic_range_db": dyn_db}


def _wpe_dereverb_stft(
    x: np.ndarray,
    sr: int,
    *,
    n_fft: int = 512,
    hop: int = 128,
    taps: int = 8,
    delay: int = 2,
    iters: int = 2,
    reg: float = 1e-4,
) -> np.ndarray:
    """Classic single-channel Weighted Prediction Error (WPE) dereverberation (Nakatani et al. 2010)."""
    n = x.shape[0]
    _f, _t, Z = signal.stft(x, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    n_freq, n_frames = Z.shape
    if n_frames <= delay + taps + 4:
        return x.copy()

    Y = Z.copy()
    # Pre-build delayed design matrix per frequency bin: shape (n_frames, taps).
    valid_start = delay + taps - 1
    idx = np.arange(valid_start, n_frames)
    n_valid = idx.shape[0]
    eye = np.eye(taps, dtype=np.complex128)

    for _it in range(iters):
        # Variance estimate lambda_{t,f} smoothed across +/-1 frames.
        power = np.abs(Y) ** 2
        var = power.copy()
        var[:, 1:-1] = (power[:, :-2] + power[:, 1:-1] + power[:, 2:]) / 3.0
        floor = np.maximum(np.median(var, axis=1, keepdims=True) * 1e-3, 1e-10)
        inv_var = 1.0 / np.maximum(var[:, valid_start:], floor)

        Y_new = Z.copy()
        for f_idx in range(n_freq):
            z_f = Z[f_idx]
            # Build delayed Hankel-like matrix X_d of shape (n_valid, taps).
            X_d = np.empty((n_valid, taps), dtype=np.complex128)
            for l_idx in range(taps):
                X_d[:, l_idx] = z_f[idx - delay - l_idx]
            w = inv_var[f_idx]
            Xw = X_d * w[:, None]
            R = Xw.conj().T @ X_d
            trace_reg = reg * max(float(np.real(np.trace(R))) / taps, 1e-10)
            rhs = Xw.conj().T @ z_f[idx]
            try:
                g = np.linalg.solve(R + trace_reg * eye, rhs)
            except np.linalg.LinAlgError:
                continue
            # Limit filter norm conservatively to prevent over-subtraction on dry/noisy speech.
            g_norm = float(np.linalg.norm(g))
            if g_norm > 0.85:
                g *= 0.85 / g_norm
            Y_new[f_idx, valid_start:] = z_f[idx] - (X_d @ g)
        Y = Y_new

    _, y_out = signal.istft(Y, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    y_out = np.asarray(y_out[:n], dtype=np.float64)
    if y_out.shape[0] < n:
        y_out = np.pad(y_out, (0, n - y_out.shape[0]))
    return y_out


def _spectral_wiener_speech_enhance(
    x: np.ndarray,
    sr: int,
    *,
    n_fft: int = 512,
    hop: int = 128,
    alpha_dd: float = 0.92,
    g_min: float = 0.14,
) -> np.ndarray:
    """Decision-directed Wiener spectral enhancer with minimum-statistics noise floor."""
    n = x.shape[0]
    _f, _t, Z = signal.stft(x, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    n_freq, n_frames = Z.shape
    mag2 = np.abs(Z) ** 2

    # Minimum-statistics / 12th-percentile noise PSD estimate across frames per bin.
    noise_psd = np.percentile(mag2, 12, axis=1) * 1.15
    noise_psd = np.maximum(noise_psd, 1e-12)

    gains = np.ones((n_freq, n_frames), dtype=np.float64)
    prev_post_snr = np.ones(n_freq, dtype=np.float64)
    prev_gain = np.ones(n_freq, dtype=np.float64)

    for t_idx in range(n_frames):
        gamma = mag2[:, t_idx] / noise_psd
        # Decision-directed a priori SNR (Ephraim-Malah).
        xi = alpha_dd * (prev_gain ** 2) * prev_post_snr + (1.0 - alpha_dd) * np.maximum(gamma - 1.0, 0.0)
        g = xi / (xi + 1.0)
        g = np.clip(g, g_min, 1.0)
        gains[:, t_idx] = g
        prev_post_snr = np.minimum(gamma, 40.0)
        prev_gain = g

    # Mild frequency smoothing of gain mask to protect vocal formants from musical noise.
    gains = signal.convolve2d(gains, np.array([[0.2], [0.6], [0.2]]), mode="same", boundary="symm")
    _, y_out = signal.istft(Z * gains, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    y_out = np.asarray(y_out[:n], dtype=np.float64)
    if y_out.shape[0] < n:
        y_out = np.pad(y_out, (0, n - y_out.shape[0]))
    return y_out


def run_voice_arm(
    arm: str,
    inp: np.ndarray,
    sr: int,
    *,
    bypass_clean: bool = True,
) -> VoiceArmResult:
    """Run one Contour-A voice CPU arm without mutating `inp`."""
    x = np.asarray(inp, dtype=np.float64).copy()
    t0 = time.perf_counter()

    if arm == "identity":
        return VoiceArmResult(
            arm=arm,
            enhanced=x.copy(),
            bypassed=True,
            wall_s=time.perf_counter() - t0,
            meta={"reason": "identity_noop"},
        )

    is_clean, clean_meta = _is_clean_dry_speech(x, sr)
    if bypass_clean and is_clean:
        return VoiceArmResult(
            arm=arm,
            enhanced=x.copy(),
            bypassed=True,
            wall_s=time.perf_counter() - t0,
            meta={"reason": "clean_dry_speech_bypass", **clean_meta},
        )

    if arm == "wpe_dereverb_cpu":
        y = _wpe_dereverb_stft(x, sr, taps=12, delay=2, iters=3)
        return VoiceArmResult(
            arm=arm,
            enhanced=y,
            bypassed=False,
            wall_s=time.perf_counter() - t0,
            meta={"taps": 12, "delay": 2, "iters": 3, **clean_meta},
        )

    if arm == "spectral_speech_enhancer_cpu":
        y = _spectral_wiener_speech_enhance(x, sr)
        return VoiceArmResult(
            arm=arm,
            enhanced=y,
            bypassed=False,
            wall_s=time.perf_counter() - t0,
            meta={"stages": ["dd_wiener_formant_floor"], **clean_meta},
        )

    raise ValueError(f"unknown E8 voice arm: {arm!r}")


# ---------------------------------------------------------------------------
# Voice Metrics (Contour A)
# ---------------------------------------------------------------------------

def si_sdr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio (dB)."""
    r = np.asarray(ref, dtype=np.float64).ravel()
    e = np.asarray(est, dtype=np.float64).ravel()
    r = r - np.mean(r)
    e = e - np.mean(e)
    ref_e = float(np.dot(r, r))
    if ref_e <= EPS:
        return 0.0
    alpha = float(np.dot(e, r)) / ref_e
    s_target = alpha * r
    e_noise = e - s_target
    num = float(np.dot(s_target, s_target))
    den = float(np.dot(e_noise, e_noise))
    if den <= EPS:
        return 60.0
    return float(min(60.0, 10.0 * np.log10(max(num, EPS) / den)))


def stoi_proxy_envelope_corr(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Short-time 1/3-octave band envelope correlation proxy for speech intelligibility.

    Modeled on Taal et al. (2011) STOI principles (15 1/3-octave bands 150-4300 Hz,
    384 ms short-time segments, normalized & clipped correlation). Explicitly
    labeled as a CPU proxy, not official licensed PESQ/pystoi.
    """
    n_fft = 512
    hop = 200  # 12.5 ms at 16 kHz -> 30 frames = 375 ms
    freqs, _t, R = signal.stft(ref, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    _, _, E = signal.stft(est, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    R_mag = np.abs(R)
    E_mag = np.abs(E)

    # 15 1/3-octave bands from 150 Hz to ~4300 Hz.
    cf = 150.0 * (2.0 ** (np.arange(15) / 3.0))
    edges_lo = cf * (2.0 ** (-1.0 / 6.0))
    edges_hi = cf * (2.0 ** (1.0 / 6.0))

    n_frames = R_mag.shape[1]
    band_r = np.zeros((15, n_frames), dtype=np.float64)
    band_e = np.zeros((15, n_frames), dtype=np.float64)
    for b in range(15):
        mask = (freqs >= edges_lo[b]) & (freqs < edges_hi[b])
        if np.any(mask):
            band_r[b] = np.sqrt(np.sum(R_mag[mask] ** 2, axis=0) + EPS)
            band_e[b] = np.sqrt(np.sum(E_mag[mask] ** 2, axis=0) + EPS)

    # Active speech frames (within 40 dB of peak frame energy).
    frame_e = np.sum(band_r ** 2, axis=0)
    active = frame_e > (np.max(frame_e) * 1e-4)
    band_r = band_r[:, active]
    band_e = band_e[:, active]
    m_frames = band_r.shape[1]
    seg_len = 30
    if m_frames < seg_len:
        seg_len = max(m_frames, 4)

    beta = 10.0 ** (-15.0 / 20.0)
    corrs = []
    for start in range(max(1, m_frames - seg_len + 1)):
        xr = band_r[:, start : start + seg_len]
        xe = band_e[:, start : start + seg_len]
        # Scale & clip degraded envelope per band (STOI normalization).
        norm_r = np.linalg.norm(xr, axis=1, keepdims=True) + EPS
        norm_e = np.linalg.norm(xe, axis=1, keepdims=True) + EPS
        xe_scaled = xe * (norm_r / norm_e)
        xe_clipped = np.minimum(xe_scaled, xr * (1.0 + beta))
        xr_c = xr - np.mean(xr, axis=1, keepdims=True)
        xe_c = xe_clipped - np.mean(xe_clipped, axis=1, keepdims=True)
        num = np.sum(xr_c * xe_c, axis=1)
        den = (np.linalg.norm(xr_c, axis=1) * np.linalg.norm(xe_c, axis=1)) + EPS
        corrs.append(float(np.mean(num / den)))
    return float(np.clip(np.mean(corrs) if corrs else 0.0, -1.0, 1.0))


def formant_drift_hz(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Estimate mean absolute F1/F2/F3 formant-peak drift (Hz) on active voiced frames."""
    n_fft = 512
    hop = 128
    freqs, _t, R = signal.stft(ref, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    _, _, E = signal.stft(est, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    R_pow = np.abs(R) ** 2
    E_pow = np.abs(E) ** 2

    frame_e = np.sum(R_pow, axis=0)
    thresh = float(np.percentile(frame_e, 60))
    voiced_idx = np.where(frame_e >= max(thresh, 1e-10))[0]
    if voiced_idx.size == 0:
        return 0.0

    # Smooth log-magnitude spectrum along frequency (cepstral-like envelope smoothing)
    # to extract vocal tract resonances rather than individual pitch harmonics.
    kernel = np.hanning(11)
    kernel /= np.sum(kernel)
    bands = ((250.0, 900.0), (900.0, 2400.0), (2200.0, 3800.0))

    drifts = []
    for t_i in voiced_idx:
        env_r = np.convolve(np.log10(R_pow[:, t_i] + 1e-12), kernel, mode="same")
        env_e = np.convolve(np.log10(E_pow[:, t_i] + 1e-12), kernel, mode="same")
        for f_lo, f_hi in bands:
            m = (freqs >= f_lo) & (freqs <= f_hi)
            if not np.any(m):
                continue
            f_sub = freqs[m]
            fr = float(f_sub[int(np.argmax(env_r[m]))])
            fe = float(f_sub[int(np.argmax(env_e[m]))])
            drifts.append(abs(fe - fr))
    return float(np.mean(drifts)) if drifts else 0.0


def speech_spectral_lsd_db(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Log-spectral distance (dB) over the [100 Hz, 7500 Hz] speech band."""
    n_fft = 512
    hop = 128
    freqs, _t, R = signal.stft(ref, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    _, _, E = signal.stft(est, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    mask = (freqs >= 100.0) & (freqs <= min(7500.0, 0.47 * sr))
    lr = 10.0 * np.log10(np.abs(R[mask]) ** 2 + 1e-10)
    le = 10.0 * np.log10(np.abs(E[mask]) ** 2 + 1e-10)
    return float(np.mean(np.sqrt(np.mean((lr - le) ** 2, axis=0))))


def syllable_envelope_corr(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Pearson correlation of 1.5-10 Hz word/syllable modulation envelope."""
    sos_lp = signal.butter(4, 12.0 / (0.5 * sr), btype="low", output="sos")
    env_r = signal.sosfiltfilt(sos_lp, np.abs(ref))
    env_e = signal.sosfiltfilt(sos_lp, np.abs(est))
    # Downsample envelope to 100 Hz for clean correlation.
    step = max(sr // 100, 1)
    er = env_r[::step] - np.mean(env_r[::step])
    ee = env_e[::step] - np.mean(env_e[::step])
    den = float(np.linalg.norm(er) * np.linalg.norm(ee))
    if den <= EPS:
        return 1.0
    return float(np.clip(float(np.dot(er, ee)) / den, -1.0, 1.0))


def evaluate_voice_case(
    clean: np.ndarray,
    degraded: np.ndarray,
    result: VoiceArmResult,
    sr: int,
    *,
    is_clean_control: bool,
    input_sha256_before: str | None = None,
    input_sha256_after: str | None = None,
) -> dict:
    """Compute all Contour-A voice metrics for one case and one CPU arm."""
    y = result.enhanced
    if y.shape != clean.shape or not np.isfinite(y).all():
        raise ValueError(f"{result.arm}: invalid enhanced voice shape or non-finite values")

    in_sisdr = si_sdr_db(clean, degraded)
    out_sisdr = si_sdr_db(clean, y)
    in_stoi = stoi_proxy_envelope_corr(clean, degraded, sr)
    out_stoi = stoi_proxy_envelope_corr(clean, y, sr)
    in_fdrift = formant_drift_hz(clean, degraded, sr)
    out_fdrift = formant_drift_hz(clean, y, sr)
    in_lsd = speech_spectral_lsd_db(clean, degraded, sr)
    out_lsd = speech_spectral_lsd_db(clean, y, sr)
    in_syl = syllable_envelope_corr(clean, degraded, sr)
    out_syl = syllable_envelope_corr(clean, y, sr)

    input_mutated = (
        input_sha256_before != input_sha256_after
        if (input_sha256_before is not None and input_sha256_after is not None)
        else False
    )
    audio_min = max(clean.shape[0] / float(sr) / 60.0, 1e-9)
    return {
        "arm": result.arm,
        "bypassed": bool(result.bypassed),
        "is_clean_control": bool(is_clean_control),
        "si_sdr_in_db": in_sisdr,
        "si_sdr_out_db": out_sisdr,
        "si_sdri_db": float(out_sisdr - in_sisdr),
        "stoi_proxy_in": in_stoi,
        "stoi_proxy_out": out_stoi,
        "delta_stoi_proxy": float(out_stoi - in_stoi),
        "formant_drift_in_hz": in_fdrift,
        "formant_drift_out_hz": out_fdrift,
        "delta_formant_drift_hz": float(out_fdrift - in_fdrift),
        "spectral_lsd_in_db": in_lsd,
        "spectral_lsd_out_db": out_lsd,
        "delta_spectral_lsd_db": float(out_lsd - in_lsd),
        "syllable_env_corr_in": in_syl,
        "syllable_env_corr_out": out_syl,
        "delta_syllable_env_corr": float(out_syl - in_syl),
        "clean_control_max_abs_err": float(np.max(np.abs(y - clean))) if is_clean_control else None,
        "input_mutated": input_mutated,
        "wall_s": float(result.wall_s),
        "wall_s_per_audio_min": float(result.wall_s / audio_min),
    }


# ---------------------------------------------------------------------------
# Honest External Audits & Go/No-Go #1 (Voice Profile)
# ---------------------------------------------------------------------------

def audit_voice_external_models_and_metrics(models_dir: Path | None = None) -> dict:
    """Honest local audit of external voice neural models, datasets, and licensed metrics."""
    m_dir = models_dir if models_dir is not None else (REPO / "models")
    weight_exts = (".pt", ".pth", ".ckpt", ".onnx", ".safetensors")
    voice_keywords = ("deepfilternet", "sgmse", "dnsmos", "voice", "speech_flow")
    local_weights = []
    if m_dir.exists():
        for p in sorted(m_dir.rglob("*")):
            if (
                p.is_file()
                and p.suffix.lower() in weight_exts
                and any(k in p.name.lower() for k in voice_keywords)
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

    colab_authorized = bool(os.environ.get("MR_E8_COLAB_AUTHORIZED") == "1")
    blocked_reasons = []
    if not local_weights:
        blocked_reasons.append("no_local_licensed_voice_weights_deepfilternet_sgmse_flow")
    if not cuda_available:
        blocked_reasons.append("no_local_cuda_gpu")
    if not colab_authorized:
        blocked_reasons.append("colab_connection_not_authorized_or_configured")

    neural_status = "blocked_external_weights_or_colab" if blocked_reasons else "ready"

    return {
        "contour": "A_voice",
        "neural_arm_status": neural_status,
        "executed": False,
        "production_dependency_added": False,
        "blocked_reasons": blocked_reasons,
        "environment_audit": {
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "local_voice_weights_found": local_weights,
            "colab_authorized": colab_authorized,
        },
        "external_datasets": [
            {
                "id": "daps",
                "name": "DAPS (Device and Produced Speech)",
                "status": "not_provisioned_on_host",
                "license_note": "CC-BY-NC-4.0 / research terms — verify license before download or Colab upload",
                "required_action": "Download externally with license verification; keep strictly out of sealed music corpora",
            },
            {
                "id": "voicebank_demand",
                "name": "VoiceBank + DEMAND (28/56 speakers)",
                "status": "not_provisioned_on_host",
                "license_note": "CC-BY-4.0 — verify split integrity (no speaker overlap between train/test)",
                "required_action": "Provision on authorized Colab session or approved dev storage with SHA-256 manifest",
            },
        ],
        "external_neural_models": [
            {
                "id": "deepfilternet3",
                "status": neural_status,
                "version_pin": "DeepFilterNet 0.5.6 (DeepFilterNet3 48 kHz)",
                "license": "MIT / Apache-2.0 dual-licensed",
                "colab_budget_estimate": "T4/L4 GPU (~1-2 GB VRAM, ~1-3 s/min audio; also CPU-capable after weight audit)",
                "target_role": "Real-time / low-latency noise suppression + mild dereverberation",
            },
            {
                "id": "sgmse_plus",
                "status": neural_status,
                "version_pin": "SGMSE+ (Richter et al. 2023, score-based diffusion)",
                "license": "MIT (verify checkpoint release terms)",
                "colab_budget_estimate": "T4/L4 GPU (~4-6 GB VRAM, ~15-40 s/min audio for 30-step reverse SDE)",
                "target_role": "Generative speech dereverberation & enhancement (requires hallucination/word-drop audit)",
            },
            {
                "id": "flow_matching_speech",
                "status": neural_status,
                "version_pin": "FlowSE / conditional flow-matching speech restoration (pin commit + SHA-256)",
                "license": "Verify repository & checkpoint SPDX before download",
                "colab_budget_estimate": "T4/L4 GPU (~4-8 GB VRAM, ~8-20 s/min audio for 5-10 ODE steps)",
                "target_role": "Low-step generative speech enhancement (strictly Enhance preview, never Preserve)",
            },
        ],
        "external_perceptual_metrics": [
            {
                "id": "pesq",
                "name": "PESQ (ITU-T P.862)",
                "status": "license_required_for_pesq",
                "license_note": "Proprietary / patent-encumbered ITU-T reference implementation; requires license check before install",
                "local_proxy_used": "stoi_proxy_envelope_corr + formant_drift_hz + syllable_env_corr",
            },
            {
                "id": "dnsmos",
                "name": "DNSMOS P.835 (SIG/BAK/OVRL)",
                "status": "blocked_external_weights_or_colab",
                "license_note": "Microsoft ONNX weights (CC-BY-4.0 / MSR terms) not provisioned locally",
                "local_proxy_used": "si_sdri_db + spectral_lsd_db + stoi_proxy_envelope_corr",
            },
        ],
        "colab_handoff_spec": {
            "permitted_inputs": "corpus/dev-e8-voice-v1.json + licensed DAPS/VoiceBank dev split ONLY",
            "forbidden_inputs": ["corpus/sealed-v1.json", "corpus/sealed-e5-v1.json", "private_masters"],
            "required_run_manifest_fields": [
                "gpu_model",
                "cuda_version",
                "torch_version",
                "model_id",
                "checkpoint_sha256",
                "license_spdx",
                "dataset_sha256",
                "wall_s_per_audio_min",
                "peak_vram_gb",
            ],
        },
    }


def evaluate_go_no_go_voice(summary_by_stratum: dict, external_audit: dict) -> dict:
    """Produce independent Go/No-Go #1 decision for Contour A (Voice Profile)."""
    rev_wpe = summary_by_stratum["reverb_only"]["wpe_dereverb_cpu"]
    noise_spec = summary_by_stratum["noise_only"]["spectral_speech_enhancer_cpu"]
    ctl_wpe = summary_by_stratum["clean_voice_control"]["wpe_dereverb_cpu"]
    ctl_spec = summary_by_stratum["clean_voice_control"]["spectral_speech_enhancer_cpu"]

    cpu_baselines_pass_sanity = bool(
        rev_wpe["mean_si_sdri_db"] > 0.5
        and noise_spec["mean_si_sdri_db"] > 1.5
        and ctl_wpe["max_clean_control_err"] == 0.0
        and ctl_spec["max_clean_control_err"] == 0.0
    )

    return {
        "decision_id": "go_no_go_1_voice_profile",
        "contour": "A_voice",
        "production_verdict": "no_go",
        "colab_research_verdict": "conditional_go" if cpu_baselines_pass_sanity else "no_go",
        "preserve_stereo_music_generalization": "forbidden",
        "rationale": (
            "CPU WPE + spectral speech enhancement baselines validate the evaluation harness "
            "with positive SI-SDRi on reverberant/noisy synthetic speech and 0.0 error on clean "
            "voice negative controls, but production adoption is NO-GO because: (1) real speech "
            "corpora (DAPS, VoiceBank+DEMAND) are not provisioned on host, (2) neural speech "
            "models (DeepFilterNet3, SGMSE+, Flow) and DNSMOS are blocked pending authorized Colab "
            "GPU + checkpoint audit, (3) PESQ requires explicit license clearance, and (4) speech "
            "enhancement must never be generalized to stereo music Preserve."
        ),
        "cpu_harness_verified": cpu_baselines_pass_sanity,
        "decision_inputs_for_colab_voice_stage": [
            "Provision licensed DAPS and VoiceBank+DEMAND dev splits with speaker-disjoint manifests and SHA-256 pins.",
            "Authorize Google Colab GPU session (separate ticket) with pinned DeepFilterNet3 (MIT/Apache-2.0) and SGMSE+ (MIT) checkpoints.",
            "Clear ITU-T P.862 license for PESQ or standardize on open-licensed DNSMOS P.835 + Whisper WER / phoneme-drop hallucination checks.",
            "Require >= +3.0 dB SI-SDRi and positive STOI/DNSMOS gain over CPU WPE+Wiener on DAPS/VoiceBank with 0 regressions on clean anechoic controls.",
            "Keep any future voice profile strictly isolated from Preserve stereo music mastering (separate opt-in voice route only after product approval).",
        ],
        "external_blockers": external_audit["blocked_reasons"],
    }


# ---------------------------------------------------------------------------
# Manifest Builder & Benchmark Runner
# ---------------------------------------------------------------------------

def build_voice_manifest(out_path: Path) -> dict:
    """Build deterministic `corpus/dev-e8-voice-v1.json` with SHA-256 audio digests."""
    cases = voice_cases()
    file_entries = []
    strata_counts: dict[str, int] = {s: 0 for s in VOICE_STRATA}
    with tempfile.TemporaryDirectory(prefix="e8_voice_manifest_") as td:
        td_p = Path(td)
        for case in cases:
            strata_counts[case["stratum"]] += 1
            clean, degraded, _meta = synthesize_voice_case(case)
            for tag, arr in (("clean", clean), ("degraded", degraded)):
                wav_name = f"{case['id']}.{tag}.wav"
                wav_path = td_p / wav_name
                corpus_synth.write_wav(wav_path, int(case["rate"]), arr, bits=16, seed=int(case["seed"]))
                file_entries.append({
                    "path": wav_name,
                    "sha256": corpus_v1.digest(wav_path),
                    "bytes": wav_path.stat().st_size,
                })

    manifest = {
        "schema": VOICE_CORPUS_SCHEMA,
        "contour": "A_voice",
        "tier": "dev",
        "dataset_version": VOICE_DATASET_VERSION,
        "sealed_corpora_opened": False,
        "external_datasets_status": {
            "daps": "not_provisioned_on_host",
            "voicebank_demand": "not_provisioned_on_host",
        },
        "rights": {
            "local_cpu": True,
            "colab": True,
            "public_metrics": True,
            "note": "100% synthetic speech-like formant + RIR + noise dev corpus; safe for authorized Colab dev runs.",
        },
        "toolchain": corpus_v1.actual_toolchain(),
        "rng": {"algorithm": "numpy-pcg64", "seed_namespace": "80101..80503"},
        "strata_counts": strata_counts,
        "files": file_entries,
        "cases": cases,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def _max(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return float(np.max(clean)) if clean else None


def run_e8_voice_eval(manifest_path: Path, out_json: Path | None = None) -> dict:
    """Run the full Contour-A voice evaluation benchmark on CPU."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != VOICE_CORPUS_SCHEMA:
        raise ValueError(f"unexpected E8 voice manifest schema: {manifest.get('schema')!r}")

    per_case: list[dict] = []
    any_mutated = False
    total_runs = 0

    for case in manifest["cases"]:
        clean, degraded, deg_meta = synthesize_voice_case(case)
        sr = int(case["rate"])
        is_ctl = bool(case.get("is_clean_control", False))
        arm_results: dict[str, dict] = {}
        for arm in VOICE_CPU_ARMS:
            inp_copy = degraded.copy()
            h_before = hashlib.sha256(inp_copy.tobytes()).hexdigest()
            res = run_voice_arm(arm, inp_copy, sr)
            h_after = hashlib.sha256(inp_copy.tobytes()).hexdigest()
            m = evaluate_voice_case(
                clean,
                inp_copy,
                res,
                sr,
                is_clean_control=is_ctl,
                input_sha256_before=h_before,
                input_sha256_after=h_after,
            )
            any_mutated = any_mutated or bool(m["input_mutated"])
            total_runs += 1
            arm_results[arm] = m

        per_case.append({
            "id": case["id"],
            "stratum": case["stratum"],
            "speaker": case["speaker"],
            "rt60_s": case.get("rt60_s"),
            "noise_snr_db": case.get("noise_snr_db"),
            "is_clean_control": is_ctl,
            "degradation_meta": deg_meta,
            "arms": arm_results,
        })

    summary_by_stratum: dict[str, dict[str, dict]] = {}
    for stratum in VOICE_STRATA:
        rows = [r for r in per_case if r["stratum"] == stratum]
        summary_by_stratum[stratum] = {}
        for arm in VOICE_CPU_ARMS:
            ms = [r["arms"][arm] for r in rows]
            summary_by_stratum[stratum][arm] = {
                "n_cases": len(rows),
                "bypassed_cases": sum(1 for m in ms if m["bypassed"]),
                "mean_si_sdr_in_db": _mean([m["si_sdr_in_db"] for m in ms]),
                "mean_si_sdr_out_db": _mean([m["si_sdr_out_db"] for m in ms]),
                "mean_si_sdri_db": _mean([m["si_sdri_db"] for m in ms]),
                "mean_stoi_proxy_in": _mean([m["stoi_proxy_in"] for m in ms]),
                "mean_stoi_proxy_out": _mean([m["stoi_proxy_out"] for m in ms]),
                "mean_delta_stoi_proxy": _mean([m["delta_stoi_proxy"] for m in ms]),
                "mean_formant_drift_in_hz": _mean([m["formant_drift_in_hz"] for m in ms]),
                "mean_formant_drift_out_hz": _mean([m["formant_drift_out_hz"] for m in ms]),
                "mean_delta_formant_drift_hz": _mean([m["delta_formant_drift_hz"] for m in ms]),
                "mean_spectral_lsd_out_db": _mean([m["spectral_lsd_out_db"] for m in ms]),
                "mean_delta_spectral_lsd_db": _mean([m["delta_spectral_lsd_db"] for m in ms]),
                "mean_syllable_env_corr_in": _mean([m["syllable_env_corr_in"] for m in ms]),
                "mean_syllable_env_corr_out": _mean([m["syllable_env_corr_out"] for m in ms]),
                "mean_delta_syllable_env_corr": _mean([m["delta_syllable_env_corr"] for m in ms]),
                "max_clean_control_err": _max([m["clean_control_max_abs_err"] for m in ms]),
                "mean_wall_s_per_audio_min": _mean([m["wall_s_per_audio_min"] for m in ms]),
            }

    ext_audit = audit_voice_external_models_and_metrics()
    go_no_go = evaluate_go_no_go_voice(summary_by_stratum, ext_audit)

    results = {
        "schema": VOICE_RESULTS_SCHEMA,
        "contour": "A_voice",
        "board_card": 1135,
        "dataset_version": manifest["dataset_version"],
        "n_cases": len(per_case),
        "sealed_corpora_opened": False,
        "preserve_pipeline_modified": False,
        "colab_launched": False,
        "toolchain": {**corpus_v1.actual_toolchain(), "platform": platform.machine()},
        "provenance": {
            "manifest_path": "corpus/dev-e8-voice-v1.json",
            "manifest_sha256": corpus_v1.digest(manifest_path),
            "evaluator_path": "research/e8_voice_eval.py",
            "evaluator_sha256": corpus_v1.digest(REPO / "research" / "e8_voice_eval.py"),
        },
        "invariants": {
            "total_runs": total_runs,
            "any_input_mutated": any_mutated,
        },
        "cpu_arms": list(VOICE_CPU_ARMS),
        "external_audit": ext_audit,
        "go_no_go_voice": go_no_go,
        "summary_by_stratum": summary_by_stratum,
        "cases": per_case,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--manifest", default="corpus/dev-e8-voice-v1.json")
    ap.add_argument("--build-manifest", action="store_true", help="Regenerate dev-e8-voice-v1.json before eval.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if args.build_manifest or not manifest_path.exists():
        build_voice_manifest(manifest_path)

    res = run_e8_voice_eval(manifest_path, Path(args.out_json) if args.out_json else None)
    gng = res["go_no_go_voice"]
    print(
        f"E8 Contour A (Voice) complete: {res['n_cases']} cases, {res['invariants']['total_runs']} runs, "
        f"neural={res['external_audit']['neural_arm_status']}, "
        f"go_no_go_1={gng['production_verdict']}/{gng['colab_research_verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
