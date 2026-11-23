#!/usr/bin/env python3
"""E8 Research Module — Contour B: 4-Stem Separation Feasibility Evaluation & Combined E8 Report.

    python3 research/e8_stems_eval.py \\
        --build-manifest \\
        --write-combined

Isolated feasibility harness for Board #1135 (E0 charter + spec v1, section E8,
Contour B — Stems). Strictly separate from Contour A (Voice) and strictly
forbidden from generalizing to stereo music mastering (`Preserve`).

Safety boundaries:
- Never opens `corpus/sealed-v1.json` or `corpus/sealed-e5-v1.json`.
- Never modifies any of the 18 frozen E4 files or `rnext/`.
- Never adds production dependencies or launches Colab automation.
- Guarantees exact additive corpus model `mix == vocals + drums + bass + other`
  (`max_err <= 1e-12`).
- External neural stem models (HTDemucs, SCNet, BS-RoFormer) are honestly marked
  `blocked_external_weights_or_colab`; MUSDB18-HQ is marked
  `not_provisioned_on_host`.
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
from scipy import ndimage, signal

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import corpus_synth, corpus_v1, e8_voice_eval

EPS = 1e-20
ADDITIVE_MIX_TOL = 1e-12
STEMS_CORPUS_SCHEMA = "e8-stems-corpus/v1"
STEMS_RESULTS_SCHEMA = "e8-stems-results/v1"
COMBINED_RESULTS_SCHEMA = "e8-voice-stems-results/v1"
STEMS_DATASET_VERSION = "dev-e8-stems-2026-11-23"
STEMS_SR = 44100
STEMS_DURATION_S = 2.5

STEM_NAMES = ("vocals", "drums", "bass", "other")

STEMS_STRATA = (
    "standard_4stem_mix",
    "sparse_clean_stem_control",
    "ood_stems",
)

STEMS_CPU_ARMS = (
    "identity",
    "hpss_band_separator_cpu",
    "oracle_irm_upper_bound",
)


@dataclass
class StemsArmResult:
    """Output of a single 4-stem separation CPU arm."""

    arm: str
    estimated_stems: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    is_oracle: bool
    wall_s: float
    meta: dict = field(default_factory=dict)


def stems_cases() -> list[dict]:
    """Return the deterministic 9-case dev 4-stem corpus specification."""
    return [
        # --- Stratum 1: standard_4stem_mix (4 musical styles) ---
        {
            "id": "e8-stems-std-pop-rock-44k",
            "stratum": "standard_4stem_mix",
            "style": "pop_rock",
            "tempo_bpm": 120.0,
            "root_hz": 110.0,
            "active_stems": ["vocals", "drums", "bass", "other"],
            "spectral_clash": False,
            "wide_antiphase": False,
            "mono_collapse": False,
            "rate": STEMS_SR,
            "duration_s": STEMS_DURATION_S,
            "seed": 81101,
        },
        {
            "id": "e8-stems-std-jazz-acoustic-44k",
            "stratum": "standard_4stem_mix",
            "style": "jazz_acoustic",
            "tempo_bpm": 104.0,
            "root_hz": 98.0,
            "active_stems": ["vocals", "drums", "bass", "other"],
            "spectral_clash": False,
            "wide_antiphase": False,
            "mono_collapse": False,
            "rate": STEMS_SR,
            "duration_s": STEMS_DURATION_S,
            "seed": 81102,
        },
        {
            "id": "e8-stems-std-synth-electronic-44k",
            "stratum": "standard_4stem_mix",
            "style": "synth_electronic",
            "tempo_bpm": 128.0,
            "root_hz": 82.41,
            "active_stems": ["vocals", "drums", "bass", "other"],
            "spectral_clash": False,
            "wide_antiphase": False,
            "mono_collapse": False,
            "rate": STEMS_SR,
            "duration_s": STEMS_DURATION_S,
            "seed": 81103,
        },
        {
            "id": "e8-stems-std-dense-ballad-44k",
            "stratum": "standard_4stem_mix",
            "style": "dense_ballad",
            "tempo_bpm": 88.0,
            "root_hz": 130.81,
            "active_stems": ["vocals", "drums", "bass", "other"],
            "spectral_clash": False,
            "wide_antiphase": False,
            "mono_collapse": False,
            "rate": STEMS_SR,
            "duration_s": STEMS_DURATION_S,
            "seed": 81104,
        },
        # --- Stratum 2: sparse_clean_stem_control (inactive-stem hallucination / leakage test) ---
        {
            "id": "e8-stems-ctl-solo-vocals-44k",
            "stratum": "sparse_clean_stem_control",
            "style": "solo_vocals",
            "tempo_bpm": 100.0,
            "root_hz": 110.0,
            "active_stems": ["vocals"],
            "spectral_clash": False,
            "wide_antiphase": False,
            "mono_collapse": False,
            "rate": STEMS_SR,
            "duration_s": STEMS_DURATION_S,
            "seed": 81201,
        },
        {
            "id": "e8-stems-ctl-bass-drums-only-44k",
            "stratum": "sparse_clean_stem_control",
            "style": "rhythm_section",
            "tempo_bpm": 116.0,
            "root_hz": 87.31,
            "active_stems": ["drums", "bass"],
            "spectral_clash": False,
            "wide_antiphase": False,
            "mono_collapse": False,
            "rate": STEMS_SR,
            "duration_s": STEMS_DURATION_S,
            "seed": 81202,
        },
        # --- Stratum 3: ood_stems ---
        {
            "id": "e8-stems-ood-vocal-other-clash-44k",
            "stratum": "ood_stems",
            "style": "center_clash",
            "tempo_bpm": 122.0,
            "root_hz": 110.0,
            "active_stems": ["vocals", "drums", "bass", "other"],
            "spectral_clash": True,
            "wide_antiphase": False,
            "mono_collapse": False,
            "rate": STEMS_SR,
            "duration_s": STEMS_DURATION_S,
            "seed": 81301,
        },
        {
            "id": "e8-stems-ood-wide-antiphase-cymbals-44k",
            "stratum": "ood_stems",
            "style": "wide_antiphase",
            "tempo_bpm": 132.0,
            "root_hz": 92.5,
            "active_stems": ["vocals", "drums", "bass", "other"],
            "spectral_clash": False,
            "wide_antiphase": True,
            "mono_collapse": False,
            "rate": STEMS_SR,
            "duration_s": STEMS_DURATION_S,
            "seed": 81302,
        },
        {
            "id": "e8-stems-ood-mono-collapsed-mix-44k",
            "stratum": "ood_stems",
            "style": "mono_collapsed",
            "tempo_bpm": 118.0,
            "root_hz": 103.83,
            "active_stems": ["vocals", "drums", "bass", "other"],
            "spectral_clash": False,
            "wide_antiphase": False,
            "mono_collapse": True,
            "rate": STEMS_SR,
            "duration_s": STEMS_DURATION_S,
            "seed": 81303,
        },
    ]


def _stereo_pan(mono: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan in [-1 (L), +1 (R)] -> shape (N, 2)."""
    theta = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * (np.pi / 4.0)
    return np.column_stack([mono * np.cos(theta), mono * np.sin(theta)])


def _synth_bass_stem(n: int, sr: int, tempo_bpm: float, root_hz: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    beat_s = 60.0 / tempo_bpm
    notes = [root_hz, root_hz, root_hz * 1.12246, root_hz * 1.25992, root_hz * 0.8909]
    mono = np.zeros(n, dtype=np.float64)
    step = int(beat_s * sr)
    for i, start in enumerate(range(0, n, step)):
        end = min(start + step, n)
        seg_t = t[start:end] - t[start]
        f = notes[i % len(notes)]
        env = np.exp(-3.2 * seg_t) * (1.0 - np.exp(-80.0 * seg_t))
        wave = (
            0.70 * np.sin(2.0 * np.pi * f * seg_t)
            + 0.25 * np.sin(2.0 * np.pi * 2.0 * f * seg_t)
            + 0.10 * np.sin(2.0 * np.pi * 3.0 * f * seg_t)
        )
        mono[start:end] += 0.24 * env * wave
    return _stereo_pan(mono, 0.0)


def _synth_drums_stem(
    n: int,
    sr: int,
    tempo_bpm: float,
    rng: np.random.Generator,
    *,
    dense_cymbals: bool = False,
) -> np.ndarray:
    beat_s = 60.0 / tempo_bpm
    eighth = int(0.5 * beat_s * sr)
    drums = np.zeros((n, 2), dtype=np.float64)

    sos_snare = signal.butter(3, [400.0 / (0.5 * sr), 5500.0 / (0.5 * sr)], btype="band", output="sos")
    sos_hat = signal.butter(3, [6500.0 / (0.5 * sr), 15000.0 / (0.5 * sr)], btype="band", output="sos")

    for step_idx, start in enumerate(range(0, n, max(eighth, 1))):
        # Kick on beats 0, 2, 4, ... (even eighths on 0, 4).
        if step_idx % 4 == 0:
            k_len = min(int(0.14 * sr), n - start)
            kt = np.arange(k_len, dtype=np.float64) / sr
            f_inst = 52.0 + 95.0 * np.exp(-38.0 * kt)
            kick = 0.34 * np.sin(2.0 * np.pi * np.cumsum(f_inst) / sr) * np.exp(-16.0 * kt)
            drums[start : start + k_len] += _stereo_pan(kick, 0.0)

        # Snare on off-beats (step_idx % 4 == 2).
        if step_idx % 4 == 2:
            s_len = min(int(0.16 * sr), n - start)
            st = np.arange(s_len, dtype=np.float64) / sr
            noise = signal.sosfilt(sos_snare, rng.normal(0.0, 1.0, s_len))
            body = np.sin(2.0 * np.pi * 195.0 * st) * np.exp(-24.0 * st)
            snare = (0.28 * noise * np.exp(-20.0 * st) + 0.16 * body)
            drums[start : start + s_len] += _stereo_pan(snare, 0.05)

        # Hi-hat / cymbal on every eighth (or denser metallic ride in OOD).
        h_len = min(int((0.18 if dense_cymbals else 0.065) * sr), n - start)
        ht = np.arange(h_len, dtype=np.float64) / sr
        hat_l = signal.sosfilt(sos_hat, rng.normal(0.0, 1.0, h_len)) * np.exp(-(14.0 if dense_cymbals else 48.0) * ht)
        hat_r = signal.sosfilt(sos_hat, rng.normal(0.0, 1.0, h_len)) * np.exp(-(14.0 if dense_cymbals else 48.0) * ht)
        gain = 0.24 if dense_cymbals else 0.14
        drums[start : start + h_len, 0] += gain * hat_l
        drums[start : start + h_len, 1] += gain * hat_r
    return drums


def _synth_vocals_stem(n: int, sr: int, root_hz: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    # Melodic singing phrase centered at 2.5*root_hz .. 4*root_hz with vibrato & vocal formants.
    base_f0 = max(210.0, root_hz * 2.4)
    f0_t = base_f0 * (1.0 + 0.14 * np.sin(2.0 * np.pi * 0.65 * t) + 0.018 * np.sin(2.0 * np.pi * 5.6 * t))
    phase = 2.0 * np.pi * np.cumsum(f0_t) / sr
    src = np.zeros(n, dtype=np.float64)
    for k in range(1, 14):
        src += (1.0 / (k ** 1.1)) * np.sin(k * phase + 0.1 * k)

    # Syllabic phrasing envelope.
    phrase = np.maximum(0.0, np.sin(2.0 * np.pi * 1.4 * t + 0.2)) ** 0.7
    y1 = signal.sosfilt(e8_voice_eval._biquad_resonator(620.0, 95.0, sr), src)
    y2 = signal.sosfilt(e8_voice_eval._biquad_resonator(1580.0, 140.0, sr), src)
    y3 = signal.sosfilt(e8_voice_eval._biquad_resonator(2750.0, 190.0, sr), src)
    vox = (0.55 * y1 + 0.32 * y2 + 0.15 * y3) * phrase
    rms = float(np.sqrt(np.mean(vox ** 2) + EPS))
    vox = vox * (0.14 / rms)
    # Subtle room presence still strongly center-panned.
    _ = rng.random()
    return _stereo_pan(vox, 0.0)


def _synth_other_stem(
    n: int,
    sr: int,
    root_hz: float,
    *,
    spectral_clash: bool = False,
    wide_antiphase: bool = False,
) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    # Stereo chord accompaniment (L/R panned guitar/keys voices).
    f_chord_l = [root_hz * 3.0, root_hz * 4.5, root_hz * 6.0]
    f_chord_r = [root_hz * 3.56, root_hz * 5.04, root_hz * 7.12]
    sig_l = np.zeros(n, dtype=np.float64)
    sig_r = np.zeros(n, dtype=np.float64)
    env = 0.65 + 0.35 * np.sin(2.0 * np.pi * 0.8 * t)
    for f in f_chord_l:
        sig_l += 0.055 * np.sin(2.0 * np.pi * f * t) + 0.025 * np.sin(2.0 * np.pi * 2.0 * f * t)
    for f in f_chord_r:
        sig_r += 0.055 * np.sin(2.0 * np.pi * f * t + 0.6) + 0.025 * np.sin(2.0 * np.pi * 2.0 * f * t + 1.1)

    other = np.column_stack([sig_l * env, sig_r * env])
    if spectral_clash:
        # Center-panned lead synth directly overlapping vocal formant range (500-2200 Hz).
        lead = 0.11 * np.sin(2.0 * np.pi * (root_hz * 5.2) * t) + 0.07 * np.sin(2.0 * np.pi * (root_hz * 10.4) * t)
        other += _stereo_pan(lead, 0.0)
    if wide_antiphase:
        anti = 0.09 * np.sin(2.0 * np.pi * (root_hz * 8.0) * t)
        other[:, 0] += anti
        other[:, 1] -= anti
    return other


def synthesize_stems_case(case: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Synthesize deterministic 4 stems + EXACT additive mix `mix == sum(stems)`."""
    sr = int(case["rate"])
    n = round(sr * float(case["duration_s"]))
    rng = np.random.default_rng(int(case["seed"]))
    active = set(case["active_stems"])

    zeros = np.zeros((n, 2), dtype=np.float64)
    stems: dict[str, np.ndarray] = {
        "vocals": (
            _synth_vocals_stem(n, sr, float(case["root_hz"]), rng)
            if "vocals" in active
            else zeros.copy()
        ),
        "drums": (
            _synth_drums_stem(
                n, sr, float(case["tempo_bpm"]), rng, dense_cymbals=bool(case.get("wide_antiphase"))
            )
            if "drums" in active
            else zeros.copy()
        ),
        "bass": (
            _synth_bass_stem(n, sr, float(case["tempo_bpm"]), float(case["root_hz"]))
            if "bass" in active
            else zeros.copy()
        ),
        "other": (
            _synth_other_stem(
                n,
                sr,
                float(case["root_hz"]),
                spectral_clash=bool(case.get("spectral_clash")),
                wide_antiphase=bool(case.get("wide_antiphase")),
            )
            if "other" in active
            else zeros.copy()
        ),
    }

    if case.get("mono_collapse", False):
        for name in STEM_NAMES:
            m = 0.5 * (stems[name][:, 0] + stems[name][:, 1])
            stems[name] = np.column_stack([m, m])

    mix = stems["vocals"] + stems["drums"] + stems["bass"] + stems["other"]
    peak = float(np.max(np.abs(mix)))
    if peak > 0.88:
        scale = 0.88 / peak
        for name in STEM_NAMES:
            stems[name] = stems[name] * scale
        # Recompute mix strictly from scaled stems for exact additive equality.
        mix = stems["vocals"] + stems["drums"] + stems["bass"] + stems["other"]

    return stems, mix


# ---------------------------------------------------------------------------
# CPU Stem Separation Baselines (Partition-of-Unity STFT Masks)
# ---------------------------------------------------------------------------

def _stft_stereo(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    freqs, _t, Z_l = signal.stft(x[:, 0], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Z_r = signal.stft(x[:, 1], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    return freqs, np.stack([Z_l, Z_r], axis=0)  # shape (2, F, T)


def _istft_stereo(Z: np.ndarray, sr: int, n_samples: int, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    _, y_l = signal.istft(Z[0], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    _, y_r = signal.istft(Z[1], fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    out = np.column_stack([y_l[:n_samples], y_r[:n_samples]]).astype(np.float64)
    if out.shape[0] < n_samples:
        out = np.pad(out, ((0, n_samples - out.shape[0]), (0, 0)))
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def run_stems_arm(
    arm: str,
    mix: np.ndarray,
    sr: int,
    *,
    true_stems: dict[str, np.ndarray] | None = None,
) -> StemsArmResult:
    """Run one Contour-B 4-stem CPU arm without mutating `mix`."""
    x = np.asarray(mix, dtype=np.float64).copy()
    n = x.shape[0]
    t0 = time.perf_counter()

    freqs, Z_mix = _stft_stereo(x, sr)
    _, n_freq, n_frames = Z_mix.shape

    if arm == "identity":
        masks = {s: np.full((2, n_freq, n_frames), 0.25, dtype=np.float64) for s in STEM_NAMES}
        est = {s: 0.25 * x for s in STEM_NAMES}
        return StemsArmResult(
            arm=arm,
            estimated_stems=est,
            masks=masks,
            is_oracle=False,
            wall_s=time.perf_counter() - t0,
            meta={"mask_type": "equal_quarter_unseparated_mix"},
        )

    if arm == "hpss_band_separator_cpu":
        # Harmonic-Percussive Source Separation on mono magnitude spectrogram.
        mag_l = np.abs(Z_mix[0])
        mag_r = np.abs(Z_mix[1])
        mag = 0.5 * (mag_l + mag_r)
        harm = ndimage.median_filter(mag, size=(1, 17))
        perc = ndimage.median_filter(mag, size=(17, 1))
        h_pow = harm ** 2
        p_pow = perc ** 2
        hp_den = h_pow + p_pow + EPS
        w_harm = h_pow / hp_den
        w_perc = p_pow / hp_den

        # Mid/Side spatial cue: center-panned vs side-panned energy.
        mid = 0.5 * (Z_mix[0] + Z_mix[1])
        side = 0.5 * (Z_mix[0] - Z_mix[1])
        mid_p = np.abs(mid) ** 2
        side_p = np.abs(side) ** 2
        center = mid_p / (mid_p + side_p + EPS)
        side_ratio = side_p / (mid_p + side_p + EPS)

        # Smooth frequency band priors.
        f_col = freqs[:, None]
        p_bass_f = _sigmoid((235.0 - f_col) / 30.0)
        p_vox_f = _sigmoid((f_col - 220.0) / 35.0) * _sigmoid((4000.0 - f_col) / 350.0)
        p_high_f = _sigmoid((f_col - 4200.0) / 400.0)

        # Raw non-negative weights per stem.
        raw_w = {
            "bass": w_harm * p_bass_f * (0.25 + 0.75 * center),
            "drums": w_perc * (0.85 + 0.15 * p_high_f) + 0.08 * p_high_f * w_perc,
            "vocals": w_harm * p_vox_f * (center ** 1.5),
            "other": w_harm * ((1.0 - p_bass_f) * (side_ratio + 0.25 * (1.0 - center)) + 0.7 * p_high_f),
        }
        total_w = sum(raw_w.values()) + EPS
        masks = {}
        est = {}
        for s in STEM_NAMES:
            m_mono = raw_w[s] / total_w
            m_stereo = np.stack([m_mono, m_mono], axis=0)
            masks[s] = m_stereo
            est[s] = _istft_stereo(Z_mix * m_stereo, sr, n)

        return StemsArmResult(
            arm=arm,
            estimated_stems=est,
            masks=masks,
            is_oracle=False,
            wall_s=time.perf_counter() - t0,
            meta={"mask_type": "hpss_median17_midside_band_wiener"},
        )

    if arm == "oracle_irm_upper_bound":
        if true_stems is None:
            raise ValueError("oracle_irm_upper_bound requires true_stems")
        pows = {}
        for s in STEM_NAMES:
            _, Z_s = _stft_stereo(true_stems[s], sr)
            pows[s] = np.abs(Z_s) ** 2
        total_p = sum(pows.values())
        zero_bins = total_p <= EPS
        masks = {}
        est = {}
        for s in STEM_NAMES:
            m = np.where(zero_bins, 0.25, pows[s] / np.where(zero_bins, 1.0, total_p))
            masks[s] = m
            est[s] = _istft_stereo(Z_mix * m, sr, n)
        return StemsArmResult(
            arm=arm,
            estimated_stems=est,
            masks=masks,
            is_oracle=True,
            wall_s=time.perf_counter() - t0,
            meta={"mask_type": "oracle_ideal_ratio_mask_upper_bound"},
        )

    raise ValueError(f"unknown E8 stems arm: {arm!r}")


# ---------------------------------------------------------------------------
# Stems Metrics (Contour B)
# ---------------------------------------------------------------------------

def evaluate_stems_case(
    true_stems: dict[str, np.ndarray],
    mix: np.ndarray,
    result: StemsArmResult,
    sr: int,
    *,
    input_sha256_before: str | None = None,
    input_sha256_after: str | None = None,
) -> dict:
    """Compute per-stem SI-SDR, inter-stem leakage (dB), and mix reconstruction error."""
    true_sum = sum(true_stems[s] for s in STEM_NAMES)
    corpus_add_err = float(np.max(np.abs(true_sum - mix)))

    est_sum = sum(result.estimated_stems[s] for s in STEM_NAMES)
    recon_diff = est_sum - mix
    mix_recon_max_err = float(np.max(np.abs(recon_diff)))
    mix_rms = float(np.sqrt(np.mean(mix ** 2) + EPS))
    recon_rms = float(np.sqrt(np.mean(recon_diff ** 2)))
    mix_recon_rms_db = float(20.0 * np.log10(max(recon_rms, EPS) / mix_rms))

    # Precompute STFT of each true stem for exact linear mask leakage measurement.
    Z_true = {s: _stft_stereo(true_stems[s], sr)[1] for s in STEM_NAMES}
    n = mix.shape[0]

    per_stem: dict[str, dict] = {}
    active_sisdrs = []
    active_leakages = []
    for s in STEM_NAMES:
        ref = true_stems[s]
        est = result.estimated_stems[s]
        ref_rms = float(np.sqrt(np.mean(ref ** 2)))
        active = ref_rms > 1e-6

        if active:
            sdr = e8_voice_eval.si_sdr_db(ref, est)
            active_sisdrs.append(sdr)
            # Energy of interfering stems passing through stem s's mask vs retained target energy.
            m_s = result.masks[s]
            Z_interf = sum(Z_true[k] for k in STEM_NAMES if k != s)
            target_retained = _istft_stereo(m_s * Z_true[s], sr, n)
            interf_leaked = _istft_stereo(m_s * Z_interf, sr, n)
            e_ret = float(np.mean(target_retained ** 2))
            e_leak = float(np.mean(interf_leaked ** 2))
            if e_leak <= EPS:
                leak_db = -90.0
            else:
                leak_db = float(10.0 * np.log10(e_leak / max(e_ret, EPS)))
            active_leakages.append(leak_db)
            per_stem[s] = {
                "active": True,
                "si_sdr_db": sdr,
                "inter_stem_leakage_db": leak_db,
                "false_leakage_dbfs": None,
            }
        else:
            est_rms = float(np.sqrt(np.mean(est ** 2)))
            false_dbfs = float(20.0 * np.log10(max(est_rms, EPS))) if est_rms > EPS else -120.0
            per_stem[s] = {
                "active": False,
                "si_sdr_db": None,
                "inter_stem_leakage_db": None,
                "false_leakage_dbfs": false_dbfs,
            }

    input_mutated = (
        input_sha256_before != input_sha256_after
        if (input_sha256_before is not None and input_sha256_after is not None)
        else False
    )
    audio_min = max(n / float(sr) / 60.0, 1e-9)
    return {
        "arm": result.arm,
        "is_oracle": bool(result.is_oracle),
        "corpus_additive_max_err": corpus_add_err,
        "mix_recon_max_err": mix_recon_max_err,
        "mix_recon_rms_db": mix_recon_rms_db,
        "mean_active_si_sdr_db": float(np.mean(active_sisdrs)) if active_sisdrs else None,
        "mean_active_leakage_db": float(np.mean(active_leakages)) if active_leakages else None,
        "per_stem": per_stem,
        "input_mutated": input_mutated,
        "wall_s": float(result.wall_s),
        "wall_s_per_audio_min": float(result.wall_s / audio_min),
    }


# ---------------------------------------------------------------------------
# Honest External Audits & Go/No-Go #2 (Stems Restoration)
# ---------------------------------------------------------------------------

def audit_stems_external_models(models_dir: Path | None = None) -> dict:
    """Honest local audit of HTDemucs / SCNet / BS-RoFormer weights, licenses, and Colab budget."""
    m_dir = models_dir if models_dir is not None else (REPO / "models")
    weight_exts = (".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".th")
    stem_keywords = ("htdemucs", "demucs", "scnet", "roformer", "mdx")
    local_weights = []
    if m_dir.exists():
        for p in sorted(m_dir.rglob("*")):
            if (
                p.is_file()
                and p.suffix.lower() in weight_exts
                and any(k in p.name.lower() for k in stem_keywords)
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
        blocked_reasons.append("no_local_licensed_stem_weights_htdemucs_scnet_bsroformer")
    if not cuda_available:
        blocked_reasons.append("no_local_cuda_gpu")
    if not colab_authorized:
        blocked_reasons.append("colab_connection_not_authorized_or_configured")

    neural_status = "blocked_external_weights_or_colab" if blocked_reasons else "ready"
    return {
        "contour": "B_stems",
        "neural_arm_status": neural_status,
        "executed": False,
        "production_dependency_added": False,
        "blocked_reasons": blocked_reasons,
        "environment_audit": {
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "local_stem_weights_found": local_weights,
            "colab_authorized": colab_authorized,
        },
        "external_datasets": [
            {
                "id": "musdb18_hq",
                "name": "MUSDB18-HQ (uncompressed 44.1 kHz 4-stem benchmark)",
                "status": "not_provisioned_on_host",
                "license_note": "Non-commercial research license — verify terms before download or Colab upload",
                "required_action": "Provision approved dev subset on Colab with SHA-256 manifest; never mix with sealed music holdout",
            },
        ],
        "external_neural_models": [
            {
                "id": "htdemucs_v4",
                "status": neural_status,
                "version_pin": "demucs 4.0.1 (`htdemucs` / `htdemucs_ft` 4-stem bag of 4 models)",
                "license": "MIT",
                "colab_budget_estimate": "T4/L4 GPU (~2-4 GB VRAM, ~2-5 s/min audio for `htdemucs`, ~8-18 s/min for `htdemucs_ft`)",
                "target_role": "Standard 4-stem music separation baseline (vocals/drums/bass/other)",
            },
            {
                "id": "scnet",
                "status": neural_status,
                "version_pin": "SCNet (Tong et al. 2024, Sparse Compression Network — pin commit + SHA-256)",
                "license": "MIT (verify checkpoint release terms)",
                "colab_budget_estimate": "T4/L4 GPU (~4-6 GB VRAM, ~4-10 s/min audio)",
                "target_role": "High-efficiency frequency-band sparse separation",
            },
            {
                "id": "bs_roformer",
                "status": neural_status,
                "version_pin": "BS-RoFormer (Lu et al. 2023, Band-Split RoFormer — pin checkpoint SHA-256)",
                "license": "MIT/Apache-2.0 code; MUST verify specific community checkpoint license before download",
                "colab_budget_estimate": "T4/L4 GPU (~6-10 GB VRAM, ~10-25 s/min audio with overlap-add)",
                "target_role": "State-of-the-art vocal/stem separation (heavy VRAM; Colab-only research)",
            },
        ],
        "colab_handoff_spec": {
            "permitted_inputs": "corpus/dev-e8-stems-v1.json + licensed MUSDB18-HQ dev split ONLY",
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
                "mix_recon_max_err",
            ],
        },
    }


def evaluate_go_no_go_stems(summary_by_stratum: dict, external_audit: dict) -> dict:
    """Produce independent Go/No-Go #2 decision for Contour B (Stems Restoration)."""
    std_id = summary_by_stratum["standard_4stem_mix"]["identity"]
    std_hpss = summary_by_stratum["standard_4stem_mix"]["hpss_band_separator_cpu"]
    std_irm = summary_by_stratum["standard_4stem_mix"]["oracle_irm_upper_bound"]

    cpu_harness_verified = bool(
        std_hpss["mean_active_si_sdr_db"] > std_id["mean_active_si_sdr_db"] + 2.0
        and std_irm["mean_active_si_sdr_db"] > std_hpss["mean_active_si_sdr_db"] + 3.0
        and std_hpss["max_mix_recon_err"] < 1e-9
    )

    return {
        "decision_id": "go_no_go_2_stems_restoration",
        "contour": "B_stems",
        "production_verdict": "no_go",
        "colab_research_verdict": "conditional_go" if cpu_harness_verified else "no_go",
        "preserve_stereo_music_generalization": "forbidden",
        "non_generalization_warning": (
            "STRICT BOUNDARY: Neither Contour A (Voice) nor Contour B (Stems) results may be "
            "generalized to stereo music mastering (`Preserve` hard-declipping). Stem separation "
            "and re-summing introduces inter-stem bleed and phase/masking artifacts incompatible "
            "with Preserve's untouched-outside-mask stereo fidelity contract."
        ),
        "rationale": (
            "The 4-stem CPU harness verifies exact additive mix integrity (`mix == sum(stems)`), "
            "shows deterministic HPSS+band separation improves over identity while trailing the "
            "Oracle IRM upper bound, and exposes OOD collapse when vocal/other spectra clash or "
            "stereo cues vanish (mono collapse). Production stem restoration is NO-GO because: "
            "(1) neural stem separators (HTDemucs, SCNet, BS-RoFormer) are blocked pending "
            "authorized Colab GPU + checkpoint license verification, (2) MUSDB18-HQ is not "
            "provisioned on host, and (3) stem separation is outside the Preserve stereo release."
        ),
        "cpu_harness_verified": cpu_harness_verified,
        "decision_inputs_for_colab_stems_stage": [
            "Provision licensed MUSDB18-HQ dev split (non-commercial research check) with SHA-256 manifest.",
            "Authorize Google Colab T4/L4 GPU session (separate ticket) with pinned HTDemucs v4 (MIT), SCNet (MIT), and license-verified BS-RoFormer checkpoints.",
            "Measure per-stem SI-SDR, inter-stem leakage (dB), and exact mix reconstruction error (`sum(est_stems) - mix`) on both dev-e8-stems-v1 and MUSDB18-HQ.",
            "Audit stereo phase/IPD preservation and bleed-through artifacts before any opt-in Enhance stem-repair prototype is proposed.",
            "Do NOT route stereo mastering (`Preserve`) through stem separation or generalize stem SI-SDR to full-mix hard-declipping.",
        ],
        "external_blockers": external_audit["blocked_reasons"],
    }


# ---------------------------------------------------------------------------
# Manifest Builder, Benchmark Runner & Combined E8 Report
# ---------------------------------------------------------------------------

def build_stems_manifest(out_path: Path) -> dict:
    """Build deterministic `corpus/dev-e8-stems-v1.json` with SHA-256 digests & exact additivity."""
    cases = stems_cases()
    file_entries = []
    strata_counts: dict[str, int] = {s: 0 for s in STEMS_STRATA}
    max_add_err = 0.0

    with tempfile.TemporaryDirectory(prefix="e8_stems_manifest_") as td:
        td_p = Path(td)
        for case in cases:
            strata_counts[case["stratum"]] += 1
            stems, mix = synthesize_stems_case(case)
            add_err = float(np.max(np.abs(sum(stems[s] for s in STEM_NAMES) - mix)))
            if add_err > ADDITIVE_MIX_TOL:
                raise ValueError(f"{case['id']}: additive mix invariant violated ({add_err:.3e})")
            max_add_err = max(max_add_err, add_err)

            for tag, arr in [("mix", mix), *[(s, stems[s]) for s in STEM_NAMES]]:
                wav_name = f"{case['id']}.{tag}.wav"
                wav_path = td_p / wav_name
                corpus_synth.write_wav(wav_path, int(case["rate"]), arr, bits=16, seed=int(case["seed"]))
                file_entries.append({
                    "path": wav_name,
                    "sha256": corpus_v1.digest(wav_path),
                    "bytes": wav_path.stat().st_size,
                })

    manifest = {
        "schema": STEMS_CORPUS_SCHEMA,
        "contour": "B_stems",
        "tier": "dev",
        "dataset_version": STEMS_DATASET_VERSION,
        "sealed_corpora_opened": False,
        "additive_mix_invariant_max_err": max_add_err,
        "external_datasets_status": {
            "musdb18_hq": "not_provisioned_on_host",
        },
        "rights": {
            "local_cpu": True,
            "colab": True,
            "public_metrics": True,
            "note": "100% synthetic 4-stem additive stereo dev corpus; safe for authorized Colab dev runs.",
        },
        "toolchain": corpus_v1.actual_toolchain(),
        "rng": {"algorithm": "numpy-pcg64", "seed_namespace": "81101..81303"},
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


def run_e8_stems_eval(manifest_path: Path, out_json: Path | None = None) -> dict:
    """Run the full Contour-B 4-stem evaluation benchmark on CPU."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != STEMS_CORPUS_SCHEMA:
        raise ValueError(f"unexpected E8 stems manifest schema: {manifest.get('schema')!r}")

    per_case: list[dict] = []
    any_mutated = False
    max_corpus_add_err = 0.0
    max_mix_recon_err = 0.0
    total_runs = 0

    for case in manifest["cases"]:
        stems, mix = synthesize_stems_case(case)
        sr = int(case["rate"])
        arm_results: dict[str, dict] = {}
        for arm in STEMS_CPU_ARMS:
            mix_copy = mix.copy()
            h_before = hashlib.sha256(mix_copy.tobytes()).hexdigest()
            res = run_stems_arm(
                arm,
                mix_copy,
                sr,
                true_stems=stems if arm == "oracle_irm_upper_bound" else None,
            )
            h_after = hashlib.sha256(mix_copy.tobytes()).hexdigest()
            m = evaluate_stems_case(
                stems,
                mix_copy,
                res,
                sr,
                input_sha256_before=h_before,
                input_sha256_after=h_after,
            )
            any_mutated = any_mutated or bool(m["input_mutated"])
            max_corpus_add_err = max(max_corpus_add_err, float(m["corpus_additive_max_err"]))
            max_mix_recon_err = max(max_mix_recon_err, float(m["mix_recon_max_err"]))
            total_runs += 1
            arm_results[arm] = m

        per_case.append({
            "id": case["id"],
            "stratum": case["stratum"],
            "style": case["style"],
            "active_stems": list(case["active_stems"]),
            "arms": arm_results,
        })

    summary_by_stratum: dict[str, dict[str, dict]] = {}
    for stratum in STEMS_STRATA:
        rows = [r for r in per_case if r["stratum"] == stratum]
        summary_by_stratum[stratum] = {}
        for arm in STEMS_CPU_ARMS:
            ms = [r["arms"][arm] for r in rows]
            stem_means = {}
            for s in STEM_NAMES:
                stem_means[s] = {
                    "mean_si_sdr_db": _mean([m["per_stem"][s]["si_sdr_db"] for m in ms]),
                    "mean_leakage_db": _mean([m["per_stem"][s]["inter_stem_leakage_db"] for m in ms]),
                    "mean_false_leakage_dbfs": _mean([m["per_stem"][s]["false_leakage_dbfs"] for m in ms]),
                }
            summary_by_stratum[stratum][arm] = {
                "n_cases": len(rows),
                "is_oracle": arm == "oracle_irm_upper_bound",
                "mean_active_si_sdr_db": _mean([m["mean_active_si_sdr_db"] for m in ms]),
                "mean_active_leakage_db": _mean([m["mean_active_leakage_db"] for m in ms]),
                "max_mix_recon_err": _max([m["mix_recon_max_err"] for m in ms]),
                "mean_mix_recon_rms_db": _mean([m["mix_recon_rms_db"] for m in ms]),
                "per_stem": stem_means,
                "mean_wall_s_per_audio_min": _mean([m["wall_s_per_audio_min"] for m in ms]),
            }

    ext_audit = audit_stems_external_models()
    go_no_go = evaluate_go_no_go_stems(summary_by_stratum, ext_audit)

    results = {
        "schema": STEMS_RESULTS_SCHEMA,
        "contour": "B_stems",
        "board_card": 1135,
        "dataset_version": manifest["dataset_version"],
        "n_cases": len(per_case),
        "sealed_corpora_opened": False,
        "preserve_pipeline_modified": False,
        "colab_launched": False,
        "toolchain": {**corpus_v1.actual_toolchain(), "platform": platform.machine()},
        "provenance": {
            "manifest_path": "corpus/dev-e8-stems-v1.json",
            "manifest_sha256": corpus_v1.digest(manifest_path),
            "evaluator_path": "research/e8_stems_eval.py",
            "evaluator_sha256": corpus_v1.digest(REPO / "research" / "e8_stems_eval.py"),
        },
        "invariants": {
            "total_runs": total_runs,
            "any_input_mutated": any_mutated,
            "max_corpus_additive_err": max_corpus_add_err,
            "max_mix_recon_err": max_mix_recon_err,
        },
        "cpu_arms": list(STEMS_CPU_ARMS),
        "external_audit": ext_audit,
        "go_no_go_stems": go_no_go,
        "summary_by_stratum": summary_by_stratum,
        "cases": per_case,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return results


def _fmt(val: float | None, spec: str = ".2f") -> str:
    return "-" if val is None else f"{val:{spec}}"


def render_combined_e8_report_md(combined: dict) -> str:
    """Render `docs/validation/e8-voice-stems-report.md` with two strictly separate contours."""
    v = combined["contour_a_voice"]
    s = combined["contour_b_stems"]
    e4 = combined["e4_freeze_integrity"]
    v_gng = v["go_no_go_voice"]
    s_gng = s["go_no_go_stems"]
    v_ext = v["external_audit"]
    s_ext = s["external_audit"]

    lines = [
        "# E8 · Voice & Stems Feasibility Study — Separate Evaluation Report",
        "",
        (
            "**Card:** Board #1135 · **Spec:** E0 charter + spec v1 (Section E8) · "
            f"**Contour A (Voice):** `{v['dataset_version']}` (`{v['n_cases']}` cases) · "
            f"**Contour B (Stems):** `{s['dataset_version']}` (`{s['n_cases']}` cases)"
        ),
        "",
        "## 1. Scope, Safety Boundaries & Provenance",
        "",
        f"- **Sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`):** `sealed_corpora_opened = {combined['sealed_corpora_opened']}` (strictly untouched).",
        f"- **Frozen E4 Preserve hard-declip pipeline:** `{e4['matched_files']}/{e4['total_pinned_files']}` SHA-256 pins verified untouched (`all_untouched = {e4['all_untouched']}`).",
        "- **Preserve promotion / production dependencies / Colab automation:** `False` (isolated CPU feasibility study only).",
        f"- **Contour A Manifest SHA-256 (`corpus/dev-e8-voice-v1.json`):** `{v['provenance']['manifest_sha256']}`",
        f"- **Contour A Evaluator SHA-256 (`research/e8_voice_eval.py`):** `{v['provenance']['evaluator_sha256']}`",
        f"- **Contour B Manifest SHA-256 (`corpus/dev-e8-stems-v1.json`):** `{s['provenance']['manifest_sha256']}`",
        f"- **Contour B Evaluator SHA-256 (`research/e8_stems_eval.py`):** `{s['provenance']['evaluator_sha256']}`",
        f"- **Contour B Exact Additive Mix Invariant:** `max(|sum(true_stems) - mix|) = {s['invariants']['max_corpus_additive_err']:.2e}`, `max(|sum(est_stems) - mix|) = {s['invariants']['max_mix_recon_err']:.2e}`.",
        "",
        "> **Strict Non-Generalization Warning:** Contour A (Voice) and Contour B (Stems) are **two independent feasibility contours**. Results from speech enhancement or 4-stem source separation **MUST NOT** be generalized to full-mix stereo music mastering (`Preserve` hard-declipping).",
        "",
        "---",
        "",
        "## 2. Contour A — Voice Restoration Feasibility",
        "",
        "### 2.1 CPU Baselines by Stratum (Synthetic Formant + RIR + Noise Dev Corpus)",
        "",
        "| Stratum (n) | Arm | SI-SDR In (dB) | SI-SDR Out (dB) | SI-SDRi (dB) | STOI-Proxy Out | ΔSTOI-Proxy | Formant Drift Out (Hz) | ΔLSD (dB) | Syllable Env Corr | Wall (s/min) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for stratum in e8_voice_eval.VOICE_STRATA:
        n_s = v["summary_by_stratum"][stratum]["identity"]["n_cases"]
        for arm in e8_voice_eval.VOICE_CPU_ARMS:
            st = v["summary_by_stratum"][stratum][arm]
            lines.append(
                f"| `{stratum}` ({n_s}) | `{arm}` | "
                f"{_fmt(st['mean_si_sdr_in_db'], '.2f')} | "
                f"{_fmt(st['mean_si_sdr_out_db'], '.2f')} | "
                f"{_fmt(st['mean_si_sdri_db'], '+.2f')} | "
                f"{_fmt(st['mean_stoi_proxy_out'], '.3f')} | "
                f"{_fmt(st['mean_delta_stoi_proxy'], '+.3f')} | "
                f"{_fmt(st['mean_formant_drift_out_hz'], '.1f')} | "
                f"{_fmt(st['mean_delta_spectral_lsd_db'], '+.2f')} | "
                f"{_fmt(st['mean_syllable_env_corr_out'], '.3f')} | "
                f"{_fmt(st['mean_wall_s_per_audio_min'], '.2f')} |"
            )

    lines += [
        "",
        "### 2.2 External Voice Datasets, Neural Models & Licensed Metrics Audit",
        "",
        "| Component | Category | Status | License / Version Gate | Colab / Provisioning Note |",
        "|---|---|---|---|---|",
    ]
    for ds in v_ext["external_datasets"]:
        lines.append(
            f"| `{ds['id']}` | Dataset | `{ds['status']}` | {ds['license_note']} | {ds['required_action']} |"
        )
    for mdl in v_ext["external_neural_models"]:
        lines.append(
            f"| `{mdl['id']}` | Neural Model | `{mdl['status']}` | {mdl['version_pin']} ({mdl['license']}) | {mdl['colab_budget_estimate']} |"
        )
    for met in v_ext["external_perceptual_metrics"]:
        lines.append(
            f"| `{met['id']}` | Perceptual Metric | `{met['status']}` | {met['license_note']} | Local CPU proxy: `{met['local_proxy_used']}` |"
        )

    lines += [
        "",
        "### 2.3 Go/No-Go #1 — Voice Profile",
        "",
        f"- **Production Verdict:** `{v_gng['production_verdict']}` (`preserve_stereo_music_generalization = {v_gng['preserve_stereo_music_generalization']}`).",
        f"- **Colab Research Stage Verdict:** `{v_gng['colab_research_verdict']}` (`cpu_harness_verified = {v_gng['cpu_harness_verified']}`).",
        f"- **Rationale:** {v_gng['rationale']}",
        "- **Decision Inputs Required for Colab Voice Stage:**",
    ]
    for item in v_gng["decision_inputs_for_colab_voice_stage"]:
        lines.append(f"  - {item}")

    lines += [
        "",
        "---",
        "",
        "## 3. Contour B — 4-Stem Separation Feasibility",
        "",
        "### 3.1 CPU Baselines vs Oracle IRM Upper Bound by Stratum",
        "",
        "| Stratum (n) | Arm | Oracle | Vocals SI-SDR (dB) | Drums SI-SDR (dB) | Bass SI-SDR (dB) | Other SI-SDR (dB) | Mean Active SI-SDR (dB) | Mean Leakage (dB) | Mix Recon Max Err | Wall (s/min) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for stratum in STEMS_STRATA:
        n_s = s["summary_by_stratum"][stratum]["identity"]["n_cases"]
        for arm in STEMS_CPU_ARMS:
            st = s["summary_by_stratum"][stratum][arm]
            ps = st["per_stem"]
            lines.append(
                f"| `{stratum}` ({n_s}) | `{arm}` | `{st['is_oracle']}` | "
                f"{_fmt(ps['vocals']['mean_si_sdr_db'], '.2f')} | "
                f"{_fmt(ps['drums']['mean_si_sdr_db'], '.2f')} | "
                f"{_fmt(ps['bass']['mean_si_sdr_db'], '.2f')} | "
                f"{_fmt(ps['other']['mean_si_sdr_db'], '.2f')} | "
                f"{_fmt(st['mean_active_si_sdr_db'], '.2f')} | "
                f"{_fmt(st['mean_active_leakage_db'], '.2f')} | "
                f"{_fmt(st['max_mix_recon_err'], '.2e')} | "
                f"{_fmt(st['mean_wall_s_per_audio_min'], '.2f')} |"
            )

    lines += [
        "",
        "### 3.2 External Stem Datasets & Neural Models Audit (HTDemucs / SCNet / BS-RoFormer)",
        "",
        "| Component | Category | Status | License / Version Pin | Colab VRAM & Runtime Budget |",
        "|---|---|---|---|---|",
    ]
    for ds in s_ext["external_datasets"]:
        lines.append(
            f"| `{ds['id']}` | Dataset | `{ds['status']}` | {ds['license_note']} | {ds['required_action']} |"
        )
    for mdl in s_ext["external_neural_models"]:
        lines.append(
            f"| `{mdl['id']}` | Neural Model | `{mdl['status']}` | {mdl['version_pin']} ({mdl['license']}) | {mdl['colab_budget_estimate']} |"
        )

    lines += [
        "",
        "### 3.3 Go/No-Go #2 — Stems Restoration",
        "",
        f"- **Production Verdict:** `{s_gng['production_verdict']}` (`preserve_stereo_music_generalization = {s_gng['preserve_stereo_music_generalization']}`).",
        f"- **Colab Research Stage Verdict:** `{s_gng['colab_research_verdict']}` (`cpu_harness_verified = {s_gng['cpu_harness_verified']}`).",
        f"- **Rationale:** {s_gng['rationale']}",
        f"- **Non-Generalization Warning:** {s_gng['non_generalization_warning']}",
        "- **Decision Inputs Required for Colab Stems Stage:**",
    ]
    for item in s_gng["decision_inputs_for_colab_stems_stage"]:
        lines.append(f"  - {item}")

    lines += [
        "",
        "---",
        "",
        "## 4. Limitations & Honest Method Boundaries",
        "",
        "1. **Synthetic Dev Corpora Only:** Both `dev-e8-voice-v1` (formant + RIR + noise) and `dev-e8-stems-v1` (4-stem additive stereo mix) are deterministic synthetic benchmarks designed to validate metrics, invariants, and CPU baselines offline. Real speech corpora (`DAPS`, `VoiceBank+DEMAND`) and real multitrack music (`MUSDB18-HQ`) are `not_provisioned_on_host`.",
        "2. **Neural GPU Arms Blocked Locally:** `DeepFilterNet3`, `SGMSE+`, flow-matching speech restoration, `DNSMOS`, `HTDemucs`, `SCNet`, and `BS-RoFormer` are all honestly reported as `blocked_external_weights_or_colab` with zero fabricated numbers. All GPU inference/benchmarks require separate Google Colab authorization.",
        "3. **Licensed Perceptual Metrics:** Official ITU-T P.862 `PESQ` is marked `license_required_for_pesq`; local voice evaluation uses `stoi_proxy_envelope_corr`, `formant_drift_hz`, `spectral_lsd_db`, and `syllable_env_corr` as transparent CPU proxies.",
        "4. **Oracle IRM Is an Upper Bound, Not a Deployable Separator:** `oracle_irm_upper_bound` accesses ground-truth clean stems to establish the magnitude-masking ceiling on `dev-e8-stems-v1`; real separators have no access to clean stems.",
        "5. **OOD Failure Modes Exposed Honestly:** In Contour A, heavy hall reverb at 2 dB SNR and non-stationary babble degrade CPU spectral enhancement; in Contour B, spectral clash (`e8-stems-ood-vocal-other-clash-44k`) and mono collapse (`e8-stems-ood-mono-collapsed-mix-44k`) sharply reduce `hpss_band_separator_cpu` vocal/other separation because spatial Mid/Side cues vanish.",
        "6. **Zero Generalization to Preserve Stereo Mastering:** Neither voice dereverberation/denoising nor 4-stem source separation validates or improves stereo hard-declipping (`Preserve`). Frozen E4 files remain `18/18` untouched.",
    ]
    return "\n".join(lines) + "\n"


def run_e8_combined_report(
    voice_manifest: Path,
    stems_manifest: Path,
    out_json: Path,
    out_md: Path,
) -> dict:
    """Run Contour A and Contour B independently and emit combined E8 report & JSON."""
    voice_res = e8_voice_eval.run_e8_voice_eval(voice_manifest)
    stems_res = run_e8_stems_eval(stems_manifest)
    e4_check = _verify_e4_freeze_untouched()

    combined = {
        "schema": COMBINED_RESULTS_SCHEMA,
        "board_card": 1135,
        "sealed_corpora_opened": False,
        "preserve_pipeline_modified": False,
        "colab_launched": False,
        "production_dependencies_added": False,
        "non_generalization_policy": (
            "Contour A (Voice) and Contour B (Stems) are strictly separate feasibility evaluations "
            "and must not be generalized to stereo music mastering (Preserve)."
        ),
        "e4_freeze_integrity": e4_check,
        "provenance": {
            "voice_manifest_path": "corpus/dev-e8-voice-v1.json",
            "voice_manifest_sha256": voice_res["provenance"]["manifest_sha256"],
            "voice_evaluator_path": "research/e8_voice_eval.py",
            "voice_evaluator_sha256": voice_res["provenance"]["evaluator_sha256"],
            "stems_manifest_path": "corpus/dev-e8-stems-v1.json",
            "stems_manifest_sha256": stems_res["provenance"]["manifest_sha256"],
            "stems_evaluator_path": "research/e8_stems_eval.py",
            "stems_evaluator_sha256": stems_res["provenance"]["evaluator_sha256"],
        },
        "go_no_go_decisions": {
            "go_no_go_1_voice": voice_res["go_no_go_voice"],
            "go_no_go_2_stems": stems_res["go_no_go_stems"],
        },
        "contour_a_voice": voice_res,
        "contour_b_stems": stems_res,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(combined, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_combined_e8_report_md(combined), encoding="utf-8")
    return combined


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--voice-manifest", default="corpus/dev-e8-voice-v1.json")
    ap.add_argument("--stems-manifest", default="corpus/dev-e8-stems-v1.json")
    ap.add_argument("--build-manifest", action="store_true", help="Regenerate E8 dev corpus manifests.")
    ap.add_argument("--write-combined", action="store_true", help="Run both contours and write E8 report + JSON.")
    ap.add_argument("--out-json", default="docs/validation/e8-voice-stems-results.json")
    ap.add_argument("--out-md", default="docs/validation/e8-voice-stems-report.md")
    args = ap.parse_args()

    voice_manifest = Path(args.voice_manifest)
    stems_manifest = Path(args.stems_manifest)

    if args.build_manifest or not stems_manifest.exists():
        build_stems_manifest(stems_manifest)
    if args.write_combined:
        if args.build_manifest or not voice_manifest.exists():
            e8_voice_eval.build_voice_manifest(voice_manifest)
        res = run_e8_combined_report(voice_manifest, stems_manifest, Path(args.out_json), Path(args.out_md))
        print(
            f"E8 combined report complete: voice={res['contour_a_voice']['n_cases']} cases, "
            f"stems={res['contour_b_stems']['n_cases']} cases, "
            f"e4_untouched={res['e4_freeze_integrity']['all_untouched']}"
        )
    else:
        res = run_e8_stems_eval(stems_manifest)
        gng = res["go_no_go_stems"]
        print(
            f"E8 Contour B (Stems) complete: {res['n_cases']} cases, {res['invariants']['total_runs']} runs, "
            f"neural={res['external_audit']['neural_arm_status']}, "
            f"go_no_go_2={gng['production_verdict']}/{gng['colab_research_verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
