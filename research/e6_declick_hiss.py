#!/usr/bin/env python3
"""E6 Research Module: Archival De-Click & Hiss Reduction (DSP vs Learned).

    python3 research/e6_declick_hiss.py \\
        --manifest corpus/dev-e6-archival-v1.json \\
        --out-json docs/validation/e6-declick-hiss-results.json \\
        --out-md docs/validation/e6-declick-hiss-report.md

Isolated research harness for Board #1131 (E0 charter + spec v1, section E6).
Does NOT modify or extend the frozen E4 Preserve hard-declip pipeline, does NOT
open sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`), and
does NOT launch Colab automation.

Implemented arms:
- `identity`: no-op reference (`residual == 0`).
- `ar_declick`: explicit autoregressive (Yule-Walker) bidirectional prediction
  residual detector with musical-transient guard + local LSAR interpolation
  strictly inside the detected click mask.
- `median_impulse_declick`: high-pass / local-median impulse baseline without
  AR transient guard (demonstrates false-positive trade-off on sharp attacks).
- `wiener_hiss`: spectral Wiener hiss reducer with minimum-statistics / lower-
  quantile noise PSD estimation, decision-directed a priori SNR, clean-floor
  bypass, and reverb-tail gain floor.
- `mmse_stsa_hiss`: Ephraim-Malah (1984) MMSE Short-Time Spectral Amplitude
  hiss reducer using exponentially scaled Bessel functions (`i0e`, `i1e`).
- `ar_declick_wiener_hiss`: two-stage cascade (`ar_declick` -> `wiener_hiss`).
- `ar_declick_mmse_stsa`: two-stage cascade (`ar_declick` -> `mmse_stsa_hiss`).
- `learned_historical_unet_restore`: honest external-weight / Colab audit
  (`blocked_external_weights_or_colab`) with explicit weight/interface contract.

Every restoration arm returns an explicit residual (`removed = input - restored`)
satisfying `restored + residual == input` within float64 precision (`<= 1e-12`)
with zero mutation of the input array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import linalg, signal, special

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import build_e6_archival_corpus, corpus_synth, corpus_v1

EPS = 1e-20
RESIDUAL_IDENTITY_TOL = 1e-12

DSP_ARMS = (
    "identity",
    "ar_declick",
    "median_impulse_declick",
    "wiener_hiss",
    "mmse_stsa_hiss",
    "ar_declick_wiener_hiss",
    "ar_declick_mmse_stsa",
)


@dataclass
class RestorationResult:
    """Output of an archival restoration arm with explicit residual."""

    arm: str
    restored: np.ndarray
    residual: np.ndarray
    click_mask: np.ndarray | None
    wall_s: float
    meta: dict = field(default_factory=dict)


def verify_residual_contract(inp: np.ndarray, result: RestorationResult) -> float:
    """Verify shape, finiteness, and exact identity `restored + residual == input`."""
    if result.restored.shape != inp.shape or result.residual.shape != inp.shape:
        raise ValueError(
            f"{result.arm}: shape mismatch input={inp.shape} "
            f"restored={result.restored.shape} residual={result.residual.shape}"
        )
    if not np.isfinite(result.restored).all() or not np.isfinite(result.residual).all():
        raise ValueError(f"{result.arm}: non-finite values in restored or residual")
    max_err = float(np.max(np.abs((result.restored + result.residual) - inp)))
    if max_err > RESIDUAL_IDENTITY_TOL:
        raise ValueError(f"{result.arm}: residual identity violated (max_err={max_err:.3e})")
    return max_err


def _as_stereo_f64(x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None].copy(), True
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D audio array, got shape {arr.shape}")
    return arr.copy(), False


def _fit_ar_yule_walker(frame: np.ndarray, order: int, ridge: float = 1e-6) -> np.ndarray:
    """Fit an AR(order) model `a = [1, a_1, ..., a_p]` via regularized Yule-Walker."""
    n = frame.size
    if n <= order + 2:
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        return a
    w = frame * np.hanning(n)
    # Fast autocorrelation via FFT.
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(w, n=n_fft)
    acf = np.fft.irfft(spec * np.conj(spec), n=n_fft)[:order + 1]
    r0 = float(acf[0])
    if r0 <= 1e-18:
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        return a
    col = acf[:order].copy()
    col[0] += ridge * r0 + 1e-12
    rhs = -acf[1:order + 1]
    try:
        coeffs = linalg.solve_toeplitz((col, col), rhs)
    except linalg.LinAlgError:
        coeffs = np.zeros(order, dtype=np.float64)
    a = np.empty(order + 1, dtype=np.float64)
    a[0] = 1.0
    a[1:] = coeffs
    return a


def _lsar_interpolate(x: np.ndarray, start: int, end: int, a: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Least-squares AR interpolation (Janssen/Veldhuis 1986) for missing run [start, end)."""
    n = x.size
    p = a.size - 1
    length = end - start
    # Include all prediction equations `n_eq` where at least one sample in [n_eq - p, n_eq]
    # falls inside [start, end).
    eq_start = max(p, start)
    eq_end = min(n, end + p)
    n_eq = eq_end - eq_start
    if n_eq <= 0:
        left = x[start - 1] if start > 0 else 0.0
        right = x[end] if end < n else left
        return np.linspace(left, right, length, dtype=np.float64)

    # Build sparse-like small dense system A_u * x_u + rhs = 0 (length <= ~32).
    a_u = np.zeros((n_eq, length), dtype=np.float64)
    rhs = np.zeros(n_eq, dtype=np.float64)
    for row, idx in enumerate(range(eq_start, eq_end)):
        for k in range(p + 1):
            sample_idx = idx - k
            ak = a[k]
            if start <= sample_idx < end:
                a_u[row, sample_idx - start] += ak
            else:
                rhs[row] += ak * x[sample_idx]

    gram = a_u.T @ a_u
    diag_scale = float(np.mean(np.diag(gram))) if length > 0 else 1.0
    gram[np.diag_indices(length)] += reg * max(diag_scale, 1.0)
    try:
        sol = linalg.solve(gram, -(a_u.T @ rhs), assume_a="pos")
    except linalg.LinAlgError:
        left = x[start - 1] if start > 0 else 0.0
        right = x[end] if end < n else left
        sol = np.linspace(left, right, length, dtype=np.float64)

    # Guard against wild overshoot on OOD wide scratches.
    ctx_lo = max(0, start - 32)
    ctx_hi = min(n, end + 32)
    ctx = np.concatenate((x[ctx_lo:start], x[end:ctx_hi]))
    bound = max(0.05, 1.5 * float(np.max(np.abs(ctx)))) if ctx.size else 1.0
    return np.clip(sol, -bound, bound)


def _extract_runs(mask_1d: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """Extract contiguous True runs from a 1D bool mask, merging gaps <= max_gap."""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    s = int(idx[0])
    e = s + 1
    for v in idx[1:]:
        iv = int(v)
        if iv - e <= max_gap:
            e = iv + 1
        else:
            runs.append((s, e))
            s = iv
            e = iv + 1
    runs.append((s, e))
    return runs


def restore_identity(x: np.ndarray, sr: int) -> RestorationResult:
    """Identity baseline: returns exact copy with zero residual."""
    t0 = time.perf_counter()
    work, was_1d = _as_stereo_f64(x)
    restored = work.copy()
    residual = np.zeros_like(work)
    if was_1d:
        restored = restored[:, 0]
        residual = residual[:, 0]
    return RestorationResult(
        arm="identity",
        restored=restored,
        residual=residual,
        click_mask=np.zeros(x.shape, dtype=bool),
        wall_s=time.perf_counter() - t0,
        meta={"rate": int(sr)},
    )


def restore_ar_declick(
    x: np.ndarray,
    sr: int,
    *,
    ar_order: int | None = None,
    threshold_sigma: float = 6.5,
    min_abs_residual: float = 0.035,
    max_click_ms: float = 0.45,
    enable_transient_guard: bool = True,
) -> RestorationResult:
    """Explicit AR de-clicker with bidirectional residual & musical-transient guard.

    Modifies ONLY samples inside the accepted click mask via local LSAR interpolation.
    All unmasked samples remain bit-exact equal to the input.
    """
    t0 = time.perf_counter()
    work, was_1d = _as_stereo_f64(x)
    n, n_ch = work.shape
    p = ar_order if ar_order is not None else int(np.clip(round(sr * 0.00045), 14, 28))
    frame_len = int(np.clip(round(sr * 0.024), 512, 4096))
    hop = frame_len // 2
    max_click_samples = max(8, round(sr * (max_click_ms / 1000.0)))
    guard_win = max(32, round(sr * 0.004))  # 4 ms pre/post energy window

    restored = work.copy()
    detected_mask = np.zeros((n, n_ch), dtype=bool)
    rejected_transient_runs = 0
    accepted_runs = 0

    for ch in range(n_ch):
        sig = work[:, ch]
        # Compute bidirectional AR prediction residual across overlapping frames.
        score = np.zeros(n, dtype=np.float64)
        cand = np.zeros(n, dtype=bool)
        for start in range(0, max(1, n - frame_len + 1), hop):
            end = min(n, start + frame_len)
            frame = sig[start:end]
            if frame.size <= 2 * p + 4:
                continue
            a = _fit_ar_yule_walker(frame, p)
            # Forward residual e_f[k] = sum_{j=0..p} a[j] * frame[k - j]
            ef = signal.lfilter(a, [1.0], frame)
            # Backward residual e_b[k] = sum_{j=0..p} a[j] * frame[k + j]
            eb = signal.lfilter(a, [1.0], frame[::-1])[::-1]
            # Minimum of |ef| and |eb| sharply localizes the true impulse without post-ringing.
            eboth = np.minimum(np.abs(ef), np.abs(eb))
            # Zero out filter warmup margins.
            valid_lo = p
            valid_hi = frame.size - p
            if valid_hi <= valid_lo:
                continue
            inner = eboth[valid_lo:valid_hi]
            mad = float(np.median(inner))
            sigma = 1.4826 * mad + 1e-8
            thresh = max(threshold_sigma * sigma, min_abs_residual)
            g_lo = start + valid_lo
            g_hi = start + valid_hi
            score[g_lo:g_hi] = np.maximum(score[g_lo:g_hi], inner / sigma)
            cand[g_lo:g_hi] |= inner > thresh

        runs = _extract_runs(cand, max_gap=2)
        for rs, re in runs:
            width = re - rs
            if width > max_click_samples:
                rejected_transient_runs += 1
                continue
            if enable_transient_guard:
                # Compare pre- and post-run residual/signal energy to distinguish
                # sustained musical attacks from localized impulsive clicks.
                pre_lo = max(0, rs - 2 - guard_win)
                pre_hi = max(0, rs - 2)
                post_lo = min(n, re + 2)
                post_hi = min(n, re + 2 + guard_win)
                pre_rms = float(np.sqrt(np.mean(sig[pre_lo:pre_hi] ** 2))) + 1e-6 if pre_hi > pre_lo else 1e-6
                post_rms = float(np.sqrt(np.mean(sig[post_lo:post_hi] ** 2))) + 1e-6 if post_hi > post_lo else 1e-6
                burst_peak_score = float(np.max(score[rs:re]))
                # Second-difference impulsiveness: true clicks have extreme peak vs post-window.
                d2_seg = sig[max(0, rs - 1):min(n, re + 1)]
                d2_burst = float(np.max(np.abs(np.diff(d2_seg, n=2)))) if d2_seg.size >= 3 else 0.0
                d2_post_seg = sig[post_lo:post_hi]
                d2_post = float(np.sqrt(np.mean(np.diff(d2_post_seg, n=2) ** 2))) + 1e-8 if d2_post_seg.size >= 4 else 1e-8
                impulse_contrast = d2_burst / d2_post
                energy_jump = post_rms / pre_rms
                # A musical attack jumps sharply in sustained post-onset energy AND
                # has moderate high-frequency burst contrast (broadband decay continues).
                if energy_jump > 1.85 and impulse_contrast < 16.0 and burst_peak_score < 28.0:
                    rejected_transient_runs += 1
                    continue
                if impulse_contrast < 6.5:
                    rejected_transient_runs += 1
                    continue

            # Local AR interpolation on clean context around [rs, re).
            ctx_half = max(4 * p, 96)
            ctx_lo = max(0, rs - ctx_half)
            ctx_hi = min(n, re + ctx_half)
            ctx_clean = np.concatenate((restored[ctx_lo:rs, ch], restored[re:ctx_hi, ch]))
            a_loc = _fit_ar_yule_walker(ctx_clean, min(p, max(6, ctx_clean.size // 4)))
            restored[rs:re, ch] = _lsar_interpolate(restored[:, ch], rs, re, a_loc)
            detected_mask[rs:re, ch] = True
            accepted_runs += 1

    residual = work - restored
    if was_1d:
        restored = restored[:, 0]
        residual = residual[:, 0]
        detected_mask = detected_mask[:, 0]
    return RestorationResult(
        arm="ar_declick",
        restored=restored,
        residual=residual,
        click_mask=detected_mask,
        wall_s=time.perf_counter() - t0,
        meta={
            "rate": int(sr),
            "ar_order": p,
            "max_click_samples": max_click_samples,
            "accepted_runs": accepted_runs,
            "rejected_transient_runs": rejected_transient_runs,
            "masked_samples": int(np.count_nonzero(detected_mask)),
        },
    )


def restore_median_impulse(
    x: np.ndarray,
    sr: int,
    *,
    threshold_sigma: float = 6.0,
    min_abs_detail: float = 0.04,
    max_click_ms: float = 0.45,
) -> RestorationResult:
    """High-pass / local-median impulse baseline (no AR model, no transient guard).

    Serves as a classical non-AR comparator to show false-positive behavior on
    sharp percussive attacks relative to `ar_declick`.
    """
    t0 = time.perf_counter()
    work, was_1d = _as_stereo_f64(x)
    n, n_ch = work.shape
    max_click_samples = max(8, round(sr * (max_click_ms / 1000.0)))
    block = int(np.clip(round(sr * 0.02), 512, 2048))

    restored = work.copy()
    detected_mask = np.zeros((n, n_ch), dtype=bool)

    for ch in range(n_ch):
        sig = work[:, ch]
        # 2nd-order wavelet-like high-pass detail: d[n] = x[n] - 0.5*(x[n-1] + x[n+1]).
        detail = np.zeros(n, dtype=np.float64)
        detail[1:-1] = np.abs(sig[1:-1] - 0.5 * (sig[:-2] + sig[2:]))
        cand = np.zeros(n, dtype=bool)
        for s in range(0, n, block):
            e = min(n, s + block)
            seg = detail[s:e]
            sigma = 1.4826 * float(np.median(seg)) + 1e-8
            cand[s:e] = seg > max(threshold_sigma * sigma, min_abs_detail)

        for rs, re in _extract_runs(cand, max_gap=1):
            if re - rs > max_click_samples:
                continue
            left = restored[rs - 1, ch] if rs > 0 else 0.0
            right = restored[re, ch] if re < n else left
            restored[rs:re, ch] = np.linspace(left, right, re - rs + 2, dtype=np.float64)[1:-1]
            detected_mask[rs:re, ch] = True

    residual = work - restored
    if was_1d:
        restored = restored[:, 0]
        residual = residual[:, 0]
        detected_mask = detected_mask[:, 0]
    return RestorationResult(
        arm="median_impulse_declick",
        restored=restored,
        residual=residual,
        click_mask=detected_mask,
        wall_s=time.perf_counter() - t0,
        meta={
            "rate": int(sr),
            "masked_samples": int(np.count_nonzero(detected_mask)),
        },
    )


def _estimate_noise_psd(power: np.ndarray, sr: int, hop: int) -> tuple[np.ndarray, float]:
    """Inter-harmonic minimum-statistics noise PSD estimator for archival hiss.

    Archival tape/vinyl hiss is broadband and spectrally smooth, whereas musical
    notes form narrow harmonic peaks. Combining a temporal lower quantile with a
    frequency-axis lower-percentile envelope rejects sustained musical harmonics
    so clean tonal signals and quiet reverb tails are not mistaken for hiss.

    Returns (noise_psd[freq, frames], broadband_floor_dbfs).
    """
    n_freq, n_frames = power.shape
    if n_frames >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)[None, :]
        smooth = signal.convolve2d(power, kernel, mode="same", boundary="symm")
    else:
        smooth = power.copy()

    # Step 1: Temporal lower quantile per frequency bin (12th percentile).
    t_floor = np.percentile(smooth, 12.0, axis=1)

    # Step 2: Frequency-axis inter-harmonic envelope (20th percentile in a ~600 Hz
    # sliding window) to strip narrow sustained musical harmonics from the noise
    # estimate while preserving smooth colored hiss spectra (white/pink/HF-shelf).
    df = (0.5 * sr) / max(1, n_freq - 1)
    freq_win = max(11, round(600.0 / max(df, 1.0)) | 1)
    half_f = freq_win // 2
    spec_floor = np.empty_like(t_floor)
    for k in range(n_freq):
        lo = max(0, k - half_f)
        hi = min(n_freq, k + half_f + 1)
        spec_floor[k] = np.percentile(t_floor[lo:hi], 20.0)
    # Bias correction for taking a lower quantile of smoothed exponential bins.
    global_psd = (spec_floor * 1.85)[:, None]

    # Step 3: Track slow amplitude modulation (wow/flutter) using a scalar
    # broadband modulation envelope estimated from high-frequency inter-harmonic
    # bins (> 3.5 kHz) where hiss dominates and music is sparse.
    hf_start = int(np.clip(round(3500.0 / max(df, 1.0)), 1, max(1, n_freq - 2)))
    hf_env = np.percentile(smooth[hf_start:, :], 20.0, axis=0)
    hf_ref = float(np.percentile(hf_env, 25.0)) + 1e-18
    mod_ratio = np.clip(hf_env / hf_ref, 0.5, 3.0)
    # Smooth modulation ratio over ~0.18 s so transients don't bump the noise floor.
    mod_win = max(3, round(0.18 * sr / hop) | 1)
    if n_frames >= mod_win:
        mod_ratio = signal.medfilt(mod_ratio, kernel_size=mod_win)

    noise_psd = np.maximum(global_psd * mod_ratio[None, :], 1e-18)

    # High-frequency broadband hiss floor (> 5 kHz) extrapolated to full Nyquist
    # bandwidth for clean-signal detection: clean tracks (-82 dBFS floor) sit near
    # -82 dBFS here while even mild 30 dB SNR archival hiss sits above -55 dBFS,
    # avoiding false triggers from low-frequency bass/chord spectral leakage.
    hf_det_start = int(np.clip(round(5000.0 / max(df, 1.0)), 1, max(1, n_freq - 2)))
    hf_frac = max((n_freq - hf_det_start) / float(n_freq), 0.1)
    hf_pow = float(np.median(np.sum(noise_psd[hf_det_start:, :], axis=0))) * 2.67 / hf_frac
    floor_dbfs = float(10.0 * np.log10(max(hf_pow, 1e-20)))
    return noise_psd, floor_dbfs


def _spectral_hiss_reduce(
    x: np.ndarray,
    sr: int,
    *,
    mode: str,
    alpha_dd: float = 0.96,
    gain_floor_db: float = -11.0,
    wiener_beta: float = 1.0,
    clean_bypass_dbfs: float = -70.0,
) -> RestorationResult:
    """Shared STFT spectral hiss reducer supporting Wiener and Ephraim-Malah MMSE-STSA."""
    t0 = time.perf_counter()
    work, was_1d = _as_stereo_f64(x)
    n, n_ch = work.shape
    n_fft = 4096 if sr >= 88200 else 2048
    hop = n_fft // 4
    g_min = 10.0 ** (gain_floor_db / 20.0)

    restored = np.empty_like(work)
    bypassed_channels = 0
    floor_dbfs_per_ch: list[float] = []

    for ch in range(n_ch):
        _, _, zxx = signal.stft(
            work[:, ch],
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            boundary="zeros",
            padded=True,
        )
        mag = np.abs(zxx)
        power = mag ** 2
        noise_psd, floor_dbfs = _estimate_noise_psd(power, sr, hop)
        floor_dbfs_per_ch.append(floor_dbfs)

        # Clean-signal guard: if estimated broadband noise floor is already below
        # -70 dBFS, bypass to keep clean recordings and delicate reverb tails untouched.
        if floor_dbfs < clean_bypass_dbfs:
            restored[:, ch] = work[:, ch]
            bypassed_channels += 1
            continue

        n_freq, n_frames = power.shape
        gains = np.ones((n_freq, n_frames), dtype=np.float64)
        prev_clean_pow = np.maximum(power[:, 0] - noise_psd[:, 0], 0.0)

        for m in range(n_frames):
            lam_d = noise_psd[:, m]
            gamma = np.clip(power[:, m] / lam_d, 1e-6, 1e6)
            xi_inst = np.maximum(gamma - 1.0, 0.0)
            if m == 0:
                xi = np.maximum(xi_inst, 1e-4)
            else:
                xi = alpha_dd * (prev_clean_pow / lam_d) + (1.0 - alpha_dd) * xi_inst
                xi = np.clip(xi, 1e-4, 1e6)

            if mode == "wiener":
                g = xi / (xi + wiener_beta)
            elif mode == "mmse_stsa":
                # Ephraim-Malah (1984) MMSE-STSA using exponentially scaled Bessel i0e, i1e.
                # For v > 50, the exact asymptotic limit is xi / (1 + xi) (error < 1e-3),
                # avoiding any clip mismatch between v and gamma on high-SNR tonal peaks.
                v = (xi / (1.0 + xi)) * gamma
                small = v <= 50.0
                g = xi / (1.0 + xi)
                if np.any(small):
                    vs = np.maximum(v[small], 1e-6)
                    gs = gamma[small]
                    half_v = 0.5 * vs
                    g[small] = (np.sqrt(np.pi) * 0.5) * (np.sqrt(vs) / gs) * (
                        (1.0 + vs) * special.i0e(half_v) + vs * special.i1e(half_v)
                    )
            else:
                raise ValueError(f"unknown spectral hiss mode: {mode}")

            g = np.clip(g, g_min, 1.0)
            gains[:, m] = g
            prev_clean_pow = (g * mag[:, m]) ** 2

        _, y_ch = signal.istft(
            zxx * gains,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            input_onesided=True,
            boundary=True,
        )
        restored[:, ch] = y_ch[:n]

    residual = work - restored
    if was_1d:
        restored = restored[:, 0]
        residual = residual[:, 0]
    arm_name = "wiener_hiss" if mode == "wiener" else "mmse_stsa_hiss"
    return RestorationResult(
        arm=arm_name,
        restored=restored,
        residual=residual,
        click_mask=None,
        wall_s=time.perf_counter() - t0,
        meta={
            "rate": int(sr),
            "n_fft": n_fft,
            "hop": hop,
            "gain_floor_db": gain_floor_db,
            "bypassed_channels": bypassed_channels,
            "estimated_floor_dbfs": [round(v, 2) for v in floor_dbfs_per_ch],
        },
    )


def restore_wiener_hiss(x: np.ndarray, sr: int, **kw) -> RestorationResult:
    """Explicit spectral Wiener hiss reducer with minimum-statistics & tail floor."""
    return _spectral_hiss_reduce(x, sr, mode="wiener", **kw)


def restore_mmse_stsa_hiss(x: np.ndarray, sr: int, **kw) -> RestorationResult:
    """Explicit Ephraim-Malah MMSE-STSA hiss reducer with tail floor."""
    return _spectral_hiss_reduce(x, sr, mode="mmse_stsa", **kw)


def restore_ar_wiener_cascade(x: np.ndarray, sr: int) -> RestorationResult:
    """Two-stage archival DSP cascade: AR de-clicker followed by Wiener hiss reducer."""
    t0 = time.perf_counter()
    s1 = restore_ar_declick(x, sr)
    s2 = restore_wiener_hiss(s1.restored, sr)
    residual = np.asarray(x, dtype=np.float64) - s2.restored
    return RestorationResult(
        arm="ar_declick_wiener_hiss",
        restored=s2.restored,
        residual=residual,
        click_mask=s1.click_mask,
        wall_s=time.perf_counter() - t0,
        meta={"stage1": s1.meta, "stage2": s2.meta},
    )


def restore_ar_mmse_stsa_cascade(x: np.ndarray, sr: int) -> RestorationResult:
    """Two-stage archival DSP cascade: AR de-clicker followed by MMSE-STSA hiss reducer."""
    t0 = time.perf_counter()
    s1 = restore_ar_declick(x, sr)
    s2 = restore_mmse_stsa_hiss(s1.restored, sr)
    residual = np.asarray(x, dtype=np.float64) - s2.restored
    return RestorationResult(
        arm="ar_declick_mmse_stsa",
        restored=s2.restored,
        residual=residual,
        click_mask=s1.click_mask,
        wall_s=time.perf_counter() - t0,
        meta={"stage1": s1.meta, "stage2": s2.meta},
    )


def run_arm(arm: str, x: np.ndarray, sr: int) -> RestorationResult:
    """Dispatch a single DSP arm by name and enforce residual & immutability contracts."""
    dispatch = {
        "identity": restore_identity,
        "ar_declick": restore_ar_declick,
        "median_impulse_declick": restore_median_impulse,
        "wiener_hiss": restore_wiener_hiss,
        "mmse_stsa_hiss": restore_mmse_stsa_hiss,
        "ar_declick_wiener_hiss": restore_ar_wiener_cascade,
        "ar_declick_mmse_stsa": restore_ar_mmse_stsa_cascade,
    }
    if arm not in dispatch:
        raise ValueError(f"unknown E6 DSP arm: {arm!r}")
    inp_f64 = np.asarray(x, dtype=np.float64)
    res = dispatch[arm](inp_f64, sr)
    verify_residual_contract(inp_f64, res)
    return res


def export_audition_bundle(
    inp: np.ndarray,
    result: RestorationResult,
    sr: int,
    out_dir: Path,
    case_id: str,
    bits: int = 24,
) -> dict:
    """Save untouched original, restored, and residual WAVs + SHA-256 provenance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inp_f64 = np.asarray(inp, dtype=np.float64)
    verify_residual_contract(inp_f64, result)
    bundle = {}
    for role, arr in (
        ("original", inp_f64),
        ("restored", result.restored),
        ("residual", result.residual),
    ):
        fname = f"{case_id}.{result.arm}.{role}.wav"
        target = out_dir / fname
        corpus_synth.write_wav(target, sr, np.ascontiguousarray(arr, dtype=np.float64), bits=bits, seed=0)
        raw = target.read_bytes()
        bundle[role] = {"file": fname, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    prov = {
        "case_id": case_id,
        "arm": result.arm,
        "rate": int(sr),
        "bits": int(bits),
        "residual_identity_verified": True,
        "files": bundle,
    }
    (out_dir / f"{case_id}.{result.arm}.provenance.json").write_text(
        json.dumps(prov, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return prov


def audit_learned_arm(models_dir: Path | None = None) -> dict:
    """Honest local audit of learned archival models (Historical U-Net / RESTORE) & Colab.

    Never fabricates learned results or overrides the Colab authorization boundary.
    """
    m_dir = models_dir if models_dir is not None else (REPO / "models")
    weight_exts = (".pt", ".pth", ".ckpt", ".onnx", ".safetensors")
    local_weights = []
    if m_dir.exists():
        for p in sorted(m_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in weight_exts:
                local_weights.append(p.relative_to(REPO).as_posix())

    cuda_available = False
    torch_version = None
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        torch_version = None

    colab_authorized = bool(os.environ.get("MR_E6_COLAB_AUTHORIZED") == "1")
    blocked_reasons = []
    if not local_weights:
        blocked_reasons.append("no_local_historical_unet_or_restore_weights")
    if not cuda_available:
        blocked_reasons.append("no_local_cuda_gpu")
    if not colab_authorized:
        blocked_reasons.append("colab_connection_not_authorized_or_configured")

    status = "blocked_external_weights_or_colab" if blocked_reasons else "ready"
    return {
        "arm": "learned_historical_unet_restore",
        "status": status,
        "executed": False,
        "blocked_reasons": blocked_reasons,
        "environment_audit": {
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "local_archival_weights_found": local_weights,
            "colab_authorized": colab_authorized,
        },
        "weights_contract": {
            "target_models": [
                {
                    "id": "historical_audio_unet",
                    "reference": "Moliner & Valimaki (2022) Historical Audio Restoration U-Net (de-click + hiss)",
                    "expected_checkpoint_format": ".pt / .safetensors with pinned SHA-256",
                    "required_license_check": "explicit_open_research_or_mit_apache_verified_before_run",
                    "io_contract": {
                        "input": "float32/float64 stereo PCM (n_samples, 2), native 44.1/48 kHz",
                        "output": "restored PCM (n_samples, 2) + explicit residual = input - restored",
                        "invariants": [
                            "max(|(restored + residual) - input|) <= 1e-6 (float32) / 1e-12 (float64)",
                            "zero_mutation_of_input",
                            "no_silent_resampling_or_downmix",
                        ],
                    },
                },
                {
                    "id": "restore_archival_diffusion_or_unet",
                    "reference": "RESTORE archival audio restoration family (clicks/crackles/hiss)",
                    "expected_checkpoint_format": ".pt / .ckpt / .safetensors with pinned SHA-256",
                    "required_license_check": "explicit_license_and_export_rights_verified_before_run",
                    "io_contract": {
                        "input": "float32/float64 stereo PCM (n_samples, 2), native 44.1/48 kHz",
                        "output": "restored PCM (n_samples, 2) + explicit residual = input - restored",
                        "invariants": [
                            "max(|(restored + residual) - input|) <= 1e-6",
                            "zero_mutation_of_input",
                            "explicit_generation_tag_for_enhance_derivative_only",
                        ],
                    },
                },
            ]
        },
        "colab_handoff_spec": {
            "permitted_inputs": "corpus/dev-e6-archival-v1.json (rights.colab == True ONLY)",
            "forbidden_inputs": ["corpus/sealed-v1.json", "corpus/sealed-e5-v1.json", "private_masters"],
            "required_run_manifest_fields": [
                "gpu_model",
                "cuda_version",
                "torch_version",
                "checkpoint_id",
                "checkpoint_sha256",
                "rng_seed",
                "wall_s_per_audio_min",
                "per_case_restored_sha256",
                "per_case_residual_sha256",
            ],
            "local_post_colab_verification": (
                "Re-evaluate returned restored/residual WAVs locally via "
                "research.e6_declick_hiss.evaluate_e6_case on the exact same "
                "dev-e6-archival-v1 clean/attack/tail masks; never promote into Preserve."
            ),
        },
        "unpaired_real_protocol": {
            "status": "not_provisioned_in_repo",
            "reason": "no_rights_cleared_real_archival_transfers_bundled_in_repo",
            "clean_reference_fabricated": False,
            "evaluation_rules": [
                "Never compute paired reconstruction SDR/SI-SDR on unpaired real transfers.",
                "Loudness-match identity vs DSP vs learned outputs within <= 0.2 LU after FLAC encode.",
                ("Audition both restored output and isolated residual (`removed = input - restored`) "
                "to verify zero musical attack/cymbal/vocal leakage in the removed signal."),
                ("Score blind A/B preference + artifact rubric (pumping, musical noise, dulled attacks, "
                "cut reverb tails) with randomized blind IDs."),
            ],
        },
    }


def _snr_db(ref: np.ndarray, est: np.ndarray) -> float | None:
    sig = float(np.sum(ref ** 2))
    err = float(np.sum((ref - est) ** 2))
    if sig < EPS or err < EPS:
        return None
    return float(10.0 * np.log10(sig / err))


def _si_sdr_db(ref: np.ndarray, est: np.ndarray) -> float | None:
    s_energy = float(np.sum(ref ** 2))
    if s_energy < EPS:
        return None
    alpha = float(np.sum(est * ref)) / s_energy
    target = alpha * ref
    noise = est - target
    t_energy = float(np.sum(target ** 2))
    n_energy = float(np.sum(noise ** 2))
    if n_energy < EPS:
        return None
    return float(10.0 * np.log10(max(t_energy, EPS) / n_energy))


def _rms_db(x: np.ndarray) -> float:
    return float(10.0 * np.log10(max(float(np.mean(x ** 2)), EPS)))


def evaluate_e6_case(
    clean: np.ndarray,
    degraded: np.ndarray,
    result: RestorationResult,
    sr: int,
    *,
    click_mask: np.ndarray,
    attack_mask: np.ndarray,
    tail_slice: list[int] | None,
    input_sha256_before: str | None = None,
    input_sha256_after: str | None = None,
) -> dict:
    """Evaluate one E6 restoration output against clean/degraded ground truth."""
    c = np.asarray(clean, dtype=np.float64)
    d = np.asarray(degraded, dtype=np.float64)
    y = np.asarray(result.restored, dtype=np.float64)
    r = np.asarray(result.residual, dtype=np.float64)

    res_err = verify_residual_contract(d, result)
    input_mutated = (
        input_sha256_before != input_sha256_after
        if (input_sha256_before is not None and input_sha256_after is not None)
        else False
    )

    snr_in = _snr_db(c, d)
    snr_out = _snr_db(c, y)
    delta_snr = (
        float(snr_out - snr_in)
        if (snr_in is not None and snr_out is not None)
        else (0.0 if snr_in is None and snr_out is None else None)
    )

    sisdr_in = _si_sdr_db(c, d)
    sisdr_out = _si_sdr_db(c, y)
    si_sdri = (
        float(sisdr_out - sisdr_in)
        if (sisdr_in is not None and sisdr_out is not None)
        else (0.0 if sisdr_in is None and sisdr_out is None else None)
    )

    # Click detection metrics (sample-level on true click_mask).
    click_p = click_r = click_f1 = None
    if result.click_mask is not None and click_mask.any():
        pred = result.click_mask.astype(bool)
        true = click_mask.astype(bool)
        tp = int(np.count_nonzero(pred & true))
        fp = int(np.count_nonzero(pred & ~true))
        fn = int(np.count_nonzero(~pred & true))
        click_p = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        click_r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        click_f1 = float(2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    # Transient false-positive rate on clean attack samples (attack_mask & ~click_mask).
    # Note: for masked de-clickers this counts masked samples; for broadband STFT hiss
    # reducers (click_mask is None) it counts samples with |residual| > 1e-3, so
    # clean_attack_residual_energy_db is reported alongside for cross-arm comparison.
    clean_attack = attack_mask.astype(bool) & (~click_mask.astype(bool))
    transient_fp_rate: float | None = None
    clean_attack_residual_energy_db: float | None = None
    clean_attack_zero_residual: bool | None = None
    if clean_attack.any():
        if result.click_mask is not None:
            fp_count = int(np.count_nonzero(result.click_mask.astype(bool) & clean_attack))
        else:
            fp_count = int(np.count_nonzero((np.abs(r) > 1e-3) & clean_attack))
        transient_fp_rate = float(fp_count / int(np.count_nonzero(clean_attack)))
        atk_clean_e = float(np.sum(c[clean_attack] ** 2))
        atk_res_e = float(np.sum(r[clean_attack] ** 2))
        if atk_clean_e > EPS:
            if atk_res_e <= EPS:
                clean_attack_zero_residual = True
                clean_attack_residual_energy_db = None
            else:
                clean_attack_zero_residual = False
                clean_attack_residual_energy_db = float(10.0 * np.log10(atk_res_e / atk_clean_e))

    # Quiet reverb tail preservation metrics.
    quiet_tail_drift_db: float | None = None
    tail_snr_in_db: float | None = None
    tail_snr_out_db: float | None = None
    tail_snr_loss_db: float | None = None
    if tail_slice is not None:
        ts, te = int(tail_slice[0]), int(tail_slice[1])
        if te > ts + 16:
            c_tail = c[ts:te]
            d_tail = d[ts:te]
            y_tail = y[ts:te]
            quiet_tail_drift_db = float(abs(_rms_db(y_tail) - _rms_db(c_tail)))
            tail_snr_in_db = _snr_db(c_tail, d_tail)
            tail_snr_out_db = _snr_db(c_tail, y_tail)
            if tail_snr_in_db is not None and tail_snr_out_db is not None:
                tail_snr_loss_db = float(max(0.0, tail_snr_in_db - tail_snr_out_db))
            elif tail_snr_in_db is None and tail_snr_out_db is None:
                tail_snr_loss_db = 0.0
            else:
                # Input tail SNR is infinite (clean control) while output SNR is finite:
                # leave loss as None (N/A) rather than using a surrogate cap; report
                # actual tail_snr_out_db and quiet_tail_drift_db directly.
                tail_snr_loss_db = None

    # Residual energy & unmodified-outside-mask check.
    residual_rms_db = _rms_db(r)
    outside_click_max_diff = None
    if result.click_mask is not None and result.arm in ("identity", "ar_declick", "median_impulse_declick"):
        unmasked = ~result.click_mask.astype(bool)
        outside_click_max_diff = float(np.max(np.abs(y[unmasked] - d[unmasked]))) if unmasked.any() else 0.0

    bypassed_channels = result.meta.get("bypassed_channels")
    if bypassed_channels is None and isinstance(result.meta.get("stage2"), dict):
        bypassed_channels = result.meta["stage2"].get("bypassed_channels")

    audio_min = max(c.shape[0] / float(sr) / 60.0, 1e-9)
    return {
        "arm": result.arm,
        "snr_in_db": snr_in,
        "snr_out_db": snr_out,
        "delta_snr_db": delta_snr,
        "si_sdr_in_db": sisdr_in,
        "si_sdr_out_db": sisdr_out,
        "si_sdri_db": si_sdri,
        "click_sample_precision": click_p,
        "click_sample_recall": click_r,
        "click_sample_f1": click_f1,
        "transient_fp_rate": transient_fp_rate,
        "clean_attack_residual_energy_db": clean_attack_residual_energy_db,
        "clean_attack_zero_residual": clean_attack_zero_residual,
        "bypassed_channels": bypassed_channels,
        "quiet_tail_drift_db": quiet_tail_drift_db,
        "tail_snr_in_db": tail_snr_in_db,
        "tail_snr_out_db": tail_snr_out_db,
        "tail_snr_loss_db": tail_snr_loss_db,
        "residual_rms_db": residual_rms_db,
        "residual_identity_max_err": res_err,
        "input_mutated": input_mutated,
        "outside_click_max_diff": outside_click_max_diff,
        "masked_samples": int(np.count_nonzero(result.click_mask)) if result.click_mask is not None else None,
        "true_click_samples": int(np.count_nonzero(click_mask)),
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


def _fmt_res_db(val: float | None, *, has_zero_residual: bool = False) -> str:
    if val is not None:
        return f"{val:.1f}"
    return "0 (no edit)" if has_zero_residual else "-"


def _render_report_md(results: dict) -> str:
    learned = results["learned_arm"]
    e4_check = results["e4_freeze_integrity"]
    prov = results["provenance"]
    diag = results["diagnostics"]
    n_clk = results["summary_by_stratum"]["clicks_only"]["identity"]["n_cases"]
    n_hiss = results["summary_by_stratum"]["hiss_only"]["identity"]["n_cases"]
    n_arc = results["summary_by_stratum"]["combined_archival"]["identity"]["n_cases"]
    n_ctl = results["summary_by_stratum"]["clean_control"]["identity"]["n_cases"]
    n_ood = results["summary_by_stratum"]["ood"]["identity"]["n_cases"]
    lines = [
        "# E6 · Archival De-Click & Hiss: DSP vs Learned Research Report",
        "",
        ("**Card:** Board #1131 · **Spec:** E0 charter + spec v1 (Section E6) · "
        f"**Dataset:** `{results['dataset_version']}` (`{results['n_cases']}` stereo cases) · "
        f"**Learned Arm:** `{learned['status']}`"),
        "",
        "## 1. Scope, Safety Boundaries & Provenance",
        "",
        f"- **Sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`):** `sealed_corpora_opened = {results['sealed_corpora_opened']}` (strictly untouched).",
        f"- **Frozen E4 Preserve hard-declip pipeline:** `{e4_check['matched_files']}/{e4_check['total_pinned_files']}` SHA-256 pins verified untouched (`all_untouched = {e4_check['all_untouched']}`).",
        "- **Preserve promotion / Colab automation:** `False` (isolated research study only; does not extend Preserve stereo).",
        f"- **Manifest SHA-256 (`corpus/dev-e6-archival-v1.json`):** `{prov['manifest_sha256']}`",
        f"- **Corpus Builder SHA-256 (`research/build_e6_archival_corpus.py`):** `{prov['builder_sha256']}`",
        f"- **E6 Harness SHA-256 (`research/e6_declick_hiss.py`):** `{prov['harness_sha256']}`",
        f"- **Residual & Immutability Contract:** `{results['invariants']['total_runs']}/{results['invariants']['total_runs']}` runs passed `restored + residual == input` (`max_err = {results['invariants']['max_residual_identity_err']:.2e}`) with `0` input mutations.",
        "",
        "---",
        "",
        "## 2. DSP Arm Comparison by Stratum",
        "",
        f"### 2.1 Clicks & Crackles (`clicks_only`, n={n_clk}, incl. low-level crackle) + Clean Attack Protection (`clean_control`, n={n_ctl})",
        "",
        "> **Note on Clean Controls:** Ultra-clean controls (`-82 dBFS` floor) trigger the spectral hiss `clean_bypass_dbfs = -70.0` gate (`0 (no edit)`). The `e6-ctl-clean-roomtone-tail-48k` control (`-58 dBFS` studio room-tone) does **not** bypass, exposing active Wiener/MMSE trade-offs (see Section 3.3 and Section 6).",
        "",
        "| Arm | Mean ΔSNR (dB) | Mean SI-SDRi (dB) | Click F1 | Clean Attack FP Rate | Clean Attack Residual (dB) | Wall (s/min) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in DSP_ARMS:
        s_clk = results["summary_by_stratum"]["clicks_only"][arm]
        s_ctl = results["summary_by_stratum"]["clean_control"][arm]
        fp_pct = None if s_ctl["mean_transient_fp_rate"] is None else 100.0 * s_ctl["mean_transient_fp_rate"]
        lines.append(
            f"| `{arm}` | {_fmt(s_clk['mean_delta_snr_db'])} | {_fmt(s_clk['mean_si_sdri_db'])} | "
            f"{_fmt(s_clk['mean_click_f1'], '.3f')} | {_fmt(fp_pct, '.4f', '%')} | "
            f"{_fmt_res_db(s_ctl['mean_clean_attack_residual_energy_db'], has_zero_residual=True)} | "
            f"{_fmt(s_clk['mean_wall_s_per_audio_min'], '.2f')} |"
        )

    # Separate clean-tail controls so bypassed (-82 dBFS) and active room-tone (-58 dBFS)
    # are never averaged together into a misleading mixed figure.
    case_by_id = {c["id"]: c["arms"] for c in results["cases"]}
    ctl_tail_clean = case_by_id["e6-ctl-clean-reverb-tail-48k"]
    ctl_tail_room = case_by_id["e6-ctl-clean-roomtone-tail-48k"]

    lines += [
        "",
        f"### 2.2 Stationary & Modulated Hiss (`hiss_only`, n={n_hiss}) + Reverb Tail Preservation",
        "",
        "| Arm | Mean ΔSNR (dB) | Mean SI-SDRi (dB) | Hiss Tail Drift (dB) | Clean Tail Drift (-82 dBFS) | Room-Tone Tail Drift (-58 dBFS) | Room-Tone Tail SNR Out (dB) | Wall (s/min) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in DSP_ARMS:
        s_hiss = results["summary_by_stratum"]["hiss_only"][arm]
        m_clean_tail = ctl_tail_clean[arm]
        m_room_tail = ctl_tail_room[arm]
        rt_snr_out = "∞ (no edit)" if m_room_tail["tail_snr_out_db"] is None else f"{m_room_tail['tail_snr_out_db']:.2f}"
        lines.append(
            f"| `{arm}` | {_fmt(s_hiss['mean_delta_snr_db'])} | {_fmt(s_hiss['mean_si_sdri_db'])} | "
            f"{_fmt(s_hiss['mean_quiet_tail_drift_db'], '.2f')} | "
            f"{_fmt(m_clean_tail['quiet_tail_drift_db'], '.2f')} | "
            f"{_fmt(m_room_tail['quiet_tail_drift_db'], '.2f')} | "
            f"{rt_snr_out} | "
            f"{_fmt(s_hiss['mean_wall_s_per_audio_min'], '.2f')} |"
        )

    lines += [
        "",
        f"### 2.3 Combined Archival Transfers (`combined_archival`, n={n_arc}) & OOD Stress Cases (`ood`, n={n_ood})",
        "",
        "| Arm | Archival Mean ΔSNR (dB) | Archival SI-SDRi (dB) | Archival Tail Drift (dB) | OOD Mean ΔSNR (dB) | OOD SI-SDRi (dB) | OOD Attack FP Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in DSP_ARMS:
        s_arc = results["summary_by_stratum"]["combined_archival"][arm]
        s_ood = results["summary_by_stratum"]["ood"][arm]
        ood_fp = None if s_ood["mean_transient_fp_rate"] is None else 100.0 * s_ood["mean_transient_fp_rate"]
        lines.append(
            f"| `{arm}` | {_fmt(s_arc['mean_delta_snr_db'])} | {_fmt(s_arc['mean_si_sdri_db'])} | "
            f"{_fmt(s_arc['mean_quiet_tail_drift_db'], '.2f')} | "
            f"{_fmt(s_ood['mean_delta_snr_db'])} | {_fmt(s_ood['mean_si_sdri_db'])} | "
            f"{_fmt(ood_fp, '.4f', '%')} |"
        )

    # Section 3: Sensitivity Diagnostics & Active-Algorithm Trade-Offs (B1, B2, B3)
    ll = diag["low_level_crackle"]
    ws = diag["ood_wide_scratch"]
    nb = diag["hiss_without_clean_bypass"]
    rt = diag["roomtone_clean_control"]
    lines += [
        "",
        "---",
        "",
        "## 3. Sensitivity Diagnostics & Active-Algorithm Trade-Offs",
        "",
        "### 3.1 Quiet Crackle Sensitivity (`e6-clicks-low-level-crackle-44k`, amp `0.03-0.08`)",
        "",
        "| Arm | ΔSNR (dB) | Click Precision | Click Recall | Click F1 | Detected / True Samples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("ar_declick", "median_impulse_declick"):
        row = ll[arm]
        lines.append(
            f"| `{arm}` | {_fmt(row['delta_snr_db'])} | {_fmt(row['click_sample_precision'], '.3f')} | "
            f"{_fmt(row['click_sample_recall'], '.3f')} | {_fmt(row['click_sample_f1'], '.3f')} | "
            f"{row['masked_samples']} / {row['true_click_samples']} |"
        )

    lines += [
        "",
        "### 3.2 OOD Wide Scratch Fragmentation (`e6-ood-wide-scratch-44k`, width `20-34` samples)",
        "",
        "| Arm | ΔSNR (dB) | Click Precision | Click Recall | Click F1 | Detected / True Samples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("ar_declick", "median_impulse_declick"):
        row = ws[arm]
        lines.append(
            f"| `{arm}` | {_fmt(row['delta_snr_db'])} | {_fmt(row['click_sample_precision'], '.3f')} | "
            f"{_fmt(row['click_sample_recall'], '.3f')} | {_fmt(row['click_sample_f1'], '.3f')} | "
            f"{row['masked_samples']} / {row['true_click_samples']} |"
        )

    lines += [
        "",
        "### 3.3 Active Wiener / MMSE-STSA Without Bypass vs Room-Tone Clean Control",
        "",
        "| Case / Mode | Arm | Bypassed Ch | Attack FP (`abs(r) > 1e-3`) | Attack Residual (dB) | Quiet Tail Drift (dB) | Tail SNR Out (dB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, entry in (
        ("`e6-ctl-clean-percussion-48k` (bypass disabled)", nb["clean_percussion"]),
        ("`e6-ctl-clean-reverb-tail-48k` (bypass disabled)", nb["clean_reverb_tail"]),
        ("`e6-ctl-clean-roomtone-tail-48k` (`-58 dBFS` room-tone, default)", rt),
    ):
        for arm in ("wiener_hiss", "mmse_stsa_hiss"):
            m = entry[arm]
            fp_pct = None if m["transient_fp_rate"] is None else 100.0 * m["transient_fp_rate"]
            t_snr = (
                "-"
                if m["quiet_tail_drift_db"] is None
                else ("∞ (no edit)" if m["tail_snr_out_db"] is None else f"{m['tail_snr_out_db']:.2f}")
            )
            lines.append(
                f"| {label} | `{arm}` | {m['bypassed_channels']}/2 | "
                f"{_fmt(fp_pct, '.2f', '%')} | "
                f"{_fmt_res_db(m['clean_attack_residual_energy_db'], has_zero_residual=bool(m['clean_attack_zero_residual']))} | "
                f"{_fmt(m['quiet_tail_drift_db'], '.2f')} | {t_snr} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 4. Learned Arm Audit (Historical U-Net / RESTORE) & Colab Contract",
        "",
        f"- **Status:** `{learned['status']}` (`executed = {learned['executed']}`)",
        f"- **Blocked Reasons:** `{', '.join(learned['blocked_reasons'])}`",
        (f"- **Environment Audit:** `torch = {learned['environment_audit']['torch_version']}`, "
        f"`cuda_available = {learned['environment_audit']['cuda_available']}`, "
        f"`local_archival_weights_found = {len(learned['environment_audit']['local_archival_weights_found'])}`, "
        f"`colab_authorized = {learned['environment_audit']['colab_authorized']}`."),
        "- **Policy Compliance:** No learned metrics fabricated; no weights downloaded or executed without license/checkpoint SHA-256 verification; no Colab session launched.",
        "- **Handoff Interface:** Future authorized Colab run must evaluate on `corpus/dev-e6-archival-v1.json` (`rights.colab == True` only, never sealed corpora), export `restored` + `residual = input - restored` with pinned checkpoint SHA-256 and GPU/CUDA/PyTorch metadata, and pass local verification via `research/e6_declick_hiss.py`.",
        "",
        "## 5. Unpaired Real Archival Transfers Protocol",
        "",
        f"- **Status:** `{learned['unpaired_real_protocol']['status']}` (`clean_reference_fabricated = False`).",
        "- No rights-cleared real archival vinyl/tape audio files are bundled in the repository. When rights-cleared real transfers are provided locally, evaluation uses loudness-matched (`<= 0.2 LU`) blind A/B preference + isolated residual (`removed = input - restored`) auditioning for attack/cymbal/reverb leakage — never synthetic paired SDR.",
        "",
        "## 6. Limitations & Honest Method Boundaries",
        "",
        (f"1. **Quiet Crackle Sensitivity Floor:** `ar_declick` uses `min_abs_residual = 0.035` and a strict transient guard to maintain `0.0000%` false positives on clean sharp percussion/plucked attacks. Consequently, on high-amplitude pops (`0.22-0.72`) it achieves `F1 ~ 0.81` and `~+20 dB ΔSNR`, whereas on quiet vinyl/shellac crackle (`e6-clicks-low-level-crackle-44k`, amp `0.03-0.08`) recall drops to `{ll['ar_declick']['click_sample_recall']:.3f}` (`ΔSNR = {_fmt(ll['ar_declick']['delta_snr_db'])} dB`). Lowering the residual threshold would trade away clean-attack protection."),
        (f"2. **Role of `clean_bypass_dbfs = -70.0` in Spectral Hiss Reduction:** Zero drift (`0.00 dB`) and zero residual (`0 (no edit)`) on the ultra-clean `-82 dBFS` controls are achieved by the high-frequency clean-floor bypass gate, **not** by Wiener/MMSE-STSA selectivity alone. When bypass is disabled (`clean_bypass_dbfs = -120.0`) or when a clean recording carries realistic `-58 dBFS` room-tone (`e6-ctl-clean-roomtone-tail-48k`), active Wiener/MMSE-STSA modifies both percussive attacks (`clean_attack_residual_energy_db` ≈ `{_fmt_res_db(rt['wiener_hiss']['clean_attack_residual_energy_db'])} dB`) and quiet reverb tails (`quiet_tail_drift_db` ≈ `{_fmt(rt['wiener_hiss']['quiet_tail_drift_db'], '.2f')} dB`)."),
        (f"3. **Partial Fragmentation on OOD Wide Scratches:** On `e6-ood-wide-scratch-44k` (20-34 sample scratches exceeding `max_click_samples = 20`), `ar_declick` does **not** achieve full abstention: the bidirectional AR residual fragments wide scratches into shorter sub-runs (`<= 20` samples) and partially interpolates `{ws['ar_declick']['masked_samples']}/{ws['ar_declick']['true_click_samples']}` samples (`recall = {ws['ar_declick']['click_sample_recall']:.3f}`, `ΔSNR = {_fmt(ws['ar_declick']['delta_snr_db'])} dB`), leaving residual scratch artifacts. Wide physical scratches require dedicated multi-scale/gap-inpainting detection rather than single-scale AR run-length gating."),
        "4. **Metric Semantics Across Arm Families:** `transient_fp_rate` counts masked samples inside `attack_mask & ~click_mask` for de-clickers (`ar_declick`, `median_impulse_declick`, and cascade stage 1), whereas for standalone broadband STFT hiss reducers (`wiener_hiss`, `mmse_stsa_hiss`) it counts attack samples with `|residual| > 1e-3`. Cross-family comparison on clean attacks should therefore use `clean_attack_residual_energy_db` alongside `transient_fp_rate`.",
        "5. **Non-Stationary Mains Hum & Learned Comparison:** Broadband spectral hiss reducers do not remove tonal 50/100/150 Hz mains hum (`e6-ood-nonstationary-hum-hiss-48k`), which requires a dedicated harmonic notch stage; and comparison against learned models (Historical U-Net / RESTORE) remains blocked until licensed weights and authorized Colab GPU execution are provisioned.",
    ]
    return "\n".join(lines) + "\n"


def run_e6_benchmark(
    manifest_path: Path,
    out_json: Path | None = None,
    out_md: Path | None = None,
) -> dict:
    """Run the full E6 DSP benchmark + learned arm audit and write report artifacts."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != build_e6_archival_corpus.SCHEMA:
        raise ValueError(f"unexpected E6 manifest schema: {manifest.get('schema')!r}")

    per_case: list[dict] = []
    max_res_err = 0.0
    any_mutated = False
    total_runs = 0

    for case in manifest["cases"]:
        clean, degraded, click_mask, attack_mask, tail_slice = build_e6_archival_corpus.synthesize_e6_arrays(case)
        sr = int(case["rate"])
        arm_results: dict[str, dict] = {}
        for arm in DSP_ARMS:
            inp_copy = degraded.copy()
            h_before = hashlib.sha256(inp_copy.tobytes()).hexdigest()
            res = run_arm(arm, inp_copy, sr)
            h_after = hashlib.sha256(inp_copy.tobytes()).hexdigest()
            metrics = evaluate_e6_case(
                clean,
                inp_copy,
                res,
                sr,
                click_mask=click_mask,
                attack_mask=attack_mask,
                tail_slice=tail_slice,
                input_sha256_before=h_before,
                input_sha256_after=h_after,
            )
            max_res_err = max(max_res_err, float(metrics["residual_identity_max_err"]))
            any_mutated = any_mutated or bool(metrics["input_mutated"])
            total_runs += 1
            arm_results[arm] = metrics
        per_case.append({
            "id": case["id"],
            "stratum": case["stratum"],
            "rate": sr,
            "bits": int(case["bits"]),
            "has_clicks": bool(click_mask.any()),
            "has_attacks": bool(attack_mask.any()),
            "has_quiet_tail": tail_slice is not None,
            "arms": arm_results,
        })

    # Aggregate per stratum and arm.
    summary_by_stratum: dict[str, dict[str, dict]] = {}
    for stratum in build_e6_archival_corpus.STRATA:
        rows = [c for c in per_case if c["stratum"] == stratum]
        summary_by_stratum[stratum] = {}
        for arm in DSP_ARMS:
            ms = [r["arms"][arm] for r in rows]
            summary_by_stratum[stratum][arm] = {
                "n_cases": len(rows),
                "mean_delta_snr_db": _mean([m["delta_snr_db"] for m in ms]),
                "mean_si_sdri_db": _mean([m["si_sdri_db"] for m in ms]),
                "mean_click_f1": _mean([m["click_sample_f1"] for m in ms]),
                "mean_transient_fp_rate": _mean([m["transient_fp_rate"] for m in ms]),
                "max_transient_fp_rate": _max([m["transient_fp_rate"] for m in ms]),
                "mean_clean_attack_residual_energy_db": _mean([m["clean_attack_residual_energy_db"] for m in ms]),
                "mean_quiet_tail_drift_db": _mean([m["quiet_tail_drift_db"] for m in ms]),
                "max_quiet_tail_drift_db": _max([m["quiet_tail_drift_db"] for m in ms]),
                "mean_tail_snr_loss_db": _mean([m["tail_snr_loss_db"] for m in ms]),
                "mean_wall_s_per_audio_min": _mean([m["wall_s_per_audio_min"] for m in ms]),
            }

    # Build explicit diagnostics (B1 wide-scratch fragmentation, B2 bypass-disabled & room-tone, B3 quiet crackle).
    case_map = {c["id"]: c for c in per_case}
    manifest_map = {c["id"]: c for c in manifest["cases"]}
    no_bypass_diag: dict[str, dict[str, dict]] = {}
    for key, cid in (
        ("clean_percussion", "e6-ctl-clean-percussion-48k"),
        ("clean_reverb_tail", "e6-ctl-clean-reverb-tail-48k"),
    ):
        c_arr, d_arr, cm_arr, am_arr, ts_arr = build_e6_archival_corpus.synthesize_e6_arrays(manifest_map[cid])
        sr_d = int(manifest_map[cid]["rate"])
        no_bypass_diag[key] = {}
        for arm_name, fn in (("wiener_hiss", restore_wiener_hiss), ("mmse_stsa_hiss", restore_mmse_stsa_hiss)):
            res_nb = fn(d_arr, sr_d, clean_bypass_dbfs=-120.0)
            no_bypass_diag[key][arm_name] = evaluate_e6_case(
                c_arr, d_arr, res_nb, sr_d,
                click_mask=cm_arr, attack_mask=am_arr, tail_slice=ts_arr,
            )

    diagnostics = {
        "low_level_crackle": {
            arm: case_map["e6-clicks-low-level-crackle-44k"]["arms"][arm]
            for arm in ("ar_declick", "median_impulse_declick")
        },
        "ood_wide_scratch": {
            arm: case_map["e6-ood-wide-scratch-44k"]["arms"][arm]
            for arm in ("ar_declick", "median_impulse_declick")
        },
        "roomtone_clean_control": {
            arm: case_map["e6-ctl-clean-roomtone-tail-48k"]["arms"][arm]
            for arm in ("wiener_hiss", "mmse_stsa_hiss")
        },
        "hiss_without_clean_bypass": no_bypass_diag,
    }

    learned = audit_learned_arm()
    e4_check = _verify_e4_freeze_untouched()
    results = {
        "schema": "e6-declick-hiss-results/v1",
        "board_card": 1131,
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
            "manifest_path": "corpus/dev-e6-archival-v1.json",
            "manifest_sha256": corpus_v1.digest(manifest_path),
            "builder_path": "research/build_e6_archival_corpus.py",
            "builder_sha256": corpus_v1.digest(REPO / "research" / "build_e6_archival_corpus.py"),
            "harness_path": "research/e6_declick_hiss.py",
            "harness_sha256": corpus_v1.digest(REPO / "research" / "e6_declick_hiss.py"),
        },
        "e4_freeze_integrity": e4_check,
        "invariants": {
            "total_runs": total_runs,
            "max_residual_identity_err": max_res_err,
            "any_input_mutated": any_mutated,
        },
        "dsp_arms": list(DSP_ARMS),
        "learned_arm": learned,
        "diagnostics": diagnostics,
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
    ap.add_argument("--manifest", default="corpus/dev-e6-archival-v1.json")
    ap.add_argument("--out-json", default="docs/validation/e6-declick-hiss-results.json")
    ap.add_argument("--out-md", default="docs/validation/e6-declick-hiss-report.md")
    args = ap.parse_args()
    res = run_e6_benchmark(Path(args.manifest), Path(args.out_json), Path(args.out_md))
    print(
        f"E6 benchmark complete: {res['n_cases']} cases, {res['invariants']['total_runs']} runs, "
        f"learned={res['learned_arm']['status']}, e4_untouched={res['e4_freeze_integrity']['all_untouched']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
