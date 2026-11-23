"""Stereo image, phase/IPD, onset/pre-echo, gapless, and album-level safety checks.

Single-owner DSP safety verification module for E4 (Stereo/Album Safety + CPU
Freeze) and E5 (Sealed Holdout) gates per spec v1 sections 3 and 4:
- Inter-channel correlation, whole-track and local active-window ILD drift;
- Cross-spectral phase coherence, energy-weighted circular IPD error (whole
  track and masked frames), and sub-sample ITD lag drift;
- Mid/Side (M/S) balance and mono-sum SDR / cancellation checks;
- Sharp transient pre-echo leakage, near-mask seam continuity, unchanged-band
  spectral stability, and quiet-tail preservation;
- Joint gapless track-boundary inspection (seam step, slope, level jump, peak
  growth);
- Preserve album-level attenuation-only headroom policy: one unified album gain
  when compatible with byte-for-byte immutability of kept tracks, or explicit
  `needs_review` + `auto_album_delivery=False` diagnostics when mixed.
"""
from __future__ import annotations

import math

import numpy as np

from rnext import features as F

EPS = 1e-20

# Frozen E4/E5 engineering safety rubric (hard gates; every threshold is
# empirically verified on dev corpora and synthetic stress tests before E5).
DEFAULT_STEREO_SAFETY_RUBRIC = {
    "max_reliable_diff_pre_gain": 0.0,
    "max_polarity_violations": 0,
    "max_lr_corr_drift_vs_clean": 0.03,
    "max_ild_drift_vs_clean_db": 0.50,
    "max_local_ild_drift_p95_db": 1.25,
    "max_phase_coherence_drift_vs_clean": 0.03,
    "max_ipd_error_mean_deg": 5.0,
    "max_masked_ipd_error_mean_deg": 15.0,
    "max_itd_drift_samples": 1.0,
    "max_ms_ratio_drift_vs_clean_db": 0.50,
    "min_mono_sum_level_drift_vs_clean_db": -0.50,
    "max_pre_echo_rise_db": 0.50,
    "max_quiet_tail_rms_drift_db": 0.25,
    "max_band_shape_rms_db": 1.50,
    "max_unchanged_band_drift_db": 0.50,
    "max_gapless_level_jump_db": 0.20,
    "max_gapless_seam_peak_growth_db": 3.0,
    "max_control_fp": 0,
}


def _validate_stereo(x: np.ndarray, name: str = "PCM") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name}: expected nonempty finite stereo PCM [samples, 2]")
    return arr


def _sdr_db(sig_energy: float, err_energy: float, eps: float = EPS) -> float | None:
    if sig_energy < eps or err_energy < eps:
        return None
    return float(10.0 * math.log10(sig_energy / err_energy))


def shared_stereo_gain(clean: np.ndarray, candidate: np.ndarray, eps: float = EPS) -> float:
    """Single whole-track scalar least-squares gain shared across L & R (delay=0)."""
    c = _validate_stereo(clean, "clean")
    y = _validate_stereo(candidate, "candidate")
    if c.shape != y.shape:
        raise ValueError("clean and candidate shapes must match")
    denom = float(np.sum(y * y))
    if denom < eps:
        return 1.0
    g = float(np.sum(c * y) / denom)
    return g if g > 0.0 else 1.0


def stereo_lr_correlation(x: np.ndarray, eps: float = EPS) -> float | None:
    """Whole-signal Pearson correlation between left and right channels."""
    arr = _validate_stereo(x)
    l_ch = arr[:, 0] - np.mean(arr[:, 0])
    r_ch = arr[:, 1] - np.mean(arr[:, 1])
    denom = math.sqrt(float(np.sum(l_ch * l_ch) * np.sum(r_ch * r_ch)))
    if denom < eps:
        return None
    return float(np.clip(float(np.sum(l_ch * r_ch)) / denom, -1.0, 1.0))


def stereo_ild_db(x: np.ndarray, eps: float = EPS) -> float | None:
    """Interaural Level Difference 20*log10(rms_L / rms_R) in dB."""
    arr = _validate_stereo(x)
    rms_l = math.sqrt(float(np.mean(arr[:, 0] ** 2)))
    rms_r = math.sqrt(float(np.mean(arr[:, 1] ** 2)))
    if rms_l < math.sqrt(eps) or rms_r < math.sqrt(eps):
        return None
    return float(20.0 * math.log10(rms_l / rms_r))


def stereo_local_ild_drift(
    ref: np.ndarray,
    cand: np.ndarray,
    sr: int,
    *,
    window_s: float = 0.1,
    hop_s: float = 0.05,
    active_floor_db: float = -50.0,
) -> dict:
    """Windowed active-region absolute ILD drift |ILD(cand) - ILD(ref)| in dB."""
    r = _validate_stereo(ref, "ref")
    y = _validate_stereo(cand, "cand")
    if r.shape != y.shape or sr <= 0:
        raise ValueError("matching stereo shapes and positive sr required")
    n = r.shape[0]
    win = min(n, max(64, round(window_s * sr)))
    hop = max(1, min(win, round(hop_s * sr)))
    peak = float(np.max(np.abs(r)))
    if peak < 1e-10:
        return {"active_windows": 0, "p50_db": None, "p95_db": None, "max_db": None}
    floor = peak * (10.0 ** (active_floor_db / 20.0))

    drifts = []
    for s in range(0, n - win + 1, hop):
        seg_r = r[s:s + win]
        seg_y = y[s:s + win]
        rl_r = math.sqrt(float(np.mean(seg_r[:, 0] ** 2)))
        rr_r = math.sqrt(float(np.mean(seg_r[:, 1] ** 2)))
        if rl_r < floor or rr_r < floor:
            continue
        rl_y = math.sqrt(float(np.mean(seg_y[:, 0] ** 2)))
        rr_y = math.sqrt(float(np.mean(seg_y[:, 1] ** 2)))
        if rl_y < floor * 1e-3 or rr_y < floor * 1e-3:
            continue
        ild_r = 20.0 * math.log10(rl_r / rr_r)
        ild_y = 20.0 * math.log10(rl_y / rr_y)
        drifts.append(abs(ild_y - ild_r))
    if not drifts:
        return {"active_windows": 0, "p50_db": None, "p95_db": None, "max_db": None}
    arr = np.asarray(drifts, dtype=np.float64)
    return {
        "active_windows": int(arr.size),
        "p50_db": float(np.percentile(arr, 50)),
        "p95_db": float(np.percentile(arr, 95)),
        "max_db": float(np.max(arr)),
    }


def _estimate_itd_samples(x: np.ndarray, sr: int, max_lag_ms: float = 2.0) -> tuple[float, float]:
    """Sub-sample Interaural Time Difference (positive = R lags L) via GCC-PHAT.

    Returns (lag_samples, max_normalized_xcorr). ITD is only physically defined
    when L and R share coherent energy (`max_normalized_xcorr >= 0.25`).
    """
    n = x.shape[0]
    max_lag = max(2, min(n // 4, round(max_lag_ms * sr / 1000.0)))
    # Use a power-of-2 FFT over up to 10 seconds of active audio to bound cost.
    seg = x[:min(n, sr * 10)]
    m = seg.shape[0]
    l0 = seg[:, 0] - np.mean(seg[:, 0])
    r0 = seg[:, 1] - np.mean(seg[:, 1])
    norm_lr = math.sqrt(float(np.sum(l0 * l0) * np.sum(r0 * r0)))
    if norm_lr < EPS:
        return 0.0, 0.0
    n_fft = 1 << math.ceil(math.log2(max(2 * m, 256)))
    sl = np.fft.rfft(l0, n=n_fft)
    sr_f = np.fft.rfft(r0, n=n_fft)
    cross = sr_f * np.conj(sl)

    # Standard normalized cross-correlation peak in [-max_lag .. +max_lag] to test coherence.
    cc_std = np.fft.irfft(cross, n=n_fft) / norm_lr
    win_std = np.concatenate((cc_std[-max_lag:], cc_std[:max_lag + 1]))
    ncc_peak = float(np.max(np.abs(win_std)))

    # Regularized PHAT weighting suppresses quiet noise bins while preserving sharp lag peak.
    mag = np.abs(cross)
    peak_mag = float(np.max(mag))
    if peak_mag < EPS:
        return 0.0, ncc_peak
    cross_phat = cross / (mag + 1e-3 * peak_mag)
    cc = np.fft.irfft(cross_phat, n=n_fft)
    window = np.concatenate((cc[-max_lag:], cc[:max_lag + 1]))
    idx = int(np.argmax(window))
    lag = float(idx - max_lag)
    # Parabolic sub-sample refinement when peak is interior.
    if 0 < idx < window.size - 1:
        y0, y1, y2 = float(window[idx - 1]), float(window[idx]), float(window[idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            lag += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
    return lag, ncc_peak


def stereo_cross_spectral_metrics(
    ref: np.ndarray,
    cand: np.ndarray,
    sr: int,
    *,
    mask: np.ndarray | None = None,
    frame_44k: int = 2048,
) -> dict:
    """Cross-spectral phase coherence, circular IPD error (deg), and ITD drift."""
    r = _validate_stereo(ref, "ref")
    y = _validate_stereo(cand, "cand")
    if r.shape != y.shape or sr <= 0:
        raise ValueError("matching stereo shapes and positive sr required")
    n = r.shape[0]
    if n < 64:
        return {
            "phase_coherence_clean": None,
            "phase_coherence_candidate": None,
            "phase_coherence_drift_vs_clean": None,
            "ipd_error_mean_deg": None,
            "ipd_error_p95_deg": None,
            "masked_frames_evaluated": 0,
            "masked_ipd_error_mean_deg": None,
            "masked_ipd_error_p95_deg": None,
            "itd_ref_ncc_peak": None,
            "itd_cand_ncc_peak": None,
            "itd_defined": False,
            "itd_clean_us": None,
            "itd_candidate_us": None,
            "itd_drift_us": None,
            "itd_drift_samples": None,
        }
    frame = max(256, min(8192, int(2 ** round(math.log2(max(1.0, frame_44k * sr / 44100.0))))))
    if n < frame:
        frame = max(64, 1 << math.floor(math.log2(n)))
    hop = max(32, frame // 2)
    win = np.hanning(frame + 1)[:-1].astype(np.float64)
    starts = np.arange(0, max(1, n - frame + 1), hop)

    freqs = np.fft.rfftfreq(frame, d=1.0 / sr)
    f_hi = min(12000.0, 0.45 * sr)
    band = (freqs >= 80.0) & (freqs <= f_hi)
    if not band.any():
        band = freqs > 0.0

    # Accumulate cross-spectra in chunks to keep memory bounded on 5-minute tracks.
    s_ll_r = np.zeros(freqs.size, dtype=np.float64)
    s_rr_r = np.zeros(freqs.size, dtype=np.float64)
    s_lr_r = np.zeros(freqs.size, dtype=np.complex128)
    s_ll_y = np.zeros(freqs.size, dtype=np.float64)
    s_rr_y = np.zeros(freqs.size, dtype=np.float64)
    s_lr_y = np.zeros(freqs.size, dtype=np.complex128)

    # Masked-frame cross-spectra (when mask is provided and nonempty).
    has_mask = mask is not None and bool(np.any(mask))
    m_lr_r = np.zeros(freqs.size, dtype=np.complex128)
    m_lr_y = np.zeros(freqs.size, dtype=np.complex128)
    m_frames = 0
    mask_any_t = np.any(mask, axis=1) if has_mask else None

    ipd_frame_errs = []
    ipd_frame_weights = []
    masked_ipd_errs = []
    masked_ipd_weights = []

    batch = 256
    for b0 in range(0, starts.size, batch):
        bs = starts[b0:b0 + batch]
        fr_l = np.stack([r[s:s + frame, 0] for s in bs]) * win[None, :]
        fr_r = np.stack([r[s:s + frame, 1] for s in bs]) * win[None, :]
        fy_l = np.stack([y[s:s + frame, 0] for s in bs]) * win[None, :]
        fy_r = np.stack([y[s:s + frame, 1] for s in bs]) * win[None, :]

        zl_r = np.fft.rfft(fr_l, axis=-1)
        zr_r = np.fft.rfft(fr_r, axis=-1)
        zl_y = np.fft.rfft(fy_l, axis=-1)
        zr_y = np.fft.rfft(fy_r, axis=-1)

        c_r = zl_r * np.conj(zr_r)
        c_y = zl_y * np.conj(zr_y)

        s_ll_r += np.sum(np.abs(zl_r) ** 2, axis=0)
        s_rr_r += np.sum(np.abs(zr_r) ** 2, axis=0)
        s_lr_r += np.sum(c_r, axis=0)

        s_ll_y += np.sum(np.abs(zl_y) ** 2, axis=0)
        s_rr_y += np.sum(np.abs(zr_y) ** 2, axis=0)
        s_lr_y += np.sum(c_y, axis=0)

        # Frame-level circular IPD error on active cross-spectral bins.
        c_rb = c_r[:, band]
        c_yb = c_y[:, band]
        w_b = np.abs(c_rb)
        # Circular angular difference via angle(c_y * conj(c_r)).
        ang_deg = np.degrees(np.abs(np.angle(c_yb * np.conj(c_rb) + 0j)))
        frame_w = np.sum(w_b, axis=1)
        peak_fw = float(np.max(frame_w)) if frame_w.size else 0.0
        for fi, s in enumerate(bs):
            fw = float(frame_w[fi])
            if fw <= 1e-12 * max(peak_fw, 1.0) or fw < EPS:
                continue
            mean_ang = float(np.sum(w_b[fi] * ang_deg[fi]) / fw)
            ipd_frame_errs.append(mean_ang)
            ipd_frame_weights.append(fw)
            if has_mask and bool(np.any(mask_any_t[s:s + frame])):
                m_lr_r += c_r[fi]
                m_lr_y += c_y[fi]
                m_frames += 1
                masked_ipd_errs.append(mean_ang)
                masked_ipd_weights.append(fw)

    def _weighted_coherence(s_ll, s_rr, s_lr):
        w = np.sqrt(np.maximum(s_ll[band] * s_rr[band], 0.0))
        tot = float(np.sum(w))
        if tot < EPS:
            return None
        coh = np.abs(s_lr[band]) / np.maximum(w, EPS)
        return float(np.clip(np.sum(w * coh) / tot, 0.0, 1.0))

    coh_ref = _weighted_coherence(s_ll_r, s_rr_r, s_lr_r)
    coh_cand = _weighted_coherence(s_ll_y, s_rr_y, s_lr_y)
    coh_drift = (abs(coh_cand - coh_ref)
                 if coh_ref is not None and coh_cand is not None else None)

    if ipd_frame_errs:
        errs = np.asarray(ipd_frame_errs, dtype=np.float64)
        wts = np.asarray(ipd_frame_weights, dtype=np.float64)
        ipd_mean = float(np.sum(errs * wts) / max(float(np.sum(wts)), EPS))
        ipd_p95 = float(np.percentile(errs, 95))
    else:
        ipd_mean = None
        ipd_p95 = None

    if masked_ipd_errs:
        m_errs = np.asarray(masked_ipd_errs, dtype=np.float64)
        m_wts = np.asarray(masked_ipd_weights, dtype=np.float64)
        m_ipd_mean = float(np.sum(m_errs * m_wts) / max(float(np.sum(m_wts)), EPS))
        m_ipd_p95 = float(np.percentile(m_errs, 95))
    else:
        m_ipd_mean = None
        m_ipd_p95 = None

    itd_ref_samp, ncc_ref = _estimate_itd_samples(r, sr)
    itd_cand_samp, ncc_cand = _estimate_itd_samples(y, sr)
    itd_defined = ncc_ref >= 0.25
    itd_drift_samp = abs(itd_cand_samp - itd_ref_samp) if itd_defined else None

    return {
        "phase_coherence_clean": coh_ref,
        "phase_coherence_candidate": coh_cand,
        "phase_coherence_drift_vs_clean": coh_drift,
        "ipd_error_mean_deg": ipd_mean,
        "ipd_error_p95_deg": ipd_p95,
        "masked_frames_evaluated": m_frames,
        "masked_ipd_error_mean_deg": m_ipd_mean,
        "masked_ipd_error_p95_deg": m_ipd_p95,
        "itd_ref_ncc_peak": ncc_ref,
        "itd_cand_ncc_peak": ncc_cand,
        "itd_defined": bool(itd_defined),
        "itd_clean_us": float(itd_ref_samp * 1e6 / sr) if itd_defined else None,
        "itd_candidate_us": float(itd_cand_samp * 1e6 / sr) if itd_defined else None,
        "itd_drift_us": float(itd_drift_samp * 1e6 / sr) if itd_defined else None,
        "itd_drift_samples": float(itd_drift_samp) if itd_defined else None,
    }


def stereo_ms_and_mono_metrics(
    clean: np.ndarray,
    degraded: np.ndarray,
    candidate_aligned: np.ndarray,
    *,
    degraded_aligned: np.ndarray | None = None,
    eps: float = EPS,
) -> dict:
    """Mid/Side balance and mono-sum compatibility metrics."""
    c = _validate_stereo(clean, "clean")
    d = _validate_stereo(degraded, "degraded")
    y = _validate_stereo(candidate_aligned, "candidate_aligned")
    d_al = d if degraded_aligned is None else _validate_stereo(degraded_aligned, "degraded_aligned")

    m_c, s_c = 0.5 * (c[:, 0] + c[:, 1]), 0.5 * (c[:, 0] - c[:, 1])
    m_d = 0.5 * (d_al[:, 0] + d_al[:, 1])
    m_y, s_y = 0.5 * (y[:, 0] + y[:, 1]), 0.5 * (y[:, 0] - y[:, 1])

    rms_mc = math.sqrt(float(np.mean(m_c ** 2)))
    rms_sc = math.sqrt(float(np.mean(s_c ** 2)))
    rms_my = math.sqrt(float(np.mean(m_y ** 2)))
    rms_sy = math.sqrt(float(np.mean(s_y ** 2)))

    ms_clean = (float(20.0 * math.log10(rms_mc / rms_sc))
                if rms_mc > 1e-8 and rms_sc > 1e-8 else None)
    ms_cand = (float(20.0 * math.log10(rms_my / rms_sy))
               if rms_my > 1e-8 and rms_sy > 1e-8 else None)
    ms_drift = (abs(ms_cand - ms_clean)
                if ms_clean is not None and ms_cand is not None else None)

    mono_sig = float(np.sum(m_c ** 2))
    mono_sdr_deg = _sdr_db(mono_sig, float(np.sum((m_c - m_d) ** 2)), eps=eps)
    mono_sdr_cand = _sdr_db(mono_sig, float(np.sum((m_c - m_y) ** 2)), eps=eps)
    mono_delta = (float(mono_sdr_cand - mono_sdr_deg)
                  if mono_sdr_cand is not None and mono_sdr_deg is not None else None)
    mono_level_drift = (float(20.0 * math.log10(rms_my / rms_mc))
                        if rms_mc > 1e-8 and rms_my > 1e-8 else None)

    return {
        "ms_ratio_clean_db": ms_clean,
        "ms_ratio_candidate_db": ms_cand,
        "ms_ratio_drift_vs_clean_db": ms_drift,
        "mono_sum_sdr_degraded_db": mono_sdr_deg,
        "mono_sum_sdr_candidate_db": mono_sdr_cand,
        "mono_sum_delta_sdr_db": mono_delta,
        "mono_sum_level_drift_vs_clean_db": mono_level_drift,
    }


def onset_and_pre_echo_metrics(
    clean: np.ndarray,
    degraded: np.ndarray,
    candidate_aligned: np.ndarray,
    sr: int,
    *,
    mask: np.ndarray | None = None,
    raw_candidate: np.ndarray | None = None,
) -> dict:
    """Measure pre-onset pre-echo leakage, near-mask seam step, and attack fidelity."""
    c = _validate_stereo(clean, "clean")
    d = _validate_stereo(degraded, "degraded")
    y = _validate_stereo(candidate_aligned, "candidate_aligned")
    n = c.shape[0]
    m = np.zeros(c.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    # 1. Pre-echo check: find sharp 10 ms onset jumps (> +12 dB over preceding 10 ms).
    pre_w = max(32, round(0.010 * sr))
    hop = max(16, pre_w // 2)
    c_env = np.mean(c ** 2, axis=1)
    peak_e = float(np.max(c_env))
    pre_echo_rises = []
    pre_gain_diffs = []
    n_onsets = 0

    if peak_e > 1e-10 and n > 2 * pre_w:
        # Cumulative sum for O(N) block energy evaluation.
        cs_c = np.concatenate(([0.0], np.cumsum(c_env)))
        cs_y = np.concatenate(([0.0], np.cumsum(np.mean(y ** 2, axis=1))))
        idx = np.arange(pre_w, n - pre_w, hop)
        e_pre_c = (cs_c[idx] - cs_c[idx - pre_w]) / pre_w
        e_post_c = (cs_c[idx + pre_w] - cs_c[idx]) / pre_w
        # Sharp onset: post-onset energy >= 1% of peak and >= 12 dB (16x) above pre-onset.
        onset_hits = np.flatnonzero((e_post_c >= 0.01 * peak_e) & (e_post_c >= 16.0 * np.maximum(e_pre_c, 1e-12)))
        # Suppress adjacent detections within 30 ms.
        min_gap = max(1, round(0.030 * sr) // hop)
        last = -min_gap - 1
        for hi in onset_hits:
            if hi - last < min_gap:
                continue
            last = hi
            pos_coarse = int(idx[hi])
            # Refine coarse hop-aligned trigger to the exact attack sample where
            # signal energy first rises above 5% of the local post-onset peak.
            seg_e = c_env[pos_coarse:pos_coarse + pre_w]
            local_pk = float(np.max(seg_e)) if seg_e.size else 0.0
            cross = np.flatnonzero(seg_e >= 0.05 * local_pk) if local_pk > 0.0 else np.array([], dtype=int)
            pos = pos_coarse + (int(cross[0]) if cross.size else 0)
            if pos < pre_w:
                continue
            n_onsets += 1
            e_pre_ref = float((cs_c[pos] - cs_c[pos - pre_w]) / pre_w)
            e_pre_cand = float((cs_y[pos] - cs_y[pos - pre_w]) / pre_w)
            # Floor at -80 dB below peak so silence quantization does not cause false inf.
            floor_e = peak_e * 1e-8
            rise_db = 10.0 * math.log10((e_pre_cand + floor_e) / (e_pre_ref + floor_e))
            pre_echo_rises.append(rise_db)
            if raw_candidate is not None:
                unm = ~m[pos - pre_w:pos]
                if unm.any():
                    rc = np.asarray(raw_candidate, dtype=np.float64)[pos - pre_w:pos]
                    pre_gain_diffs.append(float(np.max(np.abs(rc[unm] - d[pos - pre_w:pos][unm]))))

    # 2. Near-mask seam continuity: compare sample-to-sample steps right across mask edges.
    seam_ratio = 1.0
    attack_rms_drift = 0.0
    if m.any():
        edge_steps_c = []
        edge_steps_y = []
        for ch in range(2):
            mc = m[:, ch]
            if not mc.any():
                continue
            edges = np.flatnonzero(mc[1:] != mc[:-1])
            if edges.size:
                edge_steps_c.append(np.abs(c[edges + 1, ch] - c[edges, ch]))
                edge_steps_y.append(np.abs(y[edges + 1, ch] - y[edges, ch]))
        if edge_steps_c:
            sc = np.concatenate(edge_steps_c)
            sy = np.concatenate(edge_steps_y)
            p95_c = float(np.percentile(sc, 95))
            p95_y = float(np.percentile(sy, 95))
            seam_ratio = float(p95_y / max(p95_c, 1e-6))

        # Attack RMS drift on clipped samples vs clean.
        rms_mc = math.sqrt(float(np.mean(c[m] ** 2)))
        rms_my = math.sqrt(float(np.mean(y[m] ** 2)))
        if rms_mc > 1e-8 and rms_my > 1e-8:
            attack_rms_drift = float(abs(20.0 * math.log10(rms_my / rms_mc)))

    return {
        "sharp_onsets_checked": n_onsets,
        "pre_echo_rise_db": float(max(pre_echo_rises)) if pre_echo_rises else 0.0,
        "pre_echo_pre_gain_max_diff": (float(max(pre_gain_diffs))
                                       if pre_gain_diffs else (0.0 if raw_candidate is not None else None)),
        "near_mask_seam_step_ratio": seam_ratio,
        "masked_attack_rms_drift_db": attack_rms_drift,
    }


def unchanged_bands_and_tail_metrics(
    clean: np.ndarray,
    degraded: np.ndarray,
    candidate_aligned: np.ndarray,
    sr: int,
    *,
    mask: np.ndarray | None = None,
    raw_candidate: np.ndarray | None = None,
) -> dict:
    """1/3-octave band stability (overall & unchanged bands) and quiet-tail preservation."""
    c = _validate_stereo(clean, "clean")
    d = _validate_stereo(degraded, "degraded")
    y = _validate_stereo(candidate_aligned, "candidate_aligned")
    m = np.zeros(c.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    d_al = shared_stereo_gain(c, d) * d

    bands_c = np.asarray(F.band_rms_db(c.astype(np.float32), sr), dtype=np.float64)
    bands_d = np.asarray(F.band_rms_db(d_al.astype(np.float32), sr), dtype=np.float64)
    bands_y = np.asarray(F.band_rms_db(y.astype(np.float32), sr), dtype=np.float64)
    max_band_c = float(np.max(bands_c)) if bands_c.size else -240.0
    active = (bands_c >= max_band_c - 50.0) if max_band_c > -120.0 else np.zeros_like(bands_c, dtype=bool)
    if active.any():
        diff = (bands_y - bands_c)[active]
        centered = diff - np.mean(diff)
        band_shape_rms = float(np.sqrt(np.mean(centered ** 2)))
        # Truly unchanged active musical bands (within 30 dB of peak, < 0.15 dB clip shift),
        # excluding empty -40..-50 dB subsonic/noise-floor bins.
        musical = bands_c >= max_band_c - 30.0
        unchanged = musical & (np.abs(bands_d - bands_c) <= 0.15)
        if unchanged.any():
            u_diff = (bands_y - bands_c)[unchanged]
            band_max_drift = float(np.max(np.abs(u_diff)))
        else:
            band_max_drift = None
    else:
        band_shape_rms = None
        band_max_drift = None

    # Quiet tail detection: check final 20% slice if its RMS is >= 25 dB below whole-track peak
    # and contains no masked samples.
    n = c.shape[0]
    tail_start = int(n * 0.8)
    tail_c = c[tail_start:]
    tail_y = y[tail_start:]
    tail_m = m[tail_start:]
    track_peak = max(float(np.max(np.abs(c))), 1e-12)
    tail_rms_c = math.sqrt(float(np.mean(tail_c ** 2)))
    tail_rms_y = math.sqrt(float(np.mean(tail_y ** 2)))
    quiet_tail_present = (
        tail_c.shape[0] >= max(64, round(0.1 * sr))
        and tail_rms_c > 1e-7
        and (20.0 * math.log10(tail_rms_c / track_peak) <= -25.0)
        and not bool(tail_m.any())
    )
    if quiet_tail_present:
        tail_drift_db = float(abs(20.0 * math.log10(max(tail_rms_y, 1e-12) / tail_rms_c)))
        if raw_candidate is not None:
            rc_tail = np.asarray(raw_candidate, dtype=np.float64)[tail_start:]
            tail_pre_diff = float(np.max(np.abs(rc_tail - d[tail_start:])))
        else:
            tail_pre_diff = None
    else:
        tail_drift_db = None
        tail_pre_diff = None

    return {
        "band_shape_rms_vs_clean_db": band_shape_rms,
        "unchanged_band_max_drift_db": band_max_drift,
        "quiet_tail_detected": bool(quiet_tail_present),
        "quiet_tail_rms_drift_db": tail_drift_db,
        "quiet_tail_max_abs_diff_pre_gain": tail_pre_diff,
    }


def check_stereo_safety_gates(
    metrics: dict,
    *,
    is_defect: bool,
    rubric: dict | None = None,
    require_pre_gain: bool = True,
) -> dict:
    """Evaluate preregistered hard stereo safety gates against measured metrics."""
    r = {**DEFAULT_STEREO_SAFETY_RUBRIC, **(rubric or {})}
    violations = []

    pol = metrics.get("polarity_violations")
    rel_diff = metrics.get("max_reliable_diff_pre_gain")
    raw_vs_deg = metrics.get("max_abs_diff_vs_degraded_pre_gain")
    if require_pre_gain and (pol is None or rel_diff is None or raw_vs_deg is None):
        violations.append("missing_pre_gain_candidate")
    if pol is not None and pol > r["max_polarity_violations"]:
        violations.append(f"polarity_violations={pol}")
    if rel_diff is not None and rel_diff > r["max_reliable_diff_pre_gain"]:
        violations.append(f"reliable_samples_modified={rel_diff:.3e}")

    if not is_defect:
        # Strict zero-modification rule on clean/intentional negative controls:
        # both pre-gain and delivered candidate must equal degraded exactly.
        if raw_vs_deg is not None and raw_vs_deg > 0.0:
            violations.append("negative_control_modified")
        deliv_vs_deg = metrics.get("max_abs_diff_vs_degraded_delivered")
        if deliv_vs_deg is not None and deliv_vs_deg > 0.0:
            violations.append("negative_control_delivered_modified")
        return {"passed": not violations, "violations": violations}

    if not metrics.get("local_ild_active_windows"):
        violations.append("no_active_stereo_windows")

    def _hi(key: str, limit_key: str, label: str):
        val = metrics.get(key)
        if val is not None and val > r[limit_key]:
            violations.append(f"{label}={val:.4f}>{r[limit_key]}")

    _hi("lr_corr_drift_vs_clean", "max_lr_corr_drift_vs_clean", "lr_corr_drift")
    _hi("ild_drift_vs_clean_db", "max_ild_drift_vs_clean_db", "ild_drift_db")
    _hi("local_ild_drift_p95_db", "max_local_ild_drift_p95_db", "local_ild_p95_db")
    _hi("phase_coherence_drift_vs_clean", "max_phase_coherence_drift_vs_clean", "phase_coh_drift")
    _hi("ipd_error_mean_deg", "max_ipd_error_mean_deg", "ipd_mean_deg")
    _hi("masked_ipd_error_mean_deg", "max_masked_ipd_error_mean_deg", "masked_ipd_mean_deg")
    _hi("itd_drift_samples", "max_itd_drift_samples", "itd_drift_samples")
    _hi("ms_ratio_drift_vs_clean_db", "max_ms_ratio_drift_vs_clean_db", "ms_ratio_drift_db")
    _hi("pre_echo_rise_db", "max_pre_echo_rise_db", "pre_echo_rise_db")
    _hi("quiet_tail_rms_drift_db", "max_quiet_tail_rms_drift_db", "quiet_tail_drift_db")
    _hi("band_shape_rms_vs_clean_db", "max_band_shape_rms_db", "band_shape_rms_db")
    _hi("unchanged_band_max_drift_db", "max_unchanged_band_drift_db", "unchanged_band_drift_db")

    mono_lvl = metrics.get("mono_sum_level_drift_vs_clean_db")
    if mono_lvl is not None and mono_lvl < r["min_mono_sum_level_drift_vs_clean_db"]:
        violations.append(f"mono_sum_cancellation={mono_lvl:.3f}dB<{r['min_mono_sum_level_drift_vs_clean_db']}")

    return {"passed": not violations, "violations": violations}


def evaluate_stereo_safety(
    clean: np.ndarray,
    degraded: np.ndarray,
    candidate: np.ndarray,
    sr: int,
    mask: np.ndarray | None = None,
    *,
    solver_mask: np.ndarray | None = None,
    raw_candidate: np.ndarray | None = None,
    is_defect: bool | None = None,
    rubric: dict | None = None,
) -> dict:
    """Compute all stereo safety metrics and evaluate hard E4/E5 safety gates.

    `mask` is the ground-truth clip mask (used for masked-frame IPD, quiet-tail,
    and polarity checks). `solver_mask` (defaults to `mask`) is the mask actually
    passed to the solver, used to enforce the exact `0.0` pre-gain reliable-sample
    preservation invariant on blind runs while transparently reporting any
    detector shoulder difference outside `mask`.
    """
    c = _validate_stereo(clean, "clean")
    d = _validate_stereo(degraded, "degraded")
    y = _validate_stereo(candidate, "candidate")
    if c.shape != d.shape or c.shape != y.shape:
        raise ValueError("clean, degraded, and candidate shapes must match")
    m = np.zeros(c.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if m.shape != c.shape:
        raise ValueError("mask shape must match PCM")
    sm_mask = m if solver_mask is None else np.asarray(solver_mask, dtype=bool)
    if sm_mask.shape != c.shape:
        raise ValueError("solver_mask shape must match PCM")
    if is_defect is None:
        is_defect = bool(m.any())

    gain_y = shared_stereo_gain(c, y)
    gain_d = shared_stereo_gain(c, d)
    y_al = gain_y * y
    d_al = gain_d * d

    lr_c = stereo_lr_correlation(c)
    lr_d = stereo_lr_correlation(d)
    lr_y = stereo_lr_correlation(y_al)

    ild_c = stereo_ild_db(c)
    ild_d = stereo_ild_db(d)
    ild_y = stereo_ild_db(y_al)
    local_ild = stereo_local_ild_drift(c, y_al, sr)

    xspec = stereo_cross_spectral_metrics(c, y_al, sr, mask=m if m.any() else None)
    ms = stereo_ms_and_mono_metrics(c, d, y_al, degraded_aligned=d_al)
    onset = onset_and_pre_echo_metrics(c, d, y_al, sr, mask=m, raw_candidate=raw_candidate)
    bands_tail = unchanged_bands_and_tail_metrics(c, d, y_al, sr, mask=m, raw_candidate=raw_candidate)

    polarity_viol = None
    reliable_diff = None
    outside_true_mask_diff = None
    raw_vs_deg = None
    if raw_candidate is not None:
        rc = _validate_stereo(raw_candidate, "raw_candidate")
        raw_vs_deg = float(np.max(np.abs(rc - d)))
        outside_solver = ~sm_mask
        reliable_diff = (float(np.max(np.abs(rc[outside_solver] - d[outside_solver])))
                         if outside_solver.any() else 0.0)
        outside_true = ~m
        outside_true_mask_diff = (float(np.max(np.abs(rc[outside_true] - d[outside_true])))
                                  if outside_true.any() else 0.0)
        check_m = sm_mask if sm_mask.any() else m
        if check_m.any():
            pos = check_m & (d > 0.0)
            neg = check_m & (d < 0.0)
            polarity_viol = int(
                np.count_nonzero(rc[pos] < d[pos] - 1e-7)
                + np.count_nonzero(rc[neg] > d[neg] + 1e-7)
            )
        else:
            polarity_viol = 0

    metrics = {
        "gain_to_clean": gain_y,
        "lr_corr_clean": lr_c,
        "lr_corr_degraded": lr_d,
        "lr_corr_candidate": lr_y,
        "lr_corr_drift_vs_clean": abs(lr_y - lr_c) if lr_c is not None and lr_y is not None else None,
        "lr_corr_drift_vs_degraded": abs(lr_y - lr_d) if lr_d is not None and lr_y is not None else None,
        "ild_clean_db": ild_c,
        "ild_degraded_db": ild_d,
        "ild_candidate_db": ild_y,
        "ild_drift_vs_clean_db": abs(ild_y - ild_c) if ild_c is not None and ild_y is not None else None,
        "ild_drift_vs_degraded_db": abs(ild_y - ild_d) if ild_d is not None and ild_y is not None else None,
        "local_ild_active_windows": local_ild["active_windows"],
        "local_ild_drift_p50_db": local_ild["p50_db"],
        "local_ild_drift_p95_db": local_ild["p95_db"],
        "local_ild_drift_max_db": local_ild["max_db"],
        **xspec,
        **ms,
        **onset,
        **bands_tail,
        "polarity_violations": polarity_viol,
        "max_reliable_diff_pre_gain": reliable_diff,
        "outside_true_mask_max_diff_pre_gain": outside_true_mask_diff,
        "max_abs_diff_vs_degraded_pre_gain": raw_vs_deg,
        "max_abs_diff_vs_degraded_delivered": float(np.max(np.abs(y - d))),
    }
    gates = check_stereo_safety_gates(metrics, is_defect=is_defect, rubric=rubric)
    return {"metrics": metrics, "gates": gates}


# --------------------------------------------------- Gapless boundary check

def evaluate_gapless_boundary(
    a_ref: np.ndarray,
    b_ref: np.ndarray,
    a_cand: np.ndarray,
    b_cand: np.ndarray,
    sr: int,
    *,
    seam_window_ms: float = 25.0,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    rubric: dict | None = None,
) -> dict:
    """Jointly check gapless track boundary continuity between track A and track B.

    When `mask_a` / `mask_b` are provided (e.g. comparing restored output against
    a clipped source), masked samples are excluded from level-jump, peak-growth,
    step, and slope discontinuity checks so legitimately reconstructed peaks at
    track edges do not trigger false click or peak-growth alarms.
    """
    ar = _validate_stereo(a_ref, "a_ref")
    br = _validate_stereo(b_ref, "b_ref")
    ac = _validate_stereo(a_cand, "a_cand")
    bc = _validate_stereo(b_cand, "b_cand")
    if ar.shape != ac.shape or br.shape != bc.shape or sr <= 0:
        raise ValueError("matching stereo shapes and positive sr required")
    if mask_a is not None and np.asarray(mask_a).shape != ar.shape:
        raise ValueError("mask_a shape must match a_ref")
    if mask_b is not None and np.asarray(mask_b).shape != br.shape:
        raise ValueError("mask_b shape must match b_ref")
    r = {**DEFAULT_STEREO_SAFETY_RUBRIC, **(rubric or {})}

    w = min(ar.shape[0], br.shape[0], max(8, round(seam_window_ms * sr / 1000.0)))
    unm_a = (np.ones((w, 2), dtype=bool) if mask_a is None
             else ~np.asarray(mask_a[-w:], dtype=bool))
    unm_b = (np.ones((w, 2), dtype=bool) if mask_b is None
             else ~np.asarray(mask_b[:w], dtype=bool))
    unm_join = np.concatenate((unm_a, unm_b), axis=0)

    ref_join = np.concatenate((ar[-w:], br[:w]), axis=0)
    cand_join = np.concatenate((ac[-w:], bc[:w]), axis=0)

    # Align whole pair by one shared gain computed on reliable unmasked seam samples
    # (if tracks had different gains, seam level jump shows up!).
    if unm_join.any():
        denom = float(np.sum(cand_join[unm_join] ** 2))
        g = (float(np.sum(ref_join[unm_join] * cand_join[unm_join]) / denom)
             if denom >= EPS else 1.0)
        g = g if g > 0.0 else 1.0
    else:
        g = shared_stereo_gain(ref_join, cand_join)
    ac_al = g * ac[-w:]
    bc_al = g * bc[:w]
    cand_al_join = np.concatenate((ac_al, bc_al), axis=0)

    # Local reference derivative scale (95th percentile step on unmasked adjacent pairs).
    if ref_join.shape[0] >= 2:
        adj_unm = unm_join[1:] & unm_join[:-1]
        diffs = np.abs(np.diff(ref_join, axis=0))
        local_steps = diffs[adj_unm] if adj_unm.any() else diffs.reshape(-1)
        step_scale = max(float(np.percentile(local_steps, 95)), 1e-3)
    else:
        step_scale = 1e-3

    # Step continuity across the exact boundary sample (only on channels unmasked on both sides).
    seam_unm_ch = unm_a[-1] & unm_b[0]
    seam_boundary_masked = not bool(seam_unm_ch.all())
    if seam_unm_ch.any():
        step_ref = float(np.max(np.abs(br[0, seam_unm_ch] - ar[-1, seam_unm_ch])))
        step_cand = float(np.max(np.abs(bc_al[0, seam_unm_ch] - ac_al[-1, seam_unm_ch])))
        seam_step_excess = float(max(0.0, step_cand - step_ref) / step_scale)
    else:
        step_ref = None
        step_cand = None
        seam_step_excess = None

    # Second-difference (slope discontinuity) across boundary when >= 2 unmasked samples on each side.
    if w >= 2:
        slope_unm_ch = unm_a[-2] & unm_a[-1] & unm_b[0] & unm_b[1]
        if slope_unm_ch.any():
            slope_ref = float(np.max(np.abs(
                (br[1, slope_unm_ch] - br[0, slope_unm_ch])
                - (ar[-1, slope_unm_ch] - ar[-2, slope_unm_ch])
            )))
            slope_cand = float(np.max(np.abs(
                (bc_al[1, slope_unm_ch] - bc_al[0, slope_unm_ch])
                - (ac_al[-1, slope_unm_ch] - ac_al[-2, slope_unm_ch])
            )))
        else:
            slope_ref = slope_cand = None
    else:
        slope_ref = slope_cand = None

    # Level continuity across boundary on reliable (unmasked) seam samples.
    if unm_a.any() and unm_b.any():
        rms_a_ref = math.sqrt(float(np.mean(ar[-w:][unm_a] ** 2)))
        rms_b_ref = math.sqrt(float(np.mean(br[:w][unm_b] ** 2)))
        rms_a_cand = math.sqrt(float(np.mean(ac[-w:][unm_a] ** 2)))
        rms_b_cand = math.sqrt(float(np.mean(bc[:w][unm_b] ** 2)))
        if min(rms_a_ref, rms_b_ref, rms_a_cand, rms_b_cand) > 1e-7:
            ratio_ref_db = 20.0 * math.log10(rms_b_ref / rms_a_ref)
            ratio_cand_db = 20.0 * math.log10(rms_b_cand / rms_a_cand)
            level_jump_db = float(abs(ratio_cand_db - ratio_ref_db))
        else:
            level_jump_db = None
    else:
        level_jump_db = None

    # Peak growth across the stitched seam on unmasked samples at matched gain.
    if unm_join.any():
        peak_ref = max(float(np.max(np.abs(ref_join[unm_join]))), 1e-6)
        peak_cand = max(float(np.max(np.abs(cand_al_join[unm_join]))), 1e-6)
        peak_growth_db = float(20.0 * math.log10(peak_cand / peak_ref))
    else:
        peak_growth_db = None

    violations = []
    if level_jump_db is not None and level_jump_db > r["max_gapless_level_jump_db"]:
        violations.append(f"gapless_level_jump_db={level_jump_db:.3f}>{r['max_gapless_level_jump_db']}")
    if peak_growth_db is not None and peak_growth_db > r["max_gapless_seam_peak_growth_db"]:
        violations.append(f"gapless_seam_peak_growth_db={peak_growth_db:.3f}>{r['max_gapless_seam_peak_growth_db']}")
    if (step_cand is not None and step_ref is not None and seam_step_excess is not None
            and step_cand > max(0.05, 4.0 * max(step_ref, step_scale))
            and seam_step_excess > 3.0):
        violations.append(f"gapless_seam_click_step={step_cand:.4f}")

    return {
        "seam_window_samples": int(w),
        "seam_boundary_masked": seam_boundary_masked,
        "seam_step_ref": step_ref,
        "seam_step_cand_aligned": step_cand,
        "seam_step_excess_ratio": seam_step_excess,
        "seam_slope_ref": slope_ref,
        "seam_slope_cand_aligned": slope_cand,
        "seam_level_jump_db": level_jump_db,
        "seam_peak_growth_db": peak_growth_db,
        "passed": not violations,
        "violations": violations,
    }


# -------------------------------------------- Album-level headroom coordination

def plan_album_headroom(
    track_proposals: list[dict],
    *,
    ceiling_db: float = -1.0,
    max_attenuation_db: float = -24.0,
    relative_level_tol_db: float = 0.1,
) -> dict:
    """Coordinate Preserve attenuation-only album headroom per spec v1 section 3.

    Each entry in `track_proposals` must provide:
      - `track_id`: str
      - `status`: `processed_candidate` | `kept_original` | `qc_fallback_original` | `error`
      - `pre_gain_tp_dbfs`: float | None (required for `processed_candidate`)

    Contract:
      - Attenuation-only (`gain_db <= 0.0`).
      - When all tracks are repaired (or all repaired tracks need `0.0 dB` gain
        within `relative_level_tol_db`), one unified `album_gain_db = min(gains)`
        preserves 0.00 dB relative album levels and `auto_album_delivery = True`.
      - When immutable `kept_original`/`qc_fallback_original` tracks coexist with
        repaired tracks requiring attenuation (`gain < -relative_level_tol_db`),
        kept tracks remain byte-for-byte unchanged, repaired tracks share one
        unified gain to preserve relative levels among themselves, and the plan
        flags `compatible=False`, `needs_review=True`, `auto_album_delivery=False`
        instead of silently reshuffling album levels.
    """
    if not track_proposals:
        raise ValueError("track_proposals must be nonempty")

    repaired = []
    kept = []
    errors = []
    for item in track_proposals:
        tid = str(item["track_id"])
        st = str(item["status"])
        if st == "processed_candidate":
            tp = item.get("pre_gain_tp_dbfs")
            if tp is None or not math.isfinite(float(tp)):
                raise ValueError(f"track {tid}: processed_candidate missing finite pre_gain_tp_dbfs")
            prop_gain = min(0.0, float(ceiling_db) - float(tp))
            repaired.append({"track_id": tid, "pre_gain_tp_dbfs": float(tp), "proposed_gain_db": float(prop_gain)})
        elif st in ("kept_original", "qc_fallback_original"):
            kept.append({"track_id": tid, "status": st})
        else:
            errors.append({"track_id": tid, "status": st})

    if errors:
        return {
            "policy": "album_contains_errors",
            "ceiling_db": float(ceiling_db),
            "album_gain_db": None,
            "compatible": False,
            "needs_review": True,
            "auto_album_delivery": False,
            "max_relative_level_shift_db": None,
            "reason": "one_or_more_tracks_failed",
            "per_track_gain_db": {t["track_id"]: 0.0 for t in track_proposals},
            "repaired_tracks": [t["track_id"] for t in repaired],
            "kept_tracks": [t["track_id"] for t in kept],
            "error_tracks": [t["track_id"] for t in errors],
        }

    if not repaired:
        return {
            "policy": "all_kept_original",
            "ceiling_db": float(ceiling_db),
            "album_gain_db": 0.0,
            "compatible": True,
            "needs_review": False,
            "auto_album_delivery": True,
            "max_relative_level_shift_db": 0.0,
            "reason": None,
            "per_track_gain_db": {t["track_id"]: 0.0 for t in track_proposals},
            "repaired_tracks": [],
            "kept_tracks": [t["track_id"] for t in kept],
            "error_tracks": [],
        }

    unified_repaired_gain = min(t["proposed_gain_db"] for t in repaired)
    if unified_repaired_gain < float(max_attenuation_db):
        return {
            "policy": "rejected_excessive_album_attenuation",
            "ceiling_db": float(ceiling_db),
            "album_gain_db": None,
            "proposed_album_gain_db": float(unified_repaired_gain),
            "compatible": False,
            "needs_review": True,
            "auto_album_delivery": False,
            "max_relative_level_shift_db": None,
            "reason": "required_album_headroom_attenuation_over_24_db",
            "per_track_gain_db": {t["track_id"]: 0.0 for t in track_proposals},
            "repaired_tracks": [t["track_id"] for t in repaired],
            "kept_tracks": [t["track_id"] for t in kept],
            "error_tracks": [],
        }

    if not kept or abs(unified_repaired_gain) <= float(relative_level_tol_db):
        gains = {t["track_id"]: (float(unified_repaired_gain) if t["status"] == "processed_candidate" else 0.0)
                 for t in track_proposals}
        shift = 0.0 if not kept else abs(float(unified_repaired_gain))
        return {
            "policy": "unified_album_gain",
            "ceiling_db": float(ceiling_db),
            "album_gain_db": float(unified_repaired_gain),
            "compatible": True,
            "needs_review": False,
            "auto_album_delivery": True,
            "max_relative_level_shift_db": float(shift),
            "reason": None,
            "per_track_gain_db": gains,
            "repaired_tracks": [t["track_id"] for t in repaired],
            "kept_tracks": [t["track_id"] for t in kept],
            "error_tracks": [],
        }

    # Mixed album: kept tracks must remain byte-for-byte unchanged (0.0 dB),
    # while repaired tracks require attenuation under -1 dBTP. Never mutate
    # kept tracks silently; emit explicit needs_review diagnostics.
    gains = {t["track_id"]: (float(unified_repaired_gain) if t["status"] == "processed_candidate" else 0.0)
             for t in track_proposals}
    return {
        "policy": "mixed_kept_and_attenuated_incompatible",
        "ceiling_db": float(ceiling_db),
        "album_gain_db": None,
        "repaired_group_gain_db": float(unified_repaired_gain),
        "compatible": False,
        "needs_review": True,
        "auto_album_delivery": False,
        "max_relative_level_shift_db": float(abs(unified_repaired_gain)),
        "reason": "unified_album_gain_incompatible_with_immutable_kept_tracks",
        "per_track_gain_db": gains,
        "repaired_tracks": [t["track_id"] for t in repaired],
        "kept_tracks": [t["track_id"] for t in kept],
        "error_tracks": [],
    }
