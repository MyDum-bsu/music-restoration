"""Reproducible, MIT-licensed hard-declipping comparators for stereo/mono PCM.

All algorithms are implemented from first principles from published DSP
formulations without copying GPL/MATLAB code:

- `declip_aspade`: Analysis SPADE (Kitic et al., TASLP 2015; Zaviska et al.,
  JSTSP 2021) using a redundant 2x-oversampled Parseval DFT frame and batched
  per-frame ADMM with adaptive hard thresholding.
- `declip_ss_pew`: Psychoacoustically-weighted iteratively reweighted l1
  synthesis declipper (SS-PEW equivalent; Zaviska et al., ICASSP 2019 /
  JSTSP 2021) using exact Douglas-Rachford splitting in a redundant 2x
  Parseval Gabor frame with Terhardt ATH + spectral-envelope weighting.
- `declip_regularized_ar`: Regularized autoregressive declipper (Janssen et
  al., 1986; Mokry & Rajmic, 2024) alternating Tikhonov-regularized
  Yule-Walker AR estimation with exact box-constrained quadratic signal
  reconstruction per clipped frame.

Every method preserves native channels/rate, keeps unmasked reliable samples
unchanged before output headroom gain, enforces clipping polarity inequalities
on masked samples, and shares one headroom policy (`apply_headroom_policy`).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_toeplitz

from rnext.dsp.contour import true_peak_dbfs
from rnext.dsp.sparse import ReconstructionRejected


def _validate_inputs(x: np.ndarray, sr: int, mask: np.ndarray) -> None:
    if x.ndim != 2 or not x.size or x.shape[1] not in (1, 2) or not np.isfinite(x).all():
        raise ValueError("expected nonempty finite mono/stereo PCM")
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("floating PCM required")
    if mask.shape != x.shape or mask.dtype != bool or np.any(mask & (x == 0)):
        raise ValueError("clipping mask must be boolean, match PCM and exclude zero samples")
    if sr <= 0:
        raise ValueError("sample rate must be positive")


def _rate_scaled_frame(sr: int, base_44k: int, min_frame: int = 512, max_frame: int = 8192) -> int:
    scaled = int(2 ** round(np.log2(max(1.0, base_44k * sr / 44100.0))))
    return max(min_frame, min(max_frame, scaled))


def _hann_periodic(frame: int) -> np.ndarray:
    return np.hanning(frame + 1)[:-1].astype(np.float64)


def _pad_and_slice_frames(sig: np.ndarray, msk: np.ndarray, frame: int, hop: int):
    """Pad signal/mask by one frame on each side so COLA normalization is exact."""
    n = sig.size
    pad_l = frame
    rem = (n + 2 * frame - frame) % hop
    pad_r = frame + (0 if rem == 0 else hop - rem)
    sig_p = np.pad(sig, (pad_l, pad_r), mode="constant")
    msk_p = np.pad(msk, (pad_l, pad_r), mode="constant", constant_values=False)
    starts = np.arange(0, sig_p.size - frame + 1, hop)
    return sig_p, msk_p, starts, pad_l


def _project_consistent_batch(prop: np.ndarray, obs: np.ndarray,
                              pos: np.ndarray, neg: np.ndarray,
                              rel: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection onto the clipping consistency set per frame."""
    out = prop.copy()
    out[rel] = obs[rel]
    out[pos] = np.maximum(out[pos], obs[pos])
    out[neg] = np.minimum(out[neg], obs[neg])
    return out


def _hard_threshold_rfft(values: np.ndarray, k_vec: np.ndarray) -> np.ndarray:
    """Keep the top-k magnitude bins per frame in rfft space (conjugate-symmetric)."""
    n_frames, n_bins = values.shape
    mags = np.abs(values)
    out = np.zeros_like(values)
    for i in range(n_frames):
        k = int(min(n_bins, max(1, k_vec[i])))
        if k >= n_bins:
            out[i] = values[i]
        else:
            cutoff = np.partition(mags[i], n_bins - k)[n_bins - k]
            out[i] = values[i] * (mags[i] >= cutoff)
    return out


def _terhardt_ath_weights(sr: int, n_fft: int) -> np.ndarray:
    """Mild psychoacoustic frequency weights from Terhardt's ATH curve."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    f_khz = np.maximum(freqs / 1000.0, 0.02)
    ath_db = (3.64 * (f_khz ** -0.8)
              - 6.5 * np.exp(-0.6 * ((f_khz - 3.3) ** 2))
              + 1e-3 * (f_khz ** 4))
    ath_db = np.clip(ath_db, -10.0, 75.0)
    # Convert ATH relative to minimum (~ -5 dB at 3.5 kHz) into gentle linear weights.
    rel_db = ath_db - float(np.min(ath_db))
    weights = np.clip(10.0 ** (rel_db / 40.0), 1.0, 12.0)
    # Keep DC/very-low bass gently weighted so fundamental energy is preserved.
    weights[freqs < 40.0] = np.minimum(weights[freqs < 40.0], 2.5)
    return weights.astype(np.float64)


def _enforce_consistency(repaired: np.ndarray, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Enforce exact reliable-sample equality and clipping polarity inequalities."""
    out = repaired.astype(np.float64, copy=True)
    out[~mask] = x[~mask]
    pos = mask & (x > 0)
    neg = mask & (x < 0)
    out[pos] = np.maximum(out[pos], x[pos])
    out[neg] = np.minimum(out[neg], x[neg])
    out = out.astype(x.dtype)
    if not np.isfinite(out).all():
        raise ReconstructionRejected("reconstructed PCM contains nonfinite samples")
    return out


# ---------------------------------------------------------------- A-SPADE (MIT)

def _aspade_channel(
    sig: np.ndarray,
    msk: np.ndarray,
    frame: int,
    redundancy: int,
    step_k: int,
    max_iter: int,
    eps: float,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    if not msk.any():
        return sig.copy(), {"clipped_frames": 0, "total_frames": 0, "mean_iters": 0.0}

    hop = frame // 4
    n_fft = frame * redundancy
    win = _hann_periodic(frame)
    sig_p, msk_p, starts, pad_l = _pad_and_slice_frames(sig, msk, frame, hop)

    # Identify frames overlapping at least one masked sample with nonzero window weight.
    w_pos = win > 1e-12
    clipped_idx = []
    for fi, s in enumerate(starts):
        if np.any(msk_p[s:s + frame] & w_pos):
            clipped_idx.append(fi)

    acc = np.zeros_like(sig_p)
    norm = np.zeros_like(sig_p)
    win_sq = win ** 2

    # Unclipped frames contribute directly with zero ADMM work.
    clipped_set = set(clipped_idx)
    for fi, s in enumerate(starts):
        if fi not in clipped_set:
            acc[s:s + frame] += sig_p[s:s + frame] * win_sq
            norm[s:s + frame] += win_sq

    total_iters = 0
    converged = 0
    for b0 in range(0, len(clipped_idx), batch_size):
        batch_fis = clipped_idx[b0:b0 + batch_size]
        b_starts = starts[batch_fis]
        raw_frames = np.stack([sig_p[s:s + frame] for s in b_starts])
        raw_masks = np.stack([msk_p[s:s + frame] for s in b_starts])

        # Windowed frames: because win >= 0, clipping inequalities scale by win.
        obs_w = raw_frames * win[None, :]
        scales = np.maximum(np.max(np.abs(obs_w), axis=1, keepdims=True), 1e-12)
        obs = obs_w / scales
        pos = raw_masks & (obs > 0.0) & w_pos[None, :]
        neg = raw_masks & (obs < 0.0) & w_pos[None, :]
        rel = ~(pos | neg)

        n_b = obs.shape[0]
        x_cur = obs.copy()
        c_cur = np.fft.rfft(x_cur, n=n_fft, axis=-1, norm="ortho")
        dual = np.zeros_like(c_cur)
        obs_norm = np.maximum(np.linalg.norm(obs, axis=1), 1e-12)
        active = np.ones(n_b, dtype=bool)
        iters_used = np.zeros(n_b, dtype=int)
        k_val = max(2, step_k)

        for it in range(max_iter):
            if not active.any():
                break
            idx = np.flatnonzero(active)
            # 1. Hard-threshold in redundant analysis domain (rfft space).
            z_act = _hard_threshold_rfft(c_cur[idx] + dual[idx],
                                         np.full(idx.size, k_val, dtype=int))
            # 2. Parseval synthesis + exact projection onto consistent set.
            prop = np.fft.irfft(z_act - dual[idx], n=n_fft, axis=-1, norm="ortho")[:, :frame]
            x_new = _project_consistent_batch(prop, obs[idx], pos[idx], neg[idx], rel[idx])
            # 3. Analysis + dual update.
            c_new = np.fft.rfft(x_new, n=n_fft, axis=-1, norm="ortho")
            diff = c_new - z_act
            dual[idx] += diff
            x_cur[idx] = x_new
            c_cur[idx] = c_new
            iters_used[idx] = it + 1

            # Parseval residual norm (rfft counts interior bins twice).
            diff_sq = np.abs(diff) ** 2
            res_energy = diff_sq[:, 0] + diff_sq[:, -1] + 2.0 * np.sum(diff_sq[:, 1:-1], axis=1)
            rel_res = np.sqrt(np.maximum(res_energy, 0.0)) / obs_norm[idx]
            done = (it >= 4) & (rel_res <= eps)
            if done.any():
                active[idx[done]] = False
                converged += int(np.count_nonzero(done))
            k_val = min(c_cur.shape[1], k_val + step_k)

        total_iters += int(iters_used.sum())
        rebuilt_w = x_cur * scales
        for j, s in enumerate(b_starts):
            acc[s:s + frame] += rebuilt_w[j] * win
            norm[s:s + frame] += win_sq

    repaired = acc[pad_l:pad_l + sig.size] / np.maximum(norm[pad_l:pad_l + sig.size], 1e-12)
    return repaired, {
        "clipped_frames": len(clipped_idx),
        "total_frames": int(starts.size),
        "converged_frames": converged,
        "mean_iters": float(total_iters / max(1, len(clipped_idx))),
    }


def declip_aspade(
    x: np.ndarray,
    sr: int,
    mask: np.ndarray,
    *,
    frame: int | None = None,
    redundancy: int = 2,
    step_k: int = 4,
    max_iter: int = 120,
    eps: float = 1e-3,
    batch_size: int = 64,
) -> tuple[np.ndarray, dict]:
    """Analysis SPADE (MIT first-principles) in a 2x-redundant Parseval DFT frame."""
    _validate_inputs(x, sr, mask)
    if redundancy not in (1, 2, 4) or step_k < 1 or max_iter < 1:
        raise ValueError("invalid A-SPADE configuration")
    frame = _rate_scaled_frame(sr, 2048) if frame is None else int(frame)
    if frame < 64 or frame % 4:
        raise ValueError("frame must be >= 64 and divisible by 4")
    if not mask.any():
        return x.copy(), {"method": "aspade_mit", "frame": frame,
                          "redundancy": redundancy, "masked_samples": 0,
                          "channels": [], "reliable_samples_preserved": True}

    out = x.astype(np.float64, copy=True)
    ch_reports = []
    for ch in range(x.shape[1]):
        repaired_ch, info = _aspade_channel(
            x[:, ch].astype(np.float64), mask[:, ch], frame, redundancy,
            step_k, max_iter, eps, batch_size)
        out[:, ch] = repaired_ch
        ch_reports.append({"channel": ch, **info})

    out = _enforce_consistency(out, x, mask)
    return out, {
        "method": "aspade_mit",
        "frame": frame,
        "redundancy": redundancy,
        "step_k": step_k,
        "max_iter": max_iter,
        "eps": eps,
        "masked_samples": int(mask.sum()),
        "channels": ch_reports,
        "reliable_samples_preserved": True,
    }


# ---------------------------------------------------------------- SS-PEW (MIT)

def _ss_pew_channel(
    sig: np.ndarray,
    msk: np.ndarray,
    sr: int,
    frame: int,
    redundancy: int,
    outer_iters: int,
    dr_iters: int,
    gamma: float,
    delta: float,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    if not msk.any():
        return sig.copy(), {"clipped_frames": 0, "total_frames": 0}

    hop = frame // 4
    n_fft = frame * redundancy
    win = _hann_periodic(frame)
    w_pos = win > 1e-12
    ath_w = _terhardt_ath_weights(sr, n_fft)  # [n_bins]
    sig_p, msk_p, starts, pad_l = _pad_and_slice_frames(sig, msk, frame, hop)

    clipped_idx = [fi for fi, s in enumerate(starts)
                   if np.any(msk_p[s:s + frame] & w_pos)]
    clipped_set = set(clipped_idx)

    acc = np.zeros_like(sig_p)
    norm = np.zeros_like(sig_p)
    win_sq = win ** 2
    for fi, s in enumerate(starts):
        if fi not in clipped_set:
            acc[s:s + frame] += sig_p[s:s + frame] * win_sq
            norm[s:s + frame] += win_sq

    # 7-bin triangular smoothing kernel for initial spectral-envelope masking.
    kernel = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()

    for b0 in range(0, len(clipped_idx), batch_size):
        batch_fis = clipped_idx[b0:b0 + batch_size]
        b_starts = starts[batch_fis]
        raw_frames = np.stack([sig_p[s:s + frame] for s in b_starts])
        raw_masks = np.stack([msk_p[s:s + frame] for s in b_starts])

        obs_w = raw_frames * win[None, :]
        scales = np.maximum(np.max(np.abs(obs_w), axis=1, keepdims=True), 1e-12)
        obs = obs_w / scales
        pos = raw_masks & (obs > 0.0) & w_pos[None, :]
        neg = raw_masks & (obs < 0.0) & w_pos[None, :]
        rel = ~(pos | neg)

        # Initial synthesis coefficients via Parseval analysis D^* obs.
        c = np.fft.rfft(obs, n=n_fft, axis=-1, norm="ortho")
        mag0 = np.abs(c)
        # Smooth spectral envelope per frame for psychoacoustic masking weights.
        env = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, mag0)
        peak_mag = np.maximum(np.max(mag0, axis=1, keepdims=True), 1e-12)
        weights = ath_w[None, :] / (env + delta * peak_mag)
        weights /= np.maximum(np.median(weights, axis=1, keepdims=True), 1e-12)

        for outer in range(outer_iters):
            u = c.copy()
            # Threshold scaled to frame coefficient scale for loudness invariance.
            thresh = gamma * peak_mag * weights
            for _ in range(dr_iters):
                # Prox of weighted l1 norm: complex soft-thresholding.
                u_abs = np.abs(u)
                shrink = np.maximum(0.0, 1.0 - thresh / np.maximum(u_abs, 1e-20))
                z = u * shrink
                refl = 2.0 * z - u
                # Exact Parseval projection onto {c : D c in Gamma}:
                # Pi(r) = r + D^*(Pi_Gamma(D r) - D r).
                dr_time = np.fft.irfft(refl, n=n_fft, axis=-1, norm="ortho")[:, :frame]
                proj_time = _project_consistent_batch(dr_time, obs, pos, neg, rel)
                c = refl + np.fft.rfft(proj_time - dr_time, n=n_fft, axis=-1, norm="ortho")
                u = u + 1.5 * (c - z)

            if outer + 1 < outer_iters:
                # Candes-Wakin-Boyd iterative reweighting with psychoacoustic prior.
                mag_c = np.abs(c)
                peak_c = np.maximum(np.max(mag_c, axis=1, keepdims=True), 1e-12)
                weights = ath_w[None, :] / (mag_c + delta * peak_c)
                weights /= np.maximum(np.median(weights, axis=1, keepdims=True), 1e-12)

        rebuilt_w = np.fft.irfft(c, n=n_fft, axis=-1, norm="ortho")[:, :frame]
        rebuilt_w = _project_consistent_batch(rebuilt_w, obs, pos, neg, rel) * scales
        for j, s in enumerate(b_starts):
            acc[s:s + frame] += rebuilt_w[j] * win
            norm[s:s + frame] += win_sq

    repaired = acc[pad_l:pad_l + sig.size] / np.maximum(norm[pad_l:pad_l + sig.size], 1e-12)
    return repaired, {
        "clipped_frames": len(clipped_idx),
        "total_frames": int(starts.size),
    }


def declip_ss_pew(
    x: np.ndarray,
    sr: int,
    mask: np.ndarray,
    *,
    frame: int | None = None,
    redundancy: int = 2,
    outer_iters: int = 3,
    dr_iters: int = 60,
    gamma: float = 0.08,
    delta: float = 0.01,
    batch_size: int = 64,
) -> tuple[np.ndarray, dict]:
    """Psychoacoustically-weighted IRL1 synthesis declipper (SS-PEW equivalent)."""
    _validate_inputs(x, sr, mask)
    if redundancy not in (1, 2, 4) or outer_iters < 1 or dr_iters < 1 or gamma <= 0 or delta <= 0:
        raise ValueError("invalid SS-PEW configuration")
    frame = _rate_scaled_frame(sr, 2048) if frame is None else int(frame)
    if frame < 64 or frame % 4:
        raise ValueError("frame must be >= 64 and divisible by 4")
    if not mask.any():
        return x.copy(), {"method": "ss_pew_mit", "frame": frame,
                          "redundancy": redundancy, "masked_samples": 0,
                          "channels": [], "reliable_samples_preserved": True}

    out = x.astype(np.float64, copy=True)
    ch_reports = []
    for ch in range(x.shape[1]):
        repaired_ch, info = _ss_pew_channel(
            x[:, ch].astype(np.float64), mask[:, ch], sr, frame, redundancy,
            outer_iters, dr_iters, gamma, delta, batch_size)
        out[:, ch] = repaired_ch
        ch_reports.append({"channel": ch, **info})

    out = _enforce_consistency(out, x, mask)
    return out, {
        "method": "ss_pew_mit",
        "frame": frame,
        "redundancy": redundancy,
        "outer_iters": outer_iters,
        "dr_iters": dr_iters,
        "gamma": gamma,
        "delta": delta,
        "masked_samples": int(mask.sum()),
        "channels": ch_reports,
        "reliable_samples_preserved": True,
    }


# -------------------------------------------------------- Regularized AR (MIT)

def _fit_regularized_ar(x: np.ndarray, order: int, ar_reg: float) -> np.ndarray:
    """Fit Tikhonov-regularized Yule-Walker AR filter g = [1, a_1, ..., a_p]."""
    n = x.size
    p = min(order, max(2, n // 4))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, n=n_fft)
    acf = np.fft.irfft(spec.real ** 2 + spec.imag ** 2, n=n_fft)[:p + 1]
    r0 = float(acf[0])
    if r0 <= 1e-20:
        return np.array([1.0], dtype=np.float64)
    r = acf / r0
    r[0] = 1.0 + ar_reg
    try:
        a = solve_toeplitz(r[:p], -r[1:p + 1])
    except (np.linalg.LinAlgError, ValueError):
        return np.array([1.0], dtype=np.float64)
    if not np.isfinite(a).all():
        return np.array([1.0], dtype=np.float64)
    return np.concatenate(([1.0], a))


def _build_ar_gram_submatrix(c_idx: np.ndarray, g: np.ndarray, n: int) -> np.ndarray:
    """Build exact submatrix H_CC = (A(g)^T A(g))[C, C] for causal AR residual A(g).

    The residual operator e = np.convolve(x, g, mode="valid") has length n - p
    and entries e[t - p] = sum_{k=0}^p g[k] x[t - k] for t in [p, n - 1].
    Hence A[t - p, j] = g[t - j] for j <= t <= j + p, and:
      (A^T A)[i, j] = sum_{t = max(p, i, j)}^{min(n - 1, i + p, j + p)} g[t - i] g[t - j].
    For interior indices (p <= idx <= n - 1 - p) this equals acf_g[|i - j|].
    """
    p = g.size - 1
    acf_g = np.correlate(g, g, mode="full")[p:]  # length p + 1
    i_grid, j_grid = np.broadcast_arrays(c_idx[:, None], c_idx[None, :])
    d_ij = np.abs(i_grid - j_grid)
    h_cc = np.where(d_ij <= p, acf_g[np.minimum(d_ij, p)], 0.0)
    near_edge = (c_idx < p) | (c_idx > n - 1 - p)
    if near_edge.any():
        edge_pos = np.flatnonzero(near_edge)
        for row_k in edge_pos:
            ii = int(c_idx[row_k])
            cols = np.flatnonzero(np.abs(c_idx - ii) <= p)
            for col_k in cols:
                jj = int(c_idx[col_k])
                t0 = max(p, ii, jj)
                t1 = min(n - 1, ii + p, jj + p)
                val = (float(np.dot(g[t0 - ii:t1 - ii + 1], g[t0 - jj:t1 - jj + 1]))
                       if t0 <= t1 else 0.0)
                h_cc[row_k, col_k] = val
                h_cc[col_k, row_k] = val
    return h_cc


def _solve_ar_frame_qp(
    x_init: np.ndarray,
    obs: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    g: np.ndarray,
    qp_reg: float,
    qp_iters: int,
) -> np.ndarray:
    """Minimize 0.5 ||A(g) x||^2 + 0.5 * qp_reg * ||x_C - obs_C||^2 s.t. box."""
    n = obs.size
    p = g.size - 1
    if p < 1:
        return obs.copy()
    c_idx = np.sort(np.concatenate((pos_idx, neg_idx)))
    m = c_idx.size
    if m == 0:
        return obs.copy()

    # Exact linear operator H(v) = A(g)^T A(g) v via two 1D convolutions.
    g_rev = g[::-1]

    def apply_h(v: np.ndarray) -> np.ndarray:
        e = np.convolve(v, g, mode="valid")
        return np.convolve(e, g_rev, mode="full")

    # Separate reliable contribution: x = x_rel + E_C x_C.
    x_rel = obs.copy()
    x_rel[c_idx] = 0.0
    rhs_c = apply_h(x_rel)[c_idx] - qp_reg * obs[c_idx]

    is_pos = np.isin(c_idx, pos_idx)
    obs_c = obs[c_idx]

    def project_c(v_c: np.ndarray) -> np.ndarray:
        return np.where(is_pos, np.maximum(v_c, obs_c), np.minimum(v_c, obs_c))

    if m <= 320:
        h_cc = _build_ar_gram_submatrix(c_idx, g, n)
        h_cc[np.diag_indices(m)] += qp_reg
        try:
            cf = cho_factor(h_cc, lower=True, check_finite=False)
            x_unc = cho_solve(cf, -rhs_c, check_finite=False)
            if np.isfinite(x_unc).all():
                x_proj = project_c(x_unc)
                if np.allclose(x_unc, x_proj, atol=1e-12, rtol=0.0):
                    out = obs.copy()
                    out[c_idx] = x_unc
                    return out
        except np.linalg.LinAlgError:
            pass
        # Active box constraints: fast FISTA on dense H_CC.
        lip = float(np.max(np.sum(np.abs(h_cc), axis=1)))
        step = 1.0 / max(lip, 1e-12)
        z = project_c(x_init[c_idx])
        y_k = z.copy()
        t_k = 1.0
        for _ in range(qp_iters):
            grad = h_cc @ y_k + rhs_c
            z_next = project_c(y_k - step * grad)
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
            y_k = z_next + ((t_k - 1.0) / t_next) * (z_next - z)
            z = z_next
            t_k = t_next
        out = obs.copy()
        out[c_idx] = z
        return out

    # Matrix-free FISTA for heavily clipped frames (|C| > 320).
    lip = float(np.sum(np.abs(g)) ** 2) + qp_reg
    step = 1.0 / max(lip, 1e-12)
    z = project_c(x_init[c_idx])
    y_k = z.copy()
    t_k = 1.0
    buf = obs.copy()
    for _ in range(qp_iters):
        buf[c_idx] = y_k
        grad = apply_h(buf)[c_idx] + qp_reg * (y_k - obs_c)
        z_next = project_c(y_k - step * grad)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
        y_k = z_next + ((t_k - 1.0) / t_next) * (z_next - z)
        z = z_next
        t_k = t_next
    out = obs.copy()
    out[c_idx] = z
    return out


def _reg_ar_channel(
    sig: np.ndarray,
    msk: np.ndarray,
    frame: int,
    order: int,
    ar_iters: int,
    ar_reg: float,
    qp_reg: float,
    qp_iters: int,
) -> tuple[np.ndarray, dict]:
    if not msk.any():
        return sig.copy(), {"clipped_frames": 0, "total_frames": 0}

    hop = frame // 2
    win = _hann_periodic(frame)
    sig_p, msk_p, starts, pad_l = _pad_and_slice_frames(sig, msk, frame, hop)

    acc = np.zeros_like(sig_p)
    norm = np.zeros_like(sig_p)
    clipped_frames = 0
    fallback_frames = 0

    for s in starts:
        seg = sig_p[s:s + frame]
        seg_m = msk_p[s:s + frame]
        if not seg_m.any():
            acc[s:s + frame] += seg * win
            norm[s:s + frame] += win
            continue
        clipped_frames += 1
        scale = max(float(np.max(np.abs(seg))), 1e-12)
        obs = seg / scale
        pos_idx = np.flatnonzero(seg_m & (obs > 0.0))
        neg_idx = np.flatnonzero(seg_m & (obs < 0.0))
        cur = obs.copy()
        for _ in range(ar_iters):
            g = _fit_regularized_ar(cur, order, ar_reg)
            cur = _solve_ar_frame_qp(cur, obs, pos_idx, neg_idx, g, qp_reg, qp_iters)
        # Guard against rare AR instability in severely clipped frames (recorded explicitly).
        if not np.isfinite(cur).all() or float(np.max(np.abs(cur))) > 16.0:
            cur = obs
            fallback_frames += 1
        acc[s:s + frame] += (cur * scale) * win
        norm[s:s + frame] += win

    repaired = acc[pad_l:pad_l + sig.size] / np.maximum(norm[pad_l:pad_l + sig.size], 1e-12)
    return repaired, {
        "clipped_frames": clipped_frames,
        "fallback_frames": fallback_frames,
        "total_frames": int(starts.size),
    }


def declip_regularized_ar(
    x: np.ndarray,
    sr: int,
    mask: np.ndarray,
    *,
    frame: int | None = None,
    order: int = 48,
    ar_iters: int = 12,
    ar_reg: float = 1e-5,
    qp_reg: float = 1e-7,
    qp_iters: int = 60,
) -> tuple[np.ndarray, dict]:
    """Regularized autoregressive (Janssen / Mokry-Rajmic 2024) declipper."""
    _validate_inputs(x, sr, mask)
    if order < 4 or ar_iters < 1 or ar_reg <= 0 or qp_reg < 0 or qp_iters < 1:
        raise ValueError("invalid regularized AR configuration")
    frame = _rate_scaled_frame(sr, 1024, min_frame=256, max_frame=4096) if frame is None else int(frame)
    if frame < 64 or frame % 2:
        raise ValueError("frame must be >= 64 and even")
    if not mask.any():
        return x.copy(), {"method": "reg_ar_mit", "frame": frame, "order": order,
                          "masked_samples": 0, "channels": [],
                          "reliable_samples_preserved": True}

    out = x.astype(np.float64, copy=True)
    ch_reports = []
    for ch in range(x.shape[1]):
        repaired_ch, info = _reg_ar_channel(
            x[:, ch].astype(np.float64), mask[:, ch], frame, order,
            ar_iters, ar_reg, qp_reg, qp_iters)
        out[:, ch] = repaired_ch
        ch_reports.append({"channel": ch, **info})

    out = _enforce_consistency(out, x, mask)
    return out, {
        "method": "reg_ar_mit",
        "frame": frame,
        "order": order,
        "ar_iters": ar_iters,
        "ar_reg": ar_reg,
        "qp_reg": qp_reg,
        "qp_iters": qp_iters,
        "masked_samples": int(mask.sum()),
        "channels": ch_reports,
        "reliable_samples_preserved": True,
    }


# --------------------------------------------------- Unified headroom delivery

def apply_headroom_policy(
    repaired: np.ndarray,
    x_deg: np.ndarray,
    sr: int,
    mask: np.ndarray,
    *,
    ceiling_db: float = -1.0,
    max_attenuation_db: float = -24.0,
) -> tuple[np.ndarray, dict]:
    """Apply the single shared stereo headroom attenuation policy (-1 dBTP).

    Matches `rnext.dsp.repair.select_restoration`: verifies reliable samples
    before gain, rejects extreme (>24 dB) headroom collapse as
    `qc_fallback_original`, and scales stereo channels uniformly.
    """
    if not np.isfinite(repaired).all():
        return x_deg.copy(), {"status": "qc_fallback_original",
                              "reason": "nonfinite_reconstruction",
                              "output_gain_db": 0.0}
    if not np.array_equal(repaired[~mask], x_deg[~mask]):
        raise ValueError("repair changed reliable samples before output gain")
    if not np.any(repaired[mask] != x_deg[mask]):
        return x_deg.copy(), {"status": "kept_original",
                              "reason": "no_reconstructed_sample_change",
                              "output_gain_db": 0.0}
    raw_tp = true_peak_dbfs(repaired, sr)
    gain_db = min(0.0, float(ceiling_db) - raw_tp)
    if gain_db < max_attenuation_db:
        return x_deg.copy(), {"status": "qc_fallback_original",
                              "reason": "required_headroom_attenuation_over_24_db",
                              "pre_gain_tp_dbfs": float(raw_tp),
                              "proposed_output_gain_db": float(gain_db),
                              "output_gain_db": 0.0}
    delivered = (repaired.astype(np.float64) * (10.0 ** (gain_db / 20.0))).astype(x_deg.dtype)
    return delivered, {
        "status": "processed_candidate",
        "pre_gain_tp_dbfs": float(raw_tp),
        "output_gain_db": float(gain_db),
        "changed_masked_samples": int(np.count_nonzero(repaired[mask] != x_deg[mask])),
    }
