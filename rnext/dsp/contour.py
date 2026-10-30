"""STFT contour for dynamic-range restoration — chunked WOLA (no SciPy-stft).

Two-pass scheme:
  pass 1 — per-frame band levels (band-wise memory);
  computing gains per band/frame (flatten/expand);
  pass 2 — multiply bins by the gains + overlap-add.

Expand mechanics (the main mode, reproduces master restorations):
  the per-frame level of a band ABOVE the smoothed body is stretched by
  r_expand; final TP normalization pins peaks at the ceiling — crest grows by
  delta ~ r-1 from the excess transient energy (see corpus/calibration/*.jsonl).
"""
import gc
import math
from dataclasses import dataclass

import numpy as np
from scipy import signal  # get_window, resample_poly

# ---------------------------------------------------------------- WOLA basis


def _window(frame: int) -> np.ndarray:
    return signal.get_window("hann", frame, fftbins=True)


def _frame_starts(n: int, frame: int, hop: int) -> np.ndarray:
    return np.arange(0, max(n - frame, 0) + 1, hop)


def _stft_batch(x: np.ndarray, starts: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x [n, ch] float64 -> Z [nb, ch, freq] (rfft over time axis, window applied)."""
    frames = np.stack([x[s:s + w.size, :].T for s in starts])  # [nb, ch, frame]
    frames *= w[None, None, :]
    return np.fft.rfft(frames, axis=-1)


def _build_gain_matrix(levels: np.ndarray, pol: "ContourPolicy", hop_s: float,
                       band_fcs: np.ndarray) -> np.ndarray:
    """levels [nb_bands, ch, nf] -> gains_db [nb_bands, ch, nf]."""
    nb, ch, nf = levels.shape
    gains = np.zeros_like(levels, dtype=np.float64)
    for bi in range(nb):
        fc = band_fcs[bi]
        lv = levels[bi]  # [ch, nf]

        if pol.mode == "expand":
            body = _asym_smooth(lv, pol.tau_body_up_s, pol.tau_body_dn_s, hop_s)
            excess = np.clip(lv - body, 0.0, 12.0)

            if fc <= pol.bass_hz:
                w_b = pol.bass_scale
            elif fc < pol.ramp_to_hz:
                w_b = pol.bass_scale + (1.0 - pol.bass_scale) * (fc - pol.bass_hz) / (
                    pol.ramp_to_hz - pol.bass_hz)
            else:
                w_b = 1.0
            r_eff = 1.0 + w_b * (pol.r_expand - 1.0)
            g_raw = (r_eff - 1.0) * excess
            g_sm = _asym_smooth(g_raw, min(pol.tau_attack_s, 0.10), pol.tau_release_s
                                if pol.tau_release_s < 0.3 else 0.18, hop_s)
            g_sm = np.clip(g_sm, 0.0, pol.gain_max_db)
        else:  # flatten
            anchor = float(np.median(lv))
            if fc <= pol.bass_hz:
                k_eff = 1.0 - pol.bass_scale * (1.0 - pol.k)
            elif fc < pol.ramp_to_hz:
                wr = (fc - pol.bass_hz) / (pol.ramp_to_hz - pol.bass_hz)
                k_eff = 1.0 - (1.0 - wr) * pol.bass_scale * (1.0 - pol.k)
            else:
                k_eff = pol.k
            g_raw = (1.0 - k_eff) * (anchor - lv)
            g_sm = _asym_smooth(np.clip(g_raw, pol.gain_min_db, pol.gain_max_db),
                                pol.tau_attack_s, pol.tau_release_s, hop_s)
        gains[bi] = g_sm
    return gains


def _asym_smooth(g_db: np.ndarray, tau_up_s: float, tau_dn_s: float, hop_s: float) -> np.ndarray:
    """One-pole smoother with different tau for rise/fall; g_db: [ch, nf] or [nf]."""
    if g_db.ndim == 1:
        g_db = g_db[None, :]
        squeeze = True
    else:
        squeeze = False
    nf = g_db.shape[1]
    a_up = min(1.0, hop_s / max(tau_up_s, 1e-3))
    a_dn = min(1.0, hop_s / max(tau_dn_s, 1e-3))
    out = np.empty_like(g_db)
    prev = g_db[:, 0].copy()
    out[:, 0] = prev
    for i in range(1, nf):
        cur = g_db[:, i]
        a = np.where(cur > prev, a_up, a_dn)
        prev = prev + a * (cur - prev)
        out[:, i] = prev
    return out[0] if squeeze else out


# ---------------------------------------------------------------- policy


@dataclass
class ContourPolicy:
    k: float = 0.75          # mode=flatten: share of band-level spread kept
    bass_scale: float = 0.35 # share of the effect on low-frequency bands
    tau_attack_s: float = 0.25   # gain frame tau
    tau_release_s: float = 1.8   # gain rise tau (flatten) / boost fall tau (expand)
    frame: int = 8192        # ~186 ms @44.1k
    hop: int | None = None   # None -> frame//4
    gain_min_db: float = -9.0
    gain_max_db: float = 12.0
    bass_hz: float = 160.0
    ramp_to_hz: float = 900.0
    mode: str = "expand"     # "expand" | "flatten"
    r_expand: float = 1.5    # mode=expand: multiplier over (level - body)
    tau_body_up_s: float = 1.5
    tau_body_dn_s: float = 2.0
    batch_frames: int = 256  # STFT frames per batch (memory)


# ---------------------------------------------------------------- main stage


def contour_expand(x: np.ndarray, sr: int, pol: ContourPolicy | None = None,
                   return_gains: bool = False):
    """x [n, ch] float32/64 [-1,1) -> restored [n, ch]."""
    pol = pol or ContourPolicy()
    hop = pol.hop or pol.frame // 4
    w = _window(pol.frame)
    n0, ch = x.shape
    # Edge padding: without it the WOLA normalization near the borders is close
    # to zero and the edges "explode". Pad one frame of zeros left and right,
    # compute everything on the padded signal, trim the padding at the end
    # (the r=1 identity is exact).
    pad = pol.frame
    x = np.pad(x, ((pad, pad + pol.frame), (0, 0)))
    n = x.shape[0]
    starts = _frame_starts(n, pol.frame, hop)
    nf = starts.size
    freqs = np.fft.rfftfreq(pol.frame, 1.0 / sr)
    nbins = freqs.size

    # bands: 1/3-octave from 20 Hz up to Nyquist
    edges = [20.0]
    while edges[-1] < sr / 2 - 1:
        edges.append(edges[-1] * 2 ** (1 / 3))
    edges[-1] = sr / 2
    band_fcs, band_bins = [], []
    for i in range(len(edges) - 1):
        idx = np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
        if idx.size:
            band_bins.append(idx)
            band_fcs.append(math.sqrt(edges[i] * edges[i + 1]))
    nb = len(band_bins)

    # pass 1: band levels [nb, ch, nf]
    levels = np.zeros((nb, ch, nf), dtype=np.float64)
    for b0 in range(0, nf, pol.batch_frames):
        b1 = min(b0 + pol.batch_frames, nf)
        Z = _stft_batch(x, starts[b0:b1], w)  # [batch, ch, freq]
        P = np.abs(Z) ** 2  # [batch, ch, freq]
        del Z
        gc.collect()
        for bi, idx in enumerate(band_bins):
            # mean over bins -> power[batch, ch], then to dB
            pw = P[:, :, idx].mean(axis=2)  # [batch, ch]
            levels[bi, :, b0:b1] = 10.0 * np.log10(pw.T + 1e-16)
        del P
        gc.collect()

    gains_db = _build_gain_matrix(levels, pol, hop / sr, np.array(band_fcs))
    gains_lin = 10.0 ** (gains_db / 20.0)  # [nb, ch, nf]
    per_bin_gain_frames: list[np.ndarray] = []
    for bi, idx in enumerate(band_bins):
        per_bin_gain_frames.append(gains_lin[bi][:, :, None])  # [ch, 1, nf]

    # pass 2: apply the gains, WOLA assembly
    out = np.zeros((n, ch), dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)
    w_sq = w ** 2
    for b0 in range(0, nf, pol.batch_frames):
        b1 = min(b0 + pol.batch_frames, nf)
        Zb = _stft_batch(x, starts[b0:b1], w)  # [batch, ch, freq]
        Gb = np.ones((b1 - b0, ch, nbins), dtype=np.float64)
        for bi, idx in enumerate(band_bins):
            # the same band curve applied to every bin of the band: [batch, ch] -> [batch, ch, 1]
            Gb[:, :, idx] = gains_lin[bi, :, b0:b1].T[:, :, None]
        Yb = np.fft.irfft(Zb * Gb, n=pol.frame, axis=-1)
        Yb *= w[None, None, :]
        for j, s in enumerate(starts[b0:b1]):
            e = min(s + pol.frame, n)
            out[s:e, :] += Yb[j, :, : e - s].T
            norm[s:e] += w_sq[: e - s]
        del Zb, Gb, Yb
        gc.collect()

    norm = np.maximum(norm, 1e-8)
    y = out / norm[:, None]
    out = y[pad:pad + n0].astype(x.dtype)
    return (out, gains_db) if return_gains else out


def true_peak_dbfs(x: np.ndarray, sr: int, factor: int = 4) -> float:
    """TP with 4x polyphase (controls intersample overshoots)."""
    up = signal.resample_poly(x, factor, 1, axis=0)
    return float(20.0 * np.log10(max(float(np.max(np.abs(up))), 1e-12)))
