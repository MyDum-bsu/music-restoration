"""STFT-контур реставрации динамического диапазона — chunked WOLA (без SciPy-stft).

Двухпроходная схема:
  проход 1 — уровни полос по кадрам (мелочная память);
  computing gains per band/frame (flatten/expand);
  проход 2 — умножение бинов на гейны + overlap-add.

Механика expand (главный режим, воспроизводит master-реставрации):
  per-frame уровень полосы ABOVE сглаженного тела растягивается в r_expand раз;
  TP-нормализация в конце ставит пики на потолок — crest растёт на Δ≈r-1 от
  избыточной энергии транзиентов (см. corpus/calibration/*.jsonl).
"""
import gc
import math
from dataclasses import dataclass

import numpy as np
from scipy import signal  # get_window, resample_poly

# ---------------------------------------------------------------- WOLA базис


def _window(frame: int) -> np.ndarray:
    return signal.get_window("hann", frame, fftbins=True)


def _frame_starts(n: int, frame: int, hop: int) -> np.ndarray:
    return np.arange(0, max(n - frame, 0) + 1, hop)


def _stft_batch(x: np.ndarray, starts: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x [n, ch] float64 → Z [nb, ch, freq] (rfft ось времени, окно применено)."""
    frames = np.stack([x[s:s + w.size, :].T for s in starts])  # [nb, ch, frame]
    frames *= w[None, None, :]
    return np.fft.rfft(frames, axis=-1)


def _build_gain_matrix(levels: np.ndarray, pol: "ContourPolicy", hop_s: float,
                       band_fcs: np.ndarray) -> np.ndarray:
    """levels [nb_bands, ch, nf] → gains_db [nb_bands, ch, nf]."""
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
    """Однополюсник с разными τ на подъём/спад; g_db: [ch, nf] или [nf]."""
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


# ---------------------------------------------------------------- политика


@dataclass
class ContourPolicy:
    k: float = 0.75          # mode=flatten: доля сохранения разброса уровня полосы
    bass_scale: float = 0.35 # доля эффекта на НЧ-полосах
    tau_attack_s: float = 0.25   # τ кадра гейна
    tau_release_s: float = 1.8   # τ подъёма гейна (flatten) / спада буста (expand)
    frame: int = 8192        # ~186 ms @44.1k
    hop: int | None = None   # None → frame//4
    gain_min_db: float = -9.0
    gain_max_db: float = 12.0
    bass_hz: float = 160.0
    ramp_to_hz: float = 900.0
    mode: str = "expand"     # "expand" | "flatten"
    r_expand: float = 1.5    # mode=expand: множитель над (level - body)
    tau_body_up_s: float = 1.5
    tau_body_dn_s: float = 2.0
    batch_frames: int = 256  # кадров STFT за раз (память)


# ---------------------------------------------------------------- главный этап


def contour_expand(x: np.ndarray, sr: int, pol: ContourPolicy | None = None) -> np.ndarray:
    """x [n, ch] float32/64 [-1,1) → реставрированный [n, ch]."""
    pol = pol or ContourPolicy()
    hop = pol.hop or pol.frame // 4
    w = _window(pol.frame)
    n, ch = x.shape
    starts = _frame_starts(n, pol.frame, hop)
    nf = starts.size
    freqs = np.fft.rfftfreq(pol.frame, 1.0 / sr)
    nbins = freqs.size

    # полосы: 1/3-октава 20 Гц → Nyquist
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

    # проход 1: уровни полос [nb, ch, nf]
    levels = np.zeros((nb, ch, nf), dtype=np.float64)
    for b0 in range(0, nf, pol.batch_frames):
        b1 = min(b0 + pol.batch_frames, nf)
        Z = _stft_batch(x, starts[b0:b1], w)  # [batch, ch, freq]
        P = np.abs(Z) ** 2  # [batch, ch, freq]
        del Z
        gc.collect()
        for bi, idx in enumerate(band_bins):
            # mean по бинам → power[batch, ch], потом в dB
            pw = P[:, :, idx].mean(axis=2)  # [batch, ch]
            levels[bi, :, b0:b1] = 10.0 * np.log10(pw.T + 1e-16)
        del P
        gc.collect()

    gains_db = _build_gain_matrix(levels, pol, hop / sr, np.array(band_fcs))
    gains_lin = 10.0 ** (gains_db / 20.0)  # [nb, ch, nf]
    per_bin_gain_frames: list[np.ndarray] = []
    for bi, idx in enumerate(band_bins):
        per_bin_gain_frames.append(gains_lin[bi][:, :, None])  # [ch, 1, nf]

    # проход 2: применяем гейны, WOLA-сборка
    out = np.zeros((n, ch), dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)
    w_sq = w ** 2
    for b0 in range(0, nf, pol.batch_frames):
        b1 = min(b0 + pol.batch_frames, nf)
        Zb = _stft_batch(x, starts[b0:b1], w)  # [batch, ch, freq]
        Gb = np.ones((b1 - b0, ch, nbins), dtype=np.float64)
        for bi, idx in enumerate(band_bins):
            Gb[:, :, idx] = gains_lin[bi, :, b0:b1].T  # [batch, ch, 1]
        Yb = np.fft.irfft(Zb * Gb, n=pol.frame, axis=-1)
        Yb *= w[None, None, :]
        for j, s in enumerate(starts[b0:b1]):
            e = min(s + pol.frame, n)
            out[s:e, :] += Yb[j, :, : e - s].T
            norm[s:e] += w_sq[: e - s]
        del Zb, Gb, Yb
        gc.collect()

    norm = np.maximum(norm, 1e-8)
    return (out / norm[:, None]).astype(x.dtype)


def true_peak_dbfs(x: np.ndarray, sr: int, factor: int = 4) -> float:
    """TP с 4х-полифазой (контроль интерсэмплинг-овершутов)."""
    up = signal.resample_poly(x, factor, 1, axis=0)
    return float(20.0 * np.log10(max(float(np.max(np.abs(up))), 1e-12)))
