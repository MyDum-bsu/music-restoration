"""Деклип: AR-экстраполяция blended forward+backward на клиповых runs.

Подход: автофуга сторон по соседям каждого клипованного run'а продлевает
предсказание слева и справа; смешение линейно по позиции. Короткие runs
(2–150 семплов) восстанавливаются с минимумом HF-артефактов.
"""
import numpy as np
from scipy.linalg import solve_toeplitz

CLIP_THRESH = 32766  # порог детекции на 16-bit int


def _ar_coeffs(context: np.ndarray, order: int) -> np.ndarray:
    r = np.correlate(context, context, mode="full")[len(context) - 1:]
    if r[0] <= 0:
        return np.zeros(1)
    r = r / r[0]
    order = min(order, context.size - 4)
    if order < 2:
        return np.zeros(1)
    a = solve_toeplitz(r[:order], r[1:order + 1])
    # знаменатель полином: y[n] + Σ a_k y[n-k] = 0 → рекурсия y[n] = -Σ a_k y[n-k]
    return a


def _ar_extrapolate(context: np.ndarray, a: np.ndarray, n_pred: int, backward: bool) -> np.ndarray:
    if backward:
        context = context[::-1]
    y = list(context.astype(np.float64))
    out = []
    for _ in range(n_pred):
        m = len(y)
        val = 0.0
        for k in range(1, a.size + 1):
            if m - k >= 0:
                val -= a[k - 1] * y[m - k]
        out.append(val)
        y.append(val)
    arr = np.array(out)
    return arr[::-1] if backward else arr


def _extract_runs(clipped: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(clipped.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if clipped[0]:
        starts = np.r_[0, starts]
    if clipped[-1]:
        ends = np.r_[ends, clipped.size]
    if ends.size > starts.size:
        ends = ends[:starts.size]
    return list(zip(starts, ends))


def declip_track(x16: np.ndarray) -> tuple[np.ndarray, int]:
    """x16: int16 [n,2] с клипами → восстановленный int16 [n,2], + объём правок."""
    y = x16.copy()
    n_fills = 0
    for ch in (0, 1):
        xi = np.abs(x16[:, ch])
        runs = _extract_runs(xi >= CLIP_THRESH)
        for s0, e0 in runs:
            L = e0 - s0
            if L < 2 or s0 < 32 or e0 > y.shape[0] - 32:
                continue
            ctx = min(max(64, 4 * L), 2048)
            pre = x16[max(0, s0 - ctx):s0, ch].astype(np.float64)
            post = x16[e0:min(y.shape[0], e0 + ctx), ch].astype(np.float64)
            order = min(64, pre.size - 4, post.size - 4)
            if order < 8:
                continue
            a_pre = _ar_coeffs(pre, order)
            a_post = _ar_coeffs(post, order)
            fwd = _ar_extrapolate(pre, a_pre, L, backward=False)
            bwd = _ar_extrapolate(post, a_post, L, backward=True)
            w = np.linspace(1.0, 0.0, L + 2)[1:-1]
            fill = (w * fwd + (1 - w) * bwd)
            # анти-вывод: слегка «укороченные» пики, чтобы не переклать 0 dBFS
            fill = np.clip(fill, -32768.0, 32767.0).astype(np.int16)
            y[s0:e0, ch] = fill
            n_fills += L
    return y, n_fills


def clip_positions(x16: np.ndarray) -> int:
    clipped = np.abs(x16) >= CLIP_THRESH
    return int(clipped.sum())
