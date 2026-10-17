"""Адаптивный подбор r контура под целевой Δcrest.

Проблема: фиксированный r даёт Δcrest, зависящий от материала
(микроструктура плотности/транзиентов). Мастер на наших корпусных парах
стабильно добавляет Δcrest ≈ +4.4..+4.7 дБ.

Метод (2-точечная калибровка на коротком куске средней части):
    измеряем crest(r1), crest(r2) → линейно инвертируем к target.
Кост: два лёгких прогонa на куске (не на полном треке).
"""
import math

import numpy as np

from rnext import features as F
from rnext.dsp.contour import ContourPolicy, contour_expand

DEFAULT_TARGET_DCREST = 4.5


def _mid_slice(x: np.ndarray, sec: float) -> np.ndarray:
    n0 = (x.shape[0] - sec) // 2
    return x[n0:n0 + sec]


def _crest(x: np.ndarray, sr: int) -> float:
    st = F.tt_dr(x, sr)
    return st[3] - st[4]


def fit_r(x: np.ndarray, sr: int, target_dcrest: float = DEFAULT_TARGET_DCREST,
          sec: float = 90.0, lo: float = 1.2, hi: float = 2.4,
          base: ContourPolicy | None = None) -> tuple[float, float]:
    """→ (r, predicted dcrest). Оффцеливание по вечернему треку."""
    base = base or ContourPolicy()
    sl = x if sec * sr >= x.shape[0] else _mid_slice(x, int(sec * sr))
    c0 = _crest(sl, sr)
    def run(r):
        p = ContourPolicy(**{**base.__dict__, "r_expand": r})
        return _crest(contour_expand(sl, sr, p), sr)
    r1, r2 = lo, max(hi, lo + 0.4)
    c1, c2 = run(r1), run(r2)
    slope = (c2 - c1) / (r2 - r1)          # crest(r) ≈ линейно в r
    if abs(slope) < 1e-3:
        return lo, c1 - c0
    r = lo + (c0 + target_dcrest - c1) / slope
    return float(min(hi, max(lo, r))), float(min(c2, max(c1, c0 + target_dcrest)) - c0)
