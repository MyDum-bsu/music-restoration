"""Adaptive selection of the contour r for a target delta-crest.

Problem: a fixed r yields a delta-crest that depends on the material
(microstructure of density/transients). The master consistently adds
delta-crest of about +4.4..+4.7 dB on our corpus pairs.

Method: two trial passes over the FULL track (r1, r2), both through the same
post-processor as the final pass (TP limiter), then quadratic inversion
    d(u) = A*u^2 + B*u,  u = r - 1,  d(0) = 0
to the target delta-crest. The delta is measured from the RAW source
(crest_ref): the HPF itself adds +0.3..0.5 dB of crest, otherwise we
systematically overshoot.
"""
import math

import numpy as np

from rnext import features as F
from rnext.dsp.contour import ContourPolicy, contour_expand

DEFAULT_TARGET_DCREST = 4.5


def _crest(x: np.ndarray, sr: int) -> float:
    st = F.tt_dr(x, sr)
    return st[3] - st[4]


def fit_r(x: np.ndarray, sr: int, target_dcrest: float = DEFAULT_TARGET_DCREST,
          base: ContourPolicy | None = None,
          post=None, r1: float = 1.5, r2: float = 2.1,
          crest_ref: float | None = None) -> tuple[float, float]:
    """Returns (r, predicted dcrest). Two full-track trials + the zero point.

    crest(r) is superlinear (transients saturate the gains): d(u) = A*u^2 + B*u.
    """
    base = base or ContourPolicy()
    c_raw = _crest(x, sr)
    ref = c_raw if crest_ref is None else float(crest_ref)

    def run(r: float) -> float:
        cp = ContourPolicy(**{**base.__dict__, "r_expand": float(r)})
        y = contour_expand(x, sr, cp)
        if post is not None:
            y = post(y, sr)
        return _crest(y, sr) - ref

    d1 = run(r1)
    d2 = run(r2)
    u1, u2 = r1 - 1.0, r2 - 1.0
    # solve for A,B from the two points (d(0)=0 already accounted for)
    det = (u1 * u1 * u2 - u2 * u2 * u1)
    if abs(det) < 1e-12:
        return float(min(r2, max(1.1, r2 if d2 > 1.0 else 1.1))), max(d2, d1)
    A = (d1 * u2 - d2 * u1) / det
    B = (d2 * u1 * u1 - d1 * u2 * u2) / det
    if abs(A) < 1e-9:
        r = 1.0 + (target_dcrest - 0.0) / B if abs(B) > 1e-9 else r2
    else:
        disc = B * B + 4 * A * target_dcrest
        if disc < 0:
            r = r2
        else:
            r = 1.0 + (-B + math.sqrt(disc)) / (2 * A)
    r = float(min(max(r, 1.05), 3.0))
    pred = A * (r - 1.0) ** 2 + B * (r - 1.0)
    return r, float(pred)
