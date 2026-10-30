"""TP-safe limiter (v0: soft static gain + oversampled peak audit)."""
import numpy as np
from scipy import signal


def limit_track(x: np.ndarray, sr: int, ceiling_db: float = -0.3) -> np.ndarray:
    """x [n, ch] float. Guarantees TP <= ceiling; no compression if already below."""
    up = signal.resample_poly(x, 4, 1, axis=0)
    pk = float(np.max(np.abs(up)))
    ceil = 10 ** (ceiling_db / 20.0)
    if pk <= ceil:
        return x
    g = ceil / pk
    return (x * g).astype(x.dtype)
