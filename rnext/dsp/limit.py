"""TP-safe лимитер (v0: мягкая статика + oversample-аудит)."""
import numpy as np
from scipy import signal


def limit_track(x: np.ndarray, sr: int, ceiling_db: float = -0.3) -> np.ndarray:
    """x [n, ch] float. Гарантирует TP ≤ ceiling; без компрессии, если уже ниже."""
    up = signal.resample_poly(x, 4, 1, axis=0)
    pk = float(np.max(np.abs(up)))
    ceil = 10 ** (ceiling_db / 20.0)
    if pk <= ceil:
        return x
    g = ceil / pk
    return (x * g).astype(x.dtype)
