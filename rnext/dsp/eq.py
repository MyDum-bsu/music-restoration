"""EQ-этап: спектральная дельта от калибровки → пер-бин гейны (zero-phase via STFT)."""
import numpy as np


def apply_eq(x: np.ndarray, sr: int, bands_db: list[tuple[float, float]],
             frame: int = 8192) -> np.ndarray:
    """bands_db: [(hz_center, delta_db), ...] — интерполяция по log(freq).

    Применение в STFT-бинах magnitudes (zero-phase, тот же фреймворк, что contour).
    stft -> (f, t, Zxx): ПЕРВЫЙ элемент — частоты.
    """
    from scipy.signal import stft, istft
    hop = frame // 4
    centers = np.array([b[0] for b in bands_db], dtype=np.float64)
    del_db = np.array([b[1] for b in bands_db], dtype=np.float64)
    freqs, _t, Z = stft(x.T, fs=sr, nperseg=frame, noverlap=frame - hop,
                        window="hann", boundary="zeros", padded=True)
    lf = np.log(np.maximum(freqs, 5.0))
    lf_c = np.log(np.maximum(centers, 5.0))
    d = np.interp(lf, lf_c, del_db)
    H = 10.0 ** (d / 20.0)
    _, y = istft(Z * H[None, :, None], fs=sr, nperseg=frame, noverlap=frame - hop,
                 window="hann", boundary=True)
    return y.T.astype(x.dtype)
