"""Decode helpers (ffmpeg -> PCM) and metrics (TT DR etc.) — self-contained version."""
import math
import numbers
import subprocess

import numpy as np

BLOCK_TIME = 3          # s
CUT_BEST_BINS = 0.2
MAX_DYNAMIC = 24.0
AUDIO_MIN = 1e-10


def ffprobe_audio(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,bits_per_raw_sample:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    import json
    j = json.loads(out)
    s = j["streams"][0]
    return {"sr": int(s["sample_rate"]), "ch": int(s["channels"]),
            "bps": int(s.get("bits_per_raw_sample") or s.get("bits_per_sample") or 0),
            "dur": float(j["format"]["duration"])}


def decode_s16(path: str, start_s: float | None = None, dur_s: float | None = None) -> tuple[np.ndarray, int]:
    """Decode to int16 PCM [-32768..32767] (no resample/compression). For clip statistics."""
    cmd = ["ffmpeg", "-v", "error", "-err_detect", "ignore_err"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.6f}"]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s:.6f}"]
    cmd += ["-i", str(path), "-map", "0:a:0", "-ac", "2", "-c:a", "pcm_s16le", "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True)
    x = np.frombuffer(raw.stdout, dtype=np.int16).reshape(-1, 2)
    return x, ffprobe_audio(path)["sr"]


def decode_f32(path: str, start_s: float | None = None, dur_s: float | None = None) -> tuple[np.ndarray, int]:
    """Decode to float32 [-1, 1) (stereo). One decode per run — for DSP math."""
    cmd = ["ffmpeg", "-v", "error", "-err_detect", "ignore_err"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.6f}"]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s:.6f}"]
    cmd += ["-i", str(path), "-map", "0:a:0", "-ac", "2", "-c:a", "pcm_f32le", "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True)
    x = np.frombuffer(raw.stdout, dtype=np.float32).reshape(-1, 2)
    return x, ffprobe_audio(path)["sr"]


# ---------------------------------------------------------------- numbering

def dr_rms(seg: np.ndarray) -> np.ndarray:
    """RMS in the TT DR sense: sqrt(2*E[x^2]) — a full-scale sine gives 1.0."""
    return np.sqrt(2.0 * np.sum(seg.astype(np.float64) ** 2, 0) / seg.shape[0])


def tt_dr(x: np.ndarray, fs: int):
    """Full port of measure.tt_dr from restore-dr (dr14-compatible constants)."""
    delta_fs = 60 if fs == 44100 else 0
    block = BLOCK_TIME * (fs + delta_fs)
    n, ch = x.shape
    # ceil, not floor+1: at an exact multiple of n/block the last "remainder"
    # segment was empty and np.max crashed on a zero-size array.
    seg_cnt = max(1, int(math.ceil(n / block)))
    if n < block:
        return 0, 0.0, (0, 0), -100.0, -100.0
    rms = np.zeros((seg_cnt, ch))
    peaks = np.zeros((seg_cnt, ch))
    for i in range(seg_cnt):
        seg = x[i * block:(i + 1) * block, :]
        rms[i] = dr_rms(seg)
        peaks[i] = np.max(np.abs(seg), 0)

    rms.sort(0)
    peaks.sort(0)
    n_blk = max(int(math.floor(seg_cnt * CUT_BEST_BINS)), 1)
    top = rms[seg_cnt - n_blk:, :]
    rms_energy_mean = np.sqrt(np.sum(top ** 2, 0) / n_blk)
    ch_dr = -20.0 * np.log10(rms_energy_mean / peaks[seg_cnt - 2, :])
    ch_dr[np.logical_or(np.sum(top ** 2, 0) < AUDIO_MIN, np.abs(ch_dr) > MAX_DYNAMIC)] = 0.0
    dr = int(round(float(np.mean(ch_dr))))
    # full_scale: 32768 for int16, 1.0 for float (our DSP). Without this branch
    # every level read as -90 dBFS and diagnose() broke (everything -> sparse).
    full_scale = 32768.0 if np.issubdtype(x.dtype, np.integer) else 1.0
    db_peak = 20.0 * math.log10(max(float(np.max(peaks)), 1e-10) / full_scale)
    db_rms = 20.0 * math.log10(
        max(float(np.mean(dr_rms(x.astype(np.float64)))), 1e-10) / full_scale)
    return dr, float(np.mean(ch_dr)), (round(float(ch_dr[0]), 1), round(float(ch_dr[1]), 1)), db_peak, db_rms


def crest_db(x: np.ndarray) -> float:
    """Crest factor: peak vs RMS (TT DR reference)."""
    peak = float(np.max(np.abs(x)))
    rms = math.sqrt(np.mean(x.astype(np.float64) ** 2))
    if rms <= 0:
        return 99.0
    return 20.0 * math.log10(peak / rms)


def clip_stats(x16: np.ndarray) -> dict:
    out = {}
    for c, name in enumerate(("L", "R")):
        xi = np.abs(x16[:, c])
        clipped = xi >= 32767
        n_clipped = int(np.sum(clipped))
        runs = int(np.sum(clipped[1:] & ~clipped[:-1]) + (1 if clipped[0] else 0))
        d = np.diff(clipped.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if clipped[0]:
            starts = np.r_[0, starts]
        if clipped[-1]:
            ends = np.r_[ends, clipped.size]
        if ends.size > starts.size:
            ends = ends[:starts.size]
        long_runs = int(np.sum((ends - starts) >= 3)) if starts.size else 0
        out[name] = {"clipped_samples": n_clipped, "clip_events": runs, "flat_runs_ge3": long_runs}
    return out


def clip_geometry(x16: np.ndarray) -> dict:
    """Clip geometry: length of flat runs and the "flatness" of the tops.

    A soft limiter leaves long flat shelves, intersample overshoots leave short
    ones; this distinguishes different clipping causes.
    """
    out = {}
    for c, name in enumerate(("L", "R")):
        xi = np.abs(x16[:, c])
        clipped = xi >= 32767
        d = np.diff(clipped.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if clipped.size and clipped[0]:
            starts = np.r_[0, starts]
        if clipped.size and clipped[-1]:
            ends = np.r_[ends, clipped.size]
        k = min(starts.size, ends.size)
        runs = (ends[:k] - starts[:k]) if k else np.array([0])
        out[name] = {
            "longest_run": int(runs.max()) if runs.size else 0,
            "runs_ge16": int(np.sum(runs >= 16)) if runs.size else 0,
            "mean_run": round(float(runs.mean()), 2) if runs.size else 0.0,
        }
    return out


# ---------------------------------------------------------------- spectral

_BANDS_HZ = [
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
]


def band_rms_db(x: np.ndarray, fs: int, frame: int = 16384, hop: int | None = None) -> list[float]:
    """Averaged RMS spectrum in 1/3-octave bands, dB re full-scale sine."""
    import hashlib
    hop = hop or frame // 2
    win = np.hanning(frame)
    seg = x if x.ndim == 2 else x[:, None]
    n = seg.shape[0]
    acc = np.zeros(len(_BANDS_HZ) - 1)
    cnt = 0
    for st in range(0, max(n - frame, 1), hop):
        chunk = seg[st:st + frame]
        if chunk.shape[0] < frame:
            chunk = np.pad(chunk, ((0, frame - chunk.shape[0]), (0, 0)))
        spec = np.fft.rfft(chunk * win[:, None], axis=0)  # [bins, ch]
        mag = np.mean(np.abs(spec), axis=1) / (np.sum(win) / 2)
        freqs = np.fft.rfftfreq(frame, 1.0 / fs)
        for k in range(len(_BANDS_HZ) - 1):
            lo, hi = _BANDS_HZ[k], _BANDS_HZ[k + 1]
            sel = (freqs >= lo) & (freqs < hi)
            if sel.any():
                p = float(np.sqrt(np.mean(mag[sel] ** 2)))
                acc[k] += p * p
        cnt += 1
    if cnt == 0:
        return [-200.0] * (len(_BANDS_HZ) - 1)
    rms = np.sqrt(acc / cnt)
    rms = np.maximum(rms, 1e-12)
    return [round(20.0 * math.log10(float(v) / 1.0), 3) for v in rms]


def lufs(path: str) -> dict:
    """Integrated LUFS/LRA/TP via ffmpeg ebur128."""
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-filter_complex",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    txt = p.stderr.split("Summary:")[-1]
    import re
    def f(m):
        if not m:
            return None
        return float(m.group(1)) if "inf" not in m.group(1) else None
    i = re.search(r"I:\s*(-?[\d.]+|-inf)\s*LUFS", txt)
    lra = re.search(r"LRA:\s*(-?[\d.]+|-inf)\s*LU", txt)
    tp = re.search(r"Peak:\s*(-?[\d.]+|-inf)\s*dBFS", txt)
    return {"lufs": f(i), "lra": f(lra), "tp_peak_dbfs": f(tp)}


# ---------------------------------------------------------------- extended
# Second-level features: block-loudness distribution, transients, stereo image,
# true-peak overshoot, clip geometry. All computed from the same decode as the
# base metrics (see docs/dataset.md, the features section).

def _db(v: float) -> float:
    return 20.0 * math.log10(max(float(v), 1e-12))


def block_rms_stats(x: np.ndarray, fs: int, block_s: float = 3.0) -> dict:
    """Block-RMS quantiles (3 s, like a TT DR block) — "crushedness" vs dynamics.

    TT DR uses only the top 20% of blocks; here is the whole profile:
    p05 (quiet spots), p95 (dense), spread = p95 - p05.
    """
    block = int(block_s * fs)
    n = x.shape[0] // block
    if n < 2:
        return {}
    segs = x[: n * block].astype(np.float64).reshape(n, block, 2)
    rms = np.sqrt(np.mean(segs ** 2, axis=1)).max(axis=1)      # worst channel
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    q = np.percentile(db, [5, 25, 50, 75, 95])
    return {"p05": round(float(q[0]), 2), "p25": round(float(q[1]), 2),
            "p50": round(float(q[2]), 2), "p75": round(float(q[3]), 2),
            "p95": round(float(q[4]), 2),
            "spread": round(float(q[4] - q[0]), 2),
            "blocks": int(n)}


def transient_stats(x: np.ndarray, fs: int, frame: int = 1024, hop: int = 256) -> dict:
    """Attack density and spectral flux — separates "sparse" material from
    "dense" (the very Romeo t8 vs Aerosmith t15 case)."""
    mono = x.mean(axis=1).astype(np.float64)
    n = (mono.shape[0] - frame) // hop
    if n < 8:
        return {}
    win = np.hanning(frame)
    prev_mag = None
    flux = 0.0
    rms_list = []
    for i in range(n):
        seg = mono[i * hop: i * hop + frame]
        mag = np.abs(np.fft.rfft(seg * win))
        if prev_mag is not None:
            d = mag - prev_mag
            flux += float(np.sum(d[d > 0]))
        prev_mag = mag
        rms_list.append(math.sqrt(float(np.mean(seg ** 2))))
    r = np.array(rms_list)
    rdb = 20.0 * np.log10(np.maximum(r, 1e-12))
    diff = np.diff(rdb)
    dur = mono.shape[0] / fs
    attacks = int(np.sum(diff > 3.0))
    return {"attacks_per_s": round(attacks / max(dur, 1e-6), 3),
            "flux_per_s": round(flux / max(dur, 1e-6), 1),
            "rms_range_db": round(float(rdb.max() - rdb.min()), 2),
            "frames": int(n)}


def stereo_stats(x: np.ndarray, fs: int) -> dict:
    """Mid/side balance and width in two bands (a master may narrow or widen
    the base — artifact-significant, not just taste)."""
    from scipy import signal as _sig
    l, r = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    mid, side = (l + r) / 2.0, (l - r) / 2.0

    def sm(m, s):
        return round(_db(math.sqrt(float(np.mean(s ** 2))) /
                         max(math.sqrt(float(np.mean(m ** 2))), 1e-12)), 2)

    out = {"side_mid_db": sm(mid, side)}
    try:
        sos = _sig.butter(4, 6000 / (fs / 2), btype="high", output="sos")
        out["side_mid_hf_db"] = sm(_sig.sosfilt(sos, mid), _sig.sosfilt(sos, side))
    except Exception:                                     # noqa: BLE001
        pass
    return out


def tp_overshoot_db(x: np.ndarray, sr: int, up: int = 4) -> float | None:
    """True peak minus sample peak: how many dB intersample interpolation adds.
    A master on dense material produces more overshoots."""
    try:
        from scipy.signal import resample_poly
        sp = float(np.max(np.abs(x)))
        if sp <= 0:
            return None
        tp = float(np.max(np.abs(resample_poly(x, up, 1, axis=0))))
        return round(_db(tp / sp), 2)
    except Exception:                                     # noqa: BLE001
        return None


def feature_dict(x: np.ndarray, sr: int, x16: np.ndarray | None = None) -> dict:
    """Canonical per-track feature set (used by the dataset and the ML runtime)."""
    if x16 is None:
        x16 = np.clip(x * 32768.0, -32768, 32767).astype(np.int16)
    dr, dr_exact, dr_ch, peak_db, rms_db = tt_dr(x, sr)
    return {
        "dr": dr, "dr_exact": round(float(dr_exact), 2), "dr_ch": list(dr_ch),
        "peak_db": round(float(peak_db), 3), "rms_db": round(float(rms_db), 3),
        "crest_db": round(float(crest_db(x)), 3),
        "clip": clip_stats(x16),
        "clip_geom": clip_geometry(x16),
        "bands_db": band_rms_db(x, sr),
        "blocks": block_rms_stats(x, sr),
        "transient": transient_stats(x, sr),
        "stereo": stereo_stats(x, sr),
        "tp_overshoot_db": tp_overshoot_db(x, sr),
        "dur_s": round(float(x.shape[0] / sr), 2),
    }
