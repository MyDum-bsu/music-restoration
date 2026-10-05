"""Audio decode/encode helpers (ffmpeg-based, stereo).

All processing in this project is stereo-only; decode() forces -ac 2 on purpose.
"""
import subprocess

import numpy as np


def get_samplerate(file) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(file)],
        capture_output=True, text=True).stdout.strip()
    return int(out.strip().rstrip(","))  # ffprobe csv has a trailing comma


def decode(file, fmt: str = "f32le"):
    """Decode to raw PCM via ffmpeg -> (samples [n,2], samplerate).

    fmt 's16le' keeps the original integer codes (needed for clip counting on
    16-bit sources); anything else returns float32 in -32768..32767..0 scale.
    """
    cmd = ["ffmpeg", "-v", "error", "-err_detect", "ignore_err", "-i", str(file),
           "-map", "0:a:0", "-ac", "2"]
    if fmt == "s16le":
        cmd += ["-c:a", "pcm_s16le"]
    cmd += ["-f", fmt, "-"]
    raw = subprocess.run(cmd, capture_output=True)
    if raw.returncode != 0 and not raw.stdout:
        raise RuntimeError(f"ffmpeg decode failed: {raw.stderr.decode()[:300]}")
    dtype = np.int16 if fmt == "s16le" else np.float32
    x = np.frombuffer(raw.stdout, dtype=dtype).reshape(-1, 2)
    return x, get_samplerate(file)


def encode_wav(x, path, fmt: str = "pcm_f32le", fs: int = 44100) -> None:
    """Write float64 [n,2] samples as WAV. fs is the REAL samplerate of x —
    relabeling a 48k source as 44.1k pitches it down (the duration incident)."""
    raw = x.astype(np.float32).tobytes()
    p = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(fs), "-ac", "2",
         "-i", "-", "-c:a", fmt, str(path)],
        input=raw, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"encode_wav failed: {p.stderr.decode()[-300:]}")


def duration_and_fs(file):
    """(duration_seconds, samplerate) via ffprobe. Two calls on purpose: the
    default/nk output order is not guaranteed for multiple entries."""
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", str(file)],
        capture_output=True, text=True).stdout.strip().rstrip(",")
    fs = get_samplerate(file)
    return float(dur), fs
