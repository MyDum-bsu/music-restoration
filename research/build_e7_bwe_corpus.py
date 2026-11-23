#!/usr/bin/env python3
"""Build the E7 bandwidth-extension (BWE) dev corpus (`corpus/dev-e7-bwe-v1.json`).

    python3 research/build_e7_bwe_corpus.py --root corpus

Creates 17 independent 48 kHz stereo dev cases across six strata:
- bandlimit_8k (3 cases): 48 kHz -> 16 kHz -> 48 kHz polyphase band-limit
  (`observed_cutoff_hz = 8000.0`).
- bandlimit_11k (3 cases): 48 kHz -> 22.05 kHz -> 48 kHz polyphase band-limit
  (`observed_cutoff_hz = 11025.0`).
- bandlimit_16k (2 cases): 48 kHz -> 32 kHz -> 48 kHz polyphase band-limit
  (`observed_cutoff_hz = 16000.0`).
- codec_roundtrip (3 cases): real `ffmpeg` lossy codec roundtrips (`libmp3lame`
  64k / 96k and `libopus` 32k) with deterministic delay alignment and explicit
  `observed_cutoff_hz` boundary (falls back cleanly to a documented deterministic
  quantized band-limit filter if `ffmpeg` lacks the encoder).
- fullband_clean_control (3 cases): clean 48 kHz full-bandwidth stereo negative
  controls with rich natural HF content up to ~20 kHz and ZERO HF loss.
- ood (3 cases): extreme 4 kHz narrowband, hard-panned/decorrelated wide stereo
  at 8 kHz cutoff, and double lossy transcode (`mp3 64k -> opus 32k`) at 8 kHz.

Security & leakage boundaries:
- Never opens or inspects `corpus/sealed-v1.json` or `corpus/sealed-e5-v1.json`.
- Checks cross-manifest group-key disjointness against non-sealed dev/smoke
  manifests only (`smoke-v1.json`, `dev-v1.json`, `dev-e3-v1.json`,
  `dev-e6-archival-v1.json`).
- Stores relative paths and SHA-256 hashes only; zero host-specific paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import corpus_synth, corpus_v1

SCHEMA = "e7-bwe-corpus/v1"
GENERATOR = "synth/e7-bwe-v1"
DATASET_VERSION = "dev-e7-bwe-2026-11-23"

RIGHTS_DEV = {
    "local_inference": True,
    "colab": True,
    "training": True,
    "listening": True,
    "publication": False,
}

STRATA = (
    "bandlimit_8k",
    "bandlimit_11k",
    "bandlimit_16k",
    "codec_roundtrip",
    "fullband_clean_control",
    "ood",
)

_CODEC_AVAILABILITY_CACHE: dict[str, bool] = {}


def _groups(idx: int) -> dict[str, str]:
    return {
        "take": f"t-e7-{idx:02d}",
        "fingerprint": f"fp-e7-{idx:02d}",
        "session": f"ses-e7-{idx:02d}",
        "artist": f"e7-artist-{idx:02d}",
        "album": f"e7-album-{idx:02d}",
        "stem_family": f"fam-e7-{idx:02d}",
    }


def _ffmpeg_has_encoder(encoder: str) -> bool:
    if encoder in _CODEC_AVAILABILITY_CACHE:
        return _CODEC_AVAILABILITY_CACHE[encoder]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        _CODEC_AVAILABILITY_CACHE[encoder] = False
        return False
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        ok = proc.returncode == 0 and (f" {encoder} " in proc.stdout)
    except (OSError, subprocess.SubprocessError):
        ok = False
    _CODEC_AVAILABILITY_CACHE[encoder] = ok
    return ok


def _zero_phase_butter(
    x: np.ndarray,
    rate: int,
    cutoff_hz: float | tuple[float, float],
    btype: str,
    order: int = 4,
) -> np.ndarray:
    nyq = 0.5 * rate
    if isinstance(cutoff_hz, tuple):
        wn = (
            float(np.clip(cutoff_hz[0] / nyq, 1e-4, 0.98)),
            float(np.clip(cutoff_hz[1] / nyq, 1e-4, 0.98)),
        )
    else:
        wn = float(np.clip(cutoff_hz / nyq, 1e-4, 0.98))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def _bwe_clean_signal(
    seed: int,
    rate: int,
    seconds: float,
    *,
    percussive: bool = False,
    plucked: bool = False,
    bright_cymbals: bool = True,
    wide_stereo: bool = False,
    quiet_tail: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic 48 kHz stereo music with rich natural HF up to ~20 kHz.

    Returns:
        clean: float64 array of shape (n_samples, 2)
        attack_mask: bool array of shape (n_samples, 2) marking sharp musical onsets
    """
    rng = np.random.default_rng(seed)
    n = round(rate * seconds)
    t = np.arange(n, dtype=np.float64) / rate
    out = np.zeros((n, 2), dtype=np.float64)
    attack_mask = np.zeros((n, 2), dtype=bool)
    active_end = int(n * 0.78) if quiet_tail else n

    # 1. Stereo harmonic bass foundation (60-110 Hz + overtones).
    bass_f0 = float(rng.uniform(62.0, 108.0))
    bass_env = 0.68 + 0.32 * np.sin(
        2.0 * np.pi * rng.uniform(0.5, 1.1) * t + rng.uniform(0.0, 2.0 * np.pi)
    )
    bass = np.zeros(n, dtype=np.float64)
    for h in range(1, 7):
        bass += (0.24 / (h ** 1.15)) * np.sin(
            2.0 * np.pi * (bass_f0 * h) * t + rng.uniform(0.0, 2.0 * np.pi)
        )
    bass *= bass_env
    out[:, 0] += 0.90 * bass
    out[:, 1] += 0.86 * bass

    # 2. Four panned harmonic voices with rich overtones extending up to ~19.5 kHz.
    roots = bass_f0 * np.array([2.5, 3.75, 5.0, 7.5]) * rng.uniform(0.95, 1.05, size=4)
    for idx_v, f0 in enumerate(roots):
        if wide_stereo:
            pan = 0.08 if (idx_v % 2 == 0) else 0.92
        else:
            pan = float(rng.uniform(0.20, 0.80))
        gl, gr = np.cos(pan * np.pi / 2.0), np.sin(pan * np.pi / 2.0)
        mod_f = float(rng.uniform(0.4, 1.4))
        env = 0.45 + 0.55 * np.maximum(
            0.0, np.sin(2.0 * np.pi * mod_f * t + rng.uniform(0.0, 2.0 * np.pi))
        )
        voice_l = np.zeros(n, dtype=np.float64)
        voice_r = np.zeros(n, dtype=np.float64)
        max_h = min(48, int(0.42 * rate / max(f0, 1.0)))
        for h in range(1, max_h + 1):
            freq = f0 * h
            if freq >= 0.42 * rate:
                break
            amp = float(rng.uniform(0.055, 0.12)) / (h ** 1.05)
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            # Mild stereo phase offset for realism; stronger decorrelation in wide_stereo OOD.
            dphi = float(rng.uniform(-0.85, 0.85)) if wide_stereo else float(rng.uniform(-0.12, 0.12))
            voice_l += amp * np.sin(2.0 * np.pi * freq * t + phi)
            voice_r += amp * np.sin(2.0 * np.pi * freq * t + phi + dphi)
        out[:, 0] += gl * env * voice_l
        out[:, 1] += gr * env * voice_r

    # 3. Bright cymbal/hi-hat shimmer & upper-HF air (6 kHz - 20 kHz).
    if bright_cymbals or percussive:
        n_hits = max(6, int(seconds * (6 if percussive else 4)))
        step = max(1, (active_end - int(rate * 0.12)) // max(1, n_hits))
        for k_idx in range(n_hits):
            center = int(rate * 0.06) + k_idx * step + int(rng.integers(-step // 6, max(1, step // 6)))
            i = int(np.clip(center, int(rate * 0.025), max(int(rate * 0.025) + 1, active_end - int(rate * 0.09))))
            dur = min(active_end - i, int(rate * 0.085))
            if dur < 64:
                continue
            kt = np.arange(dur, dtype=np.float64) / rate
            rise_s = float(rng.uniform(0.0004, 0.0012))
            decay_s = float(rng.uniform(0.015, 0.045))
            env = (1.0 - np.exp(-kt / rise_s)) * np.exp(-kt / decay_s)

            # Metallic inharmonic partials + shaped broadband HF noise (6-20 kHz).
            raw_noise = rng.normal(0.0, 1.0, (dur, 2))
            shared_n = rng.normal(0.0, 1.0, (dur, 1))
            if wide_stereo:
                cym_noise = 0.90 * raw_noise + 0.20 * shared_n
            else:
                cym_noise = 0.55 * raw_noise + 0.83 * shared_n
            cym_hf = _zero_phase_butter(cym_noise, rate, (5500.0, 20000.0), btype="bandpass", order=3)
            cym_hf /= float(np.sqrt(np.mean(cym_hf ** 2))) + 1e-12

            body = (
                0.35 * np.sin(2.0 * np.pi * rng.uniform(220.0, 520.0) * kt)
                + 0.25 * np.sin(2.0 * np.pi * rng.uniform(2400.0, 5800.0) * kt)
                + 0.20 * np.sin(2.0 * np.pi * rng.uniform(9200.0, 14500.0) * kt)
                + 0.14 * np.sin(2.0 * np.pi * rng.uniform(14800.0, 19200.0) * kt)
            )
            pan = float(rng.uniform(0.15, 0.85))
            gl, gr = np.cos(pan * np.pi / 2.0), np.sin(pan * np.pi / 2.0)
            scale = float(rng.uniform(0.32, 0.58))
            out[i:i + dur, 0] += scale * env * (gl * body + 0.42 * gl * cym_hf[:, 0])
            out[i:i + dur, 1] += scale * env * (gr * body + 0.42 * gr * cym_hf[:, 1])

            win_pre = round(0.0015 * rate)
            win_post = round(0.012 * rate)
            attack_mask[max(0, i - win_pre):min(n, i + win_post), :] = True

    # 4. Plucked acoustic strings with bright upper harmonics up to ~18 kHz.
    if plucked:
        n_plucks = max(6, int(seconds * 4))
        step = max(1, (active_end - int(rate * 0.16)) // max(1, n_plucks))
        for k_idx in range(n_plucks):
            i = int(rate * 0.05) + k_idx * step
            dur = min(active_end - i, int(rate * 0.22))
            if dur < 64:
                continue
            kt = np.arange(dur, dtype=np.float64) / rate
            f0 = float(rng.uniform(240.0, 580.0))
            pluck = np.zeros(dur, dtype=np.float64)
            for h in range(1, 32):
                if f0 * h >= 0.39 * rate:
                    break
                h_decay = np.exp(-kt / (0.15 / (h ** 0.55)))
                pluck += (0.42 / (h ** 0.90)) * h_decay * np.sin(2.0 * np.pi * f0 * h * kt)
            pluck *= 1.0 - np.exp(-kt / 0.0005)
            pan = float(rng.uniform(0.20, 0.80))
            out[i:i + dur, 0] += np.cos(pan * np.pi / 2.0) * pluck
            out[i:i + dur, 1] += np.sin(pan * np.pi / 2.0) * pluck
            win_pre = round(0.0015 * rate)
            win_post = round(0.012 * rate)
            attack_mask[max(0, i - win_pre):min(n, i + win_post), :] = True

    # 5. Optional quiet reverb tail.
    if quiet_tail:
        fade_len = round(0.08 * rate)
        fade_end = min(n, active_end + fade_len)
        ramp = np.linspace(1.0, 0.025, fade_end - active_end, dtype=np.float64)[:, None]
        out[active_end:fade_end] *= ramp
        if fade_end < n:
            tail_t = np.arange(n - fade_end, dtype=np.float64) / rate
            out[fade_end:] *= (0.025 * np.exp(-tail_t / max(0.30, seconds * 0.20)))[:, None]

    # Low stereo clean floor (-84 dBFS).
    out += rng.normal(0.0, 6.0e-5, out.shape)
    peak = float(np.abs(out).max())
    if peak > 0.0:
        out *= 0.78 / peak
    return np.ascontiguousarray(out, dtype=np.float64), attack_mask


def polyphase_bandlimit(x: np.ndarray, rate: int, low_sr: int) -> np.ndarray:
    """Exact polyphase downsample (rate -> low_sr) then upsample (low_sr -> rate)."""
    frac = Fraction(low_sr, rate).limit_denominator(1024)
    down = signal.resample_poly(
        x, frac.numerator, frac.denominator, axis=0, window=("kaiser", 14.0)
    )
    up = signal.resample_poly(
        down, frac.denominator, frac.numerator, axis=0, window=("kaiser", 14.0)
    )
    n = x.shape[0]
    if up.shape[0] > n:
        up = up[:n]
    elif up.shape[0] < n:
        pad = np.zeros((n - up.shape[0], x.shape[1]), dtype=np.float64)
        up = np.vstack([up, pad])
    return np.ascontiguousarray(up, dtype=np.float64)


def _align_to_reference(ref: np.ndarray, cand: np.ndarray, max_lag: int = 4096) -> np.ndarray:
    """Align codec-roundtripped stereo PCM to exact length/sample lag of reference."""
    n = ref.shape[0]
    m_ref = np.mean(ref, axis=1)
    m_cand = np.mean(cand, axis=1)
    # Use middle segment to avoid start/end padding bias.
    seg_len = min(n, cand.shape[0], 32768)
    s0 = max(0, (min(n, cand.shape[0]) - seg_len) // 2)
    a = m_ref[s0:s0 + seg_len]
    b = m_cand[s0:s0 + seg_len]
    a = a - np.mean(a)
    b = b - np.mean(b)
    n_fft = 1 << int(np.ceil(np.log2(2 * seg_len)))
    corr = np.fft.irfft(np.fft.rfft(b, n=n_fft) * np.conj(np.fft.rfft(a, n=n_fft)), n=n_fft)
    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.array([corr[lag % n_fft] for lag in lags], dtype=np.float64)
    best_lag = int(lags[int(np.argmax(scores))])

    out = np.zeros_like(ref, dtype=np.float64)
    if best_lag >= 0:
        avail = min(n, cand.shape[0] - best_lag)
        if avail > 0:
            out[:avail] = cand[best_lag:best_lag + avail]
    else:
        shift = -best_lag
        avail = min(n - shift, cand.shape[0])
        if avail > 0:
            out[shift:shift + avail] = cand[:avail]
    return np.ascontiguousarray(out, dtype=np.float64)


def _ffmpeg_codec_roundtrip_once(
    x: np.ndarray,
    rate: int,
    codec: str,
    bitrate: str,
) -> tuple[np.ndarray, str]:
    """Run a single lossy codec encode->decode roundtrip via ffmpeg with honest fallback."""
    encoder = "libmp3lame" if codec == "mp3" else "libopus"
    ext = ".mp3" if codec == "mp3" else ".opus"
    if _ffmpeg_has_encoder(encoder):
        ffmpeg = shutil.which("ffmpeg")
        assert ffmpeg is not None
        with tempfile.TemporaryDirectory(prefix="e7_codec_") as td:
            td_p = Path(td)
            in_wav = td_p / "in.wav"
            enc_path = td_p / f"encoded{ext}"
            out_wav = td_p / "out.wav"
            corpus_synth.write_wav(in_wav, rate, x, bits=24, seed=0)
            enc_cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(in_wav),
                "-c:a", encoder, "-b:a", bitrate,
            ]
            if codec == "opus":
                enc_cmd += ["-vbr", "off", "-application", "audio"]
            enc_cmd.append(str(enc_path))
            p1 = subprocess.run(enc_cmd, capture_output=True, text=True, check=False, timeout=30)
            if p1.returncode == 0 and enc_path.exists():
                dec_cmd = [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(enc_path),
                    "-ar", str(rate), "-ac", "2", "-c:a", "pcm_s24le",
                    str(out_wav),
                ]
                p2 = subprocess.run(dec_cmd, capture_output=True, text=True, check=False, timeout=30)
                if p2.returncode == 0 and out_wav.exists():
                    sr_dec, raw = wavfile.read(out_wav)
                    if sr_dec == rate and raw.ndim == 2 and raw.shape[1] == 2:
                        # wavfile reads 24-bit WAV as int32 left-shifted by 8 bits.
                        scale = float(1 << 31) if raw.dtype == np.int32 else float(1 << 15)
                        pcm = raw.astype(np.float64) / scale
                        aligned = _align_to_reference(x, pcm)
                        return aligned, f"ffmpeg_{encoder}_{bitrate}"

    # Honest deterministic fallback if ffmpeg or encoder is unavailable.
    cutoff = 11025.0 if codec == "mp3" else 12000.0
    lp = _zero_phase_butter(x, rate, cutoff, btype="lowpass", order=8)
    # Simulate coarse MDCT-like quantization noise (~10-bit equivalent in HF).
    q = np.round(lp * 2048.0) / 2048.0
    return np.ascontiguousarray(q, dtype=np.float64), f"fallback_quantized_lpf_{codec}_{bitrate}"


def apply_bwe_degradation(
    clean: np.ndarray,
    rate: int,
    spec: dict,
) -> tuple[np.ndarray, dict]:
    """Apply a BWE band-limiting or codec-roundtrip degradation operator.

    Returns:
        degraded: float64 (n, 2) at native 48 kHz
        meta: dict with exact observed_cutoff_hz, codec_backend, native_low_sr
    """
    op = str(spec["operator"])
    cutoff_hz = float(spec["observed_cutoff_hz"])
    meta: dict = {
        "operator": op,
        "observed_cutoff_hz": cutoff_hz,
        "codec_backend": None,
        "native_low_sr": spec.get("native_low_sr"),
    }

    if op == "fullband_clean_control":
        return clean.copy(), meta

    if op in ("bandlimit_polyphase", "ood_narrowband_4k", "ood_wide_stereo_8k"):
        low_sr = int(spec["native_low_sr"])
        degraded = polyphase_bandlimit(clean, rate, low_sr)
        meta["codec_backend"] = "polyphase_kaiser14"
        return np.clip(degraded, -0.995, 0.995), meta

    if op == "codec_roundtrip":
        codec = str(spec["codec"])
        bitrate = str(spec["bitrate"])
        # First apply exact band-limit at observed_cutoff_hz so the observed/missing
        # boundary is crisp and unambiguous, then run real lossy codec roundtrip.
        low_sr = round(2.0 * cutoff_hz)
        bl = polyphase_bandlimit(clean, rate, low_sr)
        rt, backend = _ffmpeg_codec_roundtrip_once(bl, rate, codec, bitrate)
        # Re-apply crisp low-pass at cutoff_hz to remove any tiny codec dither above cutoff.
        rt = polyphase_bandlimit(rt, rate, low_sr)
        meta["codec_backend"] = backend
        return np.clip(rt, -0.995, 0.995), meta

    if op == "ood_double_codec_8k":
        low_sr = int(spec["native_low_sr"])
        bl = polyphase_bandlimit(clean, rate, low_sr)
        rt1, b1 = _ffmpeg_codec_roundtrip_once(bl, rate, "mp3", "64k")
        rt2, b2 = _ffmpeg_codec_roundtrip_once(rt1, rate, "opus", "32k")
        rt2 = polyphase_bandlimit(rt2, rate, low_sr)
        meta["codec_backend"] = f"{b1}->{b2}"
        return np.clip(rt2, -0.995, 0.995), meta

    raise ValueError(f"unknown E7 BWE operator: {op!r}")


def _case(
    cid: str,
    rate: int,
    seconds: float,
    seed: int,
    stratum: str,
    op: str,
    observed_cutoff_hz: float,
    bits: int,
    idx: int,
    **kw,
) -> dict:
    is_ctl = stratum == "fullband_clean_control"
    return {
        "id": cid,
        "kind": "paired_synthetic_bwe",
        "stratum": stratum,
        "split": "control" if is_ctl else "eval",
        "is_fullband_control": is_ctl,
        "observed_cutoff_hz": float(observed_cutoff_hz),
        "rate": int(rate),
        "channels": 2,
        "bits": int(bits),
        "rights": dict(RIGHTS_DEV),
        "groups": _groups(idx),
        "clean": {"path": f"{cid}.clean.wav"},
        "degraded": {"path": f"{cid}.degraded.wav"},
        "attack_mask": {"path": f"{cid}.attack_mask.json"},
        "synthesis": {
            "generator": GENERATOR,
            "seed": int(seed),
            "rate": int(rate),
            "seconds": float(seconds),
            "operator": op,
            "observed_cutoff_hz": float(observed_cutoff_hz),
            "bits": int(bits),
            **kw,
        },
    }


def e7_bwe_cases() -> list[dict]:
    """17 independent 48 kHz stereo BWE dev cases across 6 strata."""
    specs = [
        # Stratum 1: 8 kHz cutoff (48k -> 16k -> 48k), 3 cases
        ("e7-bl8k-harmonic-mix-48k", 48000, 2.5, 701, "bandlimit_8k", "bandlimit_polyphase",
         8000.0, 24, {"native_low_sr": 16000, "bright_cymbals": True}),
        ("e7-bl8k-percussion-cymbals-48k", 48000, 2.5, 702, "bandlimit_8k", "bandlimit_polyphase",
         8000.0, 24, {"native_low_sr": 16000, "percussive": True, "bright_cymbals": True}),
        ("e7-bl8k-plucked-tail-48k", 48000, 2.5, 703, "bandlimit_8k", "bandlimit_polyphase",
         8000.0, 24, {"native_low_sr": 16000, "plucked": True, "quiet_tail": True}),

        # Stratum 2: 11.025 kHz cutoff (48k -> 22.05k -> 48k), 3 cases
        ("e7-bl11k-harmonic-mix-48k", 48000, 2.5, 711, "bandlimit_11k", "bandlimit_polyphase",
         11025.0, 24, {"native_low_sr": 22050, "bright_cymbals": True}),
        ("e7-bl11k-percussion-cymbals-48k", 48000, 2.5, 712, "bandlimit_11k", "bandlimit_polyphase",
         11025.0, 24, {"native_low_sr": 22050, "percussive": True, "bright_cymbals": True}),
        ("e7-bl11k-plucked-bright-48k", 48000, 2.5, 713, "bandlimit_11k", "bandlimit_polyphase",
         11025.0, 24, {"native_low_sr": 22050, "plucked": True, "bright_cymbals": True}),

        # Stratum 3: 16 kHz cutoff (48k -> 32k -> 48k), 2 cases
        ("e7-bl16k-bright-mix-48k", 48000, 2.5, 721, "bandlimit_16k", "bandlimit_polyphase",
         16000.0, 24, {"native_low_sr": 32000, "bright_cymbals": True}),
        ("e7-bl16k-percussion-48k", 48000, 2.5, 722, "bandlimit_16k", "bandlimit_polyphase",
         16000.0, 24, {"native_low_sr": 32000, "percussive": True, "bright_cymbals": True}),

        # Stratum 4: Lossy codec roundtrips (MP3 64k, MP3 96k, Opus 32k), 3 cases
        ("e7-codec-mp3-64k-11k-48k", 48000, 2.5, 731, "codec_roundtrip", "codec_roundtrip",
         11025.0, 24, {"codec": "mp3", "bitrate": "64k", "percussive": True, "bright_cymbals": True}),
        ("e7-codec-mp3-96k-14k-48k", 48000, 2.5, 732, "codec_roundtrip", "codec_roundtrip",
         14000.0, 24, {"codec": "mp3", "bitrate": "96k", "plucked": True, "bright_cymbals": True}),
        ("e7-codec-opus-32k-12k-48k", 48000, 2.5, 733, "codec_roundtrip", "codec_roundtrip",
         12000.0, 24, {"codec": "opus", "bitrate": "32k", "percussive": True, "bright_cymbals": True}),

        # Stratum 5: Clean full-bandwidth negative controls (ZERO HF loss), 3 cases
        ("e7-ctl-fullband-percussion-48k", 48000, 2.5, 741, "fullband_clean_control", "fullband_clean_control",
         20000.0, 24, {"percussive": True, "bright_cymbals": True}),
        ("e7-ctl-fullband-plucked-48k", 48000, 2.5, 742, "fullband_clean_control", "fullband_clean_control",
         20000.0, 24, {"plucked": True, "bright_cymbals": True}),
        ("e7-ctl-fullband-harmonic-48k", 48000, 2.5, 743, "fullband_clean_control", "fullband_clean_control",
         20000.0, 24, {"bright_cymbals": True, "quiet_tail": True}),

        # Stratum 6: OOD stress cases (4 kHz narrowband, wide stereo 8 kHz, double codec 8 kHz), 3 cases
        ("e7-ood-narrowband-4k-48k", 48000, 2.5, 751, "ood", "ood_narrowband_4k",
         4000.0, 24, {"native_low_sr": 8000, "percussive": True, "bright_cymbals": True}),
        ("e7-ood-wide-stereo-8k-48k", 48000, 2.5, 752, "ood", "ood_wide_stereo_8k",
         8000.0, 24, {"native_low_sr": 16000, "wide_stereo": True, "bright_cymbals": True}),
        ("e7-ood-double-codec-8k-48k", 48000, 2.5, 753, "ood", "ood_double_codec_8k",
         8000.0, 24, {"native_low_sr": 16000, "percussive": True, "bright_cymbals": True}),
    ]
    return [
        _case(cid, rate, sec, seed, stratum, op, cutoff, bits, idx, **kw)
        for idx, (cid, rate, sec, seed, stratum, op, cutoff, bits, kw) in enumerate(specs, 1)
    ]


def synthesize_e7_arrays(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Synthesize deterministic float64 arrays for one E7 BWE case in memory."""
    syn = case["synthesis"]
    if syn.get("generator") != GENERATOR:
        raise ValueError(f"unsupported E7 generator: {syn.get('generator')!r}")
    clean, attack_mask = _bwe_clean_signal(
        int(syn["seed"]),
        int(syn["rate"]),
        float(syn["seconds"]),
        percussive=bool(syn.get("percussive", False)),
        plucked=bool(syn.get("plucked", False)),
        bright_cymbals=bool(syn.get("bright_cymbals", True)),
        wide_stereo=bool(syn.get("wide_stereo", False)),
        quiet_tail=bool(syn.get("quiet_tail", False)),
    )
    degraded, deg_meta = apply_bwe_degradation(clean, int(syn["rate"]), syn)
    return clean, degraded, attack_mask, deg_meta


def synthesize_e7_case(case: dict, audio_root: Path) -> tuple[dict, dict]:
    """Materialize WAVs and RLE attack mask for one E7 case under audio_root."""
    syn = case["synthesis"]
    clean, degraded, attack_mask, deg_meta = synthesize_e7_arrays(case)
    specs: dict[str, dict] = {}
    for role, pcm in (("clean", clean), ("degraded", degraded)):
        rel = case[role]["path"]
        target = (audio_root / rel).resolve()
        target.relative_to(audio_root.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        corpus_synth.write_wav(
            target, int(syn["rate"]), pcm, bits=int(syn.get("bits", 24)), seed=int(syn["seed"])
        )
        raw = target.read_bytes()
        specs[role] = {"path": rel, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    rel_m = case["attack_mask"]["path"]
    target_m = (audio_root / rel_m).resolve()
    target_m.relative_to(audio_root.resolve())
    blob = corpus_synth.encode_mask(attack_mask)
    target_m.write_bytes(blob)
    specs["attack_mask"] = {"path": rel_m, "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
    return specs, deg_meta


def build_e7_manifest(root: Path, *, check_dev_manifests: bool = True) -> dict:
    """Build and pin `corpus/dev-e7-bwe-v1.json` without touching sealed corpora."""
    cases = e7_bwe_cases()
    stub = {"tier": "dev", "dataset_version": DATASET_VERSION, "cases": cases}
    if check_dev_manifests:
        # Strictly check non-sealed manifests only; never open sealed-v1 or sealed-e5-v1.
        others = []
        for name in ("smoke-v1.json", "dev-v1.json", "dev-e3-v1.json", "dev-e6-archival-v1.json"):
            path = root / name
            if path.exists():
                others.append(corpus_v1.load(path))
        leaks = corpus_v1.validate_cross_manifest_leakage([stub, *others])
        if leaks:
            raise ValueError("cross-manifest group leakage: " + "; ".join(leaks))

    audio_root = root / "audio" / "dev-e7"
    audio_root.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    seen: set[str] = set()
    for case in cases:
        specs, deg_meta = synthesize_e7_case(case, audio_root)
        case["degradation_meta"] = deg_meta
        for role in ("clean", "degraded", "attack_mask"):
            case[role].update(specs[role])
            if specs[role]["path"] not in seen:
                seen.add(specs[role]["path"])
                files.append(specs[role])

    doc = {
        "schema": SCHEMA,
        "tier": "dev",
        "dataset_version": DATASET_VERSION,
        "sealed_corpora_opened": False,
        "toolchain": corpus_v1.actual_toolchain(),
        "rng": {"algorithm": "numpy-pcg64", "seed": 20261123},
        "strata_counts": {s: sum(1 for c in cases if c["stratum"] == s) for s in STRATA},
        "files": files,
        "cases": cases,
    }
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--root", default="corpus", help="Corpus root directory (default: corpus)")
    args = ap.parse_args()
    root = Path(args.root)
    doc = build_e7_manifest(root)
    target = root / "dev-e7-bwe-v1.json"
    target.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    digest = corpus_v1.manifest_digest(doc)
    print(f"{target}: {len(doc['cases'])} cases, sha256={digest[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
