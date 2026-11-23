#!/usr/bin/env python3
"""Build the E6 archival de-click and hiss dev corpus (`corpus/dev-e6-archival-v1.json`).

    python3 research/build_e6_archival_corpus.py --root corpus

Creates 18 independent stereo archival-transfer dev cases across five strata:
- clicks_only (4 cases): sparse pops, dense shellac/vinyl crackle, clicks overlaid
  directly on sharp percussive attacks, and single-channel stylus scratches;
- hiss_only (4 cases): stationary tape hiss, vinyl pink/HF hiss, amplitude-modulated
  wow/oxide hiss, and high-SNR mild archival hiss;
- combined_archival (3 cases): vinyl transfer, tape+crackle+percussion, and 78 rpm
  narrowband shellac transfer;
- clean_control (4 cases): sharp percussive attacks, plucked strings, clean quiet
  reverb tail, and clean stereo tonal mix (zero defects);
- ood (3 cases): wide 20-36 sample scratches, non-stationary mains hum + floor step,
  and 96 kHz ultrasonic percussion with sparse clicks.

Security & leakage boundaries:
- Never opens or inspects `corpus/sealed-v1.json` or `corpus/sealed-e5-v1.json`.
- Checks cross-manifest group-key disjointness against dev/smoke manifests only
  (`smoke-v1.json`, `dev-v1.json`, `dev-e3-v1.json`).
- Stores relative paths and SHA-256 hashes only; zero host-specific paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research import corpus_synth, corpus_v1

SCHEMA = "e6-archival-corpus/v1"
GENERATOR = "synth/e6-archival-v1"
DATASET_VERSION = "dev-e6-archival-2026-11-23"

RIGHTS_DEV = {
    "local_inference": True,
    "colab": True,
    "training": True,
    "listening": True,
    "publication": False,
}

STRATA = ("clicks_only", "hiss_only", "combined_archival", "clean_control", "ood")


def _groups(idx: int) -> dict[str, str]:
    return {
        "take": f"t-e6-{idx:02d}",
        "fingerprint": f"fp-e6-{idx:02d}",
        "session": f"ses-e6-{idx:02d}",
        "artist": f"e6-artist-{idx:02d}",
        "album": f"e6-album-{idx:02d}",
        "stem_family": f"fam-e6-{idx:02d}",
    }


def _butter_filter(x: np.ndarray, rate: int, cutoff_hz: float | tuple[float, float],
                   btype: str, order: int = 4) -> np.ndarray:
    nyq = 0.5 * rate
    if isinstance(cutoff_hz, tuple):
        wn = (min(0.95, cutoff_hz[0] / nyq), min(0.95, cutoff_hz[1] / nyq))
    else:
        wn = min(0.95, cutoff_hz / nyq)
    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def _archival_clean_signal(
    seed: int,
    rate: int,
    seconds: float,
    *,
    percussive: bool = False,
    plucked: bool = False,
    quiet_tail: bool = False,
    narrowband: bool = False,
    ultrasonic: bool = False,
    noise_floor_rms: float = 7.5e-5,
) -> tuple[np.ndarray, np.ndarray, list[int] | None]:
    """Generate deterministic stereo music with explicit attack_mask and tail_slice.

    Returns:
        clean: float64 array of shape (n_samples, 2)
        attack_mask: bool array of shape (n_samples, 2) marking genuine musical attacks
        tail_slice: [start, end] sample indices of quiet reverb tail, or None
    """
    rng = np.random.default_rng(seed)
    n = round(rate * seconds)
    t = np.arange(n, dtype=np.float64) / rate
    out = np.zeros((n, 2), dtype=np.float64)
    attack_mask = np.zeros((n, 2), dtype=bool)

    active_end = int(n * 0.72) if quiet_tail else n

    # 1. Stereo harmonic foundation (bass + 4 panned voices).
    bass_f0 = float(rng.uniform(60.0, 105.0))
    bass_env = 0.65 + 0.35 * np.sin(2.0 * np.pi * rng.uniform(0.45, 1.1) * t + rng.uniform(0, 2 * np.pi))
    bass = np.zeros(n, dtype=np.float64)
    for h in range(1, 5):
        bass += (0.26 / (h ** 1.15)) * np.sin(2.0 * np.pi * (bass_f0 * h) * t + rng.uniform(0, 2 * np.pi))
    bass *= bass_env
    out[:, 0] += 0.90 * bass
    out[:, 1] += 0.86 * bass

    roots = bass_f0 * np.array([2.0, 2.5, 3.0, 3.75]) * rng.uniform(0.95, 1.05, size=4)
    for f0 in roots:
        pan = float(rng.uniform(0.18, 0.82))
        gl, gr = np.cos(pan * np.pi / 2.0), np.sin(pan * np.pi / 2.0)
        mod_f = float(rng.uniform(0.35, 1.3))
        env = 0.45 + 0.55 * np.maximum(0.0, np.sin(2.0 * np.pi * mod_f * t + rng.uniform(0, 2 * np.pi)))
        voice = np.zeros(n, dtype=np.float64)
        for h in range(1, 7):
            if f0 * h >= 0.45 * rate:
                break
            amp = float(rng.uniform(0.06, 0.15)) / (h ** 0.95)
            voice += amp * np.sin(2.0 * np.pi * (f0 * h) * t + rng.uniform(0, 2 * np.pi))
        voice *= env
        out[:, 0] += gl * voice
        out[:, 1] += gr * voice

    # 2. Sharp percussive attacks (snare rimshots, hi-hats, woodblocks).
    if percussive or ultrasonic:
        n_hits = max(6, int(seconds * 5))
        step = max(1, (active_end - int(rate * 0.15)) // max(1, n_hits))
        for k_idx in range(n_hits):
            center = int(rate * 0.08) + k_idx * step + int(rng.integers(-step // 6, max(1, step // 6)))
            i = int(np.clip(center, int(rate * 0.03), max(int(rate * 0.03) + 1, active_end - int(rate * 0.1))))
            dur = min(active_end - i, int(rate * 0.09))
            if dur < 32:
                continue
            kt = np.arange(dur, dtype=np.float64) / rate
            rise_s = float(rng.uniform(0.0003, 0.0012))  # 0.3 - 1.2 ms fast musical attack
            decay_s = float(rng.uniform(0.012, 0.038))
            env = (1.0 - np.exp(-kt / rise_s)) * np.exp(-kt / decay_s)
            body = (
                0.7 * np.sin(2.0 * np.pi * rng.uniform(180.0, 480.0) * kt)
                + 0.55 * np.sin(2.0 * np.pi * rng.uniform(1400.0, 4200.0) * kt)
                + 0.45 * rng.normal(0.0, 1.0, dur)
            )
            if ultrasonic and rate >= 96000:
                body += 0.45 * np.sin(2.0 * np.pi * rng.uniform(16000.0, 26000.0) * kt)
            hit = float(rng.uniform(0.45, 0.90)) * env * body
            pan = float(rng.uniform(0.22, 0.78))
            out[i:i + dur, 0] += np.cos(pan * np.pi / 2.0) * hit
            out[i:i + dur, 1] += np.sin(pan * np.pi / 2.0) * hit
            win_pre = round(0.0015 * rate)
            win_post = round(0.015 * rate)
            attack_mask[max(0, i - win_pre):min(n, i + win_post), :] = True

    # 3. Sharp plucked strings (harpsichord / acoustic guitar onsets + harmonic decay).
    if plucked:
        n_plucks = max(6, int(seconds * 4))
        step = max(1, (active_end - int(rate * 0.18)) // max(1, n_plucks))
        for k_idx in range(n_plucks):
            i = int(rate * 0.06) + k_idx * step
            dur = min(active_end - i, int(rate * 0.25))
            if dur < 64:
                continue
            kt = np.arange(dur, dtype=np.float64) / rate
            f0 = float(rng.uniform(220.0, 660.0))
            pluck = np.zeros(dur, dtype=np.float64)
            for h in range(1, 10):
                if f0 * h >= 0.44 * rate:
                    break
                # Higher harmonics decay faster like a real plucked string.
                h_decay = np.exp(-kt / (0.14 / (h ** 0.75)))
                pluck += (0.5 / (h ** 0.85)) * h_decay * np.sin(2.0 * np.pi * f0 * h * kt)
            pluck *= 1.0 - np.exp(-kt / 0.0005)
            pan = float(rng.uniform(0.2, 0.8))
            out[i:i + dur, 0] += np.cos(pan * np.pi / 2.0) * pluck
            out[i:i + dur, 1] += np.sin(pan * np.pi / 2.0) * pluck
            win_pre = round(0.0015 * rate)
            win_post = round(0.015 * rate)
            attack_mask[max(0, i - win_pre):min(n, i + win_post), :] = True

    # 4. Smooth quiet reverb tail (last 25% of track, ~ -36 dB below main body).
    tail_slice: list[int] | None = None
    if quiet_tail:
        fade_len = round(0.08 * rate)
        tail_start = active_end
        fade_end = min(n, tail_start + fade_len)
        ramp = np.linspace(1.0, 0.015, fade_end - tail_start, dtype=np.float64)[:, None]
        out[tail_start:fade_end] *= ramp
        if fade_end < n:
            tail_t = np.arange(n - fade_end, dtype=np.float64) / rate
            tail_decay = 0.015 * np.exp(-tail_t / max(0.35, seconds * 0.22))[:, None]
            out[fade_end:] *= tail_decay
        tail_slice = [int(fade_end), int(n)]

    # 5. Narrowband archival shellac shaping (300 Hz - 7 kHz).
    if narrowband:
        out = _butter_filter(out, rate, (280.0, 7000.0), btype="bandpass", order=4)

    # Configurable clean noise floor (default -82 dBFS; room-tone control uses ~ -58 dBFS).
    out += rng.normal(0.0, noise_floor_rms, out.shape)
    peak = float(np.abs(out).max())
    if peak > 0.0:
        out *= 0.78 / peak
    return np.ascontiguousarray(out, dtype=np.float64), attack_mask, tail_slice


def _generate_colored_hiss(
    rng: np.random.Generator,
    n: int,
    rate: int,
    color: str,
    modulated: bool = False,
) -> np.ndarray:
    """Synthesize stationary or modulated stereo tape/vinyl hiss."""
    white = rng.normal(0.0, 1.0, (n, 2))
    # Slight stereo correlation (0.15) realistic for tape/vinyl surface noise.
    shared = rng.normal(0.0, 1.0, (n, 1))
    noise = 0.92 * white + 0.39 * shared

    if color == "tape_white_hf":
        # Mild HF emphasis above 3 kHz typical of analog tape bias/oxide hiss.
        hf = _butter_filter(noise, rate, 3000.0, btype="highpass", order=2)
        noise = 0.55 * noise + 0.65 * hf
    elif color == "vinyl_pink_hf":
        # 1/f-like mid/low rumble + high-frequency vinyl surface hiss.
        low = _butter_filter(noise, rate, 1200.0, btype="lowpass", order=2)
        hf = _butter_filter(noise, rate, 4500.0, btype="highpass", order=2)
        noise = 0.50 * low + 0.35 * noise + 0.55 * hf
    elif color == "shellac_broadband":
        noise = _butter_filter(noise, rate, (200.0, 8500.0), btype="bandpass", order=2)
    else:
        raise ValueError(f"unknown hiss color: {color}")

    if modulated:
        t = np.arange(n, dtype=np.float64) / rate
        # 0.55 Hz turntable revolution (33.3 RPM) + 1.8 Hz oxide flutter modulation.
        mod = 1.0 + 0.35 * np.sin(2.0 * np.pi * 0.55 * t) + 0.15 * np.sin(2.0 * np.pi * 1.8 * t + 1.1)
        noise *= mod[:, None]

    rms = float(np.sqrt(np.mean(noise ** 2)))
    if rms > 0.0:
        noise /= rms
    return noise


def _inject_clicks(
    x: np.ndarray,
    rate: int,
    rng: np.random.Generator,
    *,
    n_clicks: int,
    width_range: tuple[int, int],
    amp_range: tuple[float, float],
    channel_mode: str = "both_independent",
    on_attack_mask: np.ndarray | None = None,
    tail_slice: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject localized impulsive clicks/crackles and return (degraded, click_mask)."""
    n, n_ch = x.shape
    out = x.copy()
    mask = np.zeros((n, n_ch), dtype=bool)
    margin = max(64, round(0.01 * rate))
    max_pos = (tail_slice[0] - margin) if tail_slice is not None else (n - margin)
    max_pos = max(margin + 32, max_pos)

    attack_indices = np.flatnonzero(on_attack_mask[:, 0]) if (on_attack_mask is not None and on_attack_mask.any()) else None

    for idx in range(n_clicks):
        w_min, w_max = width_range
        w = int(rng.integers(w_min, w_max + 1))
        if attack_indices is not None and idx % 2 == 0:
            # Place half of the clicks directly near/on sharp percussive attacks.
            pos = int(rng.choice(attack_indices))
            pos = int(np.clip(pos, margin, max(margin, n - margin - w)))
        else:
            pos = int(rng.integers(margin, max(margin + 1, max_pos - w)))

        if channel_mode == "left_heavy":
            chs = (0,) if rng.random() < 0.8 else (1,)
        elif channel_mode == "stereo_linked":
            chs = (0, 1)
        else:
            chs = (int(rng.integers(0, n_ch)),) if rng.random() < 0.75 else (0, 1)

        amp = float(rng.uniform(amp_range[0], amp_range[1]))
        polarity = 1.0 if rng.random() < 0.5 else -1.0

        # Shape impulse: sharp single-sample spike or damped micro-burst.
        if w == 1:
            pulse = np.array([polarity * amp], dtype=np.float64)
        else:
            k = np.arange(w, dtype=np.float64)
            pulse = polarity * amp * np.exp(-k / max(1.0, 0.35 * w)) * np.cos(0.5 * np.pi * k / max(1.0, w))
            pulse[0] = polarity * amp

        for ch in chs:
            # Scale slightly per channel when stereo-linked.
            ch_scale = 1.0 if len(chs) == 1 else float(rng.uniform(0.85, 1.0))
            out[pos:pos + w, ch] += ch_scale * pulse
            mask[pos:pos + w, ch] = True

    return np.clip(out, -0.995, 0.995), mask


def apply_archival_operator(
    clean: np.ndarray,
    rate: int,
    spec: dict,
    seed: int,
    *,
    attack_mask: np.ndarray | None = None,
    tail_slice: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply an archival degradation operator to clean stereo PCM.

    Returns:
        degraded: float64 (n, 2)
        click_mask: bool (n, 2) exact mask of injected impulsive clicks/crackles
        meta: dict describing effective degradation parameters
    """
    rng = np.random.default_rng(seed + 101)
    op = spec["operator"]
    x = clean.copy()
    click_mask = np.zeros(clean.shape, dtype=bool)
    meta: dict = {"operator": op}

    # Active-body RMS (excluding quiet tail) so target SNR is physically calibrated.
    body_end = tail_slice[0] if tail_slice is not None else clean.shape[0]
    sig_rms = float(np.sqrt(np.mean(clean[:body_end] ** 2))) + 1e-12

    if op == "clean_control":
        pass
    elif op == "clicks_only":
        x, click_mask = _inject_clicks(
            x,
            rate,
            rng,
            n_clicks=int(spec.get("n_clicks", 28)),
            width_range=tuple(spec.get("width_range", (1, 5))),
            amp_range=tuple(spec.get("amp_range", (0.28, 0.68))),
            channel_mode=str(spec.get("channel_mode", "both_independent")),
            on_attack_mask=attack_mask if spec.get("overlay_on_attacks", False) else None,
            tail_slice=tail_slice,
        )
    elif op == "hiss_only":
        snr_db = float(spec.get("target_snr_db", 18.0))
        color = str(spec.get("hiss_color", "tape_white_hf"))
        modulated = bool(spec.get("modulated", False))
        noise = _generate_colored_hiss(rng, clean.shape[0], rate, color, modulated=modulated)
        noise_rms = sig_rms * (10.0 ** (-snr_db / 20.0))
        x = np.clip(x + noise * noise_rms, -0.995, 0.995)
        meta["noise_rms"] = noise_rms
    elif op == "combined_archival":
        snr_db = float(spec.get("target_snr_db", 16.0))
        color = str(spec.get("hiss_color", "vinyl_pink_hf"))
        modulated = bool(spec.get("modulated", False))
        noise = _generate_colored_hiss(rng, clean.shape[0], rate, color, modulated=modulated)
        noise_rms = sig_rms * (10.0 ** (-snr_db / 20.0))
        x = x + noise * noise_rms
        x, click_mask = _inject_clicks(
            x,
            rate,
            rng,
            n_clicks=int(spec.get("n_clicks", 36)),
            width_range=tuple(spec.get("width_range", (1, 6))),
            amp_range=tuple(spec.get("amp_range", (0.25, 0.65))),
            channel_mode=str(spec.get("channel_mode", "both_independent")),
            on_attack_mask=attack_mask if spec.get("overlay_on_attacks", False) else None,
            tail_slice=tail_slice,
        )
        meta["noise_rms"] = noise_rms
    elif op == "ood_wide_scratch":
        x, click_mask = _inject_clicks(
            x,
            rate,
            rng,
            n_clicks=int(spec.get("n_clicks", 12)),
            width_range=tuple(spec.get("width_range", (20, 36))),
            amp_range=tuple(spec.get("amp_range", (0.35, 0.72))),
            channel_mode="stereo_linked",
            tail_slice=tail_slice,
        )
    elif op == "ood_nonstationary_hum_hiss":
        snr_db = float(spec.get("target_snr_db", 18.0))
        noise = _generate_colored_hiss(rng, clean.shape[0], rate, "tape_white_hf", modulated=False)
        n = clean.shape[0]
        # Sudden +6 dB noise floor step mid-track + 50/100/150 Hz mains hum.
        step_env = np.ones((n, 1), dtype=np.float64)
        step_env[n // 2:] = 2.0
        t = np.arange(n, dtype=np.float64) / rate
        hum = (
            0.035 * np.sin(2.0 * np.pi * 50.0 * t)
            + 0.020 * np.sin(2.0 * np.pi * 100.0 * t + 0.4)
            + 0.010 * np.sin(2.0 * np.pi * 150.0 * t + 0.9)
        )[:, None]
        noise_rms = sig_rms * (10.0 ** (-snr_db / 20.0))
        x = np.clip(x + noise * noise_rms * step_env + hum, -0.995, 0.995)
        meta["noise_rms"] = noise_rms
    elif op == "ood_ultrasonic_perc":
        x, click_mask = _inject_clicks(
            x,
            rate,
            rng,
            n_clicks=int(spec.get("n_clicks", 16)),
            width_range=tuple(spec.get("width_range", (1, 4))),
            amp_range=tuple(spec.get("amp_range", (0.28, 0.60))),
            channel_mode="both_independent",
            on_attack_mask=attack_mask,
            tail_slice=tail_slice,
        )
    else:
        raise ValueError(f"unknown E6 archival operator: {op}")

    return np.ascontiguousarray(x, dtype=np.float64), click_mask, meta


def _case(cid: str, rate: int, seconds: float, seed: int, stratum: str, op: str,
          bits: int, idx: int, **kw) -> dict:
    return {
        "id": cid,
        "kind": "paired_synthetic_archival",
        "stratum": stratum,
        "split": "control" if stratum == "clean_control" else "eval",
        "rate": rate,
        "channels": 2,
        "bits": bits,
        "rights": dict(RIGHTS_DEV),
        "groups": _groups(idx),
        "clean": {"path": f"{cid}.clean.wav"},
        "degraded": {"path": f"{cid}.degraded.wav"},
        "click_mask": {"path": f"{cid}.click_mask.json"},
        "attack_mask": {"path": f"{cid}.attack_mask.json"},
        "synthesis": {
            "generator": GENERATOR,
            "seed": seed,
            "rate": rate,
            "seconds": seconds,
            "operator": op,
            "bits": bits,
            **kw,
        },
    }


def e6_archival_cases() -> list[dict]:
    """20 independent stereo archival-transfer dev cases across 5 strata."""
    specs = [
        # Stratum 1: Clicks & crackles only (5 cases, including low-level vinyl/shellac crackle)
        ("e6-clicks-sparse-tonal-44k", 44100, 3.0, 601, "clicks_only", "clicks_only", 16,
         {"n_clicks": 22, "width_range": [1, 5], "amp_range": [0.30, 0.72]}),
        ("e6-clicks-dense-crackle-48k", 48000, 3.0, 602, "clicks_only", "clicks_only", 24,
         {"n_clicks": 85, "width_range": [1, 4], "amp_range": [0.22, 0.60]}),
        ("e6-clicks-on-percussion-48k", 48000, 3.0, 603, "clicks_only", "clicks_only", 24,
         {"percussive": True, "overlay_on_attacks": True, "n_clicks": 32,
          "width_range": [1, 5], "amp_range": [0.32, 0.70]}),
        ("e6-clicks-single-ch-44k", 44100, 3.0, 604, "clicks_only", "clicks_only", 16,
         {"n_clicks": 36, "width_range": [1, 6], "amp_range": [0.28, 0.68],
          "channel_mode": "left_heavy"}),
        ("e6-clicks-low-level-crackle-44k", 44100, 3.0, 605, "clicks_only", "clicks_only", 24,
         {"n_clicks": 90, "width_range": [1, 4], "amp_range": [0.03, 0.08]}),

        # Stratum 2: Stationary & modulated hiss only (4 cases)
        ("e6-hiss-tape-white-20db-44k", 44100, 3.0, 611, "hiss_only", "hiss_only", 16,
         {"target_snr_db": 20.0, "hiss_color": "tape_white_hf", "quiet_tail": True}),
        ("e6-hiss-vinyl-pink-15db-48k", 48000, 3.0, 612, "hiss_only", "hiss_only", 24,
         {"target_snr_db": 15.0, "hiss_color": "vinyl_pink_hf", "quiet_tail": True}),
        ("e6-hiss-modulated-wow-18db-44k", 44100, 3.0, 613, "hiss_only", "hiss_only", 16,
         {"target_snr_db": 18.0, "hiss_color": "tape_white_hf", "modulated": True,
          "quiet_tail": True}),
        ("e6-hiss-high-snr-30db-48k", 48000, 3.0, 614, "hiss_only", "hiss_only", 24,
         {"target_snr_db": 30.0, "hiss_color": "tape_white_hf", "quiet_tail": True}),

        # Stratum 3: Combined archival profiles (3 cases)
        ("e6-archival-vinyl-44k", 44100, 3.0, 621, "combined_archival", "combined_archival", 16,
         {"target_snr_db": 18.0, "hiss_color": "vinyl_pink_hf", "n_clicks": 35,
          "width_range": [1, 5], "amp_range": [0.26, 0.65], "quiet_tail": True}),
        ("e6-archival-tape-crackle-48k", 48000, 3.0, 622, "combined_archival", "combined_archival", 24,
         {"target_snr_db": 16.0, "hiss_color": "tape_white_hf", "modulated": True,
          "percussive": True, "overlay_on_attacks": True, "n_clicks": 30,
          "width_range": [1, 5], "amp_range": [0.30, 0.68], "quiet_tail": True}),
        ("e6-archival-78rpm-shellac-44k", 44100, 3.0, 623, "combined_archival", "combined_archival", 16,
         {"target_snr_db": 12.0, "hiss_color": "shellac_broadband", "narrowband": True,
          "n_clicks": 70, "width_range": [1, 6], "amp_range": [0.24, 0.62]}),

        # Stratum 4: Clean negative controls (5 cases, ZERO defects; includes -58 dBFS room-tone control)
        ("e6-ctl-clean-percussion-48k", 48000, 3.0, 631, "clean_control", "clean_control", 24,
         {"percussive": True}),
        ("e6-ctl-clean-plucked-44k", 44100, 3.0, 632, "clean_control", "clean_control", 16,
         {"plucked": True}),
        ("e6-ctl-clean-reverb-tail-48k", 48000, 3.0, 633, "clean_control", "clean_control", 24,
         {"quiet_tail": True}),
        ("e6-ctl-clean-tonal-44k", 44100, 3.0, 634, "clean_control", "clean_control", 16,
         {}),
        ("e6-ctl-clean-roomtone-tail-48k", 48000, 3.0, 635, "clean_control", "clean_control", 24,
         {"percussive": True, "quiet_tail": True, "noise_floor_rms": 1.25e-3}),

        # Stratum 5: OOD cases (3 cases)
        ("e6-ood-wide-scratch-44k", 44100, 3.0, 641, "ood", "ood_wide_scratch", 16,
         {"n_clicks": 10, "width_range": [20, 34], "amp_range": [0.35, 0.70]}),
        ("e6-ood-nonstationary-hum-hiss-48k", 48000, 3.0, 642, "ood", "ood_nonstationary_hum_hiss", 24,
         {"target_snr_db": 18.0}),
        ("e6-ood-ultrasonic-perc-96k", 96000, 2.0, 643, "ood", "ood_ultrasonic_perc", 24,
         {"percussive": True, "ultrasonic": True, "n_clicks": 16,
          "width_range": [1, 4], "amp_range": [0.28, 0.60]}),
    ]
    return [
        _case(cid, rate, sec, seed, stratum, op, bits, idx, **kw)
        for idx, (cid, rate, sec, seed, stratum, op, bits, kw) in enumerate(specs, 1)
    ]


def synthesize_e6_arrays(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int] | None]:
    """Synthesize deterministic float64 arrays for one E6 archival case in memory."""
    syn = case["synthesis"]
    if syn.get("generator") != GENERATOR:
        raise ValueError(f"unsupported E6 generator: {syn.get('generator')!r}")
    clean, attack_mask, tail_slice = _archival_clean_signal(
        int(syn["seed"]),
        int(syn["rate"]),
        float(syn["seconds"]),
        percussive=bool(syn.get("percussive", False)),
        plucked=bool(syn.get("plucked", False)),
        quiet_tail=bool(syn.get("quiet_tail", False)),
        narrowband=bool(syn.get("narrowband", False)),
        ultrasonic=bool(syn.get("ultrasonic", False)),
        noise_floor_rms=float(syn.get("noise_floor_rms", 7.5e-5)),
    )
    degraded, click_mask, _meta = apply_archival_operator(
        clean,
        int(syn["rate"]),
        syn,
        int(syn["seed"]),
        attack_mask=attack_mask,
        tail_slice=tail_slice,
    )
    return clean, degraded, click_mask, attack_mask, tail_slice


def synthesize_e6_case(case: dict, audio_root: Path) -> tuple[dict, list[int] | None]:
    """Materialize WAVs and RLE masks for one E6 case under audio_root; return file specs."""
    syn = case["synthesis"]
    clean, degraded, click_mask, attack_mask, tail_slice = synthesize_e6_arrays(case)
    specs: dict[str, dict] = {}
    for role, pcm in (("clean", clean), ("degraded", degraded)):
        rel = case[role]["path"]
        target = (audio_root / rel).resolve()
        target.relative_to(audio_root.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        corpus_synth.write_wav(target, int(syn["rate"]), pcm, bits=int(syn.get("bits", 16)), seed=int(syn["seed"]))
        raw = target.read_bytes()
        specs[role] = {"path": rel, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    for role, mask in (("click_mask", click_mask), ("attack_mask", attack_mask)):
        rel = case[role]["path"]
        target = (audio_root / rel).resolve()
        target.relative_to(audio_root.resolve())
        blob = corpus_synth.encode_mask(mask)
        target.write_bytes(blob)
        specs[role] = {"path": rel, "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
    return specs, tail_slice


def build_e6_manifest(root: Path, *, check_dev_manifests: bool = True) -> dict:
    """Build and pin `corpus/dev-e6-archival-v1.json` without touching sealed corpora."""
    cases = e6_archival_cases()
    stub = {"tier": "dev", "dataset_version": DATASET_VERSION, "cases": cases}
    if check_dev_manifests:
        # Strictly check non-sealed manifests only; never open sealed-v1 or sealed-e5-v1.
        others = []
        for name in ("smoke-v1.json", "dev-v1.json", "dev-e3-v1.json"):
            path = root / name
            if path.exists():
                others.append(corpus_v1.load(path))
        leaks = corpus_v1.validate_cross_manifest_leakage([stub, *others])
        if leaks:
            raise ValueError("cross-manifest group leakage: " + "; ".join(leaks))

    audio_root = root / "audio" / "dev-e6"
    audio_root.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    seen: set[str] = set()
    for case in cases:
        specs, tail_slice = synthesize_e6_case(case, audio_root)
        case["tail_slice"] = tail_slice
        for role in ("clean", "degraded", "click_mask", "attack_mask"):
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
        "strata_counts": {
            s: sum(1 for c in cases if c["stratum"] == s) for s in STRATA
        },
        "files": files,
        "cases": cases,
    }
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--root", default="corpus", help="Corpus root directory (default: corpus)")
    args = ap.parse_args()
    root = Path(args.root)
    doc = build_e6_manifest(root)
    target = root / "dev-e6-archival-v1.json"
    target.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    digest = corpus_v1.manifest_digest(doc)
    print(f"{target}: {len(doc['cases'])} cases, sha256={digest[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
