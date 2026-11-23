#!/usr/bin/env python3
"""Deterministic synthetic stereo corpus generator (synth/v1).

Every case is fully described by its manifest `synthesis` block: generator name,
seed, sample rate, seconds, operator and operator parameters. Re-running the
generator must reproduce byte-identical WAV and mask files — the manifest pins
their sha256, so a different decoder, platform or edit is caught by validation.
No real audio is distributed by this module.
"""
import hashlib
import json
import wave

import numpy as np
from scipy.io import wavfile

GENERATOR = "synth/v1"
GENERATOR_V2 = "synth/v2-stereo-music"
GENERATORS = {GENERATOR, GENERATOR_V2}

# Operators that produce a defective input with a known true mask.
DEFECT_OPERATORS = {"hard_clip", "late_clip"}
# Operators that produce a non-defective negative control (mask all False).
CONTROL_OPERATORS = {"clean", "intentional_saturation", "soft_limiter"}


def _clean_signal(seed, rate, seconds, quiet_tail=False):
    """Stereo tonal/transient signal; channels decorrelated, deterministic."""
    rng = np.random.default_rng(seed)
    n = round(rate * seconds)
    t = np.arange(n) / rate
    out = np.zeros((n, 2))
    for ch in range(2):
        x = np.zeros(n)
        for _ in range(5):
            f = rng.uniform(80, 4000)
            x += rng.uniform(.05, .3) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        # sparse transients with exponential decay
        for _ in range(int(seconds * 2)):
            i = int(rng.uniform(0, max(1, n - rate // 10)))
            k = np.arange(min(n - i, int(rate * .05)))
            x[i:i + k.size] += rng.uniform(.4, .9) * np.exp(-k / (rate * .01)) * np.sin(
                2 * np.pi * rng.uniform(1500, 6000) * k / rate)
        x += rng.normal(0, 1e-4, n)  # low noise floor
        out[:, ch] = x
    peak = np.abs(out).max()
    if peak > 0:
        out *= .95 / peak
    if quiet_tail:
        tail = int(n * .8)
        out[tail:] *= .01  # -40 dB quiet tail region
    return out.astype(np.float64)


def _clean_music_signal(seed, rate, seconds, quiet_tail=False, percussive=False):
    """Deterministic stereo musical-structure signal (synth/v2-stereo-music).

    Combines a shared stereo-coherent bass foundation, harmonic chord voices
    panned across L/R with slow amplitude phrasing, and damped broadband
    percussive attacks. Designed for realistic stereo declipping evaluation
    without redistributing copyrighted recordings.
    """
    rng = np.random.default_rng(seed)
    n = round(rate * seconds)
    t = np.arange(n, dtype=np.float64) / rate
    out = np.zeros((n, 2), dtype=np.float64)

    # Shared stereo bass line with harmonic overtones (1st-4th harmonics).
    bass_f0 = rng.uniform(55.0, 98.0)
    bass_env = 0.65 + 0.35 * np.sin(2.0 * np.pi * rng.uniform(0.4, 1.1) * t + rng.uniform(0, 2 * np.pi))
    bass = np.zeros(n, dtype=np.float64)
    for h in range(1, 5):
        bass += (0.28 / (h ** 1.15)) * np.sin(2.0 * np.pi * (bass_f0 * h) * t + rng.uniform(0, 2 * np.pi))
    bass *= bass_env
    out[:, 0] += bass * 0.92
    out[:, 1] += bass * 0.88

    # Four stereo-panned harmonic instrument voices (root + 5 overtones each).
    roots = bass_f0 * np.array([2.0, 2.5, 3.0, 3.75]) * rng.uniform(0.95, 1.05, size=4)
    for f0 in roots:
        pan = rng.uniform(0.15, 0.85)
        gl, gr = np.cos(pan * np.pi / 2.0), np.sin(pan * np.pi / 2.0)
        mod_f = rng.uniform(0.3, 1.4)
        env = 0.45 + 0.55 * np.maximum(0.0, np.sin(2.0 * np.pi * mod_f * t + rng.uniform(0, 2 * np.pi)))
        voice = np.zeros(n, dtype=np.float64)
        for h in range(1, 6):
            if f0 * h >= 0.45 * rate:
                break
            amp = rng.uniform(0.06, 0.16) / (h ** 0.9)
            voice += amp * np.sin(2.0 * np.pi * (f0 * h) * t + rng.uniform(0, 2 * np.pi))
        voice *= env
        out[:, 0] += gl * voice
        out[:, 1] += gr * voice

    # Percussive damped attacks (denser when percussive=True).
    n_hits = max(2, int(seconds * (6 if percussive else 2.5)))
    for _ in range(n_hits):
        i = int(rng.uniform(0, max(1, n - int(rate * 0.12))))
        dur = min(n - i, int(rate * 0.08))
        k = np.arange(dur, dtype=np.float64) / rate
        decay = np.exp(-k / rng.uniform(0.008, 0.025))
        tone = (np.sin(2.0 * np.pi * rng.uniform(180, 600) * k)
                + 0.4 * np.sin(2.0 * np.pi * rng.uniform(1200, 4800) * k)
                + 0.15 * rng.normal(0.0, 1.0, dur))
        hit = rng.uniform(0.35, 0.85 if percussive else 0.55) * decay * tone
        pan = rng.uniform(0.2, 0.8)
        out[i:i + dur, 0] += np.cos(pan * np.pi / 2.0) * hit
        out[i:i + dur, 1] += np.sin(pan * np.pi / 2.0) * hit

    # Low stereo noise floor.
    out += rng.normal(0.0, 8e-5, out.shape)
    peak = float(np.abs(out).max())
    if peak > 0:
        out *= 0.95 / peak
    if quiet_tail:
        tail = int(n * 0.8)
        out[tail:] *= 0.01  # -40 dB quiet tail region
    return out.astype(np.float64)


def _tpdf_dither(rng, n, lsb):
    return (rng.random(n) - rng.random(n)) * lsb


def apply_operator(clean, rate, spec, seed):
    """Return (degraded float64, true_mask bool[n,ch], meta)."""
    rng = np.random.default_rng(seed + 1)
    op = spec["operator"]
    mask = np.zeros(clean.shape, dtype=bool)
    x = clean.copy()
    bits = int(spec.get("bits", 16))
    lsb = 1.0 / (2 ** (bits - 1)) if bits in (16, 24) else 1.0 / 32768.0
    if spec.get("dither"):
        x += _tpdf_dither(rng, x.shape, lsb)
    if op == "clean":
        pass
    elif op == "intentional_saturation":
        drive = spec.get("drive", 3.0)
        x = np.tanh(x * drive) / np.tanh(drive)
    elif op == "soft_limiter":
        knee = spec.get("knee", .8)
        over = np.abs(x) > knee
        x = np.where(over, np.sign(x) * (knee + (np.abs(x) - knee) / (1 + 4 * (np.abs(x) - knee))), x)
    elif op in ("hard_clip", "late_clip"):
        thr = 10 ** (spec["threshold_dbfs"] / 20)
        pos = thr * spec.get("pos_scale", 1.0)   # asymmetric clipping
        neg = thr * spec.get("neg_scale", 1.0)
        start = round(spec.get("defect_start_s", 0) * rate)
        ch_mode = spec.get("channel_mode", "both")
        chs = (0,) if ch_mode == "left_only" else (1,) if ch_mode == "right_only" else (0, 1)
        gain = spec.get("defect_gain", 1.0)
        for ch in chs:
            region = x[start:, ch] * gain
            clipped = np.clip(region, -neg, pos)
            mask[start:, ch] = clipped != region
            x[start:, ch] = clipped
    else:
        raise ValueError(f"unknown operator: {op}")
    return x, mask, {"operator": op, **{k: v for k, v in spec.items() if k != "operator"}}


def _write_wav24(path, rate, pcm, seed=0):
    """Deterministic 24-bit signed little-endian PCM WAV with TPDF dither."""
    rng = np.random.default_rng(seed + 2)
    q = np.clip(pcm, -1.0, 1.0) * 8388608.0 + _tpdf_dither(rng, pcm.shape, 1.0)
    data = np.clip(np.round(q), -8388608, 8388607).astype("<i4")
    # Pack lower 3 bytes of each int32 sample (little-endian 24-bit PCM).
    raw = data.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(int(pcm.shape[1]))
        wf.setsampwidth(3)
        wf.setframerate(int(rate))
        wf.writeframes(raw)


def write_wav(path, rate, pcm, bits=16, seed=0):
    """Deterministic PCM WAV. int16/24 with TPDF dither, or float32 for 32."""
    if bits == 16:
        rng = np.random.default_rng(seed + 2)
        q = np.clip(pcm, -1, 1) * 32768 + _tpdf_dither(rng, pcm.shape, 1.0)
        data = np.clip(np.round(q), -32768, 32767).astype(np.int16)
        wavfile.write(path, rate, data)
    elif bits == 24:
        _write_wav24(path, rate, pcm, seed=seed)
    elif bits == 32:
        data = np.clip(pcm, -1, 1).astype(np.float32)
        wavfile.write(path, rate, data)
    else:
        raise ValueError(f"unsupported bits: {bits}")


def encode_mask(mask):
    """Run-length encode a bool[n,ch] mask to deterministic JSON bytes."""
    runs = []
    flat = mask.reshape(-1)
    idx = np.flatnonzero(flat)
    if idx.size:
        splits = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits, [idx.size - 1]))
        runs = [[int(idx[s]), int(idx[e] - idx[s] + 1)] for s, e in zip(starts, ends)]
    return json.dumps({"frames": int(mask.shape[0]), "channels": int(mask.shape[1]),
                       "runs": runs}, separators=(",", ":")).encode()


def decode_mask(blob):
    doc = json.loads(blob)
    total = doc["frames"] * doc["channels"]
    flat = np.zeros(total, dtype=bool)
    for start, length in doc["runs"]:
        if start < 0 or length <= 0 or start + length > total:
            raise ValueError(f"mask run out of bounds: [{start}, +{length}) of {total}")
        flat[start:start + length] = True
    return flat.reshape(doc["frames"], doc["channels"])


def synthesize_case(case, root):
    """Materialize a paired_synthetic case under root; return {role: file spec}."""
    syn = case["synthesis"]
    gen = syn["generator"]
    if gen == GENERATOR:
        clean = _clean_signal(syn["seed"], syn["rate"], syn["seconds"],
                              quiet_tail=syn.get("quiet_tail", False))
    elif gen == GENERATOR_V2:
        clean = _clean_music_signal(syn["seed"], syn["rate"], syn["seconds"],
                                    quiet_tail=syn.get("quiet_tail", False),
                                    percussive=syn.get("percussive", False))
    else:
        raise ValueError(f"unknown generator: {gen}")
    degraded, mask, _meta = apply_operator(clean, syn["rate"], syn, syn["seed"])
    if syn["operator"] in DEFECT_OPERATORS and not mask.any():
        raise ValueError(f"defect operator '{syn['operator']}' produced an empty "
                         f"mask for case {case['id']}: adjust threshold/gain/seed")
    specs = {}
    for role, pcm in (("clean", clean), ("degraded", degraded)):
        rel = case[role]["path"]
        target = (root / rel).resolve()
        target.relative_to(root.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        write_wav(target, syn["rate"], pcm, bits=syn.get("bits", 16), seed=syn["seed"])
        raw = target.read_bytes()
        specs[role] = {"path": rel, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    mask_rel = case["clean"]["path"].replace(".wav", ".mask")
    mask_target = root / mask_rel
    blob = encode_mask(mask)
    mask_target.write_bytes(blob)
    specs["mask"] = {"path": mask_rel, "sha256": hashlib.sha256(blob).hexdigest(),
                     "bytes": len(blob)}
    return specs
