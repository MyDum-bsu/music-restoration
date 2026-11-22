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

import numpy as np
from scipy.io import wavfile

GENERATOR = "synth/v1"

# Operators that produce a defective input with a known true mask.
DEFECT_OPERATORS = {"hard_clip", "late_clip"}
# Operators that produce a non-defective negative control (mask all False).
CONTROL_OPERATORS = {"clean", "intentional_saturation", "soft_limiter"}


def _clean_signal(seed, rate, seconds, quiet_tail=False):
    """Stereo tonal/transient signal; channels decorrelated, deterministic."""
    rng = np.random.default_rng(seed)
    n = int(round(rate * seconds))
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


def _tpdf_dither(rng, n, lsb):
    return (rng.random(n) - rng.random(n)) * lsb


def apply_operator(clean, rate, spec, seed):
    """Return (degraded float64, true_mask bool[n,ch], meta)."""
    rng = np.random.default_rng(seed + 1)
    op = spec["operator"]
    mask = np.zeros(clean.shape, dtype=bool)
    x = clean.copy()
    if spec.get("dither"):
        x += _tpdf_dither(rng, x.shape, 1 / 32768)
    if op == "clean":
        pass
    elif op == "intentional_saturation":
        x = np.tanh(x * spec.get("drive", 3.0)) / np.tanh(spec.get("drive", 3.0))
    elif op == "soft_limiter":
        knee = spec.get("knee", .8)
        over = np.abs(x) > knee
        x = np.where(over, np.sign(x) * (knee + (np.abs(x) - knee) / (1 + 4 * (np.abs(x) - knee))), x)
    elif op in ("hard_clip", "late_clip"):
        thr = 10 ** (spec["threshold_dbfs"] / 20)
        pos = thr * spec.get("pos_scale", 1.0)   # asymmetric clipping
        neg = thr * spec.get("neg_scale", 1.0)
        start = int(round(spec.get("defect_start_s", 0) * rate))
        region = x[start:]
        clipped = np.clip(region, -neg, pos)
        mask[start:] = clipped != region
        x[start:] = clipped
    else:
        raise ValueError(f"unknown operator: {op}")
    return x, mask, {"operator": op, **{k: v for k, v in spec.items() if k != "operator"}}


def write_wav(path, rate, pcm, bits=16, seed=0):
    """Deterministic PCM WAV. int16 with TPDF dither, or float32 for 32."""
    if bits == 16:
        rng = np.random.default_rng(seed + 2)
        q = np.clip(pcm, -1, 1) * 32768 + _tpdf_dither(rng, pcm.shape, 1.0)
        data = np.clip(np.round(q), -32768, 32767).astype(np.int16)
    elif bits == 32:
        data = np.clip(pcm, -1, 1).astype(np.float32)
    else:
        raise ValueError(f"unsupported bits: {bits}")
    wavfile.write(path, rate, data)


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
    flat = np.zeros(doc["frames"] * doc["channels"], dtype=bool)
    for start, length in doc["runs"]:
        flat[start:start + length] = True
    return flat.reshape(doc["frames"], doc["channels"])


def synthesize_case(case, root):
    """Materialize a paired_synthetic case under root; return {role: file spec}."""
    syn = case["synthesis"]
    if syn["generator"] != GENERATOR:
        raise ValueError(f"unknown generator: {syn['generator']}")
    clean = _clean_signal(syn["seed"], syn["rate"], syn["seconds"],
                          quiet_tail=syn.get("quiet_tail", False))
    degraded, mask, meta = apply_operator(clean, syn["rate"], syn, syn["seed"])
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
