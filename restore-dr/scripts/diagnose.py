#!/usr/bin/env python3
"""
diagnose.py — track diagnosis: artifacts and dynamic deficits -> JSON profile + treatment plan.

Metrics: clipping (density, flat runs), DC offset, subsonic <25 Hz, spectral
tilt and harshness, noise floor, crest profile (punch), 3 s envelope std
(macro), M/S balance, LUFS/LRA/TP. The plan triggers: hpf/declip/nd2/contour.

Usage: diagnose.py FILE... [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import decode, ebur128


def clip_stats(x16):
    """x16: integer codes. Density per million samples and flat runs >=3."""
    out = {}
    for c, name in enumerate(("L", "R")):
        xi = np.abs(x16[:, c])
        clipped = xi >= 32767
        d = np.diff(clipped.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if clipped[0]:
            starts = np.r_[0, starts]
        if clipped[-1]:
            ends = np.r_[ends, clipped.size]
        m = min(starts.size, ends.size)
        runs3 = int(np.sum((ends[:m] - starts[:m]) >= 3)) if m else 0
        out[name] = {
            "clipped_samples": int(np.sum(clipped)),
            "per_million": round(1e6 * float(np.sum(clipped)) / x16.size * 2, 2),
            "flat_runs_ge3": runs3,
        }
    return out


def spectral(x, fs):
    f, pxx = welch(x[:, 0], fs, nperseg=8192)
    _, pxy = welch(x[:, 1], fs, nperseg=8192)
    p = (pxx + pxy) / 2
    total = p.sum()
    sub = p[f < 25].sum() / max(total, 1e-20)
    band = lambda lo, hi: 10 * np.log10(max(p[(f >= lo) & (f < hi)].sum(), 1e-20))
    tilt = (band(8000, 16000) - band(100, 200)) / np.log2(12000 / 150)  # dB/octave
    harsh = band(3000, 5000) - (band(1000, 2000) + band(5000, 8000)) / 2
    return {"subsonic_ratio_below25": round(float(sub), 5),
            "spectral_tilt_db_per_oct": round(float(tilt), 2),
            "harshness_3_5k_db": round(float(harsh), 2)}


def dynamics(x, fs):
    win = fs
    nw = len(x) // win
    if nw < 4:
        return {}
    w = x[:nw * win].reshape(nw, win, 2)
    rms_db = 20 * np.log10(np.maximum(np.sqrt(2 * np.mean(w ** 2, axis=(1, 2))), 1e-6))
    pk_db = 20 * np.log10(np.maximum(np.abs(w).max(axis=(1, 2)), 1e-6))
    crest = pk_db - rms_db
    # 3 s envelope std (macro dynamics)
    w3 = fs * 3
    n3 = len(x) // w3
    r3 = 20 * np.log10(np.maximum(np.sqrt(2 * np.mean(x[:n3 * w3].reshape(n3, w3, 2) ** 2, axis=(1, 2))), 1e-6))
    q = lambda a: [round(float(np.percentile(a, p)), 1) for p in (5, 50, 95)]
    return {"crest_1s_p5_p50_p95": q(crest), "crest_median": round(float(np.median(crest)), 1),
            "env3s_std": round(float(np.std(r3)), 2)}


def ms_balance(x, fs):
    m = x.mean(axis=1)
    s = x[:, 0] - x[:, 1]
    ems = 10 * np.log10(max(np.mean(m ** 2), 1e-12) / max(np.mean(s ** 2), 1e-12))
    sos = butter(2, [200 / (fs / 2), 5000 / (fs / 2)], "bandpass", output="sos")
    mb, sb = sosfilt(sos, m), sosfilt(sos, s)
    ems_mid = 10 * np.log10(max(np.mean(mb ** 2), 1e-12) / max(np.mean(sb ** 2), 1e-12))
    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
    return {"ms_ratio_db": round(float(ems), 2), "ms_ratio_mid_db": round(float(ems_mid), 2),
            "lr_correlation": round(corr, 3)}


def noise_floor(x, fs):
    win = int(0.1 * fs)
    nw = len(x) // win
    r = 20 * np.log10(np.maximum(np.sqrt(2 * np.mean(x[:nw * win].reshape(nw, win, 2) ** 2, axis=(1, 2))), 1e-6))
    return {"floor_p10_db": round(float(np.percentile(r, 10)), 1)}


def treatment_plan(d):
    """Treatment-plan heuristics (reference-mimicry, diagnostic triggers)."""
    clips = d["clipping"]["L"]["flat_runs_ge3"] + d["clipping"]["R"]["flat_runs_ge3"]
    return {
        "declip": clips >= 20 or (d["clipping"]["L"]["clipped_samples"] + d["clipping"]["R"]["clipped_samples"]) >= 100,
        "hpf_25hz": d["spectral"]["subsonic_ratio_below25"] > 0.0005,
        "reference_eq": True,
        "contour": {"slope": 1.08, "offset_db": -4.6},
        "nd2_strength": 0.0,           # off in mimicry mode; set manually for punch deficit
        "side_db": -3.0,
        "tp_target_db": -0.5,
        "depth_bits": 16,
    }


def diagnose(file):
    xf, fs = decode(file, "f32le")
    x16, _ = decode(file, "s16le")
    i, lra, tp = ebur128(file)
    dc = [round(float(np.mean(xf[:, c])), 6) for c in (0, 1)]
    d = {
        "file": Path(file).name,
        "fs": fs, "lufs": round(i, 1), "lra": round(lra, 1), "true_peak_db": round(tp, 1),
        "dc_offset": dc,
        "clipping": clip_stats(x16),
        "spectral": spectral(xf, fs),
        "dynamics": dynamics(xf, fs),
        "stereo": ms_balance(xf, fs),
        "noise": noise_floor(xf, fs),
    }
    d["plan"] = treatment_plan(d)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--json")
    a = p.parse_args()
    rows = [diagnose(f) for f in a.files]
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for r in rows:
        print(f"== {r['file']}")
        print(f"   LUFS {r['lufs']}  LRA {r['lra']}  TP {r['true_peak_db']}  DC {r['dc_offset']}")
        print(f"   clip: {r['clipping']['L']} | {r['clipping']['R']}")
        print(f"   spectral: {r['spectral']}")
        print(f"   dynamics: {r['dynamics']}")
        print(f"   stereo: {r['stereo']}  noise: {r['noise']}")
        print(f"   plan: {r['plan']}")


if __name__ == "__main__":
    main()
