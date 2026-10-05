#!/usr/bin/env python3
"""
measure.py — DR-restoration metrics, compatible with TT DR / dr.loudness-war.info.

Implements TT DR (the dr14_t.meter algorithm, checked against
simon-r/dr14_t.meter):
  * 3 s non-overlapping RMS blocks (132480 samples at 44.1k);
  * RMS with x√2 correction (full-scale sine -> 0 dBFS);
  * top-20% blocks by RMS, energy mean (sqrt(mean(x^2)));
  * peak = second-highest block peak;
  * channel DR = peak_dB - rms_dB; result = round(mean across channels).

Plus: LUFS/LRA/true peak via ffmpeg ebur128, clipping stats, before/after
plots (RMS envelope, crest factor, 1/3-octave spectrum).

Usage:
  measure.py metrics  FILES... [--json out.json]
  measure.py compare  BEFORE.flac AFTER.flac --outdir DIR [--label NAME]
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drlib.audio import decode, get_samplerate  # noqa: F401 (re-exported)

BLOCK_TIME = 3
CUT_BEST_BINS = 0.2
MAX_DYNAMIC = 24
AUDIO_MIN = 1e-10  # -200 dB


# ---------------------------------------------------------------- TT DR

def dr_rms(x: np.ndarray) -> np.ndarray:
    """Per-channel RMS with the x√2 (sine-reference) correction."""
    return np.sqrt(2.0 * np.sum(x.astype(np.float64) ** 2, 0) / x.shape[0])


def tt_dr(x: np.ndarray, fs: int):
    """TT DR per dr14_t.meter -> (dr:int, dr_exact:float, dr_lr:(L,R), peak_db, rms_db)."""
    delta_fs = 60 if fs == 44100 else 0
    block = BLOCK_TIME * (fs + delta_fs)
    n, ch = x.shape
    seg_cnt = int(math.floor(n / block) + 1)
    if seg_cnt < 1:
        return 0, 0.0, (0, 0), -100, -100

    rms = np.zeros((seg_cnt, ch))
    peaks = np.zeros((seg_cnt, ch))
    for i in range(seg_cnt - 1):
        seg = x[i * block:(i + 1) * block, :]
        rms[i] = dr_rms(seg)
        peaks[i] = np.max(np.abs(seg), 0)
    rest = x[(seg_cnt - 1) * block:, :]
    rms[-1] = dr_rms(rest)
    peaks[-1] = np.max(np.abs(rest), 0)

    rms.sort(0)
    peaks.sort(0)

    n_blk = max(int(math.floor(seg_cnt * CUT_BEST_BINS)), 1)
    top = rms[seg_cnt - n_blk:, :]
    rms_energy_mean = np.sqrt(np.sum(top ** 2, 0) / n_blk)

    ch_dr = -20.0 * np.log10(rms_energy_mean / peaks[seg_cnt - 2, :])
    ch_dr[np.logical_or(np.sum(top ** 2, 0) < AUDIO_MIN, np.abs(ch_dr) > MAX_DYNAMIC)] = 0.0

    dr = int(round(float(np.mean(ch_dr))))
    db_peak = 20.0 * math.log10(max(float(np.max(peaks)), 1e-10))
    db_rms = 20.0 * math.log10(max(float(np.mean(dr_rms(x))), 1e-10))
    return dr, float(np.mean(ch_dr)), (round(float(ch_dr[0]), 1), round(float(ch_dr[1]), 1)), round(db_peak, 1), round(db_rms, 1)


# ---------------------------------------------------------------- clipping

def clip_stats(x16: np.ndarray):
    """Clipping stats on integer decode: full-scale samples and run lengths."""
    out = {}
    for c, name in enumerate(("L", "R")):
        xi = np.abs(x16[:, c])
        clipped = xi >= 32767
        n_clipped = int(np.sum(clipped))
        runs = int(np.sum(clipped[1:] & ~clipped[:-1]) + (1 if clipped[0] else 0))
        # long runs (>=3 samples) are the signature of hard flat-top clipping
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


# ---------------------------------------------------------------- LUFS via ffmpeg

def ebur128(file):
    """Integrated LUFS, LRA, true peak via ffmpeg ebur128 (final Summary block)."""
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", str(file), "-filter_complex",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    txt = p.stderr.split("Summary:")[-1]
    i = re.search(r"I:\s*(-?[\d.]+|-inf)\s*LUFS", txt)
    lra = re.search(r"LRA:\s*(-?[\d.]+|-inf)\s*LU", txt)
    tp = re.search(r"Peak:\s*(-?[\d.]+|-inf)\s*dBFS", txt)
    f = lambda m: float(m.group(1)) if m and "inf" not in m.group(1) else -99.0
    return f(i), f(lra), f(tp)


# ---------------------------------------------------------------- graphs

def third_octave_spectrum(x: np.ndarray, fs: int):
    """Channel-averaged spectrum in 1/3-octave bands (dBFS per band)."""
    from scipy.signal import welch
    f, pxx = welch(x[:, 0].astype(np.float64), fs, nperseg=8192)
    _, pxy = welch(x[:, 1].astype(np.float64), fs, nperseg=8192)
    p = (pxx + pxy) / 2
    centers = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                        400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                        5000, 6300, 8000, 10000, 12500, 16000, 20000], dtype=float)
    lo = centers / 2 ** (1 / 6)
    hi = centers * 2 ** (1 / 6)
    band_p = np.array([p[(f >= l) & (f < h)].sum() for l, h in zip(lo, hi)])
    db = 10 * np.log10(np.maximum(band_p, 1e-20))
    return centers, db


def plot_compare(before_file, after_file, outdir: Path, label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xb, fs = decode(before_file)
    xa, _ = decode(after_file)
    nb = min(len(xb), len(xa))
    xb, xa = xb[:nb], xa[:nb]

    win = fs  # 1 s windows for envelopes
    nw = nb // win
    rms_b = np.sqrt(2 * np.mean(xb[:nw * win].reshape(nw, win, 2) ** 2, axis=(1, 2)))
    rms_a = np.sqrt(2 * np.mean(xa[:nw * win].reshape(nw, win, 2) ** 2, axis=(1, 2)))
    pk_b = np.abs(xb[:nw * win].reshape(nw, win, 2)).max(axis=(1, 2))
    pk_a = np.abs(xa[:nw * win].reshape(nw, win, 2)).max(axis=(1, 2))
    t = np.arange(nw)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    db = 20 * np.log10(np.maximum(rms_b, 1e-6))
    da = 20 * np.log10(np.maximum(rms_a, 1e-6))
    axes[0].plot(t, db, color="red", lw=0.8, label="original")
    axes[0].plot(t, da, color="green", lw=0.8, label="restored")
    axes[0].set_ylabel("RMS, dBFS")
    axes[0].set_title(f"{label}: loudness over time (1 s windows)")
    axes[0].legend(loc="lower left")

    cf_b = 20 * np.log10(np.maximum(pk_b, 1e-6)) - db
    cf_a = 20 * np.log10(np.maximum(pk_a, 1e-6)) - da
    axes[1].plot(t, cf_b, color="red", lw=0.8, label="original")
    axes[1].plot(t, cf_a, color="green", lw=0.8, label="restored")
    axes[1].set_ylabel("Crest factor, dB\n(peak - RMS per window)")
    axes[1].legend(loc="lower left")

    fc, sb = third_octave_spectrum(xb, fs)
    _, sa = third_octave_spectrum(xa, fs)
    axes[2].semilogx(fc, sb, color="red", marker="o", ms=3, lw=1, label="original")
    axes[2].semilogx(fc, sa, color="green", marker="o", ms=3, lw=1, label="restored")
    axes[2].set_ylabel("Band level, dB")
    axes[2].set_xlabel("Frequency, Hz")
    axes[2].set_title("Frequency response (averaged spectrum, 1/3 octave)")
    axes[2].legend()
    axes[2].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{label}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- entry

def metrics_for(file: str):
    xf, fs = decode(file, "f32le")
    x16, _ = decode(file, "s16le")
    dr, dr_exact, dr_lr, pk, rms = tt_dr(xf, fs)
    i, lra, tp = ebur128(file)
    return {
        "file": Path(file).name,
        "fs": fs,
        "dr": dr, "dr_exact": dr_exact, "dr_lr": dr_lr,
        "peak_dbfs": pk, "rms_dbfs": rms,
        "lufs": round(i, 1), "lra": round(lra, 1), "true_peak_dbfs": round(tp, 1),
        "clipping": clip_stats(x16),
    }


def cmd_metrics(args):
    rows = [metrics_for(f) for f in args.files]
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    w = max(len(r["file"]) for r in rows)
    print(f"{'file':<{w}}  DR  DRx  L     R     peak   RMS    LUFS  LRA   TP     clip(L/R samples)")
    for r in rows:
        c = r["clipping"]
        print(f"{r['file']:<{w}}  {r['dr']:>2}  {r['dr_exact']:>4.1f} {r['dr_lr'][0]:>5} {r['dr_lr'][1]:>5} "
              f"{r['peak_dbfs']:>6} {r['rms_dbfs']:>6} {r['lufs']:>5} {r['lra']:>5} {r['true_peak_dbfs']:>6} "
              f"{c['L']['clipped_samples']}/{c['R']['clipped_samples']}")
    drs = [r["dr"] for r in rows]
    print(f"\nAlbum DR (mean): {sum(drs) / len(drs):.1f} -> rounded DR{round(sum(drs) / len(drs))}")
    return rows


def cmd_compare(args):
    mb = metrics_for(args.before)
    ma = metrics_for(args.after)
    out = {"before": mb, "after": ma}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.label}_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    png = plot_compare(args.before, args.after, outdir, args.label)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"plot: {png}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("metrics")
    m.add_argument("files", nargs="+")
    m.add_argument("--json")
    m.set_defaults(func=cmd_metrics)
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--outdir", required=True)
    c.add_argument("--label", default="track")
    c.set_defaults(func=cmd_compare)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
