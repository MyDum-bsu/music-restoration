#!/usr/bin/env python3
"""Residual test without ears: ours minus original after removing the common gain.

Idea: our output = g·(original + nonlinear residual) + dither. If we remove the
common gain g = rms(orig)/rms(ours) and subtract, what remains is:
  - linear processing residue (HPF filter, contour);
  - nonlinearity products (if any) — narrow peaks/harmonics.

Metrics:
  * residual level relative to the signal (dB);
  * residual spectrum (Welch): looking for NARROW peaks — bins whose power
    exceeds the local median (in a ±W-bin window) by > P dB, with a "narrowness" (Q) estimate;
  * separately check whether the significant peaks coincide with harmonics of
    50/60 Hz (mains hum — a symptom, not a defect) and with harmonics of the
    signal's dominant tone.

  python3 scripts/residual_test.py --ours <dir> --orig <image> --cue <cue> --tracks 1,7
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rnext import features as F  # noqa: E402

NFFT = 16384
PROM_DB = 10.0     # a peak this many dB above the local median = "narrow peak"
WIN_BINS = 16      # local-median window


def narrow_peaks(e: np.ndarray, sr: int) -> dict:
    f, P = signal.welch(e, fs=sr, nperseg=NFFT, noverlap=NFFT // 2, axis=0)
    P = P[:, 0] + P[:, 1] if P.ndim > 1 else P
    Pdb = 10 * np.log10(P + 1e-30)
    k = WIN_BINS
    pad = np.pad(Pdb, (k, k), mode="edge")
    med = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(Pdb))])
    prom = Pdb - med
    # exclude the very bottom (<30 Hz: DC/HPF tail) and the very top
    band = (f > 30) & (f < sr / 2 - 500)
    idx = np.where(band)[0]
    order = idx[np.argsort(prom[idx])[::-1]]
    peaks = []
    used = set()
    for i in order:
        if prom[i] < PROM_DB:
            break
        if any(abs(i - j) <= 2 for j in used):
            continue
        used.add(i)
        peaks.append({"f_hz": round(float(f[i]), 1), "prom_db": round(float(prom[i]), 1)})
        if len(peaks) >= 12:
            break
    return {"psd_peaks": peaks, "max_prom_db": round(float(prom[band].max()), 2),
            "n_prom_gt10": int((prom[band] > PROM_DB).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--cue", default="")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = Path(args.orig)
    ours_dir = Path(args.ours)
    if src.is_file() and args.cue:
        from rnext.cue import parse_cue, track_bounds
        meta = F.ffprobe_audio(str(src))
        total = int(round(meta["dur"] * meta["sr"]))
        cs = parse_cue(args.cue, file_key=src.name)
        bounds = track_bounds(cs, total, meta["sr"])
        def grab(ti, dur=None):
            st, en = bounds[ti - 1]
            return F.decode_f32(str(src), start_s=st / meta["sr"], dur_s=(en - st) / meta["sr"])
    else:
        import os
        fs = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".flac"))
        def grab(ti, dur=None):
            return F.decode_f32(fs[ti - 1])

    out = {}
    for ti in [int(t) for t in args.tracks.split(",")]:
        o_f = next(ours_dir.glob(f"{ti:02d}*.flac"), None)
        x, sr = grab(ti)
        y, _ = F.decode_f32(str(o_f))
        n = min(x.shape[0], y.shape[0])
        x, y = x[:n], y[:n]
        g = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) /
                  np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        yg = (y.astype(np.float64) * g)
        e = yg - x.astype(np.float64)
        lvl_sig = 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-30)
        lvl_res = 10 * np.log10(np.mean(e ** 2) + 1e-30)
        npk = narrow_peaks(e, sr)
        harms = [p for p in npk["psd_peaks"]
                 if any(abs(p["f_hz"] - h * m) < 3.0
                        for h in (50, 60) for m in range(2, 12))]
        rec = {"track": ti, "gain_db": round(20 * np.log10(g), 2),
               "residual_db_rel": round(lvl_res - lvl_sig, 1),
               "max_prom_db": npk["max_prom_db"], "n_prom_gt10": npk["n_prom_gt10"],
               "peaks": npk["psd_peaks"], "peaks_at_mains_harmonics": harms}
        out[ti] = rec
        print(json.dumps(rec, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
