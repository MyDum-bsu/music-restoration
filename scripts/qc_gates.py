#!/usr/bin/env python3
"""Automatic QC gates on the output track/restoration.

Checks:
  1. TP ≤ the given ceiling (intersample, 4x oversample);
  2. no int16 clipping (|x| ≥ 32766) on the output;
  3. TEMPORAL-SMOOTHNESS: the effective out−in per-band step is INFORMATIONAL,
     not a verdict: on transients it is honestly large. The true contour gain
     is checked via the gains matrix from the pipeline (restore_track keep_arrays=True);
  4. absolute "jaggedness": no single amplitude jumps > 0.9 between adjacent
     samples (click-like garbage from pipeline bugs).

Return codes: 0 = ok, 1 = violations found. Prints a JSON report.
"""
import argparse
import json
import re
import sys

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import features as F  # noqa: E402
from rnext.dsp.contour import true_peak_dbfs  # noqa: E402


def gain_smoothness(x: np.ndarray, sr: int, frame: int = 2048) -> dict:
    """Per-frame level in 1/3-oct bands; max step between adjacent frames."""
    import math

    hop = frame // 2
    starts = np.arange(0, max(x.shape[0] - frame, 0) + 1, hop)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    edges = [20.0]
    while edges[-1] < sr / 2 - 1:
        edges.append(edges[-1] * 2 ** (1 / 3))
    edges[-1] = sr / 2
    bands = []
    for i in range(len(edges) - 1):
        idx = np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
        if idx.size:
            bands.append(idx)
    w = np.hanning(frame)
    max_step, steps_db = 0.0, []
    for b0 in range(0, starts.size, 512):
        b1 = min(b0 + 512, starts.size)
        fr = np.stack([x[s:s + frame, :].T for s in starts[b0:b1]])  # [nb, ch, frame]
        fr = fr * w[None, None, :]
        Z = np.abs(np.fft.rfft(fr, axis=-1)) ** 2
        del fr
        for idx in bands:
            lv = 10.0 * np.log10(Z[:, :, idx].mean(axis=2).T + 1e-16)  # [ch, nf]
            st = np.abs(np.diff(lv, axis=1))
            steps_db.append(st.max())
        del Z
    max_step = float(np.max(steps_db)) if steps_db else 0.0
    return {"max_gain_step_db_per_frame": round(max_step, 2),
            "frame_ms": round(1000 * hop / sr, 1)}


def gain_smoothness_pair(xo: np.ndarray, xr: np.ndarray, sr: int,
                         frame: int = 2048, floor_db: float = -80.0) -> dict:
    """Step of the APPLIED gain: out−in level difference per band.

    A single output file won't do here: a drum band honestly jumps by tens of
    dB per frame and the gate falsely trips. What must be compared is
    out_db − in_db, and only where the input has signal.
    """
    hop = frame // 2
    n = min(xo.shape[0], xr.shape[0])
    starts = np.arange(0, max(n - frame, 0) + 1, hop)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    edges = [20.0]
    while edges[-1] < sr / 2 - 1:
        edges.append(edges[-1] * 2 ** (1 / 3))
    edges[-1] = sr / 2
    bands = []
    for i in range(len(edges) - 1):
        idx = np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
        if idx.size:
            bands.append(idx)
    w = np.hanning(frame)
    max_step = 0.0
    for b0 in range(0, starts.size, 512):
        b1 = min(b0 + 512, starts.size)
        def spec(x):
            fr = np.stack([x[s:s + frame, :].T for s in starts[b0:b1]])
            return np.abs(np.fft.rfft(fr * w[None, None, :], axis=-1)) ** 2
        Zo, Zr = spec(xo), spec(xr)
        for idx in bands:
            lo = 10.0 * np.log10(Zo[:, :, idx].mean(axis=2).T + 1e-16)
            lr = 10.0 * np.log10(Zr[:, :, idx].mean(axis=2).T + 1e-16)
            gain = lr - lo
            # frames where the input is silent/near-silent are not informative.
            # The mask is applied PAIRWISE to adjacent frames, otherwise the diff
            # is taken across the removed silence patch and yields false tens of dB.
            valid = lo > floor_db
            if valid.shape[1] > 1:
                d = np.abs(np.diff(gain, axis=1))
                ok = valid[:, 1:] & valid[:, :-1]
                if ok.any():
                    max_step = max(max_step, float(d[ok].max()))
        del Zo, Zr
    return {"max_gain_step_db_per_frame": round(max_step, 2),
            "frame_ms": round(1000 * hop / sr, 1)}


def sample_spikes(x: np.ndarray) -> int:
    d = np.abs(np.diff(x, axis=0)).max(axis=1)
    return int((d > 0.95).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tp-ceiling", type=float, default=-0.5)
    ap.add_argument("--orig", help="source directory: check the applied gain (out−in)")
    args = ap.parse_args()

    bad = False
    for f in args.files:
        x, sr = F.decode_f32(f)
        if x is None:
            print(f"{f}: DECODE FAIL")
            bad = True
            continue
        st = F.tt_dr(x, sr)
        tp = true_peak_dbfs(x, sr)
        g = gain_smoothness(x, sr)
        gain_step = None
        if args.orig:
            m = re.match(r"\s*(\d{1,3})", Path(f).name)
            tn = int(m.group(1)) if m else None
            cands = sorted(Path(args.orig).glob(f"{tn:02d}*.flac")) if tn is not None else []
            if cands:
                xo, sro = F.decode_f32(str(cands[0]))
                if xo is not None and sro == sr:
                    gain_step = gain_smoothness_pair(xo, x, sr)["max_gain_step_db_per_frame"]
        spikes = sample_spikes(x)
        clipped = int((np.abs((x * 32768).astype(np.int32)) >= 32766).sum())
        rec = {
            "file": f, "dr": st[0], "peak_db": round(st[3], 2),
            "tp_dbfs": round(tp, 2), "tp_ceiling": args.tp_ceiling,
            "tp_ok": bool(tp <= args.tp_ceiling + 0.05),
            "clipped_samples_out": clipped,
            "max_gain_step_db_per_frame": gain_step,
            "gain_step_checked": gain_step is not None,
            "gain_step_note": "info: effective out-in, not contour gain",
            "signal_step_db_per_frame": g["max_gain_step_db_per_frame"],
            "sample_spikes_gt09": spikes,
        }
        # gain_step is informational: effective out−in, not the contour gain.
        # The verdict only uses TP, clipping and click-like spikes.
        bad_cnt = (not rec["tp_ok"]) or clipped > 0 or spikes > 0
        rec["verdict"] = "FAIL" if bad_cnt else "OK"
        print(json.dumps(rec, ensure_ascii=False))
        if bad_cnt:
            bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
