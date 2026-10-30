#!/usr/bin/env python3
"""Per-track restoration: parameter sweep on an excerpt, then full-track run.

For every track:
  1. sweep flatten k/cap and expand target on 30–90 s;
  2. score candidates against the master-like target Δcrest +5.5 dB,
     timbre shape error and over-expansion penalty;
  3. run the winner on the full track, write 24-bit FLAC;
  4. measure full-track result and the existing baseline for comparison.

Usage:
  python3 lab/per_track_restore.py --src SRC --out OUT --baseline BASE
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rnext import features as F                      # noqa: E402
from rnext.dsp.pipeline import Policy, restore_track  # noqa: E402
from rnext.dsp.contour import true_peak_dbfs        # noqa: E402
from restore_one import encode_flac                   # noqa: E402

TARGET_DCREST = 6.3
EXCERPT_START = 30.0
EXCERPT_DUR = 60.0


def shape(x, sr, base=None):
    b = np.array(F.band_rms_db(x, sr), dtype=np.float64)
    b -= b.mean()
    if base is None:
        return b
    d = np.abs(b - base)
    return round(float(d.max()), 2), round(float(np.sqrt((d ** 2).mean())), 2)


def short_crest(x, sr, win_s=0.05, hop_s=0.025):
    win = max(2, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    n = x.shape[0]
    vals = []
    for s in range(0, max(n - win, 1), hop):
        seg = x[s:s + win]
        p = float(np.max(np.abs(seg)))
        r = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        if r > 1e-6:
            vals.append(p / r)
    return round(float(np.percentile(vals, 95)), 3) if vals else 0.0


def metrics(x, sr, base=None):
    dr, drx, _ch, peak, rms = F.tt_dr(x, sr)
    crest_db = round(float(F.crest_db(x)), 2)
    tp = round(float(true_peak_dbfs(x, sr)), 2)
    shape_max, shape_rms = shape(x, sr, base) if base is not None else (None, None)
    return {
        "dr": dr, "dr_exact": round(float(drx), 2),
        "crest_tt": round(float(peak - rms), 2),
        "crest_db": crest_db, "tp": tp,
        "peak_db": round(float(peak), 2), "rms_db": round(float(rms), 2),
        "shape_max": shape_max, "shape_rms": shape_rms,
        "short_crest_p95": short_crest(x, sr),
        "clip": int(np.sum(np.abs(x) >= 0.9999)),
    }


def policy_of(kind, param):
    if kind == "flatten":
        k, cap = param
        return Policy(auto_policy=False, mode="flatten", adaptive=False,
                      k=k, gain_min_db=-cap, gain_max_db=cap,
                      target_tp_dbfs=-0.5)
    return Policy(auto_policy=True, mode="expand", adaptive=True,
                  target_dcrest=float(param), target_tp_dbfs=-0.5)


def candidates():
    out = [("flatten", (0.5, c)) for c in (4, 6, 8, 10)]
    out.append(("flatten", (0.3, 8)))
    out += [("expand", 6.0), ("expand", 7.0)]
    return out


def run_full(x, sr, kind, param):
    pol = policy_of(kind, param)
    y, st = restore_track(x, sr, pol, keep_arrays=False, return_stages=True)
    info = (st.get("adaptive") or {}).get("r_fit")
    return y, info


def write_24(y, sr, path, ti):
    rng = np.random.default_rng(0xC0FFEE + ti)
    dith = rng.random(y.shape) + rng.random(y.shape) - 1.0
    yq = np.clip(np.round(y * 8388608.0 + dith), -8388608, 8388607).astype(np.int32)
    encode_flac(yq, sr, path, path.parent / f"__tmp{ti}.wav", bits=24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="")
    ap.add_argument("--only", default="", help="comma-separated track numbers")
    args = ap.parse_args()
    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_dir = pathlib.Path(args.baseline) if args.baseline else None
    only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None
    files = sorted(src.glob("*.flac"))
    if only:
        files = [f for f in files if int(f.name[:2]) in only]

    rows, cand_rows = [], []
    for i, f in enumerate(files, 1):
        ti = int(f.name[:2])
        title = f.stem[3:].strip() if f.stem[:2].isdigit() else f.stem
        t0 = time.time()
        x, sr = F.decode_f32(str(f))
        base = shape(x, sr)
        orig = metrics(x, sr, base)
        ex, _ = F.decode_f32(str(f), start_s=EXCERPT_START, dur_s=EXCERPT_DUR)
        eb = shape(ex, sr)
        eo = metrics(ex, sr, eb)
        print(f"\n[{ti:02d}] {title}  orig DR {orig['dr']} crest {orig['crest_db']} "
              f"({time.time()-t0:.1f}s decode)", flush=True)

        scored = []
        for kind, param in candidates():
            try:
                ye, _ = run_full(ex, sr, kind, param)
                me = metrics(ye, sr, eb)
                dcrest = round(me["crest_db"] - eo["crest_db"], 2)
                ddr = me["dr"] - eo["dr"]
                name = (f"flatten k={param[0]} cap={param[1]}"
                        if kind == "flatten" else f"expand target={param:g}")
                score = (abs(dcrest - TARGET_DCREST)
                         + 0.8 * (me["shape_rms"] or 0)
                         + (0.5 if kind == "expand" else 0.0)
                         + max(0.0, 5.0 - dcrest) * 2.0
                         + max(0.0, dcrest - 8.0) * 1.5
                         + max(0, me["dr"] - 13) * 0.5)
                rec = {"name": name, "kind": kind, "param": param,
                       "dr": me["dr"], "ddr": ddr, "crest": me["crest_db"],
                       "dcrest": dcrest, "shape_max": me["shape_max"],
                       "shape_rms": me["shape_rms"], "tp": me["tp"],
                       "score": round(float(score), 3)}
                scored.append(rec)
                print(f"    {name:22} DR {me['dr']:2} (+{ddr}) crest {me['crest_db']:6.2f} "
                      f"({dcrest:+.2f}) shape {me['shape_rms']:.2f} score {score:.2f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    {kind} {param}: FAILED {type(e).__name__}: {e}", flush=True)
        if not scored:
            print("  !! no candidate worked", flush=True)
            continue
        best = min(scored, key=lambda r: r["score"])
        print(f"  -> choose {best['name']} (score {best['score']})", flush=True)

        # true contour gain step on the excerpt for the chosen policy
        gain_step = None
        gdiff = None
        try:
            yg, stg = restore_track(ex, sr, policy_of(best["kind"], best["param"]),
                                     keep_arrays=True, return_stages=True)
            g = stg.get("contour_gains_db")
            if g is not None:
                gain_step = round(float(np.abs(np.diff(g, axis=-1)).max()), 2)
                gdiff = round(float(np.abs(g[:, 0, :] - g[:, 1, :]).max()), 2)
        except Exception as e:  # noqa: BLE001
            print(f"  gain check failed: {e}", flush=True)

        # full track
        tf = time.time()
        y, rfit = run_full(x, sr, best["kind"], best["param"])
        fm = metrics(y, sr, base)
        fpath = out / f"{ti:02d}. {title}.flac"
        write_24(y, sr, fpath, ti)
        full_dcrest = round(fm["crest_db"] - orig["crest_db"], 2)
        row = {"track": ti, "title": title, "chosen": best["name"],
               "kind": best["kind"], "param": best["param"],
               "r_fit": rfit, "gain_step": gain_step, "gL_gR_max": gdiff,
               "orig": orig, "rest": fm, "delta": {
                   "dr": fm["dr"] - orig["dr"], "crest": full_dcrest,
                   "short_crest": round(fm["short_crest_p95"] - orig["short_crest_p95"], 2)},
               "candidates": scored, "full_seconds": round(time.time() - tf, 1)}
        rows.append(row)
        print(f"  FULL DR {orig['dr']}->{fm['dr']} crest {orig['crest_db']}->{fm['crest_db']} "
              f"(+{full_dcrest:+.2f}) shape {fm['shape_rms']} tp {fm['tp']} "
              f"gain {gain_step} ({time.time()-tf:.0f}s)", flush=True)

        if base_dir:
            bf = base_dir / f"{ti:02d}. {title}.flac"
            if bf.exists():
                xb, srb = F.decode_f32(str(bf))
                bm = metrics(xb, srb, base)
                row["baseline"] = {"dr": bm["dr"], "crest": bm["crest_db"],
                                   "crest_tt": bm["crest_tt"], "tp": bm["tp"],
                                   "shape_rms": bm["shape_rms"], "rms_db": bm["rms_db"]}
                row["better"] = {
                    "dr": fm["dr"] - bm["dr"],
                    "crest": round(fm["crest_db"] - bm["crest_db"], 2),
                    "shape_rms": round((fm["shape_rms"] or 0) - (bm["shape_rms"] or 0), 2),
                }
        cand_rows.append(row)
        (out / "per_track_report.json").write_text(
            json.dumps(cand_rows, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n== done ==", flush=True)
    for r in cand_rows:
        b = r.get("better") or {}
        print(f"{r['track']:02} {r['title'][:28]:30} {r['chosen']:22} "
              f"DR {r['orig']['dr']}->{r['rest']['dr']} crest +{r['delta']['crest']:+.2f} "
              f"vs base {b.get('dr', '?')}/{b.get('crest', '?')}/{b.get('shape_rms', '?')}")


if __name__ == "__main__":
    main()
