#!/usr/bin/env python3
"""Final judge: pick the best full-track candidate per track with the correct
objective (Δcrest over ORIGINAL, DR<=12, timbre, no regression vs baseline),
then rewrite only the files whose current output is not the winner.
"""
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rnext import features as F
from rnext.dsp.pipeline import Policy, restore_track
from rnext.dsp.contour import true_peak_dbfs
from restore_one import encode_flac

OUT = pathlib.Path("/media/mydum/Data/Music/restore/release/"
                   "Nickelback - Everything Under the Sun (2026) [restored 24-96 per-track]")
SRC = pathlib.Path.home() / "Music/Nickelback - Everything Under the Sun (2026) [WEB 24-96]"
TARGET = 6.3


def policy(kind, param):
    if kind == "flatten":
        k, cap = param
        return Policy(auto_policy=False, mode="flatten", adaptive=False,
                      k=k, gain_min_db=-cap, gain_max_db=cap, target_tp_dbfs=-0.5)
    return Policy(auto_policy=True, mode="expand", adaptive=True,
                  target_dcrest=float(param), target_tp_dbfs=-0.5)


def shape(x, sr, base):
    b = np.array(F.band_rms_db(x, sr), dtype=np.float64)
    b -= b.mean()
    d = np.abs(b - base)
    return round(float(d.max()), 2), round(float(np.sqrt((d ** 2).mean())), 2)


def met(x, sr, base):
    dr, drx, _ch, pk, rms = F.tt_dr(x, sr)
    sm, srms = shape(x, sr, base)
    return {"dr": dr, "dr_exact": round(float(drx), 2),
            "crest_db": round(float(F.crest_db(x)), 2),
            "crest_tt": round(float(pk - rms), 2),
            "tp": round(float(true_peak_dbfs(x, sr)), 2),
            "peak_db": round(float(pk), 2), "rms_db": round(float(rms), 2),
            "shape_max": sm, "shape_rms": srms}


def score(m, orig, base):
    d = m["crest_db"] - orig["crest_db"]
    s = (abs(d - TARGET) + 0.8 * m["shape_rms"]
         + max(0.0, 5.0 - d) * 2.0 + max(0.0, d - 7.5) * 2.0
         + max(0, m["dr"] - 12) * 0.5
         + max(0, base["dr"] - m["dr"]) * 0.5
         + max(0.0, base["crest"] - m["crest_db"]) * 0.5)
    return round(float(s), 3)


def run_full(f, x, sr, base, kind, param):
    y, _st = restore_track(x, sr, policy(kind, param),
                           keep_arrays=False, return_stages=True)
    return y, met(y, sr, base)


def write_24(y, sr, path, ti):
    rng = np.random.default_rng(0xC0FFEE + ti)
    dith = rng.random(y.shape) + rng.random(y.shape) - 1.0
    yq = np.clip(np.round(y * 8388608.0 + dith), -8388608, 8388607).astype(np.int32)
    encode_flac(yq, sr, path, path.parent / f"__tmp{ti}.wav", bits=24)


def gain_step_for(f, sr, kind, param):
    ex, _ = F.decode_f32(str(f), start_s=30, dur_s=60)
    _y, st = restore_track(ex, sr, policy(kind, param), keep_arrays=True, return_stages=True)
    g = st.get("contour_gains_db")
    if g is None:
        return None, None
    return (round(float(np.abs(np.diff(g, axis=-1)).max()), 2),
            round(float(np.abs(g[:, 0, :] - g[:, 1, :]).max()), 2))


# (track, candidates to test now)
EXTRA = {
    4: [("flatten", (0.5, 8))],
    5: [("flatten", (0.5, 8))],
    6: [("flatten", (0.3, 8))],
    7: [("flatten", (0.3, 8))],
    9: [("flatten", (0.5, 10)), ("flatten", (0.3, 10)), ("expand", 7.0)],
}


def main():
    rows = json.loads((OUT / "per_track_report.json").read_text())
    by = {r["track"]: r for r in rows}
    for ti in sorted(EXTRA):
        r = by[ti]
        title = r["title"]
        f = SRC / f"{ti:02d} {title}.flac"
        if not f.exists():
            f = sorted(SRC.glob(f"{ti:02d} *.flac"))[0]
        x, sr = F.decode_f32(str(f))
        base = np.array(F.band_rms_db(x, sr), dtype=np.float64)
        base -= base.mean()
        orig = met(x, sr, base)
        b = r["baseline"]
        cur = r["rest"]
        cur_score = score(cur, orig, b)
        print(f"\n[{ti:02d}] {title}  current {r['chosen']} DR {cur['dr']} "
              f"crest {cur['crest_db']} d={cur['crest_db']-orig['crest_db']:+.2f} "
              f"shape {cur['shape_rms']} score {cur_score}", flush=True)
        best = (cur_score, r["chosen"], r["kind"], r["param"], None, None)
        for kind, param in EXTRA[ti]:
            t0 = time.time()
            y, m = run_full(f, x, sr, base, kind, param)
            sc = score(m, orig, b)
            name = (f"flatten k={param[0]} cap={param[1]}" if kind == "flatten"
                    else f"expand target={param:g}")
            print(f"   {name:22} DR {m['dr']} crest {m['crest_db']} "
                  f"d={m['crest_db']-orig['crest_db']:+.2f} shape {m['shape_rms']} "
                  f"score {sc} ({time.time()-t0:.0f}s)", flush=True)
            r.setdefault("final_candidates", []).append(
                {"name": name, "kind": kind, "param": param, "dr": m["dr"],
                 "crest": m["crest_db"], "dcrest": round(m["crest_db"] - orig["crest_db"], 2),
                 "shape_max": m["shape_max"], "shape_rms": m["shape_rms"],
                 "tp": m["tp"], "score": sc})
            if sc < best[0]:
                best = (sc, name, kind, param, y, m)
        if best[4] is None:
            print("   -> keep current", flush=True)
            continue
        y, m = best[4], best[5]
        r["chosen"], r["kind"], r["param"] = best[1], best[2], best[3]
        r["rest"] = m
        r["delta"] = {"dr": m["dr"] - orig["dr"],
                      "crest": round(m["crest_db"] - orig["crest_db"], 2),
                      "short_crest": r["delta"].get("short_crest")}
        r["better"] = {"dr": m["dr"] - b["dr"],
                       "crest": round(m["crest_db"] - b["crest"], 2),
                       "shape_rms": round((m["shape_rms"] or 0) - (b.get("shape_rms") or 0), 2)}
        gs, gd = gain_step_for(f, sr, best[2], best[3])
        r["gain_step"], r["gL_gR_max"] = gs, gd
        write_24(y, sr, OUT / f"{ti:02d}. {title}.flac", ti)
        print(f"   -> REPLACED {best[1]} score {best[0]} (was {cur_score})", flush=True)
        (OUT / "per_track_report.json").write_text(
            json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nfinal judge done", flush=True)


if __name__ == "__main__":
    main()
