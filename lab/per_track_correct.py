#!/usr/bin/env python3
"""Second pass: correct per-track outliers on the FULL track.

For chosen tracks, run extra full-track candidates, score them against
Δcrest +6.3 dB / DR ≤ 12 / timbre error / no regression vs baseline,
and replace the output file only if the score improves.
"""
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

OUT = pathlib.Path("/media/mydum/Data/Music/restore/release/"
                   "Nickelback - Everything Under the Sun (2026) [restored 24-96 per-track]")
SRC = pathlib.Path.home() / "Music/Nickelback - Everything Under the Sun (2026) [WEB 24-96]"
TARGET = 6.3

SPECS = {
    3: [("flatten", (0.5, 8)), ("flatten", (0.5, 6))],
    4: [("flatten", (0.5, 8)), ("flatten", (0.5, 6))],
    5: [("expand", 6.0)],
    6: [("flatten", (0.5, 8)), ("flatten", (0.3, 8)), ("flatten", (0.5, 10))],
    7: [("flatten", (0.3, 8)), ("expand", 6.0)],
    9: [("flatten", (0.5, 8)), ("expand", 6.0)],
}


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


def score(m, base):
    d = m["crest_db"] - base["crest"]
    s = (abs(d - TARGET) + 0.8 * m["shape_rms"]
         + max(0.0, 5.0 - d) * 2.0 + max(0.0, d - 7.5) * 2.0
         + max(0, m["dr"] - 12) * 0.5
         + max(0, base["dr"] - m["dr"]) * 0.5
         + max(0.0, base["crest"] - m["crest_db"]) * 0.5)
    return round(float(s), 3), round(float(d), 2)


def main():
    rows = json.loads((OUT / "per_track_report.json").read_text())
    by = {r["track"]: r for r in rows}
    for ti, specs in SPECS.items():
        r = by[ti]
        title = r["title"]
        f = SRC / f"{ti:02d} {title}.flac"
        if not f.exists():
            cands = sorted(SRC.glob(f"{ti:02d} *.flac"))
            if not cands:
                print(f"[{ti:02d}] SOURCE MISSING", flush=True)
                continue
            f = cands[0]
        x, sr = F.decode_f32(str(f))
        base = np.array(F.band_rms_db(x, sr), dtype=np.float64)
        base -= base.mean()
        orig = met(x, sr, base)
        b = r["baseline"]
        curm = r["rest"]
        curscore, curd = score(curm, b)
        print(f"\n[{ti:02d}] {title}  current {r['chosen']} DR {curm['dr']} "
              f"crest {curm['crest_db']} ({curd:+.2f}) shape {curm['shape_rms']} "
              f"score {curscore}", flush=True)
        best = (curscore, None, None, None)
        best_y = None
        for kind, param in specs:
            t0 = time.time()
            y, _st = restore_track(x, sr, policy(kind, param),
                                   keep_arrays=False, return_stages=True)
            m = met(y, sr, base)
            sc, dc = score(m, b)
            name = (f"flatten k={param[0]} cap={param[1]}" if kind == "flatten"
                    else f"expand target={param:g}")
            print(f"   {name:22} DR {m['dr']} crest {m['crest_db']} ({dc:+.2f}) "
                  f"shape {m['shape_rms']} score {sc} ({time.time()-t0:.0f}s)", flush=True)
            r.setdefault("corrections", []).append(
                {"name": name, "kind": kind, "param": param, "dr": m["dr"],
                 "crest": m["crest_db"], "dcrest": dc,
                 "shape_max": m["shape_max"], "shape_rms": m["shape_rms"],
                 "tp": m["tp"], "score": sc})
            if sc < best[0]:
                best = (sc, name, kind, param)
                best_y = (y, m)
        if best_y is None:
            print("   -> keep current (no better candidate)", flush=True)
            continue
        y, m = best_y
        r["chosen"] = best[1]
        r["kind"] = best[2]
        r["param"] = best[3]
        r["rest"] = m
        r["delta"] = {"dr": m["dr"] - orig["dr"],
                      "crest": round(m["crest_db"] - orig["crest_db"], 2),
                      "short_crest": r["delta"].get("short_crest")}
        r["better"] = {"dr": m["dr"] - b["dr"],
                       "crest": round(m["crest_db"] - b["crest"], 2),
                       "shape_rms": round((m["shape_rms"] or 0) - (b.get("shape_rms") or 0), 2)}
        ex, _ = F.decode_f32(str(f), start_s=30, dur_s=60)
        yg, stg = restore_track(ex, sr, policy(best[2], best[3]),
                                keep_arrays=True, return_stages=True)
        g = stg.get("contour_gains_db")
        r["gain_step"] = round(float(np.abs(np.diff(g, axis=-1)).max()), 2) if g is not None else None
        r["gL_gR_max"] = round(float(np.abs(g[:, 0, :] - g[:, 1, :]).max()), 2) if g is not None else None
        rng = np.random.default_rng(0xC0FFEE + ti)
        dith = rng.random(y.shape) + rng.random(y.shape) - 1.0
        yq = np.clip(np.round(y * 8388608.0 + dith), -8388608, 8388607).astype(np.int32)
        path = OUT / f"{ti:02d}. {title}.flac"
        encode_flac(yq, sr, path, OUT / f"__tmp{ti}.wav", bits=24)
        print(f"   -> REPLACED with {best[1]} score {best[0]} (was {curscore}) "
              f"DR {m['dr']} crest {m['crest_db']} shape {m['shape_rms']}", flush=True)
        (OUT / "per_track_report.json").write_text(
            json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\ncorrected", flush=True)


if __name__ == "__main__":
    main()
