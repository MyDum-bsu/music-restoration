#!/usr/bin/env python3
"""Calibrating the contour on corpus pairs.

Our pipeline (mode=expand / flatten) is run on the track's original and
compared against master-restoration by full-track TT DR/crest/rms/peak and
band-RMS deltas (with the common RMS offset aligned, i.e. comparing shapes,
not absolute levels).

Requires corpus/feature_pairs/{key}.jsonl (for match info) + the same path
triples as pair_features.py.

Output: corpus/calibration/{key}__{tag}.jsonl + a summary on stdout.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext.cue import parse_cue, track_bounds  # noqa: E402
from rnext import features as F  # noqa: E402
from rnext import workspace  # noqa: E402
from rnext.dsp.pipeline import Policy, restore_track  # noqa: E402


def matched_rows(key: str) -> list[dict]:
    p = workspace.dir_of("corpus") / "feature_pairs" / f"{key}.jsonl"
    out = []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if "original" in r and "restored" in r and r.get("samples"):
            out.append(r)
    return out


def analyze(x: np.ndarray, sr: int) -> dict:
    st = F.tt_dr(x, sr)
    return {"dr": st[0], "dr_exact": round(float(st[1]), 2),
            "peak_dbfs": round(float(st[3]), 2), "rms_db": round(float(st[4]), 2)}


def crest(o: dict) -> float:
    return round(o["peak_dbfs"] - o["rms_db"], 2)


def band_delta_norm(ours: np.ndarray, master: np.ndarray, sr: int) -> list[float]:
    """Band delta ours-master after aligning the common RMS (shape, not level)."""
    n = min(ours.shape[0], master.shape[0])
    oy = ours[:n].astype(np.float64)
    om = master[:n].astype(np.float64)
    ry = 20.0 * np.log10(np.sqrt(np.mean(oy ** 2)) + 1e-16)
    rm = 20.0 * np.log10(np.sqrt(np.mean(om ** 2)) + 1e-16)
    shift_db = rm - ry
    b_ours = F.band_rms_db(ours[:n], sr)
    b_m = F.band_rms_db(master[:n], sr)
    return [round(b - a + shift_db, 2) for a, b in zip(b_ours, b_m)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--restored", required=True)
    ap.add_argument("--orig-image", required=True)
    ap.add_argument("--cue", required=True)
    ap.add_argument("--r-list", default="1.8,2.0,2.2")
    ap.add_argument("--mode", default="expand")
    ap.add_argument("--k", type=float, default=0.75, help="mode=flatten")
    ap.add_argument("--tracks", default="", help="comma-separated track numbers; empty = all")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    out_dir = workspace.dir_of("corpus") / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {int(t) for t in args.tracks.split(",") if t.strip()} if args.tracks else None

    img = Path(args.orig_image)
    meta = F.ffprobe_audio(str(img))
    total = int(round(meta["dur"] * meta["sr"]))
    tracks = parse_cue(args.cue, file_key=img.name)
    bounds = track_bounds(tracks, total, meta["sr"])

    for r_val in [float(x) for x in args.r_list.split(",")]:
        pol = Policy(mode=args.mode, r_expand=r_val, k=args.k, target_tp_dbfs=-0.5)
        tag = args.out_tag or f"{args.mode}-r{r_val}"
        out_path = out_dir / f"{args.key}__{tag}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in matched_rows(args.key):
                t = row["track_n"]
                if want and t not in want:
                    continue
                st, en = bounds[t - 1]
                sr = row["sr"]
                x, sr = F.decode_f32(str(img), start_s=st / sr, dur_s=(en - st) / sr)
                if x.shape[0] < sr * 5:
                    continue
                y = restore_track(x, sr, pol)
                m, _ = F.decode_f32(str(Path(args.restored) / row["restored_file"]))
                n = min(y.shape[0], m.shape[0])
                o_an = analyze(x, sr)
                y_an = analyze(y[:n], sr)
                m_an = analyze(m[:n], sr)
                rec = {
                    "key": args.key, "track_n": t, "title": row["cue_title"],
                    "mode": args.mode, "r": r_val,
                    "our": y_an, "our_crest": crest(y_an),
                    "master": m_an, "master_crest": crest(m_an),
                    "orig": o_an, "orig_crest": crest(o_an),
                    "dr_err": y_an["dr_exact"] - m_an["dr_exact"],
                    "crest_err": round(crest(y_an) - crest(m_an), 2),
                    "band_delta_norm": band_delta_norm(y[:n], m[:n], sr),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f'[r{r_val}] t{t:>2} DR our{y_an["dr"]:>3}/master{m_an["dr"]:>3} '
                      f'crest {crest(y_an):5.2f}/{crest(m_an):5.2f} '
                      f'dr_err {rec["dr_err"]:+.2f}', flush=True)

        recs = [json.loads(l) for l in out_path.read_text().splitlines()]
        if recs:
            dr_err = float(np.mean([r["dr_err"] for r in recs]))
            cr_err = float(np.mean([r["crest_err"] for r in recs]))
            bd = np.array([r["band_delta_norm"] for r in recs])
            print(f"[SUM {args.key} {tag}] mean dr_err {dr_err:+.2f} "
                  f"crest_err {cr_err:+.2f} bandΔ mean {np.round(bd.mean(0), 1).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
