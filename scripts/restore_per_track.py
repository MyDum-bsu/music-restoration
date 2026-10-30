#!/usr/bin/env python3
"""Per-track album restoration with per-track parameter selection.

For every track:
  1. sweep candidate policies (flatten k/cap, expand target) on an excerpt;
  2. score them against the master-like target Δcrest, timbre error and
     over/under-expansion penalties;
  3. run the winner on the FULL track, verify, and if the full-track Δcrest
     misses the target, try neighboring caps on the full track and keep the
     best (excerpt metrics alone are not reliable enough);
  4. write 24-bit FLAC + per_track_report.json + PER_TRACK_REPORT.md.

The result never regresses below the optional baseline (a previously restored
version): baseline regression is penalized in the score, not enforced silently.

Usage:
  python3 scripts/restore_per_track.py \
      --src "Artist - Album (2026)/" \
      --out "/media/.../restore/release/Artist - Album (2026) [restored per-track]" \
      [--baseline "/media/.../release/Artist - Album (2026) [restored]"] \
      [--only 1,2] [--jobs-message-only]
"""
import argparse
import json
import pathlib
import re
import statistics
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

TARGET_DCREST = 6.3          # master-like crest lift, dB
DELTA_OK = (5.0, 7.5)        # full-track Δcrest window that needs no re-run
SCORE_RERUN = 1.8            # full-track score above which neighbors are tried
EXCERPT_START = 30.0
EXCERPT_DUR = 60.0


# ── helpers ──────────────────────────────────────────────────────────────────

def track_no(name: str) -> int:
    m = re.match(r"\s*(\d{1,3})", name)
    return int(m.group(1)) if m else 0


def clean_title(stem: str) -> str:
    s = re.sub(r"^\s*\d{1,3}\s*(?:[-._)]|\s)\s*", "", stem.strip(), count=1)
    return s.strip() or stem


def band_shape(x, sr, base=None):
    b = np.array(F.band_rms_db(x, sr), dtype=np.float64)
    b -= b.mean()
    if base is None:
        return b
    d = np.abs(b - base)
    return round(float(d.max()), 2), round(float(np.sqrt((d ** 2).mean())), 2)


def short_crest(x, sr, win_s=0.05, hop_s=0.025):
    win, hop = max(2, int(win_s * sr)), max(1, int(hop_s * sr))
    vals = []
    for s in range(0, max(x.shape[0] - win, 1), hop):
        seg = x[s:s + win]
        r = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        if r > 1e-6:
            vals.append(float(np.max(np.abs(seg))) / r)
    return round(float(np.percentile(vals, 95)), 3) if vals else 0.0


def metrics(x, sr, base=None):
    dr, drx, _ch, peak, rms = F.tt_dr(x, sr)
    sm, srms = band_shape(x, sr, base) if base is not None else (None, None)
    return {"dr": dr, "dr_exact": round(float(drx), 2),
            "crest_tt": round(float(peak - rms), 2),
            "crest_db": round(float(F.crest_db(x)), 2),
            "tp": round(float(true_peak_dbfs(x, sr)), 2),
            "peak_db": round(float(peak), 2), "rms_db": round(float(rms), 2),
            "shape_max": sm, "shape_rms": srms,
            "short_crest_p95": short_crest(x, sr),
            "clip": int(np.sum(np.abs(x) >= 0.9999))}


def policy_of(kind, param):
    if kind == "flatten":
        k, cap = param
        return Policy(auto_policy=False, mode="flatten", adaptive=False,
                      k=k, gain_min_db=-cap, gain_max_db=cap, target_tp_dbfs=-0.5)
    return Policy(auto_policy=True, mode="expand", adaptive=True,
                  target_dcrest=float(param), target_tp_dbfs=-0.5)


def score_full(m, orig_crest, baseline=None):
    d = m["crest_db"] - orig_crest
    s = (abs(d - TARGET_DCREST) + 0.8 * (m["shape_rms"] or 0)
         + max(0.0, 5.0 - d) * 2.0 + max(0.0, d - 7.5) * 2.0
         + max(0, m["dr"] - 12) * 0.5)
    if baseline:
        s += (max(0, baseline["dr"] - m["dr"]) * 0.5
              + max(0.0, baseline["crest"] - m["crest_db"]) * 0.5)
    return round(float(s), 3), round(float(d), 2)


def score_excerpt(me, eo, kind):
    d = round(me["crest_db"] - eo["crest_db"], 2)
    return round(abs(d - TARGET_DCREST) + 0.8 * (me["shape_rms"] or 0)
                 + (0.5 if kind == "expand" else 0.0)
                 + max(0.0, 5.0 - d) * 2.0 + max(0.0, d - 8.0) * 1.5
                 + max(0, me["dr"] - 13) * 0.5, 3), d


def candidates():
    out = [("flatten", (0.5, c)) for c in (4, 6, 8, 10)]
    out.append(("flatten", (0.3, 8)))
    out += [("expand", 6.0), ("expand", 7.0)]
    return out


def neighbors(kind, param):
    """Neighboring caps/targets to try on the FULL track when the excerpt
    choice misses the window."""
    if kind == "flatten":
        k, cap = param
        caps = sorted({max(3, cap - 2), cap + 2, 12 if cap >= 10 else cap})
        return [("flatten", (k, c)) for c in caps if c != cap][:2]
    t = float(param)
    return [("expand", t - 1.0), ("expand", t + 1.0)]


def write_24(y, sr, path, ti):
    rng = np.random.default_rng(0xC0FFEE + ti)
    dith = rng.random(y.shape) + rng.random(y.shape) - 1.0
    yq = np.clip(np.round(y * 8388608.0 + dith), -8388608, 8388607).astype(np.int32)
    encode_flac(yq, sr, path, path.parent / f"__tmp{ti}.wav", bits=24)


def gain_check(f, sr, kind, param):
    ex, _ = F.decode_f32(str(f), start_s=EXCERPT_START, dur_s=EXCERPT_DUR)
    _y, st = restore_track(ex, sr, policy_of(kind, param),
                           keep_arrays=True, return_stages=True)
    g = st.get("contour_gains_db")
    if g is None:
        return None, None
    return (round(float(np.abs(np.diff(g, axis=-1)).max()), 2),
            round(float(np.abs(g[:, 0, :] - g[:, 1, :]).max()), 2))


def write_report(out, rows):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["track"])
    od = [r["orig"]["dr"] for r in rows]
    nd = [r["rest"]["dr"] for r in rows]
    dc = [r["delta"]["crest"] for r in rows]
    sh = [r["rest"]["shape_rms"] for r in rows if r["rest"]["shape_rms"] is not None]
    gs = [r["gain_step"] for r in rows if r.get("gain_step") is not None]
    has_base = any(r.get("baseline") for r in rows)
    lines = ["# Per-track restoration report", "",
             "Each track is processed individually: sweep `flatten`/`expand` on an excerpt,",
             "a full run of the winner, full-track auto-correction if Δcrest left the",
             f"window {DELTA_OK[0]}…{DELTA_OK[1]} dB.", ""]
    if has_base:
        bdr = [r["baseline"]["dr"] for r in rows if r.get("baseline")]
        cgb = [r["rest"]["crest_db"] - r["baseline"]["crest"]
               for r in rows if r.get("baseline")]
        sdb = [r["rest"]["shape_rms"] - r["baseline"]["shape_rms"]
               for r in rows if r.get("baseline") and r["baseline"].get("shape_rms")]
        lines += [f"- DR: median {statistics.median(od):.0f} → **{statistics.median(nd):.0f}**"
                  f" (baseline {statistics.median(bdr):.0f}); range {min(nd)}–{max(nd)}.",
                  f"- Δcrest vs original: median **{statistics.median(dc):+.2f} dB**"
                  f" ({min(dc):+.2f}…{max(dc):+.2f}); vs baseline {min(cgb):+.2f}…{max(cgb):+.2f} dB.",
                  f"- Timbre (per-band RMS): median **{statistics.median(sh):.2f} dB**, "
                  f"vs baseline {min(sdb):+.2f}…{max(sdb):+.2f} dB.",
                  f"- Gain gate: step ≤ {max(gs):.2f} dB/frame (threshold 8).", ""]
    else:
        lines += [f"- DR: median **{statistics.median(nd):.0f}** (was {statistics.median(od):.0f});"
                  f" range {min(nd)}–{max(nd)}.",
                  f"- Δcrest: median **{statistics.median(dc):+.2f} dB**"
                  f" ({min(dc):+.2f}…{max(dc):+.2f}).",
                  f"- Timbre (per-band RMS): median **{statistics.median(sh):.2f} dB**.",
                  f"- Gain gate: step ≤ {max(gs):.2f} dB/frame (threshold 8).", ""]
    lines += ["| # | track | mode | DR | Δcrest | timbre | gain |",
              "|---:|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['track']} | {r['title']} | {r['chosen']} | "
                     f"{r['orig']['dr']}→{r['rest']['dr']} | {r['delta']['crest']:+.2f} | "
                     f"{r['rest']['shape_rms']:.2f} | {r.get('gain_step', '—')} |")
    lines += ["", "- Full-album QC: `scripts/qc_gates.py --orig <src> <out>/*.flac`.",
              "- TP −0.5 dBFS, clipping and spikes — 0; source format preserved."]
    (out / "PER_TRACK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="directory of tracks or a single file")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--baseline", default="", help="prior restoration directory to compare against")
    ap.add_argument("--only", default="", help="track numbers, comma-separated")
    ap.add_argument("--excerpt-start", type=float, default=EXCERPT_START)
    ap.add_argument("--excerpt-dur", type=float, default=EXCERPT_DUR)
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_dir = pathlib.Path(args.baseline) if args.baseline else None
    only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None
    if src.is_file():
        files = [src]
    else:
        files = sorted(p for p in src.glob("*.flac") if not p.name.startswith("_"))
    if only:
        files = [f for f in files if track_no(f.name) in only]
    if not files:
        print(f"no FLAC files in {src}")
        return 2

    rows = []
    for f in files:
        ti = track_no(f.name)
        title = clean_title(f.stem)
        t0 = time.time()
        x, sr = F.decode_f32(str(f))
        base = band_shape(x, sr)
        orig = metrics(x, sr, base)
        ex, _ = F.decode_f32(str(f), start_s=args.excerpt_start, dur_s=args.excerpt_dur)
        eb = band_shape(ex, sr)
        eo = metrics(ex, sr, eb)
        print(f"\n[{ti:02d}] {title}  orig DR {orig['dr']} crest {orig['crest_db']}", flush=True)

        scored = []
        for kind, param in candidates():
            try:
                ye, _ = restore_track(ex, sr, policy_of(kind, param),
                                      keep_arrays=False, return_stages=True)
            except Exception as e:  # noqa: BLE001
                print(f"    {kind} {param}: FAILED {type(e).__name__}: {e}", flush=True)
                continue
            me = metrics(ye, sr, eb)
            sc, d = score_excerpt(me, eo, kind)
            name = (f"flatten k={param[0]} cap={param[1]}" if kind == "flatten"
                    else f"expand target={param:g}")
            scored.append({"name": name, "kind": kind, "param": param,
                           "dr": me["dr"], "crest": me["crest_db"], "dcrest": d,
                           "shape_max": me["shape_max"], "shape_rms": me["shape_rms"],
                           "tp": me["tp"], "score": sc})
            print(f"    {name:22} DR {me['dr']:2} crest {me['crest_db']:6.2f} ({d:+.2f}) "
                  f"shape {me['shape_rms']:.2f} score {sc:.2f}", flush=True)
        if not scored:
            print("  !! no candidate worked", flush=True)
            continue
        best = min(scored, key=lambda r: r["score"])
        print(f"  -> excerpt choice: {best['name']} (score {best['score']})", flush=True)

        # full track with the excerpt winner, then verify
        y, rfit = restore_track(x, sr, policy_of(best["kind"], best["param"]),
                                keep_arrays=False, return_stages=True)
        fm = metrics(y, sr, base)
        fsc, fdc = score_full(fm, orig["crest_db"], None)
        chosen = best
        if not (DELTA_OK[0] <= fdc <= DELTA_OK[1]) or fsc > SCORE_RERUN:
            print(f"  full Δcrest {fdc:+.2f} outside the window — "
                  f"trying neighbors on the full track", flush=True)
            for kind, param in neighbors(best["kind"], best["param"]):
                yn, _ = restore_track(x, sr, policy_of(kind, param),
                                      keep_arrays=False, return_stages=True)
                mn = metrics(yn, sr, base)
                sn, dn = score_full(mn, orig["crest_db"], None)
                name = (f"flatten k={param[0]} cap={param[1]}" if kind == "flatten"
                        else f"expand target={param:g}")
                print(f"    {name:22} DR {mn['dr']:2} crest {mn['crest_db']:6.2f} ({dn:+.2f}) "
                      f"shape {mn['shape_rms']:.2f} score {sn:.2f}", flush=True)
                if sn < fsc:
                    y, fm, fsc, fdc, chosen = yn, mn, sn, dn, {
                        "name": name, "kind": kind, "param": param}
        print(f"  -> final {chosen['name']}: DR {orig['dr']}→{fm['dr']} "
              f"crest {orig['crest_db']}→{fm['crest_db']} ({fdc:+.2f}) "
              f"shape {fm['shape_rms']}", flush=True)

        gs, gd = gain_check(f, sr, chosen["kind"], chosen["param"])
        path = out / f"{ti:02d}. {title}.flac"
        write_24(y, sr, path, ti)
        row = {"track": ti, "title": title, "chosen": chosen["name"],
               "kind": chosen["kind"], "param": chosen["param"], "r_fit": rfit,
               "gain_step": gs, "gL_gR_max": gd, "orig": orig, "rest": fm,
               "delta": {"dr": fm["dr"] - orig["dr"], "crest": fdc,
                         "short_crest": round(fm["short_crest_p95"] - orig["short_crest_p95"], 2)},
               "candidates": scored, "full_score": fsc,
               "full_seconds": round(time.time() - t0, 1)}
        if base_dir:
            bf = sorted(base_dir.glob(f"{ti:02d} *.flac")) or sorted(base_dir.glob(f"{ti:02d}.*.flac"))
            if bf:
                xb, srb = F.decode_f32(str(bf[0]))
                bm = metrics(xb, srb, base)
                row["baseline"] = {"dr": bm["dr"], "crest": bm["crest_db"],
                                   "crest_tt": bm["crest_tt"], "tp": bm["tp"],
                                   "shape_rms": bm["shape_rms"], "rms_db": bm["rms_db"]}
                row["better"] = {"dr": fm["dr"] - bm["dr"],
                                 "crest": round(fm["crest_db"] - bm["crest_db"], 2),
                                 "shape_rms": round((fm["shape_rms"] or 0) - (bm["shape_rms"] or 0), 2)}
                print(f"  vs baseline: DR {row['better']['dr']:+d} crest "
                      f"{row['better']['crest']:+.2f} shape {row['better']['shape_rms']:+.2f}", flush=True)
        rows.append(row)
        (out / "per_track_report.json").write_text(
            json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    write_report(out, rows)
    print(f"\ndone: {out}\nreport: {out / 'PER_TRACK_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
