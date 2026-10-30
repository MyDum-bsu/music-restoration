#!/usr/bin/env python3
"""Summary of a blind run (no master): our metrics vs the original.

Reads restore_report.jsonl (contains orig/rest dr/peak/rms, tp_actual,
diag_class, target_dcrest) and prints:
  - a per-track table (DR/crest/RMS/TP/Δ);
  - medians/range;
  - how many tracks landed in the target range of their diagnosis:
      * loud/sparse: |Δcrest_actual − target_dcrest| ≤ 0.5 dB
      * healthy:     |ΔDR| ≤ 0.5 (transparency)
  - a ready markdown block (--md).
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.report).read_text().splitlines() if l.strip()]
    print(f"{'t':>3} {'title':<26} {'class':>8} {'DR':>7} {'crest':>12} {'Δcr':>6} "
          f"{'target':>6} {'RMS':>12} {'TP':>6} verdict")
    hits = 0
    d_dr, d_crest, d_rms = [], [], []
    classes = {}
    md = ["| # | title | class | DR o→r | crest o→r | Δcrest | target | RMS o→r | TP | result |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ti = r["track"]; cls = r.get("diag_class", "?")
        classes[cls] = classes.get(cls, 0) + 1
        o, s = r["orig"], r["rest"]
        c_o = o["peak_dbfs"] - o["rms_db"]
        c_s = s["peak_dbfs"] - s["rms_db"]
        ddr = s["dr"] - o["dr"]
        dcr = c_s - c_o
        drms = s["rms_db"] - o["rms_db"]
        tgt = r.get("target_dcrest")
        d_dr.append(ddr); d_crest.append(dcr); d_rms.append(drms)
        if cls == "healthy":
            ok = abs(ddr) <= args.tol
            crit = f"|ΔDR|≤{args.tol}"
        else:
            ok = tgt is not None and abs(dcr - tgt) <= args.tol
            crit = f"|Δcrest−{tgt}|≤{args.tol}"
        hits += ok
        v = "OK" if ok else "miss"
        print(f"{ti:>3} {r['title'][:26]:<26} {cls:>8} {o['dr']:>2}→{s['dr']:<4} "
              f"{c_o:>5.2f}→{c_s:<5.2f} {dcr:>+6.2f} {str(tgt):>6} "
              f"{o['rms_db']:>6.1f}→{s['rms_db']:<5.1f} {r.get('tp_actual', 0):>6.2f} {v}")
        md.append(f"| {ti} | {r['title'][:26]} | {cls} | {o['dr']}→{s['dr']} | "
                  f"{c_o:.2f}→{c_s:.2f} | {dcr:+.2f} | {tgt} | {o['rms_db']:.1f}→{s['rms_db']:.1f} | "
                  f"{r.get('tp_actual',0):.2f} | {'✅' if ok else '✗'} |")

    def med(a):
        a = sorted(a)
        return a[len(a) // 2] if len(a) % 2 else 0.5 * (a[len(a)//2-1] + a[len(a)//2])

    n = len(rows)
    print("-" * 120)
    print(f"tracks {n} | classes {classes}")
    print(f"ΔDR: median {med(d_dr):+.2f} range {min(d_dr):+}..{max(d_dr):+} | "
          f"Δcrest: median {med(d_crest):+.2f} range {min(d_crest):+.2f}..{max(d_crest):+.2f} | "
          f"ΔRMS: median {med(d_rms):+.2f} range {min(d_rms):+.2f}..{max(d_rms):+.2f}")
    print(f"landed in the diagnosis target range: {hits}/{n} (tolerance ±{args.tol})")
    if args.md:
        print()
        print("\n".join(md))
        print(f"\nΔDR median {med(d_dr):+.2f} (range {min(d_dr):+}..{max(d_dr):+}); "
              f"Δcrest median {med(d_crest):+.2f} (range {min(d_crest):+.2f}..{max(d_crest):+.2f}); "
              f"ΔRMS median {med(d_rms):+.2f}; in target {hits}/{n}; classes {classes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
