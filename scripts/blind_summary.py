#!/usr/bin/env python3
"""Сводка слепого прогона (без мастера): наши метрики vs оригинал.

Читает restore_report.jsonl (содержит orig/rest dr/peak/rms, tp_actual,
diag_class, target_dcrest) и печатает:
  - пер-трековую таблицу (DR/crest/RMS/TP/Δ);
  - медианы/размах;
  - сколько треков попало в целевой диапазон диагноза:
      * loud/sparse: |Δcrest_actual − target_dcrest| ≤ 0.5 дБ
      * healthy:     |ΔDR| ≤ 0.5 (прозрачность)
  - готовый markdown-блок (--md).
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
    print(f"{'t':>3} {'title':<26} {'класс':>8} {'DR':>7} {'crest':>12} {'Δcr':>6} "
          f"{'target':>6} {'RMS':>12} {'TP':>6} вердикт")
    hits = 0
    d_dr, d_crest, d_rms = [], [], []
    classes = {}
    md = ["| # | title | класс | DR o→r | crest o→r | Δcrest | target | RMS o→r | TP | итог |",
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
        v = "OK" if ok else "мимо"
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
    print(f"треков {n} | классы {classes}")
    print(f"ΔDR: медиана {med(d_dr):+.2f} размах {min(d_dr):+}..{max(d_dr):+} | "
          f"Δcrest: медиана {med(d_crest):+.2f} размах {min(d_crest):+.2f}..{max(d_crest):+.2f} | "
          f"ΔRMS: медиана {med(d_rms):+.2f} размах {min(d_rms):+.2f}..{max(d_rms):+.2f}")
    print(f"в целевой диапазон диагноза попало: {hits}/{n} (толеранс ±{args.tol})")
    if args.md:
        print()
        print("\n".join(md))
        print(f"\nΔDR медиана {med(d_dr):+.2f} (размах {min(d_dr):+}..{max(d_dr):+}); "
              f"Δcrest медиана {med(d_crest):+.2f} (размах {min(d_crest):+.2f}..{max(d_crest):+.2f}); "
              f"ΔRMS медиана {med(d_rms):+.2f}; в цели {hits}/{n}; классы {classes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
