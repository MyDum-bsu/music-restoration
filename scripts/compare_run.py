#!/usr/bin/env python3
"""Comparing our run (restore_report.jsonl) against master-restoration.

Reads restore_report.jsonl from the out directory and
corpus/feature_pairs/{key}.jsonl, matches by track_n and prints an
"ours vs master" table: DR, crest, TP, integrated RMS and deltas.

Acceptance criteria:
  crest:  |Δcrest| ≤ 0.5 dB  (matches the master)
  DR:     |ΔDR| ≤ 1         (the master's TT-DR is usually ~1 lower at the same crest)
  RMS:    |ΔRMS| ≤ 1.0 dB   with a TP ceiling of −0.5 on both (level under the peak)
  band:   |Δband| ≤ 3 dB    (averaged over 1/3-octave bands, level-independent)

`--bands` additionally decodes our tracks and computes per-band deltas to the
master; the master uses the ready bands_db from the corpus pair, and the source
does too (no need to re-decode the source).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import features as F  # noqa: E402
from rnext import workspace  # noqa: E402

TP_CEIL = -0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--bands", action="store_true",
                    help="compute per-band deltas (decodes our FLACs)")
    args = ap.parse_args()

    report_path = Path(args.report)
    ours = {}
    for line in report_path.read_text().splitlines():
        r = json.loads(line)
        ours[r["track"]] = r

    rows = {}
    p = workspace.dir_of("corpus") / "feature_pairs" / f"{args.key}.jsonl"
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if "original" in r and "restored" in r:
            rows[r["track_n"]] = r

    hdr = (f"{'t':>3} {'title':<24} | {'DR':>8} {'ΔDR':>4} | {'crest':>11} {'Δcr':>6} | "
           f"{'RMS':>11} {'ΔRMS':>6} | {'TPour':>6} | verdict")
    print(hdr)
    print("-" * len(hdr))

    d_dr, d_crest, d_rms, verdict_bad = [], [], [], []
    band_deltas = []
    for ti, r in sorted(rows.items()):
        if ti not in ours:
            continue
        o = ours[ti]
        m = r["restored"]
        crest_ours = round(o["rest"]["peak_dbfs"] - o["rest"]["rms_db"], 2)
        crest_m = round(m["peak_dbfb"] - m["rms_db"], 2)
        rms_ours, rms_m = o["rest"]["rms_db"], m["rms_db"]
        ddr = o["rest"]["dr"] - m["dr"]
        dcr = round(crest_ours - crest_m, 2)
        drms = round(rms_ours - rms_m, 2)
        tp_ours = o.get("tp_actual", None)

        v = []
        if abs(dcr) > 0.5:
            v.append("crest")
        if abs(ddr) > 1:
            v.append("DR")
        if abs(drms) > 1.0:
            v.append("RMS")
        if tp_ours is not None and tp_ours > TP_CEIL + 0.05:
            v.append("TP")
        verdict = "OK" if not v else "FAIL:" + ",".join(v)
        if v:
            verdict_bad.append((ti, verdict))

        d_dr.append(ddr)
        d_crest.append(dcr)
        d_rms.append(drms)

        bands_str = ""
        if args.bands:
            our_f = report_path.parent / f"{ti:02d}. {o['title']}.flac"
            x, sr = F.decode_f32(str(our_f))
            if x is not None:
                ob = F.band_rms_db(x, sr)
                mb = m["bands_db"]
                if len(ob) == len(mb):
                    bd = [round(a - b, 2) for a, b in zip(ob, mb)]
                    band_deltas.append(bd)
                    bands_str = f"  bandΔ ours−master avg {sum(bd)/len(bd):+.2f}"

        print(f"{ti:>3} {r['cue_title'][:24]:<24} | {o['rest']['dr']:>3}/{m['dr']:<3} {ddr:>+5} | "
              f"{crest_ours:>5.2f}/{crest_m:<5.2f} {dcr:>+6.2f} | "
              f"{rms_ours:>5.2f}/{rms_m:<5.2f} {drms:>+6.2f} | {tp_ours if tp_ours is not None else 0:>6.2f} | {verdict}")

    n = len(d_dr)
    if not n:
        print("no matched tracks")
        return 1

    def med(a):
        s = sorted(a)
        return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s)//2 - 1] + s[len(s)//2])

    print("-" * len(hdr))
    print(f"medians: ΔDR {med(d_dr):+.2f}  Δcrest {med(d_crest):+.2f}  ΔRMS {med(d_rms):+.2f}")
    print(f"means: ΔDR {sum(d_dr)/n:+.2f}  Δcrest {sum(d_crest)/n:+.2f}  ΔRMS {sum(d_rms)/n:+.2f}")
    print(f"range:  ΔDR {min(d_dr):+}..{max(d_dr):+}  Δcrest {min(d_crest):+.2f}..{max(d_crest):+.2f}"
          f"  ΔRMS {min(d_rms):+.2f}..{max(d_rms):+.2f}")
    if band_deltas:
        nb = len(band_deltas[0])
        prof = [sum(b[i] for b in band_deltas) / len(band_deltas) for i in range(nb)]
        print("bandΔ ours−master (1/3-oct, averaged over tracks):")
        print("  " + " ".join(f"{x:+.1f}" for x in prof))
    print(f"verdicts: {n - len(verdict_bad)}/{n} OK" +
          (f"; problems: {', '.join(f't{t}({v.split(':',1)[1]})' for t, v in verdict_bad)}" if verdict_bad else ""))
    return 0 if not verdict_bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
