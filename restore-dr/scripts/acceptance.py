#!/usr/bin/env python3
"""acceptance.py — three-way acceptance table: original vs reference restoration vs ours.

Compares TT DR / LUFS / LRA / TP across up to three dirs of the same album.
Tracks are matched by their leading number ("01 - ...").

Usage:
  acceptance.py --orig DIR --mine DIR [--ref DIR] [--json out.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import decode, ebur128, tt_dr


def m(f):
    xf, fs = decode(str(f), "f32le")
    dr, drx, _, _, _ = tt_dr(xf, fs)
    i, lra, tp = ebur128(str(f))
    return {"dr": dr, "drx": round(drx, 1), "lufs": round(i, 1), "lra": round(lra, 1), "tp": round(tp, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True, help="original album dir")
    p.add_argument("--mine", required=True, help="our restoration dir")
    p.add_argument("--ref", help="reference restoration dir (a commercial 'restored' edition)")
    p.add_argument("--json", help="write the table here")
    a = p.parse_args()

    orig, mine, ref = Path(a.orig), Path(a.mine), (Path(a.ref) if a.ref else None)
    rows = []
    for o in sorted(orig.glob("*.flac")):
        n = re.match(r"^(\d+)", o.name).group(1)
        r = next(ref.glob(f"{n}. *.flac"), None) if ref else None
        mn = mine / o.name
        row = {"track": n, "orig": m(o),
               "ref": m(r) if r else None,
               "mine": m(mn) if mn.exists() else None}
        rows.append(row)
        d, rf, mn_ = row["orig"], row["ref"], row["mine"]
        ref_part = f"{rf['dr']}(ref)/" if rf else "-"
        print(f"{n}: DR {d['dr']}->{ref_part}{mn_['dr'] if mn_ else '?'}(ours"
              + (f", x{mn_['drx']}" if mn_ else "") + ")   "
              f"LUFS {d['lufs']}->{rf['lufs'] if rf else '-'}/{mn_['lufs'] if mn_ else '-'}   "
              f"LRA {d['lra']}->{rf['lra'] if rf else '-'}/{mn_['lra'] if mn_ else '-'}   "
              f"TP {mn_['tp'] if mn_ else '-'}")
    avg = lambda k, v: sum(x[k][v] for x in rows if x[k]) / max(sum(1 for x in rows if x[k]), 1)
    ref_dr = f"ref {avg('ref','dr'):.1f} / " if ref else ""
    print(f"\nMeans: DR {avg('orig','dr'):.1f} -> {ref_dr}ours {avg('mine','dr'):.1f} "
          f"(DRx {avg('mine','drx'):.1f}); LUFS {avg('orig','lufs'):.1f} -> "
          + (f"{avg('ref','lufs'):.1f} / " if ref else "") + f"{avg('mine','lufs'):.1f}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
