#!/usr/bin/env python3
"""Оценка пайплайна на собранных парах: наше Δcrest против мастерского.

Это первая честная проверка не-в-выборке: альбомы (Nickelback, Linkin Park,
Motörhead, Metallica, Artillery…), которых не было при калибровке.

Считаем по каждому треку:
    master.dcrest = Δcrest мастера (из features.json пары)
    ours.dcrest   = Δcrest нашего прогона (restore_track с авто-политикой)
    err           = ours − master
Базовая линия «ничего не делать» даёт err = −master.dcrest — с ней и сравниваем.

Варианты политики:
    flat  — текущий дефолт: цель 4.6 на всё, что не healthy/sparse
    class — цель по классу исходника из corpus/targets_v0.json (crushed/mid)

  python3 scripts/dataset_eval.py --all --max-tracks 4 --variants flat,class
"""
import argparse
import json
import statistics as S
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_extract as DX  # noqa: E402
from rnext import workspace  # noqa: E402
from rnext.dsp.pipeline import Policy, diagnose, restore_track  # noqa: E402


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def state_map() -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(dataset_dir() / "state.jsonl")}


def targets() -> dict:
    p = workspace.dir_of("corpus") / "targets_v0.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"recommendation": {}}


def policy_for(variant: str, rec: dict, dr: int, rms: float) -> Policy:
    pol = Policy(auto_policy=True, adaptive=True, mode="expand")
    if variant == "class":
        t = rec.get("recommendation", {})
        if dr >= 12:
            pol.healthy_dr_min = 12
            pol.target_dcrest = 0.0
        elif dr >= 9:
            pol.healthy_dr_min = 12
            pol.target_dcrest = t.get("target_dcrest_mid", pol.target_dcrest)
        else:
            pol.target_dcrest = t.get("target_dcrest_crushed", pol.target_dcrest)
    return pol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--key", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tracks", type=int, default=4, help="треков на пару")
    ap.add_argument("--variants", default="flat,class")
    ap.add_argument("--out", default="eval_results.jsonl")
    ap.add_argument("--summary-out", default="", help="куда записать сводку (json)")
    args = ap.parse_args()

    rec = targets()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    st = state_map()
    pairs_dir = dataset_dir() / "pairs"
    keys = sorted(p.parent.name.replace("__", "/") for p in pairs_dir.glob("*/features.json"))
    if args.key:
        keys = [k for k in keys if k == args.key]
    elif not args.all:
        ap.error("нужен --all или --key")
    if args.limit:
        keys = keys[:args.limit]
    print(f"пар к оценке: {len(keys)}, варианты {variants}, треков на пару {args.max_tracks}")

    out_path = dataset_dir() / args.out
    out_f = open(out_path, "a", encoding="utf-8")
    for key in keys:
        pdir = pairs_dir / key.replace("/", "__")
        rows = json.loads((pdir / "features.json").read_text(encoding="utf-8"))
        srec = st.get(key)
        if not srec:
            print(f"  [skip] {key}: нет записи в журнале")
            continue
        o_root = DX.side_root(srec, "orig", Path(srec.get("dir") or ""))
        if o_root is None:
            print(f"  [skip] {key}: сырьё оригинала удалено (нужен повторный fetch)")
            continue
        o_tracks = DX.side_tracks(o_root)
        loud = [r for r in rows if r["original"]["dr"] < 12]
        pick = sorted(loud, key=lambda r: r["original"]["dr"])[: args.max_tracks] or rows[:1]
        for r in pick:
            try:
                t = o_tracks[r["oi"]]
            except (IndexError, KeyError):
                continue
            x, sr = DX.F.decode_f32(t["file"], t["start_s"], t["dur_s"])
            row = {"key": key, "track": r["track"], "title": r["title"],
                   "src_dr": r["original"]["dr"], "src_rms": r["original"]["rms_db"],
                   "master_dcrest": r["delta"]["crest"], "master_ddr": r["delta"]["dr"],
                   "master_crest": r["restored"]["crest_db"]}
            for v in variants:
                pol = policy_for(v, rec, r["original"]["dr"], r["original"]["rms_db"])
                y, stages = restore_track(x, sr, pol, return_stages=True)
                fy = DX.features_from_array(y, sr)
                row[f"{v}_dcrest"] = round(fy["crest_db"] - r["original"]["crest_db"], 3)
                row[f"{v}_ddr"] = fy["dr"] - r["original"]["dr"]
                row[f"{v}_tp"] = fy["peak_db"]
                row[f"{v}_err"] = round(row[f"{v}_dcrest"] - row["master_dcrest"], 3)
                row[f"{v}_r_fit"] = (stages.get("adaptive") or {}).get("r_fit")
                row[f"{v}_class"] = (stages.get("diagnosis") or {}).get("class")
            row["noop_err"] = round(-row["master_dcrest"], 3)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            errs = " ".join(f"{v}:{row[f'{v}_err']:+.2f}" for v in variants)
            print(f"  {key} t{row['track']:<2} DR{row['src_dr']:>2} маст {row['master_dcrest']:+.2f} "
                  f"наш {errs} (без обработки: {row['noop_err']:+.2f})")
    out_f.close()

    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["key"] in keys]
    if not rows:
        return 0
    summary = {"tracks": len(rows), "pairs": len(set(r["key"] for r in rows)), "variants": {}}
    print(f"\nитог по {len(rows)} трекам (файл {out_path}):")
    print(f"{'вариант':<8}{'медиана |err|':>14}{'доля ≤0.5':>11}{'доля ≤1.0':>11}")
    for v in variants + ["noop"]:
        e = [abs(r["noop_err"] if v == "noop" else r[f"{v}_err"]) for r in rows]
        med = float(S.median(e))
        summary["variants"][v] = {
            "median_abs": round(med, 3),
            "share_le_0.5": round(sum(x <= 0.5 for x in e) / len(e), 3),
            "share_le_1.0": round(sum(x <= 1.0 for x in e) / len(e), 3),
            "bias": round(float(S.median([r["noop_err"] if v == "noop" else r[f"{v}_err"]
                                          for r in rows])), 3),
        }
        print(f"{v:<8}{med:>14.2f}{sum(x <= 0.5 for x in e) / len(e) * 100:>10.0f}%"
              f"{sum(x <= 1.0 for x in e) / len(e) * 100:>10.0f}%")
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        print(f"сводка → {args.summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
