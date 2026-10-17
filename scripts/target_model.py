#!/usr/bin/env python3
"""Целевой слой v0: какую цель Δcrest ставить по классу исходника.

Источники:
  * `dataset/pairs/*/features.json` — измеренные Δcrest мастера по классам
    (только trusted-пары из reference_audit);
  * `corpus/master_targets.jsonl` — абсолютные цели мастера (8104 трека), нужны
    для проверки, что полосы классов не противоречат корпусу.

Выход: `corpus/targets_v0.json` — таблица по классам, физические скобки и
рекомендация `target_dcrest` для пайплайна (с поправкой на недобор фиттера).

  python3 scripts/target_model.py            # построить
  python3 scripts/target_model.py --report
"""
import argparse
import json
import statistics as S
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import workspace  # noqa: E402

CLASS_EDGES = ((0, 9, "crushed"), (9, 12, "mid"), (12, 99, "healthy"))
FITTER_UNDERSHOOT = 0.4      # измеренный недобор адаптива на DR 5–6 (README)


def corpus() -> Path:
    return workspace.dir_of("corpus")


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def q(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def cls_of(dr: float) -> str:
    for lo, hi, name in CLASS_EDGES:
        if lo <= dr < hi:
            return name
    return "healthy"


def build() -> int:
    quality = {r["key"]: r for r in read_jsonl(dataset_dir() / "reference_quality.jsonl")}
    by_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_pairs = n_tracks = 0
    for d in sorted((dataset_dir() / "pairs").glob("*/features.json")):
        key = d.parent.name.replace("__", "/")
        qrec = quality.get(key)
        if qrec and qrec["label"] not in ("trusted", "weak_base"):
            continue                     # сомнительные референсы в цель не идут
        rows = json.loads(d.read_text(encoding="utf-8"))
        if not rows:
            continue
        n_pairs += 1
        for r in rows:
            c = cls_of(r["original"]["dr"])
            n_tracks += 1
            by_class[c]["dcrest_our_master"].append(r["delta"]["crest"])
            by_class[c]["ddr"].append(r["delta"]["dr"])
            by_class[c]["origin_dr"].append(r["original"]["dr"])
            by_class[c]["origin_rms"].append(r["original"]["rms_db"])

    table = {}
    for c, v in by_class.items():
        dc = v["dcrest_our_master"]
        table[c] = {
            "n_tracks": len(dc),
            "origin_dr_median": S.median(v["origin_dr"]),
            "dcrest_median": q(dc, 0.5), "dcrest_p25": q(dc, 0.25), "dcrest_p75": q(dc, 0.75),
            "dcrest_min": q(dc, 0.0), "dcrest_max": q(dc, 1.0),
            "ddr_median": q(v["ddr"], 0.5),
        }

    # абсолютные цели мастера (без исходников) — контроль непротиворечивости
    mt = read_jsonl(corpus() / "master_targets.jsonl")
    abs_by_dr: dict[str, list[float]] = defaultdict(list)
    for r in mt:
        if r.get("official_dr"):
            abs_by_dr["dr"].append(r["official_dr"])
        for t in r.get("tracks") or []:
            abs_by_dr["crest"].append(t["crest_tt"])

    rec = {}
    for c, t in table.items():
        # просим у адаптива чуть больше медианы мастера: фиттер недобирает
        rec[f"target_dcrest_{c}"] = (round(t["dcrest_median"] + FITTER_UNDERSHOOT, 2)
                                     if c != "healthy" else 0.0)
    out = {
        "by_class": table,
        "caps": {"dcrest_max": 8.0, "tilt_max_db": 3.0, "tp_ceiling_db": -0.5},
        "recommendation": rec,
        "fitter_undershoot_db": FITTER_UNDERSHOOT,
        "basis": {"pairs": n_pairs, "tracks": n_tracks,
                  "master_releases_with_dr": len(mt),
                  "master_crest_tt_median": q(abs_by_dr["crest"], 0.5) if abs_by_dr["crest"] else None},
    }
    p = corpus() / "targets_v0.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"цели v0 → {p}")
    print(json.dumps(out["by_class"], ensure_ascii=False, indent=1))
    print("рекомендация:", rec)
    print("базис:", out["basis"])
    return 0


def report() -> int:
    p = corpus() / "targets_v0.json"
    if not p.is_file():
        print("нет targets_v0.json — запустите без --report")
        return 1
    print(p.read_text(encoding="utf-8"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    return report() if args.report else build()


if __name__ == "__main__":
    sys.exit(main())
