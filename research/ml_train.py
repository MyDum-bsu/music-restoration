#!/usr/bin/env python3
"""Train a target model on the current dataset version + snapshot + metrics.

The model is picked by volume (see rnext/ml/models.py) or set explicitly.
Split by album (default) or by author (`--by-author`) — the latter is the
honest check of transfer between masters.

  python3 scripts/ml_train.py --model auto
  python3 scripts/ml_train.py --model ridge --by-author
  python3 scripts/ml_train.py --list
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext.ml import dataset as ds  # noqa: E402
from rnext.ml import metrics as M  # noqa: E402
from rnext.ml import models as Z  # noqa: E402
from rnext.ml import registry as reg  # noqa: E402


def train_one(table, base: str, train_idx, hold_idx, target: str, seed: int,
              conformal: bool = False) -> tuple[object, dict]:
    Xtr, ytr = table.X[train_idx], table.y[train_idx]
    Xho, yho = table.X[hold_idx], table.y[hold_idx]
    # calibration part of train — for the conformal band
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    n_cal = max(1, int(0.25 * len(perm)))
    cal, fit = perm[:n_cal], perm[n_cal:]

    if len(fit) < Z.MIN_N.get(base, 1):
        print(f"  [{base}] after the calibration split {len(fit)} tracks remain "
              f"(need {Z.MIN_N.get(base)}) — skipping")
        return None
    # conformal is an add-on, not a base model: built on top of ridge
    real_base = "ridge" if base == "conformal" else base
    model = Z.build(real_base, len(fit), seed=seed, feature_names=table.feature_names,
                    classes=table.classes[train_idx][fit], X_train=Xtr[fit], y_train=ytr[fit])
    if base == "conformal":
        model = Z.fit_conformal(model, Xtr[cal], ytr[cal], level=0.8)
    pred = np.asarray(model.predict(Xho))
    stats = M.evaluate(yho, pred, Xho[:, table.feature_names.index("dr")],
                       classes=table.classes[hold_idx])
    if base == "conformal":
        lo, hi = model.interval(Xho, level=0.8)
        stats["coverage_80"] = M.interval_coverage(yho, lo, hi)
    stats["vs_noop"] = M.compare_to_baseline(stats, "noop")
    stats["vs_v0table"] = M.compare_to_baseline(stats, "v0table")
    # the gate is required here too: without a verdict the snapshot is invisible to the runtime
    # (it only picks gate-passing snapshots) — manual runs produced "dead" snapshots
    stats["gate"] = M.auto_gate(yho, pred, Xho[:, table.feature_names.index("dr")])
    Xf = Xtr[fit]
    fstats = {}
    for i, name in enumerate(table.feature_names):
        col = Xf[:, i]
        col = col[~np.isnan(col)]
        if col.size:
            fstats[name] = {"mean": round(float(np.mean(col)), 4),
                            "std": round(float(np.std(col)) or 1.0, 4)}
    info = {"n_train": int(len(fit)), "n_cal": int(len(cal)), "n_hold": int(len(hold_idx)),
            "features": table.feature_names, "conformal": conformal,
            "feature_stats": fstats}
    return model, {"metrics": stats, **info}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", help="auto | table | ridge | knn | hgb | hgbq")
    ap.add_argument("--target", default="dcrest")
    ap.add_argument("--by-author", action="store_true", help="holdout = a whole author")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--list", action="store_true", help="show the snapshot registry")
    args = ap.parse_args()

    if args.list:
        for r in reg.list_snapshots():
            print(json.dumps(r, ensure_ascii=False))
        return 0

    manifest = reg.write_manifest()
    print(f"dataset version: {manifest['version']} "
          f"(pairs {manifest['pairs']}, tracks {manifest['tracks']}, "
          f"trusted {manifest['trusted_tracks']}, labels {manifest['labels']})")

    table = ds.build_table(target=args.target)
    if len(table) == 0:
        print("table is empty — no trusted pairs with features")
        return 0
    n_authors = len(set(table.authors.tolist()))
    print(f"tracks in table: {len(table)}, authors {n_authors}, "
          f"albums {len(set(table.groups.tolist()))}")

    candidates = Z.suggest_models(len(table), n_authors) if args.model == "auto" else [args.model]
    if args.model == "auto":
        print(f"volume {len(table)} → models: {candidates}")

    mode = "by-author" if args.by_author else "by-album"
    try:
        train_idx, hold_idx = ds.split(table, mode=mode, frac=args.frac, seed=args.seed)
    except ValueError as e:
        print(f"[skip] {e}")
        return 0
    print(f"split {mode}: train {len(train_idx)} / holdout {len(hold_idx)} tracks")
    if len(train_idx) < 20 or len(hold_idx) < 5:
        print("[skip] too little data for an honest evaluation — too early to train "
              f"(need ≥150 tracks for ridge, ≥400 for hgb)")
        return 0

    results = []
    for base in candidates:
        if len(train_idx) < Z.MIN_N.get(base, 1):
            print(f"  {base}: skipping (too little data, need {Z.MIN_N.get(base)})")
            continue
        model, info = train_one(table, base, train_idx, hold_idx, args.target, args.seed)
        print(f"\n== {base}: median|err| {info['metrics']['all']['median_abs']:.2f} dB, "
              f"share≤0.5 {info['metrics']['all']['share_le_0.5']:.0%}, "
              f"vs noop ×{info['metrics']['vs_noop']}")
        print(M.format_report(info["metrics"]))
        results.append((base, model, info))
        if not args.no_save:
            mid = reg.snapshot_id(base, args.target, manifest["version"])
            gate = info["metrics"].get("gate") or {}
            meta = {"model": base, "target": args.target,
                    "dataset_version": manifest["version"],
                    "promote": bool(gate.get("promote")),
                    "gate_vs_best": (gate.get("vs_best") or {}).get("improvement_dB"),
                    "gate_ci_low": (gate.get("vs_best") or {}).get("ci_low"),
                    "level": len(candidates),
                    "dataset": {k: manifest[k] for k in ("pairs", "tracks", "trusted_tracks",
                                                         "by_author", "labels")},
                    "holdout_mode": mode, "seed": args.seed, "frac": args.frac,
                    "caps": {"dcrest_max": 8.0},
                    **{k: v for k, v in info.items() if k != "metrics"},
                    "metrics": info["metrics"]}
            d = reg.save_snapshot(model, meta, mid)
            print(f"  snapshot: {d}")

    if results and not args.no_save:
        best = min(results, key=lambda r: r[2]["metrics"]["all"]["median_abs"])
        print(f"\nbest model of this run: {best[0]} "
              f"(median|err| {best[2]['metrics']['all']['median_abs']:.2f} dB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
