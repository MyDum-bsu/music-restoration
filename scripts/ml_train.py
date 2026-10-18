#!/usr/bin/env python3
"""Обучение модели цели на текущей версии датасета + снапшот + метрики.

Модель выбирается по объёму (см. rnext/ml/models.py) либо задаётся явно.
Сплит по альбомам (по умолчанию) или по авторам (`--by-author`) — последний
и есть честная проверка переноса между мастерами.

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
    # калибровочная часть train — для конформной полосы
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    n_cal = max(1, int(0.25 * len(perm)))
    cal, fit = perm[:n_cal], perm[n_cal:]

    if len(fit) < Z.MIN_N.get(base, 1):
        print(f"  [{base}] после калибровочного сплита осталось {len(fit)} треков "
              f"(нужно {Z.MIN_N.get(base)}) — пропуск")
        return None
    model = Z.build(base, len(fit), seed=seed, feature_names=table.feature_names,
                    classes=table.classes[train_idx][fit], X_train=Xtr[fit], y_train=ytr[fit])
    if conformal:
        model = Z.fit_conformal(model, Xtr[cal], ytr[cal], level=0.8)
    pred = np.asarray(model.predict(Xho))
    stats = M.evaluate(yho, pred, Xho[:, table.feature_names.index("dr")],
                       classes=table.classes[hold_idx])
    if conformal:
        lo, hi = model.interval(Xho, level=0.8)
        stats["coverage_80"] = M.interval_coverage(yho, lo, hi)
    stats["vs_noop"] = M.compare_to_baseline(stats, "noop")
    stats["vs_v0table"] = M.compare_to_baseline(stats, "v0table")
    info = {"n_train": int(len(fit)), "n_cal": int(len(cal)), "n_hold": int(len(hold_idx)),
            "features": table.feature_names, "conformal": conformal}
    return model, {"metrics": stats, **info}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", help="auto | table | ridge | knn | hgb | hgbq")
    ap.add_argument("--target", default="dcrest")
    ap.add_argument("--by-author", action="store_true", help="holdout = целый автор")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--list", action="store_true", help="показать реестр снапшотов")
    args = ap.parse_args()

    if args.list:
        for r in reg.list_snapshots():
            print(json.dumps(r, ensure_ascii=False))
        return 0

    manifest = reg.write_manifest()
    print(f"версия датасета: {manifest['version']} "
          f"(пар {manifest['pairs']}, треков {manifest['tracks']}, "
          f"trusted {manifest['trusted_tracks']}, метки {manifest['labels']})")

    table = ds.build_table(target=args.target)
    if len(table) == 0:
        print("таблица пуста — нет trusted-пар с фичами")
        return 0
    n_authors = len(set(table.authors.tolist()))
    print(f"треков в таблице: {len(table)}, авторов {n_authors}, "
          f"альбомов {len(set(table.groups.tolist()))}")

    candidates = Z.suggest_models(len(table), n_authors) if args.model == "auto" else [args.model]
    if args.model == "auto":
        print(f"объём {len(table)} → модели: {candidates}")

    mode = "by-author" if args.by_author else "by-album"
    try:
        train_idx, hold_idx = ds.split(table, mode=mode, frac=args.frac, seed=args.seed)
    except ValueError as e:
        print(f"[skip] {e}")
        return 0
    print(f"сплит {mode}: train {len(train_idx)} / holdout {len(hold_idx)} треков")
    if len(train_idx) < 20 or len(hold_idx) < 5:
        print("[skip] слишком мало данных для честной оценки — рано обучать "
              f"(нужно ≥150 треков для ridge, ≥400 для hgb)")
        return 0

    results = []
    for base in candidates:
        if len(train_idx) < Z.MIN_N.get(base, 1):
            print(f"  {base}: пропуск (мало данных, нужно {Z.MIN_N.get(base)})")
            continue
        model, info = train_one(table, base, train_idx, hold_idx, args.target, args.seed)
        print(f"\n== {base}: медиана|err| {info['metrics']['all']['median_abs']:.2f} дБ, "
              f"доля≤0.5 {info['metrics']['all']['share_le_0.5']:.0%}, "
              f"против noop ×{info['metrics']['vs_noop']}")
        print(M.format_report(info["metrics"]))
        results.append((base, model, info))
        if not args.no_save:
            mid = reg.snapshot_id(base, args.target, manifest["version"])
            meta = {"model": base, "target": args.target,
                    "dataset_version": manifest["version"],
                    "dataset": {k: manifest[k] for k in ("pairs", "tracks", "trusted_tracks",
                                                         "by_author", "labels")},
                    "holdout_mode": mode, "seed": args.seed, "frac": args.frac,
                    "caps": {"dcrest_max": 8.0},
                    **{k: v for k, v in info.items() if k != "metrics"},
                    "metrics": info["metrics"]}
            d = reg.save_snapshot(model, meta, mid)
            print(f"  снапшот: {d}")

    if results and not args.no_save:
        best = min(results, key=lambda r: r[2]["metrics"]["all"]["median_abs"])
        print(f"\nлучшая модель этого прогона: {best[0]} "
              f"(медиана|err| {best[2]['metrics']['all']['median_abs']:.2f} дБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
