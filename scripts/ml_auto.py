#!/usr/bin/env python3
"""Оркестратор: по мере роста датасета — обучение, эвалы, лидерборд.

Логика (идемпотентная, можно звать из cron или держать в цикле):

  1. пересчитать версию датасета (manifest);
  2. проверить объёмы: уровни M1=600 / M2=1500 / M3=3000 trusted-треков,
     плюс требование ≥2 авторов и ≥5 альбомов в holdout;
  3. для моделей, допустимых на этом объёме и ещё не обученных на ЭТОЙ версии
     датасета, — обучить (сплит по альбомам); при ≥2 авторах дополнительно
     обучить/оценить перенос (сплит по авторам);
  4. записать снапшоты и дописать строку в лидерборд;
  5. напечатать сводку: что обучено, что лучше базы, что ждёт объёма.

  python3 scripts/ml_auto.py             # один проход
  python3 scripts/ml_auto.py --interval 1800   # цикл, раз в 30 минут
  python3 scripts/ml_auto.py --force     # обучить всё заново на текущей версии
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext.ml import dataset as ds  # noqa: E402
from rnext.ml import metrics as M  # noqa: E402
from rnext.ml import models as Z  # noqa: E402
from rnext.ml import registry as reg  # noqa: E402

LEVELS = [(1, 600, "M1: регрессия цели"), (2, 1500, "M2: полосы/тилт/гейт"),
          (3, 3000, "M3: универсальность")]
ROOT_M = Path(__file__).resolve().parents[1] / "models"
LEADERBOARD = ROOT_M / "leaderboard.jsonl"


def level_of(tracks: int) -> int:
    lvl = 0
    for n, need, _ in LEVELS:
        if tracks >= need:
            lvl = n
    return lvl


def train_and_eval(table, base: str, target: str, holdout_mode: str, seed: int) -> dict | None:
    try:
        train_idx, hold_idx = ds.split(table, mode=holdout_mode, frac=0.3, seed=seed)
    except ValueError as e:
        print(f"  [{base}/{holdout_mode}] пропуск сплита: {e}")
        return None
    if len(train_idx) < Z.MIN_N.get(base, 1) or len(hold_idx) < 5:
        print(f"  [{base}/{holdout_mode}] мало данных: train {len(train_idx)}, hold {len(hold_idx)}")
        return None

    Xtr, ytr = table.X[train_idx], table.y[train_idx]
    Xho, yho = table.X[hold_idx], table.y[hold_idx]
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
    stats["gate"] = M.auto_gate(yho, pred, Xho[:, table.feature_names.index("dr")])
    return {"model": model, "metrics": stats, "pred": pred, "y_hold": yho,
            "n_train": int(len(fit)), "n_cal": int(n_cal),
            "n_hold": int(len(hold_idx)), "holdout_mode": holdout_mode}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=0, help="цикл, секунд (0 = один проход)")
    ap.add_argument("--force", action="store_true", help="переобучить на текущей версии")
    ap.add_argument("--target", default="dcrest")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--audio-max-pairs", type=int, default=3,
                    help="сколько пар прогнать через аудио-эвал (0 = не гонять)")
    ap.add_argument("--audio-window", default="02:00-07:00",
                    help="окно, когда аудио-эвал гоняется в полном объёме (локальное время)")
    ap.add_argument("--with-audio", action="store_true",
                    help="плюс к обучению — end-to-end аудио-эвал на парах с сырьём")
    args = ap.parse_args()
    while True:
        run_once(args)
        if args.interval <= 0:
            return 0
        time.sleep(args.interval)


def in_window(spec: str) -> bool:
    """'02:00-07:00' → сейчас внутри окна?"""
    try:
        a, b = spec.split("-")
        ha, ma = (int(x) for x in a.split(":"))
        hb, mb = (int(x) for x in b.split(":"))
        now = time.localtime()
        cur = now.tm_hour * 60 + now.tm_min
        start, end = ha * 60 + ma, hb * 60 + mb
        return start <= cur <= end if start <= end else (cur >= start or cur <= end)
    except Exception:                                     # noqa: BLE001
        return True


def run_once(args) -> None:
    manifest = reg.write_manifest()
    trusted = manifest["trusted_tracks"]
    lvl = level_of(trusted)
    print(f"[{time.strftime('%H:%M:%S')}] версия {manifest['version']}: "
          f"пар {manifest['pairs']}, trusted-треков {trusted}, уровень M{lvl}")

    table = ds.build_table(target=args.target)
    if len(table) == 0:
        print("  таблица пуста — ждём данные")
        return
    n_authors = len(set(table.authors.tolist()))
    n_albums = len(set(table.groups.tolist()))
    print(f"  треков {len(table)}, авторов {n_authors}, альбомов {n_albums}")

    todo = [m for m in Z.suggest_models(len(table), n_authors)
            if args.force or not reg.already_trained(manifest["version"], m.replace("conformal", "ridge"),
                                                      args.target)]
    # conformal строится поверх ridge/hgb — обучаем отдельно поверх лучшей базы
    modes = ["by-album"] + (["by-author"] if n_authors >= 2 else [])
    done = 0
    for mode in modes:
        for base in [m for m in todo if m != "conformal"]:
            if base == "conformal":
                continue
            res = train_and_eval(table, base, args.target, mode, args.seed)
            if not res:
                continue
            mid = reg.snapshot_id(base, args.target, manifest["version"])
            meta = {
                "model": base, "target": args.target,
                "dataset_version": manifest["version"],
                "dataset": {k: manifest[k] for k in ("pairs", "tracks", "trusted_tracks",
                                                     "by_author", "labels")},
                "level": lvl, "seed": args.seed, "frac": 0.3,
                "features": table.feature_names, "n_train": res["n_train"],
                "n_cal": res["n_cal"], "n_hold": res["n_hold"],
                "holdout_mode": mode, "caps": {"dcrest_max": 8.0},
                "promote": res["metrics"]["gate"]["promote"],
                "gate_gain_db": res["metrics"]["gate"]["vs_best"]["improvement_dB"],
                "gate_ci_low": res["metrics"]["gate"]["vs_best"]["ci_low"],
                "metrics": res["metrics"],
            }
            reg.save_snapshot(res["model"], meta, mid)
            LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
            with open(LEADERBOARD, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "model_id": mid, "model": base, "target": args.target,
                    "dataset_version": manifest["version"], "level": lvl,
                    "holdout_mode": mode, "n_train": res["n_train"],
                    "n_hold": res["n_hold"],
                    "median_abs": res["metrics"]["all"]["median_abs"],
                    "share_le_0.5": res["metrics"]["all"]["share_le_0.5"],
                    "bias": res["metrics"]["all"]["bias"],
                    "baseline_noop": res["metrics"].get("baseline_noop", {}).get("median_abs"),
                    "baseline_const46": res["metrics"].get("baseline_const46", {}).get("median_abs"),
                    "baseline_v0table": res["metrics"].get("baseline_v0table", {}).get("median_abs"),
                    "coverage_80": res["metrics"].get("coverage_80"),
                    "promote": res["metrics"]["gate"]["promote"],
                    "gate_vs_best": res["metrics"]["gate"]["vs_best"]["improvement_dB"],
                    "gate_ci_low": res["metrics"]["gate"]["vs_best"]["ci_low"],
                }, ensure_ascii=False) + "\n")
            done += 1
            g = res["metrics"]["gate"]["vs_best"]
            print(f"  ✓ {base}/{mode}: медиана|err| {res['metrics']['all']['median_abs']:.2f} "
                  f"(noop {res['metrics']['baseline_noop']['median_abs']:.2f}, "
                  f"const46 {res['metrics']['baseline_const46']['median_abs']:.2f}, "
                  f"v0 {res['metrics']['baseline_v0table']['median_abs']:.2f}) | "
                  f"гейт: {g['improvement_dB']:+.2f} дБ [{g['ci_low']:+.2f},{g['ci_high']:+.2f}] "
                  f"{'PROMOTE' if res['metrics']['gate']['promote'] else 'держим прежнее'}")

    if not done:
        print("  новых моделей нет (всё обучено на этой версии)")

    volume_checkpoints(table, manifest, args)

    nxt = next(((n, need, name) for n, need, name in LEVELS if trusted < need), None)
    if nxt:
        print(f"  до M{nxt[0]} ({nxt[2]}): ещё {nxt[1] - trusted} trusted-треков")
    else:
        print("  все уровни по объёму достигнуты — дальше только качество и слух")
    if args.with_audio:
        run_audio_eval(args, manifest)
    print(leaderboard_text())


def run_audio_eval(args, manifest) -> None:
    """End-to-end эвал: наш прогон на оригинале пары против мастера.

    Гоняется только по парам, у которых сохранилось сырьё (--keep-raw при
    извлечении). Результат прикладывается к версии датасета.
    """
    import subprocess
    root = Path(__file__).resolve().parents[1]
    n = args.audio_max_pairs
    if n <= 0:
        return
    night = in_window(args.audio_window)
    if not night:
        n = 1                     # днём — минимум, полный прогон в окне
        print(f"  аудио-эвал: днём только {n} пара (полный объём в окне {args.audio_window})")
    out = subprocess.run([sys.executable, str(root / "scripts/dataset_eval.py"),
                          "--all", "--limit", str(n), "--max-tracks", "2",
                          "--variants", "flat,class",
                          "--summary-out", str(reg.dataset_dir() / "audio_eval_summary.json")],
                         capture_output=True, text=True, cwd=str(root))
    tail = (out.stdout or "").strip().splitlines()[-6:]
    print("  аудио-эвал:")
    for line in tail:
        print("   ", line)
    sfile = reg.dataset_dir() / "audio_eval_summary.json"
    if sfile.is_file():
        summary = json.loads(sfile.read_text(encoding="utf-8"))
        summary["dataset_version"] = manifest["version"]
        summary["at"] = int(time.time())
        with open(reg.dataset_dir() / "audio_evals.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def volume_checkpoints(table, manifest, args) -> None:
    """Снапшот модели каждые +100 trusted-треков: кривая «качество против объёма».

    Это не рабочие модели, а метки роста: они позволяют увидеть, где данные
    перестают давать выигрыш (и не тратить трафик впустую).
    """
    curve = ROOT_M / "volume_curve.jsonl"
    step = 100
    trusted = manifest["trusted_tracks"]
    bucket = trusted // step
    done = set()
    if curve.is_file():
        for l in curve.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r["dataset_version"] == manifest["version"]:
                    done.add(r["model"])
    if not bucket:
        return
    for base in ("table", "ridge"):
        if base in done or len(table) < Z.MIN_N.get(base, 1):
            continue
        if reg.already_trained(manifest["version"], base, args.target):
            continue
        res = train_and_eval(table, base, args.target, "by-album", args.seed)
        if not res:
            continue
        row = {"dataset_version": manifest["version"], "tracks": trusted,
               "bucket": bucket, "model": base,
               "median_abs": res["metrics"]["all"]["median_abs"],
               "baseline_v0table": res["metrics"]["baseline_v0table"]["median_abs"],
               "baseline_const46": res["metrics"]["baseline_const46"]["median_abs"],
               "n_train": res["n_train"], "n_hold": res["n_hold"],
               "gate_promote": res["metrics"]["gate"]["promote"],
               "gate_gain": res["metrics"]["gate"]["vs_best"]["improvement_dB"]}
        with open(curve, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  кривая объёма: {base} на {trusted} треках → медиана|err| "
              f"{row['median_abs']:.2f} (v0 {row['baseline_v0table']:.2f})")


def leaderboard_text(limit: int = 8) -> str:
    if not LEADERBOARD.is_file():
        return "  лидерборд пуст"
    rows = [json.loads(l) for l in LEADERBOARD.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: (r["holdout_mode"], r["median_abs"]))
    out = ["  лидерборд (медиана|err|, дБ):"]
    for r in rows[:limit]:
        out.append(f"    {r['holdout_mode']:<9} {r['model']:<10} ds={r['dataset_version']} "
                   f"n={r['n_train']:<4} {r['median_abs']:.2f} "
                   f"(const46 {r['baseline_const46']:.2f}, v0 {r['baseline_v0table']:.2f})")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
