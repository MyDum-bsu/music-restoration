#!/usr/bin/env python3
"""Orchestrator: as the dataset grows — training, evals, leaderboard.

Logic (idempotent, can be called from cron or kept in a loop):

  1. recompute the dataset version (manifest);
  2. check volume: levels M1=600 / M2=1500 / M3=3000 trusted tracks,
     plus a requirement of ≥2 authors and ≥5 albums in the holdout;
  3. for models allowed at this volume and not yet trained on THIS dataset
     version — train (split by album); with ≥2 authors, additionally
     train/evaluate transfer (split by author);
  4. write snapshots and append a row to the leaderboard;
  5. print a summary: what was trained, what beats the baseline, what waits for volume.

  python3 scripts/ml_auto.py             # single pass
  python3 scripts/ml_auto.py --interval 1800   # loop, every 30 minutes
  python3 scripts/ml_auto.py --force     # retrain everything on the current version
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

# Levels are tied to model thresholds (rnext/ml/models.MIN_N), not eyeballed:
#   ridge/knn/conformal — from 150 in fit ≈ 300 tracks total
#   hgb/hgbq           — from 400 in fit ≈ 760 tracks total
#   cross-author validation makes sense from ~300 tracks PER AUTHOR
LEVELS = [(1, 300, "M1: target regression (ridge/knn)"),
          (2, 760, "M2: boosting (hgb)"),
          (3, 1500, "M3: cross-author validation")]
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
        print(f"  [{base}/{holdout_mode}] skipping split: {e}")
        return None
    if len(train_idx) < Z.MIN_N.get(base, 1) or len(hold_idx) < 5:
        print(f"  [{base}/{holdout_mode}] too little data: train {len(train_idx)}, hold {len(hold_idx)}")
        return None

    Xtr, ytr = table.X[train_idx], table.y[train_idx]
    Xho, yho = table.X[hold_idx], table.y[hold_idx]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    n_cal = max(1, int(0.25 * len(perm)))
    cal, fit = perm[:n_cal], perm[n_cal:]
    if len(fit) < Z.MIN_N.get(base, 1):
        print(f"  [{base}] after the calibration split {len(fit)} tracks remain "
              f"(need {Z.MIN_N.get(base)}) — skipping")
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
    elif base == "hgbq":
        lo, hi = model.interval(Xho, level=0.8)
        stats["coverage_80"] = M.interval_coverage(yho, lo, hi)
    stats["vs_noop"] = M.compare_to_baseline(stats, "noop")
    stats["vs_v0table"] = M.compare_to_baseline(stats, "v0table")
    stats["gate"] = M.auto_gate(yho, pred, Xho[:, table.feature_names.index("dr")])
    Xf = Xtr[fit]
    fstats = {}
    for i, name in enumerate(table.feature_names):
        col = Xf[:, i]
        col = col[~np.isnan(col)]
        if col.size:
            fstats[name] = {"mean": round(float(np.mean(col)), 4),
                            "std": round(float(np.std(col)) or 1.0, 4)}
    return {"model": model, "metrics": stats, "pred": pred, "y_hold": yho,
            "feature_stats": fstats,
            "n_train": int(len(fit)), "n_cal": int(n_cal),
            "n_hold": int(len(hold_idx)), "holdout_mode": holdout_mode}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=0, help="loop interval, seconds (0 = single pass)")
    ap.add_argument("--force", action="store_true", help="retrain on the current version")
    ap.add_argument("--target", default="dcrest")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--audio-max-pairs", type=int, default=3,
                    help="how many pairs to run through the audio eval (0 = none)")
    ap.add_argument("--audio-window", default="02:00-07:00",
                    help="window when the audio eval runs at full scale (local time)")
    ap.add_argument("--with-audio", action="store_true",
                    help="in addition to training — end-to-end audio eval on pairs with raw sources")
    args = ap.parse_args()
    while True:
        run_once(args)
        if args.interval <= 0:
            return 0
        time.sleep(args.interval)


def in_window(spec: str) -> bool:
    """'02:00-07:00' → is the current time inside the window?"""
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
    print(f"[{time.strftime('%H:%M:%S')}] version {manifest['version']}: "
          f"pairs {manifest['pairs']}, trusted tracks {trusted}, level M{lvl}")

    table = ds.build_table(target=args.target)
    if len(table) == 0:
        print("  table is empty — waiting for data")
        return
    n_authors = len(set(table.authors.tolist()))
    n_albums = len(set(table.groups.tolist()))
    print(f"  tracks {len(table)}, authors {n_authors}, albums {n_albums}")

    todo = [m for m in Z.suggest_models(len(table), n_authors)
            if args.force or not reg.already_trained(manifest["version"], m.replace("conformal", "ridge"),
                                                      args.target)]
    # conformal is built on top of ridge/hgb — train it separately on top of the best base
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
                "features": table.feature_names,
                "feature_stats": res.get("feature_stats"),
                "n_train": res["n_train"],
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
            print(f"  ✓ {base}/{mode}: median|err| {res['metrics']['all']['median_abs']:.2f} "
                  f"(noop {res['metrics']['baseline_noop']['median_abs']:.2f}, "
                  f"const46 {res['metrics']['baseline_const46']['median_abs']:.2f}, "
                  f"v0 {res['metrics']['baseline_v0table']['median_abs']:.2f}) | "
                  f"gate: {g['improvement_dB']:+.2f} dB [{g['ci_low']:+.2f},{g['ci_high']:+.2f}] "
                  f"{'PROMOTE' if res['metrics']['gate']['promote'] else 'keeping the previous one'}")

    if not done:
        print("  no new models (everything already trained on this version)")

    volume_checkpoints(table, manifest, args)

    nxt = next(((n, need, name) for n, need, name in LEVELS if trusted < need), None)
    if nxt:
        print(f"  until M{nxt[0]} ({nxt[2]}): {nxt[1] - trusted} more trusted tracks")
    else:
        print("  all volume levels reached — from here on it's only quality and listening")
    if args.with_audio:
        run_audio_eval(args, manifest)
    print(leaderboard_text())


def run_audio_eval(args, manifest) -> None:
    """End-to-end eval: our run on the pair's original against the master.

    Runs only on pairs whose raw sources are still kept (--keep-raw at
    extraction time). The result is attached to the dataset version.
    """
    import subprocess
    root = Path(__file__).resolve().parents[1]
    n = args.audio_max_pairs
    if n <= 0:
        return
    night = in_window(args.audio_window)
    if not night:
        n = 1                     # daytime — minimum, full run in the window
        print(f"  audio eval: only {n} pair during the day (full scale in the window {args.audio_window})")
    out = subprocess.run([sys.executable, str(root / "scripts/dataset_eval.py"),
                          "--all", "--limit", str(n), "--max-tracks", "2",
                          "--variants", "flat,class",
                          "--summary-out", str(reg.dataset_dir() / "audio_eval_summary.json")],
                         capture_output=True, text=True, cwd=str(root))
    tail = (out.stdout or "").strip().splitlines()[-6:]
    print("  audio eval:")
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
    """Model snapshot every +100 trusted tracks: a "quality vs volume" curve.

    These are not production models but growth markers: they show where data
    stops paying off (so traffic is not wasted).
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
        print(f"  volume curve: {base} at {trusted} tracks → median|err| "
              f"{row['median_abs']:.2f} (v0 {row['baseline_v0table']:.2f})")


def leaderboard_text(limit: int = 8) -> str:
    if not LEADERBOARD.is_file():
        return "  leaderboard is empty"
    rows = [json.loads(l) for l in LEADERBOARD.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: (r["holdout_mode"], r["median_abs"]))
    out = ["  leaderboard (median|err|, dB):"]
    for r in rows[:limit]:
        out.append(f"    {r['holdout_mode']:<9} {r['model']:<10} ds={r['dataset_version']} "
                   f"n={r['n_train']:<4} {r['median_abs']:.2f} "
                   f"(const46 {r['baseline_const46']:.2f}, v0 {r['baseline_v0table']:.2f})")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
