"""Target quality metrics: error computation and comparison against baselines.

Metrics are intentionally distributional, not averaged: the master's
within-album spread is sigma ~ 0.8 dB, so a "mean error" hides the tails.

Baselines (without them the error numbers are meaningless):
  * noop    — do nothing (delta = 0);
  * const46 — current pipeline default (4.6 for anything loud);
  * v0table — per-class table from corpus/targets_v0.json;
  * master  — the "ideal" (master delta-crest taken as the prediction: err = 0).
"""
import json
import statistics as S
from pathlib import Path

import numpy as np

from rnext import workspace
from rnext.ml.dataset import class_of


def v0_table() -> dict[str, float]:
    p = workspace.dir_of("corpus") / "targets_v0.json"
    if not p.is_file():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {k.replace("target_dcrest_", ""): v for k, v in d.get("recommendation", {}).items()}


def baselines(dr_values: np.ndarray) -> dict[str, np.ndarray]:
    t = v0_table()
    cls = np.array([class_of(v) for v in dr_values])
    return {
        "noop": np.zeros(len(dr_values)),
        "const46": np.full(len(dr_values), 4.6),
        "v0table": np.array([t.get(c, 4.6) for c in cls]),
    }


def error_stats(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {
        "n": int(len(a)),
        "median_abs": round(float(np.median(a)), 3),
        "mean_abs": round(float(np.mean(a)), 3),
        "p90_abs": round(float(np.quantile(a, 0.9)), 3),
        "share_le_0.5": round(float(np.mean(a <= 0.5)), 3),
        "share_le_1.0": round(float(np.mean(a <= 1.0)), 3),
        "bias": round(float(np.median(err)), 3),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, dr_values: np.ndarray,
             classes: np.ndarray | None = None) -> dict:
    err = np.asarray(y_pred) - np.asarray(y_true)
    out = {"all": error_stats(err), "by_class": {}}
    cls = classes if classes is not None else np.array([class_of(v) for v in dr_values])
    for c in sorted(set(cls.tolist())):
        m = cls == c
        out["by_class"][c] = error_stats(err[m])
    for name, base_pred in baselines(dr_values).items():
        out[f"baseline_{name}"] = error_stats(np.asarray(base_pred) - np.asarray(y_true))
    return out


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return round(float(np.mean((y_true >= lo) & (y_true <= hi))), 3)


def compare_to_baseline(stats: dict, baseline: str = "noop") -> float:
    """How many times better the model is than a baseline by median error (<1 is better)."""
    b = stats.get(f"baseline_{baseline}", {}).get("median_abs")
    m = stats["all"]["median_abs"]
    if not b or not m:
        return float("nan")
    return round(m / b, 3)


def significance(y_true, y_pred, base_pred, n_boot: int = 2000, seed: int = 7) -> dict:
    """Paired bootstrap: is the model significantly better than the baseline by median |err|.

    "10% by eye" is a bad threshold: at n~150 the standard error of the median
    is ~0.08 dB, so 10% of 0.5 dB sits inside the noise. Hence we decide by the
    absolute gain AND by the CI of the difference.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    e_m = np.abs(np.asarray(y_pred) - y_true)
    e_b = np.abs(np.asarray(base_pred) - y_true)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(float(np.median(e_b[idx]) - np.median(e_m[idx])))
    diffs = np.asarray(diffs)
    return {
        "improvement_dB": round(float(np.median(e_b) - np.median(e_m)), 3),
        "ci_low": round(float(np.quantile(diffs, 0.025)), 3),
        "ci_high": round(float(np.quantile(diffs, 0.975)), 3),
        "p_win": round(float(np.mean(diffs > 0)), 3),
        "n": int(n),
    }


def auto_gate(y_true, y_pred, dr_values, min_gain_db: float = 0.10) -> dict:
    """Decision "promote the model to production": gain over the BEST baseline.

    Criterion: absolute gain >= min_gain_db AND the lower bound of the 95% CI > 0
    (i.e. the improvement is not explained by sample spread).
    """
    bases = baselines(dr_values)
    out = {"min_gain_db": min_gain_db, "by_baseline": {}}
    scored = []
    for name, pred in bases.items():
        st = significance(y_true, y_pred, pred)
        st["median_abs"] = round(float(np.median(np.abs(np.asarray(pred) - y_true))), 3)
        st["pass"] = bool(st["improvement_dB"] >= min_gain_db and st["ci_low"] > 0)
        out["by_baseline"][name] = st
        scored.append((name, st))
    # compare against the STRONGEST baseline (lowest median error), otherwise the
    # gate would pass a model that beats a weak baseline but loses to a strong one
    best = min(scored, key=lambda kv: kv[1]["median_abs"])
    out["best_baseline"] = best[0]
    out["vs_best"] = best[1]
    out["promote"] = bool(best[1]["improvement_dB"] >= min_gain_db and best[1]["ci_low"] > 0)
    return out


def format_report(stats: dict) -> str:
    a = stats["all"]
    lines = [f"n={a['n']} median|err|={a['median_abs']:.2f} share<=0.5={a['share_le_0.5']:.0%}"
             f" share<=1.0={a['share_le_1.0']:.0%} bias={a['bias']:+.2f}"]
    for c, s in stats["by_class"].items():
        lines.append(f"   {c:<8} n={s['n']:<4} median={s['median_abs']:.2f} "
                     f"<=0.5={s['share_le_0.5']:.0%} bias={s['bias']:+.2f}")
    for k in ("noop", "const46", "v0table"):
        b = stats.get(f"baseline_{k}")
        if b:
            lines.append(f"   base {k:<8} median={b['median_abs']:.2f} <=0.5={b['share_le_0.5']:.0%}")
    return "\n".join(lines)
