"""Single-owner fidelity, detection, and statistical metrics for declipping bake-offs.

Implements the exact metric contract of spec v1 section 4:
- Unified gain/delay policy: delay_samples = 0, one shared stereo scalar gain
  `gain_to_clean` across both channels, identical across all methods.
- Primary paired fidelity: gain-controlled masked SDR on the true clip mask,
  alongside uncompensated masked SDR, near-mask (seam) SDR, outside-mask SDR,
  and whole-track stereo SI-SDR / SI-SDRi.
- Explicit roll-off epsilon (1e-20) and exclusion reporting so JSON stays finite.
- Sample-level detection precision/recall/F1, case-level coverage and negative-
  control false-positive rates, and risk-coverage curves with all failures and
  abstentions accounted as identity in all-input effects.
- Cluster-level 95% paired bootstrap and Student-t CIs with Holm-Bonferroni
  multiplicity adjustment across pre-planned comparisons.
"""
import math

import numpy as np
from scipy import stats

from research.bench_declipping import _sdr, reconstruction_metrics

EPS = 1e-20
NEAR_MASK_RADIUS = 32


def _dilate_mask_1d(mask: np.ndarray, radius: int) -> np.ndarray:
    """Boolean dilation along time axis (axis=0) for seam collar evaluation."""
    if not mask.any() or radius <= 0:
        return np.zeros_like(mask, dtype=bool)
    kernel = np.ones(2 * radius + 1, dtype=np.int32)
    out = np.zeros_like(mask, dtype=bool)
    for ch in range(mask.shape[1]):
        conv = np.convolve(mask[:, ch].astype(np.int32), kernel, mode="same")
        out[:, ch] = conv > 0
    return out


def si_sdr_db(clean: np.ndarray, candidate: np.ndarray, eps: float = EPS) -> float | None:
    """Whole-track stereo Scale-Invariant SDR (dB), no time shift, zero mean not removed."""
    s = clean.astype(np.float64)
    y = candidate.astype(np.float64)
    s_energy = float(np.sum(s ** 2))
    if s_energy < eps:
        return None
    alpha = float(np.sum(y * s)) / s_energy
    s_target = alpha * s
    e_noise = y - s_target
    t_energy = float(np.sum(s_target ** 2))
    n_energy = float(np.sum(e_noise ** 2))
    if n_energy < eps:
        return None
    return float(10.0 * np.log10(max(t_energy, eps) / n_energy))


def evaluate_waveform(
    clean: np.ndarray,
    degraded: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray | None,
    *,
    raw_candidate: np.ndarray | None = None,
    eps: float = EPS,
    near_radius: int = NEAR_MASK_RADIUS,
) -> dict:
    """Evaluate one candidate against clean/degraded under the unified policy.

    Returns finite floats or None (with explicit `exclusions` entries).
    """
    c = clean.astype(np.float64)
    d = degraded.astype(np.float64)
    y = candidate.astype(np.float64)
    if c.shape != d.shape or c.shape != y.shape:
        raise ValueError("clean, degraded, and candidate shapes must match")

    exclusions = []
    whole = reconstruction_metrics(c, y)
    deg_whole = reconstruction_metrics(c, d)
    gain = float(whole["gain_to_clean"])
    y_aligned = gain * y

    sisdr_cand = si_sdr_db(c, y, eps=eps)
    sisdr_deg = si_sdr_db(c, d, eps=eps)
    if sisdr_cand is None or sisdr_deg is None:
        sisdri = None
        exclusions.append("whole_track_sisdri_undefined_near_zero_error")
    else:
        sisdri = float(sisdr_cand - sisdr_deg)

    out = {
        "delay_samples": 0,
        "epsilon": eps,
        "gain_to_clean": gain,
        "whole_track_fixed_gain_sdr_db": whole["fixed_gain_sdr_db"],
        "whole_track_scale_aligned_sdr_db": whole["scale_aligned_sdr_db"],
        "whole_track_sisdr_db": sisdr_cand,
        "whole_track_sisdri_db": sisdri,
        "masked_sdr_gain_controlled_db": None,
        "masked_sdr_uncompensated_db": None,
        "masked_delta_sdr_db": None,
        "near_mask_sdr_db": None,
        "outside_mask_sdr_db": None,
        "polarity_violations": None,
        "max_reliable_diff": None,
    }

    if mask is None or not mask.any():
        exclusions.append("empty_true_mask")
        out["outside_mask_sdr_db"] = _sdr(
            float(np.sum(c ** 2)), float(np.sum((c - y_aligned) ** 2)))
        if raw_candidate is not None:
            out["polarity_violations"] = 0
            out["max_reliable_diff"] = float(np.max(np.abs(raw_candidate.astype(np.float64) - d)))
        else:
            exclusions.append("pre_gain_candidate_unavailable")
        out["exclusions"] = exclusions
        return out

    m = mask.astype(bool)
    sig_m = float(np.sum(c[m] ** 2))
    err_gc = float(np.sum((c[m] - y_aligned[m]) ** 2))
    err_unc = float(np.sum((c[m] - y[m]) ** 2))
    deg_gain = float(deg_whole["gain_to_clean"])
    err_deg_gc = float(np.sum((c[m] - (deg_gain * d)[m]) ** 2))

    sdr_gc = _sdr(sig_m, err_gc)
    sdr_unc = _sdr(sig_m, err_unc)
    sdr_deg_gc = _sdr(sig_m, err_deg_gc)
    out["masked_sdr_gain_controlled_db"] = sdr_gc
    out["masked_sdr_uncompensated_db"] = sdr_unc
    if sdr_gc is not None and sdr_deg_gc is not None:
        out["masked_delta_sdr_db"] = float(sdr_gc - sdr_deg_gc)
    else:
        exclusions.append("masked_delta_sdr_undefined_near_zero_error")

    near = _dilate_mask_1d(m, near_radius) & (~m)
    if near.any():
        out["near_mask_sdr_db"] = _sdr(
            float(np.sum(c[near] ** 2)), float(np.sum((c[near] - y_aligned[near]) ** 2)))
    outside = ~m
    if outside.any():
        out["outside_mask_sdr_db"] = _sdr(
            float(np.sum(c[outside] ** 2)), float(np.sum((c[outside] - y_aligned[outside]) ** 2)))

    # Pre-gain invariants are only defined when the pre-gain candidate PCM is supplied.
    # Measuring them on a gain-aligned post-gain output creates false violations.
    if raw_candidate is None:
        exclusions.append("pre_gain_candidate_unavailable")
    else:
        check = raw_candidate.astype(np.float64)
        out["max_reliable_diff"] = float(np.max(np.abs(check[outside] - d[outside]))) if outside.any() else 0.0
        pos = m & (d > 0.0)
        neg = m & (d < 0.0)
        viol = int(np.count_nonzero(check[pos] < d[pos] - 1e-7)
                   + np.count_nonzero(check[neg] > d[neg] + 1e-7))
        out["polarity_violations"] = viol
    out["exclusions"] = exclusions
    return out


def evaluate_detection(true_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    """Sample-level confusion counts and precision/recall/F1."""
    if true_mask.shape != pred_mask.shape:
        raise ValueError("true_mask and pred_mask shapes must match")
    t = true_mask.astype(bool)
    p = pred_mask.astype(bool)
    tp = int(np.count_nonzero(t & p))
    fp = int(np.count_nonzero((~t) & p))
    fn = int(np.count_nonzero(t & (~p)))
    tn = int(np.count_nonzero((~t) & (~p)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0 if (tp + fp + fn) > 0 else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "true_count": tp + fn, "pred_count": tp + fp,
        "precision": precision, "recall": recall, "f1": f1,
    }


def paired_cluster_ci(
    cluster_diffs: dict[str, float],
    *,
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 20261123,
) -> dict:
    """Paired cluster-level 95% bootstrap + Student-t CI and sign-flip p-value.

    `cluster_diffs` maps each independent cluster id (e.g. take/artist) to the
    paired difference (method_A - method_B). Multiple slices from one cluster
    must be collapsed before calling this function.
    """
    items = [(k, float(v)) for k, v in sorted(cluster_diffs.items()) if v is not None and np.isfinite(v)]
    n = len(items)
    if n == 0:
        return {"n_clusters": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None, "worst_cluster": None,
                "bootstrap_ci_low": None, "bootstrap_ci_high": None,
                "t_ci_low": None, "t_ci_high": None, "sign_flip_p_value": None}

    vals = np.array([v for _, v in items], dtype=np.float64)
    worst_idx = int(np.argmin(vals))
    mean_v = float(np.mean(vals))
    med_v = float(np.median(vals))
    std_v = float(np.std(vals, ddof=1)) if n > 1 else 0.0

    alpha = 1.0 - confidence
    if n == 1:
        boot_lo = boot_hi = mean_v
        t_lo = t_hi = None
        p_val = 1.0
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = np.mean(vals[idx], axis=1)
        boot_lo = float(np.quantile(boot_means, alpha / 2.0))
        boot_hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
        se = std_v / math.sqrt(n)
        if se == 0.0:
            t_lo = t_hi = mean_v
        else:
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
            t_lo = float(mean_v - t_crit * se)
            t_hi = float(mean_v + t_crit * se)
        # Exact sign-flip permutation test for n <= 16, Monte Carlo otherwise.
        obs_abs = abs(mean_v)
        if n <= 16:
            bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) * 2 - 1
            perm_means = np.abs(bits @ vals) / n
            p_val = float(np.mean(perm_means >= obs_abs - 1e-12))
        else:
            signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
            perm_means = np.abs(signs @ vals) / n
            p_val = float((1 + np.count_nonzero(perm_means >= obs_abs - 1e-12)) / (n_boot + 1))

    return {
        "n_clusters": n,
        "mean": mean_v,
        "median": med_v,
        "std": std_v,
        "min": float(vals[worst_idx]),
        "max": float(np.max(vals)),
        "worst_cluster": items[worst_idx][0],
        "bootstrap_ci_low": boot_lo,
        "bootstrap_ci_high": boot_hi,
        "t_ci_low": t_lo,
        "t_ci_high": t_hi,
        "sign_flip_p_value": p_val,
    }


def holm_bonferroni(p_values: dict[str, float | None]) -> dict[str, float | None]:
    """Step-down Holm-Bonferroni adjusted p-values for planned comparisons."""
    valid = [(k, float(p)) for k, p in p_values.items() if p is not None and np.isfinite(p)]
    m = len(valid)
    adjusted = {k: None for k in p_values}
    if m == 0:
        return adjusted
    valid.sort(key=lambda kv: kv[1])
    running_max = 0.0
    for rank, (key, p) in enumerate(valid):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[key] = float(running_max)
    return adjusted


def risk_coverage_curve(case_records: list[dict]) -> dict:
    """Compute risk-coverage curve over detector confidence thresholds.

    Each record must contain:
      - `is_defect`: bool
      - `confidence`: float in [0, 1]
      - `attempted`: bool (whether the method produced a candidate if threshold passed)
      - `delta_sdr_db`: float | None (masked delta SDR when attempted, 0.0 when abstained/failed)
      - `status`: str (`processed_candidate` / `kept_original` / `qc_fallback_original` / `error`)
    """
    defects = [r for r in case_records if r["is_defect"]]
    controls = [r for r in case_records if not r["is_defect"]]
    n_def = len(defects)
    n_ctl = len(controls)
    if n_def == 0:
        return {"n_defects": 0, "n_controls": n_ctl, "points": [], "aurc_regression": None}

    confs = sorted({0.0, 1.0, *(round(float(r["confidence"]), 6) for r in case_records)},
                   reverse=True)
    points = []
    for thr in confs:
        sel_def = [r for r in defects if r["confidence"] >= thr and r["attempted"]]
        sel_ctl = [r for r in controls if r["confidence"] >= thr and r["attempted"]]
        cov = float(len(sel_def) / n_def)
        ctl_fpr = float(len(sel_ctl) / n_ctl) if n_ctl > 0 else 0.0
        # All-input effect counts abstentions and failures as 0.0 dB (identity).
        all_deltas = [
            float(r["delta_sdr_db"])
            if (r["confidence"] >= thr and r["attempted"] and r["status"] == "processed_candidate"
                and r["delta_sdr_db"] is not None)
            else 0.0
            for r in defects
        ]
        cond_deltas = [
            float(r["delta_sdr_db"])
            for r in sel_def
            if r["status"] == "processed_candidate" and r["delta_sdr_db"] is not None
        ]
        n_reg = sum(1 for v in cond_deltas if v < 0.0)
        reg_risk = float(n_reg / len(sel_def)) if sel_def else 0.0
        points.append({
            "threshold": float(thr),
            "coverage": cov,
            "n_attempted_defects": len(sel_def),
            "control_fp_count": len(sel_ctl),
            "control_fpr": ctl_fpr,
            "selective_regression_risk": reg_risk,
            "all_input_mean_delta_sdr_db": float(np.mean(all_deltas)),
            "conditional_mean_delta_sdr_db": float(np.mean(cond_deltas)) if cond_deltas else None,
        })

    # Area under selective regression risk vs coverage curve (trapezoidal).
    pts_sorted = sorted(points, key=lambda p: p["coverage"])
    covs = np.array([p["coverage"] for p in pts_sorted], dtype=np.float64)
    risks = np.array([p["selective_regression_risk"] for p in pts_sorted], dtype=np.float64)
    aurc = float(np.trapezoid(risks, covs)) if hasattr(np, "trapezoid") else float(np.trapz(risks, covs))
    return {
        "n_defects": n_def,
        "n_controls": n_ctl,
        "points": points,
        "aurc_regression": aurc,
    }
