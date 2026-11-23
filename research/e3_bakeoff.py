#!/usr/bin/env python3
"""E3 hard-declipping bake-off runner and statistical report generator.

    python3 research/e3_bakeoff.py --manifest corpus/dev-e3-v1.json \
        --audio-root corpus/audio/dev-e3 --out out/e3-dev \
        --check-against corpus/sealed-v1.json

Compares on identical stereo inputs under one unified gain/delay/ceiling policy:
  - identity
  - gain_only (level-only headroom attenuation without repair)
  - current_sparse (frozen current production snapshot)
  - aspade_mit (MIT first-principles Analysis SPADE, 2x redundant Parseval DFT)
  - ss_pew_mit (MIT first-principles psychoacoustically-weighted IRL1 synthesis)
  - reg_ar_mit (MIT first-principles regularized AR / Janssen declipper)
  - neural_colab (explicit `not_run` record unless Colab auth/rights confirmed)

Separates oracle-mask evaluation from blind-detector evaluation (`v1_current`
vs `v2_windowed`), accounts for every failure/abstention as identity in
all-input effects, computes 95% paired cluster CIs with Holm-Bonferroni
correction, precision/recall, negative-control FPR, risk-coverage, worst cases,
and end-to-end CPU p50/p95 wall time and peak RSS.
"""
import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import corpus_synth, corpus_v1
from research import declip_metrics as M
from research.run_benchmark import _clean_error, _decode, _git_commit, _safe_argv
from rnext.dsp.clip_detector import detect_hard_clip_v2, score_hard_clip_v1
from rnext.dsp.contour import true_peak_dbfs
from rnext.dsp.declip_methods import (
    apply_headroom_policy,
    declip_aspade,
    declip_regularized_ar,
    declip_ss_pew,
)
from rnext.dsp.predictive import periodic_ar_proposal
from rnext.dsp.repair import select_restoration
from rnext.dsp.sparse import ReconstructionRejected, repair_clipping

SOLVER_METHODS = ("identity", "gain_only", "current_sparse", "aspade_mit",
                  "ss_pew_mit", "reg_ar_mit")
REPAIR_SOLVERS = ("current_sparse", "aspade_mit", "ss_pew_mit", "reg_ar_mit")
DETECTORS = ("v1_current", "v2_windowed")


def _implementation_hashes() -> dict[str, str]:
    files = [
        REPO / "research" / n for n in (
            "e3_bakeoff.py", "declip_metrics.py", "run_benchmark.py",
            "corpus_v1.py", "corpus_synth.py", "bench_declipping.py",
            "build_e3_corpus.py",
        )
    ]
    files += [
        REPO / "rnext/dsp" / (n + ".py") for n in (
            "sparse", "predictive", "repair", "selection", "contour",
            "declip", "clip_detector", "declip_methods",
        )
    ]
    files.append(REPO / "rnext/features.py")
    return {str(p.relative_to(REPO)): corpus_v1.digest(p) for p in files if p.exists()}


def _run_current_sparse_with_mask(x: np.ndarray, sr: int, mask: np.ndarray,
                                  ceiling_db: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Current snapshot solver core given an explicit mask (no dynamics fallback)."""
    if not mask.any():
        return x.copy(), x.copy(), {"status": "kept_original",
                                    "reason": "empty_mask",
                                    "method": "current_sparse",
                                    "output_gain_db": 0.0}
    frame = max(1024, min(16384, 2 ** round(np.log2(4096 * sr / 44100.0))))
    proposed, pred = periodic_ar_proposal(x, mask)
    if pred["accepted"]:
        raw, info = proposed, pred
    else:
        try:
            raw, info = repair_clipping(x, sr, mask, frame=frame)
        except ReconstructionRejected as exc:
            return x.copy(), x.copy(), {"status": "qc_fallback_original",
                                        "reason": str(exc),
                                        "method": "current_sparse",
                                        "output_gain_db": 0.0}
    delivered, head = apply_headroom_policy(raw, x, sr, mask, ceiling_db=ceiling_db)
    return delivered, raw, {**head, "solver_method": info.get("method", "current_sparse")}


def _run_solver(method: str, x: np.ndarray, sr: int, mask: np.ndarray | None,
                ceiling_db: float) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Run one solver method on stereo PCM given a mask (for repair solvers)."""
    if method == "identity":
        return x.copy(), x.copy(), {"status": "kept_original", "output_gain_db": 0.0}
    if method == "gain_only":
        # Same true-peak headroom ceiling policy (-1.0 dBTP) as repair arms, with zero repair.
        raw_tp = true_peak_dbfs(x, sr)
        gain_db = min(0.0, float(ceiling_db) - raw_tp)
        out = (x.astype(np.float64) * (10.0 ** (gain_db / 20.0))).astype(x.dtype)
        # Pre-gain waveform is unmodified input `x` (static attenuation is not a repair).
        return out, x.copy(), {"status": "gain_only",
                               "pre_gain_tp_dbfs": float(raw_tp),
                               "output_gain_db": float(gain_db)}
    if mask is None or not mask.any():
        return x.copy(), x.copy(), {"status": "kept_original",
                                    "reason": "empty_mask",
                                    "output_gain_db": 0.0}
    if method == "current_sparse":
        return _run_current_sparse_with_mask(x, sr, mask, ceiling_db)
    if method == "aspade_mit":
        raw, info = declip_aspade(x, sr, mask)
    elif method == "ss_pew_mit":
        raw, info = declip_ss_pew(x, sr, mask)
    elif method == "reg_ar_mit":
        raw, info = declip_regularized_ar(x, sr, mask)
    else:
        raise ValueError(f"unknown solver method: {method}")
    delivered, head = apply_headroom_policy(raw, x, sr, mask, ceiling_db=ceiling_db)
    return delivered, raw, {**head, "solver_method": info.get("method", method)}


def _current_rss_mb() -> float:
    """Zero-overhead current resident set size in MiB (Linux /proc, fallback ru_maxrss)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _timed_eval(fn, clean, degraded, true_mask, audio_seconds):
    """Execute solver/pipeline callable while measuring wall time and RSS (zero overhead)."""
    rss_before = _current_rss_mb()
    t0 = time.monotonic()
    try:
        delivered, raw, meta = fn()
        wall_s = time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001 — benchmark runner records per-method failures instead of dropping cases
        wall_s = time.monotonic() - t0
        rss_after = _current_rss_mb()
        rss_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        # Failures count as identity in all-input metrics.
        ev = M.evaluate_waveform(clean, degraded, degraded, true_mask, raw_candidate=degraded)
        return {
            "status": "error",
            "error": _clean_error(str(exc), exc)[:300],
            "metrics": ev,
            "wall_seconds": float(wall_s),
            "wall_s_per_audio_min": float(wall_s * 60.0 / max(audio_seconds, 1e-9)),
            "rss_delta_mb": float(max(0.0, rss_after - rss_before)),
            "process_max_rss_mb": float(rss_max),
        }
    rss_after = _current_rss_mb()
    rss_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    ev = M.evaluate_waveform(clean, degraded, delivered, true_mask, raw_candidate=raw)
    return {
        "status": meta.get("status", "ok"),
        "reason": meta.get("reason"),
        "output_gain_db": float(meta.get("output_gain_db", 0.0)),
        "metrics": ev,
        "wall_seconds": float(wall_s),
        "wall_s_per_audio_min": float(wall_s * 60.0 / max(audio_seconds, 1e-9)),
        "rss_delta_mb": float(max(0.0, rss_after - rss_before)),
        "process_max_rss_mb": float(rss_max),
    }


def evaluate_case(doc: dict, root: Path, case: dict, methods: list[str],
                  allow_custodian: bool = False) -> dict:
    """Run oracle-mask and blind-detector arms on one stereo manifest case."""
    ceiling_db = float(doc.get("output_ceiling_dbfs", -1.0))
    row = {
        "id": case["id"],
        "split": case.get("split"),
        "profile": case.get("profile"),
        "rate": case.get("rate"),
        "channels": case.get("channels"),
        "cluster_id": (case.get("groups") or {}).get("take") or case["id"],
        "artist_id": (case.get("groups") or {}).get("artist") or case["id"],
        "input_sha256": case["degraded"]["sha256"],
        "oracle": {},
        "detectors": {},
        "blind": {},
    }
    degraded_path = corpus_v1.resolve(doc, root, case["degraded"],
                                      allow_custodian=allow_custodian)
    clean_path = corpus_v1.resolve(doc, root, case["clean"],
                                   allow_custodian=allow_custodian)
    mask_path = corpus_v1.resolve(doc, root, case["mask"],
                                  allow_custodian=allow_custodian)
    degraded, rate, info = _decode(degraded_path)
    clean, clean_rate, _ = _decode(clean_path)
    if clean_rate != rate or clean.shape != degraded.shape:
        raise ValueError("clean/degraded misaligned")
    true_mask = corpus_synth.decode_mask(mask_path.read_bytes())
    if true_mask.shape != degraded.shape:
        raise ValueError("mask/degraded shape mismatch")

    bits = int(info["bps"] or 16)
    audio_seconds = float(len(degraded) / rate)
    row["bits"] = bits
    row["audio_seconds"] = audio_seconds
    row["is_defect"] = bool(true_mask.any())
    row["true_masked_samples"] = int(true_mask.sum())
    row["true_masked_fraction"] = float(true_mask.mean())

    # 1. Oracle-mask arm (ground-truth mask passed directly to solvers).
    for m in methods:
        row["oracle"][m] = _timed_eval(
            lambda m=m: _run_solver(m, degraded, rate, true_mask, ceiling_db),
            clean, degraded, true_mask, audio_seconds,
        )

    # 2. Blind detectors: v1_current vs v2_windowed.
    det_masks = {}
    det_confs = {}
    for det_name in DETECTORS:
        t0 = time.monotonic()
        if det_name == "v1_current":
            pred_mask, diag = score_hard_clip_v1(degraded, bits=bits)
        else:
            pred_mask, diag = detect_hard_clip_v2(degraded, rate, bits=bits)
        det_s = time.monotonic() - t0
        det_eval = M.evaluate_detection(true_mask, pred_mask)
        det_masks[det_name] = pred_mask
        det_confs[det_name] = float(diag.get("track_confidence", 0.0))
        row["detectors"][det_name] = {
            **det_eval,
            "detected_case": bool(pred_mask.any()),
            "track_confidence": det_confs[det_name],
            "wall_seconds": float(det_s),
        }

    # 3. Blind end-to-end arms:
    #    - `current_snapshot_auto`: exact `select_restoration(..., mode=case["mode"])`
    #    - `<solver>__<detector>`: each repair solver fed each blind detector mask.
    def _run_current_snapshot():
        out, rep = select_restoration(degraded, rate, mode=case.get("mode", "auto"), bits=bits)
        gain_db = float((rep.get("repair") or {}).get("output_gain_db", 0.0))
        return out, None, {"status": rep["status"], "reason": rep.get("reason"),
                           "output_gain_db": gain_db}

    snap_entry = _timed_eval(_run_current_snapshot, clean, degraded, true_mask, audio_seconds)
    snap_entry["detector"] = "v1_current"
    snap_entry["track_confidence"] = det_confs["v1_current"]
    row["blind"]["current_snapshot_auto"] = snap_entry

    for det_name in DETECTORS:
        pm = det_masks[det_name]
        det_s = row["detectors"][det_name]["wall_seconds"]
        for m in methods:
            if m not in REPAIR_SOLVERS:
                continue
            key = f"{m}__{det_name}"
            entry = _timed_eval(
                lambda m=m, pm=pm: _run_solver(m, degraded, rate, pm, ceiling_db),
                clean, degraded, true_mask, audio_seconds,
            )
            # End-to-end blind wall time includes detection + solver + headroom.
            entry["wall_seconds"] = float(entry["wall_seconds"] + det_s)
            entry["wall_s_per_audio_min"] = float(
                entry["wall_seconds"] * 60.0 / max(audio_seconds, 1e-9))
            entry["detector"] = det_name
            entry["track_confidence"] = det_confs[det_name]
            row["blind"][key] = entry

    return row


def _aggregate_arm(rows: list[dict], arm: str, name: str) -> dict:
    """Summarize all-input, conditional, worst-case, and runtime metrics for one method."""
    defects = [r for r in rows if r["is_defect"]]
    controls = [r for r in rows if not r["is_defect"]]

    all_masked_sdr = []
    all_masked_delta = []
    all_sisdri = []
    cond_masked_delta = []
    cond_sisdri = []
    worst_entries = []
    wall_per_min = []
    rss_vals = []
    peak_mem_vals = []
    total_wall = 0.0
    n_processed = 0
    n_abstained = 0
    n_failed = 0
    polarity_violations_vals = []
    reliable_diff_vals = []

    for r in defects:
        e = r[arm][name]
        st = e["status"]
        m = e["metrics"]
        wall_per_min.append(e["wall_s_per_audio_min"])
        rss_vals.append(e["process_max_rss_mb"])
        peak_mem_vals.append(e["rss_delta_mb"])
        total_wall += e["wall_seconds"]
        if m.get("polarity_violations") is not None:
            polarity_violations_vals.append(int(m["polarity_violations"]))
        if m.get("max_reliable_diff") is not None:
            reliable_diff_vals.append(float(m["max_reliable_diff"]))

        if st == "processed_candidate":
            n_processed += 1
            d_sdr = m["masked_delta_sdr_db"]
            sisdri = m["whole_track_sisdri_db"]
            if d_sdr is not None:
                cond_masked_delta.append(float(d_sdr))
                all_masked_delta.append(float(d_sdr))
            if sisdri is not None:
                cond_sisdri.append(float(sisdri))
                all_sisdri.append(float(sisdri))
            if m["masked_sdr_gain_controlled_db"] is not None:
                all_masked_sdr.append(float(m["masked_sdr_gain_controlled_db"]))
        else:
            if st in ("error", "qc_fallback_original"):
                n_failed += 1
            else:
                n_abstained += 1
            # Identity fallback in all-input accounting.
            all_masked_delta.append(0.0)
            all_sisdri.append(0.0)
            id_sdr = r["oracle"]["identity"]["metrics"]["masked_sdr_gain_controlled_db"]
            if id_sdr is not None:
                all_masked_sdr.append(float(id_sdr))

        eff_delta = float(m["masked_delta_sdr_db"]) if (
            st == "processed_candidate" and m["masked_delta_sdr_db"] is not None) else 0.0
        eff_sisdri = float(m["whole_track_sisdri_db"]) if (
            st == "processed_candidate" and m["whole_track_sisdri_db"] is not None) else 0.0
        worst_entries.append({
            "id": r["id"], "profile": r["profile"], "status": st,
            "masked_delta_sdr_db": eff_delta, "whole_track_sisdri_db": eff_sisdri,
        })

    ctl_fp_ids = []
    for r in controls:
        e = r[arm][name]
        wall_per_min.append(e["wall_s_per_audio_min"])
        rss_vals.append(e["process_max_rss_mb"])
        peak_mem_vals.append(e["rss_delta_mb"])
        total_wall += e["wall_seconds"]
        if e["status"] == "processed_candidate":
            ctl_fp_ids.append(r["id"])

    worst_entries.sort(key=lambda item: item["masked_delta_sdr_db"])
    n_def = len(defects)
    n_ctl = len(controls)
    return {
        "n_defects": n_def,
        "n_controls": n_ctl,
        "n_processed_defects": n_processed,
        "n_abstained_defects": n_abstained,
        "n_failed_defects": n_failed,
        "coverage": float(n_processed / n_def) if n_def else 0.0,
        "control_fp_count": len(ctl_fp_ids),
        "control_fp_rate": float(len(ctl_fp_ids) / n_ctl) if n_ctl else 0.0,
        "control_fp_ids": ctl_fp_ids,
        "all_input_mean_masked_sdr_db": float(np.mean(all_masked_sdr)) if all_masked_sdr else None,
        "all_input_median_masked_sdr_db": float(np.median(all_masked_sdr)) if all_masked_sdr else None,
        "all_input_mean_masked_delta_sdr_db": float(np.mean(all_masked_delta)) if all_masked_delta else None,
        "all_input_median_masked_delta_sdr_db": float(np.median(all_masked_delta)) if all_masked_delta else None,
        "all_input_min_masked_delta_sdr_db": float(np.min(all_masked_delta)) if all_masked_delta else None,
        "all_input_mean_sisdri_db": float(np.mean(all_sisdri)) if all_sisdri else None,
        "all_input_median_sisdri_db": float(np.median(all_sisdri)) if all_sisdri else None,
        "conditional_mean_masked_delta_sdr_db": float(np.mean(cond_masked_delta)) if cond_masked_delta else None,
        "conditional_median_masked_delta_sdr_db": float(np.median(cond_masked_delta)) if cond_masked_delta else None,
        "conditional_mean_sisdri_db": float(np.mean(cond_sisdri)) if cond_sisdri else None,
        "polarity_violations_total": int(sum(polarity_violations_vals)) if polarity_violations_vals else None,
        "max_reliable_diff": float(max(reliable_diff_vals)) if reliable_diff_vals else None,
        "worst_cases": worst_entries[:3],
        "wall_seconds_total": float(total_wall),
        "wall_s_per_audio_min_p50": float(np.percentile(wall_per_min, 50)) if wall_per_min else None,
        "wall_s_per_audio_min_p95": float(np.percentile(wall_per_min, 95)) if wall_per_min else None,
        "rss_delta_mb_p95": float(np.percentile(peak_mem_vals, 95)) if peak_mem_vals else None,
        "process_max_rss_mb": float(max(rss_vals)) if rss_vals else None,
    }


def _cluster_metric_map(rows: list[dict], arm: str, name: str, metric: str) -> dict[str, float]:
    """Collapse per-case defect metric to cluster-level mean (all-input accounting)."""
    buckets: dict[str, list[float]] = {}
    for r in rows:
        if not r["is_defect"]:
            continue
        e = r[arm][name]
        if e["status"] == "processed_candidate" and e["metrics"].get(metric) is not None:
            val = float(e["metrics"][metric])
        elif name in ("identity", "gain_only"):
            v = e["metrics"].get(metric)
            val = 0.0 if v is None else float(v)
        else:
            # Abstention / failure counts as identity (0 dB delta).
            val = 0.0 if "delta" in metric or "sisdri" in metric else float(
                r["oracle"]["identity"]["metrics"].get(metric) or 0.0)
        buckets.setdefault(r["cluster_id"], []).append(val)
    return {cid: float(np.mean(vals)) for cid, vals in buckets.items()}


def _detector_summary(rows: list[dict], det_name: str) -> dict:
    tp = fp = fn = tn = 0
    precs, recs, f1s = [], [], []
    def_detected = 0
    ctl_fp_cases = []
    per_profile: dict[str, dict] = {}
    for r in rows:
        d = r["detectors"][det_name]
        tp += d["tp"]
        fp += d["fp"]
        fn += d["fn"]
        tn += d["tn"]
        prof = per_profile.setdefault(r["profile"], {"cases": 0, "detected": 0, "tp": 0, "fp": 0, "fn": 0})
        prof["cases"] += 1
        prof["detected"] += int(d["detected_case"])
        prof["tp"] += d["tp"]
        prof["fp"] += d["fp"]
        prof["fn"] += d["fn"]
        if r["is_defect"]:
            if d["detected_case"]:
                def_detected += 1
            if d["precision"] is not None:
                precs.append(d["precision"])
            if d["recall"] is not None:
                recs.append(d["recall"])
            if d["f1"] is not None:
                f1s.append(d["f1"])
        else:
            if d["detected_case"]:
                ctl_fp_cases.append(r["id"])

    n_def = sum(1 for r in rows if r["is_defect"])
    n_ctl = sum(1 for r in rows if not r["is_defect"])
    micro_p = float(tp / (tp + fp)) if (tp + fp) else None
    micro_r = float(tp / (tp + fn)) if (tp + fn) else None
    micro_f1 = (float(2 * micro_p * micro_r / (micro_p + micro_r))
                if (micro_p is not None and micro_r is not None and (micro_p + micro_r) > 0)
                else 0.0)
    return {
        "detector": det_name,
        "sample_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_defect_precision": float(np.mean(precs)) if precs else None,
        "macro_defect_recall": float(np.mean(recs)) if recs else None,
        "macro_defect_f1": float(np.mean(f1s)) if f1s else None,
        "case_defect_recall": float(def_detected / n_def) if n_def else 0.0,
        "cases_detected": def_detected,
        "n_defects": n_def,
        "control_fp_cases": len(ctl_fp_cases),
        "control_fp_rate": float(len(ctl_fp_cases) / n_ctl) if n_ctl else 0.0,
        "control_fp_ids": ctl_fp_cases,
        "per_profile": per_profile,
    }


def summarize_bakeoff(rows: list[dict], methods: list[str]) -> dict:
    """Compute full statistical summary, paired CIs, risk-coverage, and dev decision."""
    oracle_summary = {m: _aggregate_arm(rows, "oracle", m) for m in methods}
    blind_keys = ["current_snapshot_auto"] + [
        f"{m}__{d}" for d in DETECTORS for m in methods if m in REPAIR_SOLVERS
    ]
    blind_summary = {k: _aggregate_arm(rows, "blind", k) for k in blind_keys}
    detector_summary = {d: _detector_summary(rows, d) for d in DETECTORS}

    # Planned paired cluster-level 95% CIs on primary metric (masked_delta_sdr_db)
    # and secondary metric (whole_track_sisdri_db), with Holm-Bonferroni correction.
    planned_pairs = []
    for m in ("aspade_mit", "ss_pew_mit", "reg_ar_mit"):
        if m in methods:
            planned_pairs.append(("oracle", m, "identity"))
            planned_pairs.append(("oracle", m, "gain_only"))
            if "current_sparse" in methods:
                planned_pairs.append(("oracle", m, "current_sparse"))
            planned_pairs.append(("blind", f"{m}__v2_windowed", "current_snapshot_auto"))
    if "current_sparse" in methods:
        planned_pairs.append(("oracle", "current_sparse", "identity"))
        planned_pairs.append(("blind", "current_sparse__v2_windowed", "current_snapshot_auto"))

    paired_masked = {}
    paired_sisdri = {}
    p_masked = {}
    p_sisdri = {}
    for arm, a_name, b_name in planned_pairs:
        label = f"{arm}:{a_name}_vs_{b_name}"
        a_m = _cluster_metric_map(rows, arm, a_name, "masked_delta_sdr_db")
        b_m = _cluster_metric_map(rows, arm, b_name, "masked_delta_sdr_db")
        diffs_m = {k: a_m[k] - b_m[k] for k in sorted(set(a_m) & set(b_m))}
        ci_m = M.paired_cluster_ci(diffs_m)
        paired_masked[label] = ci_m
        p_masked[label] = ci_m["sign_flip_p_value"]

        a_s = _cluster_metric_map(rows, arm, a_name, "whole_track_sisdri_db")
        b_s = _cluster_metric_map(rows, arm, b_name, "whole_track_sisdri_db")
        diffs_s = {k: a_s[k] - b_s[k] for k in sorted(set(a_s) & set(b_s))}
        ci_s = M.paired_cluster_ci(diffs_s)
        paired_sisdri[label] = ci_s
        p_sisdri[label] = ci_s["sign_flip_p_value"]

    adj_m = M.holm_bonferroni(p_masked)
    adj_s = M.holm_bonferroni(p_sisdri)
    for label, ci_m in paired_masked.items():
        ci_m["holm_bonferroni_p"] = adj_m[label]
        paired_sisdri[label]["holm_bonferroni_p"] = adj_s[label]

    # Risk-coverage curves for blind arms.
    risk_coverage = {}
    for k in blind_keys:
        recs = []
        for r in rows:
            e = r["blind"][k]
            recs.append({
                "is_defect": r["is_defect"],
                "confidence": float(e.get("track_confidence", 0.0)),
                "attempted": e["status"] == "processed_candidate",
                "delta_sdr_db": e["metrics"].get("masked_delta_sdr_db"),
                "status": e["status"],
            })
        risk_coverage[k] = M.risk_coverage_curve(recs)

    # Exploratory dev selection gate: select best blind candidate satisfying
    # zero negative-control FP, zero polarity violations, zero failures, and
    # positive exploratory 95% cluster bootstrap CI over current_snapshot_auto.
    # Holm-Bonferroni p-values and Student-t CIs are reported alongside; formal
    # confirmatory gates are preregistered in E4 before opening sealed in E5.
    candidates = [k for k in blind_keys if k != "current_snapshot_auto"]
    eligible = []
    for k in candidates:
        s = blind_summary[k]
        ci_vs_snap = paired_masked.get(f"blind:{k}_vs_current_snapshot_auto")
        if (s["control_fp_count"] == 0
                and s["polarity_violations_total"] == 0
                and s["n_failed_defects"] == 0
                and (s["all_input_mean_masked_delta_sdr_db"] or 0.0) > 0.0
                and ci_vs_snap is not None
                and (ci_vs_snap["bootstrap_ci_low"] or 0.0) > 0.0):
            eligible.append((s["all_input_mean_masked_delta_sdr_db"],
                             s["all_input_mean_sisdri_db"] or 0.0, k))
    eligible.sort(reverse=True)
    if eligible:
        winner = eligible[0][2]
        ci_win = paired_masked.get(f"blind:{winner}_vs_current_snapshot_auto")
        decision = {
            "verdict": "select_dev_candidate_for_e4_freeze",
            "gate_basis": "exploratory_dev_bootstrap_ci_95",
            "selected_candidate": winner,
            "baseline": "current_snapshot_auto",
            "sealed_opened": False,
            "selected_paired_stats": {
                "bootstrap_ci_95": [ci_win["bootstrap_ci_low"], ci_win["bootstrap_ci_high"]],
                "student_t_ci_95": [ci_win["t_ci_low"], ci_win["t_ci_high"]],
                "sign_flip_p_value": ci_win["sign_flip_p_value"],
                "holm_bonferroni_p": ci_win["holm_bonferroni_p"],
            } if ci_win else None,
            "reason": (
                f"{winner} passes zero negative-control FP, zero polarity violations, "
                f"zero failures, and achieves positive exploratory 95% paired cluster "
                f"bootstrap CI over current_snapshot_auto on dev without touching sealed."
            ),
            "caveat": (
                "Exploratory dev selection only (unadjusted bootstrap CI gate; see reported "
                "Student-t CI and Holm-Bonferroni p) — NOT a release claim or perceptual "
                "preference claim. Requires E4 stereo/album safety + CPU freeze and E5 sealed + blind panel."
            ),
        }
    else:
        decision = {
            "verdict": "no_promote_on_dev",
            "gate_basis": "exploratory_dev_bootstrap_ci_95",
            "selected_candidate": None,
            "baseline": "current_snapshot_auto",
            "sealed_opened": False,
            "reason": "No blind candidate cleared zero-FP, zero-violation, and exploratory 95% bootstrap CI gates.",
            "caveat": "Keep current safe route; do not open sealed.",
        }

    neural_colab = {
        "arm": "neural_colab",
        "status": "not_run",
        "reason": "colab_connection_not_authorized_or_configured",
        "spec_reference": (
            "spec-v1 section 5: separate Colab authorization/setup step is required "
            "before GPU neural runs; local CPU bake-off proceeds independently"
        ),
    }

    return {
        "oracle": oracle_summary,
        "blind": blind_summary,
        "detectors": detector_summary,
        "paired_ci_masked_delta_sdr_db": paired_masked,
        "paired_ci_whole_track_sisdri_db": paired_sisdri,
        "risk_coverage": risk_coverage,
        "neural_colab": neural_colab,
        "dev_decision": decision,
    }


def render_markdown_report(protocol: dict, summary: dict, rows: list[dict]) -> str:
    """Render concise, publication-grade Markdown tables from bake-off summary."""
    lines = [
        f"# E3 Hard-Declipping Bake-Off Report (`{protocol['dataset_version']}`)",
        "",
        f"- **Manifest SHA-256:** `{protocol['manifest_sha256'][:16]}...`",
        f"- **Git commit:** `{protocol['toolchain'].get('git_commit')}`",
        (f"- **CPU:** `{protocol['toolchain']['cpu']['model']}` "
         f"({protocol['toolchain']['cpu']['logical_cores']} logical cores, "
         f"`OMP_NUM_THREADS={protocol['toolchain']['cpu']['numpy_threads']}`)"),
        f"- **Metric / Gain / Delay policy:** `{protocol['metric']}`",
        (f"- **Neural / Google Colab arm:** `{summary['neural_colab']['status']}` "
         f"(`{summary['neural_colab']['reason']}`)"),
        (f"- **Dev decision:** `{summary['dev_decision']['verdict']}` -> "
         f"`{summary['dev_decision']['selected_candidate']}` "
         f"(sealed opened: `{summary['dev_decision']['sealed_opened']}`)"),
        "",
        "## 1. Oracle-Mask Comparison (All-Input Defect Accounting)",
        "",
        "| Method | Mean Masked SDR (dB) | Mean ΔMasked SDR (dB) | Median ΔMasked SDR (dB) | Worst ΔMasked (dB) | Mean SI-SDRi (dB) | Polarity Viol | Wall p50 (s/min) | Wall p95 (s/min) | Peak RSS (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def _fmt(v, s="{:.2f}"):
        return "-" if v is None else s.format(v)

    for m, s in summary["oracle"].items():
        lines.append(
            f"| `{m}` | {_fmt(s['all_input_mean_masked_sdr_db'])} | "
            f"{_fmt(s['all_input_mean_masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_fmt(s['all_input_median_masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_fmt(s['all_input_min_masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_fmt(s['all_input_mean_sisdri_db'], '{:+.2f}')} | "
            f"{'-' if s['polarity_violations_total'] is None else s['polarity_violations_total']} | "
            f"{_fmt(s['wall_s_per_audio_min_p50'])} | "
            f"{_fmt(s['wall_s_per_audio_min_p95'])} | "
            f"{_fmt(s['process_max_rss_mb'], '{:.1f}')} |"
        )

    lines += [
        "",
        "## 2. Blind Detector Comparison",
        "",
        "| Detector | Sample Micro P | Sample Micro R | Sample Micro F1 | Macro Defect R | Case Recall | Control FP | Control FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dname, ds in summary["detectors"].items():
        lines.append(
            f"| `{dname}` | {_fmt(ds['micro_precision'], '{:.4f}')} | "
            f"{_fmt(ds['micro_recall'], '{:.4f}')} | "
            f"{_fmt(ds['micro_f1'], '{:.4f}')} | "
            f"{_fmt(ds['macro_defect_recall'], '{:.4f}')} | "
            f"{ds['cases_detected']}/{ds['n_defects']} ({100*ds['case_defect_recall']:.1f}%) | "
            f"{ds['control_fp_cases']} | {100*ds['control_fp_rate']:.1f}% |"
        )

    lines += [
        "",
        "## 3. End-to-End Blind Pipeline Comparison (Failures/Abstentions = Identity)",
        "",
        "| Blind Arm | Coverage | Fail | Control FP | All-Input Mean ΔMasked (dB) | Cond Mean ΔMasked (dB) | All-Input Mean SI-SDRi (dB) | Worst ΔMasked (dB) | Wall p95 (s/min) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, s in summary["blind"].items():
        lines.append(
            f"| `{k}` | {s['n_processed_defects']}/{s['n_defects']} ({100*s['coverage']:.1f}%) | "
            f"{s['n_failed_defects']} | {s['control_fp_count']} | "
            f"{_fmt(s['all_input_mean_masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_fmt(s['conditional_mean_masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_fmt(s['all_input_mean_sisdri_db'], '{:+.2f}')} | "
            f"{_fmt(s['all_input_min_masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_fmt(s['wall_s_per_audio_min_p95'])} |"
        )

    lines += [
        "",
        "## 4. Paired 95% Cluster Confidence Intervals (Masked ΔSDR, dB)",
        "",
        "| Comparison | n | Mean Diff (dB) | Median Diff (dB) | Bootstrap 95% CI | Student-t 95% CI | Worst Cluster (dB) | Sign-Flip p | Holm-Bonf p |",
        "|---|---:|---:|---:|---|---|---|---:|---:|",
    ]
    for label, ci in summary["paired_ci_masked_delta_sdr_db"].items():
        bci = f"[{_fmt(ci['bootstrap_ci_low'], '{:+.2f}')}, {_fmt(ci['bootstrap_ci_high'], '{:+.2f}')}]"
        tci = f"[{_fmt(ci['t_ci_low'], '{:+.2f}')}, {_fmt(ci['t_ci_high'], '{:+.2f}')}]"
        worst = f"`{ci['worst_cluster']}` ({_fmt(ci['min'], '{:+.2f}')})" if ci["worst_cluster"] else "-"
        lines.append(
            f"| `{label}` | {ci['n_clusters']} | {_fmt(ci['mean'], '{:+.2f}')} | "
            f"{_fmt(ci['median'], '{:+.2f}')} | {bci} | {tci} | {worst} | "
            f"{_fmt(ci['sign_flip_p_value'], '{:.4f}')} | {_fmt(ci['holm_bonferroni_p'], '{:.4f}')} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--operation", default="local_inference", choices=corpus_v1.OPERATIONS)
    ap.add_argument("--methods", default=",".join(SOLVER_METHODS))
    ap.add_argument("--strict-toolchain", action="store_true")
    ap.add_argument("--check-against", action="append", default=[], metavar="MANIFEST")
    args = ap.parse_args()

    doc = corpus_v1.load(args.manifest)
    root = Path(args.audio_root).resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in SOLVER_METHODS]
    if unknown:
        print(f"manifest error: unknown methods {unknown}", file=sys.stderr)
        return 2

    # E3 never opens sealed tiers.
    if doc.get("tier") == "sealed":
        print("manifest error: E3 bake-off refuses sealed tier", file=sys.stderr)
        return 2
    sealed = [c for c in doc.get("cases", []) if corpus_v1.is_sealed_case(doc, c)]
    view = corpus_v1.without_sealed(doc)

    # Rights checked BEFORE touching any file bytes.
    errors = []
    for case in view.get("cases", []):
        if not corpus_v1.case_allowed(case, args.operation):
            errors.append(f"case {case['id']}: operation '{args.operation}' not permitted")
    if errors:
        for e in errors:
            print(f"manifest error: {e}", file=sys.stderr)
        return 2

    if args.check_against:
        others = [corpus_v1.load(m) for m in args.check_against]
        errors += corpus_v1.validate_cross_manifest_leakage([view, *others])
    errors += corpus_v1.validate_manifest(view, check_hashes_root=root)

    try:
        detected = corpus_v1.actual_toolchain()
    except Exception as exc:  # noqa: BLE001 — CLI boundary reports toolchain probe failures as exit code 2
        detected, errors = {}, errors + [f"toolchain detection failed: {type(exc).__name__}"]
    for field, declared in (doc.get("toolchain") or {}).items():
        got = detected.get(field)
        if got is None:
            continue
        if args.strict_toolchain and got != declared:
            errors.append(f"toolchain.{field}: detected {got} != declared {declared}")
    if errors:
        for e in errors:
            print(f"manifest error: {e}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        ap.error("output directory must be empty")

    protocol = {
        "schema": "e3-bakeoff/v1",
        "manifest_sha256": corpus_v1.manifest_digest(doc),
        "dataset_version": doc["dataset_version"],
        "tier": doc["tier"],
        "excluded_sealed_count": len(sealed),
        "methods": methods,
        "detectors": list(DETECTORS),
        "output_ceiling_dbfs": float(doc.get("output_ceiling_dbfs", -1.0)),
        "rng": doc.get("rng"),
        "numeric_tolerance": doc.get("numeric_tolerance", 1e-6),
        "metric": (
            "gain-controlled masked SDR (single stereo whole-track scalar gain, delay=0, "
            "eps=1e-20) + uncompensated masked SDR + whole-track stereo SI-SDRi"
        ),
        "toolchain": {
            "declared": doc.get("toolchain", {}),
            "detected": detected,
            "implementation_sha256": _implementation_hashes(),
            "git_commit": _git_commit(),
            "cpu": {
                "model": platform.processor() or platform.machine(),
                "logical_cores": os.cpu_count(),
                "numpy_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            },
        },
        "argv": _safe_argv(),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")

    rows = []
    for case in sorted(view["cases"], key=lambda c: c["id"]):
        row = evaluate_case(view, root, case, methods, allow_custodian=False)
        rows.append(row)
        print(
            f"{case['id']}: defect={row['is_defect']} "
            f"v1_det={row['detectors']['v1_current']['detected_case']} "
            f"v2_det={row['detectors']['v2_windowed']['detected_case']}",
            flush=True,
        )

    summary = summarize_bakeoff(rows, methods)
    (out / "results.json").write_text(json.dumps(
        {"excluded_sealed_count": len(sealed), "cases": rows},
        indent=2, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "report.md").write_text(render_markdown_report(protocol, summary, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
