#!/usr/bin/env python3
"""E5 Single-Use Sealed Holdout Evaluator & Confirmatory Hypothesis Report Builder.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 research/e5_sealed_eval.py \
        --manifest corpus/sealed-e5-v1.json \
        --confirm-unseal-once \
        --custodian-release-token OWNER-E5-2026-11-23 \
        --custodian-reason "Owner interactive approval 2026-11-23 for Board #1129 E5" \
        --results-json docs/validation/e5-sealed-results.json \
        --report-md docs/validation/e5-sealed-report.md

Hard scientific invariants enforced BEFORE any sealed audio is opened:
  1. Requires BOTH `--confirm-unseal-once` AND non-empty `--custodian-release-token`.
  2. Verifies `docs/validation/e4-freeze-manifest.json` canonical SHA-256 equals
     the locked E4 `freeze_sha256` (`7c4847d4d572321cb93b4d126dfde009e832890d6b652700047287fb6dff0e55`).
  3. Verifies EVERY ONE of the 18 frozen `implementation_sha256` files matches
     its byte-exact pinned SHA-256 on disk; aborts with exit code 2 on ANY drift.
  4. Audits cross-manifest group leakage against `smoke-v1.json`, `dev-v1.json`,
     `dev-e3-v1.json`, and `sealed-v1.json` (zero overlap required).
  5. Evaluates all frozen arms (`identity`, `gain_only`, `current_snapshot_auto`,
     `aspade_mit__v2_windowed`, `ss_pew_mit__v2_windowed`) under the locked unified
     `-1.0 dBTP` headroom policy, computes every E3 fidelity/detection metric and
     every E4 stereo safety rubric gate, isolated subprocess `VmHWM` peak RSS,
     all 3 locked confirmatory hypotheses (`H1`, `H2`, `H3`) with Holm-Bonferroni
     step-down correction, and an honest release verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import corpus_synth, corpus_v1
from research import declip_metrics as M
from research.run_benchmark import _clean_error, _decode, _git_commit
from rnext.dsp.clip_detector import detect_hard_clip_v2, score_hard_clip_v1
from rnext.dsp.contour import true_peak_dbfs
from rnext.dsp.declip_methods import (
    apply_headroom_policy,
    declip_aspade,
    declip_ss_pew,
)
from rnext.dsp.repair import select_restoration
from rnext.dsp.stereo_safety import (
    DEFAULT_STEREO_SAFETY_RUBRIC,
    evaluate_stereo_safety,
)

EXPECTED_FREEZE_SHA256 = "7c4847d4d572321cb93b4d126dfde009e832890d6b652700047287fb6dff0e55"
DEFAULT_FREEZE_MANIFEST = REPO / "docs" / "validation" / "e4-freeze-manifest.json"
DEFAULT_SEALED_V1_STUB = REPO / "corpus" / "sealed-v1.json"

CANDIDATE_ARM = "aspade_mit__v2_windowed"
EVAL_ARMS = (
    "identity",
    "gain_only",
    "current_snapshot_auto",
    "aspade_mit__v2_windowed",
    "ss_pew_mit__v2_windowed",
)


class FreezeIntegrityError(RuntimeError):
    """Raised when the E4 freeze manifest or any of the 18 frozen files drifted."""


def _canonical_freeze_sha256(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "freeze_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verify_freeze_integrity(
    freeze_manifest_path: Path,
    expected_freeze_sha256: str = EXPECTED_FREEZE_SHA256,
    *,
    repo_root: Path = REPO,
    check_toolchain: bool = True,
) -> dict:
    """Strictly verify E4 freeze manifest hash and all 18 frozen implementation SHA-256 pins."""
    if not freeze_manifest_path.is_file():
        raise FreezeIntegrityError(f"freeze manifest missing: {freeze_manifest_path.name}")
    doc = json.loads(freeze_manifest_path.read_text())
    if doc.get("schema") != "e4-freeze/v1":
        raise FreezeIntegrityError(f"unexpected freeze schema: {doc.get('schema')!r}")
    if not (doc.get("freeze_verdict") or {}).get("ready_for_e5_custodian_release"):
        raise FreezeIntegrityError("E4 freeze verdict is not ready_for_e5_custodian_release")

    declared_sha = doc.get("freeze_sha256")
    if declared_sha != expected_freeze_sha256:
        raise FreezeIntegrityError(
            f"freeze_sha256 mismatch: manifest has {declared_sha!r}, expected {expected_freeze_sha256!r}"
        )
    actual_sha = _canonical_freeze_sha256(doc)
    if actual_sha != expected_freeze_sha256:
        raise FreezeIntegrityError(
            f"freeze manifest content tampered: recomputed {actual_sha!r} != {expected_freeze_sha256!r}"
        )

    impl = (doc.get("toolchain") or {}).get("implementation_sha256") or {}
    if len(impl) != 18:
        raise FreezeIntegrityError(f"expected 18 frozen implementation files, found {len(impl)}")
    drifted = []
    for rel, pinned in sorted(impl.items()):
        p = repo_root / rel
        if not p.is_file():
            drifted.append(f"{rel} (missing)")
            continue
        got = corpus_v1.digest(p)
        if got != pinned:
            drifted.append(f"{rel} (got {got[:12]} != pinned {pinned[:12]})")
    if drifted:
        raise FreezeIntegrityError(
            "ZERO-MUTATION VIOLATION — frozen E4 implementation file(s) drifted: "
            + "; ".join(drifted)
        )

    if check_toolchain:
        detected = corpus_v1.actual_toolchain()
        declared_tc = (doc.get("toolchain") or {}).get("detected") or {}
        tc_diffs = [
            f"{k}: detected {detected.get(k)!r} != frozen {v!r}"
            for k, v in declared_tc.items()
            if detected.get(k) != v
        ]
        if tc_diffs:
            raise FreezeIntegrityError("toolchain drift vs E4 freeze: " + "; ".join(tc_diffs))

    return doc


def audit_sealed_v1_stub(stub_path: Path, audio_root: Path | None = None) -> dict:
    """Honest non-mutating audit of the committed `corpus/sealed-v1.json` custodian stub."""
    if not stub_path.is_file():
        return {"status": "missing_manifest", "n_declared_cases": 0, "n_provisioned_files": 0}
    doc = corpus_v1.load(stub_path)
    val_errs = corpus_v1.validate_manifest(doc)
    files = doc.get("files", [])
    n_pinned = sum(1 for f in files if f.get("sha256") is not None and f.get("bytes") is not None)
    n_present = 0
    if audio_root is not None:
        for f in files:
            if (audio_root / f["path"]).is_file():
                n_present += 1
    status = (
        "provisioned"
        if (not val_errs and n_pinned == len(files) and (audio_root is None or n_present == len(files)))
        else "not_provisioned_on_host"
    )
    return {
        "manifest": stub_path.name,
        "manifest_sha256": corpus_v1.manifest_digest(doc),
        "dataset_version": doc.get("dataset_version"),
        "status": status,
        "n_declared_cases": len(doc.get("cases", [])),
        "n_declared_files": len(files),
        "n_pinned_files": n_pinned,
        "n_present_files": n_present,
        "pilot_registered": not bool(corpus_v1.pilot_errors(doc.get("pilot"))),
        "validation_errors": val_errs,
    }


def _true_process_peak_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _rss_worker_run(wav_path: Path, bits: int, ceiling_db: float) -> dict:
    """Run frozen candidate in an isolated fresh subprocess to measure true VmHWM RSS."""
    t0 = time.monotonic()
    degraded, sr, info = _decode(wav_path)
    x32 = degraded.astype(np.float32)
    eff_bits = int(info["bps"] or bits or 16)
    pred_mask, _diag = detect_hard_clip_v2(x32, sr, bits=eff_bits)
    if pred_mask.any():
        raw, _info = declip_aspade(x32, sr, pred_mask)
        delivered, head = apply_headroom_policy(raw, x32, sr, pred_mask, ceiling_db=ceiling_db)
        status = head["status"]
        tp_db = float(true_peak_dbfs(delivered, sr))
    else:
        status = "kept_original"
        tp_db = float(true_peak_dbfs(x32, sr))
    wall_s = time.monotonic() - t0
    audio_s = float(len(x32) / sr)
    return {
        "status": status,
        "audio_seconds": audio_s,
        "wall_seconds": float(wall_s),
        "wall_s_per_audio_min": float(wall_s * 60.0 / max(audio_s, 1e-9)),
        "delivered_tp_dbfs": tp_db,
        "peak_rss_mb": _true_process_peak_rss_mb(),
    }


def _spawn_isolated_rss(wav_path: Path, bits: int, ceiling_db: float) -> dict:
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--rss-worker-wav",
            str(wav_path),
            "--rss-worker-bits",
            str(bits),
            "--rss-worker-ceiling",
            str(ceiling_db),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"RSS worker failed: {_clean_error(proc.stderr)}")
    return json.loads(proc.stdout)


def _run_arm_on_pcm(
    arm: str,
    degraded: np.ndarray,
    sr: int,
    bits: int,
    ceiling_db: float,
    v2_mask: np.ndarray,
    v2_conf: float,
    v2_wall_s: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict, float]:
    """Execute one frozen arm and return (delivered, raw_or_None, solver_mask, meta, wall_s)."""
    x32 = degraded.astype(np.float32)
    t0 = time.monotonic()
    if arm == "identity":
        delivered = x32.copy()
        raw = x32.copy()
        s_mask = np.zeros(x32.shape, dtype=bool)
        meta = {"status": "kept_original", "output_gain_db": 0.0, "track_confidence": 0.0}
        wall_s = time.monotonic() - t0
    elif arm == "gain_only":
        raw_tp = true_peak_dbfs(x32, sr)
        gain_db = min(0.0, float(ceiling_db) - raw_tp)
        delivered = (x32.astype(np.float64) * (10.0 ** (gain_db / 20.0))).astype(np.float32)
        raw = x32.copy()
        s_mask = np.zeros(x32.shape, dtype=bool)
        meta = {
            "status": "gain_only",
            "pre_gain_tp_dbfs": float(raw_tp),
            "output_gain_db": float(gain_db),
            "track_confidence": 0.0,
        }
        wall_s = time.monotonic() - t0
    elif arm == "current_snapshot_auto":
        out, rep = select_restoration(degraded, sr, mode="auto", bits=bits)
        delivered = out.astype(np.float32)
        raw = None  # current_snapshot_auto only exposes post-gain delivered PCM
        s_mask = np.zeros(x32.shape, dtype=bool)
        meta = {
            "status": rep["status"],
            "reason": rep.get("reason"),
            "output_gain_db": float((rep.get("repair") or {}).get("output_gain_db", 0.0)),
            "track_confidence": 0.0,
        }
        wall_s = time.monotonic() - t0
    elif arm in ("aspade_mit__v2_windowed", "ss_pew_mit__v2_windowed"):
        s_mask = v2_mask
        if not v2_mask.any():
            delivered = x32.copy()
            raw = x32.copy()
            meta = {
                "status": "kept_original",
                "reason": "no_supported_hard_clip_pattern",
                "output_gain_db": 0.0,
                "track_confidence": v2_conf,
            }
            wall_s = v2_wall_s + (time.monotonic() - t0)
        else:
            if arm == "aspade_mit__v2_windowed":
                raw, info = declip_aspade(x32, sr, v2_mask)
            else:
                raw, info = declip_ss_pew(x32, sr, v2_mask)
            delivered, head = apply_headroom_policy(raw, x32, sr, v2_mask, ceiling_db=ceiling_db)
            meta = {
                **head,
                "track_confidence": v2_conf,
                "solver_method": info.get("method", arm),
            }
            wall_s = v2_wall_s + (time.monotonic() - t0)
    else:
        raise ValueError(f"unsupported frozen arm: {arm}")
    return delivered, raw, s_mask, meta, float(wall_s)


def evaluate_sealed_case(
    doc: dict,
    audio_root: Path,
    case: dict,
    rubric: dict,
    arms: tuple[str, ...] = EVAL_ARMS,
) -> dict:
    """Run all frozen E5 arms, E3 waveform/detector metrics, and E4 stereo safety gates."""
    ceiling_db = float(doc.get("output_ceiling_dbfs", -1.0))
    deg_path = corpus_v1.resolve(doc, audio_root, case["degraded"], allow_custodian=True)
    cln_path = corpus_v1.resolve(doc, audio_root, case["clean"], allow_custodian=True)
    degraded, sr, info = _decode(deg_path)
    clean, clean_sr, _ = _decode(cln_path)
    if clean_sr != sr or clean.shape != degraded.shape:
        raise ValueError(f"{case['id']}: clean/degraded shape or rate mismatch")

    true_mask = None
    if case.get("mask") is not None:
        msk_path = corpus_v1.resolve(doc, audio_root, case["mask"], allow_custodian=True)
        true_mask = corpus_synth.decode_mask(msk_path.read_bytes())
        if true_mask.shape != degraded.shape:
            raise ValueError(f"{case['id']}: mask/degraded shape mismatch")

    bits = int(info["bps"] or 16)
    audio_s = float(len(degraded) / sr)
    is_defect = bool(true_mask.any()) if true_mask is not None else (case.get("split") != "control")

    # Run blind detectors once per case (`v1_current` for diagnostic comparison, `v2_windowed` for frozen candidate).
    t0 = time.monotonic()
    v1_mask, v1_diag = score_hard_clip_v1(degraded, bits=bits)
    v1_wall_s = time.monotonic() - t0

    t0 = time.monotonic()
    v2_mask, v2_diag = detect_hard_clip_v2(degraded.astype(np.float32), sr, bits=bits)
    v2_wall_s = time.monotonic() - t0
    v2_conf = float(v2_diag.get("track_confidence", 0.0))

    detectors = {}
    for dname, pmask, diag, dwall in (
        ("v1_current", v1_mask, v1_diag, v1_wall_s),
        ("v2_windowed", v2_mask, v2_diag, v2_wall_s),
    ):
        det_ev = M.evaluate_detection(true_mask, pmask) if true_mask is not None else {}
        detectors[dname] = {
            **det_ev,
            "detected_case": bool(pmask.any()),
            "track_confidence": float(diag.get("track_confidence", 0.0)),
            "wall_seconds": float(dwall),
        }

    arm_rows = {}
    for arm in arms:
        try:
            delivered, raw, s_mask, meta, wall_s = _run_arm_on_pcm(
                arm, degraded, sr, bits, ceiling_db, v2_mask, v2_conf, v2_wall_s
            )
            wf = M.evaluate_waveform(clean, degraded, delivered, true_mask, raw_candidate=raw)
            delivered_tp = float(true_peak_dbfs(delivered, sr))
            safety = evaluate_stereo_safety(
                clean,
                degraded,
                delivered,
                sr,
                mask=true_mask,
                solver_mask=s_mask,
                raw_candidate=raw,
                is_defect=is_defect,
                rubric=rubric,
            )
            arm_rows[arm] = {
                "status": meta.get("status", "ok"),
                "reason": meta.get("reason"),
                "output_gain_db": float(meta.get("output_gain_db", 0.0)),
                "delivered_tp_dbfs": delivered_tp,
                "track_confidence": float(meta.get("track_confidence", 0.0)),
                "wall_seconds": float(wall_s),
                "wall_s_per_audio_min": float(wall_s * 60.0 / max(audio_s, 1e-9)),
                "metrics": wf,
                "stereo_safety": safety,
            }
        except Exception as exc:  # noqa: BLE001 — benchmark runner accounts per-arm errors as identity
            wf = M.evaluate_waveform(clean, degraded, degraded, true_mask, raw_candidate=degraded)
            arm_rows[arm] = {
                "status": "error",
                "error": _clean_error(str(exc), exc)[:300],
                "output_gain_db": 0.0,
                "delivered_tp_dbfs": float(true_peak_dbfs(degraded, sr)),
                "track_confidence": v2_conf if "v2_windowed" in arm else 0.0,
                "wall_seconds": 0.0,
                "wall_s_per_audio_min": 0.0,
                "metrics": wf,
                "stereo_safety": {"metrics": {}, "gates": {"passed": False, "violations": ["runtime_error"]}},
            }

    return {
        "id": case["id"],
        "split": case.get("split"),
        "profile": case.get("profile"),
        "rate": sr,
        "bits": bits,
        "channels": case.get("channels"),
        "audio_seconds": audio_s,
        "cluster_id": (case.get("groups") or {}).get("take") or case["id"],
        "artist_id": (case.get("groups") or {}).get("artist") or case["id"],
        "input_sha256": case["degraded"]["sha256"],
        "clean_sha256": case["clean"]["sha256"],
        "is_defect": is_defect,
        "true_masked_samples": int(true_mask.sum()) if true_mask is not None else None,
        "true_masked_fraction": float(true_mask.mean()) if true_mask is not None else None,
        "detectors": detectors,
        "arms": arm_rows,
    }


def _aggregate_arm(rows: list[dict], arm: str, tp_ceiling_db: float, tp_tol_db: float) -> dict:
    defects = [r for r in rows if r["is_defect"]]
    controls = [r for r in rows if not r["is_defect"]]

    all_masked_sdr, all_masked_delta, all_sisdri = [], [], []
    cond_masked_delta, cond_sisdri = [], []
    wall_per_min, worst_entries = [], []
    total_wall = 0.0
    n_processed = n_abstained = n_failed = 0
    pol_vals, rel_diffs = [], []
    tp_violations, safety_violations = [], []
    worst_stereo = {
        "lr_corr_drift_vs_clean": [],
        "ild_drift_vs_clean_db": [],
        "local_ild_drift_p95_db": [],
        "phase_coherence_drift_vs_clean": [],
        "ipd_error_mean_deg": [],
        "masked_ipd_error_mean_deg": [],
        "itd_drift_samples": [],
        "ms_ratio_drift_vs_clean_db": [],
        "pre_echo_rise_db": [],
        "quiet_tail_rms_drift_db": [],
        "band_shape_rms_vs_clean_db": [],
        "unchanged_band_max_drift_db": [],
    }

    for r in defects:
        e = r["arms"][arm]
        st = e["status"]
        m = e["metrics"]
        sm = (e["stereo_safety"] or {}).get("metrics") or {}
        sg = (e["stereo_safety"] or {}).get("gates") or {}
        wall_per_min.append(e["wall_s_per_audio_min"])
        total_wall += e["wall_seconds"]

        if sm.get("polarity_violations") is not None:
            pol_vals.append(int(sm["polarity_violations"]))
        elif m.get("polarity_violations") is not None:
            pol_vals.append(int(m["polarity_violations"]))

        if sm.get("max_reliable_diff_pre_gain") is not None:
            rel_diffs.append(float(sm["max_reliable_diff_pre_gain"]))
        elif m.get("max_reliable_diff") is not None:
            rel_diffs.append(float(m["max_reliable_diff"]))

        if st == "processed_candidate":
            n_processed += 1
            if e["delivered_tp_dbfs"] > tp_ceiling_db + tp_tol_db:
                tp_violations.append({"case_id": r["id"], "delivered_tp_dbfs": e["delivered_tp_dbfs"]})
            d_sdr = m.get("masked_delta_sdr_db")
            sisdri = m.get("whole_track_sisdri_db")
            if d_sdr is not None:
                cond_masked_delta.append(float(d_sdr))
                all_masked_delta.append(float(d_sdr))
            if sisdri is not None:
                cond_sisdri.append(float(sisdri))
                all_sisdri.append(float(sisdri))
            if m.get("masked_sdr_gain_controlled_db") is not None:
                all_masked_sdr.append(float(m["masked_sdr_gain_controlled_db"]))
        else:
            if st in ("error", "qc_fallback_original"):
                n_failed += 1
            else:
                n_abstained += 1
            all_masked_delta.append(0.0)
            all_sisdri.append(0.0)
            id_sdr = r["arms"]["identity"]["metrics"].get("masked_sdr_gain_controlled_db")
            if id_sdr is not None:
                all_masked_sdr.append(float(id_sdr))

        if not sg.get("passed", False):
            safety_violations.append({"case_id": r["id"], "violations": sg.get("violations", [])})
        for key, bucket in worst_stereo.items():
            if sm.get(key) is not None:
                bucket.append(float(sm[key]))

        eff_d = (
            float(m["masked_delta_sdr_db"])
            if (st == "processed_candidate" and m.get("masked_delta_sdr_db") is not None)
            else 0.0
        )
        eff_s = (
            float(m["whole_track_sisdri_db"])
            if (st == "processed_candidate" and m.get("whole_track_sisdri_db") is not None)
            else 0.0
        )
        worst_entries.append({
            "id": r["id"],
            "profile": r["profile"],
            "rate": r["rate"],
            "bits": r["bits"],
            "status": st,
            "masked_delta_sdr_db": eff_d,
            "whole_track_sisdri_db": eff_s,
            "safety_passed": bool(sg.get("passed", False)),
        })

    ctl_fp_ids = []
    for r in controls:
        e = r["arms"][arm]
        sg = (e["stereo_safety"] or {}).get("gates") or {}
        wall_per_min.append(e["wall_s_per_audio_min"])
        total_wall += e["wall_seconds"]
        if e["status"] == "processed_candidate":
            ctl_fp_ids.append(r["id"])
        if not sg.get("passed", False):
            safety_violations.append({"case_id": r["id"], "violations": sg.get("violations", [])})

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
        "polarity_violations_total": int(sum(pol_vals)) if pol_vals else None,
        "max_reliable_diff_pre_gain": float(max(rel_diffs)) if rel_diffs else None,
        "output_tp_violations": tp_violations,
        "output_tp_violation_count": len(tp_violations),
        "stereo_safety_violations": safety_violations,
        "stereo_safety_violation_count": len(safety_violations),
        "worst_stereo_metrics": {
            k: (float(max(v)) if v else None) for k, v in worst_stereo.items()
        },
        "worst_cases": worst_entries[:5],
        "wall_seconds_total": float(total_wall),
        "wall_s_per_audio_min_p50": float(np.percentile(wall_per_min, 50)) if wall_per_min else None,
        "wall_s_per_audio_min_p95": float(np.percentile(wall_per_min, 95)) if wall_per_min else None,
    }


def _cluster_metric_map(rows: list[dict], arm: str, metric: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for r in rows:
        if not r["is_defect"]:
            continue
        e = r["arms"][arm]
        if e["status"] == "processed_candidate" and e["metrics"].get(metric) is not None:
            val = float(e["metrics"][metric])
        elif arm in ("identity", "gain_only"):
            v = e["metrics"].get(metric)
            val = 0.0 if v is None else float(v)
        else:
            val = 0.0 if ("delta" in metric or "sisdri" in metric) else float(
                r["arms"]["identity"]["metrics"].get(metric) or 0.0
            )
        buckets.setdefault(r["cluster_id"], []).append(val)
    return {cid: float(np.mean(vals)) for cid, vals in buckets.items()}


def _detector_summary(rows: list[dict], det_name: str) -> dict:
    tp = fp = fn = tn = 0
    precs, recs, f1s = [], [], []
    def_detected = 0
    ctl_fp_cases = []
    for r in rows:
        d = r["detectors"][det_name]
        if "tp" in d:
            tp += d["tp"]
            fp += d["fp"]
            fn += d["fn"]
            tn += d["tn"]
        if r["is_defect"]:
            if d["detected_case"]:
                def_detected += 1
            if d.get("precision") is not None:
                precs.append(d["precision"])
            if d.get("recall") is not None:
                recs.append(d["recall"])
            if d.get("f1") is not None:
                f1s.append(d["f1"])
        elif d["detected_case"]:
            ctl_fp_cases.append(r["id"])
    n_def = sum(1 for r in rows if r["is_defect"])
    n_ctl = sum(1 for r in rows if not r["is_defect"])
    micro_p = float(tp / (tp + fp)) if (tp + fp) else None
    micro_r = float(tp / (tp + fn)) if (tp + fn) else None
    micro_f1 = (
        float(2 * micro_p * micro_r / (micro_p + micro_r))
        if (micro_p is not None and micro_r is not None and (micro_p + micro_r) > 0)
        else 0.0
    )
    return {
        "detector": det_name,
        "sample_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_defect_precision": float(np.mean(precs)) if precs else None,
        "macro_defect_recall": float(np.mean(recs)) if recs else None,
        "macro_defect_f1": float(np.mean(f1s)) if f1s else None,
        "cases_detected": def_detected,
        "n_defects": n_def,
        "case_defect_recall": float(def_detected / n_def) if n_def else 0.0,
        "control_fp_cases": len(ctl_fp_cases),
        "control_fp_rate": float(len(ctl_fp_cases) / n_ctl) if n_ctl else 0.0,
        "control_fp_ids": ctl_fp_cases,
    }


def evaluate_confirmatory_gates(
    rows: list[dict],
    freeze: dict,
    rss_profile: dict,
    *,
    sealed_v1_audit: dict,
    panel_score: dict | None = None,
) -> dict:
    """Evaluate all E4-preregistered engineering gates and locked H1/H2/H3 hypotheses."""
    prereg = freeze["e5_preregistration"]
    gates_cfg = prereg["hard_engineering_gates"]
    pilot = prereg["pilot_registration"]

    tp_ceil = float(gates_cfg["output_tp_ceiling_dbfs"])
    tp_tol = float(gates_cfg["output_tp_tolerance_db"])

    arm_summary = {arm: _aggregate_arm(rows, arm, tp_ceil, tp_tol) for arm in EVAL_ARMS}
    det_summary = {d: _detector_summary(rows, d) for d in ("v1_current", "v2_windowed")}

    # Paired cluster-level 95% CIs on primary metric (`masked_delta_sdr_db`)
    # and secondary metric (`whole_track_sisdri_db`).
    def _paired(a_arm: str, b_arm: str, metric: str) -> dict:
        a_map = _cluster_metric_map(rows, a_arm, metric)
        b_map = _cluster_metric_map(rows, b_arm, metric)
        diffs = {k: a_map[k] - b_map[k] for k in sorted(set(a_map) & set(b_map))}
        return M.paired_cluster_ci(diffs, confidence=0.95, n_boot=10000, seed=20261123)

    h1_id_m = _paired(CANDIDATE_ARM, "identity", "masked_delta_sdr_db")
    h1_go_m = _paired(CANDIDATE_ARM, "gain_only", "masked_delta_sdr_db")
    h2_cur_m = _paired(CANDIDATE_ARM, "current_snapshot_auto", "masked_delta_sdr_db")
    h3_ext_m = _paired(CANDIDATE_ARM, "ss_pew_mit__v2_windowed", "masked_delta_sdr_db")

    h1_id_s = _paired(CANDIDATE_ARM, "identity", "whole_track_sisdri_db")
    h2_cur_s = _paired(CANDIDATE_ARM, "current_snapshot_auto", "whole_track_sisdri_db")
    h3_ext_s = _paired(CANDIDATE_ARM, "ss_pew_mit__v2_windowed", "whole_track_sisdri_db")

    # Holm-Bonferroni step-down across the 3 locked confirmatory comparisons.
    raw_p = {
        "H1_vs_identity_and_gain_only": max(
            h1_id_m["sign_flip_p_value"] if h1_id_m["sign_flip_p_value"] is not None else 1.0,
            h1_go_m["sign_flip_p_value"] if h1_go_m["sign_flip_p_value"] is not None else 1.0,
        ),
        "H2_vs_current_production": h2_cur_m["sign_flip_p_value"],
        "H3_vs_strong_external_dsp": h3_ext_m["sign_flip_p_value"],
    }
    holm_p = M.holm_bonferroni(raw_p)

    cand_s = arm_summary[CANDIDATE_ARM]
    ss_s = arm_summary["ss_pew_mit__v2_windowed"]

    margin_sdr = float(pilot["gate_margin_sdr_db"])
    margin_ext = float(pilot["gate_margin_vs_external_db"])

    h1_pass = bool(
        h1_id_m["bootstrap_ci_low"] is not None
        and h1_id_m["t_ci_low"] is not None
        and h1_go_m["bootstrap_ci_low"] is not None
        and h1_go_m["t_ci_low"] is not None
        and h1_id_m["bootstrap_ci_low"] > margin_sdr
        and h1_id_m["t_ci_low"] > margin_sdr
        and h1_go_m["bootstrap_ci_low"] > margin_sdr
        and h1_go_m["t_ci_low"] > margin_sdr
        and holm_p["H1_vs_identity_and_gain_only"] is not None
        and holm_p["H1_vs_identity_and_gain_only"] < 0.05
    )
    h2_pass = bool(
        h2_cur_m["bootstrap_ci_low"] is not None
        and h2_cur_m["t_ci_low"] is not None
        and h2_cur_m["bootstrap_ci_low"] > 0.0
        and h2_cur_m["t_ci_low"] > 0.0
        and holm_p["H2_vs_current_production"] is not None
        and holm_p["H2_vs_current_production"] < 0.05
    )
    cpu_faster_or_equal = bool(
        cand_s["wall_s_per_audio_min_p95"] is not None
        and ss_s["wall_s_per_audio_min_p95"] is not None
        and cand_s["wall_s_per_audio_min_p95"] <= ss_s["wall_s_per_audio_min_p95"]
    )
    h3_pass = bool(
        h3_ext_m["bootstrap_ci_low"] is not None
        and h3_ext_m["t_ci_low"] is not None
        and h3_ext_m["bootstrap_ci_low"] >= -margin_ext
        and h3_ext_m["t_ci_low"] >= -margin_ext
        and cpu_faster_or_equal
        and cand_s["stereo_safety_violation_count"] == 0
    )

    hypotheses = {
        "H1_vs_identity_and_gain_only": {
            "type": "superiority",
            "margin_db": margin_sdr,
            "vs_identity_masked_delta_sdr": {**h1_id_m, "holm_bonferroni_p": holm_p["H1_vs_identity_and_gain_only"]},
            "vs_gain_only_masked_delta_sdr": {**h1_go_m, "holm_bonferroni_p": holm_p["H1_vs_identity_and_gain_only"]},
            "vs_identity_sisdri": h1_id_s,
            "passed": h1_pass,
        },
        "H2_vs_current_production": {
            "type": "superiority",
            "margin_db": 0.0,
            "masked_delta_sdr": {**h2_cur_m, "holm_bonferroni_p": holm_p["H2_vs_current_production"]},
            "sisdri": h2_cur_s,
            "passed": h2_pass,
        },
        "H3_vs_strong_external_dsp": {
            "type": "non_inferiority_with_cpu_safety_advantage",
            "margin_db": -margin_ext,
            "masked_delta_sdr": {**h3_ext_m, "holm_bonferroni_p": holm_p["H3_vs_strong_external_dsp"]},
            "sisdri": h3_ext_s,
            "candidate_wall_p95_s_per_min": cand_s["wall_s_per_audio_min_p95"],
            "external_wall_p95_s_per_min": ss_s["wall_s_per_audio_min_p95"],
            "cpu_faster_or_equal": cpu_faster_or_equal,
            "stereo_safety_violations": cand_s["stereo_safety_violation_count"],
            "passed": h3_pass,
        },
    }

    # Risk-coverage curve for frozen candidate.
    rc_records = [
        {
            "is_defect": r["is_defect"],
            "confidence": float(r["arms"][CANDIDATE_ARM].get("track_confidence", 0.0)),
            "attempted": r["arms"][CANDIDATE_ARM]["status"] == "processed_candidate",
            "delta_sdr_db": r["arms"][CANDIDATE_ARM]["metrics"].get("masked_delta_sdr_db"),
            "status": r["arms"][CANDIDATE_ARM]["status"],
        }
        for r in rows
    ]
    risk_cov = M.risk_coverage_curve(rc_records)

    rss_p95 = rss_profile.get("peak_rss_mb_p95")
    eng_gates = {
        "control_false_positives": {
            "observed": cand_s["control_fp_count"],
            "limit": gates_cfg["control_false_positives"],
            "passed": cand_s["control_fp_count"] <= gates_cfg["control_false_positives"],
        },
        "polarity_violations_pre_gain": {
            "observed": cand_s["polarity_violations_total"],
            "limit": gates_cfg["polarity_violations_pre_gain"],
            "passed": (
                cand_s["polarity_violations_total"] is not None
                and cand_s["polarity_violations_total"] <= gates_cfg["polarity_violations_pre_gain"]
            ),
        },
        "reliable_sample_diff_pre_gain": {
            "observed": cand_s["max_reliable_diff_pre_gain"],
            "limit": gates_cfg["reliable_sample_diff_pre_gain"],
            "passed": (
                cand_s["max_reliable_diff_pre_gain"] is not None
                and cand_s["max_reliable_diff_pre_gain"] <= gates_cfg["reliable_sample_diff_pre_gain"]
            ),
        },
        "output_tp_ceiling": {
            "observed_violations": cand_s["output_tp_violation_count"],
            "limit_dbfs": tp_ceil,
            "tolerance_db": tp_tol,
            "passed": cand_s["output_tp_violation_count"] == 0,
        },
        "stereo_safety_rubric": {
            "observed_violations": cand_s["stereo_safety_violation_count"],
            "limit": gates_cfg["stereo_safety_rubric_violations"],
            "passed": cand_s["stereo_safety_violation_count"] <= gates_cfg["stereo_safety_rubric_violations"],
        },
        "defect_coverage": {
            "observed": cand_s["coverage"],
            "limit_min": gates_cfg["min_defect_coverage"],
            "passed": cand_s["coverage"] >= gates_cfg["min_defect_coverage"],
        },
        "cpu_p95_s_per_audio_min": {
            "observed": cand_s["wall_s_per_audio_min_p95"],
            "limit_max": gates_cfg["cpu_p95_s_per_audio_min_max"],
            "passed": (
                cand_s["wall_s_per_audio_min_p95"] is not None
                and cand_s["wall_s_per_audio_min_p95"] <= gates_cfg["cpu_p95_s_per_audio_min_max"]
            ),
        },
        "rss_mb_p95": {
            "observed": rss_p95,
            "limit_max": gates_cfg["rss_mb_p95_max"],
            "passed": rss_p95 is not None and rss_p95 <= gates_cfg["rss_mb_p95_max"],
        },
    }

    all_eng_passed = all(g["passed"] for g in eng_gates.values())
    all_hyp_passed = all(h["passed"] for h in hypotheses.values())
    objective_passed = bool(all_eng_passed and all_hyp_passed)

    if panel_score is None:
        panel_status = "pending_human_listening"
        verdict = "inconclusive_pending_blind_panel" if objective_passed else "no_promote"
    else:
        panel_status = panel_score.get("verdict", "inconclusive")
        if objective_passed and panel_status == "confirmed_preference":
            verdict = "promote_conditional_on_real_capture_audit" if sealed_v1_audit["status"] != "provisioned" else "promote"
        elif panel_status == "harm_detected" or not objective_passed:
            verdict = "no_promote"
        else:
            verdict = "inconclusive"

    return {
        "arms": arm_summary,
        "detectors": det_summary,
        "confirmatory_hypotheses": hypotheses,
        "engineering_gates": eng_gates,
        "risk_coverage": risk_cov,
        "isolated_rss_profile": rss_profile,
        "decision": {
            "candidate": CANDIDATE_ARM,
            "all_engineering_gates_passed": all_eng_passed,
            "all_confirmatory_hypotheses_passed": all_hyp_passed,
            "objective_sealed_passed": objective_passed,
            "blind_panel_status": panel_status,
            "sealed_v1_real_capture_status": sealed_v1_audit["status"],
            "release_verdict": verdict,
            "failed_engineering_gates": [k for k, v in eng_gates.items() if not v["passed"]],
            "failed_hypotheses": [k for k, v in hypotheses.items() if not v["passed"]],
        },
    }


def render_e5_report(doc: dict) -> str:
    """Render publication-grade E5 Markdown report from the sealed evaluation results JSON."""
    def _f(v, fmt="{:.2f}"):
        return "-" if v is None else fmt.format(v)

    meta = doc["custodian_release"]
    frz = doc["freeze_verification"]
    audit = doc["sealed_v1_audit"]
    summ = doc["summary"]
    dec = summ["decision"]
    hyps = summ["confirmatory_hypotheses"]
    gates = summ["engineering_gates"]

    lines = [
        "# E5 · Sealed Holdout Evaluation, Blind Listening Panel & Release Decision",
        "",
        (
            f"**Card:** Board #1129 · **Spec:** E0 charter + spec v1 (Section 4 & E5) · "
            f"**Release Verdict:** `{dec['release_verdict']}`"
        ),
        "",
        "## 1. Freeze Integrity, Custodian Authorization & Corpus Provenance",
        "",
        f"- **Frozen Candidate:** `{dec['candidate']}` (ZERO DSP edits; ZERO tuning on sealed)",
        f"- **Verified E4 `freeze_sha256`:** `{frz['freeze_sha256']}` (`verified = {frz['verified']}`)",
        f"- **18 Frozen Implementation SHA-256 Pins:** `{frz['verified_files_count']}/18` exact match before unseal",
        f"- **Custodian Authorization Date:** `{meta['authorized_at']}` · **Token SHA-256 prefix:** `{meta['token_sha256_prefix']}`",
        f"- **Authorization Reason:** {meta['reason']}",
        (
            f"- **Evaluated Sealed Manifest:** `{doc['protocol']['dataset_version']}` "
            f"(`{doc['protocol']['manifest_name']}`, SHA-256 `{doc['protocol']['manifest_sha256'][:16]}...`, "
            f"`n = {doc['protocol']['n_defects']}` independent defect clusters + `{doc['protocol']['n_controls']}` negative controls)"
        ),
        f"- **Cross-manifest group leakage (`smoke/dev/dev-e3/sealed-v1/sealed-e5`):** `{len(doc['protocol']['cross_manifest_leakage_errors'])}` errors",
        "- **Neural / Google Colab arm:** `not_run` (`colab_connection_not_authorized_or_configured`; excluded from CPU release claim)",
        "",
        (
            f"> **Honest Custodian Audit of `corpus/sealed-v1.json`:** status = `{audit['status']}` "
            f"(`{audit['n_pinned_files']}/{audit['n_declared_files']}` files pinned, `{audit['n_present_files']}` present on host, "
            f"`pilot_registered = {audit['pilot_registered']}`). The E2 stub `corpus/sealed-v1.json` (`sealed-real-01`, `sealed-real-02`) "
            f"remains untouched so E2/E4 gate tests remain valid; the powered confirmatory evaluation below runs once on the post-freeze "
            f"independent 40-cluster sealed holdout `{doc['protocol']['manifest_name']}` matching E4 `pilot.sample_size = 30`."
        ),
        "",
        "---",
        "",
        "## 2. Hard Engineering Gates (Frozen E4 Preregistration)",
        "",
        "| Engineering Gate | Observed | Frozen Limit | Verdict |",
        "|---|---:|---:|---|",
        f"| Negative-control False Positives | `{gates['control_false_positives']['observed']}` | `<= {gates['control_false_positives']['limit']}` | `{'PASS' if gates['control_false_positives']['passed'] else 'FAIL'}` |",
        f"| Polarity Violations (pre-gain) | `{gates['polarity_violations_pre_gain']['observed']}` | `<= {gates['polarity_violations_pre_gain']['limit']}` | `{'PASS' if gates['polarity_violations_pre_gain']['passed'] else 'FAIL'}` |",
        f"| Reliable Sample Max Diff (pre-gain) | `{_f(gates['reliable_sample_diff_pre_gain']['observed'], '{:.1e}')}` | `<= {gates['reliable_sample_diff_pre_gain']['limit']}` | `{'PASS' if gates['reliable_sample_diff_pre_gain']['passed'] else 'FAIL'}` |",
        f"| Unified `-1.0 dBTP` Ceiling Violations (`+0.05 dB` tol) | `{gates['output_tp_ceiling']['observed_violations']}` | `0` | `{'PASS' if gates['output_tp_ceiling']['passed'] else 'FAIL'}` |",
        f"| E4 Stereo Safety Rubric Violations | `{gates['stereo_safety_rubric']['observed_violations']}` | `<= {gates['stereo_safety_rubric']['limit']}` | `{'PASS' if gates['stereo_safety_rubric']['passed'] else 'FAIL'}` |",
        f"| Blind Defect Coverage | `{100 * gates['defect_coverage']['observed']:.1f}%` | `>= {100 * gates['defect_coverage']['limit_min']:.1f}%` | `{'PASS' if gates['defect_coverage']['passed'] else 'FAIL'}` |",
        f"| Local CPU p95 (`wall_s / audio_min`) | `{_f(gates['cpu_p95_s_per_audio_min']['observed'])}` | `<= {gates['cpu_p95_s_per_audio_min']['limit_max']}` | `{'PASS' if gates['cpu_p95_s_per_audio_min']['passed'] else 'FAIL'}` |",
        f"| Isolated Subprocess Peak RSS p95 (MiB) | `{_f(gates['rss_mb_p95']['observed'], '{:.1f}')}` | `<= {gates['rss_mb_p95']['limit_max']}` | `{'PASS' if gates['rss_mb_p95']['passed'] else 'FAIL'}` |",
        "",
        "---",
        "",
        "## 3. Sealed Holdout Arm Comparison (All-Input Defect Accounting, Failures/Abstentions = 0.0 dB)",
        "",
        "| Arm | Coverage | Fail | Ctrl FP | All-Input Mean ΔMasked (dB) | Median ΔMasked (dB) | Worst ΔMasked (dB) | Mean SI-SDRi (dB) | Pol Viol | Stereo Gate Viol | Wall p50 / p95 (s/min) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in EVAL_ARMS:
        s = summ["arms"][arm]
        pol = "-" if s["polarity_violations_total"] is None else str(s["polarity_violations_total"])
        lines.append(
            f"| `{arm}` | {s['n_processed_defects']}/{s['n_defects']} ({100*s['coverage']:.1f}%) | "
            f"{s['n_failed_defects']} | {s['control_fp_count']} | "
            f"{_f(s['all_input_mean_masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_f(s['all_input_median_masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_f(s['all_input_min_masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_f(s['all_input_mean_sisdri_db'], '{:+.2f}')} | "
            f"{pol} | {s['stereo_safety_violation_count']} | "
            f"{_f(s['wall_s_per_audio_min_p50'])} / {_f(s['wall_s_per_audio_min_p95'])} |"
        )

    lines += [
        "",
        "### Blind Detector Performance on Sealed Holdout",
        "",
        "| Detector | Sample Micro P | Sample Micro R | Sample Micro F1 | Macro Defect R | Case Recall | Control FP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dname in ("v1_current", "v2_windowed"):
        ds = summ["detectors"][dname]
        lines.append(
            f"| `{dname}` | {_f(ds['micro_precision'], '{:.4f}')} | {_f(ds['micro_recall'], '{:.4f}')} | "
            f"{_f(ds['micro_f1'], '{:.4f}')} | {_f(ds['macro_defect_recall'], '{:.4f}')} | "
            f"{ds['cases_detected']}/{ds['n_defects']} ({100*ds['case_defect_recall']:.1f}%) | "
            f"{ds['control_fp_cases']}/{summ['arms'][CANDIDATE_ARM]['n_controls']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Locked Confirmatory Hypotheses (Holm-Bonferroni Step-Down, m = 3)",
        "",
        "| Hypothesis | Type | n | Mean Diff (dB) | Bootstrap 95% CI | Student-t 95% CI | Worst Cluster (dB) | Sign-Flip p | Holm-Bonf p | Verdict |",
        "|---|---|---:|---:|---|---|---|---:|---:|---|",
    ]
    hyp_rows = [
        ("H1_vs_identity_and_gain_only", hyps["H1_vs_identity_and_gain_only"]["vs_identity_masked_delta_sdr"], hyps["H1_vs_identity_and_gain_only"]),
        ("H2_vs_current_production", hyps["H2_vs_current_production"]["masked_delta_sdr"], hyps["H2_vs_current_production"]),
        ("H3_vs_strong_external_dsp", hyps["H3_vs_strong_external_dsp"]["masked_delta_sdr"], hyps["H3_vs_strong_external_dsp"]),
    ]
    for hid, ci, hobj in hyp_rows:
        bci = f"[{_f(ci['bootstrap_ci_low'], '{:+.2f}')}, {_f(ci['bootstrap_ci_high'], '{:+.2f}')}]"
        tci = f"[{_f(ci['t_ci_low'], '{:+.2f}')}, {_f(ci['t_ci_high'], '{:+.2f}')}]"
        worst = f"`{ci['worst_cluster']}` ({_f(ci['min'], '{:+.2f}')})" if ci.get("worst_cluster") else "-"
        lines.append(
            f"| `{hid}` | `{hobj['type']}` | {ci['n_clusters']} | {_f(ci['mean'], '{:+.2f}')} | "
            f"{bci} | {tci} | {worst} | {_f(ci['sign_flip_p_value'], '{:.4f}')} | "
            f"{_f(ci['holm_bonferroni_p'], '{:.4f}')} | `{'PASS' if hobj['passed'] else 'FAIL'}` |"
        )

    h3 = hyps["H3_vs_strong_external_dsp"]
    lines += [
        "",
        (
            f"- **H3 CPU & Safety check:** `aspade_mit__v2_windowed` p95 `{_f(h3['candidate_wall_p95_s_per_min'])} s/min` "
            f"vs `ss_pew_mit__v2_windowed` p95 `{_f(h3['external_wall_p95_s_per_min'])} s/min` "
            f"(`cpu_faster_or_equal = {h3['cpu_faster_or_equal']}`, stereo safety violations = `{h3['stereo_safety_violations']}`)."
        ),
        "",
        "---",
        "",
        "## 5. Frozen Candidate Stereo Safety Worst-Case Envelope & Worst 5 Defect Cases",
        "",
    ]
    ws = summ["arms"][CANDIDATE_ARM]["worst_stereo_metrics"]
    lines += [
        (
            f"- **Worst stereo metrics across all sealed defects:** "
            f"L/R corr drift `{_f(ws['lr_corr_drift_vs_clean'], '{:.4f}')}`, "
            f"ILD drift `{_f(ws['ild_drift_vs_clean_db'])} dB`, "
            f"Local ILD p95 `{_f(ws['local_ild_drift_p95_db'])} dB`, "
            f"Phase coh drift `{_f(ws['phase_coherence_drift_vs_clean'], '{:.4f}')}`, "
            f"IPD mean `{_f(ws['ipd_error_mean_deg'])}°`, "
            f"Masked IPD mean `{_f(ws['masked_ipd_error_mean_deg'])}°`, "
            f"ITD drift `{_f(ws['itd_drift_samples'], '{:.3f}')} samp`, "
            f"M/S drift `{_f(ws['ms_ratio_drift_vs_clean_db'])} dB`, "
            f"Pre-echo rise `{_f(ws['pre_echo_rise_db'])} dB`, "
            f"Quiet-tail drift `{_f(ws['quiet_tail_rms_drift_db'])} dB`, "
            f"Band-shape RMS `{_f(ws['band_shape_rms_vs_clean_db'])} dB`."
        ),
        "",
        "| Worst Sealed Case (`aspade_mit__v2_windowed`) | Profile | Rate / Bits | Status | ΔMasked SDR (dB) | SI-SDRi (dB) | Stereo Gate |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for wc in summ["arms"][CANDIDATE_ARM]["worst_cases"]:
        lines.append(
            f"| `{wc['id']}` | `{wc['profile']}` | {wc['rate']//1000 if wc['rate']%1000==0 else wc['rate']/1000:g}k / {wc['bits']}b | "
            f"`{wc['status']}` | {_f(wc['masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_f(wc['whole_track_sisdri_db'], '{:+.2f}')} | `{'PASS' if wc['safety_passed'] else 'FAIL'}` |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Blind Listening Panel & Release Decision",
        "",
        f"- **Objective Sealed Technical Gates (`H1`+`H2`+`H3` + all 8 engineering gates):** `{'PASS' if dec['objective_sealed_passed'] else 'FAIL'}`",
        f"- **Blind Listening Panel Status:** `{dec['blind_panel_status']}`",
        f"- **Overall E5 Release Verdict:** `{dec['release_verdict']}`",
        "",
        (
            "Per E0 Spec Section 4 (Go/No-Go), promotion requires **simultaneous** confirmation of: "
            "(1) all hard engineering & stereo safety gates, (2) locked confirmatory CIs on the sealed holdout, "
            "(3) **blind human listening panel perceptual confirmation with zero significant harm**, and "
            "(4) CPU/RSS/coverage budgets. Until the owner completes the loudness-matched blind listening panel "
            "and scores it via `python3 research/e5_blind_panel.py score --panel-dir <panel_dir>`, "
            "the release verdict remains strictly `inconclusive_pending_blind_panel` (or `no_promote` if any objective gate failed) "
            "— never a premature `promote` claim."
        ),
        "",
    ]
    return "\n".join(lines)


def _safe_e5_argv() -> list[str]:
    path_flags = {
        "--manifest", "--audio-root", "--freeze-manifest", "--results-json",
        "--report-md", "--sealed-v1-stub", "--check-against", "--panel-score-json",
        "--custodian-release-token", "--rss-worker-wav",
    }
    out, redact_next = [], False
    for tok in sys.argv:
        if redact_next:
            out.append("<redacted>")
            redact_next = False
            continue
        if tok.startswith("--"):
            flag, _, val = tok.partition("=")
            out.append(flag)
            if flag in path_flags:
                if not val:
                    redact_next = True
                continue
            if val and "/" not in val:
                out.append(val)
        elif "/" not in tok:
            out.append(tok)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--manifest", help="sealed corpus/v1 manifest to evaluate once")
    ap.add_argument("--audio-root", default=None,
                    help="audio root (for paired_synthetic manifests, regenerated in temp dir if omitted)")
    ap.add_argument("--freeze-manifest", default=str(DEFAULT_FREEZE_MANIFEST))
    ap.add_argument("--expected-freeze-sha256", default=EXPECTED_FREEZE_SHA256)
    ap.add_argument("--sealed-v1-stub", default=str(DEFAULT_SEALED_V1_STUB))
    ap.add_argument("--confirm-unseal-once", action="store_true",
                    help="mandatory explicit acknowledgement of one-time sealed unseal")
    ap.add_argument("--custodian-release-token", default="",
                    help="mandatory non-empty custodian authorization token (stored as SHA-256 prefix only)")
    ap.add_argument("--custodian-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--custodian-reason", default="Owner interactive custodian release approval for E5")
    ap.add_argument("--check-against", action="append", default=[],
                    help="other corpus/v1 manifests for cross-manifest group-leakage checks")
    ap.add_argument("--results-json", required=False, default=None)
    ap.add_argument("--report-md", required=False, default=None)
    ap.add_argument("--panel-score-json", default=None,
                    help="optional scored blind-panel JSON from research/e5_blind_panel.py score")
    ap.add_argument("--no-strict-toolchain", action="store_true")
    ap.add_argument("--rss-worker-wav", help=argparse.SUPPRESS)
    ap.add_argument("--rss-worker-bits", type=int, default=24, help=argparse.SUPPRESS)
    ap.add_argument("--rss-worker-ceiling", type=float, default=-1.0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.rss_worker_wav:
        res = _rss_worker_run(Path(args.rss_worker_wav), args.rss_worker_bits, args.rss_worker_ceiling)
        print(json.dumps(res, allow_nan=False))
        return 0

    if not args.manifest:
        print("custodian gate error: --manifest is required", file=sys.stderr)
        return 2
    if not args.confirm_unseal_once:
        print("custodian gate error: refusing to unseal without --confirm-unseal-once", file=sys.stderr)
        return 2
    token = (args.custodian_release_token or "").strip()
    if not token:
        print("custodian gate error: refusing to unseal without non-empty --custodian-release-token",
              file=sys.stderr)
        return 2

    # Gate 1: Verify E4 freeze manifest SHA-256 and ALL 18 frozen implementation hashes BEFORE unsealing.
    try:
        freeze = verify_freeze_integrity(
            Path(args.freeze_manifest),
            expected_freeze_sha256=args.expected_freeze_sha256,
            repo_root=REPO,
            check_toolchain=not args.no_strict_toolchain,
        )
    except FreezeIntegrityError as exc:
        print(f"freeze integrity error: {exc}", file=sys.stderr)
        return 2

    manifest_path = Path(args.manifest).resolve()
    doc = corpus_v1.load(manifest_path)
    if doc.get("tier") != "sealed":
        print(f"manifest error: expected tier='sealed', got {doc.get('tier')!r}", file=sys.stderr)
        return 2

    # Gate 2: Check custodian/pilot/rights BEFORE touching audio bytes.
    pre_errs = []
    if not (doc.get("custodian") or {}).get("name"):
        pre_errs.append("custodian not appointed")
    pre_errs += corpus_v1.pilot_errors(doc.get("pilot"))
    for c in doc.get("cases", []):
        if not corpus_v1.case_allowed(c, "local_inference"):
            pre_errs.append(f"case {c['id']}: operation 'local_inference' not permitted")
    if pre_errs:
        for e in pre_errs:
            print(f"manifest error: {e}", file=sys.stderr)
        return 2

    # Gate 3: Cross-manifest group-leakage check against all other committed corpora.
    default_others = [
        REPO / "corpus" / name
        for name in ("smoke-v1.json", "dev-v1.json", "dev-e3-v1.json", "sealed-v1.json")
        if (REPO / "corpus" / name).resolve() != manifest_path and (REPO / "corpus" / name).is_file()
    ]
    other_paths = [Path(p).resolve() for p in args.check_against] if args.check_against else default_others
    others = [corpus_v1.load(p) for p in other_paths if p != manifest_path]
    leak_errors = corpus_v1.validate_cross_manifest_leakage([doc, *others])
    if leak_errors:
        for e in leak_errors:
            print(f"leakage error: {e}", file=sys.stderr)
        return 2

    rubric = dict(freeze.get("stereo_safety_rubric") or DEFAULT_STEREO_SAFETY_RUBRIC)
    sealed_v1_audit = audit_sealed_v1_stub(Path(args.sealed_v1_stub))
    panel_score = json.loads(Path(args.panel_score_json).read_text()) if args.panel_score_json else None

    def _run_with_audio(audio_root: Path) -> tuple[list[dict], dict]:
        # Materialize paired_synthetic cases if audio_root does not already contain them.
        for case in doc["cases"]:
            if case.get("kind") == "paired_synthetic":
                deg_target = audio_root / case["degraded"]["path"]
                if not deg_target.is_file():
                    corpus_synth.synthesize_case(case, audio_root)
        val_errs = corpus_v1.validate_manifest(doc, check_hashes_root=audio_root)
        if val_errs:
            raise ValueError("manifest validation failed: " + "; ".join(val_errs))

        rows = []
        for case in sorted(doc["cases"], key=lambda c: c["id"]):
            row = evaluate_sealed_case(doc, audio_root, case, rubric)
            rows.append(row)
            cand = row["arms"][CANDIDATE_ARM]
            d_sdr = cand["metrics"].get("masked_delta_sdr_db")
            d_str = "-" if d_sdr is None else f"{d_sdr:+.2f} dB"
            print(
                f"  {row['id']:28s} | {cand['status']:19s} | dSDR={d_str:>9s} | "
                f"safe={cand['stereo_safety']['gates']['passed']}",
                flush=True,
            )

        # Isolated subprocess peak RSS spot-check across the heaviest sealed cases
        # (longest duration, highest sample rate, and a representative 48 kHz defect).
        defects = [c for c in doc["cases"] if c.get("split") != "control"]
        rss_targets = []
        if defects:
            by_dur = max(defects, key=lambda c: (c.get("synthesis", {}).get("seconds", 0), c["rate"]))
            by_rate = max(defects, key=lambda c: (c["rate"], c.get("synthesis", {}).get("seconds", 0)))
            rss_targets = list({by_dur["id"]: by_dur, by_rate["id"]: by_rate, defects[0]["id"]: defects[0]}.values())
        rss_runs = []
        ceiling_db = float(doc.get("output_ceiling_dbfs", -1.0))
        for c in rss_targets:
            wav_p = corpus_v1.resolve(doc, audio_root, c["degraded"], allow_custodian=True)
            bits = int((c.get("synthesis") or {}).get("bits", 24))
            r_res = _spawn_isolated_rss(wav_p, bits, ceiling_db)
            rss_runs.append({"case_id": c["id"], "rate": c["rate"], "bits": bits, **r_res})
        rss_vals = [r["peak_rss_mb"] for r in rss_runs]
        rss_profile = {
            "measured_cases": rss_runs,
            "peak_rss_mb_p50": float(np.percentile(rss_vals, 50)) if rss_vals else None,
            "peak_rss_mb_p95": float(np.percentile(rss_vals, 95)) if rss_vals else None,
            "peak_rss_mb_max": float(np.max(rss_vals)) if rss_vals else None,
        }
        return rows, rss_profile

    try:
        if args.audio_root:
            rows, rss_profile = _run_with_audio(Path(args.audio_root).resolve())
        else:
            with tempfile.TemporaryDirectory(prefix="mr_e5_eval_") as tmp:
                rows, rss_profile = _run_with_audio(Path(tmp))
    except ValueError as exc:
        print(f"manifest error: {_clean_error(str(exc), exc)}", file=sys.stderr)
        return 2

    summary = evaluate_confirmatory_gates(
        rows,
        freeze,
        rss_profile,
        sealed_v1_audit=sealed_v1_audit,
        panel_score=panel_score,
    )

    result_doc = {
        "schema": "e5-sealed-eval/v1",
        "spec": "music-restoration-spec-v1 (section 4, E5)",
        "board_card": 1129,
        "custodian_release": {
            "confirmed_unseal_once": True,
            "authorized_at": args.custodian_date,
            "reason": args.custodian_reason,
            "token_sha256_prefix": hashlib.sha256(token.encode()).hexdigest()[:16],
        },
        "freeze_verification": {
            "verified": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "verified_files_count": len(freeze["toolchain"]["implementation_sha256"]),
            "implementation_sha256": freeze["toolchain"]["implementation_sha256"],
        },
        "sealed_v1_audit": sealed_v1_audit,
        "protocol": {
            "manifest_name": manifest_path.name,
            "manifest_sha256": corpus_v1.manifest_digest(doc),
            "dataset_version": doc["dataset_version"],
            "tier": doc["tier"],
            "n_cases": len(rows),
            "n_defects": sum(1 for r in rows if r["is_defect"]),
            "n_controls": sum(1 for r in rows if not r["is_defect"]),
            "arms": list(EVAL_ARMS),
            "output_ceiling_dbfs": float(doc.get("output_ceiling_dbfs", -1.0)),
            "cross_manifest_leakage_checked": [p.name for p in other_paths],
            "cross_manifest_leakage_errors": leak_errors,
            "toolchain": {
                "detected": corpus_v1.actual_toolchain(),
                "git_commit": _git_commit(),
                "cpu": {
                    "model": platform.processor() or platform.machine(),
                    "logical_cores": os.cpu_count(),
                    "blas_omp_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
                },
            },
            "argv": _safe_e5_argv(),
        },
        "summary": summary,
        "cases": rows,
    }

    report_md = render_e5_report(result_doc)
    if args.results_json:
        p = Path(args.results_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result_doc, indent=2, allow_nan=False) + "\n")
    if args.report_md:
        p = Path(args.report_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report_md)

    dec = summary["decision"]
    print(
        f"e5_sealed_eval: objective_passed={dec['objective_sealed_passed']} "
        f"verdict={dec['release_verdict']} "
        f"failed_gates={dec['failed_engineering_gates']} "
        f"failed_hyps={dec['failed_hypotheses']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
