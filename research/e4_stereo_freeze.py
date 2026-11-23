#!/usr/bin/env python3
"""E4 Stereo/Album Safety Verification, Local CPU Profiling, and E5 Freeze Builder.

    python3 research/e4_stereo_freeze.py --out out/e4-run \
        --freeze-manifest docs/validation/e4-freeze-manifest.json \
        --freeze-report docs/validation/e4-stereo-freeze-report.md

Implements spec v1 sections 3, 4 and card E4 (Board #1127):
  - Suite A: Stereo safety & invariant audit across committed non-sealed corpora
    (`smoke-v1.json`, `dev-v1.json`, `dev-e3-v1.json`). Cross-manifest group-
    leakage audit against `sealed-v1.json` metadata runs strictly AFTER pilot
    registration is computed; sealed audio is NEVER opened, accessed, or
    executed.
  - Suite B: Dedicated synthetic stereo & album safety stress tests (L/R
    correlation, ILD/ITD, stereo image drift, phase coherence/IPD, M/S &
    mono-sum, single-channel L/R clipping, asymmetric dithered clip, transient
    pre-echo, quiet-tail & unchanged bands, negative controls, and two real
    24-bit FLAC multi-track gapless albums through `scripts/restore_safe.py`
    and `scripts/evaluate_album.py`).
  - Suite C: Isolated-subprocess cold/warm local CPU profiling on 5-minute
    stereo FLAC tracks (44.1/48/96 kHz, 16/24-bit) and a multi-track stereo
    album (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`), measuring per-stage
    timings, RTF (`wall_time / audio_time`) p50/p95, `wall_s_per_audio_min`
    p50/p95, and isolated peak RSS (MiB).
  - Suite D: Power-derived pilot registration block and immutable E5 freeze
    manifest + publication-grade Markdown report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import corpus_synth, corpus_v1
from research import declip_metrics as M
from research.run_benchmark import _clean_error, _decode, _git_commit
from rnext import features as F
from rnext.dsp.clip_detector import detect_hard_clip_v2
from rnext.dsp.declip_methods import apply_headroom_policy, declip_aspade
from rnext.dsp.stereo_safety import (
    DEFAULT_STEREO_SAFETY_RUBRIC,
    evaluate_gapless_boundary,
    evaluate_stereo_safety,
)
from scripts.restore_one import encode_flac

FREEZE_FILES = (
    "rnext/dsp/clip_detector.py",
    "rnext/dsp/declip_methods.py",
    "rnext/dsp/stereo_safety.py",
    "rnext/dsp/repair.py",
    "rnext/dsp/selection.py",
    "rnext/dsp/contour.py",
    "rnext/dsp/sparse.py",
    "rnext/dsp/predictive.py",
    "rnext/dsp/declip.py",
    "rnext/features.py",
    "scripts/restore_safe.py",
    "scripts/evaluate_album.py",
    "research/corpus_v1.py",
    "research/corpus_synth.py",
    "research/declip_metrics.py",
    "research/run_benchmark.py",
    "research/e3_bakeoff.py",
    "research/e4_stereo_freeze.py",
)

FROZEN_CANDIDATE = {
    "name": "aspade_mit__v2_windowed",
    "license": "MIT (first-principles implementation)",
    "detector": {
        "id": "v2_windowed",
        "entry": "rnext.dsp.clip_detector.detect_hard_clip_v2",
        "parameters": {
            "window_s": 2.0,
            "hop_s": 1.0,
            "min_confidence": 0.5,
            "bits": "from_source_ffprobe",
        },
    },
    "solver": {
        "id": "aspade_mit",
        "entry": "rnext.dsp.declip_methods.declip_aspade",
        "parameters": {
            "frame_44k": 2048,
            "redundancy": 2,
            "step_k": 4,
            "max_iter": 120,
            "eps": 0.001,
            "batch_size": 64,
        },
    },
    "headroom": {
        "entry": "rnext.dsp.declip_methods.apply_headroom_policy",
        "parameters": {
            "ceiling_db": -1.0,
            "tp_tolerance_db": 0.05,
            "max_attenuation_db": -24.0,
            "tp_measurer": "polyphase_4x",
            "album_policy": "rnext.dsp.stereo_safety.plan_album_headroom",
        },
    },
}

FROZEN_COMPARATORS = [
    {"name": "identity", "role": "no_op_baseline", "parameters": {}},
    {
        "name": "gain_only",
        "role": "level_only_headroom_baseline",
        "parameters": {"ceiling_db": -1.0, "tp_measurer": "polyphase_4x"},
    },
    {
        "name": "current_snapshot_auto",
        "role": "current_production_baseline",
        "detector": {"entry": "rnext.dsp.sparse.hard_clip_mask"},
        "solver": {
            "entry": "rnext.dsp.repair.select_restoration",
            "parameters": {"mode": "auto", "declip_engine": "current_snapshot", "ceiling_dbfs": -1.0},
        },
    },
    {
        "name": "ss_pew_mit__v2_windowed",
        "role": "primary_external_dsp_comparator",
        "detector": {"entry": "rnext.dsp.clip_detector.detect_hard_clip_v2"},
        "solver": {
            "entry": "rnext.dsp.declip_methods.declip_ss_pew",
            "parameters": {
                "frame_44k": 2048,
                "redundancy": 2,
                "outer_iters": 3,
                "dr_iters": 60,
                "gamma": 0.08,
                "delta": 0.01,
            },
        },
    },
    {
        "name": "reg_ar_mit__v2_windowed",
        "role": "secondary_external_dsp_comparator",
        "detector": {"entry": "rnext.dsp.clip_detector.detect_hard_clip_v2"},
        "solver": {
            "entry": "rnext.dsp.declip_methods.declip_regularized_ar",
            "parameters": {
                "frame_44k": 1024,
                "order": 48,
                "ar_iters": 12,
                "ar_reg": 1e-5,
                "qp_reg": 1e-7,
                "qp_iters": 60,
            },
        },
    },
]


def _safe_e4_argv() -> list[str]:
    """Redact all path-bearing CLI flag values from logged argv."""
    path_flags = {"--manifest", "--out", "--freeze-manifest", "--freeze-report",
                  "--reuse-suite-c", "--profile-worker-in", "--profile-worker-out"}
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


def _implementation_hashes() -> dict[str, str]:
    out = {}
    for rel in FREEZE_FILES:
        p = REPO / rel
        if p.exists():
            out[rel] = corpus_v1.digest(p)
    return out


def _write_flac(path: Path, pcm: np.ndarray, sr: int, *, bits: int = 24, seed: int = 1234) -> str:
    """Write deterministic 16/24-bit stereo FLAC with TPDF dither and return SHA-256."""
    rng = np.random.default_rng(seed)
    dith = rng.random(pcm.shape) - rng.random(pcm.shape)
    scale = float(1 << (bits - 1))
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    q = np.clip(np.round(np.clip(pcm, -1.0, 1.0) * scale + dith), lo, hi).astype(np.int32)
    tmp_wav = path.with_suffix(".tmp.wav")
    try:
        encode_flac(q, sr, path, tmp_wav, bits=bits)
    finally:
        tmp_wav.unlink(missing_ok=True)
    return corpus_v1.digest(path)


def _run_candidate_pcm(degraded: np.ndarray, sr: int, bits: int, ceiling_db: float = -1.0):
    """Run frozen E4 candidate (`aspade_mit__v2_windowed`) on PCM array."""
    x32 = degraded.astype(np.float32)
    pred_mask, diag = detect_hard_clip_v2(x32, sr, bits=bits)
    if not pred_mask.any():
        return x32.copy(), x32.copy(), pred_mask, {
            "status": "kept_original",
            "reason": "no_supported_hard_clip_pattern",
            "output_gain_db": 0.0,
            "track_confidence": float(diag.get("track_confidence", 0.0)),
        }
    raw, info = declip_aspade(x32, sr, pred_mask)
    delivered, head = apply_headroom_policy(raw, x32, sr, pred_mask, ceiling_db=ceiling_db)
    return delivered, raw, pred_mask, {
        **head,
        "track_confidence": float(diag.get("track_confidence", 0.0)),
        "solver_method": info.get("method", "aspade_mit"),
    }


# ------------------------------------------- Suite A: Existing non-sealed corpora

def run_suite_a_corpora(manifests: list[Path], work_dir: Path) -> dict:
    """Verify stereo safety, zero FP, and zero polarity violations on smoke/dev corpora."""
    results = {}
    for mpath in manifests:
        doc = corpus_v1.load(mpath)
        if doc.get("tier") == "sealed":
            raise PermissionError(f"E4 refuses sealed manifest: {mpath.name}")
        view = corpus_v1.without_sealed(doc)
        audio_dir = work_dir / "suite_a" / mpath.stem
        for case in view["cases"]:
            specs = corpus_synth.synthesize_case(case, audio_dir)
            for role in ("clean", "degraded", "mask"):
                if specs[role]["sha256"] != case[role]["sha256"]:
                    raise ValueError(f"{mpath.name}:{case['id']}: regenerated {role} hash drifted")
        errs = corpus_v1.validate_manifest(view, check_hashes_root=audio_dir)
        if errs:
            raise ValueError(f"{mpath.name} validation errors: {errs}")

        ceiling_db = float(view.get("output_ceiling_dbfs", -1.0))
        cases_out = []
        control_fp = 0
        polarity_viol_total = 0
        max_reliable_diff = 0.0
        n_defects = 0
        n_processed_defects = 0
        gate_failures = []

        for case in sorted(view["cases"], key=lambda c: c["id"]):
            deg_p = corpus_v1.resolve(view, audio_dir, case["degraded"])
            cln_p = corpus_v1.resolve(view, audio_dir, case["clean"])
            msk_p = corpus_v1.resolve(view, audio_dir, case["mask"])
            degraded, sr, info = _decode(deg_p)
            clean, _, _ = _decode(cln_p)
            true_mask = corpus_synth.decode_mask(msk_p.read_bytes())
            bits = int(info["bps"] or 16)
            is_defect = bool(true_mask.any())
            # Identify localized gain-jump cases (E3 diagnostic: `late_clip` with defect_gain > 1)
            # where whole-track clean reference differs by an unrecorded local gain jump.
            syn = case.get("synthesis") or {}
            has_local_gain_jump = (syn.get("operator") == "late_clip"
                                   and float(syn.get("defect_gain", 1.0)) != 1.0)

            t0 = time.monotonic()
            delivered, raw, pred_mask, meta = _run_candidate_pcm(degraded, sr, bits, ceiling_db)
            wall_s = time.monotonic() - t0

            if is_defect:
                n_defects += 1
                if meta["status"] == "processed_candidate":
                    n_processed_defects += 1
            else:
                if pred_mask.any() or meta["status"] != "kept_original":
                    control_fp += 1

            wf = M.evaluate_waveform(clean, degraded, delivered, true_mask, raw_candidate=raw)
            safety = evaluate_stereo_safety(
                clean, degraded, delivered, sr, mask=true_mask,
                solver_mask=pred_mask, raw_candidate=raw, is_defect=is_defect,
            )
            sm = safety["metrics"]
            if sm["polarity_violations"]:
                polarity_viol_total += int(sm["polarity_violations"])
            if sm["max_reliable_diff_pre_gain"] is not None:
                max_reliable_diff = max(max_reliable_diff, float(sm["max_reliable_diff_pre_gain"]))

            # On local-gain-jump synthetic late_clip cases, clean vs degraded itself has
            # an artificial gain step after 90s; check pre-gain & stereo consistency invariants.
            if not safety["gates"]["passed"] and not has_local_gain_jump:
                gate_failures.append({"case_id": case["id"], "violations": safety["gates"]["violations"]})

            cases_out.append({
                "id": case["id"],
                "profile": case.get("profile"),
                "rate": sr,
                "bits": bits,
                "is_defect": is_defect,
                "has_local_gain_jump": has_local_gain_jump,
                "status": meta["status"],
                "output_gain_db": float(meta.get("output_gain_db", 0.0)),
                "masked_delta_sdr_db": wf["masked_delta_sdr_db"],
                "whole_track_sisdri_db": wf["whole_track_sisdri_db"],
                "polarity_violations": sm["polarity_violations"],
                "max_reliable_diff": sm["max_reliable_diff_pre_gain"],
                "outside_true_mask_max_diff": sm["outside_true_mask_max_diff_pre_gain"],
                "lr_corr_drift_vs_clean": sm["lr_corr_drift_vs_clean"],
                "ild_drift_vs_clean_db": sm["ild_drift_vs_clean_db"],
                "local_ild_drift_p95_db": sm["local_ild_drift_p95_db"],
                "phase_coherence_drift_vs_clean": sm["phase_coherence_drift_vs_clean"],
                "ipd_error_mean_deg": sm["ipd_error_mean_deg"],
                "masked_ipd_error_mean_deg": sm["masked_ipd_error_mean_deg"],
                "itd_drift_samples": sm["itd_drift_samples"],
                "ms_ratio_drift_vs_clean_db": sm["ms_ratio_drift_vs_clean_db"],
                "mono_sum_delta_sdr_db": sm["mono_sum_delta_sdr_db"],
                "mono_sum_level_drift_vs_clean_db": sm["mono_sum_level_drift_vs_clean_db"],
                "pre_echo_rise_db": sm["pre_echo_rise_db"],
                "quiet_tail_rms_drift_db": sm["quiet_tail_rms_drift_db"],
                "band_shape_rms_vs_clean_db": sm["band_shape_rms_vs_clean_db"],
                "safety_passed": safety["gates"]["passed"],
                "safety_violations": safety["gates"]["violations"],
                "wall_s": float(wall_s),
            })

        non_jump_def = [c for c in cases_out if c["is_defect"] and not c["has_local_gain_jump"]]
        def _max_field(rows, field):
            vals = [float(r[field]) for r in rows if r.get(field) is not None]
            return float(max(vals)) if vals else None

        results[mpath.stem] = {
            "manifest_path": (str(mpath.relative_to(REPO))
                              if mpath.is_relative_to(REPO) else mpath.name),
            "manifest_sha256": corpus_v1.manifest_digest(doc),
            "n_cases": len(cases_out),
            "n_defects": n_defects,
            "n_controls": len(cases_out) - n_defects,
            "defect_coverage": float(n_processed_defects / max(1, n_defects)),
            "control_fp": control_fp,
            "polarity_violations_total": polarity_viol_total,
            "max_reliable_diff_pre_gain": max_reliable_diff,
            "worst_lr_corr_drift": _max_field(non_jump_def, "lr_corr_drift_vs_clean"),
            "worst_ild_drift_db": _max_field(non_jump_def, "ild_drift_vs_clean_db"),
            "worst_local_ild_p95_db": _max_field(non_jump_def, "local_ild_drift_p95_db"),
            "worst_phase_coh_drift": _max_field(non_jump_def, "phase_coherence_drift_vs_clean"),
            "worst_ipd_mean_deg": _max_field(non_jump_def, "ipd_error_mean_deg"),
            "worst_masked_ipd_mean_deg": _max_field(non_jump_def, "masked_ipd_error_mean_deg"),
            "worst_itd_drift_samples": _max_field(non_jump_def, "itd_drift_samples"),
            "worst_ms_drift_db": _max_field(non_jump_def, "ms_ratio_drift_vs_clean_db"),
            "worst_pre_echo_rise_db": _max_field(non_jump_def, "pre_echo_rise_db"),
            "gate_failures": gate_failures,
            "all_gates_passed": (control_fp == 0 and polarity_viol_total == 0
                                 and max_reliable_diff == 0.0 and not gate_failures),
            "cases": cases_out,
        }
    return results


# ----------------------------- Suite B: Dedicated Stereo & Album Safety Suite

def _synth_ild_itd_stereo(seed: int, sr: int, seconds: float, *, itd_samples: int = 16,
                          ild_db: float = 3.5) -> np.ndarray:
    """Stereo music signal with explicit ILD (L hotter than R) and ITD lag on R."""
    base = corpus_synth._clean_music_signal(seed=seed, rate=sr, seconds=seconds + 0.05)
    n = round(sr * seconds)
    # Mix a strongly correlated stereo core delayed on R by `itd_samples` plus panned overtones.
    core = 0.5 * (base[:, 0] + base[:, 1])
    out = np.zeros((n, 2), dtype=np.float64)
    g_r = 10.0 ** (-ild_db / 20.0)
    out[:, 0] = 0.70 * core[:n] + 0.30 * base[:n, 0]
    out[:, 1] = g_r * (0.70 * core[itd_samples:itd_samples + n] + 0.30 * base[:n, 1])
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out *= 0.95 / peak
    return out


def _synth_pre_echo_stress(seed: int, sr: int, seconds: float) -> np.ndarray:
    """Stereo signal with quiet pre-onset floors (-45 dB) followed by sharp loud attacks."""
    rng = np.random.default_rng(seed)
    n = round(sr * seconds)
    t = np.arange(n, dtype=np.float64) / sr
    # Quiet continuous bed at -45 dBFS
    bed = 0.005 * np.column_stack([
        np.sin(2.0 * np.pi * 196.0 * t) + 0.5 * np.sin(2.0 * np.pi * 392.0 * t),
        np.sin(2.0 * np.pi * 196.0 * t + 0.15) + 0.5 * np.sin(2.0 * np.pi * 392.0 * t + 0.25),
    ])
    out = bed + rng.normal(0.0, 2e-5, (n, 2))
    # Insert sharp damped tonal+percussive hits every 0.45 s starting at 0.3 s.
    pos = round(0.30 * sr)
    step = round(0.45 * sr)
    dur = round(0.14 * sr)
    while pos + dur < n:
        k = np.arange(dur, dtype=np.float64) / sr
        env = np.exp(-k / 0.025)
        hit = env * (
            0.65 * np.sin(2.0 * np.pi * 330.0 * k)
            + 0.35 * np.sin(2.0 * np.pi * 990.0 * k + 0.3)
            + 0.20 * np.sin(2.0 * np.pi * 2400.0 * k)
        )
        out[pos:pos + dur, 0] += 0.95 * hit
        out[pos:pos + dur, 1] += 0.85 * hit
        pos += step
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out *= 0.96 / peak
    return out


def _run_restore_safe_cli(src_dir: Path, out_dir: Path, *, declip_engine: str = "aspade_v2",
                          album_headroom: bool = True) -> dict:
    cmd = [
        sys.executable, str(REPO / "scripts" / "restore_safe.py"),
        "--src", str(src_dir), "--out", str(out_dir),
        "--mode", "declip", "--declip-engine", declip_engine,
    ]
    if album_headroom:
        cmd.append("--album-headroom")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"restore_safe failed (rc={proc.returncode}): {_clean_error(proc.stderr)}")
    report = json.loads((out_dir / "report.json").read_text())
    album_safety = (json.loads((out_dir / "album_safety.json").read_text())
                    if (out_dir / "album_safety.json").exists() else None)
    return {"report": report, "album_safety": album_safety}


def run_suite_b_stereo_and_album(work_dir: Path, *, quick: bool = False) -> dict:
    """Run dedicated stereo safety stress tests and real 24-bit FLAC album runs."""
    dur = 1.5 if quick else 4.0
    stress_specs = [
        ("stereo-coherent-corr-44k", 44100, 24,
         lambda sr: corpus_synth._clean_music_signal(seed=8101, rate=sr, seconds=dur),
         {"operator": "hard_clip", "threshold_dbfs": -5.0, "bits": 24}),
        ("stereo-ild-itd-panned-48k", 48000, 24,
         lambda sr: _synth_ild_itd_stereo(seed=8102, sr=sr, seconds=dur, itd_samples=16, ild_db=3.5),
         {"operator": "hard_clip", "threshold_dbfs": -5.0, "bits": 24}),
        ("stereo-single-ch-left-44k", 44100, 24,
         lambda sr: corpus_synth._clean_music_signal(seed=8103, rate=sr, seconds=dur),
         {"operator": "hard_clip", "threshold_dbfs": -6.0, "channel_mode": "left_only", "bits": 24}),
        ("stereo-single-ch-right-asym-16b", 48000, 16,
         lambda sr: corpus_synth._clean_music_signal(seed=8104, rate=sr, seconds=dur),
         {"operator": "hard_clip", "threshold_dbfs": -5.0, "pos_scale": 0.72, "neg_scale": 1.0,
          "channel_mode": "right_only", "bits": 16, "dither": True}),
        ("stereo-transient-pre-echo-48k", 48000, 24,
         lambda sr: _synth_pre_echo_stress(seed=8105, sr=sr, seconds=dur),
         {"operator": "hard_clip", "threshold_dbfs": -5.5, "bits": 24}),
        ("stereo-quiet-tail-96k", 96000, 24,
         lambda sr: corpus_synth._clean_music_signal(seed=8106, rate=sr, seconds=1.5 if quick else 2.5, quiet_tail=True),
         {"operator": "hard_clip", "threshold_dbfs": -5.0, "bits": 24}),
        ("ctrl-clean-stereo-44k", 44100, 24,
         lambda sr: corpus_synth._clean_music_signal(seed=8107, rate=sr, seconds=dur),
         {"operator": "clean", "bits": 24}),
        ("ctrl-clean-percussive-48k", 48000, 24,
         lambda sr: corpus_synth._clean_music_signal(seed=8108, rate=sr, seconds=dur, percussive=True),
         {"operator": "clean", "bits": 24}),
        ("ctrl-intentional-tanh-44k", 44100, 16,
         lambda sr: corpus_synth._clean_music_signal(seed=8109, rate=sr, seconds=dur),
         {"operator": "intentional_saturation", "drive": 5.0, "bits": 16}),
        ("ctrl-soft-limiter-48k", 48000, 24,
         lambda sr: corpus_synth._clean_music_signal(seed=8110, rate=sr, seconds=dur),
         {"operator": "soft_limiter", "knee": 0.70, "bits": 24}),
    ]

    stress_rows = []
    stress_dir = work_dir / "suite_b" / "stress"
    stress_dir.mkdir(parents=True, exist_ok=True)
    for idx, (cid, sr, bits, gen_fn, op_spec) in enumerate(stress_specs):
        clean = gen_fn(sr)
        deg, true_mask, _ = corpus_synth.apply_operator(clean, sr, op_spec, seed=8200 + idx)
        is_defect = bool(true_mask.any())
        # Encode degraded stereo track to real FLAC and strictly decode back.
        flac_in = stress_dir / f"{cid}.flac"
        _write_flac(flac_in, deg, sr, bits=bits, seed=8300 + idx)
        deg_dec, sr_dec = F.decode_f32(str(flac_in), strict=True, channels=None)
        assert sr_dec == sr and deg_dec.shape == deg.shape

        delivered, raw, pred_mask, meta = _run_candidate_pcm(deg_dec, sr, bits, ceiling_db=-1.0)
        wf = M.evaluate_waveform(clean, deg_dec, delivered, true_mask, raw_candidate=raw)
        safety = evaluate_stereo_safety(
            clean, deg_dec, delivered, sr, mask=true_mask,
            solver_mask=pred_mask, raw_candidate=raw, is_defect=is_defect,
        )
        # Single-channel invariance check: if only one channel was clipped, the clean
        # opposite channel must be bit-identical before gain.
        untouched_ch_diff = None
        if op_spec.get("channel_mode") == "left_only":
            untouched_ch_diff = float(np.max(np.abs(raw[:, 1] - deg_dec[:, 1])))
        elif op_spec.get("channel_mode") == "right_only":
            untouched_ch_diff = float(np.max(np.abs(raw[:, 0] - deg_dec[:, 0])))

        sm = safety["metrics"]
        stress_rows.append({
            "id": cid,
            "rate": sr,
            "bits": bits,
            "is_defect": is_defect,
            "channel_mode": op_spec.get("channel_mode", "both"),
            "status": meta["status"],
            "output_gain_db": float(meta.get("output_gain_db", 0.0)),
            "masked_delta_sdr_db": wf["masked_delta_sdr_db"],
            "whole_track_sisdri_db": wf["whole_track_sisdri_db"],
            "polarity_violations": sm["polarity_violations"],
            "max_reliable_diff": sm["max_reliable_diff_pre_gain"],
            "outside_true_mask_max_diff": sm["outside_true_mask_max_diff_pre_gain"],
            "untouched_channel_max_diff": untouched_ch_diff,
            "lr_corr_drift_vs_clean": sm["lr_corr_drift_vs_clean"],
            "ild_drift_vs_clean_db": sm["ild_drift_vs_clean_db"],
            "local_ild_drift_p95_db": sm["local_ild_drift_p95_db"],
            "phase_coherence_drift_vs_clean": sm["phase_coherence_drift_vs_clean"],
            "ipd_error_mean_deg": sm["ipd_error_mean_deg"],
            "masked_ipd_error_mean_deg": sm["masked_ipd_error_mean_deg"],
            "itd_drift_samples": sm["itd_drift_samples"],
            "itd_drift_us": sm["itd_drift_us"],
            "ms_ratio_drift_vs_clean_db": sm["ms_ratio_drift_vs_clean_db"],
            "mono_sum_delta_sdr_db": sm["mono_sum_delta_sdr_db"],
            "mono_sum_level_drift_vs_clean_db": sm["mono_sum_level_drift_vs_clean_db"],
            "pre_echo_rise_db": sm["pre_echo_rise_db"],
            "quiet_tail_rms_drift_db": sm["quiet_tail_rms_drift_db"],
            "band_shape_rms_vs_clean_db": sm["band_shape_rms_vs_clean_db"],
            "safety_passed": safety["gates"]["passed"] and (untouched_ch_diff in (None, 0.0)),
            "safety_violations": safety["gates"]["violations"],
        })

    # Real FLAC Album 1: 4-track gapless all-clipped stereo album (unified album gain!).
    alb1_src = work_dir / "suite_b" / "album_all_clipped_src"
    alb1_out = work_dir / "suite_b" / "album_all_clipped_out"
    alb1_eval = work_dir / "suite_b" / "album_all_clipped_eval"
    alb1_src.mkdir(parents=True, exist_ok=True)
    sr_alb = 44100
    seg_s = 2.0 if quick else 4.0
    # Tracks 01 & 02 form a continuous gapless stereo passage split mid-phrase!
    cont_clean = corpus_synth._clean_music_signal(seed=8401, rate=sr_alb, seconds=2.0 * seg_s)
    cont_deg, _, _ = corpus_synth.apply_operator(
        cont_clean, sr_alb, {"operator": "hard_clip", "threshold_dbfs": -5.5, "bits": 24}, seed=8401)
    mid = cont_deg.shape[0] // 2
    t1_deg, t2_deg = cont_deg[:mid], cont_deg[mid:]
    # Track 03: single-channel left clip (different headroom demand)
    t3_clean = corpus_synth._clean_music_signal(seed=8403, rate=sr_alb, seconds=seg_s)
    t3_deg, _, _ = corpus_synth.apply_operator(
        t3_clean, sr_alb, {"operator": "hard_clip", "threshold_dbfs": -6.5, "channel_mode": "left_only", "bits": 24}, seed=8403)
    # Track 04: heavier symmetric clip (sets the album-wide attenuation ceiling)
    t4_clean = corpus_synth._clean_music_signal(seed=8404, rate=sr_alb, seconds=seg_s)
    t4_deg, _, _ = corpus_synth.apply_operator(
        t4_clean, sr_alb, {"operator": "hard_clip", "threshold_dbfs": -8.0, "bits": 24}, seed=8404)

    for fname, pcm, sd in (
        ("01 - Gapless Part I.flac", t1_deg, 8411),
        ("02 - Gapless Part II.flac", t2_deg, 8412),
        ("03 - Left Clip.flac", t3_deg, 8413),
        ("04 - Heavy Clip.flac", t4_deg, 8414),
    ):
        _write_flac(alb1_src / fname, pcm, sr_alb, bits=24, seed=sd)

    alb1_run = _run_restore_safe_cli(alb1_src, alb1_out, declip_engine="aspade_v2", album_headroom=True)
    eval_proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "evaluate_album.py"),
         "--src", str(alb1_src), "--restored", str(alb1_out), "--out", str(alb1_eval)],
        capture_output=True, text=True, check=False,
    )
    if eval_proc.returncode != 0:
        raise RuntimeError(f"evaluate_album failed: {_clean_error(eval_proc.stderr)}")
    alb1_eval_summary = json.loads((alb1_eval / "summary.json").read_text())

    # Also verify Track 01 -> Track 02 gapless seam directly against clean ground truth!
    d1, _ = F.decode_f32(str(alb1_out / "01 - Gapless Part I.flac"), strict=True, channels=None)
    d2, _ = F.decode_f32(str(alb1_out / "02 - Gapless Part II.flac"), strict=True, channels=None)
    gapless_vs_clean = evaluate_gapless_boundary(cont_clean[:mid], cont_clean[mid:], d1, d2, sr_alb)

    # Real FLAC Album 2: 3-track mixed album (1 clipped + 1 clean gapless + 1 intentional tanh).
    alb2_src = work_dir / "suite_b" / "album_mixed_src"
    alb2_out = work_dir / "suite_b" / "album_mixed_out"
    alb2_src.mkdir(parents=True, exist_ok=True)
    m_clean = corpus_synth._clean_music_signal(seed=8501, rate=sr_alb, seconds=2.0 * seg_s)
    # Realistic hot-clipped master on track 1 (pre-clip peaks > 0 dBFS, clipped at -1.0 dBFS)
    # requiring ~2.5 dB headroom attenuation, incompatible with immutable kept track 2.
    m_t1, _, _ = corpus_synth.apply_operator(
        m_clean[:mid], sr_alb,
        {"operator": "hard_clip", "threshold_dbfs": -1.0, "defect_gain": 1.35, "bits": 24},
        seed=8501,
    )
    m_t2 = m_clean[mid:]  # clean unclipped continuation -> kept_original!
    m_t3_base = corpus_synth._clean_music_signal(seed=8503, rate=sr_alb, seconds=seg_s)
    m_t3, _, _ = corpus_synth.apply_operator(
        m_t3_base, sr_alb, {"operator": "intentional_saturation", "drive": 5.0, "bits": 24}, seed=8503)

    alb2_src_hashes = {}
    for fname, pcm, sd in (
        ("01 - Clipped Track.flac", m_t1, 8511),
        ("02 - Clean Continuation.flac", m_t2, 8512),
        ("03 - Intentional Saturation.flac", m_t3, 8513),
    ):
        alb2_src_hashes[fname] = _write_flac(alb2_src / fname, pcm, sr_alb, bits=24, seed=sd)

    alb2_run = _run_restore_safe_cli(alb2_src, alb2_out, declip_engine="aspade_v2", album_headroom=True)
    kept_byte_exact = all(
        corpus_v1.digest(alb2_out / fn) == alb2_src_hashes[fn]
        for fn in ("02 - Clean Continuation.flac", "03 - Intentional Saturation.flac")
    )

    alb1_plan = alb1_run["album_safety"]["headroom_plan"]
    alb2_plan = alb2_run["album_safety"]["headroom_plan"]
    alb1_tp_ok = all(r["delivery_qc"]["delivered_tp_dbfs"] <= -1.0 + 0.05 for r in alb1_run["report"])
    alb1_clip_zero = all(r["delivery_qc"]["delivered_clip_samples"] == 0 for r in alb1_run["report"])

    stress_passed = all(r["safety_passed"] for r in stress_rows)
    stress_ctrl_fp = sum(1 for r in stress_rows if (not r["is_defect"]) and r["status"] != "kept_original")
    alb1_passed = (
        alb1_plan["policy"] == "unified_album_gain"
        and alb1_plan["compatible"] is True
        and alb1_plan["needs_review"] is False
        and alb1_plan["auto_album_delivery"] is True
        and alb1_plan["max_relative_level_shift_db"] == 0.0
        and alb1_tp_ok and alb1_clip_zero
        and alb1_run["album_safety"]["all_boundaries_passed"] is True
        and gapless_vs_clean["passed"] is True
        and alb1_eval_summary["verdict_counts"]["orig_better"] == 0
    )
    alb2_passed = (
        alb2_plan["policy"] == "mixed_kept_and_attenuated_incompatible"
        and alb2_plan["compatible"] is False
        and alb2_plan["needs_review"] is True
        and alb2_plan["auto_album_delivery"] is False
        and kept_byte_exact is True
        and alb2_run["report"][0]["status"] == "processed_candidate"
        and alb2_run["report"][1]["status"] == "kept_original"
        and alb2_run["report"][2]["status"] == "kept_original"
    )

    return {
        "stress_cases": stress_rows,
        "stress_control_fp": stress_ctrl_fp,
        "stress_all_passed": stress_passed and stress_ctrl_fp == 0,
        "album_all_clipped": {
            "headroom_plan": alb1_plan,
            "all_delivered_tp_under_ceiling": alb1_tp_ok,
            "all_delivered_zero_clips": alb1_clip_zero,
            "gapless_boundaries_vs_source": alb1_run["album_safety"]["gapless_boundaries"],
            "gapless_part1_part2_vs_clean": gapless_vs_clean,
            "evaluate_album_verdict": alb1_eval_summary["verdict_overall"],
            "evaluate_album_verdict_counts": alb1_eval_summary["verdict_counts"],
            "passed": alb1_passed,
        },
        "album_mixed_kept": {
            "headroom_plan": alb2_plan,
            "kept_tracks_byte_identical": kept_byte_exact,
            "track_statuses": {Path(r["source"]).name: r["status"] for r in alb2_run["report"]},
            "gapless_boundaries_vs_source": alb2_run["album_safety"]["gapless_boundaries"],
            "passed": alb2_passed,
        },
        "all_passed": bool(stress_passed and stress_ctrl_fp == 0 and alb1_passed and alb2_passed),
    }


# ------------------------- Suite C: Isolated Local CPU Cold/Warm Profiling

def _true_process_peak_rss_mb() -> float:
    """True isolated peak RSS (MiB) of THIS process after execve().

    On Linux, `ru_maxrss` of a forked+exec'd child inherits the parent's peak
    high-water mark, whereas `/proc/self/status` `VmHWM` tracks only this
    process's own resident high-water mark after execve().
    """
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _profile_worker_clean(flac_path: Path, out_flac: Path) -> dict:
    """Single clean pass in isolated subprocess for exact cold/warm timing & peak RSS."""
    t_total0 = time.monotonic()
    info = F.ffprobe_audio(str(flac_path))
    bits = int(info["bps"] or 24)

    t0 = time.monotonic()
    x, sr = F.decode_f32(str(flac_path), strict=True, channels=None)
    t_decode = time.monotonic() - t0
    audio_s = float(len(x) / sr)

    t0 = time.monotonic()
    mask, _diag = detect_hard_clip_v2(x, sr, bits=bits)
    t_detect = time.monotonic() - t0

    t_repair = 0.0
    t_headroom = 0.0
    t_encode_qc = 0.0
    if not mask.any():
        status = "kept_original"
        output_gain_db = 0.0
    else:
        t0 = time.monotonic()
        raw, _info = declip_aspade(x, sr, mask)
        t_repair = time.monotonic() - t0

        t0 = time.monotonic()
        delivered, head = apply_headroom_policy(raw, x, sr, mask, ceiling_db=-1.0)
        t_headroom = time.monotonic() - t0
        status = head["status"]
        output_gain_db = float(head.get("output_gain_db", 0.0))

        if status == "processed_candidate":
            t0 = time.monotonic()
            _write_flac(out_flac, delivered, sr, bits=24, seed=0xC0FFEE)
            dec_back, sr_back = F.decode_f32(str(out_flac), strict=True, channels=None)
            if sr_back != sr or dec_back.shape != delivered.shape:
                status = "qc_fallback_original"
            t_encode_qc = time.monotonic() - t0

    t_total = time.monotonic() - t_total0
    peak_rss_mb = _true_process_peak_rss_mb()
    return {
        "rate": sr,
        "bits": bits,
        "audio_seconds": audio_s,
        "masked_fraction": float(mask.mean()),
        "status": status,
        "output_gain_db": output_gain_db,
        "stages_s": {
            "decode": float(t_decode),
            "detect": float(t_detect),
            "repair": float(t_repair),
            "headroom": float(t_headroom),
            "encode_qc": float(t_encode_qc),
            "total": float(t_total),
        },
        "rtf": float(t_total / max(audio_s, 1e-9)),
        "wall_s_per_audio_min": float(t_total * 60.0 / max(audio_s, 1e-9)),
        "peak_rss_mb": peak_rss_mb,
    }


def _spawn_isolated_profile(flac_path: Path, out_flac: Path) -> dict:
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()),
         "--profile-worker-in", str(flac_path),
         "--profile-worker-out", str(out_flac)],
        capture_output=True, text=True, check=False, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"profile worker failed: {_clean_error(proc.stderr)}")
    return json.loads(proc.stdout)


def _run_suite_c_album_benchmark(prof_dir: Path, *, quick: bool = False) -> dict:
    """Run multi-track stereo FLAC album benchmark via real CLI."""
    dur_alb_track = 3.0 if quick else 150.0
    alb_src = prof_dir / "album_cpu_src"
    alb_out = prof_dir / "album_cpu_out"
    alb_src.mkdir(parents=True, exist_ok=True)
    alb_total_s = 0.0
    for idx, thr in enumerate((-4.0, -4.5, -4.5, -4.0), start=1):
        clean = corpus_synth._clean_music_signal(seed=9200 + idx, rate=44100, seconds=dur_alb_track)
        deg, _, _ = corpus_synth.apply_operator(
            clean, 44100, {"operator": "hard_clip", "threshold_dbfs": thr, "bits": 24}, seed=9200 + idx)
        _write_flac(alb_src / f"{idx:02d} - Track {idx}.flac", deg, 44100, bits=24, seed=9300 + idx)
        del clean, deg
        alb_total_s += dur_alb_track

    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "restore_safe.py"),
         "--src", str(alb_src), "--out", str(alb_out),
         "--mode", "declip", "--declip-engine", "aspade_v2", "--album-headroom"],
        capture_output=True, text=True, check=False, env=env,
    )
    alb_wall_s = time.monotonic() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"CPU album CLI failed: {_clean_error(proc.stderr)}")
    alb_report = json.loads((alb_out / "report.json").read_text())
    alb_safety = json.loads((alb_out / "album_safety.json").read_text())
    return {
        "n_tracks": len(alb_report),
        "total_audio_seconds": alb_total_s,
        "wall_seconds": float(alb_wall_s),
        "rtf": float(alb_wall_s / max(alb_total_s, 1e-9)),
        "wall_s_per_audio_min": float(alb_wall_s * 60.0 / max(alb_total_s, 1e-9)),
        "all_processed": all(r["status"] == "processed_candidate" for r in alb_report),
        "album_policy": alb_safety["headroom_plan"]["policy"],
        "album_gain_db": alb_safety["headroom_plan"]["album_gain_db"],
        "all_boundaries_passed": alb_safety["all_boundaries_passed"],
    }


def _reuse_suite_c_tracks(prior_manifest: Path, work_dir: Path, *, quick: bool = False) -> dict:
    """Reuse verified 5-min single-track CPU profiles when core DSP hashes match,
    while freshly re-running the multi-track album CLI benchmark."""
    prior = json.loads(prior_manifest.read_text())
    prior_hashes = prior["toolchain"]["implementation_sha256"]
    cur_hashes = _implementation_hashes()
    dsp_core = (
        "rnext/dsp/clip_detector.py",
        "rnext/dsp/declip_methods.py",
        "rnext/dsp/contour.py",
        "rnext/features.py",
    )
    for rel in dsp_core:
        if prior_hashes.get(rel) != cur_hashes.get(rel):
            raise ValueError(f"cannot reuse Suite C tracks: {rel} hash changed")
    prof_dir = work_dir / "suite_c"
    prof_dir.mkdir(parents=True, exist_ok=True)
    prior_c = prior["e4_verification"]["suite_c_cpu_profile"]
    return {
        **prior_c,
        "tracks_reused_from_sha256": prior["freeze_sha256"],
        "album_benchmark": _run_suite_c_album_benchmark(prof_dir, quick=quick),
    }


def run_suite_c_cpu_profile(work_dir: Path, *, quick: bool = False) -> dict:
    """Run cold/warm isolated subprocess CPU/RSS benchmarks on 5-min tracks & album."""
    prof_dir = work_dir / "suite_c"
    prof_dir.mkdir(parents=True, exist_ok=True)
    dur_5m = 5.0 if quick else 300.0

    # Build deterministic 5-minute stereo FLAC tracks covering 44.1/48/96 kHz, 16/24-bit,
    # realistic moderate/heavier clipping, and a clean negative control.
    track_specs = [
        ("track-5m-44k-16b-mod", 44100, 16, dur_5m, {"operator": "hard_clip", "threshold_dbfs": -4.0, "bits": 16}, 9101),
        ("track-5m-44k-24b-heavy", 44100, 24, dur_5m, {"operator": "hard_clip", "threshold_dbfs": -5.0, "bits": 24}, 9102),
        ("track-5m-48k-24b-mod", 48000, 24, dur_5m, {"operator": "hard_clip", "threshold_dbfs": -4.0, "bits": 24}, 9103),
        ("track-5m-48k-24b-clean", 48000, 24, dur_5m, {"operator": "clean", "bits": 24}, 9104),
        ("track-5m-96k-24b-mod", 96000, 24, dur_5m, {"operator": "hard_clip", "threshold_dbfs": -4.0, "bits": 24}, 9105),
    ]

    track_runs = []
    for tid, sr, bits, secs, op_spec, seed in track_specs:
        clean = corpus_synth._clean_music_signal(seed=seed, rate=sr, seconds=secs)
        deg, _mask, _ = corpus_synth.apply_operator(clean, sr, op_spec, seed=seed)
        flac_in = prof_dir / f"{tid}.flac"
        _write_flac(flac_in, deg, sr, bits=bits, seed=seed + 10)
        del clean, deg

        # Cold run (fresh subprocess) followed by warm run (fresh subprocess, hot OS page cache).
        cold = _spawn_isolated_profile(flac_in, prof_dir / f"{tid}.cold.out.flac")
        warm = _spawn_isolated_profile(flac_in, prof_dir / f"{tid}.warm.out.flac")
        (prof_dir / f"{tid}.cold.out.flac").unlink(missing_ok=True)
        (prof_dir / f"{tid}.warm.out.flac").unlink(missing_ok=True)
        flac_in.unlink(missing_ok=True)

        track_runs.append({
            "id": tid,
            "rate": sr,
            "bits": bits,
            "audio_seconds": secs,
            "operator": op_spec["operator"],
            "cold": cold,
            "warm": warm,
        })

    album_bench = _run_suite_c_album_benchmark(prof_dir, quick=quick)

    # Aggregate cold/warm RTF, s/min, and peak RSS across all 5-min tracks.
    cold_s_per_min = [r["cold"]["wall_s_per_audio_min"] for r in track_runs]
    warm_s_per_min = [r["warm"]["wall_s_per_audio_min"] for r in track_runs]
    all_s_per_min = cold_s_per_min + warm_s_per_min
    all_rtf = [r["cold"]["rtf"] for r in track_runs] + [r["warm"]["rtf"] for r in track_runs]
    all_rss = [r["cold"]["peak_rss_mb"] for r in track_runs] + [r["warm"]["peak_rss_mb"] for r in track_runs]
    failures = sum(1 for r in track_runs for k in ("cold", "warm")
                   if r[k]["status"] in ("error", "qc_fallback_original"))
    abstentions = sum(1 for r in track_runs for k in ("cold", "warm")
                      if r["operator"] != "clean" and r[k]["status"] == "kept_original")
    clean_fp = sum(1 for r in track_runs for k in ("cold", "warm")
                   if r["operator"] == "clean" and r[k]["status"] != "kept_original")

    return {
        "mode": "quick" if quick else "full_5min",
        "threading_env": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        "tracks": track_runs,
        "summary": {
            "n_tracks": len(track_runs),
            "rtf_p50": float(np.percentile(all_rtf, 50)),
            "rtf_p95": float(np.percentile(all_rtf, 95)),
            "wall_s_per_audio_min_p50": float(np.percentile(all_s_per_min, 50)),
            "wall_s_per_audio_min_p95": float(np.percentile(all_s_per_min, 95)),
            "cold_s_per_audio_min_p95": float(np.percentile(cold_s_per_min, 95)),
            "warm_s_per_audio_min_p95": float(np.percentile(warm_s_per_min, 95)),
            "peak_rss_mb_p50": float(np.percentile(all_rss, 50)),
            "peak_rss_mb_p95": float(np.percentile(all_rss, 95)),
            "peak_rss_mb_max": float(np.max(all_rss)),
            "failures": failures,
            "defect_abstentions": abstentions,
            "control_fp": clean_fp,
        },
        "album_benchmark": album_bench,
    }


# --------------------------- Suite D: Pilot Power & Freeze Manifest Builder

def compute_pilot_registration(e3_manifest_path: Path, suite_c: dict, registered_at: str) -> dict:
    """Derive formal E5 pilot registration values from E3 dev variance + E4 CPU pilot."""
    e3 = json.loads(e3_manifest_path.read_text())
    dev_e3 = e3["runs"]["dev-e3-v1"]
    ci_vs_curr = dev_e3["paired_ci_masked_delta_sdr_db"]["blind:aspade_mit__v2_windowed_vs_current_snapshot_auto"]
    dev_std_db = float(ci_vs_curr["std"])

    # Two-sided paired power calculation for familywise alpha=0.05 with m=3 confirmatory
    # comparisons (Holm-Bonferroni worst-step alpha_eff = 0.05/3), 80% power,
    # target superiority effect = 0.75 dB vs current_snapshot_auto.
    alpha_family = 0.05
    m_planned = 3
    alpha_eff = alpha_family / m_planned
    power_target = 0.80
    target_effect_db = 0.75
    z_alpha = float(stats.norm.ppf(1.0 - alpha_eff / 2.0))
    z_beta = float(stats.norm.ppf(power_target))
    # Add +2 degrees-of-freedom correction for Student-t vs normal quantile.
    n_clusters = math.ceil(((z_alpha + z_beta) * dev_std_db / target_effect_db) ** 2) + 2

    # CPU & RSS budgets directly from measured local CPU pilot (5-min tracks + E3 dev-e3-v1),
    # with explicit 1.25x operational headroom above observed p95 (never an arbitrary SLA).
    e3_dev_p95 = float(dev_e3["blind"]["aspade_mit__v2_windowed"]["wall_s_per_audio_min_p95"])
    c_p95 = float(suite_c["summary"]["wall_s_per_audio_min_p95"])
    cpu_p95_budget = round(max(c_p95, e3_dev_p95) * 1.25, 1)

    rss_obs_max = float(suite_c["summary"]["peak_rss_mb_max"])
    rss_budget = round(max(rss_obs_max * 1.25, 1024.0), 1)

    return {
        "pilot_fields": {
            "sample_size": n_clusters,
            "gate_margin_sdr_db": 1.0,
            "gate_margin_vs_external_db": 0.5,
            "min_coverage": 0.9,
            "cpu_p95_s_per_audio_min": cpu_p95_budget,
            "rss_mb_p95": rss_budget,
            "registered_at": registered_at,
        },
        "derivation": {
            "power_formula": "ceil(((z_{1-alpha_eff/2} + z_{power}) * dev_std / target_effect)^2) + 2",
            "alpha_family": alpha_family,
            "m_planned_comparisons": m_planned,
            "alpha_eff": round(alpha_eff, 6),
            "power_target": power_target,
            "dev_e3_blind_vs_current_std_db": round(dev_std_db, 4),
            "target_detectable_effect_db": target_effect_db,
            "observed_5min_cpu_p95_s_per_audio_min": round(c_p95, 2),
            "observed_dev_e3_cpu_p95_s_per_audio_min": round(e3_dev_p95, 2),
            "observed_5min_peak_rss_mb_max": round(rss_obs_max, 1),
            "headroom_multiplier": 1.25,
        },
    }


def build_freeze_manifest(suite_a: dict, suite_b: dict, suite_c: dict,
                          pilot: dict, cross_leak_errors: list[str],
                          registered_at: str) -> dict:
    """Assemble the immutable E4 freeze & E5 preregistration contract."""
    detected = corpus_v1.actual_toolchain()
    doc = {
        "schema": "e4-freeze/v1",
        "spec": "music-restoration-spec-v1 (sections 3, 4, E4)",
        "board_card": 1127,
        "frozen_at": registered_at,
        "sealed_audio_opened": False,
        "sealed_manifest_metadata_leakage_checked_only": True,
        "sealed_manifest_status": "metadata_only_leakage_checked_post_pilot_audio_custodian_held_until_e5",
        "cross_manifest_leakage_checked": [
            "corpus/smoke-v1.json",
            "corpus/dev-v1.json",
            "corpus/dev-e3-v1.json",
            "corpus/sealed-v1.json",
        ],
        "cross_manifest_leakage_errors": cross_leak_errors,
        "toolchain": {
            "detected": detected,
            "git_base_commit": _git_commit(),
            "implementation_sha256": _implementation_hashes(),
            "reference_cpu": {
                "model": platform.processor() or platform.machine(),
                "logical_cores": os.cpu_count(),
                "blas_omp_threads": 1,
            },
        },
        "frozen_candidate": FROZEN_CANDIDATE,
        "frozen_comparators": FROZEN_COMPARATORS,
        "neural_colab": {
            "arm": "neural_colab",
            "status": "not_run",
            "reason": "colab_connection_not_authorized_or_configured",
            "excluded_from_cpu_release_claim": True,
        },
        "stereo_safety_rubric": DEFAULT_STEREO_SAFETY_RUBRIC,
        "e5_preregistration": {
            "primary_metric": (
                "gain-controlled masked delta SDR (dB) on true clip mask, "
                "delay_samples=0, single shared whole-track stereo scalar gain, "
                "eps=1e-20, all-input accounting (failures and abstentions = 0.0 dB)"
            ),
            "cluster_unit": "independent recording cluster (take/artist); multiple slices collapsed first",
            "ci_method": "95% paired cluster bootstrap (10000 resamples, seed=20261123) + Student-t 95% CI",
            "multiplicity_correction": "Holm-Bonferroni step-down across the 3 pre-planned confirmatory comparisons",
            "locked_hypotheses": [
                {
                    "id": "H1_vs_identity_and_gain_only",
                    "type": "superiority",
                    "comparison": "aspade_mit__v2_windowed vs identity & gain_only",
                    "rule": "95% cluster CI lower bound > +1.0 dB (gate_margin_sdr_db) AND Holm-Bonferroni p < 0.05",
                },
                {
                    "id": "H2_vs_current_production",
                    "type": "superiority",
                    "comparison": "aspade_mit__v2_windowed vs current_snapshot_auto",
                    "rule": "95% cluster CI lower bound > 0.0 dB AND Holm-Bonferroni p < 0.05",
                },
                {
                    "id": "H3_vs_strong_external_dsp",
                    "type": "non_inferiority_with_cpu_safety_advantage",
                    "comparison": "aspade_mit__v2_windowed vs ss_pew_mit__v2_windowed",
                    "rule": (
                        "95% cluster CI lower bound >= -0.50 dB (-gate_margin_vs_external_db) "
                        "with verified faster or equal local CPU wall_s_per_audio_min p95 and "
                        "zero stereo safety rubric violations"
                    ),
                },
            ],
            "hard_engineering_gates": {
                "control_false_positives": 0,
                "polarity_violations_pre_gain": 0,
                "reliable_sample_diff_pre_gain": 0.0,
                "output_tp_ceiling_dbfs": -1.0,
                "output_tp_tolerance_db": 0.05,
                "stereo_safety_rubric_violations": 0,
                "min_defect_coverage": pilot["pilot_fields"]["min_coverage"],
                "cpu_p95_s_per_audio_min_max": pilot["pilot_fields"]["cpu_p95_s_per_audio_min"],
                "rss_mb_p95_max": pilot["pilot_fields"]["rss_mb_p95"],
            },
            "blind_listening_panel_protocol": {
                "loudness_matching": "equal achievable loudness verified <= 0.2 LU after FLAC encode (scripts/listen_pack.py)",
                "design": "double-blind A/B preference + artifact check (BS.1116 reference anchor), randomized blind IDs, custodian-held key, balanced AB/BA order with hidden repeats",
                "comparators": ["identity", "aspade_mit__v2_windowed", "ss_pew_mit__v2_windowed"],
                "cluster_ci": "95% CI accounting for both recording cluster and listener cluster",
                "promotion_requirement": "positive perceptual preference claim confirmed with no significant listening harm; insufficient power or inconclusive -> no-promote/inconclusive",
            },
            "pilot_registration": pilot["pilot_fields"],
            "pilot_derivation": pilot["derivation"],
            "immutability_rule": (
                "Any parameter, detector, solver, or threshold edit after this freeze "
                "invalidates this preregistration and requires a new sealed corpus version."
            ),
        },
        "e4_verification": {
            "suite_a_corpora": suite_a,
            "suite_b_stereo_and_album": suite_b,
            "suite_c_cpu_profile": suite_c,
        },
    }
    all_ok = (
        not cross_leak_errors
        and all(v["all_gates_passed"] for v in suite_a.values())
        and suite_b["all_passed"]
        and suite_c["summary"]["failures"] == 0
        and suite_c["summary"]["control_fp"] == 0
        and suite_c["album_benchmark"]["all_processed"]
        and suite_c["album_benchmark"]["all_boundaries_passed"]
    )
    doc["freeze_verdict"] = {
        "status": "frozen_for_e5" if all_ok else "blocked_safety_or_budget_failure",
        "ready_for_e5_custodian_release": bool(all_ok),
        "sealed_audio_opened": False,
    }
    # Deterministic content hash over the entire freeze specification (excluding the hash field itself).
    doc["freeze_sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return doc


def render_freeze_report(freeze: dict) -> str:
    """Render publication-grade E4 Markdown report from the freeze manifest."""
    def _f(v, fmt="{:.2f}"):
        return "-" if v is None else fmt.format(v)

    tc = freeze["toolchain"]["detected"]
    cpu = freeze["toolchain"]["reference_cpu"]
    sa = freeze["e4_verification"]["suite_a_corpora"]
    sb = freeze["e4_verification"]["suite_b_stereo_and_album"]
    sc = freeze["e4_verification"]["suite_c_cpu_profile"]
    pilot = freeze["e5_preregistration"]["pilot_registration"]
    deriv = freeze["e5_preregistration"]["pilot_derivation"]
    verdict = freeze["freeze_verdict"]

    lines = [
        "# E4 · Stereo/Album Safety Verification & CPU Candidate Freeze",
        "",
        (f"**Card:** Board #1127 · **Spec:** E0 charter + spec v1 (sections 3, 4, E4) · "
         f"**Freeze status:** `{verdict['status']}` · **Sealed audio opened:** `false` "
         "(custodian-held; `corpus/sealed-v1.json` metadata inspected post-pilot for group-leakage audit only, audio never accessed or executed)."),
        "",
        (f"- **Frozen Candidate:** `{freeze['frozen_candidate']['name']}` (`detect_hard_clip_v2` + `declip_aspade` + unified `-1.0 dBTP` headroom)"),
        f"- **Freeze SHA-256:** `{freeze['freeze_sha256']}`",
        f"- **Reference CPU:** `{cpu['model']}` ({cpu['logical_cores']} logical cores, single-threaded BLAS/OpenMP `= {cpu['blas_omp_threads']}`)",
        (f"- **Toolchain Pins:** Python `{tc['python']}`, NumPy `{tc['numpy']}`, SciPy `{tc['scipy']}`, "
         f"FFmpeg `{tc['ffmpeg']}`, libflac `{tc['libflac']}`"),
        f"- **Cross-manifest group leakage (`smoke/dev/dev-e3/sealed`):** `{len(freeze['cross_manifest_leakage_errors'])}` errors",
        f"- **Neural / Google Colab arm:** `{freeze['neural_colab']['status']}` (`{freeze['neural_colab']['reason']}`; excluded from CPU release claim)",
        "- **Machine-verifiable freeze manifest:** [`docs/validation/e4-freeze-manifest.json`](e4-freeze-manifest.json)",
        "",
        "---",
        "",
        "## 1. Suite A — Stereo Safety Audit on Non-Sealed Corpora (`smoke-v1`, `dev-v1`, `dev-e3-v1`)",
        "",
        "| Corpus | Defects / Ctrls | Coverage | Ctrl FP | Pol Viol | Max Rel Diff | Worst L/R Corr Drift | Worst ILD Drift (dB) | Worst Local ILD p95 (dB) | Worst IPD Mean (°) | Worst ITD Drift (samp) | Worst M/S Drift (dB) | Worst Pre-Echo (dB) | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, s in sa.items():
        lines.append(
            f"| `{name}` | {s['n_defects']} / {s['n_controls']} | {100*s['defect_coverage']:.1f}% | "
            f"{s['control_fp']} | {s['polarity_violations_total']} | {_f(s['max_reliable_diff_pre_gain'], '{:.1e}')} | "
            f"{_f(s['worst_lr_corr_drift'], '{:.4f}')} | {_f(s['worst_ild_drift_db'])} | "
            f"{_f(s['worst_local_ild_p95_db'])} | {_f(s['worst_ipd_mean_deg'])} | "
            f"{_f(s['worst_itd_drift_samples'], '{:.3f}')} | {_f(s['worst_ms_drift_db'])} | "
            f"{_f(s['worst_pre_echo_rise_db'])} | `{'PASS' if s['all_gates_passed'] else 'FAIL'}` |"
        )

    lines += [
        "",
        "*(Note: localized gain-jump `late_clip` cases with `defect_gain > 1` are evaluated for pre-gain invariants and zero polarity violations; as documented in E3 Section 4, their whole-track clean reference contains an unrecorded local gain step.)*",
        "",
        "---",
        "",
        "## 2. Suite B — Dedicated Stereo Stress Suite & Real 24-Bit FLAC Albums",
        "",
        "### 2.1 Stereo Image, Phase/IPD, Single-Channel, Pre-Echo & Quiet-Tail Stress Cases (Real FLAC Input)",
        "",
        "| Case | Rate / Bits | Status | ΔMasked SDR (dB) | L/R Corr Drift | ILD Drift (dB) | Local ILD p95 (dB) | IPD Mean (°) | ITD Drift (samp) | M/S Drift (dB) | Mono ΔSDR (dB) | Pre-Echo (dB) | Untouched Ch Diff | Gates |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in sb["stress_cases"]:
        lines.append(
            f"| `{r['id']}` | {r['rate']//1000 if r['rate']%1000==0 else r['rate']/1000:g}k / {r['bits']}b | "
            f"`{r['status']}` | {_f(r['masked_delta_sdr_db'], '{:+.2f}')} | "
            f"{_f(r['lr_corr_drift_vs_clean'], '{:.4f}')} | {_f(r['ild_drift_vs_clean_db'])} | "
            f"{_f(r['local_ild_drift_p95_db'])} | {_f(r['ipd_error_mean_deg'])} | "
            f"{_f(r['itd_drift_samples'], '{:.3f}')} | {_f(r['ms_ratio_drift_vs_clean_db'])} | "
            f"{_f(r['mono_sum_delta_sdr_db'], '{:+.2f}')} | {_f(r['pre_echo_rise_db'])} | "
            f"{_f(r['untouched_channel_max_diff'], '{:.1e}')} | `{'PASS' if r['safety_passed'] else 'FAIL'}` |"
        )

    a1 = sb["album_all_clipped"]
    a2 = sb["album_mixed_kept"]
    g_cln = a1["gapless_part1_part2_vs_clean"]
    lines += [
        "",
        "### 2.2 Real Multi-Track Stereo FLAC Albums (`restore_safe.py --album-headroom` + `evaluate_album.py`)",
        "",
        f"1. **4-Track Gapless All-Clipped Album (`album_all_clipped`)** — `{'PASS' if a1['passed'] else 'FAIL'}`:",
        f"   - Policy: `{a1['headroom_plan']['policy']}`, unified album gain `{a1['headroom_plan']['album_gain_db']:+.2f} dB`, `max_relative_level_shift_db = {a1['headroom_plan']['max_relative_level_shift_db']:.2f} dB`, `auto_album_delivery = {a1['headroom_plan']['auto_album_delivery']}`, `needs_review = {a1['headroom_plan']['needs_review']}`.",
        f"   - Unified `-1.0 dBTP` headroom: all delivered tracks `<= -1.0 + 0.05 dBTP` (`{a1['all_delivered_tp_under_ceiling']}`), `0` clipped samples (`{a1['all_delivered_zero_clips']}`).",
        f"   - Gapless Track 01 → Track 02 seam vs clean ground truth: level jump `{g_cln['seam_level_jump_db']:.4f} dB`, seam peak growth `{g_cln['seam_peak_growth_db']:+.3f} dB`, seam step `{g_cln['seam_step_cand_aligned']:.5f}` vs ref `{g_cln['seam_step_ref']:.5f}` (`passed = {g_cln['passed']}`).",
        f"   - `scripts/evaluate_album.py` verdict: `{a1['evaluate_album_verdict']}` (`{a1['evaluate_album_verdict_counts']}`).",
        f"2. **3-Track Mixed Clipped + Immutable Kept Album (`album_mixed_kept`)** — `{'PASS' if a2['passed'] else 'FAIL'}`:",
        f"   - Policy: `{a2['headroom_plan']['policy']}`, `compatible = {a2['headroom_plan']['compatible']}`, `needs_review = {a2['headroom_plan']['needs_review']}`, `auto_album_delivery = {a2['headroom_plan']['auto_album_delivery']}`, `max_relative_level_shift_db = {a2['headroom_plan']['max_relative_level_shift_db']:.2f} dB`.",
        f"   - Track statuses: `{a2['track_statuses']}`; kept tracks byte-for-byte SHA-256 identical to source (`{a2['kept_tracks_byte_identical']}`).",
        "   - Confirms Spec Section 3 contract: never silently attenuates/re-encodes immutable `kept_original` tracks and never silently delivers a level-shifted album.",
        "",
        "---",
        "",
        "## 3. Suite C — Local CPU Profiling (Cold/Warm, 5-Minute Stereo FLAC Tracks & Album)",
        "",
        "All timings and peak RSS values were measured in **isolated fresh subprocesses** (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`) on real stereo FLAC files via `/proc/self/status` `VmHWM` (true post-`execve` resident high-water mark, zero parent/synthesis contamination; chunked 4x polyphase `true_peak_dbfs`).",
        "",
        "| Track | Rate / Bits | Dur (s) | Masked (%) | Run | Decode (s) | Detect (s) | Repair (s) | Headroom (s) | Encode+QC (s) | Total (s) | RTF | s / audio min | Peak RSS (MiB) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in sc["tracks"]:
        for kind in ("cold", "warm"):
            r = t[kind]
            st = r["stages_s"]
            lines.append(
                f"| `{t['id']}` | {t['rate']//1000 if t['rate']%1000==0 else t['rate']/1000:g}k / {t['bits']}b | "
                f"{t['audio_seconds']:.0f} | {100*r['masked_fraction']:.3f}% | {kind} | "
                f"{st['decode']:.2f} | {st['detect']:.2f} | {st['repair']:.2f} | "
                f"{st['headroom']:.2f} | {st['encode_qc']:.2f} | {st['total']:.2f} | "
                f"{r['rtf']:.3f}x | {r['wall_s_per_audio_min']:.2f} | {r['peak_rss_mb']:.1f} |"
            )

    sm_c = sc["summary"]
    ab_c = sc["album_benchmark"]
    lines += [
        "",
        (f"- **5-min aggregate:** RTF p50 `{sm_c['rtf_p50']:.3f}x`, p95 `{sm_c['rtf_p95']:.3f}x`; "
         f"`wall_s_per_audio_min` p50 `{sm_c['wall_s_per_audio_min_p50']:.2f}`, p95 `{sm_c['wall_s_per_audio_min_p95']:.2f}` "
         f"(cold p95 `{sm_c['cold_s_per_audio_min_p95']:.2f}`, warm p95 `{sm_c['warm_s_per_audio_min_p95']:.2f}`); "
         f"isolated Peak RSS p50 `{sm_c['peak_rss_mb_p50']:.1f} MiB`, p95 `{sm_c['peak_rss_mb_p95']:.1f} MiB`, max `{sm_c['peak_rss_mb_max']:.1f} MiB`; "
         f"failures `{sm_c['failures']}`, defect abstentions `{sm_c['defect_abstentions']}`, control FP `{sm_c['control_fp']}`."),
        (f"- **Multi-track album benchmark ({ab_c['n_tracks']} tracks, {ab_c['total_audio_seconds']/60:.1f} min stereo FLAC):** "
         f"wall `{ab_c['wall_seconds']:.2f} s` (RTF `{ab_c['rtf']:.3f}x`, `{ab_c['wall_s_per_audio_min']:.2f} s/audio min`), "
         f"policy `{ab_c['album_policy']}` (`{ab_c['album_gain_db']:+.2f} dB`), all processed `{ab_c['all_processed']}`, all boundaries passed `{ab_c['all_boundaries_passed']}`."),
        "",
        "---",
        "",
        "## 4. Frozen E5 Preregistration & Pilot Registration Block",
        "",
        "Locked **before** opening sealed audio (`corpus/sealed-v1.json` audio remains custodian-held and unexecuted; manifest group metadata is inspected post-pilot strictly for cross-manifest leakage audit). Any edit after freeze requires a new sealed corpus version.",
        "",
        "| Pilot Field (`corpus_v1.PILOT_FIELDS`) | Frozen Value | Basis |",
        "|---|---:|---|",
        f"| `pilot.sample_size` | `{pilot['sample_size']}` | Paired power (`alpha_eff={deriv['alpha_eff']}`, 80% power, `dev_std={deriv['dev_e3_blind_vs_current_std_db']} dB`, effect `{deriv['target_detectable_effect_db']} dB`) |",
        f"| `pilot.gate_margin_sdr_db` | `{pilot['gate_margin_sdr_db']}` dB | Practical superiority margin vs `identity` and `gain_only` |",
        f"| `pilot.gate_margin_vs_external_db` | `{pilot['gate_margin_vs_external_db']}` dB | Locked non-inferiority margin vs `ss_pew_mit__v2_windowed` (with CPU/safety advantage) |",
        f"| `pilot.min_coverage` | `{pilot['min_coverage']}` | Minimum blind defect attempt coverage (100% observed on `dev-v1` and `dev-e3-v1`) |",
        f"| `pilot.cpu_p95_s_per_audio_min` | `{pilot['cpu_p95_s_per_audio_min']}` s/min | `1.25x` max(dev-e3 p95 `{deriv['observed_dev_e3_cpu_p95_s_per_audio_min']}`, 5-min p95 `{deriv['observed_5min_cpu_p95_s_per_audio_min']}`) |",
        f"| `pilot.rss_mb_p95` | `{pilot['rss_mb_p95']}` MiB | `1.25x` max isolated 5-min peak RSS (`{deriv['observed_5min_peak_rss_mb_max']} MiB`) |",
        f"| `pilot.registered_at` | `{pilot['registered_at']}` | E4 freeze timestamp |",
        "",
        "### Locked Confirmatory Hypotheses for E5 (No Post-Hoc Switching Allowed)",
        "",
    ]
    for h in freeze["e5_preregistration"]["locked_hypotheses"]:
        lines.append(f"- **`{h['id']}` (`{h['type']}`):** {h['comparison']} — {h['rule']}.")
    lines += [
        "",
        "### Blind Listening Panel & Promotion Gate",
        "",
        "- Equal achievable loudness verified `<= 0.2 LU` after FLAC encode (`scripts/listen_pack.py`), double-blind A/B + BS.1116 artifact anchor against `identity` and `ss_pew_mit__v2_windowed`, custodian-held randomization key, 95% CIs over recording and listener clusters.",
        "- Promotion requires passing **all** engineering gates (0 control FP, 0 polarity violations, unified `-1.0 dBTP` headroom, 0 stereo safety rubric violations, CPU p95 & RSS budgets, `>= 90%` coverage), all 3 locked confirmatory statistical hypotheses on sealed, AND perceptual confirmation on the blind panel. Otherwise verdict is `no-promote` / `inconclusive`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--out", help="output directory for E4 run artifacts (protocol.json, summary.json)")
    ap.add_argument("--mode", choices=("full", "quick"), default="full")
    ap.add_argument("--manifest", action="append", default=[],
                    help="override Suite A manifests (defaults to smoke-v1, dev-v1, dev-e3-v1)")
    ap.add_argument("--freeze-manifest", help="optional path to write e4-freeze-manifest.json")
    ap.add_argument("--freeze-report", help="optional path to write e4-stereo-freeze-report.md")
    ap.add_argument("--reuse-suite-c", help="reuse verified 5-min Suite C single-track profiles from prior freeze manifest when core DSP hashes match")
    ap.add_argument("--profile-worker-in", help=argparse.SUPPRESS)
    ap.add_argument("--profile-worker-out", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.profile_worker_in and args.profile_worker_out:
        res = _profile_worker_clean(Path(args.profile_worker_in), Path(args.profile_worker_out))
        print(json.dumps(res, allow_nan=False))
        return 0

    manifests = (
        [Path(m).resolve() for m in args.manifest]
        if args.manifest
        else [
            REPO / "corpus" / "smoke-v1.json",
            REPO / "corpus" / "dev-v1.json",
            REPO / "corpus" / "dev-e3-v1.json",
        ]
    )
    # Refuse any sealed manifest before touching audio bytes.
    for mpath in manifests:
        doc = corpus_v1.load(mpath)
        if doc.get("tier") == "sealed":
            print("manifest error: E4 refuses sealed tier", file=sys.stderr)
            return 2

    quick = args.mode == "quick"
    registered_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with tempfile.TemporaryDirectory(prefix="mr_e4_") as tmp:
        work = Path(tmp)
        suite_a = run_suite_a_corpora(manifests, work)
        suite_b = run_suite_b_stereo_and_album(work, quick=quick)
        if args.reuse_suite_c:
            suite_c = _reuse_suite_c_tracks(Path(args.reuse_suite_c), work, quick=quick)
        else:
            suite_c = run_suite_c_cpu_profile(work, quick=quick)

    # Compute pilot registration strictly BEFORE reading sealed-v1.json metadata.
    pilot = compute_pilot_registration(
        REPO / "docs" / "validation" / "e3-manifests.json",
        suite_c,
        registered_at,
    )

    # Post-pilot metadata-only cross-manifest group-leakage audit (sealed audio is never accessed).
    leak_docs = [
        corpus_v1.load(REPO / "corpus" / name)
        for name in ("smoke-v1.json", "dev-v1.json", "dev-e3-v1.json", "sealed-v1.json")
    ]
    cross_leak_errors = corpus_v1.validate_cross_manifest_leakage(leak_docs)
    if cross_leak_errors:
        for e in cross_leak_errors:
            print(f"leakage error: {e}", file=sys.stderr)
        return 2

    freeze = build_freeze_manifest(suite_a, suite_b, suite_c, pilot, cross_leak_errors, registered_at)
    report_md = render_freeze_report(freeze)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        if any(out_dir.iterdir()):
            ap.error("output directory must be empty")
        (out_dir / "protocol.json").write_text(json.dumps({
            "schema": "e4-run-protocol/v1",
            "mode": args.mode,
            "argv": _safe_e4_argv(),
            "sealed_audio_opened": False,
            "sealed_manifest_metadata_leakage_checked_only": True,
            "freeze_sha256": freeze["freeze_sha256"],
        }, indent=2, allow_nan=False) + "\n")
        (out_dir / "e4-freeze-manifest.json").write_text(
            json.dumps(freeze, indent=2, allow_nan=False) + "\n")
        (out_dir / "e4-stereo-freeze-report.md").write_text(report_md)

    if args.freeze_manifest:
        p = Path(args.freeze_manifest)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(freeze, indent=2, allow_nan=False) + "\n")
    if args.freeze_report:
        p = Path(args.freeze_report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report_md)

    print(
        f"e4_stereo_freeze: verdict={freeze['freeze_verdict']['status']} "
        f"freeze_sha256={freeze['freeze_sha256'][:16]}",
        flush=True,
    )
    return 0 if freeze["freeze_verdict"]["ready_for_e5_custodian_release"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
