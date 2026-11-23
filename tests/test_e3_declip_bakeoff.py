"""E3 hard-declipping bake-off: DSP solvers, blind detector, metrics, and CLI."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from research import corpus_synth, corpus_v1
from research import declip_metrics as M
from rnext.dsp.clip_detector import detect_hard_clip_v2
from rnext.dsp.declip_methods import (
    _build_ar_gram_submatrix,
    _fit_regularized_ar,
    _solve_ar_frame_qp,
    apply_headroom_policy,
    declip_aspade,
    declip_regularized_ar,
    declip_ss_pew,
)

ROOT = Path(__file__).resolve().parents[1]
BAKEOFF_CLI = ROOT / "research" / "e3_bakeoff.py"


def _stereo_harmonic(sr=44100, seconds=0.6, seed=11):
    rng = np.random.default_rng(seed)
    n = round(sr * seconds)
    t = np.arange(n, dtype=np.float64) / sr
    left = (0.35 * np.sin(2 * np.pi * 220 * t)
            + 0.25 * np.sin(2 * np.pi * 440 * t + 0.4)
            + 0.18 * np.sin(2 * np.pi * 660 * t + 0.9))
    right = (0.32 * np.sin(2 * np.pi * 261.63 * t + 0.2)
             + 0.22 * np.sin(2 * np.pi * 523.25 * t + 0.7)
             + 0.16 * np.sin(2 * np.pi * 784.0 * t + 1.1))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 1.5 * t)
    x = np.column_stack([left * env, right * env]) + rng.normal(0, 1e-4, (n, 2))
    x *= 0.95 / np.max(np.abs(x))
    return x.astype(np.float32)


@pytest.mark.parametrize("name,solver", [
    ("aspade_mit", declip_aspade),
    ("ss_pew_mit", declip_ss_pew),
    ("reg_ar_mit", declip_regularized_ar),
])
def test_solvers_improve_stereo_clip_and_preserve_invariants(name, solver):
    sr = 44100
    clean = _stereo_harmonic(sr=sr, seconds=0.5)
    thr = 0.52
    deg = np.clip(clean, -thr, thr).astype(np.float32)
    mask = np.abs(clean) > thr
    assert mask.any()

    raw, info = solver(deg, sr, mask)
    assert raw.shape == deg.shape and raw.dtype == deg.dtype
    assert np.isfinite(raw).all()
    assert info["reliable_samples_preserved"] is True
    np.testing.assert_array_equal(raw[~mask], deg[~mask])
    assert np.all(raw[mask & (deg > 0)] >= deg[mask & (deg > 0)])
    assert np.all(raw[mask & (deg < 0)] <= deg[mask & (deg < 0)])

    delivered, head = apply_headroom_policy(raw, deg, sr, mask)
    assert head["status"] == "processed_candidate"
    ev = M.evaluate_waveform(clean, deg, delivered, mask, raw_candidate=raw)
    assert ev["polarity_violations"] == 0
    assert ev["max_reliable_diff"] == 0.0
    assert ev["masked_delta_sdr_db"] > 2.0, f"{name} delta SDR too low: {ev}"
    assert ev["whole_track_sisdri_db"] > 2.0, f"{name} SI-SDRi too low: {ev}"


@pytest.mark.parametrize("solver", [declip_aspade, declip_ss_pew, declip_regularized_ar])
def test_solvers_empty_mask_is_exact_identity(solver):
    x = _stereo_harmonic(sr=48000, seconds=0.25)
    empty = np.zeros(x.shape, dtype=bool)
    y, info = solver(x, 48000, empty)
    np.testing.assert_array_equal(y, x)
    assert info["masked_samples"] == 0


def test_headroom_policy_rejects_extreme_attenuation():
    sr = 44100
    x = _stereo_harmonic(sr=sr, seconds=0.2)
    deg = np.clip(x, -0.5, 0.5).astype(np.float32)
    mask = np.abs(x) > 0.5
    wild = deg.copy()
    wild[mask] = np.sign(deg[mask]) * 50.0
    delivered, head = apply_headroom_policy(wild, deg, sr, mask)
    np.testing.assert_array_equal(delivered, deg)
    assert head["status"] == "qc_fallback_original"
    assert "over_24_db" in head["reason"]


def test_detector_v2_catches_asymmetric_single_channel_dither_and_late_clip():
    # 1. Single-channel asymmetric sub-full-scale + post-clip TPDF dither (16-bit)
    sr = 48000
    clean = corpus_synth._clean_music_signal(seed=501, rate=sr, seconds=3.0)
    deg, true_mask, _ = corpus_synth.apply_operator(
        clean, sr,
        {"operator": "hard_clip", "threshold_dbfs": -5.0, "pos_scale": 0.7,
         "channel_mode": "left_only", "bits": 16, "dither": True},
        seed=501,
    )
    rng = np.random.default_rng(503)
    q = np.clip(deg, -1, 1) * 32768.0 + (rng.random(deg.shape) - rng.random(deg.shape))
    deg16 = (np.clip(np.round(q), -32768, 32767) / 32768.0).astype(np.float32)
    pred, diag = detect_hard_clip_v2(deg16, sr, bits=16)
    ev = M.evaluate_detection(true_mask, pred)
    assert ev["recall"] > 0.85
    assert ev["precision"] > 0.90
    assert not pred[:, 1].any(), "clean right channel must not be flagged"
    assert diag["track_confidence"] >= 0.5

    # 2. Late-onset clip below earlier unclipped peak (6s clean @0.95, 2s clip @0.65)
    clean_late = corpus_synth._clean_music_signal(seed=502, rate=44100, seconds=8.0)
    deg_late, true_late, _ = corpus_synth.apply_operator(
        clean_late, 44100,
        {"operator": "late_clip", "threshold_dbfs": -3.8, "defect_start_s": 6.0,
         "defect_gain": 2.0, "bits": 16},
        seed=502,
    )
    pred_late, _ = detect_hard_clip_v2(deg_late.astype(np.float32), 44100, bits=16)
    ev_late = M.evaluate_detection(true_late, pred_late)
    assert ev_late["recall"] > 0.90
    assert ev_late["precision"] > 0.95


@pytest.mark.parametrize("op,extra", [
    ("clean", {}),
    ("clean", {"percussive": True}),
    ("clean", {"quiet_tail": True}),
    ("intentional_saturation", {"drive": 4.0}),
    ("intentional_saturation", {"drive": 6.0}),
    ("soft_limiter", {"knee": 0.70}),
])
@pytest.mark.parametrize("bits", [16, 24])
def test_detector_v2_zero_fp_on_negative_controls(op, extra, bits):
    sr = 44100
    clean = corpus_synth._clean_music_signal(
        seed=601, rate=sr, seconds=3.0,
        quiet_tail=extra.get("quiet_tail", False),
        percussive=extra.get("percussive", False),
    )
    deg, _, _ = corpus_synth.apply_operator(
        clean, sr, {"operator": op, "bits": bits, **extra}, seed=601)
    q_scale = float(2 ** (bits - 1))
    deg_q = (np.clip(np.round(deg * q_scale), -q_scale, q_scale - 1) / q_scale).astype(np.float32)
    pred, diag = detect_hard_clip_v2(deg_q, sr, bits=bits)
    assert not pred.any(), f"false positive on {op} ({bits}-bit): {diag}"


def test_reg_ar_gram_submatrix_and_qp_match_exact_ata_at_boundaries():
    """Verify fast H_CC matches exact causal A^T A to machine precision at left/right edges."""
    n, p = 256, 24
    rng = np.random.default_rng(77)
    t = np.arange(n, dtype=np.float64) / 44100.0
    sig = 0.6 * np.sin(2 * np.pi * 220 * t) + 0.35 * np.sin(2 * np.pi * 550 * t + 0.4)
    g = _fit_regularized_ar(sig, order=p, ar_reg=1e-5)

    # Explicit causal AR residual matrix A: (A @ x)[t - p] = sum_{k=0}^p g[k] x[t - k].
    a_mat = np.zeros((n - p, n), dtype=np.float64)
    for m in range(n - p):
        a_mat[m, m:m + p + 1] = g[::-1]
    h_exact = a_mat.T @ a_mat

    # Clipped indices spanning left boundary (<p), exact seam (p-1, p), interior, and right edge (>n-1-p).
    pos_idx = np.array([0, 5, p - 1, p, 120, n - p - 1, n - p, n - 1])
    neg_idx = np.array([10, p + 1, 130, n - p + 3, n - 2])
    c_idx = np.sort(np.concatenate((pos_idx, neg_idx)))

    h_cc_fast = _build_ar_gram_submatrix(c_idx, g, n)
    np.testing.assert_allclose(h_cc_fast, h_exact[np.ix_(c_idx, c_idx)], atol=1e-13, rtol=0.0)

    # Also verify exact unconstrained QP match via _solve_ar_frame_qp (box strictly inactive).
    qp_reg = 1e-5
    obs_free = sig + rng.normal(0.0, 1e-4, size=n)
    obs_free[pos_idx] = sig[pos_idx] - 2.0
    obs_free[neg_idx] = sig[neg_idx] + 2.0
    sol = _solve_ar_frame_qp(obs_free.copy(), obs_free, pos_idx, neg_idx, g, qp_reg=qp_reg, qp_iters=60)
    x_rel = obs_free.copy()
    x_rel[c_idx] = 0.0
    rhs_exact = (h_exact @ x_rel)[c_idx] - qp_reg * obs_free[c_idx]
    x_c_exact = np.linalg.solve(h_exact[np.ix_(c_idx, c_idx)] + qp_reg * np.eye(c_idx.size), -rhs_exact)
    is_pos = np.isin(c_idx, pos_idx)
    assert np.all(x_c_exact[is_pos] >= obs_free[c_idx[is_pos]])
    assert np.all(x_c_exact[~is_pos] <= obs_free[c_idx[~is_pos]])
    np.testing.assert_allclose(sol[c_idx], x_c_exact, atol=1e-10, rtol=0.0)


def test_metrics_gain_invariance_and_statistics():
    clean = _stereo_harmonic(sr=44100, seconds=0.4)
    deg = np.clip(clean, -0.55, 0.55).astype(np.float32)
    mask = np.abs(clean) > 0.55

    ev_id = M.evaluate_waveform(clean, deg, deg, mask, raw_candidate=deg)
    ev_gain = M.evaluate_waveform(clean, deg, deg * 0.5, mask, raw_candidate=deg)
    ev_no_raw = M.evaluate_waveform(clean, deg, deg * 0.5, mask, raw_candidate=None)
    assert ev_id["masked_delta_sdr_db"] == pytest.approx(0.0, abs=1e-9)
    assert ev_gain["masked_delta_sdr_db"] == pytest.approx(0.0, abs=1e-6)
    assert ev_gain["whole_track_sisdri_db"] == pytest.approx(0.0, abs=1e-6)
    assert ev_gain["polarity_violations"] == 0
    assert ev_gain["max_reliable_diff"] == 0.0
    assert ev_no_raw["polarity_violations"] is None
    assert ev_no_raw["max_reliable_diff"] is None
    assert "pre_gain_candidate_unavailable" in ev_no_raw["exclusions"]
    assert ev_gain["masked_sdr_uncompensated_db"] < ev_id["masked_sdr_uncompensated_db"] - 5.0
    json.dumps(ev_gain, allow_nan=False)
    json.dumps(ev_no_raw, allow_nan=False)

    ci = M.paired_cluster_ci({f"c{i}": 2.0 + 0.2 * i for i in range(8)})
    assert ci["n_clusters"] == 8
    assert ci["bootstrap_ci_low"] > 2.0
    assert ci["t_ci_low"] > 2.0
    assert ci["sign_flip_p_value"] < 0.02

    adj = M.holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["b"] == pytest.approx(0.08)
    assert adj["c"] == pytest.approx(0.20)

    rc = M.risk_coverage_curve([
        {"is_defect": True, "confidence": 0.9, "attempted": True, "delta_sdr_db": 6.0, "status": "processed_candidate"},
        {"is_defect": True, "confidence": 0.4, "attempted": False, "delta_sdr_db": None, "status": "kept_original"},
        {"is_defect": False, "confidence": 0.0, "attempted": False, "delta_sdr_db": None, "status": "kept_original"},
    ])
    p0 = next(p for p in rc["points"] if p["threshold"] == 0.0)
    assert p0["coverage"] == pytest.approx(0.5)
    assert p0["all_input_mean_delta_sdr_db"] == pytest.approx(3.0)
    assert p0["conditional_mean_delta_sdr_db"] == pytest.approx(6.0)
    assert p0["control_fp_count"] == 0


def test_committed_e3_manifest_is_valid_and_leak_free():
    dev_e3 = corpus_v1.load(ROOT / "corpus" / "dev-e3-v1.json")
    smoke = corpus_v1.load(ROOT / "corpus" / "smoke-v1.json")
    dev = corpus_v1.load(ROOT / "corpus" / "dev-v1.json")
    sealed = corpus_v1.load(ROOT / "corpus" / "sealed-v1.json")
    assert corpus_v1.validate_manifest(dev_e3) == []
    assert corpus_v1.validate_cross_manifest_leakage([smoke, dev, dev_e3, sealed]) == []
    assert {c["rate"] for c in dev_e3["cases"]} == {44100, 48000, 96000}
    profiles = {c["profile"] for c in dev_e3["cases"]}
    for required in ("asymmetric_sub_full_scale", "dithered_clip", "single_channel_stereo",
                     "defect_after_90s", "clean_control", "intentional_control",
                     "soft_limiter_control"):
        assert required in profiles


def test_e3_bakeoff_cli_refuses_sealed_and_runs_cleanly(tmp_path):
    # 1. Refuses sealed manifest before touching audio bytes.
    proc_sealed = subprocess.run(
        [sys.executable, str(BAKEOFF_CLI), "--manifest", str(ROOT / "corpus" / "sealed-v1.json"),
         "--audio-root", str(tmp_path), "--out", str(tmp_path / "sealed-out")],
        capture_output=True, text=True, check=False,
    )
    assert proc_sealed.returncode == 2
    assert not (tmp_path / "sealed-out" / "results.json").exists()

    # 2. Mini 3-case stereo manifest end-to-end CLI check.
    audio = tmp_path / "audio"
    cases = [
        {"id": "t-clip", "kind": "paired_synthetic", "profile": "dense_symmetric_moderate",
         "rate": 44100, "channels": 2, "mode": "auto", "split": "eval",
         "rights": {k: True for k in corpus_v1.OPERATIONS},
         "groups": {k: f"{k}-1" for k in corpus_v1.GROUP_KEYS},
         "clean": {"path": "t-clip.clean.wav"}, "degraded": {"path": "t-clip.degraded.wav"},
         "synthesis": {"generator": corpus_synth.GENERATOR_V2, "seed": 901, "rate": 44100,
                       "seconds": 1, "operator": "hard_clip", "bits": 16, "threshold_dbfs": -6.0}},
        {"id": "t-clean", "kind": "paired_synthetic", "profile": "clean_control",
         "rate": 44100, "channels": 2, "mode": "auto", "split": "control",
         "rights": {k: True for k in corpus_v1.OPERATIONS},
         "groups": {k: f"{k}-2" for k in corpus_v1.GROUP_KEYS},
         "clean": {"path": "t-clean.clean.wav"}, "degraded": {"path": "t-clean.degraded.wav"},
         "synthesis": {"generator": corpus_synth.GENERATOR_V2, "seed": 902, "rate": 44100,
                       "seconds": 1, "operator": "clean", "bits": 16}},
        {"id": "t-soft", "kind": "paired_synthetic", "profile": "soft_limiter_control",
         "rate": 48000, "channels": 2, "mode": "auto", "split": "control",
         "rights": {k: True for k in corpus_v1.OPERATIONS},
         "groups": {k: f"{k}-3" for k in corpus_v1.GROUP_KEYS},
         "clean": {"path": "t-soft.clean.wav"}, "degraded": {"path": "t-soft.degraded.wav"},
         "synthesis": {"generator": corpus_synth.GENERATOR_V2, "seed": 903, "rate": 48000,
                       "seconds": 1, "operator": "soft_limiter", "bits": 24, "knee": 0.72}},
    ]
    files = []
    for c in cases:
        specs = corpus_synth.synthesize_case(c, audio)
        for role in ("clean", "degraded", "mask"):
            c.setdefault(role, {}).update(specs[role])
            files.append(specs[role])
    doc = {"schema": corpus_v1.SCHEMA, "tier": "dev", "dataset_version": "mini-e3",
           "toolchain": corpus_v1.actual_toolchain(),
           "rng": {"algorithm": "numpy-pcg64", "seed": 1},
           "numeric_tolerance": 1e-6, "output_ceiling_dbfs": -1.0,
           "files": files, "cases": cases}
    mpath = tmp_path / "mini.json"
    mpath.write_text(json.dumps(doc))

    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(BAKEOFF_CLI), "--manifest", str(mpath),
         "--audio-root", str(audio), "--out", str(out),
         "--methods", "identity,gain_only,aspade_mit,reg_ar_mit"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    proto = json.loads((out / "protocol.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert proto["schema"] == "e3-bakeoff/v1"
    assert str(tmp_path) not in (out / "protocol.json").read_text()
    assert summary["neural_colab"]["status"] == "not_run"
    assert summary["dev_decision"]["sealed_opened"] is False
    assert summary["detectors"]["v2_windowed"]["control_fp_cases"] == 0
    assert summary["oracle"]["gain_only"]["polarity_violations_total"] == 0
    assert summary["oracle"]["gain_only"]["max_reliable_diff"] == 0.0
    assert summary["blind"]["current_snapshot_auto"]["polarity_violations_total"] is None
    assert summary["oracle"]["aspade_mit"]["all_input_mean_masked_delta_sdr_db"] > 0.0
    assert (out / "report.md").exists()
