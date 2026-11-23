"""E4 Stereo/Album Safety & CPU Candidate Freeze tests.

Covers:
1. Chunked `true_peak_dbfs` exact parity vs full-array 4x polyphase resampling.
2. Stereo safety metrics (L/R correlation, ILD, local ILD, phase coherence,
   circular IPD error, ITD lag, M/S balance, mono-sum compatibility, pre-echo,
   quiet tail, unchanged bands) on real candidate output AND intentional defect
   injection (proving every gate catches real regressions).
3. Album-level headroom coordination (`plan_album_headroom`) across all 4
   regimes (all kept, unified album gain, mixed immutable-kept + attenuated ->
   `needs_review=True` & `auto_album_delivery=False`, >24 dB rejection).
4. Gapless boundary joint verification (passing continuous seam vs failing
   uncoordinated gain jump / click).
5. Real 24-bit FLAC CLI end-to-end runs of `scripts/restore_safe.py` with
   `--declip-engine aspade_v2 --album-headroom` + `scripts/evaluate_album.py`.
6. Sealed refusal, quick E4 CLI run without host path leaks, and committed
   E4 freeze manifest integrity & SHA-256 pin verification.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from research import corpus_synth, corpus_v1
from rnext.dsp.clip_detector import detect_hard_clip_v2
from rnext.dsp.contour import true_peak_dbfs
from rnext.dsp.declip_methods import apply_headroom_policy, declip_aspade
from rnext.dsp.repair import select_restoration
from rnext.dsp.stereo_safety import (
    evaluate_gapless_boundary,
    evaluate_stereo_safety,
    plan_album_headroom,
)
from scripts.restore_one import encode_flac

ROOT = Path(__file__).resolve().parents[1]
E4_CLI = ROOT / "research" / "e4_stereo_freeze.py"
RESTORE_CLI = ROOT / "scripts" / "restore_safe.py"
EVAL_ALBUM_CLI = ROOT / "scripts" / "evaluate_album.py"


def _write_flac24(path: Path, pcm: np.ndarray, sr: int, seed: int = 1) -> str:
    rng = np.random.default_rng(seed)
    dith = rng.random(pcm.shape) - rng.random(pcm.shape)
    q = np.clip(np.round(np.clip(pcm, -1.0, 1.0) * 8388608.0 + dith), -8388608, 8388607).astype(np.int32)
    tmp_wav = path.with_suffix(".tmp.wav")
    try:
        encode_flac(q, sr, path, tmp_wav, bits=24)
    finally:
        tmp_wav.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_chunked_true_peak_matches_full_resample_poly():
    """Chunked 4x polyphase true-peak matches full-array computation to machine precision."""
    rng = np.random.default_rng(42)
    # Length > chunk (524_288) so chunked path executes across multiple boundaries.
    n = 620_000
    t = np.arange(n, dtype=np.float64) / 44100.0
    x = np.column_stack([
        0.72 * np.sin(2.0 * np.pi * 11025.0 * t + 0.4),
        0.68 * np.sin(2.0 * np.pi * 8820.0 * t + 1.1),
    ]).astype(np.float32)
    # Put an intersample-heavy peak right across the 524_288 chunk seam!
    x[524_280:524_296, 0] = np.tile([0.92, -0.92], 8).astype(np.float32)
    x += rng.normal(0.0, 1e-4, x.shape).astype(np.float32)

    chunked_tp = true_peak_dbfs(x, 44100, factor=4)
    up_full = signal.resample_poly(x, 4, 1, axis=0)
    full_tp = float(20.0 * np.log10(max(float(np.max(np.abs(up_full))), 1e-12)))
    assert chunked_tp == pytest.approx(full_tp, abs=1e-6)


def test_stereo_safety_passes_on_single_channel_and_ild_itd_declip():
    sr = 48000
    base = corpus_synth._clean_music_signal(seed=710, rate=sr, seconds=2.0)
    # Introduce explicit 12-sample ITD (~250 us) and 3 dB ILD.
    n = sr * 2 - 20
    core = 0.5 * (base[:, 0] + base[:, 1])
    clean = np.column_stack([
        0.7 * core[:n] + 0.3 * base[:n, 0],
        (10.0 ** (-3.0 / 20.0)) * (0.7 * core[12:12 + n] + 0.3 * base[:n, 1]),
    ])
    clean *= 0.95 / np.max(np.abs(clean))

    # Left-only hard clip: right channel must remain bit-exact before gain.
    deg, mask, _ = corpus_synth.apply_operator(
        clean, sr,
        {"operator": "hard_clip", "threshold_dbfs": -5.5, "channel_mode": "left_only", "bits": 24},
        seed=710,
    )
    deg32 = deg.astype(np.float32)
    pred, _ = detect_hard_clip_v2(deg32, sr, bits=24)
    assert pred[:, 0].any() and not pred[:, 1].any()
    raw, _ = declip_aspade(deg32, sr, pred)
    np.testing.assert_array_equal(raw[:, 1], deg32[:, 1])
    delivered, head = apply_headroom_policy(raw, deg32, sr, pred, ceiling_db=-1.0)
    assert head["status"] == "processed_candidate"

    res = evaluate_stereo_safety(clean, deg32, delivered, sr, mask=mask, raw_candidate=raw)
    assert res["gates"]["passed"], res["gates"]["violations"]
    assert res["metrics"]["polarity_violations"] == 0
    assert res["metrics"]["max_reliable_diff_pre_gain"] == 0.0
    assert res["metrics"]["itd_drift_samples"] < 0.5
    assert res["metrics"]["ild_drift_vs_clean_db"] < 0.25
    assert res["metrics"]["ipd_error_mean_deg"] < 3.0
    json.dumps(res, allow_nan=False)


def test_stereo_safety_catches_injected_ild_ipd_itd_and_preecho_defects():
    """Prove the E4 safety rubric catches genuine stereo & transient corruptions."""
    sr = 44100
    clean = corpus_synth._clean_music_signal(seed=720, rate=sr, seconds=1.5)
    mask = np.zeros(clean.shape, dtype=bool)
    mask[1000:1020, 0] = True

    # 1. Injected ILD shift (+2.0 dB on L only)
    bad_ild = clean.copy()
    bad_ild[:, 0] *= 10.0 ** (2.0 / 20.0)
    r_ild = evaluate_stereo_safety(clean, clean, bad_ild, sr, mask=mask)
    assert not r_ild["gates"]["passed"]
    assert any("ild_drift" in v for v in r_ild["gates"]["violations"])

    # 2. Injected phase inversion on R (180 deg IPD + mono-sum cancellation)
    bad_phase = clean.copy()
    bad_phase[:, 1] *= -1.0
    r_phase = evaluate_stereo_safety(clean, clean, bad_phase, sr, mask=mask)
    assert not r_phase["gates"]["passed"]
    assert any("ipd_mean_deg" in v for v in r_phase["gates"]["violations"])
    assert any("lr_corr_drift" in v for v in r_phase["gates"]["violations"])

    # 3. Injected ITD shift (delay R by 10 samples)
    bad_itd = clean.copy()
    bad_itd[10:, 1] = clean[:-10, 1]
    r_itd = evaluate_stereo_safety(clean, clean, bad_itd, sr, mask=mask)
    assert not r_itd["gates"]["passed"]
    assert any("itd_drift_samples" in v for v in r_itd["gates"]["violations"])

    # 4. Injected pre-echo right before a sharp transient onset
    n = sr
    t = np.arange(n, dtype=np.float64) / sr
    quiet_hit = 0.002 * np.column_stack([np.sin(2 * np.pi * 220 * t)] * 2)
    pos = sr // 2
    dur = round(0.08 * sr)
    k = np.arange(dur, dtype=np.float64) / sr
    burst = 0.9 * np.exp(-k / 0.02) * np.sin(2 * np.pi * 440 * k)
    quiet_hit[pos:pos + dur, 0] += burst
    quiet_hit[pos:pos + dur, 1] += burst
    bad_pe = quiet_hit.copy()
    # Inject audible pre-echo 5 ms before onset
    pre = round(0.005 * sr)
    bad_pe[pos - pre:pos] += 0.08 * np.sin(2 * np.pi * 880 * np.arange(pre)[:, None] / sr)
    m_hit = np.zeros(quiet_hit.shape, dtype=bool)
    m_hit[pos:pos + 10, :] = True
    r_pe = evaluate_stereo_safety(quiet_hit, quiet_hit, bad_pe, sr, mask=m_hit)
    assert not r_pe["gates"]["passed"]
    assert any("pre_echo_rise_db" in v for v in r_pe["gates"]["violations"])


def test_gapless_boundary_passes_continuous_and_catches_gain_jump():
    sr = 44100
    cont = corpus_synth._clean_music_signal(seed=730, rate=sr, seconds=2.0)
    mid = cont.shape[0] // 2
    a, b = cont[:mid], cont[mid:]

    # Same uniform gain on both tracks -> exact gapless continuity
    g = 10.0 ** (-2.5 / 20.0)
    ok = evaluate_gapless_boundary(a, b, a * g, b * g, sr)
    assert ok["passed"] and ok["seam_level_jump_db"] < 1e-6

    # Uncoordinated per-track gain (-2.5 dB on A, 0 dB on B) -> fails level jump gate!
    bad = evaluate_gapless_boundary(a, b, a * g, b, sr)
    assert not bad["passed"]
    assert bad["seam_level_jump_db"] == pytest.approx(2.5, abs=0.05)
    assert any("gapless_level_jump_db" in v for v in bad["violations"])


def test_gapless_boundary_respects_clip_masks_at_track_edge():
    """Reconstructed clip in the final 3 samples of track A must not trigger false
    click or peak-growth alarms when mask_a marks those samples, while an unmasked
    seam click or gain jump still fails."""
    sr = 44100
    cont = corpus_synth._clean_music_signal(seed=731, rate=sr, seconds=2.0)
    mid = cont.shape[0] // 2
    a_clean, b_clean = cont[:mid].copy(), cont[mid:].copy()
    # Clip the last 3 samples of track A to a low plateau while candidate restores a higher peak.
    a_ref = a_clean.copy()
    a_cand = a_clean.copy()
    a_ref[-3:, :] = np.sign(a_ref[-3:, :] + 1e-9) * 0.45
    a_cand[-3:, :] = np.sign(a_ref[-3:, :]) * 1.15
    mask_a = np.zeros(a_ref.shape, dtype=bool)
    mask_a[-3:, :] = True
    mask_b = np.zeros(b_clean.shape, dtype=bool)

    g = 10.0 ** (-2.0 / 20.0)
    ok = evaluate_gapless_boundary(
        a_ref, b_clean, a_cand * g, b_clean * g, sr,
        mask_a=mask_a, mask_b=mask_b,
    )
    assert ok["passed"], ok["violations"]
    assert ok["seam_boundary_masked"] is True
    assert ok["seam_step_cand_aligned"] is None
    assert ok["seam_level_jump_db"] < 1e-6
    assert ok["seam_peak_growth_db"] == pytest.approx(0.0, abs=1e-6)

    # Without masks, the exact same peak difference at the edge fails click/peak gates.
    unmasked = evaluate_gapless_boundary(a_ref, b_clean, a_cand * g, b_clean * g, sr)
    assert not unmasked["passed"]

    # Even WITH mask_a on the last 3 samples, an uncoordinated gain jump STILL fails!
    bad_gain = evaluate_gapless_boundary(
        a_ref, b_clean, a_cand * g, b_clean, sr,
        mask_a=mask_a, mask_b=mask_b,
    )
    assert not bad_gain["passed"]
    assert any("gapless_level_jump_db" in v for v in bad_gain["violations"])


def test_stereo_safety_returns_none_on_silence_and_rejects_missing_pre_gain():
    """Unmeasurable metrics on silence must return None (never fake 0.0) and fail
    defect gates; omitting raw_candidate or modifying a negative control must fail."""
    sr = 44100
    silence = np.zeros((4800, 2), dtype=np.float64)
    m_sil = np.zeros(silence.shape, dtype=bool)
    m_sil[10:20, 0] = True
    r_sil = evaluate_stereo_safety(silence, silence, silence, sr, mask=m_sil, raw_candidate=silence)
    sm = r_sil["metrics"]
    assert sm["local_ild_active_windows"] == 0
    assert sm["local_ild_drift_p95_db"] is None
    assert sm["ms_ratio_drift_vs_clean_db"] is None
    assert sm["mono_sum_level_drift_vs_clean_db"] is None
    assert sm["band_shape_rms_vs_clean_db"] is None
    assert sm["unchanged_band_max_drift_db"] is None
    assert sm["ipd_error_mean_deg"] is None
    assert not r_sil["gates"]["passed"]
    assert "no_active_stereo_windows" in r_sil["gates"]["violations"]

    # Half-amplitude candidate without raw_candidate must fail on BOTH negative-control and defect paths.
    clean = corpus_synth._clean_music_signal(seed=732, rate=sr, seconds=1.0)
    half = clean * 0.5
    r_ctrl_no_raw = evaluate_stereo_safety(clean, clean, half, sr, is_defect=False)
    assert not r_ctrl_no_raw["gates"]["passed"]
    assert "missing_pre_gain_candidate" in r_ctrl_no_raw["gates"]["violations"]
    assert "negative_control_delivered_modified" in r_ctrl_no_raw["gates"]["violations"]

    m_def = np.zeros(clean.shape, dtype=bool)
    m_def[100:120, 0] = True
    r_def_no_raw = evaluate_stereo_safety(clean, clean, half, sr, mask=m_def, is_defect=True)
    assert not r_def_no_raw["gates"]["passed"]
    assert "missing_pre_gain_candidate" in r_def_no_raw["gates"]["violations"]


@pytest.mark.parametrize("n_samples", [1, 2, 17, 63])
def test_stereo_safety_and_gapless_handle_short_tracks_under_64_samples(n_samples):
    """Short tracks (< 64 samples, including n=1) must not crash with shape/index errors."""
    sr = 44100
    rng = np.random.default_rng(733 + n_samples)
    x = rng.normal(0.0, 0.2, size=(n_samples, 2))
    res = evaluate_stereo_safety(x, x, x, sr, raw_candidate=x, is_defect=False)
    assert res["metrics"]["ipd_error_mean_deg"] is None
    assert res["metrics"]["itd_defined"] is False
    assert res["gates"]["passed"] is True
    json.dumps(res, allow_nan=False)

    gb = evaluate_gapless_boundary(x, x, x, x, sr)
    assert gb["passed"] is True
    if n_samples < 2:
        assert gb["seam_slope_ref"] is None
    json.dumps(gb, allow_nan=False)


def test_restore_safe_album_headroom_rejects_invalid_combinations(tmp_path):
    """--album-headroom must fail loudly (exit 2) for wrong mode, single file, or --loudness-target."""
    sr = 44100
    x = corpus_synth._clean_music_signal(seed=735, rate=sr, seconds=0.5)
    src = tmp_path / "alb"
    src.mkdir()
    f1 = src / "01 - One.flac"
    f2 = src / "02 - Two.flac"
    _write_flac24(f1, x, sr, seed=1)
    _write_flac24(f2, x, sr, seed=2)

    # 1. Wrong mode (--mode auto)
    p1 = subprocess.run(
        [sys.executable, str(RESTORE_CLI), "--src", str(src), "--out", str(tmp_path / "o1"),
         "--mode", "auto", "--album-headroom"],
        capture_output=True, text=True, check=False,
    )
    assert p1.returncode == 2 and "--mode declip" in p1.stderr

    # 2. Single file
    p2 = subprocess.run(
        [sys.executable, str(RESTORE_CLI), "--src", str(f1), "--out", str(tmp_path / "o2"),
         "--mode", "declip", "--album-headroom"],
        capture_output=True, text=True, check=False,
    )
    assert p2.returncode == 2 and "at least 2 FLAC" in p2.stderr

    # 3. Incompatible --loudness-target
    p3 = subprocess.run(
        [sys.executable, str(RESTORE_CLI), "--src", str(src), "--out", str(tmp_path / "o3"),
         "--mode", "declip", "--album-headroom", "--loudness-target", "-14"],
        capture_output=True, text=True, check=False,
    )
    assert p3.returncode == 2 and "--loudness-target" in p3.stderr


def test_plan_album_headroom_all_four_regimes():
    # 1. All kept original
    p1 = plan_album_headroom([
        {"track_id": "01.flac", "status": "kept_original", "pre_gain_tp_dbfs": None},
        {"track_id": "02.flac", "status": "kept_original", "pre_gain_tp_dbfs": None},
    ])
    assert p1["policy"] == "all_kept_original"
    assert p1["compatible"] and not p1["needs_review"] and p1["auto_album_delivery"]

    # 2. All repaired -> unified single album gain (minimum / most conservative attenuation)
    p2 = plan_album_headroom([
        {"track_id": "01.flac", "status": "processed_candidate", "pre_gain_tp_dbfs": 0.5},
        {"track_id": "02.flac", "status": "processed_candidate", "pre_gain_tp_dbfs": 2.2},
    ], ceiling_db=-1.0)
    assert p2["policy"] == "unified_album_gain"
    assert p2["album_gain_db"] == pytest.approx(-3.2)
    assert p2["per_track_gain_db"] == {"01.flac": pytest.approx(-3.2), "02.flac": pytest.approx(-3.2)}
    assert p2["max_relative_level_shift_db"] == 0.0
    assert p2["compatible"] and not p2["needs_review"] and p2["auto_album_delivery"]

    # 3. Mixed kept + attenuated -> incompatible with immutable kept tracks -> needs_review!
    p3 = plan_album_headroom([
        {"track_id": "01.flac", "status": "processed_candidate", "pre_gain_tp_dbfs": 1.8},
        {"track_id": "02.flac", "status": "kept_original", "pre_gain_tp_dbfs": None},
    ], ceiling_db=-1.0)
    assert p3["policy"] == "mixed_kept_and_attenuated_incompatible"
    assert p3["compatible"] is False
    assert p3["needs_review"] is True
    assert p3["auto_album_delivery"] is False
    assert p3["per_track_gain_db"]["02.flac"] == 0.0
    assert p3["per_track_gain_db"]["01.flac"] == pytest.approx(-2.8)
    assert p3["max_relative_level_shift_db"] == pytest.approx(2.8)

    # 4. Excessive (>24 dB) headroom attenuation rejected
    p4 = plan_album_headroom([
        {"track_id": "01.flac", "status": "processed_candidate", "pre_gain_tp_dbfs": 26.0},
    ], ceiling_db=-1.0)
    assert p4["policy"] == "rejected_excessive_album_attenuation"
    assert p4["needs_review"] is True and p4["auto_album_delivery"] is False


def test_real_flac_album_cli_unified_and_mixed_with_aspade_v2(tmp_path):
    """End-to-end real 24-bit FLAC CLI tests for unified & mixed album headroom."""
    sr = 44100
    cont = corpus_synth._clean_music_signal(seed=740, rate=sr, seconds=3.0)
    mid = cont.shape[0] // 2

    # Album A: 2 gapless clipped tracks -> unified album gain, gapless passes, evaluate_album passes.
    deg_cont, _, _ = corpus_synth.apply_operator(
        cont, sr, {"operator": "hard_clip", "threshold_dbfs": -6.0, "bits": 24}, seed=740)
    src_a, out_a, eval_a = tmp_path / "alb_a_src", tmp_path / "alb_a_out", tmp_path / "alb_a_eval"
    src_a.mkdir()
    h1 = _write_flac24(src_a / "01 - Part1.flac", deg_cont[:mid], sr, seed=1)
    h2 = _write_flac24(src_a / "02 - Part2.flac", deg_cont[mid:], sr, seed=2)

    p_a = subprocess.run(
        [sys.executable, str(RESTORE_CLI), "--src", str(src_a), "--out", str(out_a),
         "--mode", "declip", "--declip-engine", "aspade_v2", "--album-headroom"],
        capture_output=True, text=True, check=False,
    )
    assert p_a.returncode == 0, p_a.stderr
    rep_a = json.loads((out_a / "report.json").read_text())
    as_a = json.loads((out_a / "album_safety.json").read_text())
    assert [r["status"] for r in rep_a] == ["processed_candidate", "processed_candidate"]
    assert as_a["headroom_plan"]["policy"] == "unified_album_gain"
    assert as_a["headroom_plan"]["auto_album_delivery"] is True
    assert as_a["all_boundaries_passed"] is True
    assert rep_a[0]["repair"]["output_gain_db"] == pytest.approx(rep_a[1]["repair"]["output_gain_db"], abs=1e-9)
    assert all(r["delivery_qc"]["delivered_tp_dbfs"] <= -1.0 + 0.05 for r in rep_a)
    assert hashlib.sha256((src_a / "01 - Part1.flac").read_bytes()).hexdigest() == h1
    assert hashlib.sha256((src_a / "02 - Part2.flac").read_bytes()).hexdigest() == h2

    p_ev = subprocess.run(
        [sys.executable, str(EVAL_ALBUM_CLI), "--src", str(src_a), "--restored", str(out_a),
         "--out", str(eval_a)],
        capture_output=True, text=True, check=False,
    )
    assert p_ev.returncode == 0, p_ev.stderr
    ev_sum = json.loads((eval_a / "summary.json").read_text())
    assert ev_sum["verdict_counts"]["orig_better"] == 0

    # Album B: 1 clipped + 1 clean kept_original -> kept is byte-identical, needs_review=True!
    src_b, out_b = tmp_path / "alb_b_src", tmp_path / "alb_b_out"
    src_b.mkdir()
    _write_flac24(src_b / "01 - Clipped.flac", deg_cont[:mid], sr, seed=3)
    h_clean = _write_flac24(src_b / "02 - Clean.flac", cont[mid:], sr, seed=4)
    p_b = subprocess.run(
        [sys.executable, str(RESTORE_CLI), "--src", str(src_b), "--out", str(out_b),
         "--mode", "declip", "--declip-engine", "aspade_v2", "--album-headroom"],
        capture_output=True, text=True, check=False,
    )
    assert p_b.returncode == 0, p_b.stderr
    rep_b = json.loads((out_b / "report.json").read_text())
    as_b = json.loads((out_b / "album_safety.json").read_text())
    assert [r["status"] for r in rep_b] == ["processed_candidate", "kept_original"]
    assert hashlib.sha256((out_b / "02 - Clean.flac").read_bytes()).hexdigest() == h_clean
    assert as_b["headroom_plan"]["policy"] == "mixed_kept_and_attenuated_incompatible"
    assert as_b["headroom_plan"]["needs_review"] is True
    assert as_b["headroom_plan"]["auto_album_delivery"] is False
    assert all(r["needs_review"] is True for r in rep_b)


def test_select_restoration_default_engine_unchanged_and_aspade_v2_opt_in():
    sr = 44100
    clean = corpus_synth._clean_music_signal(seed=750, rate=sr, seconds=1.0)
    deg, _, _ = corpus_synth.apply_operator(
        clean, sr, {"operator": "hard_clip", "threshold_dbfs": -5.5, "bits": 24}, seed=750)
    y_v2, rep_v2 = select_restoration(deg.astype(np.float32), sr, mode="declip", bits=24,
                                      declip_engine="aspade_v2")
    assert rep_v2["status"] == "processed_candidate"
    assert rep_v2["chosen"] == "aspade_mit__v2_windowed"
    assert rep_v2["rest"]["tp_dbfs"] <= -1.0 + 0.05
    assert y_v2.shape == deg.shape

    # Negative control must be kept_original under aspade_v2
    y_cln, rep_cln = select_restoration(clean.astype(np.float32), sr, mode="declip", bits=24,
                                        declip_engine="aspade_v2")
    assert rep_cln["status"] == "kept_original"
    np.testing.assert_array_equal(y_cln, clean.astype(np.float32))


def test_e4_cli_refuses_sealed_and_runs_quick_without_host_paths(tmp_path):
    # 1. Strictly refuses sealed manifest before touching audio bytes.
    proc_s = subprocess.run(
        [sys.executable, str(E4_CLI), "--manifest", str(ROOT / "corpus" / "sealed-v1.json"),
         "--out", str(tmp_path / "sealed-out"), "--mode", "quick"],
        capture_output=True, text=True, check=False,
    )
    assert proc_s.returncode == 2
    assert not (tmp_path / "sealed-out" / "e4-freeze-manifest.json").exists()

    # 2. Quick end-to-end E4 run on a mini 2-case stereo manifest + stress/album/CPU profile.
    audio = tmp_path / "mini_audio"
    cases = [
        {"id": "q-clip", "kind": "paired_synthetic", "profile": "dense_symmetric_moderate",
         "rate": 44100, "channels": 2, "mode": "auto", "split": "eval",
         "rights": {k: True for k in corpus_v1.OPERATIONS},
         "groups": {k: f"q-{k}-1" for k in corpus_v1.GROUP_KEYS},
         "clean": {"path": "q-clip.clean.wav"}, "degraded": {"path": "q-clip.degraded.wav"},
         "synthesis": {"generator": corpus_synth.GENERATOR_V2, "seed": 951, "rate": 44100,
                       "seconds": 1, "operator": "hard_clip", "bits": 16, "threshold_dbfs": -5.5}},
        {"id": "q-clean", "kind": "paired_synthetic", "profile": "clean_control",
         "rate": 48000, "channels": 2, "mode": "auto", "split": "control",
         "rights": {k: True for k in corpus_v1.OPERATIONS},
         "groups": {k: f"q-{k}-2" for k in corpus_v1.GROUP_KEYS},
         "clean": {"path": "q-clean.clean.wav"}, "degraded": {"path": "q-clean.degraded.wav"},
         "synthesis": {"generator": corpus_synth.GENERATOR_V2, "seed": 952, "rate": 48000,
                       "seconds": 1, "operator": "clean", "bits": 24}},
    ]
    files = []
    for c in cases:
        specs = corpus_synth.synthesize_case(c, audio)
        for role in ("clean", "degraded", "mask"):
            c.setdefault(role, {}).update(specs[role])
            files.append(specs[role])
    mini_doc = {"schema": corpus_v1.SCHEMA, "tier": "dev", "dataset_version": "mini-e4",
                "toolchain": corpus_v1.actual_toolchain(),
                "rng": {"algorithm": "numpy-pcg64", "seed": 1},
                "numeric_tolerance": 1e-6, "output_ceiling_dbfs": -1.0,
                "files": files, "cases": cases}
    mini_manifest = tmp_path / "mini-e4.json"
    mini_manifest.write_text(json.dumps(mini_doc))

    out = tmp_path / "quick-out"
    proc = subprocess.run(
        [sys.executable, str(E4_CLI), "--manifest", str(mini_manifest),
         "--out", str(out), "--mode", "quick"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    manifest_text = (out / "e4-freeze-manifest.json").read_text()
    report_text = (out / "e4-stereo-freeze-report.md").read_text()
    proto_text = (out / "protocol.json").read_text()
    for blob in (manifest_text, report_text, proto_text):
        assert str(tmp_path) not in blob
        assert "/home/" not in blob
        assert "/media/" not in blob
    freeze = json.loads(manifest_text)
    assert freeze["schema"] == "e4-freeze/v1"
    assert freeze["sealed_audio_opened"] is False
    assert freeze["sealed_manifest_metadata_leakage_checked_only"] is True
    assert freeze["freeze_verdict"]["ready_for_e5_custodian_release"] is True
    assert freeze["e4_verification"]["suite_b_stereo_and_album"]["all_passed"] is True
    assert freeze["e4_verification"]["suite_c_cpu_profile"]["summary"]["failures"] == 0


def test_committed_e4_freeze_manifest_is_valid_and_pinned():
    freeze_path = ROOT / "docs" / "validation" / "e4-freeze-manifest.json"
    report_path = ROOT / "docs" / "validation" / "e4-stereo-freeze-report.md"
    assert freeze_path.exists(), "E4 freeze manifest must be generated and committed"
    assert report_path.exists(), "E4 freeze report must be generated and committed"

    text = freeze_path.read_text()
    rep_text = report_path.read_text()
    for blob in (text, rep_text):
        assert "/home/" not in blob
        assert "/media/" not in blob

    doc = json.loads(text)
    assert doc["schema"] == "e4-freeze/v1"
    assert doc["sealed_audio_opened"] is False
    assert doc["sealed_manifest_metadata_leakage_checked_only"] is True
    assert doc["freeze_verdict"]["ready_for_e5_custodian_release"] is True
    assert doc["cross_manifest_leakage_errors"] == []

    # Verify content digest integrity
    expected_sha = doc["freeze_sha256"]
    body = {k: v for k, v in doc.items() if k != "freeze_sha256"}
    actual_sha = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert actual_sha == expected_sha

    # Verify all pilot fields satisfy corpus_v1.pilot_errors (ready for E5 custodian registration)
    pilot = doc["e5_preregistration"]["pilot_registration"]
    assert corpus_v1.pilot_errors(pilot) == []

    # Verify every pinned implementation file hash matches the committed workspace!
    for rel, pinned in doc["toolchain"]["implementation_sha256"].items():
        assert corpus_v1.digest(ROOT / rel) == pinned, f"implementation hash drifted: {rel}"

    # Verify committed sealed-v1.json remains untouched & blocked!
    sealed = corpus_v1.load(ROOT / "corpus" / "sealed-v1.json")
    assert any("sealed tier blocked" in e for e in corpus_v1.validate_manifest(sealed))
