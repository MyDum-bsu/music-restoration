#!/usr/bin/env python3
"""Unit & integration tests for E6 archival de-click and hiss research harness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research import build_e6_archival_corpus as e6_corpus
from research import e6_declick_hiss as e6

REPO = Path(__file__).resolve().parents[1]


def _case_by_id(cid: str) -> dict:
    for case in e6_corpus.e6_archival_cases():
        if case["id"] == cid:
            return case
    raise KeyError(f"unknown E6 case id: {cid}")


@pytest.mark.parametrize("arm", e6.DSP_ARMS)
def test_residual_identity_and_zero_input_mutation_stereo_and_mono(arm: str) -> None:
    """Every DSP arm preserves restored + residual == input and never mutates input."""
    case = _case_by_id("e6-archival-vinyl-44k")
    _clean, degraded, _cm, _am, _ts = e6_corpus.synthesize_e6_arrays(case)
    sr = int(case["rate"])
    # Slice 0.6 s for fast deterministic test execution.
    seg_stereo = degraded[: int(sr * 0.6)].copy()
    h_before = hashlib.sha256(seg_stereo.tobytes()).hexdigest()
    res_stereo = e6.run_arm(arm, seg_stereo, sr)
    h_after = hashlib.sha256(seg_stereo.tobytes()).hexdigest()
    assert h_before == h_after, f"{arm} mutated stereo input array"
    assert e6.verify_residual_contract(seg_stereo, res_stereo) <= e6.RESIDUAL_IDENTITY_TOL

    seg_mono = seg_stereo[:, 0].copy()
    hm_before = hashlib.sha256(seg_mono.tobytes()).hexdigest()
    res_mono = e6.run_arm(arm, seg_mono, sr)
    hm_after = hashlib.sha256(seg_mono.tobytes()).hexdigest()
    assert hm_before == hm_after, f"{arm} mutated mono input array"
    assert e6.verify_residual_contract(seg_mono, res_mono) <= e6.RESIDUAL_IDENTITY_TOL


def test_ar_declick_removes_clicks_and_preserves_unmasked_samples() -> None:
    """AR de-clicker improves SNR on click-corrupted audio and never edits outside mask."""
    case = _case_by_id("e6-clicks-sparse-tonal-44k")
    clean, degraded, click_mask, attack_mask, tail_slice = e6_corpus.synthesize_e6_arrays(case)
    sr = int(case["rate"])
    res = e6.run_arm("ar_declick", degraded, sr)
    metrics = e6.evaluate_e6_case(
        clean, degraded, res, sr,
        click_mask=click_mask, attack_mask=attack_mask, tail_slice=tail_slice,
    )
    assert metrics["delta_snr_db"] is not None and metrics["delta_snr_db"] > 10.0
    assert metrics["si_sdri_db"] is not None and metrics["si_sdri_db"] > 10.0
    assert metrics["click_sample_f1"] is not None and metrics["click_sample_f1"] > 0.6
    assert metrics["outside_click_max_diff"] == 0.0


def test_clean_sharp_attacks_protected_from_false_positives() -> None:
    """AR de-clicker protects clean percussive/plucked attacks; hiss arms bypass on -82 dBFS floor."""
    for cid in ("e6-ctl-clean-percussion-48k", "e6-ctl-clean-plucked-44k"):
        case = _case_by_id(cid)
        clean, degraded, click_mask, attack_mask, tail_slice = e6_corpus.synthesize_e6_arrays(case)
        sr = int(case["rate"])
        assert attack_mask.any()
        assert not click_mask.any()

        ar_res = e6.run_arm("ar_declick", degraded, sr)
        ar_m = e6.evaluate_e6_case(
            clean, degraded, ar_res, sr,
            click_mask=click_mask, attack_mask=attack_mask, tail_slice=tail_slice,
        )
        assert ar_m["transient_fp_rate"] == 0.0
        assert ar_m["clean_attack_zero_residual"] is True
        assert ar_m["clean_attack_residual_energy_db"] is None

        for hiss_arm in ("wiener_hiss", "mmse_stsa_hiss", "ar_declick_wiener_hiss", "ar_declick_mmse_stsa"):
            h_res = e6.run_arm(hiss_arm, degraded, sr)
            h_m = e6.evaluate_e6_case(
                clean, degraded, h_res, sr,
                click_mask=click_mask, attack_mask=attack_mask, tail_slice=tail_slice,
            )
            assert h_m["bypassed_channels"] == 2
            assert h_m["transient_fp_rate"] == 0.0


def test_active_hiss_reducers_without_bypass_and_on_roomtone_control() -> None:
    """Measure real active Wiener/MMSE-STSA behavior when clean bypass does NOT fire."""
    # 1. Room-tone clean control (-58 dBFS floor > -70 dBFS bypass threshold): active suppression runs.
    rt_case = _case_by_id("e6-ctl-clean-roomtone-tail-48k")
    c_rt, d_rt, cm_rt, am_rt, ts_rt = e6_corpus.synthesize_e6_arrays(rt_case)
    sr_rt = int(rt_case["rate"])
    assert ts_rt is not None and am_rt.any()
    for arm in ("wiener_hiss", "mmse_stsa_hiss"):
        res = e6.run_arm(arm, d_rt, sr_rt)
        m = e6.evaluate_e6_case(
            c_rt, d_rt, res, sr_rt,
            click_mask=cm_rt, attack_mask=am_rt, tail_slice=ts_rt,
        )
        assert m["bypassed_channels"] == 0, f"{arm} unexpectedly bypassed on -58 dBFS room-tone"
        assert m["residual_identity_max_err"] <= e6.RESIDUAL_IDENTITY_TOL
        assert m["clean_attack_zero_residual"] is False
        assert m["clean_attack_residual_energy_db"] is not None
        assert m["quiet_tail_drift_db"] is not None and m["quiet_tail_drift_db"] > 0.0

    # 2. Explicit bypass-disabled check on clean reverb tail (-120 dBFS bypass threshold).
    tail_case = _case_by_id("e6-ctl-clean-reverb-tail-48k")
    c_t, d_t, cm_t, am_t, ts_t = e6_corpus.synthesize_e6_arrays(tail_case)
    sr_t = int(tail_case["rate"])
    for fn in (e6.restore_wiener_hiss, e6.restore_mmse_stsa_hiss):
        res_nb = fn(d_t, sr_t, clean_bypass_dbfs=-120.0)
        m_nb = e6.evaluate_e6_case(
            c_t, d_t, res_nb, sr_t,
            click_mask=cm_t, attack_mask=am_t, tail_slice=ts_t,
        )
        assert m_nb["bypassed_channels"] == 0
        assert m_nb["residual_identity_max_err"] <= e6.RESIDUAL_IDENTITY_TOL
        assert m_nb["quiet_tail_drift_db"] is not None and m_nb["quiet_tail_drift_db"] < 3.0


def test_low_level_crackle_and_wide_scratch_honest_diagnostics() -> None:
    """Verify quiet crackle sensitivity case and wide-scratch partial-fragmentation behavior."""
    ll_case = _case_by_id("e6-clicks-low-level-crackle-44k")
    c_ll, d_ll, cm_ll, am_ll, ts_ll = e6_corpus.synthesize_e6_arrays(ll_case)
    res_ll = e6.run_arm("ar_declick", d_ll, int(ll_case["rate"]))
    m_ll = e6.evaluate_e6_case(
        c_ll, d_ll, res_ll, int(ll_case["rate"]),
        click_mask=cm_ll, attack_mask=am_ll, tail_slice=ts_ll,
    )
    assert m_ll["click_sample_recall"] is not None and m_ll["click_sample_recall"] < 0.25

    ws_case = _case_by_id("e6-ood-wide-scratch-44k")
    c_ws, d_ws, cm_ws, am_ws, ts_ws = e6_corpus.synthesize_e6_arrays(ws_case)
    res_ws = e6.run_arm("ar_declick", d_ws, int(ws_case["rate"]))
    m_ws = e6.evaluate_e6_case(
        c_ws, d_ws, res_ws, int(ws_case["rate"]),
        click_mask=cm_ws, attack_mask=am_ws, tail_slice=ts_ws,
    )
    # Honest partial fragmentation (neither full repair nor zero-sample abstention).
    assert m_ws["click_sample_recall"] is not None
    assert 0.0 < m_ws["click_sample_recall"] < 0.5


def test_median_impulse_linear_interpolation_does_not_duplicate_endpoints() -> None:
    """Linear interpolation across a detected click run excludes left/right boundary values."""
    import numpy as np

    sig = np.linspace(0.0, 0.2, 2048, dtype=np.float64)
    # Inject a 3-sample impulse at indices 1000..1002.
    corrupted = sig.copy()
    corrupted[1000:1003] += 0.65
    res = e6.restore_median_impulse(corrupted, 44100)
    assert res.click_mask is not None and res.click_mask[1000:1003].all()
    left = float(corrupted[999])
    right = float(corrupted[1003])
    # Interior interpolated points must lie strictly between left and right (no endpoint duplication).
    assert float(res.restored[1000]) > left
    assert float(res.restored[1002]) < right


def test_hiss_reducers_improve_snr_and_preserve_clean_reverb_tails() -> None:
    """Wiener and MMSE-STSA improve SNR on hiss while preserving clean reverb tails."""
    hiss_case = _case_by_id("e6-hiss-tape-white-20db-44k")
    clean, degraded, click_mask, attack_mask, tail_slice = e6_corpus.synthesize_e6_arrays(hiss_case)
    sr = int(hiss_case["rate"])
    for arm in ("wiener_hiss", "mmse_stsa_hiss"):
        res = e6.run_arm(arm, degraded, sr)
        m = e6.evaluate_e6_case(
            clean, degraded, res, sr,
            click_mask=click_mask, attack_mask=attack_mask, tail_slice=tail_slice,
        )
        assert m["delta_snr_db"] is not None and m["delta_snr_db"] > 4.0
        assert m["si_sdri_db"] is not None and m["si_sdri_db"] > 4.0

    # Clean reverb tail must remain untouched (zero drift, zero SNR loss).
    tail_case = _case_by_id("e6-ctl-clean-reverb-tail-48k")
    c_t, d_t, cm_t, am_t, ts_t = e6_corpus.synthesize_e6_arrays(tail_case)
    sr_t = int(tail_case["rate"])
    assert ts_t is not None
    for arm in ("ar_declick", "wiener_hiss", "mmse_stsa_hiss", "ar_declick_wiener_hiss", "ar_declick_mmse_stsa"):
        res = e6.run_arm(arm, d_t, sr_t)
        m = e6.evaluate_e6_case(
            c_t, d_t, res, sr_t,
            click_mask=cm_t, attack_mask=am_t, tail_slice=ts_t,
        )
        assert m["quiet_tail_drift_db"] is not None and m["quiet_tail_drift_db"] <= 0.1
        assert m["tail_snr_loss_db"] is not None and m["tail_snr_loss_db"] <= 0.5


def test_combined_cascade_beats_identity_on_archival_transfer() -> None:
    """Two-stage AR + spectral hiss cascade improves combined vinyl archival SNR."""
    case = _case_by_id("e6-archival-vinyl-44k")
    clean, degraded, click_mask, attack_mask, tail_slice = e6_corpus.synthesize_e6_arrays(case)
    sr = int(case["rate"])
    for arm in ("ar_declick_wiener_hiss", "ar_declick_mmse_stsa"):
        res = e6.run_arm(arm, degraded, sr)
        m = e6.evaluate_e6_case(
            clean, degraded, res, sr,
            click_mask=click_mask, attack_mask=attack_mask, tail_slice=tail_slice,
        )
        assert m["delta_snr_db"] is not None and m["delta_snr_db"] > 4.0
        assert m["si_sdri_db"] is not None and m["si_sdri_db"] > 4.0


def test_export_audition_bundle_writes_original_restored_residual(tmp_path: Path) -> None:
    """Operator audition bundle saves untouched original, restored, residual, and SHA-256."""
    case = _case_by_id("e6-clicks-sparse-tonal-44k")
    _clean, degraded, _cm, _am, _ts = e6_corpus.synthesize_e6_arrays(case)
    sr = int(case["rate"])
    seg = degraded[: int(sr * 0.5)].copy()
    res = e6.run_arm("ar_declick", seg, sr)
    prov = e6.export_audition_bundle(seg, res, sr, tmp_path, case["id"], bits=24)
    assert prov["residual_identity_verified"] is True
    for role in ("original", "restored", "residual"):
        info = prov["files"][role]
        p = tmp_path / info["file"]
        assert p.exists()
        assert hashlib.sha256(p.read_bytes()).hexdigest() == info["sha256"]


def test_learned_arm_honest_blocker_and_unpaired_protocol() -> None:
    """Learned arm honestly reports blocked_external_weights_or_colab with contract."""
    audit = e6.audit_learned_arm()
    assert audit["status"] == "blocked_external_weights_or_colab"
    assert audit["executed"] is False
    assert "no_local_historical_unet_or_restore_weights" in audit["blocked_reasons"]
    assert audit["unpaired_real_protocol"]["clean_reference_fabricated"] is False
    assert len(audit["weights_contract"]["target_models"]) >= 2
    assert "corpus/sealed-v1.json" in audit["colab_handoff_spec"]["forbidden_inputs"]
    assert "corpus/sealed-e5-v1.json" in audit["colab_handoff_spec"]["forbidden_inputs"]


def test_e4_freeze_untouched_and_no_host_paths_in_e6_artifacts() -> None:
    """Verify all 18 E4 frozen files are untouched and no host paths leak into E6 files."""
    e4_check = e6._verify_e4_freeze_untouched()
    assert e4_check["total_pinned_files"] == 18
    assert e4_check["all_untouched"] is True

    tracked_files = [
        REPO / "research" / "build_e6_archival_corpus.py",
        REPO / "research" / "e6_declick_hiss.py",
        REPO / "tests" / "test_e6_declick_hiss.py",
        REPO / "corpus" / "dev-e6-archival-v1.json",
        REPO / "docs" / "validation" / "e6-declick-hiss-results.json",
        REPO / "docs" / "validation" / "e6-declick-hiss-report.md",
    ]
    forbidden_tokens = ("/home/" + "mydum", "/media/" + "mydum")
    for path in tracked_files:
        assert path.exists(), f"missing expected E6 artifact: {path.name}"
        text = path.read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            assert tok not in text, f"host path {tok!r} leaked into {path.name}"

    res_doc = json.loads((REPO / "docs" / "validation" / "e6-declick-hiss-results.json").read_text(encoding="utf-8"))
    assert res_doc["sealed_corpora_opened"] is False
    assert res_doc["preserve_pipeline_modified"] is False
    assert res_doc["colab_launched"] is False
    assert res_doc["invariants"]["any_input_mutated"] is False
    assert res_doc["invariants"]["max_residual_identity_err"] <= e6.RESIDUAL_IDENTITY_TOL
