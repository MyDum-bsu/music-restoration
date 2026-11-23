#!/usr/bin/env python3
"""Unit & integration tests for E7 BWE preview with marked synthesized band."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from mutagen.flac import FLAC

from research import build_e7_bwe_corpus as e7_corpus
from research import corpus_synth
from research import e7_bwe_preview as e7

REPO = Path(__file__).resolve().parents[1]


def _case_by_id(cid: str) -> dict:
    for case in e7_corpus.e7_bwe_cases():
        if case["id"] == cid:
            return case
    raise KeyError(f"unknown E7 case id: {cid}")


@pytest.mark.parametrize("arm", e7.DSP_ARMS)
def test_additive_identity_and_zero_input_mutation_stereo_and_mono(arm: str) -> None:
    """Every DSP arm preserves preview == input + synthesized_band and never mutates input."""
    case = _case_by_id("e7-bl8k-harmonic-mix-48k")
    _clean, degraded, _am, _meta = e7_corpus.synthesize_e7_arrays(case)
    sr = int(case["rate"])
    cutoff = float(case["observed_cutoff_hz"])

    seg_stereo = degraded[: int(sr * 0.5)].copy()
    h_before = hashlib.sha256(seg_stereo.tobytes()).hexdigest()
    res_stereo = e7.run_arm(arm, seg_stereo, sr, cutoff)
    h_after = hashlib.sha256(seg_stereo.tobytes()).hexdigest()
    assert h_before == h_after, f"{arm} mutated stereo input array"
    assert e7.verify_additive_contract(seg_stereo, res_stereo) <= e7.ADDITIVE_IDENTITY_TOL

    seg_mono = seg_stereo[:, 0].copy()
    hm_before = hashlib.sha256(seg_mono.tobytes()).hexdigest()
    res_mono = e7.run_arm(arm, seg_mono, sr, cutoff)
    hm_after = hashlib.sha256(seg_mono.tobytes()).hexdigest()
    assert hm_before == hm_after, f"{arm} mutated mono input array"
    assert e7.verify_additive_contract(seg_mono, res_mono) <= e7.ADDITIVE_IDENTITY_TOL


def test_safe_resample_only_polyphase_and_identity_zero_synth() -> None:
    """safe_resample_only produces zero synthesized HF in both identity and polyphase modes."""
    case = _case_by_id("e7-bl11k-harmonic-mix-48k")
    _clean, degraded, _am, _meta = e7_corpus.synthesize_e7_arrays(case)
    sr = int(case["rate"])
    cutoff = float(case["observed_cutoff_hz"])

    res_id = e7.run_safe_resample_only(degraded, sr, cutoff)
    assert res_id.synthesized_hf_band is False
    assert res_id.is_generative_preview is False
    assert float(np.max(np.abs(res_id.synthesized_band))) == 0.0
    assert np.array_equal(res_id.preview, degraded)

    # Polyphase upsample from 16 kHz -> 48 kHz (zero hallucinated HF above 8 kHz).
    rng = np.random.default_rng(123)
    t16 = np.arange(8000, dtype=np.float64) / 16000.0
    low_sig = np.column_stack([
        0.5 * np.sin(2.0 * np.pi * 1000.0 * t16),
        0.5 * np.sin(2.0 * np.pi * 2500.0 * t16 + 0.3),
    ]) + rng.normal(0.0, 1e-4, (8000, 2))
    res_up = e7.run_safe_resample_only(low_sig, 48000, 8000.0, input_sr=16000)
    assert res_up.meta["resampled"] is True
    assert res_up.preview.shape == (24000, 2)
    assert res_up.synthesized_hf_band is False
    assert float(np.max(np.abs(res_up.synthesized_band))) == 0.0


def test_dsp_bwe_preserves_observed_band_and_synthesizes_missing_band() -> None:
    """DSP BWE strictly preserves [0, cutoff] and improves missing-band envelope LSD."""
    for cid in ("e7-bl8k-harmonic-mix-48k", "e7-bl11k-percussion-cymbals-48k", "e7-codec-mp3-64k-11k-48k"):
        case = _case_by_id(cid)
        clean, degraded, attack_mask, _meta = e7_corpus.synthesize_e7_arrays(case)
        sr = int(case["rate"])
        cutoff = float(case["observed_cutoff_hz"])

        res = e7.run_arm("dsp_harmonic_bwe_preview", degraded, sr, cutoff)
        assert res.synthesized_hf_band is True
        assert res.is_generative_preview is True
        assert res.bypassed is False

        m = e7.evaluate_e7_case(
            clean, degraded, res, sr,
            observed_cutoff_hz=cutoff, attack_mask=attack_mask, is_fullband_control=False,
        )
        # Observed band strictly preserved: zero SNR loss, tiny max error, zero sub-cutoff leak.
        assert m["observed_band"]["delta_snr_db"] is not None
        assert abs(m["observed_band"]["delta_snr_db"]) < 0.02
        assert m["observed_band"]["vs_input_max_err"] < 1e-4
        assert m["artifacts"]["sub_cutoff_leakage_db"] is not None
        assert m["artifacts"]["sub_cutoff_leakage_db"] < -60.0

        # Missing band receives synthesized HF envelope closer to clean than silence.
        assert m["missing_band"]["delta_lsd_db"] is not None
        assert m["missing_band"]["delta_lsd_db"] < -5.0
        assert m["missing_band"]["synth_energy_dbfs"] is not None


def test_fullband_clean_control_abstain_and_forced_diagnostic() -> None:
    """Full-band clean negative controls abstain by default (zero false HF)."""
    case = _case_by_id("e7-ctl-fullband-percussion-48k")
    clean, degraded, attack_mask, _meta = e7_corpus.synthesize_e7_arrays(case)
    sr = int(case["rate"])
    cutoff = float(case["observed_cutoff_hz"])

    res_def = e7.run_arm("dsp_harmonic_bwe_preview", degraded, sr, cutoff, bypass_fullband=True)
    m_def = e7.evaluate_e7_case(
        clean, degraded, res_def, sr,
        observed_cutoff_hz=cutoff, attack_mask=attack_mask, is_fullband_control=True,
    )
    assert res_def.bypassed is True
    assert res_def.synthesized_hf_band is False
    assert m_def["artifacts"]["false_hf_injected"] is False
    assert float(np.max(np.abs(res_def.synthesized_band))) == 0.0

    # Forced mode honestly shows that disabling bypass injects HF on a clean full-band signal.
    res_forced = e7.run_arm("dsp_harmonic_bwe_preview", degraded, sr, cutoff, bypass_fullband=False)
    m_forced = e7.evaluate_e7_case(
        clean, degraded, res_forced, sr,
        observed_cutoff_hz=cutoff, attack_mask=attack_mask, is_fullband_control=True,
    )
    assert res_forced.bypassed is False
    assert m_forced["artifacts"]["false_hf_injected"] is True


def test_export_bwe_preview_tags_flac_and_sidecar_without_touching_source(tmp_path: Path) -> None:
    """export_bwe_preview leaves source untouched and writes mandatory FLAC + JSON tags."""
    case = _case_by_id("e7-bl8k-percussion-cymbals-48k")
    _clean, degraded, _am, _meta = e7_corpus.synthesize_e7_arrays(case)
    sr = int(case["rate"])
    cutoff = float(case["observed_cutoff_hz"])
    seg = degraded[: int(sr * 0.5)].copy()

    src_dir = tmp_path / "library"
    out_dir = tmp_path / "enhance_preview"
    src_dir.mkdir()
    src_path = src_dir / "original_track.wav"
    corpus_synth.write_wav(src_path, sr, seg, bits=24, seed=0)
    src_sha_before = hashlib.sha256(src_path.read_bytes()).hexdigest()

    res = e7.run_arm("dsp_harmonic_bwe_preview", seg, sr, cutoff)
    sidecar = e7.export_bwe_preview(src_path, res, sr, out_dir, case["id"], bits=24)

    # 1. Source file 100% unchanged.
    src_sha_after = hashlib.sha256(src_path.read_bytes()).hexdigest()
    assert src_sha_before == src_sha_after
    assert sidecar["source_untouched_verified"] is True
    assert sidecar["source_sha256"] == src_sha_before

    # 2. Required sidecar fields & SHA-256 verification on disk.
    assert sidecar["synthesized_hf_band"] is True
    assert sidecar["observed_cutoff_hz"] == cutoff
    assert sidecar["is_generative_preview"] is True
    assert sidecar["not_original_restoration"] is True
    assert sidecar["enhance_opt_in_derivative"] is True

    deriv_path = out_dir / sidecar["files"]["derivative_flac"]
    synth_path = out_dir / sidecar["files"]["synthesized_band_only_flac"]
    sidecar_path = out_dir / sidecar["files"]["sidecar_json"]
    assert deriv_path.exists() and synth_path.exists() and sidecar_path.exists()
    assert hashlib.sha256(deriv_path.read_bytes()).hexdigest() == sidecar["derivative_sha256"]
    assert hashlib.sha256(synth_path.read_bytes()).hexdigest() == sidecar["synthesized_band_sha256"]

    # 3. Required Vorbis comment tags embedded in BOTH FLAC files.
    for flac_p, expected_role in (
        (deriv_path, "bwe_preview_derivative"),
        (synth_path, "synthesized_band_only"),
    ):
        meta = FLAC(str(flac_p))
        assert meta["SYNTHESIZED_HF_BAND"] == ["true"]
        assert meta["OBSERVED_CUTOFF_HZ"] == [f"{cutoff:.1f}"]
        assert meta["IS_GENERATIVE_PREVIEW"] == ["true"]
        assert meta["NOT_ORIGINAL_RESTORATION"] == ["true"]
        assert meta["ENHANCE_OPT_IN_DERIVATIVE"] == ["true"]
        assert meta["SOURCE_SHA256"] == [src_sha_before]
        assert meta["BWE_TRACK_ROLE"] == [expected_role]


def test_neural_bwe_arm_honest_blocker_and_colab_contract() -> None:
    """Neural BWE arm honestly reports blocked_external_weights_or_colab with contract."""
    audit = e7.audit_neural_bwe_arm()
    assert audit["status"] == "blocked_external_weights_or_colab"
    assert audit["executed"] is False
    assert audit["production_dependency_added"] is False
    assert "no_local_licensed_bwe_weights_audiosr_universr_babe2_samecfm" in audit["blocked_reasons"]
    target_ids = {m["id"] for m in audit["weights_and_license_contract"]["target_models"]}
    assert {"audiosr", "universr", "babe_2", "samecfm"}.issubset(target_ids)
    assert "corpus/sealed-v1.json" in audit["colab_handoff_spec"]["forbidden_inputs"]
    assert "corpus/sealed-e5-v1.json" in audit["colab_handoff_spec"]["forbidden_inputs"]


def test_e4_freeze_untouched_and_no_host_paths_in_e7_artifacts() -> None:
    """Verify all 18 E4 frozen files are untouched and no host paths leak into E7 files."""
    e4_check = e7._verify_e4_freeze_untouched()
    assert e4_check["total_pinned_files"] == 18
    assert e4_check["all_untouched"] is True

    tracked_files = [
        REPO / "research" / "build_e7_bwe_corpus.py",
        REPO / "research" / "e7_bwe_preview.py",
        REPO / "tests" / "test_e7_bwe_preview.py",
        REPO / "corpus" / "dev-e7-bwe-v1.json",
        REPO / "docs" / "validation" / "e7-bwe-results.json",
        REPO / "docs" / "validation" / "e7-bwe-report.md",
    ]
    forbidden_tokens = ("/home/" + "mydum", "/media/" + "mydum")
    for path in tracked_files:
        assert path.exists(), f"missing expected E7 artifact: {path.name}"
        text = path.read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            assert tok not in text, f"host path {tok!r} leaked into {path.name}"

    res_doc = json.loads((REPO / "docs" / "validation" / "e7-bwe-results.json").read_text(encoding="utf-8"))
    assert res_doc["sealed_corpora_opened"] is False
    assert res_doc["preserve_pipeline_modified"] is False
    assert res_doc["colab_launched"] is False
    assert res_doc["invariants"]["any_input_mutated"] is False
    assert res_doc["invariants"]["max_additive_identity_err"] <= e7.ADDITIVE_IDENTITY_TOL

    # Verify every GFM table in the report has consistent column counts per table.
    report_lines = (REPO / "docs" / "validation" / "e7-bwe-report.md").read_text(encoding="utf-8").splitlines()
    expected_cols: int | None = None
    for line in report_lines:
        if line.startswith("|") and line.endswith("|"):
            cols = len(line.split("|")) - 2
            if expected_cols is None:
                expected_cols = cols
            else:
                assert cols == expected_cols, f"misaligned GFM table row: {line}"
        else:
            expected_cols = None
