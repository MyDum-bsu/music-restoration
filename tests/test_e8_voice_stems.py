#!/usr/bin/env python3
"""Unit & integration tests for E8 Voice & Stems separate feasibility evaluations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research import corpus_v1, e8_stems_eval, e8_voice_eval

REPO = Path(__file__).resolve().parents[1]


def _voice_case(cid: str) -> dict:
    for case in e8_voice_eval.voice_cases():
        if case["id"] == cid:
            return case
    raise KeyError(f"unknown E8 voice case id: {cid}")


def _stems_case(cid: str) -> dict:
    for case in e8_stems_eval.stems_cases():
        if case["id"] == cid:
            return case
    raise KeyError(f"unknown E8 stems case id: {cid}")


# ---------------------------------------------------------------------------
# Contour A — Voice Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", e8_voice_eval.VOICE_CPU_ARMS)
def test_voice_arms_zero_input_mutation_and_finite_metrics(arm: str) -> None:
    """Every Contour-A CPU arm preserves input immutability and yields finite voice metrics."""
    case = _voice_case("e8-voice-revnoise-alto-mod-16k")
    clean, degraded, _meta = e8_voice_eval.synthesize_voice_case(case)
    sr = int(case["rate"])

    inp = degraded.copy()
    h_before = hashlib.sha256(inp.tobytes()).hexdigest()
    res = e8_voice_eval.run_voice_arm(arm, inp, sr)
    h_after = hashlib.sha256(inp.tobytes()).hexdigest()
    assert h_before == h_after, f"{arm} mutated voice input"

    m = e8_voice_eval.evaluate_voice_case(
        clean, inp, res, sr,
        is_clean_control=False,
        input_sha256_before=h_before,
        input_sha256_after=h_after,
    )
    assert m["input_mutated"] is False
    for key in (
        "si_sdr_in_db",
        "si_sdr_out_db",
        "si_sdri_db",
        "stoi_proxy_out",
        "formant_drift_out_hz",
        "spectral_lsd_out_db",
        "syllable_env_corr_out",
    ):
        assert np.isfinite(m[key]), f"{arm}: non-finite metric {key}"
    assert -1.0 <= m["stoi_proxy_out"] <= 1.0
    assert -1.0 <= m["syllable_env_corr_out"] <= 1.0


def test_voice_wpe_and_spectral_enhancer_improve_degraded_and_bypass_clean_controls() -> None:
    """WPE improves reverb_only, spectral enhancer improves noise_only, clean controls bypass."""
    # 1. Reverb-only case: WPE improves SI-SDR.
    rev_case = _voice_case("e8-voice-rev-alto-medium-16k")
    c_rev, d_rev, _ = e8_voice_eval.synthesize_voice_case(rev_case)
    res_wpe = e8_voice_eval.run_voice_arm("wpe_dereverb_cpu", d_rev, int(rev_case["rate"]))
    m_wpe = e8_voice_eval.evaluate_voice_case(
        c_rev, d_rev, res_wpe, int(rev_case["rate"]), is_clean_control=False
    )
    assert res_wpe.bypassed is False
    assert m_wpe["si_sdri_db"] > 0.5

    # 2. Noise-only case: spectral speech enhancer improves SI-SDR & LSD.
    n_case = _voice_case("e8-voice-noise-alto-10db-16k")
    c_n, d_n, _ = e8_voice_eval.synthesize_voice_case(n_case)
    res_spec = e8_voice_eval.run_voice_arm("spectral_speech_enhancer_cpu", d_n, int(n_case["rate"]))
    m_spec = e8_voice_eval.evaluate_voice_case(
        c_n, d_n, res_spec, int(n_case["rate"]), is_clean_control=False
    )
    assert res_spec.bypassed is False
    assert m_spec["si_sdri_db"] > 1.5
    assert m_spec["delta_spectral_lsd_db"] < 0.0

    # 3. Clean negative control: both CPU enhancers abstain with 0.0 error.
    ctl_case = _voice_case("e8-voice-ctl-clean-baritone-16k")
    c_ctl, d_ctl, _ = e8_voice_eval.synthesize_voice_case(ctl_case)
    for arm in ("wpe_dereverb_cpu", "spectral_speech_enhancer_cpu"):
        res_ctl = e8_voice_eval.run_voice_arm(arm, d_ctl, int(ctl_case["rate"]))
        m_ctl = e8_voice_eval.evaluate_voice_case(
            c_ctl, d_ctl, res_ctl, int(ctl_case["rate"]), is_clean_control=True
        )
        assert res_ctl.bypassed is True
        assert m_ctl["clean_control_max_abs_err"] == 0.0


def test_voice_external_audit_honest_blockers_and_licenses() -> None:
    """Contour-A audit honestly marks DAPS/VoiceBank, GPU models, DNSMOS, and PESQ."""
    audit = e8_voice_eval.audit_voice_external_models_and_metrics()
    assert audit["contour"] == "A_voice"
    assert audit["neural_arm_status"] == "blocked_external_weights_or_colab"
    assert audit["executed"] is False
    assert audit["production_dependency_added"] is False

    ds_map = {d["id"]: d["status"] for d in audit["external_datasets"]}
    assert ds_map["daps"] == "not_provisioned_on_host"
    assert ds_map["voicebank_demand"] == "not_provisioned_on_host"

    mdl_map = {m["id"]: m["status"] for m in audit["external_neural_models"]}
    assert mdl_map["deepfilternet3"] == "blocked_external_weights_or_colab"
    assert mdl_map["sgmse_plus"] == "blocked_external_weights_or_colab"
    assert mdl_map["flow_matching_speech"] == "blocked_external_weights_or_colab"

    met_map = {m["id"]: m["status"] for m in audit["external_perceptual_metrics"]}
    assert met_map["pesq"] == "license_required_for_pesq"
    assert met_map["dnsmos"] == "blocked_external_weights_or_colab"


# ---------------------------------------------------------------------------
# Contour B — Stems Tests
# ---------------------------------------------------------------------------

def test_stems_corpus_exact_additive_mix_invariant_all_cases() -> None:
    """Every case in dev-e8-stems-v1 satisfies exact additivity `mix == sum(stems)`."""
    for case in e8_stems_eval.stems_cases():
        stems, mix = e8_stems_eval.synthesize_stems_case(case)
        assert set(stems.keys()) == set(e8_stems_eval.STEM_NAMES)
        stem_sum = sum(stems[s] for s in e8_stems_eval.STEM_NAMES)
        err = float(np.max(np.abs(stem_sum - mix)))
        assert err <= e8_stems_eval.ADDITIVE_MIX_TOL, f"{case['id']}: additive err={err:.3e}"


@pytest.mark.parametrize("arm", e8_stems_eval.STEMS_CPU_ARMS)
def test_stems_arms_mix_reconstruction_and_zero_input_mutation(arm: str) -> None:
    """Every Contour-B CPU arm reconstructs mix (`sum(est_stems) == mix`) and preserves input."""
    case = _stems_case("e8-stems-std-pop-rock-44k")
    stems, mix = e8_stems_eval.synthesize_stems_case(case)
    sr = int(case["rate"])

    mix_copy = mix.copy()
    h_before = hashlib.sha256(mix_copy.tobytes()).hexdigest()
    res = e8_stems_eval.run_stems_arm(
        arm, mix_copy, sr, true_stems=stems if arm == "oracle_irm_upper_bound" else None
    )
    h_after = hashlib.sha256(mix_copy.tobytes()).hexdigest()
    assert h_before == h_after, f"{arm} mutated mix input"

    m = e8_stems_eval.evaluate_stems_case(
        stems, mix_copy, res, sr,
        input_sha256_before=h_before,
        input_sha256_after=h_after,
    )
    assert m["input_mutated"] is False
    assert m["corpus_additive_max_err"] <= e8_stems_eval.ADDITIVE_MIX_TOL
    assert m["mix_recon_max_err"] < 1e-9
    assert m["mix_recon_rms_db"] < -150.0


def test_stems_separation_hierarchy_and_sparse_control_reporting() -> None:
    """Oracle IRM > HPSS+band > identity on standard 4-stem mix; sparse controls report silent stems."""
    case = _stems_case("e8-stems-std-pop-rock-44k")
    stems, mix = e8_stems_eval.synthesize_stems_case(case)
    sr = int(case["rate"])

    res_id = e8_stems_eval.run_stems_arm("identity", mix, sr)
    res_hpss = e8_stems_eval.run_stems_arm("hpss_band_separator_cpu", mix, sr)
    res_irm = e8_stems_eval.run_stems_arm("oracle_irm_upper_bound", mix, sr, true_stems=stems)

    m_id = e8_stems_eval.evaluate_stems_case(stems, mix, res_id, sr)
    m_hpss = e8_stems_eval.evaluate_stems_case(stems, mix, res_hpss, sr)
    m_irm = e8_stems_eval.evaluate_stems_case(stems, mix, res_irm, sr)

    assert m_hpss["mean_active_si_sdr_db"] > m_id["mean_active_si_sdr_db"] + 2.0
    assert m_irm["mean_active_si_sdr_db"] > m_hpss["mean_active_si_sdr_db"] + 3.0
    assert m_hpss["mean_active_leakage_db"] < m_id["mean_active_leakage_db"]
    assert m_irm["mean_active_leakage_db"] < m_hpss["mean_active_leakage_db"]

    # Sparse control: solo vocals only -> drums/bass/other inactive with false_leakage_dbfs reported.
    sparse = _stems_case("e8-stems-ctl-solo-vocals-44k")
    sp_stems, sp_mix = e8_stems_eval.synthesize_stems_case(sparse)
    sp_res = e8_stems_eval.run_stems_arm("hpss_band_separator_cpu", sp_mix, int(sparse["rate"]))
    sp_m = e8_stems_eval.evaluate_stems_case(sp_stems, sp_mix, sp_res, int(sparse["rate"]))
    assert sp_m["per_stem"]["vocals"]["active"] is True
    for inactive in ("drums", "bass", "other"):
        assert sp_m["per_stem"][inactive]["active"] is False
        assert sp_m["per_stem"][inactive]["si_sdr_db"] is None
        assert sp_m["per_stem"][inactive]["false_leakage_dbfs"] is not None


def test_stems_external_audit_honest_blockers_and_colab_budget() -> None:
    """Contour-B audit honestly marks HTDemucs/SCNet/BS-RoFormer and MUSDB18-HQ."""
    audit = e8_stems_eval.audit_stems_external_models()
    assert audit["contour"] == "B_stems"
    assert audit["neural_arm_status"] == "blocked_external_weights_or_colab"
    assert audit["executed"] is False
    assert audit["production_dependency_added"] is False

    ds_map = {d["id"]: d["status"] for d in audit["external_datasets"]}
    assert ds_map["musdb18_hq"] == "not_provisioned_on_host"

    mdl_ids = {m["id"]: m for m in audit["external_neural_models"]}
    for mid in ("htdemucs_v4", "scnet", "bs_roformer"):
        assert mid in mdl_ids
        assert mdl_ids[mid]["status"] == "blocked_external_weights_or_colab"
        assert mdl_ids[mid]["license"]
        assert mdl_ids[mid]["colab_budget_estimate"]


# ---------------------------------------------------------------------------
# Combined Artifacts, Safety, Separate Go/No-Go & GFM Alignment Tests
# ---------------------------------------------------------------------------

def test_e8_artifacts_separate_go_no_go_provenance_e4_untouched_and_gfm_tables() -> None:
    """Verify two separate Go/No-Go decisions, E4 18/18 untouched, sealed untouched, 0 host paths, GFM tables."""
    e4_check = e8_stems_eval._verify_e4_freeze_untouched()
    assert e4_check["total_pinned_files"] == 18
    assert e4_check["all_untouched"] is True

    tracked_files = [
        REPO / "research" / "e8_voice_eval.py",
        REPO / "research" / "e8_stems_eval.py",
        REPO / "tests" / "test_e8_voice_stems.py",
        REPO / "corpus" / "dev-e8-voice-v1.json",
        REPO / "corpus" / "dev-e8-stems-v1.json",
        REPO / "docs" / "validation" / "e8-voice-stems-results.json",
        REPO / "docs" / "validation" / "e8-voice-stems-report.md",
    ]
    forbidden_tokens = ("/home/" + "mydum", "/media/" + "mydum")
    for path in tracked_files:
        assert path.exists(), f"missing expected E8 artifact: {path.name}"
        text = path.read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            assert tok not in text, f"host path {tok!r} leaked into {path.name}"

    res_doc = json.loads((REPO / "docs" / "validation" / "e8-voice-stems-results.json").read_text(encoding="utf-8"))
    assert res_doc["sealed_corpora_opened"] is False
    assert res_doc["preserve_pipeline_modified"] is False
    assert res_doc["colab_launched"] is False
    assert res_doc["production_dependencies_added"] is False
    assert res_doc["e4_freeze_integrity"]["all_untouched"] is True
    assert res_doc["e4_freeze_integrity"]["total_pinned_files"] == 18

    # Verify two distinct Go/No-Go decisions with non-generalization guard.
    gng_v = res_doc["go_no_go_decisions"]["go_no_go_1_voice"]
    gng_s = res_doc["go_no_go_decisions"]["go_no_go_2_stems"]
    assert gng_v["decision_id"] == "go_no_go_1_voice_profile"
    assert gng_s["decision_id"] == "go_no_go_2_stems_restoration"
    assert gng_v["contour"] == "A_voice"
    assert gng_s["contour"] == "B_stems"
    assert gng_v["production_verdict"] == "no_go"
    assert gng_s["production_verdict"] == "no_go"
    assert gng_v["preserve_stereo_music_generalization"] == "forbidden"
    assert gng_s["preserve_stereo_music_generalization"] == "forbidden"
    assert len(gng_v["decision_inputs_for_colab_voice_stage"]) >= 3
    assert len(gng_s["decision_inputs_for_colab_stems_stage"]) >= 3

    # Verify SHA-256 provenance matches files on disk.
    prov = res_doc["provenance"]
    assert prov["voice_manifest_sha256"] == corpus_v1.digest(REPO / prov["voice_manifest_path"])
    assert prov["voice_evaluator_sha256"] == corpus_v1.digest(REPO / prov["voice_evaluator_path"])
    assert prov["stems_manifest_sha256"] == corpus_v1.digest(REPO / prov["stems_manifest_path"])
    assert prov["stems_evaluator_sha256"] == corpus_v1.digest(REPO / prov["stems_evaluator_path"])

    # Verify GFM table alignment and required sections in the markdown report.
    report_text = (REPO / "docs" / "validation" / "e8-voice-stems-report.md").read_text(encoding="utf-8")
    assert "Go/No-Go #1" in report_text
    assert "Go/No-Go #2" in report_text
    assert "Limitations" in report_text
    assert "blocked_external_weights_or_colab" in report_text
    assert "license_required_for_pesq" in report_text
    assert "not_provisioned_on_host" in report_text

    expected_cols: int | None = None
    table_count = 0
    for line in report_text.splitlines():
        if line.startswith("|") and line.endswith("|"):
            # Ensure no unescaped pipes distort column counts.
            cols = len(line.split("|")) - 2
            if expected_cols is None:
                expected_cols = cols
                table_count += 1
            else:
                assert cols == expected_cols, f"misaligned GFM table row: {line}"
        else:
            expected_cols = None
    assert table_count >= 4


def test_sealed_path_rejected_before_io(tmp_path: Path) -> None:
    """Sealed paths are rejected before any read/write in both voice and stems CLIs."""
    with pytest.raises(ValueError, match="sealed_corpus_forbidden_in_e8"):
        e8_voice_eval.main(["--manifest", "corpus/sealed-v1.json"])
    with pytest.raises(ValueError, match="sealed_corpus_forbidden_in_e8"):
        e8_stems_eval.main(["--stems-manifest", "corpus/sealed-v1.json"])

    sealed_fake = tmp_path / "sealed-fake.json"
    with pytest.raises(ValueError, match="sealed_corpus_forbidden_in_e8"):
        e8_voice_eval.main(["--manifest", str(sealed_fake), "--build-manifest"])
    assert not sealed_fake.exists()

    with pytest.raises(ValueError, match="sealed_corpus_forbidden_in_e8"):
        e8_stems_eval.main(["--stems-manifest", str(sealed_fake), "--build-manifest"])
    assert not sealed_fake.exists()


def test_missing_manifest_and_disguised_tier_fail_closed(tmp_path: Path) -> None:
    """Missing manifests raise FileNotFoundError and tier=='sealed' JSON is rejected before decoding."""
    # 1. Manifest with tier=='sealed' under an innocuous filename rejected before decoding.
    disguised = tmp_path / "disguised-dev.json"
    disguised.write_text(json.dumps({"tier": "sealed", "schema": e8_voice_eval.VOICE_CORPUS_SCHEMA}), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed_corpus_forbidden_in_e8"):
        e8_voice_eval.run_e8_voice_eval(disguised)

    disguised_stems = tmp_path / "disguised-stems.json"
    disguised_stems.write_text(
        json.dumps({"tier": "sealed", "schema": e8_stems_eval.STEMS_CORPUS_SCHEMA}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sealed_corpus_forbidden_in_e8"):
        e8_stems_eval.run_e8_stems_eval(disguised_stems)

    # 2. Missing manifest without --build-manifest fails closed with FileNotFoundError.
    missing = tmp_path / "does-not-exist.json"
    with pytest.raises(FileNotFoundError):
        e8_voice_eval.main(["--manifest", str(missing)])
    assert not missing.exists()

    with pytest.raises(FileNotFoundError):
        e8_stems_eval.main(["--stems-manifest", str(missing)])
    assert not missing.exists()


def test_per_model_weight_audit_does_not_mark_all_models_ready_from_one_file(tmp_path: Path) -> None:
    """A single unverified weight file cannot mark other models or itself ready."""
    (tmp_path / "deepfilternet_unverified.pt").write_bytes(b"fake")
    v_audit = e8_voice_eval.audit_voice_external_models_and_metrics(models_dir=tmp_path)
    assert v_audit["neural_arm_status"] == "blocked_external_weights_or_colab"
    v_models = {m["id"]: m for m in v_audit["external_neural_models"]}
    assert len(v_models["deepfilternet3"]["local_weights_found"]) == 1
    assert v_models["deepfilternet3"]["license_verified"] is False
    assert v_models["deepfilternet3"]["status"] == "blocked_external_weights_or_colab"
    assert v_models["sgmse_plus"]["local_weights_found"] == []
    assert v_models["flow_matching_speech"]["local_weights_found"] == []

    (tmp_path / "htdemucs_unverified.th").write_bytes(b"fake")
    s_audit = e8_stems_eval.audit_stems_external_models(models_dir=tmp_path)
    assert s_audit["neural_arm_status"] == "blocked_external_weights_or_colab"
    s_models = {m["id"]: m for m in s_audit["external_neural_models"]}
    assert len(s_models["htdemucs_v4"]["local_weights_found"]) == 1
    assert s_models["htdemucs_v4"]["license_verified"] is False
    assert s_models["htdemucs_v4"]["status"] == "blocked_external_weights_or_colab"
    assert s_models["scnet"]["local_weights_found"] == []
    assert s_models["bs_roformer"]["local_weights_found"] == []


def test_zero_output_scores_floor_not_ceiling_on_active_speech() -> None:
    """Silent output on active speech/stems scores honest floor (-60 dB / 0.0), never ceiling."""
    v_case = _voice_case("e8-voice-ctl-clean-alto-16k")
    clean, _deg, _ = e8_voice_eval.synthesize_voice_case(v_case)
    sr = int(v_case["rate"])
    zeros = np.zeros_like(clean)

    assert e8_voice_eval.si_sdr_db(clean, zeros) <= -50.0
    assert e8_voice_eval.stoi_proxy_envelope_corr(clean, zeros, sr) <= 0.05
    assert e8_voice_eval.syllable_envelope_corr(clean, zeros, sr) <= 0.05

    # Also verify stems SI-SDR on silent estimate of an active stem.
    s_case = _stems_case("e8-stems-std-pop-rock-44k")
    stems, _mix = e8_stems_eval.synthesize_stems_case(s_case)
    assert e8_voice_eval.si_sdr_db(stems["vocals"], np.zeros_like(stems["vocals"])) <= -50.0


def test_wpe_solver_fallback_is_recorded_in_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linear solver failure inside WPE is explicitly recorded as solver_fallback=True."""
    v_case = _voice_case("e8-voice-rev-baritone-small-16k")
    _clean, degraded, _ = e8_voice_eval.synthesize_voice_case(v_case)

    def _boom(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced singular matrix")

    monkeypatch.setattr(np.linalg, "solve", _boom)
    res = e8_voice_eval.run_voice_arm("wpe_dereverb_cpu", degraded, int(v_case["rate"]), bypass_clean=False)
    assert res.meta["solver_fallback"] is True
    assert res.meta["solver_fallback_bins"] > 0
