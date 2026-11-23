"""E5 Sealed Holdout Evaluation & Blind Listening Panel tests.

All unit tests run strictly offline on synthetic data in `tmp_path` without
re-opening or re-executing the committed sealed holdout corpus.

Covers:
1. Custodian gate refusal without `--confirm-unseal-once` or `--custodian-release-token`.
2. Freeze integrity refusal on tampered `freeze_sha256` or manifest body.
3. Zero-mutation guard across ALL 18 frozen implementation SHA-256 pins.
4. Honest refusal on unprovisioned `corpus/sealed-v1.json` custodian stub.
5. End-to-end synthetic mini sealed run verifying finite JSON, E3 + E4 stereo
   safety metrics, and zero host path leakage (`/home/`, `/media/`).
6. Blind listening panel build: post-encode LUFS matching (`<= 0.2 LU`),
   `-1.0 dBTP` ceiling, zero method spoilers in `listening_sheet.md`, and
   custodian-held `answer_key_DO_NOT_OPEN_BEFORE_LISTENING.json`.
7. Blind listening panel scoring from both JSON responses and filled-in
   `listening_sheet.md`: binomial sign test, Wilcoxon signed-rank test,
   recording-cluster 95% CI, hidden-repeat consistency, and perceptual verdicts.
8. Committed E5 artifacts (`sealed-e5-v1.json`, `e5-sealed-results.json`,
   `e5-sealed-report.md`) integrity, zero host paths, zero leakage, and intact
   18-file E4 freeze.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research import corpus_synth, corpus_v1
from research.e5_blind_panel import (
    build_blind_panel,
    parse_listening_sheet_md,
    score_blind_panel,
)
from research.e5_sealed_eval import (
    EXPECTED_FREEZE_SHA256,
    FreezeIntegrityError,
    verify_freeze_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
E5_EVAL_CLI = ROOT / "research" / "e5_sealed_eval.py"
E5_PANEL_CLI = ROOT / "research" / "e5_blind_panel.py"
FREEZE_MANIFEST = ROOT / "docs" / "validation" / "e4-freeze-manifest.json"


def _make_mini_sealed_manifest(tmp_path: Path) -> tuple[Path, Path]:
    """Create a tiny 3-case synthetic sealed corpus/v1 manifest (2 defects + 1 control)."""
    audio = tmp_path / "mini_sealed_audio"
    freeze = json.loads(FREEZE_MANIFEST.read_text())
    pilot = dict(freeze["e5_preregistration"]["pilot_registration"])
    rights = {
        "local_inference": True,
        "colab": False,
        "training": False,
        "listening": True,
        "publication": False,
    }
    cases = [
        {
            "id": "ms-clip-1",
            "kind": "paired_synthetic",
            "sealed": True,
            "profile": "dense_symmetric_moderate",
            "rate": 44100,
            "channels": 2,
            "mode": "auto",
            "split": "eval",
            "rights": dict(rights),
            "groups": {k: f"ms-{k}-1" for k in corpus_v1.GROUP_KEYS},
            "clean": {"path": "ms-clip-1.clean.wav"},
            "degraded": {"path": "ms-clip-1.degraded.wav"},
            "synthesis": {
                "generator": corpus_synth.GENERATOR_V2,
                "seed": 61001,
                "rate": 44100,
                "seconds": 2,
                "operator": "hard_clip",
                "bits": 24,
                "threshold_dbfs": -6.0,
            },
        },
        {
            "id": "ms-clip-2",
            "kind": "paired_synthetic",
            "sealed": True,
            "profile": "asymmetric_sub_full_scale",
            "rate": 48000,
            "channels": 2,
            "mode": "auto",
            "split": "eval",
            "rights": dict(rights),
            "groups": {k: f"ms-{k}-2" for k in corpus_v1.GROUP_KEYS},
            "clean": {"path": "ms-clip-2.clean.wav"},
            "degraded": {"path": "ms-clip-2.degraded.wav"},
            "synthesis": {
                "generator": corpus_synth.GENERATOR_V2,
                "seed": 61002,
                "rate": 48000,
                "seconds": 2,
                "operator": "hard_clip",
                "bits": 16,
                "threshold_dbfs": -5.0,
                "pos_scale": 0.70,
            },
        },
        {
            "id": "ms-clean-ctrl",
            "kind": "paired_synthetic",
            "sealed": True,
            "profile": "clean_control",
            "rate": 44100,
            "channels": 2,
            "mode": "auto",
            "split": "control",
            "rights": dict(rights),
            "groups": {k: f"ms-{k}-3" for k in corpus_v1.GROUP_KEYS},
            "clean": {"path": "ms-clean-ctrl.clean.wav"},
            "degraded": {"path": "ms-clean-ctrl.degraded.wav"},
            "synthesis": {
                "generator": corpus_synth.GENERATOR_V2,
                "seed": 61003,
                "rate": 44100,
                "seconds": 2,
                "operator": "clean",
                "bits": 24,
            },
        },
    ]
    files = []
    for c in cases:
        specs = corpus_synth.synthesize_case(c, audio)
        for role in ("clean", "degraded", "mask"):
            c.setdefault(role, {}).update(specs[role])
            files.append(specs[role])
    doc = {
        "schema": corpus_v1.SCHEMA,
        "tier": "sealed",
        "dataset_version": "mini-sealed-test-1",
        "toolchain": corpus_v1.actual_toolchain(),
        "rng": {"algorithm": "numpy-pcg64", "seed": 61000},
        "numeric_tolerance": 1e-6,
        "output_ceiling_dbfs": -1.0,
        "custodian": {"name": "test custodian", "appointed_at": "2026-11-23"},
        "pilot": pilot,
        "files": files,
        "cases": cases,
    }
    mpath = tmp_path / "mini-sealed.json"
    mpath.write_text(json.dumps(doc, indent=2))
    return mpath, audio


def test_e5_refuses_without_confirm_unseal_once_and_token(tmp_path):
    mpath, _audio = _make_mini_sealed_manifest(tmp_path)
    out_json = tmp_path / "res.json"

    # Missing both flags
    p1 = subprocess.run(
        [sys.executable, str(E5_EVAL_CLI), "--manifest", str(mpath), "--results-json", str(out_json)],
        capture_output=True, text=True, check=False,
    )
    assert p1.returncode == 2
    assert "--confirm-unseal-once" in p1.stderr
    assert not out_json.exists()

    # Has --confirm-unseal-once but missing/empty token
    p2 = subprocess.run(
        [sys.executable, str(E5_EVAL_CLI), "--manifest", str(mpath),
         "--confirm-unseal-once", "--custodian-release-token", "   ",
         "--results-json", str(out_json)],
        capture_output=True, text=True, check=False,
    )
    assert p2.returncode == 2
    assert "--custodian-release-token" in p2.stderr
    assert not out_json.exists()


def test_e5_refuses_tampered_freeze_sha256_and_all_18_frozen_files(tmp_path):
    orig = json.loads(FREEZE_MANIFEST.read_text())
    impl = orig["toolchain"]["implementation_sha256"]
    assert len(impl) == 18

    # 1. Tampered freeze_sha256 string
    bad_hash_manifest = tmp_path / "bad_hash.json"
    bad_doc = dict(orig)
    bad_doc["freeze_sha256"] = "0" * 64
    bad_hash_manifest.write_text(json.dumps(bad_doc))
    with pytest.raises(FreezeIntegrityError, match="freeze_sha256 mismatch"):
        verify_freeze_integrity(bad_hash_manifest, EXPECTED_FREEZE_SHA256, repo_root=ROOT)

    # 2. Tampered manifest body while keeping old freeze_sha256
    bad_body_manifest = tmp_path / "bad_body.json"
    bad_body = json.loads(FREEZE_MANIFEST.read_text())
    bad_body["frozen_at"] = "2099-01-01"
    bad_body_manifest.write_text(json.dumps(bad_body))
    with pytest.raises(FreezeIntegrityError, match="content tampered"):
        verify_freeze_integrity(bad_body_manifest, EXPECTED_FREEZE_SHA256, repo_root=ROOT)

    # 3. Every single one of the 18 frozen implementation files must be verified!
    mpath, _audio = _make_mini_sealed_manifest(tmp_path)
    for idx, rel in enumerate(sorted(impl)):
        tampered = json.loads(FREEZE_MANIFEST.read_text())
        tampered["toolchain"]["implementation_sha256"][rel] = "f" * 64
        # Recompute canonical freeze_sha256 so ONLY the implementation file check triggers!
        body = {k: v for k, v in tampered.items() if k != "freeze_sha256"}
        import hashlib
        recomputed = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        tampered["freeze_sha256"] = recomputed
        tpath = tmp_path / f"tampered_{idx}.json"
        tpath.write_text(json.dumps(tampered))
        with pytest.raises(FreezeIntegrityError, match="ZERO-MUTATION VIOLATION"):
            verify_freeze_integrity(tpath, recomputed, repo_root=ROOT, check_toolchain=False)

    # Verify CLI exits with code 2 on drifted frozen file too.
    first_tampered = tmp_path / "tampered_0.json"
    first_sha = json.loads(first_tampered.read_text())["freeze_sha256"]
    proc = subprocess.run(
        [sys.executable, str(E5_EVAL_CLI), "--manifest", str(mpath),
         "--freeze-manifest", str(first_tampered),
         "--expected-freeze-sha256", first_sha,
         "--confirm-unseal-once", "--custodian-release-token", "TEST-TOKEN"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 2
    assert "ZERO-MUTATION VIOLATION" in proc.stderr


def test_e5_refuses_unprovisioned_sealed_v1_stub(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(E5_EVAL_CLI),
         "--manifest", str(ROOT / "corpus" / "sealed-v1.json"),
         "--confirm-unseal-once", "--custodian-release-token", "TEST-TOKEN",
         "--results-json", str(tmp_path / "never.json")],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 2
    assert "sealed tier blocked" in proc.stderr
    assert not (tmp_path / "never.json").exists()


def test_e5_mini_synthetic_run_and_no_host_path_leaks(tmp_path):
    mpath, audio = _make_mini_sealed_manifest(tmp_path)
    out_json = tmp_path / "mini_results.json"
    out_md = tmp_path / "mini_report.md"

    proc = subprocess.run(
        [sys.executable, str(E5_EVAL_CLI),
         "--manifest", str(mpath),
         "--audio-root", str(audio),
         "--confirm-unseal-once",
         "--custodian-release-token", "SECRET-UNIT-TOKEN-12345",
         "--custodian-reason", "Unit test synthetic run",
         "--results-json", str(out_json),
         "--report-md", str(out_md)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    jtxt = out_json.read_text()
    mtxt = out_md.read_text()
    for blob in (jtxt, mtxt):
        assert str(tmp_path) not in blob
        assert "/home/" not in blob
        assert "/media/" not in blob
        assert "SECRET-UNIT-TOKEN-12345" not in blob

    doc = json.loads(jtxt)
    assert doc["schema"] == "e5-sealed-eval/v1"
    assert doc["freeze_verification"]["verified"] is True
    assert doc["freeze_verification"]["verified_files_count"] == 18
    assert doc["freeze_verification"]["freeze_sha256"] == EXPECTED_FREEZE_SHA256
    assert doc["sealed_v1_audit"]["status"] == "not_provisioned_on_host"

    cand = doc["summary"]["arms"]["aspade_mit__v2_windowed"]
    assert cand["control_fp_count"] == 0
    assert cand["polarity_violations_total"] == 0
    assert cand["max_reliable_diff_pre_gain"] == 0.0
    assert cand["output_tp_violation_count"] == 0
    assert cand["stereo_safety_violation_count"] == 0
    assert cand["coverage"] == 1.0
    assert doc["summary"]["decision"]["blind_panel_status"] == "pending_human_listening"
    assert doc["summary"]["decision"]["release_verdict"] in (
        "inconclusive_pending_blind_panel", "no_promote"
    )


def test_blind_panel_build_lufs_matching_blindness_and_scoring(tmp_path):
    # Build 4 synthetic stereo sources (3 clipped defects + 1 clean control)
    sources = []
    for idx, (sr, thr, is_def) in enumerate(
        [(44100, -5.5, True), (48000, -6.5, True), (44100, -5.0, True), (48000, None, False)],
        start=1,
    ):
        clean = corpus_synth._clean_music_signal(seed=72000 + idx, rate=sr, seconds=2.0)
        if is_def:
            deg, mask, _ = corpus_synth.apply_operator(
                clean, sr, {"operator": "hard_clip", "threshold_dbfs": thr, "bits": 24},
                seed=72100 + idx,
            )
            assert mask.any()
        else:
            deg = clean.copy()
        sources.append({
            "source_id": f"syn-{idx:02d}",
            "cluster_id": f"cluster-{idx:02d}",
            "stratum": "unit_test_synth",
            "profile": "hard_clip" if is_def else "clean_control",
            "is_defect": is_def,
            "sr": sr,
            "bits": 24,
            "clean": clean,
            "degraded": deg,
        })

    panel_dir = tmp_path / "blind_panel"
    key = build_blind_panel(
        sources,
        panel_dir,
        seed=1234567,
        target_lufs=-18.0,
        comparators=("identity", "ss_pew_mit__v2_windowed"),
        n_hidden_repeats=2,
    )
    assert key["n_items"] == 9  # 3*2 defect trials + 1 control + 2 hidden repeats
    assert key["n_hidden_repeats"] == 2

    # 1. Verify audio files, LUFS tolerance <= 0.2 LU, and TP <= -1.0 + 0.05 dBTP
    for it in key["items"]:
        assert it["lufs_diff_ab"] <= 0.2
        for lbl in ("REF", "A", "B"):
            assert (panel_dir / it["files"][lbl]["path"]).is_file()
        assert it["files"]["A"]["tp_dbfs"] <= -0.95
        assert it["files"]["B"]["tp_dbfs"] <= -0.95

    # 2. Verify blindness of listening_sheet.md (zero method spoilers, zero seed, zero host paths)
    sheet_text = (panel_dir / "listening_sheet.md").read_text(encoding="utf-8")
    for forbidden in ("aspade", "ss_pew", "current_snapshot", "1234567", "/home/", "/media/"):
        assert forbidden not in sheet_text.lower()

    # 3. Score via JSON responses where listener strongly prefers candidate over identity
    #    (5/5 wins with +20 score diff across 2 listeners -> confirmed_preference!)
    resp_a = {"schema": "e5-blind-panel-responses/v1", "listener_id": "L1", "responses": []}
    resp_b = {"schema": "e5-blind-panel-responses/v1", "listener_id": "L2", "responses": []}
    for it in key["items"]:
        cand_lbl = it["candidate_label"]
        if it["is_defect"] and it["comparator"] == "identity":
            sa = 88.0 if cand_lbl == "A" else 64.0
            sb = 64.0 if cand_lbl == "A" else 88.0
            pref = cand_lbl
        elif not it["is_defect"]:
            sa, sb, pref = 95.0, 95.0, "tie"
        else:
            sa = 84.0 if cand_lbl == "A" else 82.0
            sb = 82.0 if cand_lbl == "A" else 84.0
            pref = cand_lbl
        resp_a["responses"].append({"item_id": it["item_id"], "preference": pref, "score_a": sa, "score_b": sb})
        resp_b["responses"].append({"item_id": it["item_id"], "preference": pref, "score_a": sa - 1, "score_b": sb - 1})

    scored = score_blind_panel(key, [resp_a, resp_b])
    assert scored["n_listeners"] == 2
    assert scored["verdict"] == "confirmed_preference"
    assert scored["comparisons"]["identity"]["wins_candidate"] == 6
    assert scored["comparisons"]["identity"]["losses_comparator"] == 0
    assert scored["comparisons"]["identity"]["binomial_two_sided_p"] < 0.05
    assert scored["comparisons"]["identity"]["wilcoxon_two_sided_p"] < 0.05
    assert scored["repeat_reliability"]["preference_agreement_rate"] == 1.0
    json.dumps(scored, allow_nan=False)

    # 4. Score when listener detects harm (always prefers identity over candidate) -> harm_detected!
    resp_harm = {"schema": "e5-blind-panel-responses/v1", "listener_id": "L_harm", "responses": []}
    for it in key["items"]:
        comp_lbl = it["comparator_label"]
        sa = 85.0 if comp_lbl == "A" else 55.0
        sb = 55.0 if comp_lbl == "A" else 85.0
        resp_harm["responses"].append({"item_id": it["item_id"], "preference": comp_lbl, "score_a": sa, "score_b": sb})
    scored_harm = score_blind_panel(key, [resp_harm, resp_harm])
    assert scored_harm["verdict"] == "harm_detected"

    # 5. Test Markdown listening_sheet.md parsing + CLI `e5_blind_panel.py score`
    filled_lines = []
    key_map = {it["item_id"]: it for it in key["items"]}
    for line in sheet_text.splitlines():
        if line.startswith("| `item_"):
            iid = line.split("`")[1]
            it = key_map[iid]
            pref = it["candidate_label"] if it["is_defect"] else "tie"
            sa = 85 if pref == "A" else (70 if pref == "B" else 90)
            sb = 70 if pref == "A" else (85 if pref == "B" else 90)
            parts = line.split("|")
            parts[5] = f" {pref} "
            parts[6] = f" {sa} "
            parts[7] = f" {sb} "
            parts[8] = " clean "
            filled_lines.append("|".join(parts))
        else:
            filled_lines.append(line)
    filled_md = panel_dir / "listening_sheet_filled.md"
    filled_md.write_text("\n".join(filled_lines) + "\n", encoding="utf-8")

    parsed = parse_listening_sheet_md(filled_md)
    assert len(parsed["responses"]) == key["n_items"]

    proc_sc = subprocess.run(
        [sys.executable, str(E5_PANEL_CLI), "score",
         "--panel-dir", str(panel_dir),
         "--responses", str(filled_md)],
        capture_output=True, text=True, check=False,
    )
    assert proc_sc.returncode == 0, proc_sc.stderr
    assert (panel_dir / "panel_score_summary.json").is_file()
    assert (panel_dir / "panel_score_report.md").is_file()


def test_committed_e5_artifacts_valid_and_e4_freeze_intact():
    # 1. All 18 E4 frozen files and freeze_sha256 remain 100% untouched!
    freeze = verify_freeze_integrity(FREEZE_MANIFEST, EXPECTED_FREEZE_SHA256, repo_root=ROOT, check_toolchain=False)
    assert freeze["freeze_sha256"] == EXPECTED_FREEZE_SHA256

    # 2. Committed sealed-e5-v1.json, e5-sealed-results.json, and e5-sealed-report.md exist and are clean.
    manifest_p = ROOT / "corpus" / "sealed-e5-v1.json"
    results_p = ROOT / "docs" / "validation" / "e5-sealed-results.json"
    report_p = ROOT / "docs" / "validation" / "e5-sealed-report.md"
    for p in (manifest_p, results_p, report_p):
        assert p.is_file(), f"missing E5 artifact: {p.name}"
        text = p.read_text(encoding="utf-8")
        assert "/home/" not in text, f"host path leaked in {p.name}"
        assert "/media/" not in text, f"host path leaked in {p.name}"

    # 3. Zero cross-manifest group leakage across ALL 5 committed manifests.
    all_manifests = [
        corpus_v1.load(ROOT / "corpus" / name)
        for name in ("smoke-v1.json", "dev-v1.json", "dev-e3-v1.json", "sealed-v1.json", "sealed-e5-v1.json")
    ]
    assert corpus_v1.validate_cross_manifest_leakage(all_manifests) == []

    # 4. E5 results schema & freeze pin checks.
    res = json.loads(results_p.read_text(encoding="utf-8"))
    assert res["schema"] == "e5-sealed-eval/v1"
    assert res["freeze_verification"]["verified"] is True
    assert res["freeze_verification"]["freeze_sha256"] == EXPECTED_FREEZE_SHA256
    assert res["freeze_verification"]["verified_files_count"] == 18
    assert res["protocol"]["n_defects"] == 30
    assert res["protocol"]["n_controls"] == 10
    assert res["protocol"]["cross_manifest_leakage_errors"] == []
