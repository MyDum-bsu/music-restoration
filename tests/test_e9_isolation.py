#!/usr/bin/env python3
"""E9 Legacy & ML Isolation, Offline Preserve E2E Smoke, and ML Governance tests."""
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

from research import e9_isolation_audit as AUDIT
from research import e9_ml_governance as GOV

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = REPO_ROOT / "docs" / "validation" / "e9-legacy-isolation-report.md"
RESULTS_JSON = REPO_ROOT / "docs" / "validation" / "e9-legacy-isolation-results.json"


# ── 1. Legacy compatibility smoke ────────────────────────────────────────────


def test_legacy_modules_and_helpers_remain_importable():
    """Legacy runners and ML modules are preserved (never deleted) and import cleanly."""
    for mod_name in (
        "rnext.ml.dataset",
        "rnext.ml.metrics",
        "rnext.ml.models",
        "rnext.ml.registry",
        "rnext.ml.runtime",
        "scripts.restore_one",
        "scripts.restore_per_track",
    ):
        mod = importlib.import_module(mod_name)
        assert mod is not None

    restore_one = importlib.import_module("scripts.restore_one")
    restore_per_track = importlib.import_module("scripts.restore_per_track")
    assert callable(restore_one.encode_flac)
    assert restore_one._clean_title("01 - Track Name") == "Track Name"
    assert restore_per_track.track_no("03. Song.flac") == 3


def test_repaired_research_runbook_scripts_point_to_existing_files():
    """Verify repaired subprocess script paths in research/dataset_loop.py & status_report.py."""
    pat = re.compile(r"(?:scripts|research)/([A-Za-z0-9_-]+\.py)")
    for rel in ("research/dataset_loop.py", "research/status_report.py"):
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for match in pat.finditer(src):
            full_rel = match.group(0)
            assert (REPO_ROOT / full_rel).is_file(), (
                f"Broken script reference '{full_rel}' in {rel}"
            )


# ── 2. Static AST + Isolated Offline/No-ML Preserve E2E smoke ────────────────


def test_preserve_static_import_audit_passes():
    """Preserve modules must never import joblib, torch, sklearn, rnext.ml, research, or lab."""
    res = AUDIT.audit_preserve_static_imports(REPO_ROOT)
    assert res["static_audit_passed"] is True
    assert res["preserve_to_ml_violations"] == []
    assert res["ml_to_preserve_violations"] == []
    assert res["restore_one_top_level_forbidden_imports"] == []
    assert res["preserve_modules_checked"] >= 18


def test_preserve_e2e_smoke_without_ml_and_without_network():
    """Run Preserve CLI in an isolated subprocess with ML libraries & sockets hard-blocked."""
    smoke = AUDIT.run_preserve_isolated_e2e_smoke(REPO_ROOT)
    assert smoke["returncode"] == 0
    assert smoke["network_blocked"] is True
    assert smoke["loaded_forbidden_modules"] == []
    assert smoke["clean_control"]["status"] == "kept_original"
    assert smoke["clean_control"]["byte_identical_to_source"] is True
    assert smoke["hard_clipped_case"]["status"] == "processed_candidate"
    assert smoke["hard_clipped_case"]["encoded_tp_dbfs"] <= -0.95
    assert smoke["hard_clipped_case"]["encoded_clip_samples"] == 0
    assert smoke["e2e_smoke_passed"] is True


# ── 3. ML-target governance & split-boundary contract ────────────────────────


def _valid_four_stage_records() -> list[GOV.MLRecord]:
    return [
        GOV.MLRecord("r1", "rec-01", "grp-01", "approved_labels", "approved", "1" * 64),
        GOV.MLRecord("r2", "rec-02", "grp-02", "dev_selection", "approved", "2" * 64),
        GOV.MLRecord("r3", "rec-03", "grp-03", "promotion", "approved", "3" * 64),
        GOV.MLRecord("r4", "rec-04", "grp-04", "sealed_test", "approved", "4" * 64),
    ]


def test_ml_governance_valid_four_stage_manifest_and_versioning():
    records = _valid_four_stage_records()
    summary = GOV.validate_split_boundaries(records, require_all_stages=True)
    assert summary["valid"] is True
    assert summary["total_records"] == 4
    assert summary["cross_stage_leakage_errors"] == []
    assert summary["dataset_version"].startswith("ml-ds-v1-")

    # Deterministic dataset_version regardless of input order.
    summary_rev = GOV.validate_split_boundaries(list(reversed(records)), require_all_stages=True)
    assert summary_rev["dataset_version"] == summary["dataset_version"]

    # Versioned result integrity + tamper detection.
    res = GOV.build_versioned_result(
        stage="dev_selection",
        records=records,
        metrics={"median_abs_err_db": 0.35},
        model_id="ridge-dev-01",
        model_sha256="a" * 64,
        git_commit="test-sha",
    )
    assert GOV.verify_versioned_result(res) is True

    tampered = dict(res)
    tampered["metrics"] = {"median_abs_err_db": 0.01}
    with pytest.raises(GOV.GovernanceError, match="results_sha256 mismatch"):
        GOV.verify_versioned_result(tampered)


def test_ml_governance_recording_deduplication():
    records = [
        GOV.MLRecord("r2", "rec-dup", "grp-01", "approved_labels", "approved", "1" * 64),
        GOV.MLRecord("r1", "rec-dup", "grp-01", "approved_labels", "approved", "1" * 64),
        GOV.MLRecord("r3", "rec-02", "grp-02", "dev_selection", "approved", "2" * 64),
    ]
    # Direct validation fails closed on un-deduplicated recording_id.
    with pytest.raises(GOV.GovernanceError, match="Duplicate recording_id"):
        GOV.validate_split_boundaries(records)

    kept, dropped = GOV.deduplicate_by_recording(records)
    assert len(kept) == 2
    assert len(dropped) == 1
    assert dropped[0]["dropped_record_id"] == "r2"
    assert dropped[0]["kept_record_id"] == "r1"

    summary = GOV.validate_split_boundaries(kept)
    assert summary["valid"] is True
    assert summary["unique_recordings"] == 2


@pytest.mark.parametrize(
    "leak_field,r_a,r_b",
    [
        (
            "recording_id",
            GOV.MLRecord("a1", "shared-rec", "grp-1", "approved_labels", "approved", "1" * 64),
            GOV.MLRecord("b1", "shared-rec", "grp-2", "dev_selection", "approved", "2" * 64),
        ),
        (
            "group_id",
            GOV.MLRecord("a1", "rec-1", "shared-grp", "dev_selection", "approved", "1" * 64),
            GOV.MLRecord("b1", "rec-2", "shared-grp", "sealed_test", "approved", "2" * 64),
        ),
        (
            "content_sha256",
            GOV.MLRecord("a1", "rec-1", "grp-1", "approved_labels", "approved", "f" * 64),
            GOV.MLRecord("b1", "rec-2", "grp-2", "sealed_test", "approved", "f" * 64),
        ),
    ],
)
def test_ml_governance_blocks_cross_stage_leakage(leak_field, r_a, r_b):
    with pytest.raises(GOV.GovernanceError, match=leak_field):
        GOV.validate_split_boundaries([r_a, r_b])


def test_ml_governance_rejects_unapproved_label_and_sealed_in_dev():
    unapproved = GOV.MLRecord("u1", "rec-u", "grp-u", "approved_labels", "provisional", "9" * 64)
    with pytest.raises(GOV.GovernanceError, match="label_status='approved'"):
        GOV.validate_split_boundaries([unapproved])

    sealed_leak = GOV.MLRecord(
        "s1",
        "rec-s",
        "grp-s",
        "dev_selection",
        "approved",
        "8" * 64,
        source_manifest="corpus/sealed-v1.json",
    )
    with pytest.raises(GOV.GovernanceError, match="Sealed corpus reference"):
        GOV.validate_split_boundaries([sealed_leak])


def test_ml_governance_bidirectional_boundary_audit():
    audit = GOV.audit_ml_preserve_boundary(REPO_ROOT)
    assert audit["bidirectional_isolation_passed"] is True
    assert audit["preserve_to_ml_violations"] == []
    assert audit["ml_to_preserve_violations"] == []
    # Transitive traversal covers local helper modules beyond the core seed list.
    assert len(audit["preserve_transitive_modules"]) > len(GOV.PRESERVE_CORE_MODULES)
    assert "rnext/dsp/adaptive.py" in audit["preserve_transitive_modules"]
    assert "rnext/phone/config.py" in audit["preserve_transitive_modules"]


def test_ml_governance_dedup_catches_cross_split_leak_before_dropping_and_case_sha():
    """Blocker #1: dropped duplicate cannot hide cross-split leak; SHA-256 is case-insensitive."""
    # Case-insensitive SHA-256 collision across stages.
    upper_sha = "ABCDEF" + "0" * 58
    lower_sha = upper_sha.lower()
    with pytest.raises(GOV.GovernanceError, match="content_sha256"):
        GOV.validate_split_boundaries(
            [
                GOV.MLRecord("a1", "rec-1", "grp-1", "approved_labels", "approved", upper_sha),
                GOV.MLRecord("b1", "rec-2", "grp-2", "sealed_test", "approved", lower_sha),
            ]
        )

    # Dropped duplicate in approved_labels shares group_id with sealed_test.
    records_group_leak = [
        GOV.MLRecord("a1", "rec-dup", "grp-safe", "approved_labels", "approved", "1" * 64),
        GOV.MLRecord("a2", "rec-dup", "grp-leaked", "approved_labels", "approved", "2" * 64),
        GOV.MLRecord("s1", "rec-sealed", "grp-leaked", "sealed_test", "approved", "3" * 64),
    ]
    with pytest.raises(GOV.GovernanceError, match="group_id"):
        GOV.deduplicate_by_recording(records_group_leak)

    # Dropped duplicate in approved_labels shares case-variant content_sha256 with sealed_test.
    records_sha_leak = [
        GOV.MLRecord("a1", "rec-dup", "grp-1", "approved_labels", "approved", "1" * 64),
        GOV.MLRecord("a2", "rec-dup", "grp-1", "approved_labels", "approved", upper_sha),
        GOV.MLRecord("s1", "rec-sealed", "grp-2", "sealed_test", "approved", lower_sha),
    ]
    with pytest.raises(GOV.GovernanceError, match="content_sha256"):
        GOV.deduplicate_by_recording(records_sha_leak)


def test_ml_governance_versioning_fail_closed_rules():
    """Blocker #2: forbid 0-record sealed_test, sealed_audio_opened in non-sealed, empty dataset_version."""
    dev_only = [
        GOV.MLRecord("r1", "rec-01", "grp-01", "approved_labels", "approved", "1" * 64),
        GOV.MLRecord("r2", "rec-02", "grp-02", "dev_selection", "approved", "2" * 64),
    ]
    with pytest.raises(GOV.GovernanceError, match="0 records in stage 'sealed_test'"):
        GOV.build_versioned_result(
            stage="sealed_test",
            records=dev_only,
            metrics={"median_abs_err_db": 0.2},
            model_id="m1",
            model_sha256="a" * 64,
        )

    # Even with a recomputed valid results_sha256, verify_versioned_result rejects
    # sealed_audio_opened=True for non-sealed stages and empty dataset_version.
    valid = GOV.build_versioned_result(
        stage="dev_selection",
        records=_valid_four_stage_records(),
        metrics={"median_abs_err_db": 0.3},
        model_id="m1",
        model_sha256="a" * 64,
    )
    bad_sealed = {k: v for k, v in valid.items() if k != "results_sha256"}
    bad_sealed["sealed_audio_opened"] = True
    bad_sealed["results_sha256"] = GOV.compute_sha256_json(bad_sealed)
    with pytest.raises(GOV.GovernanceError, match="sealed_audio_opened=True"):
        GOV.verify_versioned_result(bad_sealed)

    bad_dv = {k: v for k, v in valid.items() if k != "results_sha256"}
    bad_dv["dataset_version"] = "garbage"
    bad_dv["results_sha256"] = GOV.compute_sha256_json(bad_dv)
    with pytest.raises(GOV.GovernanceError, match="Invalid dataset_version"):
        GOV.verify_versioned_result(bad_dv)

    bad_sha = {k: v for k, v in valid.items() if k != "results_sha256"}
    bad_sha["model_sha256"] = "not-a-sha"
    bad_sha["results_sha256"] = GOV.compute_sha256_json(bad_sha)
    with pytest.raises(GOV.GovernanceError, match="Invalid model_sha256"):
        GOV.verify_versioned_result(bad_sha)

    null_sha = {k: v for k, v in valid.items() if k != "results_sha256"}
    null_sha["model_sha256"] = None
    null_sha["results_sha256"] = GOV.compute_sha256_json(null_sha)
    with pytest.raises(GOV.GovernanceError, match="Invalid model_sha256"):
        GOV.verify_versioned_result(null_sha)

    with pytest.raises(GOV.GovernanceError, match="model_sha256"):
        GOV.build_versioned_result(
            stage="dev_selection",
            records=_valid_four_stage_records(),
            metrics={"median_abs_err_db": 0.3},
            model_id="m1",
            model_sha256="not-a-sha",
        )


def test_sealed_symlink_alias_and_resolve_oserror_fail_closed(tmp_path, monkeypatch):
    """Blocker #3: any symlink component and resolve()/lstat OSError/EACCES fail closed."""
    import os

    sealed_real = REPO_ROOT / "corpus" / "sealed-v1.json"
    alias = tmp_path / "harmless_dev_alias.json"
    alias.symlink_to(sealed_real)

    with pytest.raises(GOV.GovernanceError, match="symlink component"):
        GOV.assert_not_sealed_source(alias, stage="dev_selection")

    with pytest.raises(ValueError, match="symlink component"):
        AUDIT._assert_not_sealed_path(alias)

    # Even a symlink inside a parent directory component is rejected fail-closed.
    dir_link = tmp_path / "dir_alias"
    dir_link.symlink_to(REPO_ROOT / "corpus")
    with pytest.raises(ValueError, match="symlink component"):
        AUDIT._assert_not_sealed_path(dir_link / "any_file.json")

    def _eacces(_path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "lstat", _eacces)
    with pytest.raises(ValueError, match="Fail-closed: unable to resolve path"):
        AUDIT._assert_not_sealed_path(tmp_path / "any_out.json")
    with pytest.raises(GOV.GovernanceError, match="Fail-closed: unable to resolve source_manifest"):
        GOV.assert_not_sealed_source(tmp_path / "any_dev.json", stage="dev_selection")


def test_ast_import_audit_catches_from_rnext_import_ml_and_transitive_leak(tmp_path):
    """Blocker #4: `from pkg import name` expands to `pkg.name` and walks transitive local modules + parent __init__.py."""
    # Minimal synthetic repo with Preserve seed -> local helper -> `from rnext import ml`.
    for rel in GOV.PRESERVE_CORE_MODULES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# clean\n", encoding="utf-8")
    for rel in GOV.ML_CONTOUR_MODULES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# clean\n", encoding="utf-8")

    (tmp_path / "rnext" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "rnext" / "dsp" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "rnext" / "helper_transitive.py").write_text(
        "from rnext import ml\n", encoding="utf-8"
    )
    (tmp_path / "rnext" / "features.py").write_text(
        "from rnext import helper_transitive\n", encoding="utf-8"
    )

    audit = GOV.audit_ml_preserve_boundary(tmp_path)
    assert audit["bidirectional_isolation_passed"] is False
    assert "rnext/helper_transitive.py" in audit["preserve_transitive_modules"]
    assert any(
        v["file"] == "rnext/helper_transitive.py" and v["import"] == "rnext.ml"
        for v in audit["preserve_to_ml_violations"]
    )

    # Also verify parent package __init__.py is always audited when importing a clean submodule.
    (tmp_path / "rnext" / "helper_transitive.py").write_text("# clean\n", encoding="utf-8")
    subpkg = tmp_path / "rnext" / "subpkg"
    subpkg.mkdir(parents=True, exist_ok=True)
    (subpkg / "__init__.py").write_text("import joblib\n", encoding="utf-8")
    (subpkg / "leaf.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "rnext" / "features.py").write_text(
        "from rnext.subpkg.leaf import VALUE\n", encoding="utf-8"
    )

    audit_init = GOV.audit_ml_preserve_boundary(tmp_path)
    assert audit_init["bidirectional_isolation_passed"] is False
    assert "rnext/subpkg/__init__.py" in audit_init["preserve_transitive_modules"]
    assert any(
        v["file"] == "rnext/subpkg/__init__.py" and v["import"] == "joblib"
        for v in audit_init["preserve_to_ml_violations"]
    )


def test_resume_sh_fails_closed_when_data_root_unresolved(tmp_path):
    """Blocker #5: research/resume.sh exits non-zero on unresolved data_root or missing MR_DATA_ROOT without calling udisksctl."""
    import os
    import subprocess

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_py = fake_bin / "python3"
    fake_py.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    fake_py.chmod(0o755)
    marker = tmp_path / "udisksctl_called"
    fake_udisks = fake_bin / "udisksctl"
    fake_udisks.write_text(
        f"#!/usr/bin/env bash\ntouch '{marker}'\nexit 0\n", encoding="utf-8"
    )
    fake_udisks.chmod(0o755)

    env = os.environ.copy()
    env.pop("MR_DATA_ROOT", None)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"

    proc = subprocess.run(
        ["bash", str(REPO_ROOT / "research" / "resume.sh")],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "failed to resolve data_root" in proc.stderr
    assert not marker.exists()

    # Custom MR_DATA_ROOT pointing to non-existent directory must fail closed without udisksctl.
    env_custom = os.environ.copy()
    env_custom["MR_DATA_ROOT"] = str(tmp_path / "does_not_exist")
    env_custom["PATH"] = f"{fake_bin}:{env_custom.get('PATH', '')}"
    proc_custom = subprocess.run(
        ["bash", str(REPO_ROOT / "research" / "resume.sh")],
        cwd=str(REPO_ROOT),
        env=env_custom,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc_custom.returncode != 0
    assert "custom MR_DATA_ROOT does not exist" in proc_custom.stderr
    assert not marker.exists()


# ── 4. Security, Freeze, Portability & Report Integrity ──────────────────────


def test_e4_freeze_18_of_18_untouched_and_sealed_guard():
    e4 = AUDIT.verify_e4_freeze_untouched(REPO_ROOT)
    assert e4["total_pinned"] == 18
    assert e4["untouched_count"] == 18
    assert e4["all_untouched"] is True
    assert e4["mismatches"] == []

    for sealed in ("corpus/sealed-v1.json", "corpus/sealed-e5-v1.json"):
        with pytest.raises(ValueError, match="sealed corpus"):
            AUDIT._assert_not_sealed_path(REPO_ROOT / sealed)


def test_e9_artifacts_integrity_zero_host_paths_and_gfm_tables():
    assert RESULTS_JSON.is_file(), "Missing docs/validation/e9-legacy-isolation-results.json"
    assert REPORT_MD.is_file(), "Missing docs/validation/e9-legacy-isolation-report.md"

    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    assert data["schema_version"] == AUDIT.SCHEMA_VERSION
    assert data["sealed_corpora_opened"] is False
    assert data["e4_freeze_verification"]["all_untouched"] is True
    assert data["e4_freeze_verification"]["untouched_count"] == 18
    assert data["static_import_audit"]["static_audit_passed"] is True
    assert data["isolated_preserve_e2e_smoke"]["e2e_smoke_passed"] is True
    assert data["all_checks_passed"] is True

    # Verify results_sha256 matches payload.
    recorded_sha = data["results_sha256"]
    body = {k: v for k, v in data.items() if k != "results_sha256"}
    assert GOV.compute_sha256_json(body) == recorded_sha

    # Zero host paths across all new/modified E9 files.
    e9_files = [
        REPO_ROOT / "research" / "e9_ml_governance.py",
        REPO_ROOT / "research" / "e9_isolation_audit.py",
        REPO_ROOT / "tests" / "test_e9_isolation.py",
        RESULTS_JSON,
        REPORT_MD,
        REPO_ROOT / "research" / "dataset_loop.py",
        REPO_ROOT / "research" / "status_report.py",
        REPO_ROOT / "research" / "lab" / "server.py",
        REPO_ROOT / "research" / "lab" / "README.md",
        REPO_ROOT / "research" / "docs" / "ml.md",
        REPO_ROOT / "research" / "docs" / "dataset.md",
        REPO_ROOT / "research" / "skills" / "restore-status" / "SKILL.md",
        REPO_ROOT / "research" / "resume.sh",
        REPO_ROOT / "scripts" / "release_batch.py",
        REPO_ROOT / "scripts" / "restore_one.py",
        REPO_ROOT / "docs" / "setup.md",
        REPO_ROOT / "README.md",
    ]
    host_audit = AUDIT.audit_no_host_paths(e9_files, REPO_ROOT)
    assert host_audit["passed"] is True, f"Found host paths: {host_audit['host_path_hits']}"

    # Verify GFM table alignment in docs/validation/e9-legacy-isolation-report.md.
    md_lines = REPORT_MD.read_text(encoding="utf-8").splitlines()
    tables_checked = 0
    i = 0
    while i < len(md_lines):
        if md_lines[i].startswith("|") and md_lines[i].endswith("|"):
            block: list[str] = []
            while i < len(md_lines) and md_lines[i].startswith("|") and md_lines[i].endswith("|"):
                block.append(md_lines[i])
                i += 1
            assert len(block) >= 3, f"GFM table too short: {block}"
            col_counts = [len([c for c in row.strip("|").split("|")]) for row in block]
            assert len(set(col_counts)) == 1, f"Misaligned GFM table columns: {col_counts}"
            sep_cells = [c.strip() for c in block[1].strip("|").split("|")]
            assert all(re.fullmatch(r":?-{3,}:?", c) for c in sep_cells), (
                f"Invalid GFM separator row: {block[1]}"
            )
            tables_checked += 1
        else:
            i += 1
    assert tables_checked >= 4
