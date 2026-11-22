"""corpus/v1: validator gates and two-environment runner reproducibility."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from research import corpus_synth, corpus_v1, run_benchmark

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "research" / "run_benchmark.py"


def mini_manifest(tmp_path):
    """Two tiny synthetic cases generated into tmp_path/audio."""
    audio = tmp_path / "audio"
    cases = [
        {"id": "m-clip", "kind": "paired_synthetic", "profile": "dense",
         "rate": 44100, "channels": 2, "mode": "auto", "split": "eval",
         "rights": {"local_inference": True, "colab": False, "training": False,
                    "listening": True, "publication": False},
         "groups": {k: f"{k}-a" for k in corpus_v1.GROUP_KEYS},
         "clean": {"path": "m-clip.clean.wav"},
         "degraded": {"path": "m-clip.degraded.wav"},
         "synthesis": {"generator": corpus_synth.GENERATOR, "seed": 7,
                       "rate": 44100, "seconds": 2, "operator": "hard_clip",
                       "bits": 16, "threshold_dbfs": -4.0}},
        {"id": "m-clean", "kind": "paired_synthetic", "profile": "clean_control",
         "rate": 48000, "channels": 2, "mode": "auto", "split": "control",
         "rights": {"local_inference": True, "colab": False, "training": False,
                    "listening": True, "publication": False},
         "groups": {k: f"{k}-b" for k in corpus_v1.GROUP_KEYS},
         "clean": {"path": "m-clean.clean.wav"},
         "degraded": {"path": "m-clean.degraded.wav"},
         "synthesis": {"generator": corpus_synth.GENERATOR, "seed": 8,
                       "rate": 48000, "seconds": 2, "operator": "clean",
                       "bits": 16}},
    ]
    files = []
    for case in cases:
        specs = corpus_synth.synthesize_case(case, audio)
        for role in ("clean", "degraded", "mask"):
            case.setdefault(role, {}).update(specs[role])
            files.append(specs[role])
    doc = {"schema": corpus_v1.SCHEMA, "tier": "dev",
           "dataset_version": "test-1",
           "toolchain": corpus_v1.actual_toolchain(),
           "rng": {"algorithm": "numpy-pcg64", "seed": 1},
           "numeric_tolerance": 1e-6, "files": files, "cases": cases}
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    return doc


def run_cli(manifest, audio_root, out, *extra):
    return subprocess.run(
        [sys.executable, str(RUNNER), "--manifest", str(manifest),
         "--audio-root", str(audio_root), "--out", str(out), *extra],
        capture_output=True, text=True)


def test_validator_rejects_hash_mismatch(tmp_path):
    doc = mini_manifest(tmp_path)
    target = tmp_path / "audio" / doc["cases"][0]["degraded"]["path"]
    target.write_bytes(b"corrupted")
    errors = corpus_v1.validate_manifest(doc, check_hashes_root=tmp_path / "audio")
    assert any("hash mismatch" in e for e in errors)
    proc = run_cli(tmp_path / "manifest.json", tmp_path / "audio", tmp_path / "out")
    assert proc.returncode == 2 and "hash mismatch" in proc.stderr
    assert not (tmp_path / "out" / "results.json").exists()


def test_validator_rejects_disallowed_operation(tmp_path):
    doc = mini_manifest(tmp_path)
    proc = run_cli(tmp_path / "manifest.json", tmp_path / "audio",
                   tmp_path / "out", "--operation", "colab")
    assert proc.returncode == 2 and "not permitted" in proc.stderr


def test_group_leakage_across_splits_is_rejected(tmp_path):
    doc = mini_manifest(tmp_path)
    doc["cases"][1]["groups"]["take"] = "take-a"  # same take in eval and control
    errors = corpus_v1.validate_manifest(doc)
    assert any("appears in both" in e for e in errors)


def test_cross_manifest_group_leakage_is_rejected(tmp_path):
    doc = mini_manifest(tmp_path)
    sealed = corpus_v1.load(ROOT / "corpus" / "sealed-v1.json")
    dev = corpus_v1.load(ROOT / "corpus" / "dev-v1.json")
    assert corpus_v1.validate_cross_manifest_leakage([doc, dev, sealed]) == []
    sealed["cases"][0]["groups"]["take"] = "take-a"  # collides with mini dev case
    errors = corpus_v1.validate_cross_manifest_leakage([doc, sealed])
    assert any("shared between" in e for e in errors)


def test_rights_values_must_be_bool(tmp_path):
    doc = mini_manifest(tmp_path)
    doc["cases"][0]["rights"]["colab"] = "false"  # string, not bool
    errors = corpus_v1.validate_manifest(doc)
    assert any("rights.colab must be a bool" in e for e in errors)


def test_file_spec_format_is_checked(tmp_path):
    doc = mini_manifest(tmp_path)
    doc["cases"][0]["degraded"]["sha256"] = "ZZZ-not-hex"
    errors = corpus_v1.validate_manifest(doc)
    assert any("64 lowercase hex" in e for e in errors)
    doc = mini_manifest(tmp_path)
    doc["cases"][0]["degraded"]["bytes"] = -1
    assert any("positive int" in e for e in corpus_v1.validate_manifest(doc))


def test_toolchain_placeholders_are_rejected(tmp_path):
    doc = mini_manifest(tmp_path)
    doc["toolchain"] = {"python": ">=3.10", "ffmpeg": "pinned-by-environment"}
    errors = corpus_v1.validate_manifest(doc)
    assert any("toolchain.libflac" in e for e in errors)
    assert any("toolchain.python" in e for e in errors)
    assert any("toolchain.ffmpeg" in e for e in errors)


def test_mask_rle_rejects_out_of_bounds():
    import pytest as _pytest
    blob = json.dumps({"frames": 10, "channels": 2,
                       "runs": [[18, 5]]}).encode()
    with _pytest.raises(ValueError, match="out of bounds"):
        corpus_synth.decode_mask(blob)


def test_sealed_cases_never_enter_dev_report(tmp_path):
    doc = mini_manifest(tmp_path)
    held = {"path": "sealed/x.flac", "sha256": None, "bytes": None,
            "held_by": "custodian"}
    doc["cases"].append({
        "id": "m-sealed", "kind": "paired_file", "sealed": True,
        "profile": "real", "rate": 44100, "channels": 2, "split": "eval",
        "rights": {k: False for k in corpus_v1.OPERATIONS},
        "groups": {k: f"{k}-s" for k in corpus_v1.GROUP_KEYS},
        "clean": dict(held), "degraded": dict(held)})
    doc["files"].append(dict(held))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(doc))
    proc = run_cli(manifest, tmp_path / "audio", tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    report = json.loads((tmp_path / "out" / "results.json").read_text())
    assert report["excluded_sealed_count"] == 1
    assert "m-sealed" not in proc.stdout
    assert all(r["id"] != "m-sealed" for r in report["cases"])
    assert not (tmp_path / "out" / "results_sealed.json").exists()


def test_paired_file_without_mask_reports_null_mask_metrics(tmp_path):
    doc = corpus_v1.load(ROOT / "corpus" / "smoke-v1.json")
    case = doc["cases"][0]
    audio = tmp_path / "audio"
    corpus_synth.synthesize_case(case, audio)
    case = dict(case)
    del case["mask"]
    case["kind"] = "paired_file"
    doc = {**doc, "files": [f for f in doc["files"]
                            if not f["path"].endswith(".mask")],
           "cases": [case]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(doc))
    proc = run_cli(manifest, audio, tmp_path / "out", "--methods", "identity")
    assert proc.returncode == 0, proc.stderr
    rows = json.loads((tmp_path / "out" / "results.json").read_text())["cases"]
    assert rows[0]["methods"]["identity"]["masked_fixed_gain_sdr_db"] is None


def test_sealed_tier_blocked_until_pilot_registered():
    doc = corpus_v1.load(ROOT / "corpus" / "sealed-v1.json")
    errors = corpus_v1.validate_manifest(doc)
    assert any("sealed tier blocked" in e for e in errors)
    pilot = {"sample_size": 40, "gate_margin_sdr_db": 1.0,
             "gate_margin_vs_external_db": 0.5, "min_coverage": .9,
             "cpu_p95_s_per_audio_min": 30, "rss_mb_p95": 2048,
             "registered_at": "2026-11-23"}
    doc["pilot"] = pilot
    doc["rng"] = {"algorithm": "numpy-pcg64", "seed": 5}
    assert corpus_v1.validate_manifest(doc) == []


def test_sealed_manifest_refuses_plain_run():
    proc = run_cli(ROOT / "corpus" / "sealed-v1.json", ROOT / "corpus" / "audio",
                   "/tmp/never-sealed-out")
    assert proc.returncode == 2


def test_seeded_synthesis_is_byte_reproducible(tmp_path):
    case = {"clean": {"path": "a.clean.wav"}, "degraded": {"path": "a.degraded.wav"},
            "synthesis": {"generator": corpus_synth.GENERATOR, "seed": 42,
                          "rate": 48000, "seconds": 1, "operator": "hard_clip",
                          "bits": 16, "threshold_dbfs": -3.0}}
    a = corpus_synth.synthesize_case(case, tmp_path / "x")
    b = corpus_synth.synthesize_case(case, tmp_path / "y")
    assert a == b and a["degraded"]["sha256"] == b["degraded"]["sha256"]


def test_two_clean_runs_reproduce_ids_hashes_statuses(tmp_path):
    doc = mini_manifest(tmp_path)
    manifest = tmp_path / "manifest.json"
    runs = []
    for name in ("env1", "env2"):
        out = tmp_path / name
        proc = run_cli(manifest, tmp_path / "audio", out)
        assert proc.returncode == 0, proc.stderr
        runs.append(json.loads((out / "results.json").read_text())["cases"])
        assert json.loads((out / "protocol.json").read_text())["manifest_sha256"] \
            == corpus_v1.manifest_digest(doc)
    diffs = run_benchmark.compare_reports(runs[0], runs[1], 1e-6)
    assert diffs == []
    rows = {r["id"]: r for r in runs[0]}
    assert rows["m-clip"]["methods"]["identity"]["fixed_gain_sdr_db"] is not None
    assert rows["m-clean"]["methods"]["identity"]["masked_fixed_gain_sdr_db"] is None


def test_committed_smoke_manifest_runs_via_cli(tmp_path):
    """Real smoke corpus: validate pins, run the CLI, get comparable rows."""
    audio = tmp_path / "audio" / "smoke"
    doc = corpus_v1.load(ROOT / "corpus" / "smoke-v1.json")
    for case in doc["cases"]:
        specs = corpus_synth.synthesize_case(case, audio)
        for role in ("clean", "degraded", "mask"):
            assert specs[role]["sha256"] == case[role]["sha256"], \
                f"{case['id']}: regenerated {role} drifted from pinned hash"
    assert corpus_v1.validate_manifest(doc, check_hashes_root=audio) == []
    proc = run_cli(ROOT / "corpus" / "smoke-v1.json", audio, tmp_path / "out",
                   "--methods", "identity,gain_only")
    assert proc.returncode == 0, proc.stderr
    rows = json.loads((tmp_path / "out" / "results.json").read_text())
    assert {c["id"] for c in doc["cases"]} == {r["id"] for r in rows["cases"]}
    profiles = {c["profile"] for c in doc["cases"]}
    for required in ("clean_control", "intentional_control",
                     "rare_asymmetric_sub_full_scale", "dithered_clip",
                     "quiet_tail", "defect_after_90s"):
        assert required in profiles
    assert {c["rate"] for c in doc["cases"]} == {44100, 48000, 96000}
