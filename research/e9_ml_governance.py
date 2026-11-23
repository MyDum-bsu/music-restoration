#!/usr/bin/env python3
"""E9 ML-target governance, recording deduplication, and split-boundary contract.

Enforces fail-closed governance for future ML research contours (target
estimators, neural prototypes) while keeping them strictly isolated from the
local CPU `Preserve` release path (`rnext/dsp/*`, `scripts/restore_safe.py`).

Stages (strictly disjoint by recording_id, group_id, and content_sha256):
  1. `approved_labels` — verified training/calibration records only.
  2. `dev_selection`   — model/hyperparameter selection and dev bake-off.
  3. `promotion`       — held-out promotion gate verification before sealed.
  4. `sealed_test`     — custodian-held final test split (never opened in dev).

CLI usage:
  python3 research/e9_ml_governance.py --self-check
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "e9-ml-governance/v1"
ALLOWED_STAGES = (
    "approved_labels",
    "dev_selection",
    "promotion",
    "sealed_test",
)
STAGE_ORDER = {stage: idx for idx, stage in enumerate(ALLOWED_STAGES)}

ALLOWED_LABEL_STATUSES = ("approved", "provisional", "unlabeled", "rejected")
SEALED_CORPUS_NAMES = ("sealed-v1.json", "sealed-e5-v1.json")

# Modules that belong strictly to the Preserve local CPU delivery contour.
PRESERVE_CORE_MODULES = (
    "rnext/features.py",
    "rnext/dsp/clip_detector.py",
    "rnext/dsp/contour.py",
    "rnext/dsp/declip.py",
    "rnext/dsp/declip_methods.py",
    "rnext/dsp/eq.py",
    "rnext/dsp/limit.py",
    "rnext/dsp/pipeline.py",
    "rnext/dsp/predictive.py",
    "rnext/dsp/repair.py",
    "rnext/dsp/selection.py",
    "rnext/dsp/sparse.py",
    "rnext/dsp/stereo_safety.py",
    "scripts/restore_safe.py",
    "scripts/audio_measure.py",
    "scripts/evaluate_album.py",
    "scripts/qc_gates.py",
    "scripts/finish_release.py",
)

# Preserve-side forbidden imports (no ML/research/lab dependencies).
FORBIDDEN_IN_PRESERVE = (
    "joblib",
    "torch",
    "sklearn",
    "scikit_learn",
    "rnext.ml",
    "research",
    "lab",
)

# ML-contour modules that must never import Preserve declip/delivery entrypoints.
ML_CONTOUR_MODULES = (
    "rnext/ml/__init__.py",
    "rnext/ml/dataset.py",
    "rnext/ml/metrics.py",
    "rnext/ml/models.py",
    "rnext/ml/registry.py",
    "rnext/ml/runtime.py",
    "research/ml_auto.py",
    "research/ml_train.py",
    "research/e9_ml_governance.py",
)

FORBIDDEN_IN_ML_CONTOUR = (
    "scripts.restore_safe",
    "rnext.dsp.repair",
    "rnext.dsp.declip_methods",
    "rnext.dsp.clip_detector",
    "rnext.dsp.stereo_safety",
    "rnext.dsp.selection",
)


class GovernanceError(ValueError):
    """Fail-closed violation of ML split, deduplication, versioning, or boundary rules."""


@dataclass(frozen=True)
class MLRecord:
    """One track/recording entry in a governed ML dataset manifest."""

    record_id: str
    recording_id: str
    group_id: str
    stage: str
    label_status: str
    content_sha256: str
    source_manifest: str = ""
    target_name: str = "dcrest"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_canonical_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_sha256_json(obj: Any) -> str:
    """Deterministic SHA-256 over canonical JSON."""
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def assert_not_sealed_source(path_or_name: str | Path, stage: str) -> None:
    """Fail closed if a sealed corpus path is referenced outside `sealed_test`."""
    raw = str(path_or_name)
    name = Path(raw).name
    if (
        stage != "sealed_test"
        and any(sealed in raw or name == sealed for sealed in SEALED_CORPUS_NAMES)
    ):
        raise GovernanceError(
            f"Sealed corpus reference '{raw}' is forbidden in stage '{stage}'"
        )


def validate_record(record: MLRecord) -> None:
    """Validate required fields and per-stage label rules for one record."""
    for field_name in (
        "record_id",
        "recording_id",
        "group_id",
        "stage",
        "label_status",
        "content_sha256",
    ):
        val = getattr(record, field_name)
        if not isinstance(val, str) or not val.strip():
            raise GovernanceError(
                f"Record '{record.record_id}' missing non-empty '{field_name}'"
            )

    if record.stage not in STAGE_ORDER:
        raise GovernanceError(
            f"Record '{record.record_id}' has invalid stage '{record.stage}'; "
            f"expected one of {ALLOWED_STAGES}"
        )

    if record.label_status not in ALLOWED_LABEL_STATUSES:
        raise GovernanceError(
            f"Record '{record.record_id}' has invalid label_status '{record.label_status}'; "
            f"expected one of {ALLOWED_LABEL_STATUSES}"
        )

    if record.stage == "approved_labels" and record.label_status != "approved":
        raise GovernanceError(
            f"Record '{record.record_id}' in stage 'approved_labels' must have "
            f"label_status='approved', got '{record.label_status}'"
        )

    if record.source_manifest:
        assert_not_sealed_source(record.source_manifest, record.stage)


def deduplicate_by_recording(
    records: list[MLRecord],
) -> tuple[list[MLRecord], list[dict[str, str]]]:
    """Deterministically deduplicate records by `recording_id` within each stage.

    Cross-stage duplicates are NOT silently dropped — they raise GovernanceError
    because a recording appearing in two stages is a split-boundary violation.
    Within the same stage, the lexicographically smallest `record_id` is kept
    and dropped duplicates are explicitly reported.
    """
    for rec in records:
        validate_record(rec)

    stage_by_recording: dict[str, str] = {}
    for rec in records:
        prev_stage = stage_by_recording.get(rec.recording_id)
        if prev_stage is not None and prev_stage != rec.stage:
            raise GovernanceError(
                f"Cross-stage recording_id leakage for '{rec.recording_id}': "
                f"present in both '{prev_stage}' and '{rec.stage}'"
            )
        stage_by_recording[rec.recording_id] = rec.stage

    ordered = sorted(records, key=lambda r: (STAGE_ORDER[r.stage], r.recording_id, r.record_id))
    kept: list[MLRecord] = []
    dropped: list[dict[str, str]] = []
    seen_recording: dict[str, str] = {}

    for rec in ordered:
        if rec.recording_id in seen_recording:
            dropped.append(
                {
                    "dropped_record_id": rec.record_id,
                    "kept_record_id": seen_recording[rec.recording_id],
                    "recording_id": rec.recording_id,
                    "stage": rec.stage,
                    "reason": "duplicate_recording_id_in_stage",
                }
            )
            continue
        seen_recording[rec.recording_id] = rec.record_id
        kept.append(rec)

    return kept, dropped


def compute_dataset_version(records: list[MLRecord]) -> str:
    """Compute a deterministic content-addressed dataset_version string."""
    canonical_rows = sorted(
        (rec.to_canonical_dict() for rec in records),
        key=lambda d: (STAGE_ORDER[d["stage"]], d["recording_id"], d["record_id"]),
    )
    digest = compute_sha256_json(canonical_rows)
    return f"ml-ds-v1-{digest[:16]}"


def validate_split_boundaries(
    records: list[MLRecord],
    *,
    require_all_stages: bool = False,
) -> dict[str, Any]:
    """Fail-closed validation of stage separation, deduplication, and provenance.

    Checks:
      - Every record has valid stage, label_status, and non-empty IDs/hashes.
      - Unique `record_id` across the entire manifest.
      - Unique `recording_id` (no un-deduplicated duplicates within or across stages).
      - Zero overlap of `recording_id`, `group_id`, or `content_sha256` across any
        pair of stages (`approved_labels`, `dev_selection`, `promotion`, `sealed_test`).
      - Sealed corpus sources never appear outside `sealed_test`.
    """
    if not records:
        raise GovernanceError("Manifest contains zero records")

    seen_record_ids: set[str] = set()
    seen_recordings_in_stage: dict[tuple[str, str], str] = {}

    recording_stages: dict[str, set[str]] = {}
    group_stages: dict[str, set[str]] = {}
    hash_stages: dict[str, set[str]] = {}
    stage_counts: dict[str, int] = {stage: 0 for stage in ALLOWED_STAGES}

    for rec in records:
        validate_record(rec)

        if rec.record_id in seen_record_ids:
            raise GovernanceError(f"Duplicate record_id '{rec.record_id}'")
        seen_record_ids.add(rec.record_id)

        rec_key = (rec.stage, rec.recording_id)
        if rec_key in seen_recordings_in_stage:
            raise GovernanceError(
                f"Duplicate recording_id '{rec.recording_id}' within stage '{rec.stage}' "
                f"(records '{seen_recordings_in_stage[rec_key]}' and '{rec.record_id}'); "
                "run deduplicate_by_recording() before freezing"
            )
        seen_recordings_in_stage[rec_key] = rec.record_id

        recording_stages.setdefault(rec.recording_id, set()).add(rec.stage)
        group_stages.setdefault(rec.group_id, set()).add(rec.stage)
        hash_stages.setdefault(rec.content_sha256, set()).add(rec.stage)
        stage_counts[rec.stage] += 1

    violations: list[str] = []
    for rec_id, stages in sorted(recording_stages.items()):
        if len(stages) > 1:
            violations.append(
                f"recording_id '{rec_id}' leaks across stages {sorted(stages)}"
            )
    for grp_id, stages in sorted(group_stages.items()):
        if len(stages) > 1:
            violations.append(
                f"group_id '{grp_id}' leaks across stages {sorted(stages)}"
            )
    for digest, stages in sorted(hash_stages.items()):
        if len(stages) > 1:
            violations.append(
                f"content_sha256 '{digest}' leaks across stages {sorted(stages)}"
            )

    if require_all_stages:
        missing = [s for s in ALLOWED_STAGES if stage_counts[s] == 0]
        if missing:
            violations.append(f"missing required stages: {missing}")

    if violations:
        raise GovernanceError(
            "Split-boundary validation failed: " + "; ".join(violations)
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": compute_dataset_version(records),
        "total_records": len(records),
        "unique_recordings": len(recording_stages),
        "unique_groups": len(group_stages),
        "stage_counts": stage_counts,
        "cross_stage_leakage_errors": [],
        "valid": True,
    }


def build_versioned_result(
    *,
    stage: str,
    records: list[MLRecord],
    metrics: dict[str, Any],
    model_id: str,
    model_sha256: str,
    git_commit: str = "unknown",
    sealed_audio_opened: bool = False,
) -> dict[str, Any]:
    """Create a tamper-evident versioned ML evaluation result payload."""
    if stage not in STAGE_ORDER:
        raise GovernanceError(f"Invalid result stage '{stage}'")
    if stage != "sealed_test" and sealed_audio_opened:
        raise GovernanceError(
            f"sealed_audio_opened=True is forbidden for stage '{stage}'"
        )
    if not model_id.strip() or not model_sha256.strip():
        raise GovernanceError("model_id and model_sha256 must be non-empty")

    summary = validate_split_boundaries(records)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "dataset_version": summary["dataset_version"],
        "stage_counts": summary["stage_counts"],
        "model_id": model_id,
        "model_sha256": model_sha256,
        "git_commit": git_commit,
        "preserve_isolated": True,
        "sealed_audio_opened": bool(sealed_audio_opened),
        "metrics": metrics,
    }
    payload["results_sha256"] = compute_sha256_json(payload)
    return payload


def verify_versioned_result(payload: dict[str, Any]) -> bool:
    """Verify schema_version, dataset_version, and results_sha256 integrity."""
    required = (
        "schema_version",
        "stage",
        "dataset_version",
        "model_id",
        "model_sha256",
        "preserve_isolated",
        "metrics",
        "results_sha256",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        raise GovernanceError(f"Versioned result missing required fields: {missing}")

    if payload["schema_version"] != SCHEMA_VERSION:
        raise GovernanceError(
            f"Unsupported schema_version '{payload['schema_version']}' "
            f"(expected '{SCHEMA_VERSION}')"
        )
    if payload["stage"] not in STAGE_ORDER:
        raise GovernanceError(f"Invalid stage '{payload['stage']}'")
    if payload["preserve_isolated"] is not True:
        raise GovernanceError("preserve_isolated must be True for ML results")

    recorded = str(payload["results_sha256"])
    body = {k: v for k, v in payload.items() if k != "results_sha256"}
    expected = compute_sha256_json(body)
    if recorded != expected:
        raise GovernanceError(
            f"results_sha256 mismatch: recorded={recorded}, expected={expected}"
        )
    return True


def _extract_imports(py_path: Path) -> list[dict[str, Any]]:
    """Extract all import statements from a Python file via AST."""
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_path))
    imports: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"module": alias.name, "lineno": node.lineno})
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append({"module": node.module, "lineno": node.lineno})
    return imports


def _matches_prefix(module_name: str, forbidden: tuple[str, ...]) -> str | None:
    for prefix in forbidden:
        if module_name == prefix or module_name.startswith(prefix + "."):
            return prefix
    return None


def audit_ml_preserve_boundary(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Bidirectional AST audit ensuring Preserve and ML contours stay isolated."""
    preserve_violations: list[dict[str, Any]] = []
    ml_violations: list[dict[str, Any]] = []

    for rel in PRESERVE_CORE_MODULES:
        p = repo_root / rel
        if not p.is_file():
            preserve_violations.append({"file": rel, "error": "missing_module"})
            continue
        for imp in _extract_imports(p):
            hit = _matches_prefix(imp["module"], FORBIDDEN_IN_PRESERVE)
            if hit is not None:
                preserve_violations.append(
                    {
                        "file": rel,
                        "lineno": imp["lineno"],
                        "import": imp["module"],
                        "forbidden_prefix": hit,
                    }
                )

    for rel in ML_CONTOUR_MODULES:
        p = repo_root / rel
        if not p.is_file():
            ml_violations.append({"file": rel, "error": "missing_module"})
            continue
        for imp in _extract_imports(p):
            hit = _matches_prefix(imp["module"], FORBIDDEN_IN_ML_CONTOUR)
            if hit is not None:
                ml_violations.append(
                    {
                        "file": rel,
                        "lineno": imp["lineno"],
                        "import": imp["module"],
                        "forbidden_prefix": hit,
                    }
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "preserve_modules_checked": len(PRESERVE_CORE_MODULES),
        "ml_modules_checked": len(ML_CONTOUR_MODULES),
        "preserve_to_ml_violations": preserve_violations,
        "ml_to_preserve_violations": ml_violations,
        "bidirectional_isolation_passed": (
            len(preserve_violations) == 0 and len(ml_violations) == 0
        ),
    }


def _run_self_check() -> dict[str, Any]:
    sample_records = [
        MLRecord("rec-001", "take-a", "artist-a", "approved_labels", "approved", "a" * 64),
        MLRecord("rec-002", "take-b", "artist-b", "dev_selection", "approved", "b" * 64),
        MLRecord("rec-003", "take-c", "artist-c", "promotion", "approved", "c" * 64),
        MLRecord("rec-004", "take-d", "artist-d", "sealed_test", "approved", "d" * 64),
    ]
    split_summary = validate_split_boundaries(sample_records, require_all_stages=True)
    result = build_versioned_result(
        stage="dev_selection",
        records=sample_records,
        metrics={"median_abs_err_db": 0.42},
        model_id="self-check-model",
        model_sha256="0" * 64,
        git_commit="self-check",
    )
    verify_versioned_result(result)
    boundary = audit_ml_preserve_boundary()
    return {
        "split_summary": split_summary,
        "versioned_result": result,
        "boundary": boundary,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="Run governance contract and bidirectional isolation self-check.",
    )
    args = ap.parse_args(argv)
    if args.self_check:
        out = _run_self_check()
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0 if out["boundary"]["bidirectional_isolation_passed"] else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
