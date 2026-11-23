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
import os
import re
import stat
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
DATASET_VERSION_RE = re.compile(
    r"^[a-z0-9][a-z0-9_-]*-(?:v\d+|\d{4}-\d{2}-\d{2})(?:-[0-9a-f]{8,64})?$"
)
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

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

    def normalized_sha256(self) -> str:
        return self.content_sha256.strip().lower()

    def to_canonical_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["content_sha256"] = self.normalized_sha256()
        return d


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_sha256_json(obj: Any) -> str:
    """Deterministic SHA-256 over canonical JSON."""
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def assert_not_sealed_source(path_or_name: str | Path, stage: str) -> None:
    """Fail closed if a sealed corpus path, any symlink component, or unresolvable path is referenced outside `sealed_test`."""
    if stage == "sealed_test":
        return
    raw = str(path_or_name)
    candidates = [raw, Path(raw).name]
    try:
        abs_path = Path(os.path.abspath(raw))
        prefix = Path(abs_path.anchor) if abs_path.anchor else Path(".")
        for part in abs_path.parts[1:] if abs_path.anchor else abs_path.parts:
            prefix = prefix / part
            try:
                st = os.lstat(prefix)
            except FileNotFoundError:
                break
            if stat.S_ISLNK(st.st_mode):
                raise GovernanceError(
                    f"Fail-closed: symlink component '{prefix}' in source_manifest '{raw}' "
                    f"is forbidden in stage '{stage}'"
                )
        real = os.path.realpath(raw)
        resolved = str(Path(raw).resolve(strict=False))
        candidates.extend([real, Path(real).name, resolved, Path(resolved).name])
    except GovernanceError:
        raise
    except OSError as exc:
        raise GovernanceError(
            f"Fail-closed: unable to resolve source_manifest '{raw}' in stage '{stage}': {exc}"
        ) from exc

    for cand in candidates:
        if any(sealed in cand or Path(cand).name == sealed for sealed in SEALED_CORPUS_NAMES):
            raise GovernanceError(
                f"Sealed corpus reference '{raw}' (resolved '{cand}') is forbidden in stage '{stage}'"
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

    Fail-closed cross-stage leakage checks (`recording_id`, `group_id`, and
    case-normalized `content_sha256`) run on the FULL input set BEFORE any
    within-stage duplicate is dropped, so a dropped duplicate can never hide
    a cross-stage leak. Within the same stage, the lexicographically smallest
    `record_id` is kept and dropped duplicates are explicitly reported.
    """
    for rec in records:
        validate_record(rec)

    recording_stages: dict[str, set[str]] = {}
    group_stages: dict[str, set[str]] = {}
    hash_stages: dict[str, set[str]] = {}
    for rec in records:
        recording_stages.setdefault(rec.recording_id, set()).add(rec.stage)
        group_stages.setdefault(rec.group_id, set()).add(rec.stage)
        hash_stages.setdefault(rec.normalized_sha256(), set()).add(rec.stage)

    leaks: list[str] = []
    for rec_id, stages in sorted(recording_stages.items()):
        if len(stages) > 1:
            leaks.append(f"recording_id '{rec_id}' leaks across stages {sorted(stages)}")
    for grp_id, stages in sorted(group_stages.items()):
        if len(stages) > 1:
            leaks.append(f"group_id '{grp_id}' leaks across stages {sorted(stages)}")
    for digest, stages in sorted(hash_stages.items()):
        if len(stages) > 1:
            leaks.append(f"content_sha256 '{digest}' leaks across stages {sorted(stages)}")
    if leaks:
        raise GovernanceError(
            "Cross-stage leakage detected prior to deduplication: " + "; ".join(leaks)
        )

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
        hash_stages.setdefault(rec.normalized_sha256(), set()).add(rec.stage)
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
    if not model_id.strip() or not isinstance(model_sha256, str):
        raise GovernanceError("model_id must be non-empty and model_sha256 must be a 64-hex string")
    norm_model_sha = model_sha256.strip().lower()
    if not SHA256_HEX_RE.fullmatch(norm_model_sha):
        raise GovernanceError(
            f"model_sha256 must be 64 lowercase hex characters, got '{model_sha256}'"
        )

    summary = validate_split_boundaries(records)
    if summary["stage_counts"].get(stage, 0) <= 0:
        raise GovernanceError(
            f"Cannot build '{stage}' result with 0 records in stage '{stage}'"
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "dataset_version": summary["dataset_version"],
        "stage_counts": summary["stage_counts"],
        "model_id": model_id,
        "model_sha256": norm_model_sha,
        "git_commit": git_commit,
        "preserve_isolated": True,
        "sealed_audio_opened": bool(sealed_audio_opened),
        "metrics": metrics,
    }
    payload["results_sha256"] = compute_sha256_json(payload)
    return payload


def verify_versioned_result(payload: dict[str, Any]) -> bool:
    """Verify schema_version, dataset_version, model_sha256, sealed flags, and results_sha256 integrity."""
    required = (
        "schema_version",
        "stage",
        "dataset_version",
        "stage_counts",
        "model_id",
        "model_sha256",
        "preserve_isolated",
        "sealed_audio_opened",
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
    stage = payload["stage"]
    if stage not in STAGE_ORDER:
        raise GovernanceError(f"Invalid stage '{stage}'")

    dataset_version = payload["dataset_version"]
    if not isinstance(dataset_version, str) or not DATASET_VERSION_RE.fullmatch(dataset_version):
        raise GovernanceError(
            f"Invalid dataset_version '{dataset_version}'; expected versioned identifier"
        )

    model_sha = payload["model_sha256"]
    if not isinstance(model_sha, str) or not SHA256_HEX_RE.fullmatch(model_sha):
        raise GovernanceError(
            f"Invalid model_sha256 '{model_sha}'; expected 64 lowercase hex characters"
        )

    if not isinstance(payload["sealed_audio_opened"], bool):
        raise GovernanceError("sealed_audio_opened must be a boolean")
    if stage != "sealed_test" and payload["sealed_audio_opened"] is True:
        raise GovernanceError(
            f"sealed_audio_opened=True is forbidden for non-sealed stage '{stage}'"
        )

    stage_counts = payload["stage_counts"]
    if not isinstance(stage_counts, dict) or int(stage_counts.get(stage, 0)) <= 0:
        raise GovernanceError(
            f"stage_counts['{stage}'] must be > 0 for stage '{stage}'"
        )

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


def _module_name_for_path(py_path: Path, repo_root: Path) -> str:
    try:
        rel = py_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return py_path.stem
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _extract_imports(
    py_path: Path, repo_root: Path = REPO_ROOT, *, top_level_only: bool = False
) -> list[dict[str, Any]]:
    """Extract all import statements from a Python file via AST.

    Expands `from pkg import name` to emit both `pkg` and `pkg.name` (so
    `from rnext import ml` is caught as `rnext.ml`), and resolves relative
    imports against the file's package path.
    """
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_path))

    parent_map: dict[ast.AST, ast.AST] = {}
    if top_level_only:
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

    def _is_top_level(node: ast.AST) -> bool:
        cur = parent_map.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return False
            cur = parent_map.get(cur)
        return True

    pkg_parts = _module_name_for_path(py_path, repo_root).split(".")
    if py_path.name != "__init__.py" and pkg_parts:
        pkg_parts = pkg_parts[:-1]

    imports: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if top_level_only and not _is_top_level(node):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"module": alias.name, "lineno": node.lineno})
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level > 0:
                up = node.level - 1
                anchor = pkg_parts[: len(pkg_parts) - up] if up <= len(pkg_parts) else []
                base = ".".join([*anchor, base]) if base else ".".join(anchor)
            if base:
                imports.append({"module": base, "lineno": node.lineno})
            for alias in node.names:
                if alias.name == "*":
                    continue
                full = f"{base}.{alias.name}" if base else alias.name
                if full:
                    imports.append({"module": full, "lineno": node.lineno})
    return imports


def _matches_prefix(module_name: str, forbidden: tuple[str, ...]) -> str | None:
    for prefix in forbidden:
        if module_name == prefix or module_name.startswith(prefix + "."):
            return prefix
    return None


def _resolve_local_module_file(module_name: str, repo_root: Path) -> Path | None:
    """Resolve a local repo module name (under rnext/ or scripts/) to its .py file."""
    parts = module_name.split(".")
    if not parts or parts[0] not in ("rnext", "scripts"):
        return None
    cand_file = repo_root.joinpath(*parts).with_suffix(".py")
    if cand_file.is_file():
        return cand_file
    cand_pkg = repo_root.joinpath(*parts, "__init__.py")
    if cand_pkg.is_file():
        return cand_pkg
    if len(parts) > 1:
        parent_file = repo_root.joinpath(*parts[:-1]).with_suffix(".py")
        if parent_file.is_file():
            return parent_file
    return None


def audit_ml_preserve_boundary(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Bidirectional AST audit ensuring Preserve (incl. transitive local closure) and ML stay isolated."""
    preserve_violations: list[dict[str, Any]] = []
    ml_violations: list[dict[str, Any]] = []

    visited_preserve: set[str] = set()
    queue: list[str] = list(PRESERVE_CORE_MODULES)

    while queue:
        rel = queue.pop(0)
        if rel in visited_preserve:
            continue
        visited_preserve.add(rel)

        p = repo_root / rel
        if not p.is_file():
            preserve_violations.append({"file": rel, "error": "missing_module"})
            continue

        # Python always executes every parent package's __init__.py when importing
        # a submodule, so enqueue all parent __init__.py files into the closure.
        parent = p.parent
        while parent != repo_root and parent.is_relative_to(repo_root):
            init_py = parent / "__init__.py"
            if init_py.is_file():
                init_rel = str(init_py.relative_to(repo_root))
                if init_rel not in visited_preserve:
                    queue.append(init_rel)
            parent = parent.parent

        # scripts/restore_one.py is imported by scripts/restore_safe.py solely for
        # `encode_flac`; check its top-level imports here (its lazy opt-in ML import
        # inside process_track is audited separately). All other Preserve modules
        # are checked at every scope (top-level and function-level).
        top_only = rel == "scripts/restore_one.py"
        for imp in _extract_imports(p, repo_root, top_level_only=top_only):
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
                continue
            local_target = _resolve_local_module_file(imp["module"], repo_root)
            if local_target is not None:
                target_rel = str(local_target.relative_to(repo_root))
                if target_rel not in visited_preserve:
                    queue.append(target_rel)

    for rel in ML_CONTOUR_MODULES:
        p = repo_root / rel
        if not p.is_file():
            ml_violations.append({"file": rel, "error": "missing_module"})
            continue
        for imp in _extract_imports(p, repo_root):
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
        "preserve_modules_checked": len(visited_preserve),
        "preserve_transitive_modules": sorted(visited_preserve),
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
