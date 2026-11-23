# E9 · Legacy & ML Research Isolation Report

**Card:** Board #1137 · **Spec:** E0 charter + spec v1 (Section E9) · **Schema:** `e9-isolation-audit/v1`

## 1. Executive Summary & Safety Boundaries

- **Sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`):** `sealed_corpora_opened = False` (strictly untouched).
- **Frozen E4 Preserve files:** `18/18` SHA-256 pins verified untouched (`all_untouched = True`).
- **Static AST import audit (Preserve ↔ ML bidirectional):** `static_audit_passed = True` (`preserve_modules_checked = 23`, `ml_modules_checked = 9`, `preserve_to_ml_violations = 0`, `ml_to_preserve_violations = 0`).
- **Isolated Preserve E2E smoke (ML libs + network hard-blocked):** `e2e_smoke_passed = True` (`loaded_forbidden_modules = []`).
- **ML-target governance schema:** `e9-ml-governance/v1` (`stages = approved_labels, dev_selection, promotion, sealed_test`, fail-closed deduplication and split isolation).
- **Production dependencies & GPU policy:** Zero new production dependencies added; no training launched; GPU remains Google Colab-only under separate authorization.
- **Results SHA-256:** `63a28a41d95e738bf63707263386aa2e73b7684dcf27c932722958b3dcd7c9d4`

## 2. Legacy Runners, Dependencies & Migration Inventory

| Component | Category | Status | ML / Research Dependencies | Migration Route |
|---|---|---|---|---|
| `scripts/restore_safe.py` | `preserve_default` | `active_production_default` | none (pure CPU/DSP) | Canonical Preserve target |
| `scripts/restore_one.py` | `legacy_runner` | `legacy_preserved_manual` | rnext.ml (lazy, opt-in --ml-snapshot only) | legacy -> Preserve: python3 scripts/restore_safe.py --src <in> --out <out> |
| `scripts/restore_per_track.py` | `legacy_runner` | `legacy_preserved_comparison` | none (pure CPU/DSP) | legacy -> Preserve: python3 scripts/restore_safe.py --src <in> --out <out> |
| `scripts/release_batch.py` | `legacy_runner` | `deprecated_legacy_wrapper` | transitive via restore_one.py --ml-snapshot | legacy -> Preserve: python3 scripts/restore_safe.py --src <album/> --out <out/> |
| `rnext/ml/*` | `legacy_ml_layer` | `isolated_ml_research_layer` | scikit-learn, joblib | research -> Preserve: governed by research/e9_ml_governance.py; zero auto-promotion |
| `research/lab/server.py` | `research_lab_ui` | `research_only` | rnext.ml (lazy), dataset_manifests/ | research -> Preserve: inspect Preserve runs via report.json + scripts/evaluate_album.py |
| `research/ml_auto.py & research/ml_train.py` | `research_ml_training` | `research_only` | rnext.ml, scikit-learn, joblib, dataset_manifests/ | research -> Preserve: offline evaluation + E9 split governance before any proposal |
| `research/dataset_loop.py & research/dataset_*.py` | `research_dataset_pipeline` | `research_only` | rnext.ml (dataset_eval.py), research/* | research -> Preserve: use versioned corpus/*.json manifests + research/run_benchmark.py |
| `research/e6_declick_hiss.py, e7_bwe_preview.py, e8_*_eval.py` | `research_feasibility` | `research_only_colab_gated` | torch (optional runtime probe only), research/* | research -> Preserve: separate product approval required; no generalization to Preserve |

### 2.1 Caller Summary Across Repository

| Target | Caller Count | Sample Callers (non-Preserve unless noted) |
|---|---:|---|
| `joblib` | 11 | `README.md`, `docs/portability-report.md`, `docs/setup.md`, `docs/validation/e9-legacy-isolation-report.md`, `research/docs/ml.md`, `research/e9_isolation_audit.py` (+5 more) |
| `research/lab/server.py` | 4 | `docs/validation/e9-legacy-isolation-report.md`, `research/e9_isolation_audit.py`, `research/lab/README.md`, `research/lab/server.py` |
| `rnext.ml` | 13 | `README.md`, `docs/validation/e9-legacy-isolation-report.md`, `research/dataset_eval.py`, `research/docs/ml.md`, `research/e9_isolation_audit.py`, `research/e9_ml_governance.py` (+7 more) |
| `scripts/release_batch.py` | 6 | `README.md`, `docs/code-review-2026-11-02.md`, `docs/setup.md`, `docs/validation/e9-legacy-isolation-report.md`, `research/e9_isolation_audit.py`, `scripts/release_batch.py` |
| `scripts/restore_one.py` | 11 | `README.md`, `docs/setup.md`, `docs/validation/e9-legacy-isolation-report.md`, `research/docs/ml.md`, `research/e9_isolation_audit.py`, `research/e9_ml_governance.py` (+5 more) |
| `scripts/restore_per_track.py` | 12 | `README.md`, `docs/eval-vs-masters-run.md`, `docs/experiment/independent-method.md`, `docs/portability-report.md`, `docs/setup.md`, `docs/validation/e9-legacy-isolation-report.md` (+6 more) |
| `torch` | 12 | `README.md`, `docs/validation/e6-declick-hiss-report.md`, `docs/validation/e7-bwe-report.md`, `docs/validation/e9-legacy-isolation-report.md`, `research/docs/ml.md`, `research/e6_declick_hiss.py` (+6 more) |

> **Note on `scripts/restore_safe.py` -> `scripts/restore_one.py`:** `scripts/restore_safe.py` (frozen in E4) imports only the pure numpy/wave/flac helper `encode_flac` from `scripts/restore_one.py`. Static AST inspection proves `scripts/restore_one.py` has **0 top-level** ML/research imports (`rnext.ml` is imported lazily inside `process_track()` only when `--ml-snapshot` is requested), and the isolated E2E subprocess proves `restore_safe.py` completes with `0` ML modules loaded.

## 3. Programmatic Isolation Proof (AST + Offline/No-ML E2E Smoke)

| Check | Result | Evidence |
|---|---|---|
| Preserve static AST audit | `PASS` | 23 Preserve modules, 0 forbidden imports (`joblib`, `torch`, `sklearn`, `rnext.ml`, `research`, `lab`) |
| ML -> Preserve reverse AST audit | `PASS` | 9 ML modules, 0 imports of Preserve declip/delivery entrypoints |
| Isolated E2E clean control FLAC | `PASS` | `status=kept_original`, `byte_identical=True` |
| Isolated E2E hard-clipped FLAC | `PASS` | `status=processed_candidate`, `tp_dbfs=-1.0`, `clip_samples=0` |
| Blocked modules & network guard | `PASS` | `loaded_forbidden=[]`, `socket` blocked in subprocess |

## 4. ML-Target Governance & Split-Boundary Contract

Implemented in `research/e9_ml_governance.py` (`schema_version = e9-ml-governance/v1`):

| Stage | Purpose | Required Label Gate | Leakage Rule |
|---|---|---|---|
| `approved_labels` | Training / calibration source records | `label_status == 'approved'` (fail-closed) | Disjoint `recording_id`, `group_id`, `content_sha256` from all other stages |
| `dev_selection` | Model / hyperparameter selection on dev | Validated provenance, no sealed paths | Zero overlap with `approved_labels`, `promotion`, `sealed_test` |
| `promotion` | Held-out promotion gate verification | Frozen `dataset_version` + `model_sha256` | Zero overlap with `approved_labels`, `dev_selection`, `sealed_test` |
| `sealed_test` | Custodian-held final evaluation | Single-shot per frozen protocol | Never opened in dev; strictly disjoint from all earlier stages |

- **Recording Deduplication:** `deduplicate_by_recording()` enforces one canonical record per `recording_id` within each stage and raises `GovernanceError` immediately if any `recording_id`, `group_id`, or `content_sha256` crosses stage boundaries.
- **Result Versioning:** `build_versioned_result()` pins `schema_version`, deterministic `dataset_version`, `model_sha256`, `preserve_isolated = True`, and tamper-evident `results_sha256`, verified fail-closed by `verify_versioned_result()`.

## 5. Runbook Repairs & Migration Guide

### 5.1 Migration Routes

- **`legacy -> Preserve`:** Replace `scripts/restore_one.py`, `scripts/restore_per_track.py`, and `scripts/release_batch.py` with `python3 scripts/restore_safe.py --src <album/> --out <empty-out/>`, followed by `python3 scripts/evaluate_album.py --src <album/> --restored <out/> --out <eval/>`. Legacy scripts remain intact for historical reproduction and are not deleted.
- **`research -> Preserve`:** Research scripts (`research/ml_auto.py`, `research/ml_train.py`, `research/lab/server.py`, `research/e6_*`–`e8_*`) stay strictly outside `Preserve`. Any future ML candidate must satisfy `research/e9_ml_governance.py`, pass E3/E4/E5 freeze & blind-panel gates, and receive explicit owner/CTO release approval before touching `Preserve`.

### 5.2 Repaired Stale Runbook References

| File | Repair Applied |
|---|---|
| `research/dataset_loop.py` | Replaced 11 stale `scripts/<research_script>.py` subprocess/doc calls with `research/<script>.py`. |
| `research/status_report.py` | Fixed `scripts/dataset_audit.py` subprocess call and stale `scripts/` hints to `research/`. |
| `research/lab/server.py` | Fixed CLI docstring from `python3 lab/server.py` to `python3 research/lab/server.py`. |
| `research/lab/README.md` | Fixed `lab/server.py` -> `research/lab/server.py`, removed host paths, added Preserve migration note. |
| `research/docs/ml.md` | Fixed `scripts/ml_auto.py`, `scripts/ml_train.py`, `scripts/dataset_loop.py` -> `research/...`; documented Preserve isolation and E9 governance. |
| `research/docs/dataset.md` | Fixed `scripts/master_targets.py`, `scripts/dataset_plan.py`, `scripts/dataset_fetch.py` -> `research/...`; removed host path. |
| `research/skills/restore-status/SKILL.md` | Fixed all moved research script paths to `research/...`, removed host paths, added `legacy -> Preserve` migration note. |
| `research/resume.sh` | Fixed `scripts/status.py` -> `research/status.py` and parameterized `MR_DATA_ROOT` (removed hardcoded host mount path). |
| `scripts/release_batch.py` | Updated deprecation docstring to direct new runs to `scripts/restore_safe.py` (Preserve). |
| `scripts/restore_one.py` | Updated `--eq-curve` help text from `scripts/fit_tilt.py` to `research/fit_tilt.py` (behavior unchanged). |
| `docs/setup.md` | Clarified that `scikit-learn`/`joblib` are legacy/research ML deps unused by Preserve and documented migration routes. |
| `README.md` | Documented `legacy -> Preserve` and `research -> Preserve` isolation boundary with link to E9 report. |

## 6. Limitations & Honest Boundaries

1. `pyproject.toml`, `requirements.txt`, and `rnext/doctor.py` retain `scikit-learn` and `joblib` for backward compatibility with `rnext/ml/*` and legacy `restore_one.py --ml-snapshot`. They are not imported or required at runtime by `Preserve` (`scripts/restore_safe.py`).
2. No legacy scripts, historical manifests, or corpus inputs were deleted; backward compatibility is verified by `tests/test_e9_isolation.py`.
3. No ML model training or Colab GPU execution was run in E9.
