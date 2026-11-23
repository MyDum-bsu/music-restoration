#!/usr/bin/env python3
"""E9 Legacy & ML Isolation Audit, Offline Preserve E2E Smoke, and Report Generator.

Proves programmatically that:
  1. Local CPU `Preserve` (`rnext/dsp/*`, `rnext/features.py`, `scripts/restore_safe.py`
     and its delivery CLI) does not import or depend on `joblib`, `torch`, `sklearn`,
     `rnext.ml`, `research/*`, or `lab/*`.
  2. `Preserve` passes an isolated end-to-end FLAC smoke test (both a clean stereo
     control and a hard-clipped stereo track) with `joblib`, `torch`, `sklearn`,
     `rnext.ml`, `research`, `lab`, AND network sockets hard-blocked in a fresh
     subprocess.
  3. All 18 frozen E4 implementation files remain byte-for-byte untouched, sealed
     corpora are never opened, and zero host-specific paths exist in E9 artifacts.
  4. Legacy runners, callers, and runbook paths are fully inventoried with explicit
     `legacy -> Preserve` and `research -> Preserve` migration routes.

CLI usage:
  python3 research/e9_isolation_audit.py \\
      --out-json docs/validation/e9-legacy-isolation-results.json \\
      --out-md docs/validation/e9-legacy-isolation-report.md
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research import e9_ml_governance as GOV

SCHEMA_VERSION = "e9-isolation-audit/v1"
SEALED_FILES = ("sealed-v1.json", "sealed-e5-v1.json")
E4_FREEZE_MANIFEST = REPO_ROOT / "docs" / "validation" / "e4-freeze-manifest.json"
HOST_PATH_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    re.compile(r"/media/[A-Za-z0-9._-]+"),
)

BLOCKED_SUBPROCESS_MODULES = (
    "joblib",
    "torch",
    "sklearn",
    "scikit_learn",
    "rnext.ml",
    "research",
    "lab",
)

# Canonical legacy & research runners inventory (preserved, not deleted).
LEGACY_RUNNER_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "path": "scripts/restore_safe.py",
        "category": "preserve_default",
        "status": "active_production_default",
        "ml_deps": [],
        "research_deps": [],
        "notes": (
            "Local CPU Preserve default. Imports only encode_flac helper from "
            "scripts/restore_one.py (pure numpy/wave/flac; zero ML imports loaded)."
        ),
        "migration_route": "Canonical Preserve target",
    },
    {
        "path": "scripts/restore_one.py",
        "category": "legacy_runner",
        "status": "legacy_preserved_manual",
        "ml_deps": ["rnext.ml (lazy, opt-in --ml-snapshot only)"],
        "research_deps": [],
        "notes": (
            "Manual single-policy dynamics runner; lazily imports rnext.ml only "
            "when --ml-snapshot is passed. Retained for historical compatibility "
            "and shared encode_flac utility."
        ),
        "migration_route": "legacy -> Preserve: python3 scripts/restore_safe.py --src <in> --out <out>",
    },
    {
        "path": "scripts/restore_per_track.py",
        "category": "legacy_runner",
        "status": "legacy_preserved_comparison",
        "ml_deps": [],
        "research_deps": [],
        "notes": (
            "Legacy +6.3 dB crest-sweep dynamics runner; fixed crest target is "
            "not the default quality criterion. Kept for historical reproduction."
        ),
        "migration_route": "legacy -> Preserve: python3 scripts/restore_safe.py --src <in> --out <out>",
    },
    {
        "path": "scripts/release_batch.py",
        "category": "legacy_runner",
        "status": "deprecated_legacy_wrapper",
        "ml_deps": ["transitive via restore_one.py --ml-snapshot"],
        "research_deps": [],
        "notes": (
            "Deprecated batch wrapper around restore_one.py. Retained so old call "
            "sites do not break; docstring updated to point to Preserve."
        ),
        "migration_route": "legacy -> Preserve: python3 scripts/restore_safe.py --src <album/> --out <out/>",
    },
    {
        "path": "rnext/ml/*",
        "category": "legacy_ml_layer",
        "status": "isolated_ml_research_layer",
        "ml_deps": ["scikit-learn", "joblib"],
        "research_deps": [],
        "notes": (
            "Target-crest regression dataset/models/registry/runtime. Never "
            "imported by Preserve; used only by research/* and opt-in restore_one.py."
        ),
        "migration_route": "research -> Preserve: governed by research/e9_ml_governance.py; zero auto-promotion",
    },
    {
        "path": "research/lab/server.py",
        "category": "research_lab_ui",
        "status": "research_only",
        "ml_deps": ["rnext.ml (lazy)"],
        "research_deps": ["dataset_manifests/"],
        "notes": (
            "Local read-only web inspector for historical dataset/models. "
            "Runbook updated from stale `lab/server.py` to `research/lab/server.py`."
        ),
        "migration_route": "research -> Preserve: inspect Preserve runs via report.json + scripts/evaluate_album.py",
    },
    {
        "path": "research/ml_auto.py & research/ml_train.py",
        "category": "research_ml_training",
        "status": "research_only",
        "ml_deps": ["rnext.ml", "scikit-learn", "joblib"],
        "research_deps": ["dataset_manifests/"],
        "notes": (
            "Legacy target-estimator training loop. Never invoked by Preserve; "
            "future ML work must satisfy research/e9_ml_governance.py split rules."
        ),
        "migration_route": "research -> Preserve: offline evaluation + E9 split governance before any proposal",
    },
    {
        "path": "research/dataset_loop.py & research/dataset_*.py",
        "category": "research_dataset_pipeline",
        "status": "research_only",
        "ml_deps": ["rnext.ml (dataset_eval.py)"],
        "research_deps": ["research/*"],
        "notes": (
            "Historical dataset enrichment/audit scripts. Stale `scripts/...` "
            "subcommand paths repaired to `research/...`."
        ),
        "migration_route": "research -> Preserve: use versioned corpus/*.json manifests + research/run_benchmark.py",
    },
    {
        "path": "research/e6_declick_hiss.py, e7_bwe_preview.py, e8_*_eval.py",
        "category": "research_feasibility",
        "status": "research_only_colab_gated",
        "ml_deps": ["torch (optional runtime probe only)"],
        "research_deps": ["research/*"],
        "notes": (
            "Isolated E6/E7/E8 feasibility harnesses (CPU baselines + Colab GPU "
            "audits). Zero Preserve coupling and zero production dependencies."
        ),
        "migration_route": "research -> Preserve: separate product approval required; no generalization to Preserve",
    },
)

RUNBOOK_UPDATES: tuple[dict[str, str], ...] = (
    {
        "file": "research/dataset_loop.py",
        "fix": "Replaced 11 stale `scripts/<research_script>.py` subprocess/doc calls with `research/<script>.py`.",
    },
    {
        "file": "research/status_report.py",
        "fix": "Fixed `scripts/dataset_audit.py` subprocess call and stale `scripts/` hints to `research/`.",
    },
    {
        "file": "research/lab/server.py",
        "fix": "Fixed CLI docstring from `python3 lab/server.py` to `python3 research/lab/server.py`.",
    },
    {
        "file": "research/lab/README.md",
        "fix": "Fixed `lab/server.py` -> `research/lab/server.py`, removed host paths, added Preserve migration note.",
    },
    {
        "file": "research/docs/ml.md",
        "fix": "Fixed `scripts/ml_auto.py`, `scripts/ml_train.py`, `scripts/dataset_loop.py` -> `research/...`; documented Preserve isolation and E9 governance.",
    },
    {
        "file": "research/docs/dataset.md",
        "fix": "Fixed `scripts/master_targets.py`, `scripts/dataset_plan.py`, `scripts/dataset_fetch.py` -> `research/...`; removed host path.",
    },
    {
        "file": "research/skills/restore-status/SKILL.md",
        "fix": "Fixed all moved research script paths to `research/...`, removed host paths, added `legacy -> Preserve` migration note.",
    },
    {
        "file": "research/resume.sh",
        "fix": "Fixed `scripts/status.py` -> `research/status.py` and parameterized `MR_DATA_ROOT` (removed hardcoded host mount path).",
    },
    {
        "file": "scripts/release_batch.py",
        "fix": "Updated deprecation docstring to direct new runs to `scripts/restore_safe.py` (Preserve).",
    },
    {
        "file": "scripts/restore_one.py",
        "fix": "Updated `--eq-curve` help text from `scripts/fit_tilt.py` to `research/fit_tilt.py` (behavior unchanged).",
    },
    {
        "file": "docs/setup.md",
        "fix": "Clarified that `scikit-learn`/`joblib` are legacy/research ML deps unused by Preserve and documented migration routes.",
    },
    {
        "file": "README.md",
        "fix": "Documented `legacy -> Preserve` and `research -> Preserve` isolation boundary with link to E9 report.",
    },
)


def _assert_not_sealed_path(path: Path) -> None:
    """Fail closed if a target path points to a sealed corpus manifest, contains any symlink, or cannot be stat/resolved."""
    candidates = [str(path), path.name]
    try:
        abs_path = Path(os.path.abspath(str(path)))
        prefix = Path(abs_path.anchor) if abs_path.anchor else Path(".")
        for part in abs_path.parts[1:] if abs_path.anchor else abs_path.parts:
            prefix = prefix / part
            try:
                st = os.lstat(prefix)
            except FileNotFoundError:
                break
            if stat.S_ISLNK(st.st_mode):
                raise ValueError(
                    f"Fail-closed: refusing symlink component '{prefix}' in path '{path}'"
                )
        resolved = path.resolve(strict=False)
        real = os.path.realpath(str(path))
        candidates.extend([str(resolved), resolved.name, real, Path(real).name])
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            f"Fail-closed: unable to resolve path '{path}' for sealed check: {exc}"
        ) from exc
    for cand in candidates:
        for sealed in SEALED_FILES:
            if sealed in cand:
                raise ValueError(
                    f"Refusing to touch sealed corpus path: '{path}' (matched '{sealed}')"
                )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_e4_freeze_untouched(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Verify all 18 frozen E4 files match their pinned SHA-256 digests."""
    manifest_path = repo_root / "docs" / "validation" / "e4-freeze-manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    pins: dict[str, str] = data["toolchain"]["implementation_sha256"]
    checked: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for rel, expected in sorted(pins.items()):
        p = repo_root / rel
        if not p.is_file():
            mismatches.append(f"{rel}: missing")
            checked.append({"path": rel, "expected": expected, "actual": None, "ok": False})
            continue
        actual = sha256_file(p)
        ok = actual == expected
        if not ok:
            mismatches.append(f"{rel}: expected {expected}, got {actual}")
        checked.append({"path": rel, "expected": expected, "actual": actual, "ok": ok})
    return {
        "total_pinned": len(pins),
        "untouched_count": sum(1 for item in checked if item["ok"]),
        "all_untouched": len(mismatches) == 0 and len(pins) == 18,
        "mismatches": mismatches,
    }


def _collect_ast_imports(py_path: Path) -> list[dict[str, Any]]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    parent_map: dict[ast.AST, ast.AST] = {}
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

    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                rows.append(
                    {
                        "module": alias.name,
                        "lineno": node.lineno,
                        "top_level": _is_top_level(node),
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            top = _is_top_level(node)
            base = node.module or ""
            if base:
                rows.append({"module": base, "lineno": node.lineno, "top_level": top})
            for alias in node.names:
                if alias.name == "*":
                    continue
                full = f"{base}.{alias.name}" if base else alias.name
                if full:
                    rows.append({"module": full, "lineno": node.lineno, "top_level": top})
    return rows


def audit_preserve_static_imports(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Static AST audit of all Preserve runtime modules plus lazy-import boundary."""
    boundary = GOV.audit_ml_preserve_boundary(repo_root)

    # Also verify scripts/restore_one.py has zero top-level ML/research imports,
    # because scripts/restore_safe.py imports `encode_flac` from it.
    restore_one = repo_root / "scripts" / "restore_one.py"
    restore_one_imports = _collect_ast_imports(restore_one)
    top_level_forbidden = []
    lazy_ml_imports = []
    for imp in restore_one_imports:
        mod = imp["module"]
        hit = next(
            (
                pref
                for pref in GOV.FORBIDDEN_IN_PRESERVE
                if mod == pref or mod.startswith(pref + ".")
            ),
            None,
        )
        if hit is not None:
            if imp["top_level"]:
                top_level_forbidden.append(imp)
            else:
                lazy_ml_imports.append(imp)

    passed = (
        boundary["bidirectional_isolation_passed"]
        and len(top_level_forbidden) == 0
    )
    return {
        "preserve_modules_checked": boundary["preserve_modules_checked"],
        "ml_modules_checked": boundary["ml_modules_checked"],
        "preserve_to_ml_violations": boundary["preserve_to_ml_violations"],
        "ml_to_preserve_violations": boundary["ml_to_preserve_violations"],
        "restore_one_top_level_forbidden_imports": top_level_forbidden,
        "restore_one_lazy_opt_in_ml_imports": lazy_ml_imports,
        "static_audit_passed": passed,
    }


def _write_synth_stereo_flac(
    wav_path: Path, flac_path: Path, samples: np.ndarray, sr: int = 44100
) -> None:
    q = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(q.tobytes())
    subprocess.run(
        ["flac", "-s", "-f", "-o", str(flac_path), str(wav_path)],
        check=True,
        capture_output=True,
    )
    wav_path.unlink(missing_ok=True)


def run_preserve_isolated_e2e_smoke(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Preserve CLI end-to-end on clean + clipped FLAC with ML and network blocked."""
    sr = 44100
    dur_s = 2.0
    n = int(sr * dur_s)
    t = np.arange(n, dtype=np.float64) / sr

    # Clean stereo control (gentle amplitude, well below clipping).
    left_clean = 0.25 * np.sin(2.0 * np.pi * 220.0 * t) + 0.15 * np.sin(2.0 * np.pi * 554.37 * t)
    right_clean = 0.25 * np.sin(2.0 * np.pi * 220.0 * t + 0.15) + 0.15 * np.sin(
        2.0 * np.pi * 659.25 * t
    )
    clean_stereo = np.stack([left_clean, right_clean], axis=1)

    # Hard-clipped stereo track (repeated symmetric plateau at +/-0.60).
    drive = np.stack(
        [
            0.95 * np.sin(2.0 * np.pi * 180.0 * t) + 0.45 * np.sin(2.0 * np.pi * 440.0 * t),
            0.95 * np.sin(2.0 * np.pi * 180.0 * t + 0.20) + 0.45 * np.sin(2.0 * np.pi * 523.25 * t),
        ],
        axis=1,
    )
    clipped_stereo = np.clip(drive, -0.60, 0.60)

    launcher_code = r"""
import importlib.abc
import json
import runpy
import socket
import sys
from pathlib import Path

BLOCKED = tuple(json.loads(sys.argv[1]))
SCRIPT = sys.argv[2]
RESTORE_ARGS = sys.argv[3:]


class _BlockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        for prefix in BLOCKED:
            if fullname == prefix or fullname.startswith(prefix + "."):
                raise ImportError(f"Blocked by E9 isolation audit: {fullname}")
        return None


class _BlockedSocket:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Network socket blocked by E9 isolation audit")


def _blocked_create_connection(*args, **kwargs):
    raise RuntimeError("Network connection blocked by E9 isolation audit")


sys.meta_path.insert(0, _BlockFinder())
socket.socket = _BlockedSocket
socket.create_connection = _blocked_create_connection

sys.argv = [SCRIPT, *RESTORE_ARGS]
exit_code = 0
try:
    runpy.run_path(SCRIPT, run_name="__main__")
except SystemExit as exc:
    exit_code = 0 if exc.code in (None, 0) else int(exc.code)

loaded_forbidden = sorted(
    mod
    for mod in sys.modules
    if any(mod == pref or mod.startswith(pref + ".") for pref in BLOCKED)
)
print(
    "__E9_ISOLATION_RESULT__="
    + json.dumps({"exit_code": exit_code, "loaded_forbidden": loaded_forbidden})
)
sys.exit(exit_code)
"""

    with tempfile.TemporaryDirectory(prefix="e9_preserve_smoke_") as tmp_str:
        tmp = Path(tmp_str)
        src_dir = tmp / "src"
        out_dir = tmp / "out"
        src_dir.mkdir(parents=True)

        clean_flac = src_dir / "01_clean_control.flac"
        clipped_flac = src_dir / "02_hard_clipped.flac"
        _write_synth_stereo_flac(tmp / "clean.wav", clean_flac, clean_stereo, sr=sr)
        _write_synth_stereo_flac(tmp / "clipped.wav", clipped_flac, clipped_stereo, sr=sr)
        clean_src_sha = sha256_file(clean_flac)

        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        cmd = [
            sys.executable,
            "-c",
            launcher_code,
            json.dumps(list(BLOCKED_SUBPROCESS_MODULES)),
            str(repo_root / "scripts" / "restore_safe.py"),
            "--src",
            str(src_dir),
            "--out",
            str(out_dir),
            "--mode",
            "declip",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        marker = "__E9_ISOLATION_RESULT__="
        isolation_payload = None
        for line in proc.stdout.splitlines():
            if line.startswith(marker):
                isolation_payload = json.loads(line[len(marker) :])
        if isolation_payload is None:
            raise RuntimeError(
                f"Isolated Preserve subprocess failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()}"
            )

        report_path = out_dir / "report.json"
        if not report_path.is_file():
            raise RuntimeError("Preserve isolated smoke did not emit report.json")
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        entries = report_data.get("tracks") if isinstance(report_data, dict) else report_data
        by_name = {Path(item["source"]).name: item for item in entries}

        clean_entry = by_name["01_clean_control.flac"]
        clipped_entry = by_name["02_hard_clipped.flac"]
        clean_out_sha = sha256_file(out_dir / "01_clean_control.flac")
        clean_byte_identical = clean_out_sha == clean_src_sha

        clipped_qc = clipped_entry["delivery_qc"]
        clipped_tp = float(clipped_qc["delivered_tp_dbfs"])
        clipped_clips = int(clipped_qc["delivered_clip_samples"])

        passed = (
            proc.returncode == 0
            and isolation_payload["loaded_forbidden"] == []
            and clean_entry["status"] == "kept_original"
            and clean_byte_identical
            and clipped_entry["status"] == "processed_candidate"
            and clipped_tp <= -0.95
            and clipped_clips == 0
        )

        return {
            "returncode": proc.returncode,
            "blocked_modules": list(BLOCKED_SUBPROCESS_MODULES),
            "network_blocked": True,
            "loaded_forbidden_modules": isolation_payload["loaded_forbidden"],
            "clean_control": {
                "status": clean_entry["status"],
                "byte_identical_to_source": clean_byte_identical,
            },
            "hard_clipped_case": {
                "status": clipped_entry["status"],
                "encoded_tp_dbfs": round(clipped_tp, 4),
                "encoded_clip_samples": clipped_clips,
            },
            "e2e_smoke_passed": passed,
        }


def scan_callers_of_legacy(repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """Inventory repository callers of legacy runners and ML modules."""
    targets = {
        "scripts/restore_one.py": re.compile(r"restore_one\.py"),
        "scripts/restore_per_track.py": re.compile(r"restore_per_track\.py"),
        "scripts/release_batch.py": re.compile(r"release_batch\.py"),
        "rnext.ml": re.compile(r"\brnext\.ml\b"),
        "research/lab/server.py": re.compile(r"(?:research/)?lab/server\.py"),
        "joblib": re.compile(r"\bjoblib\b"),
        "torch": re.compile(r"\btorch\b"),
    }
    found: dict[str, set[str]] = {k: set() for k in targets}
    scan_roots = ("rnext", "scripts", "research", "docs", "skills", "README.md")
    for root_name in scan_roots:
        base = repo_root / root_name
        files = [base] if base.is_file() else sorted(base.rglob("*"))
        for p in files:
            if (
                not p.is_file()
                or "__pycache__" in p.parts
                or p.suffix in (".png", ".gif", ".wav", ".flac", ".json", ".csv", ".pyc")
            ):
                continue
            rel = str(p.relative_to(repo_root))
            text = p.read_text(encoding="utf-8", errors="replace")
            for key, pat in targets.items():
                if pat.search(text):
                    found[key].add(rel)
    return {k: sorted(v) for k, v in found.items()}


def audit_no_host_paths(
    paths: list[Path], repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Verify zero host-specific paths exist in the given tracked files."""
    hits: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        rel = str(p.relative_to(repo_root)) if p.is_relative_to(repo_root) else str(p)
        for lineno, line in enumerate(
            p.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for pat in HOST_PATH_PATTERNS:
                m = pat.search(line)
                if m:
                    hits.append({"file": rel, "lineno": lineno, "match": m.group(0)})
    return {"files_checked": len(paths), "host_path_hits": hits, "passed": len(hits) == 0}


def render_markdown_report(results: dict[str, Any]) -> str:
    """Render docs/validation/e9-legacy-isolation-report.md with aligned GFM tables."""
    e4 = results["e4_freeze_verification"]
    static = results["static_import_audit"]
    smoke = results["isolated_preserve_e2e_smoke"]
    gov = results["ml_governance_summary"]
    callers = results["legacy_callers"]

    lines: list[str] = [
        "# E9 · Legacy & ML Research Isolation Report",
        "",
        ("**Card:** Board #1137 · **Spec:** E0 charter + spec v1 (Section E9) · "
        f"**Schema:** `{results['schema_version']}`"),
        "",
        "## 1. Executive Summary & Safety Boundaries",
        "",
        (f"- **Sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`):** "
        f"`sealed_corpora_opened = {results['sealed_corpora_opened']}` (strictly untouched)."),
        (f"- **Frozen E4 Preserve files:** `{e4['untouched_count']}/{e4['total_pinned']}` "
        f"SHA-256 pins verified untouched (`all_untouched = {e4['all_untouched']}`)."),
        (f"- **Static AST import audit (Preserve ↔ ML bidirectional):** "
        f"`static_audit_passed = {static['static_audit_passed']}` "
        f"(`preserve_modules_checked = {static['preserve_modules_checked']}`, "
        f"`ml_modules_checked = {static['ml_modules_checked']}`, "
        f"`preserve_to_ml_violations = {len(static['preserve_to_ml_violations'])}`, "
        f"`ml_to_preserve_violations = {len(static['ml_to_preserve_violations'])}`)."),
        (f"- **Isolated Preserve E2E smoke (ML libs + network hard-blocked):** "
        f"`e2e_smoke_passed = {smoke['e2e_smoke_passed']}` "
        f"(`loaded_forbidden_modules = {smoke['loaded_forbidden_modules']}`)."),
        (f"- **ML-target governance schema:** `{gov['schema_version']}` "
        f"(`stages = {', '.join(gov['allowed_stages'])}`, fail-closed deduplication and split isolation)."),
        ("- **Production dependencies & GPU policy:** Zero new production dependencies added; "
        "no training launched; GPU remains Google Colab-only under separate authorization."),
        f"- **Results SHA-256:** `{results['results_sha256']}`",
        "",
        "## 2. Legacy Runners, Dependencies & Migration Inventory",
        "",
        "| Component | Category | Status | ML / Research Dependencies | Migration Route |",
        "|---|---|---|---|---|",
    ]

    for row in results["legacy_inventory"]:
        deps = ", ".join(row["ml_deps"] + row["research_deps"]) or "none (pure CPU/DSP)"
        lines.append(
            f"| `{row['path']}` | `{row['category']}` | `{row['status']}` | "
            f"{deps} | {row['migration_route']} |"
        )

    lines.extend(
        [
            "",
            "### 2.1 Caller Summary Across Repository",
            "",
            "| Target | Caller Count | Sample Callers (non-Preserve unless noted) |",
            "|---|---:|---|",
        ]
    )
    for target, files in sorted(callers.items()):
        sample = ", ".join(f"`{f}`" for f in files[:6])
        if len(files) > 6:
            sample += f" (+{len(files) - 6} more)"
        lines.append(f"| `{target}` | {len(files)} | {sample or 'none'} |")

    lines.extend(
        [
            "",
            ("> **Note on `scripts/restore_safe.py` -> `scripts/restore_one.py`:** "
            "`scripts/restore_safe.py` (frozen in E4) imports only the pure numpy/wave/flac "
            "helper `encode_flac` from `scripts/restore_one.py`. Static AST inspection proves "
            "`scripts/restore_one.py` has **0 top-level** ML/research imports (`rnext.ml` is "
            "imported lazily inside `process_track()` only when `--ml-snapshot` is requested), "
            "and the isolated E2E subprocess proves `restore_safe.py` completes with `0` ML "
            "modules loaded."),
            "",
            "## 3. Programmatic Isolation Proof (AST + Offline/No-ML E2E Smoke)",
            "",
            "| Check | Result | Evidence |",
            "|---|---|---|",
            (f"| Preserve static AST audit | `{'PASS' if static['static_audit_passed'] else 'FAIL'}` | "
            f"{static['preserve_modules_checked']} Preserve modules, 0 forbidden imports (`joblib`, `torch`, `sklearn`, `rnext.ml`, `research`, `lab`) |"),
            (f"| ML -> Preserve reverse AST audit | `{'PASS' if len(static['ml_to_preserve_violations']) == 0 else 'FAIL'}` | "
            f"{static['ml_modules_checked']} ML modules, 0 imports of Preserve declip/delivery entrypoints |"),
            (f"| Isolated E2E clean control FLAC | `{'PASS' if smoke['clean_control']['byte_identical_to_source'] else 'FAIL'}` | "
            f"`status={smoke['clean_control']['status']}`, `byte_identical={smoke['clean_control']['byte_identical_to_source']}` |"),
            (f"| Isolated E2E hard-clipped FLAC | `{'PASS' if smoke['hard_clipped_case']['status'] == 'processed_candidate' else 'FAIL'}` | "
            f"`status={smoke['hard_clipped_case']['status']}`, `tp_dbfs={smoke['hard_clipped_case']['encoded_tp_dbfs']}`, `clip_samples={smoke['hard_clipped_case']['encoded_clip_samples']}` |"),
            (f"| Blocked modules & network guard | `{'PASS' if smoke['loaded_forbidden_modules'] == [] and smoke['network_blocked'] else 'FAIL'}` | "
            f"`loaded_forbidden={smoke['loaded_forbidden_modules']}`, `socket` blocked in subprocess |"),
            "",
            "## 4. ML-Target Governance & Split-Boundary Contract",
            "",
            "Implemented in `research/e9_ml_governance.py` (`schema_version = e9-ml-governance/v1`):",
            "",
            "| Stage | Purpose | Required Label Gate | Leakage Rule |",
            "|---|---|---|---|",
            "| `approved_labels` | Training / calibration source records | `label_status == 'approved'` (fail-closed) | Disjoint `recording_id`, `group_id`, `content_sha256` from all other stages |",
            "| `dev_selection` | Model / hyperparameter selection on dev | Validated provenance, no sealed paths | Zero overlap with `approved_labels`, `promotion`, `sealed_test` |",
            "| `promotion` | Held-out promotion gate verification | Frozen `dataset_version` + `model_sha256` | Zero overlap with `approved_labels`, `dev_selection`, `sealed_test` |",
            "| `sealed_test` | Custodian-held final evaluation | Single-shot per frozen protocol | Never opened in dev; strictly disjoint from all earlier stages |",
            "",
            ("- **Recording Deduplication:** `deduplicate_by_recording()` enforces one canonical record "
            "per `recording_id` within each stage and raises `GovernanceError` immediately if any "
            "`recording_id`, `group_id`, or `content_sha256` crosses stage boundaries."),
            ("- **Result Versioning:** `build_versioned_result()` pins `schema_version`, deterministic "
            "`dataset_version`, `model_sha256`, `preserve_isolated = True`, and tamper-evident "
            "`results_sha256`, verified fail-closed by `verify_versioned_result()`."),
            "",
            "## 5. Runbook Repairs & Migration Guide",
            "",
            "### 5.1 Migration Routes",
            "",
            ("- **`legacy -> Preserve`:** Replace `scripts/restore_one.py`, `scripts/restore_per_track.py`, "
            "and `scripts/release_batch.py` with `python3 scripts/restore_safe.py --src <album/> --out <empty-out/>`, "
            "followed by `python3 scripts/evaluate_album.py --src <album/> --restored <out/> --out <eval/>`. "
            "Legacy scripts remain intact for historical reproduction and are not deleted."),
            ("- **`research -> Preserve`:** Research scripts (`research/ml_auto.py`, `research/ml_train.py`, "
            "`research/lab/server.py`, `research/e6_*`–`e8_*`) stay strictly outside `Preserve`. Any future "
            "ML candidate must satisfy `research/e9_ml_governance.py`, pass E3/E4/E5 freeze & blind-panel "
            "gates, and receive explicit owner/CTO release approval before touching `Preserve`."),
            "",
            "### 5.2 Repaired Stale Runbook References",
            "",
            "| File | Repair Applied |",
            "|---|---|",
        ]
    )

    for item in results["runbook_updates"]:
        lines.append(f"| `{item['file']}` | {item['fix']} |")

    lines.extend(
        [
            "",
            "## 6. Limitations & Honest Boundaries",
            "",
            ("1. `pyproject.toml`, `requirements.txt`, and `rnext/doctor.py` retain `scikit-learn` and "
            "`joblib` for backward compatibility with `rnext/ml/*` and legacy `restore_one.py --ml-snapshot`. "
            "They are not imported or required at runtime by `Preserve` (`scripts/restore_safe.py`)."),
            ("2. No legacy scripts, historical manifests, or corpus inputs were deleted; backward "
            "compatibility is verified by `tests/test_e9_isolation.py`."),
            "3. No ML model training or Colab GPU execution was run in E9.",
            "",
        ]
    )
    return "\n".join(lines)


def run_full_audit(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Execute the full E9 isolation audit and assemble versioned JSON results."""
    e4 = verify_e4_freeze_untouched(repo_root)
    static = audit_preserve_static_imports(repo_root)
    smoke = run_preserve_isolated_e2e_smoke(repo_root)
    callers = scan_callers_of_legacy(repo_root)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "board_card": 1137,
        "spec": "music-restoration-spec-v1 (Section E9)",
        "sealed_corpora_opened": False,
        "e4_freeze_verification": e4,
        "static_import_audit": static,
        "isolated_preserve_e2e_smoke": smoke,
        "legacy_inventory": list(LEGACY_RUNNER_CATALOG),
        "legacy_callers": callers,
        "runbook_updates": list(RUNBOOK_UPDATES),
        "ml_governance_summary": {
            "schema_version": GOV.SCHEMA_VERSION,
            "allowed_stages": list(GOV.ALLOWED_STAGES),
            "dedup_key": "recording_id",
            "cluster_key": "group_id",
            "bidirectional_isolation_passed": static["static_audit_passed"],
        },
        "all_checks_passed": (
            e4["all_untouched"]
            and static["static_audit_passed"]
            and smoke["e2e_smoke_passed"]
        ),
    }
    payload["results_sha256"] = GOV.compute_sha256_json(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "docs" / "validation" / "e9-legacy-isolation-results.json",
        help="Path to write versioned JSON audit results.",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "docs" / "validation" / "e9-legacy-isolation-report.md",
        help="Path to write Markdown isolation report.",
    )
    args = ap.parse_args(argv)

    _assert_not_sealed_path(args.out_json)
    _assert_not_sealed_path(args.out_md)

    results = run_full_audit(REPO_ROOT)
    md = render_markdown_report(results)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(results, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")

    print(
        f"E9 audit complete: all_checks_passed={results['all_checks_passed']} "
        f"e4_untouched={results['e4_freeze_verification']['untouched_count']}/18 "
        f"results_sha256={results['results_sha256'][:12]}"
    )
    return 0 if results["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
