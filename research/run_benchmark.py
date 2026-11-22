#!/usr/bin/env python3
"""Reproducible benchmark runner over a corpus/v1 manifest.

    python3 research/run_benchmark.py --manifest corpus/smoke-v1.json \
        --audio-root corpus/audio/smoke --out out/run1

The runner validates the manifest (hashes, rights, split leakage, sealed
gates), executes each case in its declared mode, and writes:

  protocol.json  manifest digest, toolchain pins, implementation hashes, argv
  results.json   one row per case: id, split, input sha256, per-method status
                 and metrics — sorted, finite, no NaN/Infinity

Two clean checkouts running the same manifest must produce identical case ids,
input hashes and statuses, with metrics inside `numeric_tolerance`. Sealed
cases never appear in a dev report: they are excluded unless
--custodian-release is passed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research import corpus_v1, corpus_synth
from research.bench_declipping import reconstruction_metrics, _sdr
from rnext import features as F
from rnext.dsp.repair import select_restoration

METHODS = ("identity", "gain_only", "restore")


def _decode(spec_path):
    info = F.ffprobe_audio(str(spec_path))
    pcm, rate = F.decode_f32(str(spec_path), strict=True, channels=None)
    return pcm.astype(np.float64), rate, info


def _masked_sdr(clean, cand, mask):
    """Fixed-gain SDR restricted to the true clipped region."""
    if not mask.any():
        return None
    return _sdr(float(np.sum(clean[mask] ** 2)),
                float(np.sum((clean[mask] - cand[mask]) ** 2)))


def run_case(doc, root, case, methods, tolerance):
    """One manifest case -> results row (statuses never raise)."""
    row = {"id": case["id"], "split": case.get("split"), "profile": case.get("profile"),
           "mode": case.get("mode", "auto"), "methods": {}}
    try:
        degraded_path = corpus_v1.resolve(doc, root, case["degraded"])
        degraded, rate, info = _decode(degraded_path)
        row["input_sha256"] = case["degraded"]["sha256"]
        clean = mask = None
        if case["kind"] != "unpaired":
            clean_path = corpus_v1.resolve(doc, root, case["clean"])
            clean, clean_rate, _ = _decode(clean_path)
            if clean_rate != rate or clean.shape != degraded.shape:
                raise ValueError("clean/degraded misaligned")
            if case.get("mask") is not None:
                mask_path = corpus_v1.resolve(doc, root, case["mask"])
                mask = corpus_synth.decode_mask(mask_path.read_bytes())
            row["clean_sha256"] = case["clean"]["sha256"]
        expected = case.get("expected_input_sdr_db")
        if expected is not None and clean is not None:
            actual = reconstruction_metrics(clean, degraded)["fixed_gain_sdr_db"]
            if actual is None or abs(actual - expected) > max(tolerance, .1):
                raise ValueError(f"wrong degradation: expected input SDR {expected}, got {actual}")
        outputs = {"identity": degraded}
        ceiling = 10 ** (doc.get("output_ceiling_dbfs", -1.0) / 20)
        peak = float(np.abs(degraded).max())
        outputs["gain_only"] = degraded * min(1.0, ceiling / peak) if peak > 0 else degraded
        for name in methods:
            if name not in METHODS:
                row["methods"][name] = {"status": "error", "error": f"unknown method {name}"}
                continue
            try:
                if name == "restore":
                    restored, report = select_restoration(
                        degraded, rate, mode=case.get("mode", "auto"),
                        bits=info["bps"] or 16)
                    cand, extra = restored, {"repair_status": report["status"]}
                else:
                    cand, extra = outputs[name], {}
                entry = {"status": "ok", **extra}
                if clean is not None:
                    m = reconstruction_metrics(clean, cand)
                    entry["fixed_gain_sdr_db"] = m["fixed_gain_sdr_db"]
                    entry["scale_aligned_sdr_db"] = m["scale_aligned_sdr_db"]
                    entry["masked_fixed_gain_sdr_db"] = _masked_sdr(clean, cand, mask) \
                        if mask is not None else None
                row["methods"][name] = entry
            except Exception as exc:
                row["methods"][name] = {"status": "error",
                                        "error": _clean_error(str(exc))[:300]}
    except Exception as exc:
        row["methods"] = {name: {"status": "error",
                                 "error": _clean_error(str(exc))[:300]}
                          for name in methods}
        row["status"] = "error"
        row["error"] = _clean_error(str(exc))[:300]
        return row
    row["status"] = "error" if any(m["status"] == "error" for m in row["methods"].values()) else "ok"
    return row


def _walk_errors(a, b, path, tol, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(f"{path}: keys {sorted(set(a) ^ set(b))} differ")
        for k in sorted(set(a) & set(b)):
            _walk_errors(a[k], b[k], f"{path}.{k}", tol, diffs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool):
        if not (np.isfinite(a) and np.isfinite(b)) or abs(a - b) > tol:
            diffs.append(f"{path}: {a} vs {b} (tol {tol})")
    elif a != b:
        diffs.append(f"{path}: {a!r} vs {b!r}")


def compare_reports(a_rows, b_rows, tolerance):
    """Diff two results.json payloads; empty list = reproducible."""
    diffs = []
    if [r["id"] for r in a_rows] != [r["id"] for r in b_rows]:
        return ["case id sets differ"]
    for ra, rb in zip(a_rows, b_rows):
        for key in ("status", "input_sha256", "clean_sha256", "split"):
            if ra.get(key) != rb.get(key):
                diffs.append(f"{ra['id']}.{key}: {ra.get(key)!r} vs {rb.get(key)!r}")
        _walk_errors(ra["methods"], rb["methods"], ra["id"], tolerance, diffs)
    return diffs


def _clean_error(text):
    """Strip local absolute paths from error text before it reaches reports."""
    for prefix in (str(Path.cwd()), str(REPO), str(Path.home())):
        text = text.replace(prefix, ".")
    return re.sub(r"(?<![\w.])/(?:[\w.\-]+/)+", "/", text)


def _safe_argv():
    """argv with absolute paths reduced to basenames (protocol record)."""
    return [a if not a.startswith("/") else Path(a).name for a in sys.argv]


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _implementation_hashes():
    files = [REPO / "research" / n for n in
             ("run_benchmark.py", "corpus_v1.py", "corpus_synth.py",
              "bench_declipping.py", "fetch_declipping.py")]
    files += [REPO / "rnext/dsp" / (n + ".py") for n in
              ("sparse", "predictive", "repair", "selection", "contour", "declip")]
    files.append(REPO / "rnext/features.py")
    return {str(p.relative_to(REPO)): corpus_v1.digest(p) for p in files if p.exists()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--audio-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--operation", default="local_inference", choices=corpus_v1.OPERATIONS)
    p.add_argument("--methods", default="identity,gain_only,restore")
    p.add_argument("--custodian-release", action="store_true",
                   help="run sealed cases; only valid with custodian approval")
    args = p.parse_args()

    doc = corpus_v1.load(args.manifest)
    root = Path(args.audio_root).resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Sealed cases are partitioned BEFORE any rights/file checks: a mixed
    # manifest must not fail — or leak — on inputs the caller cannot touch.
    sealed = [c for c in doc.get("cases", []) if c.get("sealed")]
    if doc.get("tier") == "sealed" and not args.custodian_release:
        print("manifest error: sealed manifest requires --custodian-release",
              file=sys.stderr)
        return 2
    view = doc if args.custodian_release else corpus_v1.without_sealed(doc)
    errors = corpus_v1.validate_manifest(view, check_hashes_root=root)
    for case in view.get("cases", []):
        if not corpus_v1.case_allowed(case, args.operation):
            errors.append(f"case {case['id']}: operation '{args.operation}' not permitted")
    if args.custodian_release and (sealed or doc.get("tier") == "sealed"):
        errors += corpus_v1.pilot_errors(doc.get("pilot"))
    if errors:
        for e in errors:
            print(f"manifest error: {e}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error("output directory must be empty")

    try:
        detected = corpus_v1.actual_toolchain()
    except Exception:
        detected = {}
    toolchain = {"declared": doc.get("toolchain", {}), "detected": detected,
                 "implementation_sha256": _implementation_hashes(),
                 "git_commit": _git_commit(),
                 "cpu": {"model": platform.processor() or platform.machine(),
                         "logical_cores": os.cpu_count(),
                         "numpy_threads": os.environ.get("OMP_NUM_THREADS", "unset")}}
    (out / "protocol.json").write_text(json.dumps({
        "manifest_sha256": corpus_v1.manifest_digest(doc),
        "dataset_version": doc["dataset_version"], "tier": doc["tier"],
        "toolchain": toolchain, "rng": doc.get("rng"),
        "numeric_tolerance": doc.get("numeric_tolerance", 1e-6),
        "metric": "fixed-gain and scale-aligned waveform SDR, true-mask SDR; no time shift",
        "argv": _safe_argv()}, indent=2))

    rows, sealed_rows = [], []
    for case in sorted(view["cases"], key=lambda c: c["id"]):
        row = run_case(view, root, case, methods,
                       doc.get("numeric_tolerance", 1e-6))
        (sealed_rows if case.get("sealed") else rows).append(row)
        statuses = {k: v["status"] for k, v in row["methods"].items()}
        print(f"{case['id']}: {row['status']} {statuses}", flush=True)
    (out / "results.json").write_text(json.dumps(
        {"excluded_sealed_count": len(sealed) if not args.custodian_release else 0,
         "cases": rows}, indent=2, allow_nan=False))
    if sealed_rows:
        # sealed metrics never mix into a dev report
        (out / "results_sealed.json").write_text(
            json.dumps({"cases": sealed_rows}, indent=2, allow_nan=False))
    return int(any(r["status"] == "error" for r in rows + sealed_rows))


if __name__ == "__main__":
    raise SystemExit(main())
