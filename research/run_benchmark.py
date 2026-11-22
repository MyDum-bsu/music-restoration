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
from pathlib import Path
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
                row["methods"][name] = {"status": "error", "error": str(exc)[:300]}
    except Exception as exc:
        row["methods"] = {name: {"status": "error", "error": str(exc)[:300]}
                          for name in methods}
        row["status"] = "error"
        row["error"] = str(exc)[:300]
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
    errors = corpus_v1.validate_manifest(doc, check_hashes_root=root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for case in doc.get("cases", []):
        if not corpus_v1.case_allowed(case, args.operation):
            errors.append(f"case {case['id']}: operation '{args.operation}' not permitted")
    if doc.get("tier") == "sealed" and not args.custodian_release:
        errors.append("sealed manifest requires --custodian-release")
    if errors:
        for e in errors:
            print(f"manifest error: {e}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error("output directory must be empty")

    toolchain = dict(doc.get("toolchain", {}))
    toolchain["implementation_sha256"] = _implementation_hashes()
    (out / "protocol.json").write_text(json.dumps({
        "manifest_sha256": corpus_v1.manifest_digest(doc),
        "dataset_version": doc["dataset_version"], "tier": doc["tier"],
        "toolchain": toolchain, "rng": doc.get("rng"),
        "numeric_tolerance": doc.get("numeric_tolerance", 1e-6),
        "metric": "fixed-gain and scale-aligned waveform SDR, true-mask SDR; no time shift",
        "argv": sys.argv}, indent=2))

    rows, excluded = [], []
    for case in sorted(doc["cases"], key=lambda c: c["id"]):
        if case.get("sealed") and not args.custodian_release:
            excluded.append(case["id"])  # sealed never enters a dev report
            continue
        row = run_case(doc, root, case, methods,
                       doc.get("numeric_tolerance", 1e-6))
        rows.append(row)
        tmp = out / "results.json.tmp"
        tmp.write_text(json.dumps({"excluded_sealed": sorted(excluded),
                                   "cases": rows}, indent=2, allow_nan=False))
        tmp.replace(out / "results.json")
        statuses = {k: v["status"] for k, v in row["methods"].items()}
        print(f"{case['id']}: {row['status']} {statuses}", flush=True)
    return int(any(r["status"] == "error" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
