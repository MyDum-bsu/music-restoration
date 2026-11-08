#!/usr/bin/env python3
"""Evaluate explicitly paired, hash-pinned clipping cases against known clean PCM.

The manifest names exact files and expected input SDR. No directory glob chooses
an edition or severity. Native mono/stereo channels are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rnext import features as F
from rnext.dsp.repair import select_restoration


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(spec):
    path = Path(spec["path"])
    if _digest(path) != spec["sha256"]:
        raise ValueError(f"input hash mismatch: {path}")
    info = F.ffprobe_audio(str(path))
    if info["ch"] not in (1, 2):
        raise ValueError("benchmark supports mono/stereo sources only")
    pcm, rate = F.decode_f32(str(path), strict=True, channels=None)
    return pcm, rate, info


def _sdr(signal_energy, error_energy):
    if error_energy < 1e-20:
        return None  # perfect/quantization-limited match, not a JSON Infinity
    return float(10 * np.log10(max(signal_energy, 1e-20) / error_energy))


def reconstruction_metrics(clean, candidate):
    clean, candidate = clean.astype(np.float64), candidate.astype(np.float64)
    signal = float(np.sum(clean ** 2))
    gain = float(np.sum(clean * candidate) / max(1e-20, np.sum(candidate ** 2)))
    error = float(np.sum((clean - gain * candidate) ** 2))
    return {"scale_aligned_sdr_db": _sdr(signal, error), "gain_to_clean": gain,
            "relative_error": float(np.sqrt(error / max(1e-20, signal))),
            "fixed_gain_sdr_db": _sdr(signal, float(np.sum((clean - candidate) ** 2)))}


def evaluate_case(case, out):
    clean, sr, clean_info = _read(case["clean"])
    degraded, rate, info = _read(case["degraded"])
    if sr != rate or clean.shape != degraded.shape:
        raise ValueError("reference and input must have identical sample rate and frame count")
    before = reconstruction_metrics(clean, degraded)
    expected = case.get("expected_input_sdr_db")
    actual = before["fixed_gain_sdr_db"]
    if expected is not None and (actual is None or abs(actual - expected) > .1):
        raise ValueError(f"wrong degradation/edition: expected input SDR {expected}, got {actual}")
    started = time.monotonic()
    restored, report = select_restoration(degraded, sr, mode=case.get("mode", "auto"),
                                          bits=info["bps"] or 24)
    measured = reconstruction_metrics(clean, restored)
    references = {}
    for name, spec in case.get("references", {}).items():
        reference, reference_sr, _ = _read(spec)
        if reference_sr != sr or reference.shape != clean.shape:
            raise ValueError(f"unaligned reference: {name}")
        references[name] = reconstruction_metrics(clean, reference)
    target = out / f"{case['id']}.wav"
    wavfile.write(target, sr, restored)
    delta = (None if before["scale_aligned_sdr_db"] is None or measured["scale_aligned_sdr_db"] is None
             else measured["scale_aligned_sdr_db"] - before["scale_aligned_sdr_db"])
    return {"id": case["id"], "split": case.get("split"), "input": case,
            "input_format": info, "clean_format": clean_info,
            "native_channels_preserved": True, "status": report["status"],
            "seconds": time.monotonic() - started, "before": before, "after": measured,
            "delta_sdr_db": delta, "references": references, "report": report,
            "output": str(target), "output_sha256": _digest(target),
            "listening_verdict": "not_evaluated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids):
        parser.error("case IDs must be unique simple names")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("output directory must be empty")
    implementation = {str(path.relative_to(REPO)): _digest(path) for path in
                      [REPO / "rnext/dsp" / (name + ".py")
                       for name in ("sparse", "predictive", "repair", "selection", "contour", "declip")]
                      + [REPO / "rnext/features.py", Path(__file__).resolve()]}
    (out / "protocol.json").write_text(json.dumps({"manifest": manifest,
                                                   "implementation_sha256": implementation,
                                                   "metric": "scale-aligned waveform SDR; no time shift",
                                                   "argv": sys.argv}, indent=2))
    rows = []
    for case in cases:
        try:
            row = evaluate_case(case, out)
        except Exception as exc:
            row = {"id": case["id"], "status": "error", "error": str(exc)}
        rows.append(row)
        temp = out / "results.json.tmp"
        temp.write_text(json.dumps(rows, indent=2, allow_nan=False))
        temp.replace(out / "results.json")
        print(f"{case['id']}: {row['status']}; delta SDR {row.get('delta_sdr_db')}", flush=True)
    return int(any(row["status"] == "error" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
