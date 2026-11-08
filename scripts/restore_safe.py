#!/usr/bin/env python3
"""Restore mono/stereo FLAC with strict decode and encoded QC."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rnext import features as F
from rnext.dsp.selection import measure, output_violations, compare
from rnext.dsp.repair import select_restoration
from scripts.restore_per_track import write_24


def _encoded_qc(x, y, dest, report, source_sr):
    decoded, out_sr = F.decode_f32(str(dest), strict=True, channels=None)
    encoded = measure(decoded, out_sr)
    if report.get("mode") == "declip":
        # Repair may legitimately change source-relative spectrum/DR. Verify
        # that encoding preserved the accepted waveform, not those proxies.
        errors = []
        if decoded.shape != y.shape:
            errors.append("sample_count_or_channels_changed")
        elif float(np.max(np.abs(decoded - y))) > 2. / 8388608:
            errors.append("encoded_waveform_differs_from_accepted_repair")
        if encoded["tp_dbfs"] > -.95 or encoded["clip_samples"]:
            errors.append("encoded_peak_or_clipping_failure")
    else:
        errors = output_violations(x, decoded, out_sr, report["orig"], encoded)
    if out_sr != source_sr:
        errors.append("sample_rate_changed")
    return encoded, errors


def process(source, dest, mode="auto", clip_threshold=None):
    info = F.ffprobe_audio(str(source))
    if info["ch"] not in (1, 2):
        raise ValueError("mono/stereo input required; multichannel conversion must be explicit")
    x, sr = F.decode_f32(str(source), strict=True, channels=None)
    if abs(len(x) / sr - info["dur"]) > max(1e-5, 2 / sr):
        raise ValueError("decoded duration differs from container")
    y, report = select_restoration(x, sr, mode=mode, bits=info["bps"] or 24,
                                  clip_threshold=clip_threshold)
    if report["status"] == "processed_candidate":
        write_24(y, sr, dest, 0)
        encoded, errors = _encoded_qc(x, y, dest, report, sr)
        report["encoded_metrics"] = encoded
        if errors:
            report["violations"] += errors
            report["status"] = "qc_fallback_original"
            report["rest"] = report["orig"]
            report["comparison"] = compare(report["orig"], report["orig"], report["plan"]["target"])
    if report["status"] != "processed_candidate":
        shutil.copy2(source, dest)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    report.update(source=str(source.resolve()), output=str(dest.resolve()),
                  source_sha256=digest.hexdigest(),
                  source_format=info, provenance="not_verified_by_container",
                  output_claim="candidate enhancement; listening evaluation pending")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("auto", "dynamics", "declip"), default="auto")
    ap.add_argument("--clip-threshold", type=float, help="explicit symmetric hard-clip level in PCM units")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    files = [src] if src.is_file() else sorted(src.glob("*.flac"))
    if not files or any(p.suffix.lower() != ".flac" for p in files):
        ap.error("provide a FLAC file or a directory containing FLAC tracks")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        ap.error("output must be an empty directory")
    rows = []
    for source in files:
        try:
            row = process(source, out / source.name, mode=args.mode,
                          clip_threshold=args.clip_threshold)
        except Exception as exc:
            row = {"source": str(source), "status": "error", "error": str(exc)}
        rows.append(row)
        tmp = out / "report.json.tmp"
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
        tmp.replace(out / "report.json")
        print(f"{source.name}: {row['status']}", flush=True)
    return 1 if any(r["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
