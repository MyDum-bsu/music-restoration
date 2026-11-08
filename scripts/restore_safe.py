#!/usr/bin/env python3
"""Restore stereo FLAC with an identity candidate, strict decode and encoded QC."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rnext import features as F
from rnext.dsp.selection import select_track, measure, output_violations, compare
from scripts.restore_per_track import write_24


def process(source, dest):
    info = F.ffprobe_audio(str(source))
    if info["ch"] != 2:
        raise ValueError("stereo input required; channel conversion must be explicit")
    x, sr = F.decode_f32(str(source), strict=True)
    if abs(len(x) / sr - info["dur"]) > max(1e-5, 2 / sr):
        raise ValueError("decoded duration differs from container")
    y, report = select_track(x, sr)
    if report["status"] == "processed_candidate":
        write_24(y, sr, dest, 0)
        decoded, out_sr = F.decode_f32(str(dest), strict=True)
        encoded = measure(decoded, out_sr)
        errors = output_violations(x, decoded, out_sr, report["orig"], encoded)
        if out_sr != sr:
            errors.append("sample_rate_changed")
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
            row = process(source, out / source.name)
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
