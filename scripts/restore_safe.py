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
from rnext.dsp import limit
from rnext.dsp.selection import measure, output_violations, compare
from rnext.dsp.selection import release_target as compute_release_target
from rnext.dsp.repair import select_restoration
from scripts.restore_per_track import write_24
from scripts import audio_measure


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


def _loudness_pass(x, y, dest, report, sr):
    """Opt-in release loudness: gain under the plan ceiling, then full re-QC."""
    lufs, _tp = audio_measure.lufs_tp(dest)
    plan_gain = {"measured_lufs": lufs, "target_lufs": report["loudness_target"]}
    report["loudness_plan"] = plan_gain
    if lufs is None:
        raise ValueError("loudness pass: LUFS is unmeasurable for the encoded candidate")
    delta = float(report["loudness_target"]) - lufs
    if abs(delta) <= 0.2:
        plan_gain["applied"] = False
        return y
    plan_gain["gain_db"] = round(delta, 2)
    plan_gain["applied"] = True
    # The plan's own ceiling; the declip branch records no policy ceiling and
    # repairs to -1.0 dBFS by construction.
    ceiling = float((report.get("policy") or {}).get("target_tp_dbfs", -1.0))
    y2 = limit.limit_track((y * 10 ** (delta / 20.)).astype(np.float32), sr,
                           ceiling_db=ceiling)
    write_24(y2, sr, dest, 0)
    encoded, errors = _encoded_qc(x, y2, dest, report, sr)
    report["encoded_metrics"] = encoded
    if errors:
        raise ValueError("loudness pass violated output constraints: "
                         + ", ".join(errors))
    return y2


def process(source, dest, mode="auto", clip_threshold=None, release_target=None,
            loudness_target=None):
    info = F.ffprobe_audio(str(source))
    if info["ch"] not in (1, 2):
        raise ValueError("mono/stereo input required; multichannel conversion must be explicit")
    x, sr = F.decode_f32(str(source), strict=True, channels=None)
    if abs(len(x) / sr - info["dur"]) > max(1e-5, 2 / sr):
        raise ValueError("decoded duration differs from container")
    y, report = select_restoration(x, sr, mode=mode, bits=info["bps"] or 24,
                                  clip_threshold=clip_threshold,
                                  release_target=release_target)
    if release_target is not None:
        report["release_target"] = release_target
    if report["status"] == "processed_candidate":
        write_24(y, sr, dest, 0)
        encoded, errors = _encoded_qc(x, y, dest, report, sr)
        report["encoded_metrics"] = encoded
        if errors:
            report["violations"] += errors
            report["status"] = "qc_fallback_original"
            report["restoration_state"] = "encoded_candidate_rejected"
            report["needs_review"] = True
            report["rest"] = report["orig"]
            report["comparison"] = compare(report["orig"], report["orig"], report["plan"]["target"])
    if report["status"] == "processed_candidate" and loudness_target is not None:
        report["loudness_target"] = loudness_target
        try:
            y = _loudness_pass(x, y, dest, report, sr)
        except ValueError as exc:
            report["violations"] += [str(exc)]
            report["status"] = "qc_fallback_original"
            report["restoration_state"] = "encoded_candidate_rejected"
            report["needs_review"] = True
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
    ap.add_argument("--loudness-target", type=float, default=None,
                    help="opt-in release loudness in LUFS: gain the accepted candidate "
                         "under the plan's true-peak ceiling and re-run the full encoded QC")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    files = [src] if src.is_file() else sorted(src.glob("*.flac"))
    if not files or any(p.suffix.lower() != ".flac" for p in files):
        ap.error("provide a FLAC file or a directory containing FLAC tracks")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        ap.error("output must be an empty directory")

    # Release-level dynamics prior: the least-compressed third carries the
    # release's own ceiling. Advisory only — a file that cannot be decoded
    # here fails loudly again in its own per-track run.
    release_target = None
    if len(files) > 1:
        measures = []
        for source in files:
            try:
                sx, ssr = F.decode_f32(str(source), strict=True, channels=None)
                measures.append(measure(sx, ssr))
            except Exception as exc:  # noqa: BLE001 — the per-track run reports it
                print(f"release stats skip {source.name}: {exc}", flush=True)
        release_target = compute_release_target(measures)
        print(f"release target: {release_target}", flush=True)

    rows = []
    for source in files:
        try:
            row = process(source, out / source.name, mode=args.mode,
                          clip_threshold=args.clip_threshold,
                          release_target=release_target,
                          loudness_target=args.loudness_target)
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
