#!/usr/bin/env python3
"""Restore mono/stereo FLAC with strict decode and encoded QC."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rnext import features as F
from rnext.dsp import limit
from rnext.dsp.selection import measure, output_violations, compare
from rnext.dsp.selection import release_target as compute_release_target
from rnext.dsp.repair import select_restoration
from scripts.restore_one import encode_flac
from scripts import audio_measure

# One QC contract for the whole delivery path: the defective Preserve branch
# repairs under -1 dBTP; the dynamics branch keeps its material ceilings.
# The same limit and the same 4x polyphase true-peak measurer apply to the
# in-memory candidate, the decoded encode output and the loudness pass.
PRESERVE_CEILING_DBFS = -1.0
TP_TOLERANCE_DB = 0.05
LOUDNESS_TOLERANCE_LU = 0.2


def _effective_ceiling(report):
    """Effective output TP ceiling in dBFS, derived once from the profile."""
    if report.get("mode") == "declip":
        return PRESERVE_CEILING_DBFS
    plan = report.get("plan") or {}
    ceiling = plan.get("tp_ceiling")
    return PRESERVE_CEILING_DBFS if ceiling is None else float(ceiling)


def _tmp_path(dest):
    return dest.parent / f".{dest.name}.{uuid.uuid4().hex}.tmp"


def _encode_atomic(y, sr, dest):
    """Encode to a unique temp on the same volume, then atomically commit."""
    tmp = _tmp_path(dest)
    wav_tmp = tmp.parent / f"__tmp_{uuid.uuid4().hex}.wav"
    try:
        rng = np.random.default_rng(0xC0FFEE)
        dith = rng.random(y.shape) + rng.random(y.shape) - 1.0
        yq = np.clip(np.round(y * 8388608.0 + dith), -8388608, 8388607).astype(np.int32)
        encode_flac(yq, sr, tmp, wav_tmp, bits=24)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
        wav_tmp.unlink(missing_ok=True)


def _copy_atomic(source, dest):
    """Byte-for-byte fallback copy via unique temp and atomic commit."""
    tmp = _tmp_path(dest)
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _encoded_qc(x, y, dest, report, source_sr, ceiling):
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
        if encoded["tp_dbfs"] > ceiling + TP_TOLERANCE_DB or encoded["clip_samples"]:
            errors.append("encoded_peak_or_clipping_failure")
    else:
        errors = output_violations(x, decoded, out_sr, report["orig"], encoded,
                                   tp_ceiling=ceiling)
    if out_sr != source_sr:
        errors.append("sample_rate_changed")
    return encoded, errors


def _loudness_pass(x, y, dest, report, sr, ceiling):
    """Opt-in release loudness: gain under the same effective ceiling, re-encode,
    then report what was actually achieved. Never adds gain above the TP
    ceiling and never hides an unreached target behind a valid FLAC."""
    loud = {"requested_lufs": float(report["loudness_target"]),
            "tolerance_lu": LOUDNESS_TOLERANCE_LU,
            "tp_ceiling_dbfs": ceiling}
    report["loudness"] = loud
    lufs, _tp = audio_measure.lufs_tp(dest)
    if lufs is None:
        loud.update(result="not_measured",
                    reason="lufs_unmeasurable_on_encoded_candidate")
        return y
    loud["pre_gain_lufs"] = lufs
    delta = float(report["loudness_target"]) - lufs
    if abs(delta) <= LOUDNESS_TOLERANCE_LU:
        loud.update(result="met", applied_gain_db=0.0, achieved_lufs=lufs)
        return y
    loud["requested_gain_db"] = round(delta, 2)
    y2 = limit.limit_track((y * 10 ** (delta / 20.)).astype(np.float32), sr,
                           ceiling_db=ceiling)
    _encode_atomic(y2, sr, dest)
    encoded, errors = _encoded_qc(x, y2, dest, report, sr, ceiling)
    report["encoded_metrics"] = encoded
    if errors:
        raise ValueError("loudness pass violated output constraints: "
                         + ", ".join(errors))
    achieved, _ = audio_measure.lufs_tp(dest)
    if achieved is None:
        loud.update(result="not_measured",
                    reason="lufs_unmeasurable_after_gain")
        return y2
    applied = round(achieved - lufs, 2)
    loud.update(achieved_lufs=achieved, applied_gain_db=applied)
    if abs(achieved - float(report["loudness_target"])) <= LOUDNESS_TOLERANCE_LU:
        loud["result"] = "met"
    else:
        loud.update(result="limited_by_peak",
                    reason=(f"true-peak ceiling {ceiling} dBTP admits "
                            f"{applied} dB of {round(delta, 2)} dB requested"))
    return y2


def _reject_candidate(report, reason=None):
    if reason:
        report["violations"] += [reason] if isinstance(reason, str) else reason
    report["status"] = "qc_fallback_original"
    report["restoration_state"] = "encoded_candidate_rejected"
    report["needs_review"] = True
    report["rest"] = report["orig"]
    report["comparison"] = compare(report["orig"], report["orig"],
                                   report["plan"]["target"])


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process(source, dest, mode="auto", clip_threshold=None, release_target=None,
            loudness_target=None):
    info = F.ffprobe_audio(str(source))
    if info["ch"] not in (1, 2):
        raise ValueError("mono/stereo input required; multichannel conversion must be explicit")
    source_sha256 = _sha256(source)
    x, sr = F.decode_f32(str(source), strict=True, channels=None)
    if abs(len(x) / sr - info["dur"]) > max(1e-5, 2 / sr):
        raise ValueError("decoded duration differs from container")
    y, report = select_restoration(x, sr, mode=mode, bits=info["bps"] or 24,
                                  clip_threshold=clip_threshold,
                                  release_target=release_target)
    if release_target is not None:
        report["release_target"] = release_target
    # The single delivery QC contract: one effective ceiling and tolerance for
    # candidate selection, decoded-encode verification and the loudness pass.
    ceiling = _effective_ceiling(report)
    report["delivery_qc"] = {"tp_ceiling_dbfs": ceiling,
                             "tp_tolerance_db": TP_TOLERANCE_DB,
                             "tp_measurer": "polyphase_4x"}
    if report["status"] == "processed_candidate":
        _encode_atomic(y, sr, dest)
        encoded, errors = _encoded_qc(x, y, dest, report, sr, ceiling)
        report["encoded_metrics"] = encoded
        if errors:
            _reject_candidate(report, errors)
    if report["status"] == "processed_candidate" and loudness_target is not None:
        report["loudness_target"] = loudness_target
        try:
            y = _loudness_pass(x, y, dest, report, sr, ceiling)
        except ValueError as exc:
            _reject_candidate(report, str(exc))
    if report["status"] != "processed_candidate":
        _copy_atomic(source, dest)
    # The output is accepted only together with this report: hash the committed
    # file so a stale/orphan artifact can never pass as the verified result.
    report.update(source=str(source.resolve()), output=str(dest.resolve()),
                  source_sha256=source_sha256,
                  output_sha256=_sha256(dest),
                  source_format=info, provenance="not_verified_by_container",
                  output_claim="candidate enhancement; listening evaluation pending")
    if _sha256(source) != source_sha256:
        report["source_mutated_during_run"] = True
        raise ValueError("source hash changed during processing")
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
