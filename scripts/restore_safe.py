#!/usr/bin/env python3
"""Restore mono/stereo FLAC with strict decode and encoded QC."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rnext import features as F
from rnext.dsp import limit
from rnext.dsp.repair import (
    DECLIP_ENGINES,
    finalize_declip_candidate,
    propose_declip_candidate,
    select_restoration,
)
from rnext.dsp.selection import compare, measure, output_violations
from rnext.dsp.selection import release_target as compute_release_target
from rnext.dsp.stereo_safety import evaluate_gapless_boundary, plan_album_headroom
from scripts import audio_measure
from scripts.restore_one import encode_flac

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


def _encode_to(y, sr, path):
    """Encode to `path` (a staging name); the caller commits via os.replace."""
    wav_tmp = path.parent / f"__tmp_{uuid.uuid4().hex}.wav"
    try:
        rng = np.random.default_rng(0xC0FFEE)
        dith = rng.random(y.shape) + rng.random(y.shape) - 1.0
        yq = np.clip(np.round(y * 8388608.0 + dith), -8388608, 8388607).astype(np.int32)
        encode_flac(yq, sr, path, wav_tmp, bits=24)
    finally:
        wav_tmp.unlink(missing_ok=True)


def _copy_atomic(source, dest):
    """Byte-for-byte fallback copy via unique temp and atomic commit."""
    tmp = _tmp_path(dest)
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _encoded_qc(x, y, staged, report, source_sr, ceiling):
    decoded, out_sr = F.decode_f32(str(staged), strict=True, channels=None)
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


def _loudness_pass(x, y, staged, report, sr, ceiling):
    """Opt-in release loudness: gain under the same effective ceiling, re-encode
    on the staging file, then report what was actually achieved. Never adds
    gain above the TP ceiling and never hides an unreached target behind a
    valid FLAC."""
    loud = {"requested_lufs": float(report["loudness_target"]),
            "tolerance_lu": LOUDNESS_TOLERANCE_LU,
            "tp_ceiling_dbfs": ceiling}
    report["loudness"] = loud
    lufs, _tp = audio_measure.lufs_tp(staged)
    if lufs is None:
        loud.update(result="not_measured",
                    reason="lufs_unmeasurable_on_encoded_candidate")
        return y
    loud["pre_gain_lufs"] = lufs
    delta = float(report["loudness_target"]) - lufs
    loud["requested_gain_db"] = round(delta, 2)
    # Total delivered gain = this pass + the gain the repair already applied
    # (the declip branch attenuates for headroom by construction).
    repair_gain = float((report.get("repair") or {}).get("output_gain_db") or 0.0)
    if abs(delta) <= LOUDNESS_TOLERANCE_LU:
        loud.update(result="met", applied_gain_db=0.0, achieved_lufs=lufs,
                    total_applied_gain_db=round(repair_gain, 2))
        return y
    y2 = limit.limit_track((y * 10 ** (delta / 20.)).astype(np.float32), sr,
                           ceiling_db=ceiling)
    staged2 = _tmp_path(staged.with_name(staged.name + ".loud"))
    try:
        _encode_to(y2, sr, staged2)
        encoded, errors = _encoded_qc(x, y2, staged2, report, sr, ceiling)
        report["encoded_metrics"] = encoded
        if errors:
            raise ValueError("loudness pass violated output constraints: "
                             + ", ".join(errors))
        achieved, _ = audio_measure.lufs_tp(staged2)
        if achieved is None:
            loud.update(result="not_measured",
                        reason="lufs_unmeasurable_after_gain")
            return y
        applied = round(achieved - lufs, 2)
        loud.update(achieved_lufs=achieved, applied_gain_db=applied,
                    total_applied_gain_db=round(applied + repair_gain, 2))
        if abs(achieved - float(report["loudness_target"])) <= LOUDNESS_TOLERANCE_LU:
            loud["result"] = "met"
        elif delta > 0 and applied < delta - LOUDNESS_TOLERANCE_LU:
            loud.update(result="limited_by_peak",
                        reason=(f"true-peak ceiling {ceiling} dBTP admits "
                                f"{applied} dB of {round(delta, 2)} dB requested"))
        else:
            loud.update(result="not_met",
                        reason=(f"applied gain {applied} dB was not limited by "
                                f"the peak ceiling yet achieved {achieved} LUFS, "
                                f"outside ±{LOUDNESS_TOLERANCE_LU} LU of the target"))
        os.replace(staged2, staged)
    finally:
        staged2.unlink(missing_ok=True)
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


def _deliver_track(source, dest, x, sr, info, source_sha256, y, report,
                   loudness_target=None):
    """Run the single delivery QC contract, atomic commit, and output hashing."""
    ceiling = _effective_ceiling(report)
    report["delivery_qc"] = {"tp_ceiling_dbfs": ceiling,
                             "tp_tolerance_db": TP_TOLERANCE_DB,
                             "tp_measurer": "polyphase_4x"}
    # Everything lands on a unique staging name first; `dest` is committed by a
    # single os.replace only after encode, decode-back QC and the loudness pass
    # have all succeeded — an interrupted run can never leave an unverified
    # candidate at the output path.
    staged = _tmp_path(dest)
    try:
        if report["status"] == "processed_candidate":
            _encode_to(y, sr, staged)
            encoded, errors = _encoded_qc(x, y, staged, report, sr, ceiling)
            report["encoded_metrics"] = encoded
            if errors:
                report["rejected_candidate_metrics"] = report.pop("encoded_metrics")
                _reject_candidate(report, errors)
            elif loudness_target is not None:
                report["loudness_target"] = loudness_target
                try:
                    y = _loudness_pass(x, y, staged, report, sr, ceiling)
                except ValueError as exc:
                    rejected = report.pop("encoded_metrics", None)
                    if rejected is not None:
                        report["rejected_candidate_metrics"] = rejected
                    _reject_candidate(report, str(exc))
        if report["status"] == "processed_candidate":
            os.replace(staged, dest)
        else:
            _copy_atomic(source, dest)
    finally:
        staged.unlink(missing_ok=True)
    # delivery_qc describes the file actually handed over — measured on the
    # committed output, never on a rejected attempt.
    delivered, dsr = F.decode_f32(str(dest), strict=True, channels=None)
    delivered_metrics = measure(delivered, dsr)
    output_sha256 = _sha256(dest)
    report["delivery_qc"].update(
        delivered_tp_dbfs=delivered_metrics["tp_dbfs"],
        delivered_clip_samples=delivered_metrics["clip_samples"],
        delivered_dr=delivered_metrics["dr"],
        delivered_sha256=output_sha256)
    if (report["status"] != "processed_candidate"
            and report["orig"]["tp_dbfs"] > ceiling + TP_TOLERANCE_DB):
        report["delivery_qc"]["source_exceeds_ceiling"] = True
    # restoration_evidence records the attempt itself: what was tried, why it
    # was kept or rejected — separate from the delivered-file facts above.
    evidence = {"mode": report.get("mode"), "chosen": report.get("chosen"),
                "restoration_state": report.get("restoration_state"),
                "reason": report.get("reason"),
                "violations": list(report.get("violations", [])),
                "repair": report.get("repair"),
                "rejected_candidate_metrics":
                    report.get("rejected_candidate_metrics")}
    report["restoration_evidence"] = {k: v for k, v in evidence.items()
                                    if v is not None}
    # The output is accepted only together with this report: hash the committed
    # file so a stale/orphan artifact can never pass as the verified result.
    report.update(source=str(source.resolve()), output=str(dest.resolve()),
                  source_sha256=source_sha256,
                  output_sha256=output_sha256,
                  source_format=info, provenance="not_verified_by_container",
                  output_claim="candidate enhancement; listening evaluation pending")
    if _sha256(source) != source_sha256:
        report["source_mutated_during_run"] = True
        raise ValueError("source hash changed during processing")
    return report


def process(source, dest, mode="auto", clip_threshold=None, release_target=None,
            loudness_target=None, declip_engine="current_snapshot",
            album_gain_db=None):
    info = F.ffprobe_audio(str(source))
    if info["ch"] not in (1, 2):
        raise ValueError("mono/stereo input required; multichannel conversion must be explicit")
    source_sha256 = _sha256(source)
    x, sr = F.decode_f32(str(source), strict=True, channels=None)
    if abs(len(x) / sr - info["dur"]) > max(1e-5, 2 / sr):
        raise ValueError("decoded duration differs from container")
    opts = {
        "mode": mode,
        "bits": info["bps"] or 24,
        "clip_threshold": clip_threshold,
        "release_target": release_target,
    }
    if declip_engine != "current_snapshot":
        opts["declip_engine"] = declip_engine
    if album_gain_db is not None:
        opts["album_gain_db"] = album_gain_db
    y, report = select_restoration(x, sr, **opts)
    if release_target is not None:
        report["release_target"] = release_target
    return _deliver_track(source, dest, x, sr, info, source_sha256, y, report,
                          loudness_target=loudness_target)


def _process_album_declip(files, out, *, clip_threshold=None,
                          loudness_target=None,
                          declip_engine="current_snapshot"):
    """Two-pass Preserve album declip with unified attenuation-only headroom.

    Pass 1 computes pre-gain declip proposals per track. `plan_album_headroom`
    decides whether a single unified album gain is compatible with immutable
    kept tracks; if mixed, kept tracks stay byte-for-byte unchanged and the
    album is explicitly flagged `needs_review=True`, `auto_album_delivery=False`.
    Pass 2 delivers each track via the identical atomic FLAC + QC contract and
    verifies adjacent gapless track boundaries jointly.
    """
    if loudness_target is not None:
        raise ValueError(
            "album headroom coordination cannot be combined with per-track loudness_target"
        )
    staged_items = []
    proposals = []
    for source in files:
        try:
            info = F.ffprobe_audio(str(source))
            if info["ch"] not in (1, 2):
                raise ValueError("mono/stereo input required; multichannel conversion must be explicit")
            source_sha256 = _sha256(source)
            x, sr = F.decode_f32(str(source), strict=True, channels=None)
            if abs(len(x) / sr - info["dur"]) > max(1e-5, 2 / sr):
                raise ValueError("decoded duration differs from container")
            repaired, mask, prop = propose_declip_candidate(
                x, sr, bits=info["bps"] or 24, clip_threshold=clip_threshold,
                declip_engine=declip_engine)
            staged_items.append({"source": source, "info": info,
                                 "source_sha256": source_sha256,
                                 "x": x, "sr": sr, "repaired": repaired,
                                 "mask": mask, "proposal": prop, "error": None})
            proposals.append({"track_id": source.name, "status": prop["status"],
                              "pre_gain_tp_dbfs": prop.get("pre_gain_tp_dbfs")})
        except Exception as exc:  # noqa: BLE001 — per-track error recorded in batch report
            staged_items.append({"source": source, "error": str(exc)})
            proposals.append({"track_id": source.name, "status": "error",
                              "pre_gain_tp_dbfs": None})

    plan = plan_album_headroom(proposals, ceiling_db=PRESERVE_CEILING_DBFS)
    rows = []
    for item in staged_items:
        source = item["source"]
        if item["error"] is not None:
            row = {"source": str(source), "status": "error", "error": item["error"]}
        else:
            try:
                tid = source.name
                prop = item["proposal"]
                gain_db = plan["per_track_gain_db"].get(tid, 0.0)
                if plan["policy"] == "rejected_excessive_album_attenuation":
                    prop = {**prop, "status": "qc_fallback_original",
                            "reason": plan["reason"]}
                    y, report = finalize_declip_candidate(
                        item["x"], item["repaired"], item["mask"], prop, item["sr"])
                else:
                    use_gain = gain_db if prop["status"] == "processed_candidate" else None
                    y, report = finalize_declip_candidate(
                        item["x"], item["repaired"], item["mask"], prop, item["sr"],
                        album_gain_db=use_gain, ceiling_db=PRESERVE_CEILING_DBFS)
                if plan["needs_review"]:
                    report["needs_review"] = True
                report["album_headroom"] = {
                    "policy": plan["policy"],
                    "compatible": plan["compatible"],
                    "auto_album_delivery": plan["auto_album_delivery"],
                    "album_gain_db": plan["album_gain_db"],
                    "applied_track_gain_db": float(gain_db),
                    "max_relative_level_shift_db": plan["max_relative_level_shift_db"],
                    "reason": plan["reason"],
                }
                row = _deliver_track(source, out / source.name, item["x"], item["sr"],
                                     item["info"], item["source_sha256"], y, report,
                                     loudness_target=loudness_target)
            except Exception as exc:  # noqa: BLE001 — per-track error recorded in batch report
                row = {"source": str(source), "status": "error", "error": str(exc)}
        rows.append(row)
        tmp = _tmp_path(out / "report.json")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
        tmp.replace(out / "report.json")
        print(f"{source.name}: {row['status']}", flush=True)

    # Joint gapless boundary verification on consecutive delivered stereo tracks.
    boundaries = []
    for i in range(len(staged_items) - 1):
        a, b = staged_items[i], staged_items[i + 1]
        if a["error"] is not None or b["error"] is not None:
            continue
        if a["sr"] != b["sr"] or a["info"]["ch"] != 2 or b["info"]["ch"] != 2:
            continue
        dest_a, dest_b = out / a["source"].name, out / b["source"].name
        if not (dest_a.exists() and dest_b.exists()):
            continue
        da, sr_a = F.decode_f32(str(dest_a), strict=True, channels=None)
        db, sr_b = F.decode_f32(str(dest_b), strict=True, channels=None)
        if sr_a == sr_b == a["sr"] and da.shape == a["x"].shape and db.shape == b["x"].shape:
            res = evaluate_gapless_boundary(
                a["x"], b["x"], da, db, a["sr"],
                mask_a=a["mask"], mask_b=b["mask"],
            )
            boundaries.append({
                "from_track": a["source"].name,
                "to_track": b["source"].name,
                **res,
            })

    album_safety = {
        "schema": "album-safety/v1",
        "declip_engine": declip_engine,
        "headroom_plan": plan,
        "gapless_boundaries": boundaries,
        "all_boundaries_passed": all(b["passed"] for b in boundaries),
    }
    tmp_as = _tmp_path(out / "album_safety.json")
    tmp_as.write_text(json.dumps(album_safety, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp_as.replace(out / "album_safety.json")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("auto", "dynamics", "declip"), default="auto")
    ap.add_argument("--declip-engine", choices=DECLIP_ENGINES, default="current_snapshot",
                    help="hard-declipping engine (default current_snapshot; aspade_v2 is the E4 freeze candidate)")
    ap.add_argument("--album-headroom", action="store_true",
                    help="coordinate attenuation-only unified album gain and joint gapless checks in declip mode")
    ap.add_argument("--clip-threshold", type=float, help="explicit symmetric hard-clip level in PCM units")
    ap.add_argument("--loudness-target", type=float, default=None,
                    help="opt-in release loudness in LUFS: gain the accepted candidate "
                         "under the plan's true-peak ceiling and re-run the full encoded QC")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    files = [src] if src.is_file() else sorted(src.glob("*.flac"))
    if not files or any(p.suffix.lower() != ".flac" for p in files):
        ap.error("provide a FLAC file or a directory containing FLAC tracks")
    if args.album_headroom:
        if args.mode != "declip":
            ap.error("--album-headroom requires --mode declip")
        if len(files) < 2:
            ap.error("--album-headroom requires a directory with at least 2 FLAC tracks")
        if args.loudness_target is not None:
            ap.error(
                "--album-headroom cannot be combined with --loudness-target "
                "(per-track loudness pass overrides unified album headroom)"
            )
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        ap.error("output must be an empty directory")

    if args.album_headroom:
        rows = _process_album_declip(
            files, out, clip_threshold=args.clip_threshold,
            loudness_target=None,
            declip_engine=args.declip_engine)
        return 1 if any(r["status"] == "error" for r in rows) else 0

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
                          loudness_target=args.loudness_target,
                          declip_engine=args.declip_engine)
        except Exception as exc:  # noqa: BLE001 — CLI boundary records per-track error in batch report
            row = {"source": str(source), "status": "error", "error": str(exc)}
        rows.append(row)
        tmp = _tmp_path(out / "report.json")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
        tmp.replace(out / "report.json")
        print(f"{source.name}: {row['status']}", flush=True)
    return 1 if any(r["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
