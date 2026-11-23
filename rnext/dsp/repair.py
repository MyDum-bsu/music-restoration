"""Choose a defect-specific repair before considering dynamics enhancement."""
import numpy as np

from rnext.dsp.clip_detector import detect_hard_clip_v2
from rnext.dsp.contour import true_peak_dbfs
from rnext.dsp.declip_methods import declip_aspade
from rnext.dsp.predictive import periodic_ar_proposal
from rnext.dsp.selection import compare, measure, select_track
from rnext.dsp.sparse import ReconstructionRejected, hard_clip_mask, repair_clipping

DECLIP_ENGINES = ("current_snapshot", "aspade_v2")


def _unchanged(x, common, reason, status="kept_original", repair=None):
    orig = common["orig"]
    result = {**common, "status": status, "rest": orig, "reason": reason,
              "comparison": compare(orig, orig, 0.)}
    if status == "qc_fallback_original":
        result["violations"] = [reason]
    if repair is not None:
        result["repair"] = repair
    return x.copy(), result


def propose_declip_candidate(x, sr, *, bits=24, clip_threshold=None,
                             declip_engine="aspade_v2"):
    """Diagnose and reconstruct pre-gain hard-clipped PCM without applying headroom gain.

    Returns (pre_gain_pcm, mask, proposal_dict) where proposal_dict carries
    `status` (`processed_candidate`, `kept_original`, or `qc_fallback_original`),
    `pre_gain_tp_dbfs`, `proposed_gain_db`, `clip_diagnosis`, and `repair`.
    Used by album-level headroom coordination so two-pass album delivery shares
    one exact detector/solver implementation.
    """
    if declip_engine not in DECLIP_ENGINES:
        raise ValueError(f"unknown declip engine: {declip_engine}")
    if declip_engine == "aspade_v2":
        mask, diagnosis = detect_hard_clip_v2(x, sr, bits=bits)
        chosen = "aspade_mit__v2_windowed"
    else:
        mask, diagnosis = hard_clip_mask(x, bits)
        chosen = "consistent_sparse_declip"

    if clip_threshold is not None:
        if not np.isfinite(clip_threshold) or clip_threshold <= 0:
            raise ValueError("clip threshold must be finite and positive")
        unit = max(2. ** (1 - int(bits or 24)), clip_threshold * 1e-7)
        mask = np.abs(x) >= clip_threshold - .51 * unit
        mask &= (x != 0)
        diagnosis.update(explicit_threshold=clip_threshold, masked_samples=int(mask.sum()),
                         mask_source="explicit_threshold")
    else:
        diagnosis["mask_source"] = "windowed_mode_plateau_v2" if declip_engine == "aspade_v2" else "plateau_detector"

    orig = measure(x, sr)
    common = {"mode": "declip", "plan": {"class": "hard_clip", "target": 0.},
              "orig": orig, "clip_diagnosis": diagnosis, "violations": [],
              "listening_verdict": "not_evaluated", "chosen": chosen,
              "declip_engine": declip_engine}
    if not mask.any():
        return x.copy(), mask, {**common, "status": "kept_original",
                                "reason": "no_supported_hard_clip_pattern",
                                "pre_gain_tp_dbfs": None, "proposed_gain_db": 0.0,
                                "repair": None}

    if declip_engine == "aspade_v2":
        try:
            repaired, info = declip_aspade(x, sr, mask)
        except ReconstructionRejected as exc:
            return x.copy(), mask, {**common, "status": "qc_fallback_original",
                                    "reason": str(exc), "pre_gain_tp_dbfs": None,
                                    "proposed_gain_db": 0.0, "repair": None}
    else:
        frame = max(1024, min(16384, 2 ** round(np.log2(4096 * sr / 44100.))))
        proposed, prediction = periodic_ar_proposal(x, mask)
        if prediction["accepted"]:
            repaired, info = proposed, prediction
            info.update(frame=None, iterations_limit=None)
            common["chosen"] = "periodic_float_ar"
        else:
            try:
                repaired, info = repair_clipping(x, sr, mask, frame=frame)
            except ReconstructionRejected as exc:
                return x.copy(), mask, {**common, "status": "qc_fallback_original",
                                        "reason": str(exc), "pre_gain_tp_dbfs": None,
                                        "proposed_gain_db": 0.0, "repair": None}
            info["predictive_confidence"] = {k: v for k, v in prediction.items() if k != "gaps"}

    if not np.array_equal(repaired[~mask], x[~mask]):
        raise ValueError("repair changed reliable samples")
    if not np.any(repaired[mask] != x[mask]):
        return x.copy(), mask, {**common, "status": "kept_original",
                                "reason": "no_reconstructed_sample_change",
                                "pre_gain_tp_dbfs": None, "proposed_gain_db": 0.0,
                                "repair": info}
    raw_tp = true_peak_dbfs(repaired, sr)
    proposed_gain = min(0., -1. - raw_tp)
    if proposed_gain < -24.:
        info.update(pre_gain_tp_dbfs=float(raw_tp), proposed_output_gain_db=float(proposed_gain))
        return x.copy(), mask, {**common, "status": "qc_fallback_original",
                                "reason": "required_headroom_attenuation_over_24_db",
                                "pre_gain_tp_dbfs": float(raw_tp),
                                "proposed_gain_db": float(proposed_gain),
                                "repair": info}
    return repaired, mask, {**common, "status": "processed_candidate",
                            "pre_gain_tp_dbfs": float(raw_tp),
                            "proposed_gain_db": float(proposed_gain),
                            "repair": info}


def finalize_declip_candidate(x, repaired, mask, proposal, sr, *,
                              album_gain_db=None, ceiling_db=-1.0):
    """Apply unified or per-track headroom attenuation to a pre-gain declip proposal."""
    common = {k: proposal[k] for k in ("mode", "plan", "orig", "clip_diagnosis",
                                       "violations", "listening_verdict", "chosen")
              if k in proposal}
    if proposal.get("declip_engine") != "current_snapshot":
        common["declip_engine"] = proposal.get("declip_engine")
    if proposal["status"] != "processed_candidate":
        return _unchanged(x, common, proposal["reason"],
                          status=proposal["status"], repair=proposal.get("repair"))
    info = dict(proposal["repair"])
    raw_tp = float(proposal["pre_gain_tp_dbfs"])
    per_track_gain = min(0., float(ceiling_db) - raw_tp)
    if album_gain_db is not None:
        if album_gain_db > 0.0:
            raise ValueError("Preserve album_gain_db must be attenuation-only (<= 0.0)")
        gain_db = min(per_track_gain, float(album_gain_db))
    else:
        gain_db = per_track_gain
    if gain_db < -24.:
        info.update(pre_gain_tp_dbfs=raw_tp, proposed_output_gain_db=float(gain_db))
        return _unchanged(x, common, "required_headroom_attenuation_over_24_db",
                          status="qc_fallback_original", repair=info)
    y = (repaired.astype(np.float64) * (10. ** (gain_db / 20.))).astype(x.dtype)
    rest = measure(y, sr)
    info.update(pre_gain_tp_dbfs=raw_tp, output_gain_db=float(gain_db),
                changed_masked_samples=int(np.count_nonzero(repaired[mask] != x[mask])))
    if album_gain_db is not None:
        info["album_gain_applied_db"] = float(gain_db)
        info["per_track_proposed_gain_db"] = float(per_track_gain)
    return y, {**common, "status": "processed_candidate", "rest": rest,
               "repair": info, "comparison": compare(rest, proposal["orig"], 0.),
               "comparison_note": "dynamics proxies are descriptive; selection is defect-specific",
               "policy": {"method": info["method"], "frame": info.get("frame"),
                          "iterations": info.get("iterations_limit", info.get("max_iter"))}}


def select_restoration(x, sr, mode="auto", bits=24, clip_threshold=None,
                       release_target=None, *, declip_engine="current_snapshot",
                       album_gain_db=None):
    if mode not in ("auto", "dynamics", "declip"):
        raise ValueError("unknown restoration mode")
    if declip_engine not in DECLIP_ENGINES:
        raise ValueError(f"unknown declip engine: {declip_engine}")
    if mode == "dynamics":
        return select_track(x, sr, allow_sparse=True, release_target=release_target)

    if declip_engine == "aspade_v2":
        repaired, mask, proposal = propose_declip_candidate(
            x, sr, bits=bits, clip_threshold=clip_threshold, declip_engine="aspade_v2")
        if not mask.any() and mode == "auto":
            y, report = select_track(x, sr, release_target=release_target)
            report["clip_diagnosis"] = proposal["clip_diagnosis"]
            report["declip_engine"] = "aspade_v2"
            return y, report
        return finalize_declip_candidate(x, repaired, mask, proposal, sr,
                                         album_gain_db=album_gain_db)

    # Frozen current_snapshot production baseline (preserved unmodified).
    mask, diagnosis = hard_clip_mask(x, bits)
    if clip_threshold is not None:
        if not np.isfinite(clip_threshold) or clip_threshold <= 0:
            raise ValueError("clip threshold must be finite and positive")
        unit = max(2. ** (1 - int(bits or 24)), clip_threshold * 1e-7)
        mask = np.abs(x) >= clip_threshold - .51 * unit
        diagnosis.update(explicit_threshold=clip_threshold, masked_samples=int(mask.sum()),
                         mask_source="explicit_threshold")
    else:
        diagnosis["mask_source"] = "plateau_detector"
    if not mask.any() and mode == "auto":
        y, report = select_track(x, sr, release_target=release_target)
        report["clip_diagnosis"] = diagnosis
        return y, report
    orig = measure(x, sr)
    common = {"mode": "declip", "plan": {"class": "hard_clip", "target": 0.},
              "orig": orig, "clip_diagnosis": diagnosis, "violations": [],
              "listening_verdict": "not_evaluated", "chosen": "consistent_sparse_declip"}
    if not mask.any():
        return _unchanged(x, common, "no_supported_hard_clip_pattern")
    # Window duration stays approximately constant at other native sample rates.
    frame = max(1024, min(16384, 2 ** round(np.log2(4096 * sr / 44100.))))
    proposed, prediction = periodic_ar_proposal(x, mask)
    if prediction["accepted"]:
        repaired, info = proposed, prediction
        info.update(frame=None, iterations_limit=None)
        common["chosen"] = "periodic_float_ar"
    else:
        try:
            repaired, info = repair_clipping(x, sr, mask, frame=frame)
        except ReconstructionRejected as exc:
            return _unchanged(x, common, str(exc), status="qc_fallback_original")
        info["predictive_confidence"] = {k: v for k, v in prediction.items() if k != "gaps"}
    if not np.array_equal(repaired[~mask], x[~mask]):
        raise ValueError("repair changed reliable samples")
    if not np.any(repaired[mask] != x[mask]):
        return _unchanged(x, common, "no_reconstructed_sample_change", repair=info)
    raw_tp = true_peak_dbfs(repaired, sr)
    gain_db = min(0., -1. - raw_tp)
    if album_gain_db is not None:
        if album_gain_db > 0.0:
            raise ValueError("Preserve album_gain_db must be attenuation-only (<= 0.0)")
        gain_db = min(gain_db, float(album_gain_db))
    if gain_db < -24.:
        info.update(pre_gain_tp_dbfs=float(raw_tp), proposed_output_gain_db=float(gain_db))
        return _unchanged(x, common, "required_headroom_attenuation_over_24_db",
                          status="qc_fallback_original", repair=info)
    y = (repaired * 10 ** (gain_db / 20.)).astype(x.dtype)
    rest = measure(y, sr)
    info.update(pre_gain_tp_dbfs=float(raw_tp), output_gain_db=float(gain_db),
                changed_masked_samples=int(np.count_nonzero(repaired[mask] != x[mask])))
    return y, {**common, "status": "processed_candidate", "rest": rest,
               "repair": info, "comparison": compare(rest, orig, 0.),
               "comparison_note": "dynamics proxies are descriptive; selection is defect-specific",
               "policy": {"method": info["method"], "frame": frame,
                          "iterations": info["iterations_limit"]}}
