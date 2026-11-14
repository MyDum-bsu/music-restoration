"""Choose a defect-specific repair before considering dynamics enhancement."""
import numpy as np

from rnext.dsp.contour import true_peak_dbfs
from rnext.dsp.selection import measure, compare, select_track
from rnext.dsp.sparse import hard_clip_mask, repair_clipping, ReconstructionRejected
from rnext.dsp.predictive import periodic_ar_proposal


def _unchanged(x, common, reason, status="kept_original", repair=None):
    orig = common["orig"]
    result = {**common, "status": status, "rest": orig, "reason": reason,
              "comparison": compare(orig, orig, 0.)}
    if status == "qc_fallback_original":
        result["violations"] = [reason]
    if repair is not None:
        result["repair"] = repair
    return x.copy(), result


def select_restoration(x, sr, mode="auto", bits=24, clip_threshold=None,
                       release_target=None):
    if mode not in ("auto", "dynamics", "declip"):
        raise ValueError("unknown restoration mode")
    if mode == "dynamics":
        return select_track(x, sr, allow_sparse=True, release_target=release_target)
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
