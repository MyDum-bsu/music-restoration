"""Conservative candidate selection; metrics are safeguards, not listening scores."""
from dataclasses import replace
import json

import numpy as np

from rnext import features as F
from rnext.dsp.contour import true_peak_dbfs
from rnext.dsp.pipeline import Policy, restore_track


def measure(x, sr):
    if x.ndim != 2 or x.shape[1] not in (1, 2) or not x.size or not np.isfinite(x).all():
        raise ValueError("expected nonempty finite mono/stereo PCM")
    dr, exact, _, peak, rms = F.tt_dr(x, sr)
    bands = np.asarray(F.band_rms_db(x, sr))
    return {"dr": int(dr), "dr_exact": float(exact), "crest_db": F.crest_db(x),
            "rms_tt": float(rms), "peak_db": float(peak),
            "tp_dbfs": float(true_peak_dbfs(x, sr)),
            "clip_samples": int(np.count_nonzero(np.abs(x) >= 0.9999)),
            "spikes": int(np.count_nonzero(np.max(np.abs(np.diff(x, axis=0)), axis=1) > .95)),
            "shape": (bands - bands.mean()).tolist()}


def material_plan(m, duration, allow_sparse=False):
    """Class-table priors; quiet/high-DR material retains an identity candidate."""
    if duration < 6 or m["rms_tt"] < -70:
        return {"class": "short_or_silent", "target": 0.0}
    if m["dr"] >= 11:
        return {"class": "healthy", "target": 0.0}
    if m["rms_tt"] <= -13:
        return {"class": "sparse", "target": .8 if allow_sparse else 0.,
                "reason": "quiet sparse material needs an explicit dynamics request"}
    return {"class": "crushed" if m["dr"] <= 7 else "mid",
            "target": 5.3 if m["dr"] <= 7 else 2.3}


def candidate_policies(plan):
    # A ceiling only attenuates. No automatic peak/loudness boost or EQ match.
    out = {"original": None}
    if plan["target"] == 0:
        return out
    # Rumble removal needs a diagnosis; avoid filtering every clean master.
    base = Policy(auto_policy=False, hpf_hz=0., declip=False, normalize_tp=False, target_tp_dbfs=-1.0)
    out["gentle"] = replace(base, k=.75, gain_min_db=-4., gain_max_db=4.)
    out["moderate"] = replace(base, k=.5, gain_min_db=-6., gain_max_db=6.)
    out["adaptive"] = replace(base, mode="expand", adaptive=True,
                              target_dcrest=plan["target"])
    out["adaptive_neutral"] = replace(out["adaptive"], bass_scale=1.)
    return out


def compare(m, orig, target):
    delta = m["crest_db"] - orig["crest_db"]
    delta_dr = m["dr_exact"] - orig["dr_exact"]
    shape_in = np.asarray(orig["shape"])
    active = shape_in >= shape_in.max() - 60.
    change = (np.asarray(m["shape"]) - shape_in)[active]
    # Quantization noise in empty bands is not a tonal change. Re-center on
    # source-active bands so unrelated noise floors cannot alter the gain reference.
    change -= change.mean()
    shape = float(np.sqrt(np.mean(change ** 2)))
    # A single boundary peak can raise global crest while local dynamics stay
    # unchanged. Use block-based TT DR for the objective; report crest separately.
    return {"delta_crest": delta, "delta_dr": delta_dr, "shape_rms": shape,
            "score": abs(delta_dr - target) + .8 * shape + max(0., delta_dr - target - 1.) * 2}


def output_violations(x, y, sr, orig, measured):
    errors = []
    if x.shape != y.shape:
        errors.append("sample_count_or_channels_changed")
    if not np.isfinite(y).all():
        errors.append("nonfinite_pcm")
    if measured["tp_dbfs"] > -.95:
        errors.append("true_peak_exceeds_minus_1")
    if measured["clip_samples"]:
        errors.append("clipped_output")
    if measured["spikes"] > orig["spikes"]:
        errors.append("additional_sample_jumps")
    if compare(measured, orig, 0)["shape_rms"] > 1.5:
        errors.append("band_shape_change_over_1_5_db")
    if measured["dr_exact"] < orig["dr_exact"] - .25:
        errors.append("local_dr_regression")
    if x.shape == y.shape:
        n = min(len(x), max(1, int(.05 * sr)))
        edge_x = max(float(np.max(np.abs(x[:n]))), float(np.max(np.abs(x[-n:]))), 1e-6)
        edge_y = max(float(np.max(np.abs(y[:n]))), float(np.max(np.abs(y[-n:]))), 1e-6)
        gain = 10 ** ((orig["rms_tt"] - measured["rms_tt"]) / 20.)
        if 20 * np.log10(edge_y * gain / edge_x) > 3.:
            errors.append("boundary_peak_growth_over_3_db_at_matched_rms")
    return errors


def select_track(x, sr, allow_sparse=False):
    orig = measure(x, sr)
    plan = material_plan(orig, len(x) / sr, allow_sparse=allow_sparse)
    policies = candidate_policies(plan)
    # Spread windows across the track; include short files without seeking past EOF.
    length = min(len(x), 15 * sr)
    starts = sorted({int(f * (len(x) - length)) for f in (.15, .5, .85)})
    trials = []
    for name, pol in policies.items():
        windows = []
        for start in starts:
            ex = x[start:start + length]
            eo = measure(ex, sr)
            ye = ex if pol is None else restore_track(ex, sr, pol, keep_arrays=False)
            em = measure(ye, sr)
            cmp = compare(em, eo, plan["target"])
            cmp["violations"] = [] if pol is None else output_violations(ex, ye, sr, eo, em)
            windows.append(cmp)
        valid = all(not w["violations"] for w in windows)
        trials.append({"name": name, "valid": valid, "windows": windows,
                       "score": float(np.mean([w["score"] for w in windows]))})
    best = min((t for t in trials if t["valid"]), key=lambda t: t["score"])
    pol = policies[best["name"]]
    y = x.copy() if pol is None else restore_track(x, sr, pol, keep_arrays=False)
    rest = measure(y, sr)
    full = compare(rest, orig, plan["target"])
    violations = [] if pol is None else output_violations(x, y, sr, orig, rest)
    # A full-track regression cannot be rescued by good excerpt scores.
    if pol is not None and full["score"] >= plan["target"]:
        violations.append("full_track_no_proxy_benefit")
    status = "kept_original" if pol is None else "processed_candidate"
    if violations:
        y, rest, status = x.copy(), orig, "qc_fallback_original"
    return y, {"status": status, "plan": plan, "chosen": best["name"],
               "policy": None if pol is None else json.loads(pol.to_json()),
               "orig": orig, "rest": rest, "attempted_comparison": full,
               "comparison": compare(rest, orig, plan["target"]),
               "violations": violations, "windows_start_samples": starts,
               "candidates": trials, "listening_verdict": "not_evaluated"}
