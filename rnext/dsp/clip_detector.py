"""Blind hard-clipping detectors for native mono/stereo float PCM.

`detect_hard_clip_v2` addresses four structural blind spots found in the
v1 global-maximum plateau detector (`rnext.dsp.sparse.hard_clip_mask`):

1. Overlapping local windows detect late-onset or sub-maximum clipping even
   when an earlier unclipped transient exceeds the clipping threshold.
2. Mode-anchored clustering within 3 LSB of each window extremum handles
   post-clip TPDF dither outliers that sit 1 LSB above the actual plateau.
3. Laplace-smoothed interior shoulder density avoids the hard `max(1.0, ...)`
   background floor that suppressed rare or high-sample-rate clips.
4. Boundary step sharpness and shoulder-tail checks reject smooth maxima,
   soft limiters, and intentional `tanh` saturation while emitting a
   continuous confidence score in [0, 1] for risk-coverage analysis.
"""
import numpy as np

from rnext.dsp.declip import _extract_runs
from rnext.dsp.sparse import hard_clip_mask as detect_hard_clip_v1


def _window_starts(n_samples: int, win: int, hop: int) -> list[int]:
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if starts[-1] + win < n_samples:
        starts.append(n_samples - win)
    return starts


def _analyze_side_segment(values: np.ndarray, bits: int) -> dict:
    """Analyze one polarity of one channel segment for a hard-clipped plateau."""
    n = values.size
    level_max = float(values.max()) if n else 0.0
    if level_max <= 1e-4:
        return {"detected": False, "confidence": 0.0, "level": level_max,
                "runs": 0, "plateau_samples": 0, "spike_ratio": 0.0,
                "sharp_runs": 0, "bins": 0, "mask": np.zeros(n, dtype=bool)}

    unit = max(2.0 ** (1 - bits), level_max * 1e-7)
    dist_from_max = level_max - values

    # Count samples in the top 5 LSBs to locate the true plateau mode bin
    # even when post-clip TPDF dither places 1-2 outlier samples at +1 LSB.
    top_counts = [
        int(np.count_nonzero(np.abs(dist_from_max - b * unit) <= 0.51 * unit))
        for b in range(5)
    ]
    # Search for the dominant mode inside the top 3 bins (within 2 LSB of max).
    mode_bin = int(np.argmax(top_counts[:3]))
    mode_level = level_max - mode_bin * unit
    dist = mode_level - values  # 0 at mode_level, negative above, positive below

    # Interior shoulder [4..64 LSB] below mode_level measures smooth background
    # density without being contaminated by a dithered plateau (+/-2 LSB).
    shoulder = int(np.count_nonzero((dist > 3.51 * unit) & (dist <= 63.51 * unit)))
    # Near shoulder [3..10 LSB] catches smooth compression ramps (tanh / soft limiter)
    # whose density ramps smoothly toward the maximum instead of jumping sharply.
    near_shoulder = int(np.count_nonzero((dist > 2.51 * unit) & (dist <= 10.51 * unit)))
    bg_rate = max(0.12, (shoulder + 1.0) / 60.0)
    near_rate = max(0.12, (near_shoulder + 0.5) / 8.0)

    # Build a contiguous dither cluster around mode_bin (at most 3 bins wide:
    # mode_bin and immediate +/-1 LSB neighbors with significant excess mass).
    lo_off, hi_off = 0, 0
    mode_count = top_counts[mode_bin]
    if mode_bin > 0:
        above_count = top_counts[mode_bin - 1]
        if above_count >= max(1, int(np.ceil(3.0 * bg_rate))) and above_count >= 0.02 * mode_count:
            lo_off = -1
    if mode_bin + 1 < len(top_counts):
        below_count = top_counts[mode_bin + 1]
        if below_count >= max(2, int(np.ceil(4.0 * bg_rate))) and below_count >= 0.05 * mode_count:
            hi_off = 1
        # Allow a 3rd bin below only when there is no bin above and strong excess mass.
        if lo_off == 0 and hi_off == 1 and mode_bin + 2 < len(top_counts):
            below2 = top_counts[mode_bin + 2]
            if below2 >= max(4, int(np.ceil(8.0 * bg_rate))) and below2 >= 0.05 * mode_count:
                hi_off = 2

    n_bins = hi_off - lo_off + 1
    in_band = (dist >= (lo_off - 0.51) * unit) & (dist <= (hi_off + 0.51) * unit)
    band_count = int(np.count_nonzero(in_band))

    runs_all = _extract_runs(in_band)
    flat_runs = [(int(s), int(e)) for s, e in runs_all if e - s >= 2]
    plateau_samples = int(sum(e - s for s, e in flat_runs))

    # Measure boundary entry/exit sharpness: hard clips enter/exit the plateau
    # abruptly (>= 2 LSB step on at least one side of the run), unlike smooth
    # low-frequency sine crests or soft limiters that creep bin by bin.
    sharp_runs = 0
    for s, e in flat_runs:
        left_step = abs(float(values[s] - values[s - 1])) if s > 0 else 0.0
        right_step = abs(float(values[e] - values[e - 1])) if e < n else 0.0
        if max(left_step, right_step) >= 2.0 * unit:
            sharp_runs += 1

    spike_ratio = float(band_count / max(1, n_bins) / bg_rate)
    # Edge contrast against the immediate near-shoulder [3..10 LSB] separates
    # a hard wall from smooth tanh/soft-limiter compression.
    edge_contrast = float(band_count / max(1, n_bins) / near_rate)

    # Continuous confidence score in [0, 1] combining histogram discontinuity,
    # near-shoulder contrast, flat-run count, and corner sharpness.
    if band_count < 4 or not flat_runs:
        confidence = 0.0
    else:
        c_spike = min(1.0, max(0.0, (np.log10(max(spike_ratio, 1.0)) - 0.6) / 1.0))
        c_edge = min(1.0, max(0.0, (np.log10(max(edge_contrast, 1.0)) - 0.45) / 0.85))
        c_runs = min(1.0, len(flat_runs) / 3.0)
        c_samples = min(1.0, plateau_samples / 6.0)
        c_sharp = min(1.0, sharp_runs / max(1.0, min(2.0, float(len(flat_runs)))))
        confidence = float(c_spike * c_edge * (0.45 + 0.55 * c_runs * c_samples) * (0.2 + 0.8 * c_sharp))

    # Hard structural guards keep false positives at zero on clean, tanh, and
    # soft-limiter controls even if confidence threshold is lowered.
    structural_ok = (
        band_count >= 5
        and len(flat_runs) >= 1
        and plateau_samples >= 2
        and sharp_runs >= 1
        and spike_ratio >= 8.0
        and edge_contrast >= 4.0
        and (len(flat_runs) >= 2 or (plateau_samples >= 4 and spike_ratio >= 15.0))
    )
    if not structural_ok:
        confidence = min(confidence, 0.25)

    return {
        "detected": bool(structural_ok),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "level": float(mode_level),
        "runs": len(flat_runs),
        "sharp_runs": sharp_runs,
        "plateau_samples": plateau_samples,
        "band_samples": band_count,
        "spike_ratio": spike_ratio,
        "edge_contrast": edge_contrast,
        "bins": n_bins,
        "mask": in_band,
    }


def detect_hard_clip_v2(
    x: np.ndarray,
    sr: int,
    bits: int = 24,
    *,
    window_s: float = 2.0,
    hop_s: float = 1.0,
    min_confidence: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Windowed mode-anchored hard-clip detector with continuous confidence.

    Inputs:
        x: float PCM [samples, channels], finite, nonempty.
        sr: sample rate in Hz (> 0).
        bits: source PCM bit depth (8..32) used for quantization bin width.
        window_s: local analysis window duration in seconds.
        hop_s: hop between consecutive windows in seconds.
        min_confidence: minimum confidence in [0, 1] to activate a detected
            window/side mask.

    Returns:
        (mask bool[samples, channels], diagnosis dict with track_confidence).
    """
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("expected nonempty finite PCM [samples, channels]")
    if sr <= 0 or window_s <= 0 or hop_s <= 0:
        raise ValueError("sr, window_s and hop_s must be positive")
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence must lie in [0, 1]")

    bits = min(32, max(8, int(bits or 24)))
    n_samples, n_channels = x.shape
    win = max(1024, round(window_s * sr))
    hop = max(512, round(hop_s * sr))
    starts = _window_starts(n_samples, win, hop)

    mask = np.zeros(x.shape, dtype=bool)
    max_conf = 0.0
    sides_summary = []

    for ch in range(n_channels):
        col = x[:, ch].astype(np.float64)
        for sign in (-1, 1):
            signed = col * sign
            # Always evaluate full-track segment (catches rare clips spread
            # across >4s) plus overlapping local windows (catches late clips
            # whose threshold is below an earlier unclipped peak).
            segments = [(0, n_samples)]
            if len(starts) > 1:
                segments.extend((s, min(n_samples, s + win)) for s in starts)

            best_conf = 0.0
            best_meta = None
            active_windows = 0
            for s0, e0 in segments:
                res = _analyze_side_segment(signed[s0:e0], bits)
                if res["confidence"] > best_conf:
                    best_conf = res["confidence"]
                    best_meta = {k: v for k, v in res.items() if k != "mask"}
                    best_meta["window_start_sample"] = int(s0)
                    best_meta["window_end_sample"] = int(e0)
                if res["detected"] and res["confidence"] >= min_confidence:
                    # Only mark positive-polarity samples in the matching sign
                    # (guarantees mask excludes zeros).
                    seg_mask = res["mask"] & (signed[s0:e0] > 0.0)
                    mask[s0:e0, ch] |= seg_mask
                    active_windows += 1

            max_conf = max(max_conf, best_conf)
            sides_summary.append({
                "channel": ch,
                "sign": sign,
                "best_confidence": float(best_conf),
                "active_windows": active_windows,
                "supported": active_windows > 0,
                **(best_meta or {"level": 0.0, "runs": 0, "sharp_runs": 0,
                                 "plateau_samples": 0, "band_samples": 0,
                                 "spike_ratio": 0.0, "edge_contrast": 0.0,
                                 "bins": 0}),
            })

    # Never mask exact zero samples (required by all downstream solvers).
    mask &= (x != 0.0)
    return mask, {
        "detector": "windowed_mode_plateau_v2",
        "pattern": "windowed_mode_anchored_plateaus",
        "bits": bits,
        "window_s": float(window_s),
        "hop_s": float(hop_s),
        "min_confidence": float(min_confidence),
        "track_confidence": float(max_conf),
        "masked_samples": int(mask.sum()),
        "sides": sides_summary,
        "intent": "not_inferable_from_pcm",
    }


def score_hard_clip_v1(x: np.ndarray, bits: int = 24) -> tuple[np.ndarray, dict]:
    """Run current v1 detector (`rnext.dsp.sparse.hard_clip_mask`) and attach a
    continuous confidence score derived strictly from its own side diagnostics."""
    mask, diag = detect_hard_clip_v1(x, bits=bits)
    scores = []
    min_samples = max(6.0, len(x) * 1e-5)
    for side in diag.get("sides", []):
        r_ratio = min(1.0, float(side.get("mass_ratio", 0.0)) / 10.0)
        r_runs = min(1.0, float(side.get("runs", 0)) / 3.0)
        r_samp = min(1.0, float(side.get("plateau_samples", 0)) / min_samples)
        if side.get("supported"):
            score = 0.5 + 0.5 * min(1.0, np.log10(max(float(side.get("mass_ratio", 10.0)), 10.0)) / 3.0)
        else:
            score = 0.49 * (r_ratio * r_runs * r_samp)
        scores.append(float(np.clip(score, 0.0, 1.0)))
    diag = dict(diag)
    diag["detector"] = "global_plateau_v1_current"
    diag["track_confidence"] = float(max(scores)) if scores else 0.0
    return mask, diag
