"""Consistent hard-clipping repair using a Parseval STFT and hard projections.

An independent structured-sparsity implementation inspired by analysis sparse
declipping. It is not the published A-SPADE implementation or its parameter set.
Reliable samples are fixed; only an explicitly supplied clipping mask is repaired.
"""
import numpy as np

from rnext.dsp.contour import _frame_starts
from rnext.dsp.declip import _extract_runs


class ReconstructionRejected(ValueError):
    """The numerical candidate cannot be delivered; the source remains usable."""


class _ParsevalSTFT:
    def __init__(self, samples, frame):
        self.frame = frame
        self.hop = frame // 4
        self.starts = _frame_starts(samples, frame, self.hop)
        # Match the frozen research implementation exactly: tiny window changes
        # can change ties in a nonconvex hard-threshold iteration.
        window = np.hanning(frame + 1)[:-1]
        energy = np.zeros(samples)
        for start in self.starts:
            energy[start:start + frame] += window ** 2
        self.weights = np.stack([
            window / np.sqrt(np.maximum(energy[s:s + frame], 1e-20))
            for s in self.starts])
        self.samples = samples

    def analysis(self, x):
        frames = np.lib.stride_tricks.sliding_window_view(x, self.frame)[::self.hop]
        return np.fft.fft(frames * self.weights, axis=-1, norm="ortho")

    def synthesis(self, coeff):
        frames = np.fft.ifft(coeff, axis=-1, norm="ortho").real * self.weights
        result = np.zeros(self.samples)
        for idx, start in enumerate(self.starts):
            result[start:start + self.frame] += frames[idx]
        return result


def hard_clip_mask(x, bits=24):
    """Find repeated extreme plateaus; this identifies a pattern, not intent.

    Natural isolated maxima and smooth soft saturation are left unsupported.
    Positive/negative levels are separate to handle asymmetric PCM quantization.
    """
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("expected nonempty finite PCM [samples, channels]")
    bits = min(32, max(8, int(bits or 24)))
    mask = np.zeros(x.shape, dtype=bool)
    sides = []
    for channel in range(x.shape[1]):
        for sign in (-1, 1):
            values = x[:, channel].astype(np.float64) * sign
            level = float(values.max())
            if level <= 1e-5:
                continue
            unit = max(2. ** (1 - bits), level * 1e-7)
            distance = level - values
            # TPDF dither can spread a true plateau over several PCM bins.
            # Extend only through adjacent bins with their own excess mass;
            # smooth natural maxima do not have this histogram discontinuity.
            counts = [int(np.count_nonzero(np.abs(distance - b * unit) <= .51 * unit))
                      for b in range(3)]
            shoulder = np.count_nonzero((distance > 2.51 * unit) & (distance <= 50.5 * unit))
            background = max(1., shoulder / 48.)
            bins = 1
            for index in (1, 2):
                if counts[index] < max(10 * background, .01 * counts[0]):
                    break
                bins += 1
            near = (distance >= -.51 * unit) & (distance <= (bins - .49) * unit)
            runs = [(s, e) for s, e in _extract_runs(near) if e - s >= 2]
            samples = int(sum(e - s for s, e in runs))
            ratio = float(np.count_nonzero(near) / bins / background)
            supported = len(runs) >= 3 and samples >= max(6, len(x) * 1e-5) and ratio >= 10
            if supported:
                # Repeated runs establish the level. Isolated samples at that
                # same supported level are clipped too, especially at high Hz.
                mask[:, channel] |= near
            sides.append({"channel": channel, "sign": sign, "level": level,
                          "runs": len(runs), "plateau_samples": samples,
                          "mass_ratio": ratio, "quantization_bins": bins,
                          "supported": bool(supported)})
    return mask, {"pattern": "repeated_extreme_plateaus", "bits": bits,
                  "masked_samples": int(mask.sum()), "sides": sides,
                  "intent": "not_inferable_from_pcm"}


def _repair_channel(x, mask, frame, iterations):
    scale = float(np.max(np.abs(x)))
    if not mask.any() or scale == 0:
        return x.copy(), {"iterations": 0, "residual": 0., "stopping_reason": "no_mask"}
    hop, pad = frame // 4, frame
    length = int(np.ceil((len(x) + 2 * pad - frame) / hop)) * hop + frame
    observed = np.pad(x.astype(np.float64) / scale, (pad, length - len(x) - pad))
    unknown = np.pad(mask, (pad, length - len(x) - pad))
    positive, negative = unknown & (observed > 0), unknown & (observed < 0)
    transform = _ParsevalSTFT(length, frame)
    current = observed.copy()
    coeff = transform.analysis(current)
    identity_error = float(np.max(np.abs(transform.synthesis(coeff)[pad:pad + len(x)]
                                         - current[pad:pad + len(x)])))
    if identity_error > 1e-10:
        raise ValueError("STFT is not Parseval on the source interval")
    dual = np.zeros_like(coeff)
    sparse = coeff.copy()
    reason = "iteration_budget"
    for iteration in range(iterations):
        proposal = transform.synthesis(sparse - dual)
        current = observed.copy()
        current[positive] = np.maximum(proposal[positive], observed[positive])
        current[negative] = np.minimum(proposal[negative], observed[negative])
        coeff = transform.analysis(current)
        values = coeff + dual
        retained = min(frame, 20 + 10 * (iteration // 2))
        cutoff = np.partition(np.abs(values), frame - retained, axis=-1)[:, frame - retained]
        sparse = values * (np.abs(values) >= cutoff[:, None])
        difference = coeff - sparse
        dual += difference
        residual = float(np.sqrt(np.sum(np.abs(difference) ** 2)
                                 / max(1e-20, np.sum(np.abs(coeff) ** 2))))
        if iteration >= 20 and residual < 1e-4:
            reason = "residual_tolerance"
            break
    result = current[pad:pad + len(x)] * scale
    if not np.isfinite(result).all():
        raise ReconstructionRejected("sparse repair produced nonfinite samples")
    result[~mask] = x[~mask]
    return result, {"iterations": iteration + 1, "residual": residual,
                    "stopping_reason": reason, "identity_error": identity_error}


def repair_clipping(x, sr, mask, frame=4096, iterations=160, chunk_seconds=12):
    """Repair mono/stereo float PCM; preserve reliable samples before output gain.

    Blocks overlap by half their duration, keeping memory bounded for long songs.
    A positive weighted average preserves clipping consistency at block seams.
    No peak normalization or integer encoding occurs in this function.
    """
    if x.ndim != 2 or not x.size or x.shape[1] not in (1, 2) or not np.isfinite(x).all():
        raise ValueError("expected nonempty finite mono/stereo PCM")
    if mask.shape != x.shape or mask.dtype != bool or np.any(mask & (x == 0)):
        raise ValueError("clipping mask must be boolean, match PCM and exclude zero samples")
    if sr <= 0 or iterations < 1 or frame < 32 or frame % 4 or chunk_seconds <= 0:
        raise ValueError("invalid sparse repair configuration")
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("floating PCM required")
    block = max(frame * 4, int(chunk_seconds * sr))
    starts = sorted(set(range(0, max(1, len(x) - block + 1), max(1, block // 2)))
                    | {max(0, len(x) - block)})
    result = x.astype(np.float64).copy()
    reports = []
    for channel in range(x.shape[1]):
        if not mask[:, channel].any():
            continue
        total, weights = np.zeros(len(x)), np.zeros(len(x))
        for start in starts:
            end = min(len(x), start + block)
            repaired, report = _repair_channel(x[start:end, channel], mask[start:end, channel],
                                                frame, iterations)
            weight = np.ones(end - start)
            overlap = min(block // 2, len(weight))
            if start:
                weight[:overlap] *= np.linspace(0., 1., overlap)
            if end < len(x):
                weight[-overlap:] *= np.linspace(1., 0., overlap)
            total[start:end] += repaired * weight
            weights[start:end] += weight
            reports.append({"channel": channel, "start_sample": start, **report})
        if np.any(weights <= 0):
            raise ValueError("block coverage gap")
        result[:, channel] = total / weights
    result[~mask] = x[~mask]
    positive, negative = mask & (x > 0), mask & (x < 0)
    result[positive] = np.maximum(result[positive], x[positive])
    result[negative] = np.minimum(result[negative], x[negative])
    result = result.astype(x.dtype)
    if not np.isfinite(result).all():
        raise ReconstructionRejected("repair exceeds the source floating-point representation")
    return result, {"method": "parseval_stft_hard_projection", "frame": frame,
                                   "iterations_limit": iterations, "chunk_seconds": chunk_seconds,
                                   "masked_samples": int(mask.sum()), "blocks": reports,
                                   "reliable_samples_preserved": True}
