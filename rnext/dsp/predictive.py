"""Float AR repair proposal with conservative periodic-prediction confidence.

    Validation on reliable context and forward/backward agreement decide
    whether this proposal is usable. There is no integer/full-scale clamp.
"""
import numpy as np

from rnext.dsp.declip import _extract_runs, _ar_extrapolate


def _fit_context(context, unreliable, order):
    windows = np.lib.stride_tricks.sliding_window_view(context, order + 1)
    masks = np.lib.stride_tricks.sliding_window_view(unreliable, order + 1)
    windows = windows[~masks.any(axis=1)]
    training, validation = windows[::2], windows[1::2]
    if len(training) < order + 2 or len(validation) < order + 2:
        raise ValueError("insufficient reliable prediction context")
    predictors, target = training[:, :-1][:, ::-1], training[:, -1]
    gram = predictors.T @ predictors
    ridge = max(1e-12, float(np.trace(gram)) * 1e-8 / order)
    coeff = np.linalg.solve(gram + np.eye(order) * ridge, predictors.T @ target)
    error = validation[:, -1] - validation[:, :-1][:, ::-1] @ coeff
    relative_error = float(np.sqrt(np.mean(error ** 2)
                                  / max(1e-20, np.mean(validation[:, -1] ** 2))))
    return coeff, relative_error


def periodic_ar_proposal(x, mask, order=16, max_gap=1024):
    """Propose changes only for gaps with highly predictable reliable context."""
    y = x.astype(np.float64).copy()
    diagnostics = []
    inspected, covered, confident_gaps = 0, 0, 0
    count = int(mask.sum())
    for channel in range(x.shape[1]):
        peak = max(1e-6, float(np.max(np.abs(x[:, channel]))))
        for start, end in _extract_runs(mask[:, channel]):
            if inspected - covered > count * .1:
                return x.copy(), {"method": "periodic_float_ar", "accepted": False,
                                  "confident_coverage": covered / max(1, count),
                                  "changed_fraction": 0., "gaps": diagnostics,
                                  "stopping_reason": "confidence_gate_unreachable"}
            start, end = int(start), int(end)
            length = end - start
            inspected += length
            if length < 2 or length > max_gap or start < 32 or end > len(x) - 32:
                continue
            context = min(2048, max(256, 8 * length))
            left, right = max(0, start - context), min(len(x), end + context)
            pre, post = x[left:start, channel].astype(float), x[end:right, channel].astype(float)
            pre_mask, post_mask = mask[left:start, channel], mask[end:right, channel]
            before, after = np.flatnonzero(pre_mask[::-1]), np.flatnonzero(post_mask)
            npre = int(before[0]) if len(before) else len(pre)
            npost = int(after[0]) if len(after) else len(post)
            size = min(order, npre // 2, npost // 2)
            if size < 2:
                continue
            try:
                a, epre = _fit_context(pre, pre_mask, size)
                b, epost = _fit_context(post[::-1], post_mask[::-1], size)
                forward = _ar_extrapolate(pre, a, length, False)
                backward = _ar_extrapolate(post, b, length, True)
            except (ValueError, np.linalg.LinAlgError):
                continue
            weight = np.linspace(1., 0., length + 2)[1:-1]
            fill = weight * forward + (1 - weight) * backward
            agreement = float(np.sqrt(np.mean((forward - backward) ** 2)
                                      / max(1e-20, np.mean(fill ** 2))))
            source = x[start:end, channel]
            supported = (max(epre, epost) <= .001 and agreement <= .01
                         and np.isfinite(fill).all()
                         and np.max(np.abs(fill)) <= peak * 8)
            if not supported:
                continue
            sign = np.sign(source)
            fill = sign * np.maximum(sign * fill, np.abs(source))
            y[start:end, channel] = fill
            covered += length
            confident_gaps += 1
            if len(diagnostics) < 20:
                diagnostics.append({"channel": channel, "start_sample": start,
                                    "samples": length, "context_error": max(epre, epost),
                                    "forward_backward_error": agreement})
    y = y.astype(x.dtype)
    coverage = float(covered / max(1, count))
    changed = float(np.count_nonzero((y != x) & mask) / max(1, count))
    return y, {"method": "periodic_float_ar", "confident_coverage": coverage,
               "changed_fraction": changed, "accepted": coverage >= .9 and changed >= .8,
               "confidence_note": "engineering confidence, not calibrated probability",
               "confident_gaps": confident_gaps, "gaps": diagnostics,
               "reliable_samples_preserved": True}
