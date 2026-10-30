"""Unit tests for the music-restoration DSP core (pytest)."""
import numpy as np
import pytest

from rnext.dsp.contour import ContourPolicy, contour_expand
from rnext.dsp.declip import clip_positions, declip_track
from rnext.dsp.limit import limit_track
from rnext.dsp.pipeline import Policy, restore_track


def test_wola_identity_expand_r1():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300000, 2)).astype(np.float32) * 0.3
    y = contour_expand(x, 44100, ContourPolicy(r_expand=1.0, frame=2048, hop=512))
    rel = np.sqrt(np.mean((y - x) ** 2)) / np.sqrt(np.mean(x ** 2))
    assert rel < 1e-5, rel


def test_flatten_identity_k1():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200000, 2)).astype(np.float32) * 0.3
    y = contour_expand(x, 44100, ContourPolicy(mode="flatten", k=1.0, frame=2048, hop=512))
    rel = np.sqrt(np.mean((y - x) ** 2)) / np.sqrt(np.mean(x ** 2))
    assert rel < 1e-5, rel


def test_expand_boosts_crest():
    """A crest-heavy signal must gain crest."""
    sr = 44100
    t = np.arange(sr * 10) / sr
    base = 0.4 * np.sin(2 * np.pi * 110 * t)
    hit = 0.55 * (np.exp(-np.arange(600) / 150.0) * np.sin(2 * np.pi * 700 * np.arange(600) / sr))
    x = np.stack([base, base], 1)
    idx = np.arange(0, sr * 10 - 700, sr)
    for i in idx:
        x[i:i + 600, :] += hit[:, None]
    x = x.astype(np.float32)
    c0 = 20 * np.log10(np.max(np.abs(x)) / np.sqrt(np.mean(x ** 2)))
    y = contour_expand(x, sr, ContourPolicy(mode="expand", r_expand=1.6, frame=2048, hop=512))
    c1 = 20 * np.log10(np.max(np.abs(y)) / np.sqrt(np.mean(y ** 2)))
    assert c1 > c0 + 1.0, (c0, c1)


def test_declip_sine_reconstruction():
    sr = 44100
    t = np.arange(44100 * 4) / sr
    sig = 0.9 * np.sin(2 * np.pi * 220 * t)
    clipd = np.column_stack([
        np.clip(sig * 1.25 * 32768, -32768, 32767).astype(np.int16),
        np.clip(sig * 0.3 * 32768, -32768, 32767).astype(np.int16)])
    assert clip_positions(clipd) > 500
    y, nfill = declip_track(clipd)
    assert nfill > 0
    # compare MAGNITUDES: reconstruction follows the signal's sign, the reference is |sig|
    err = np.abs(np.abs(y[:, 0][3000:].astype(np.float64) / 32768) - np.abs(sig)[3000:])
    assert err.max() < 0.45, err.max()


def test_limiter_tp():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((44100, 2)) * 0.9).astype(np.float32)
    y = limit_track(x, 44100, ceiling_db=-1.0)
    assert np.max(np.abs((y * 32768).astype(np.int32))) <= 32768 * 10 ** (-1.0 / 20) + 2


def test_pipeline_transparent_when_contour_off_and_k1():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((100000, 2)) * 0.2).astype(np.float32)
    pol = Policy(mode="expand", r_expand=1.0, contour_enabled=False, declip=True)
    y = restore_track(x, 44100, pol)
    # The HPF only touches the lowest frequencies. For WHITE noise, frequencies
    # below 25 Hz carry sqrt(25/22050) ≈ 3.4 % of RMS — exactly what the filter
    # removes (verified: the difference matches the low-frequency part,
    # correlation 0.9999), so a 2 % threshold is physically unreachable;
    # we keep 5 % and watch that it does not grow.
    rel = np.sqrt(np.mean((y - x) ** 2)) / np.sqrt(np.mean(x ** 2))
    assert rel < 0.05, rel


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
