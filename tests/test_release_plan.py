"""Material-dependent ceilings and the release-level dynamics target.

The first eval-vs-masters run (docs/eval-vs-masters-run.md) showed one
universal ceiling and one per-track dynamics constant simultaneously
undershooting and overshooting different releases. These tests pin the
replacement: the ceiling follows the material class, the target follows the
release's own least-compressed third, and a ceiling never boosts anything.
"""
import numpy as np
import pytest

from rnext.dsp.selection import (TP_CEILING_BY_CLASS, candidate_policies,
                                 material_plan, output_violations,
                                 release_target, select_track)


def test_tp_ceiling_follows_material_class():
    crushed = material_plan({"dr": 5, "rms_tt": -9.0}, 30.0)
    mid = material_plan({"dr": 9, "rms_tt": -9.0}, 30.0)
    sparse = material_plan({"dr": 5, "rms_tt": -20.0}, 30.0, allow_sparse=True)
    assert crushed["tp_ceiling"] == TP_CEILING_BY_CLASS["crushed"] == -0.3
    assert mid["tp_ceiling"] == TP_CEILING_BY_CLASS["mid"] == -0.5
    assert sparse["tp_ceiling"] == TP_CEILING_BY_CLASS["sparse"] == -1.5
    assert material_plan({"dr": 12, "rms_tt": -9.0}, 30.0)["tp_ceiling"] is None
    assert candidate_policies(crushed)["gentle"].target_tp_dbfs == -0.3
    assert candidate_policies(sparse)["adaptive"].target_tp_dbfs == -1.5
    # A ceiling only attenuates: no class may reach a hot ceiling.
    for plan in (crushed, mid, sparse):
        assert plan["tp_ceiling"] <= -0.3


def test_output_violations_use_the_plan_ceiling():
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    tone = np.sin(2 * np.pi * 1000.0 * t)
    x = np.stack([tone * 0.5, tone * 0.5], axis=1).astype(np.float32)
    y = np.stack([tone * 10 ** (-0.4 / 20)] * 2, axis=1).astype(np.float32)
    from rnext.dsp.selection import measure
    orig, meas = measure(x, sr), measure(y, sr)
    assert "true_peak_above_ceiling" in output_violations(x, y, sr, orig, meas)
    assert "true_peak_above_ceiling" not in output_violations(
        x, y, sr, orig, meas, tp_ceiling=-0.3)


def test_release_target_from_least_compressed_third():
    # A uniform release says nothing: the class-table priors apply.
    assert release_target([{"dr": 6} for _ in range(6)]) is None
    # Median DR of the least-compressed third minus the release median.
    assert release_target([{"dr": d} for d in (4, 5, 5, 6, 9, 10)]) == 4.0
    # The gap is capped to the conservative class-table range.
    assert release_target([{"dr": d} for d in (1, 1, 1, 20, 20, 20)]) == 5.3
    assert release_target([{"dr": 3}, {"dr": 9}]) is None


def test_material_plan_uses_release_target():
    crushed = {"dr": 5, "rms_tt": -9.0}
    assert material_plan(crushed, 30.0)["target"] == 5.3
    assert material_plan(crushed, 30.0, release_target=2.0)["target"] == 2.0
    # Sparse material keeps its explicit-request target regardless.
    sparse = {"dr": 5, "rms_tt": -20.0}
    plan = material_plan(sparse, 30.0, release_target=2.0, allow_sparse=True)
    assert plan["target"] == 0.8


def test_select_track_threads_release_target():
    sr = 4000
    seconds = 24.0
    t = np.arange(int(sr * seconds)) / sr
    x = np.stack([np.sin(2 * np.pi * 220.0 * t) * 0.5] * 2, axis=1).astype(np.float32)
    report = select_track(x.copy(), sr, release_target=1.0)[1]
    assert report["plan"]["class"] == "crushed"
    assert report["plan"]["target"] == 1.0
    assert report["plan"]["tp_ceiling"] == -0.3


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
