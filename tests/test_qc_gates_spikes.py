"""`qc_gates.py --orig`: the spike gate is relative to the source take.

The review's real case: a hot master carries 748 click-like jumps, the
restoration 35 — an absolute gate failed the better result. The absolute
verdict must stay the default; `--orig` only compares against the source.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.qc_gates import sample_spikes, source_take, spike_verdict

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qc_gates.py"


def silent_or_spiky(spikes: int) -> np.ndarray:
    x = np.zeros((64, 2), dtype=np.float32)
    for i in range(spikes):
        x[2 + 2 * i, 0] = 1.0
    return x


def test_sample_spikes_counts_click_like_jumps():
    assert sample_spikes(silent_or_spiky(0)) == 0
    # one click is two >0.9 sample-to-sample jumps: the rise and the fall
    assert sample_spikes(silent_or_spiky(1)) == 2
    assert sample_spikes(silent_or_spiky(5)) == 10


def test_without_orig_every_spike_fails():
    assert spike_verdict(0, None) == (False, "absolute")
    assert spike_verdict(1, None) == (True, "absolute")


def test_with_orig_only_growth_fails():
    # the real case: 748 spikes in the source, 35 in the restoration — passes
    assert spike_verdict(35, 748) == (False, "vs_orig")
    assert spike_verdict(748, 748) == (False, "vs_orig")
    assert spike_verdict(749, 748) == (True, "vs_orig")
    assert spike_verdict(1, 0) == (True, "vs_orig")


def test_source_take_is_matched_by_track_number(tmp_path: Path):
    src = tmp_path / "01 - Balls to the Wall.flac"
    src.write_bytes(b"")
    (tmp_path / "02 - London Leatherboys.flac").write_bytes(b"")

    assert source_take(str(tmp_path), "01 - Balls to the Wall (restored).flac") == src
    assert source_take(str(tmp_path), "02 - London Leatherboys.flac") is not None
    assert source_take(str(tmp_path), "07 - Unrelated.flac") is None
    assert source_take(str(tmp_path), "no-track-number.flac") is None
    assert source_take(str(tmp_path / "missing"), "01 - X.flac") is None


def test_missing_source_take_is_an_explicit_failure(tmp_path: Path):
    """No source for --orig must fail loudly, not fall back to the absolute gate."""
    out = tmp_path / "01 - Track.flac"
    out.write_bytes(b"")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(out), "--orig", str(tmp_path / "empty-src")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "no source take" in proc.stdout
    assert '"verdict": "FAIL"' in proc.stdout


def test_help_lists_the_optional_orig_flag():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )

    assert proc.returncode == 0
    assert "--orig" in proc.stdout
    assert "required" not in proc.stdout


@pytest.mark.parametrize("declared", ["--orig"])
def test_orig_is_not_a_required_argument(declared: str):
    src = SCRIPT.read_text(encoding="utf-8")
    lines = [line for line in src.splitlines() if "add_argument" in line and declared in line]
    assert lines, "qc_gates.py does not declare --orig"
    assert all("required=True" not in line for line in lines)
