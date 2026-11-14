"""Oracle: the eval harness must measure honestly, select deterministically and fail loudly.

Criterion for t-20261104-235752-цель-оценить-насколько-наш-скил-7a46f8.
Fails before `scripts/eval_vs_masters.py` exists, passes when the contract holds.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "eval_vs_masters.py"

METRIC_KEYS = {
    "dr14",
    "crest_db",
    "lufs",
    "true_peak_db",
    "clip_samples",
    "spikes_over_09",
    "band_rms_db",
}
TRACK_COLUMNS = [
    "release",
    "track",
    "side",
    "dr14",
    "crest_db",
    "lufs",
    "true_peak_db",
    "clip_samples",
    "spikes_over_09",
]
DELTA_COLUMNS = [
    "release",
    "track",
    "metric",
    "abs_delta_ours_master",
    "abs_delta_orig_master",
]
ALBUM_COLUMNS = [
    "release",
    "metric",
    "abs_delta_ours_master",
    "abs_delta_orig_master",
    "verdict",
]
REPORT_HEADINGS = ["## Verdict", "## Per track", "## Per album", "## Reproduce"]
SEED = 20261104


@pytest.fixture(scope="session")
def mod():
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    spec = importlib.util.spec_from_file_location("eval_vs_masters", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_vs_masters"] = module
    spec.loader.exec_module(module)
    return module


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def write_tone(path: Path, amplitude: float = 0.05, clipped: bool = False, seconds: float = 8.0) -> None:
    sr = 48000
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    tone = np.sin(2.0 * np.pi * 1000.0 * t)
    x = np.clip(tone * 4.0, -1.0, 1.0) if clipped else tone * amplitude
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def make_release(root: Path, name: str, kinds: dict[str, bool] | None = None) -> dict:
    """Three side directories for one album: orig quiet, ours quiet, master crushed."""
    kinds = kinds or {"orig": False, "ours": False, "master": True}
    sides = {}
    for side, clipped in kinds.items():
        d = root / name / side
        d.mkdir(parents=True, exist_ok=True)
        write_tone(d / "01 - Track One.wav", clipped=clipped)
        write_tone(d / "02 - Track Two.wav", clipped=clipped)
        sides[side] = str(d)
    return {"release": name, **sides}


# --- measurement -------------------------------------------------------------


def test_measure_side_normalizes_all_metrics(mod, tmp_path):
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    write_tone(quiet / "01.wav")

    loud = tmp_path / "loud"
    loud.mkdir()
    write_tone(loud / "01.wav", clipped=True)

    q = mod.measure_side(quiet)
    l = mod.measure_side(loud)

    assert METRIC_KEYS <= set(q), f"measure_side is missing {METRIC_KEYS - set(q)}"
    assert METRIC_KEYS <= set(l)
    for key in ("dr14", "crest_db"):
        assert isinstance(q[key], (int, float)) and np.isfinite(q[key]), key
        assert l[key] < q[key], f"{key}: a hard-clipped tone must not measure taller than a quiet one"
    assert l["spikes_over_09"] > q["spikes_over_09"]
    assert q["band_rms_db"] is not None and l["band_rms_db"] is not None


# --- verdict rule ------------------------------------------------------------


def test_verdict_thresholds_and_tolerance(mod):
    assert mod.verdict(1.0, 5.0) == "better"
    assert mod.verdict(5.0, 1.0) == "worse"
    assert mod.verdict(5.0, 5.0) == "comparable"
    assert mod.verdict(5.1, 5.0) == "comparable"  # inside the 5% band
    assert mod.verdict(4.8, 5.0) == "comparable"
    assert mod.verdict(4.0, 5.0) == "better"
    assert mod.verdict(6.5, 5.0) == "worse"
    # original already equals the master: no division, no exception
    assert mod.verdict(0.0, 0.0) == "comparable"


def test_overall_verdict_is_a_majority_vote(mod):
    assert mod.overall_verdict({"dr14": "better", "crest_db": "better", "lufs": "worse"}) == "better"
    assert mod.overall_verdict({"dr14": "worse", "crest_db": "worse", "lufs": "better"}) == "worse"
    assert mod.overall_verdict({"dr14": "better", "crest_db": "worse"}) == "comparable"
    assert mod.overall_verdict({}) == "comparable"


# --- selection ---------------------------------------------------------------


def single(i: int, seeds: int = 10) -> dict:
    return {"topic_id": 1000 + i, "title": f"Artist {i} - Album {i} (2020)", "seeds": seeds}


def disco(i: int, seeds: int = 10) -> dict:
    return {"topic_id": 5000 + i, "title": f"Artist {i} - Дискография (2020)", "seeds": seeds}


def ids(picks) -> list[int]:
    return [int(r["topic_id"]) for r in picks]


def test_pick_releases_is_deterministic_for_a_seed(mod):
    records = [single(i) for i in range(40)]
    first = ids(mod.pick_releases(records, SEED, per_master=5))
    second = ids(mod.pick_releases(list(reversed(records)), SEED, per_master=5))
    assert first == second, "selection must not depend on input order"
    assert len(first) == 5
    assert first == sorted(first)


def test_pick_releases_seed_changes_the_sample(mod):
    records = [single(i) for i in range(300)]
    a = ids(mod.pick_releases(records, SEED, per_master=5))
    b = ids(mod.pick_releases(records, SEED + 1, per_master=5))
    assert a != b, "the seed must actually drive the sample"


def test_pick_releases_requires_seeds_of_at_least_three(mod):
    dead = [single(i, seeds=s) for i, s in zip(range(3), (0, 1, 2))]
    alive = [single(i) for i in range(3, 13)]
    picks = ids(mod.pick_releases(dead + alive, SEED, per_master=20))
    assert len(picks) == 10
    for r in dead:
        assert int(r["topic_id"]) not in picks


def test_pick_releases_prefers_single_albums(mod):
    records = [single(i) for i in range(6)] + [disco(i) for i in range(6)]
    picks = mod.pick_releases(records, SEED, per_master=5)
    assert len(picks) == 5
    for r in picks:
        assert int(r["topic_id"]) < 5000, f"discography picked while single albums remained: {r['title']}"


def test_pick_releases_falls_back_to_discographies_and_reports_shortfall(mod):
    only_disco = [disco(i) for i in range(4)]
    picks = mod.pick_releases(only_disco, SEED, per_master=5)
    assert len(picks) == 4  # shortfall is returned, not padded


# --- end to end --------------------------------------------------------------


def test_cli_end_to_end_produces_consistent_tables(mod, tmp_path):
    root = tmp_path / "work"
    release = make_release(root, "Artist - Album")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"seed": SEED, "releases": [release]}), encoding="utf-8")
    out = tmp_path / "out"
    report = out / "REPORT.md"

    proc = run_cli("--manifest", str(manifest), "--out-dir", str(out), "--report", str(report))
    assert proc.returncode == 0, proc.stderr

    for name in ("tracks.csv", "deltas.csv", "albums.csv", "summary.json"):
        assert (out / name).exists(), name

    track_cols, tracks = read_csv(out / "tracks.csv")
    assert set(TRACK_COLUMNS) <= set(track_cols)
    assert {r["side"] for r in tracks} == {"orig", "ours", "master"}
    assert len(tracks) == 6, "2 tracks x 3 sides"
    assert len({(r["release"], r["track"], r["side"]) for r in tracks}) == 6, "no duplicated measurement"

    delta_cols, deltas = read_csv(out / "deltas.csv")
    assert set(DELTA_COLUMNS) <= set(delta_cols)
    assert deltas
    for row in deltas:
        assert float(row["abs_delta_ours_master"]) >= 0.0
        assert float(row["abs_delta_orig_master"]) >= 0.0

    album_cols, albums = read_csv(out / "albums.csv")
    assert set(ALBUM_COLUMNS) <= set(album_cols)
    assert albums
    for row in albums:
        assert row["verdict"] in {"better", "worse", "comparable"}

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["seed"] == SEED
    assert summary["verdict_overall"] in {"better", "worse", "comparable"}
    per_metric = summary["per_metric"]
    assert per_metric, "per_metric must not be empty"
    for metric, cell in per_metric.items():
        assert cell["verdict"] in {"better", "worse", "comparable"}, metric
        assert float(cell["abs_ours"]) >= 0.0 and float(cell["abs_orig"]) >= 0.0
    better = sum(1 for v in per_metric.values() if v["verdict"] == "better")
    worse = sum(1 for v in per_metric.values() if v["verdict"] == "worse")
    expected = "better" if better > worse else "worse" if worse > better else "comparable"
    assert summary["verdict_overall"] == expected

    assert report.exists()
    text = report.read_text(encoding="utf-8")
    for heading in REPORT_HEADINGS:
        assert heading in text, f"report is missing {heading}"
    assert "1" in text  # measured count present


def test_cli_fails_loud_on_missing_side(tmp_path):
    root = tmp_path / "work"
    release = make_release(root, "Artist - Album")
    release["master"] = str(root / "does-not-exist")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"seed": SEED, "releases": [release]}), encoding="utf-8")
    out = tmp_path / "out"

    proc = run_cli("--manifest", str(manifest), "--out-dir", str(out))
    assert proc.returncode == 2, f"expected exit 2, got {proc.returncode}"
    assert "does-not-exist" in (proc.stderr + proc.stdout)
    assert not (out / "summary.json").exists(), "a failed run must not leave a summary behind"


def test_select_cli_requires_seed_and_is_reproducible(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps(r) for r in [single(i) for i in range(6)] + [disco(i) for i in range(6)]),
        encoding="utf-8",
    )

    no_seed = run_cli("--select", "--corpus", str(corpus), "--out", str(tmp_path / "p0.json"))
    assert no_seed.returncode == 2, "selection must not default the seed"

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    for target in (a, b):
        proc = run_cli(
            "--select",
            "--corpus",
            str(corpus),
            "--per-master",
            "5",
            "--seed",
            str(SEED),
            "--out",
            str(target),
        )
        assert proc.returncode == 0, proc.stderr
    assert a.read_bytes() == b.read_bytes()

    picks = json.loads(a.read_text(encoding="utf-8"))
    assert picks["seed"] == SEED
    assert len(picks["picks"]) == 5
    assert {int(p["topic_id"]) for p in picks["picks"]} <= {r["topic_id"] for r in [single(i) for i in range(6)]}
    for p in picks["picks"]:
        assert int(p["seeds"]) >= 3


def test_report_values_match_summary(tmp_path):
    root = tmp_path / "work"
    release = make_release(root, "Artist - Album")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"seed": SEED, "releases": [release]}), encoding="utf-8")
    out = tmp_path / "out"
    report = out / "REPORT.md"

    proc = run_cli("--manifest", str(manifest), "--out-dir", str(out), "--report", str(report))
    assert proc.returncode == 0, proc.stderr

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    text = report.read_text(encoding="utf-8")
    assert summary["verdict_overall"] in text, "rendered verdict must be the one from summary.json"
    for metric, cell in summary["per_metric"].items():
        assert metric in text, f"report omits metric {metric}"
        assert cell["verdict"] in text
