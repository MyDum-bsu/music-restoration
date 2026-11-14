"""Oracle: the album evaluator must judge by the declared ladder only.

Fails before scripts/evaluate_album.py exists, passes when the contract of
t-20261114-225449-цель-оценить-качество-альбома-6cea50 holds.
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
SCRIPT = REPO / "scripts" / "evaluate_album.py"

TRACK_COLUMNS = [
    "track", "dr14_orig", "dr14_ours", "crest_db_orig", "crest_db_ours",
    "lufs_orig", "lufs_ours", "true_peak_db_orig", "true_peak_db_ours",
    "clip_samples_orig", "clip_samples_ours",
    "spikes_over_09_orig", "spikes_over_09_ours",
    "band_shift_db", "band_delta_mean_db", "punch_retention_db",
]
VERDICT_COLUMNS = [
    "track", "verdict", "reason", "comparability", "status", "needs_review",
    "restoration_state",
]
REPORT_HEADINGS = ["## Verdict", "## Systematics", "## Per track", "## Limitations"]


@pytest.fixture(scope="session")
def mod():
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    spec = importlib.util.spec_from_file_location("evaluate_album", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_album"] = module
    spec.loader.exec_module(module)
    return module


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(REPO),
    )


def write_tone(path: Path, amplitude: float = 0.05, clipped: bool = False,
               seconds: float = 8.0) -> None:
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


# --- verdict ladder ----------------------------------------------------------


def test_verdict_ladder(mod):
    ours_ok = {"dr14": 10, "crest_db": 14.0, "lufs": -15.0, "true_peak_db": -1.0,
               "clip_samples": 0, "spikes_over_09": 0}
    orig_crushed = {"dr14": 6, "crest_db": 8.0, "lufs": -8.0, "true_peak_db": 0.4,
                    "clip_samples": 900, "spikes_over_09": 5000}
    assert mod.track_verdict(ours_ok, orig_crushed) == ("ours_better", "dr", "loudness_mismatch")

    assert mod.track_verdict(ours_ok, ours_ok) == ("comparable", "tie", "ok")

    spiky = dict(ours_ok, spikes_over_09=50)
    assert mod.track_verdict(spiky, ours_ok) == ("orig_better", "spikes", "ok")
    cleaned = dict(orig_crushed, spikes_over_09=10, true_peak_db=-1.0)
    assert mod.track_verdict(cleaned, orig_crushed)[0:2] == ("ours_better", "spikes")

    hot = dict(ours_ok, true_peak_db=-0.5)
    assert mod.track_verdict(hot, orig_crushed) == ("orig_better", "ceiling", "loudness_mismatch")
    assert mod.track_verdict(hot, orig_crushed, tp_ceiling=-0.4)[0:2] == ("ours_better", "dr")

    regression = dict(ours_ok, dr14=4)
    assert mod.track_verdict(regression, orig_crushed) == ("orig_better", "dr", "loudness_mismatch")


def test_album_verdict_majority(mod):
    assert mod.album_verdict(["ours_better", "ours_better", "orig_better"]) == ("ours_better", "majority")
    assert mod.album_verdict(["orig_better", "orig_better", "ours_better"]) == ("orig_better", "majority")
    assert mod.album_verdict(["ours_better", "orig_better", "comparable"]) == ("comparable", "tie")
    assert mod.album_verdict(["inconclusive"]) == ("inconclusive", "tie")


# --- track matching ----------------------------------------------------------


def test_match_tracks_number_then_name(mod):
    def f(p):
        return Path(p)

    src = [f("01 - Alpha.flac"), f("02 - Beta (Japan bonus track).flac"),
           f("17 - The Reverend King (Japan bonus track).flac"), f("98 - Hidden Track.flac")]
    res = [f("01 - Alpha.flac"), f("02 - Beta.flac"),
           f("17 - The Reverend King.flac"), f("18 - Hidden Track.flac")]
    matched, skipped = mod.match_tracks(src, res)
    via = {(Path(s).name, Path(r).name, how) for _label, s, r, how in matched}
    assert ("01 - Alpha.flac", "01 - Alpha.flac", "number") in via
    assert ("02 - Beta (Japan bonus track).flac", "02 - Beta.flac", "number") in via
    assert ("98 - Hidden Track.flac", "18 - Hidden Track.flac", "name") in via
    assert not skipped


def test_match_tracks_reports_unmatched(mod):
    def f(p):
        return Path(p)

    src = [f("01 - Alpha.flac"), f("02 - Only Here.flac")]
    res = [f("01 - Alpha.flac")]
    matched, skipped = mod.match_tracks(src, res)
    assert len(matched) == 1
    assert any(s["reason"] == "track_mismatch" and s["side"] == "src"
               for s in skipped)


def test_normalize_name(mod):
    assert mod.normalize_name("07 - Song (Japan bonus track)") == "07 song"
    assert mod.normalize_name("07 — Song!! (live)") == "07 song"


# --- end to end --------------------------------------------------------------


def make_album_pair(root: Path) -> tuple[Path, Path]:
    src, res = root / "orig", root / "restored"
    src.mkdir(parents=True)
    res.mkdir(parents=True)
    write_tone(src / "01 - Loud One.flac", clipped=True)
    write_tone(src / "02 - Loud Two.flac", clipped=True)
    write_tone(src / "03 - Quiet Three.flac", amplitude=0.05)
    write_tone(src / "07 - Bonus (Japan bonus track).flac", amplitude=0.05)
    write_tone(res / "01 - Loud One.flac", amplitude=0.3)
    write_tone(res / "02 - Loud Two.flac", amplitude=0.4)
    write_tone(res / "03 - Quiet Three.flac", amplitude=0.05)
    return src, res


def test_cli_end_to_end(mod, tmp_path):
    src, res = make_album_pair(tmp_path / "work")
    out = tmp_path / "eval"

    proc = run_cli("--src", str(src), "--restored", str(res), "--out", str(out))
    assert proc.returncode == 0, proc.stderr

    for name in ("tracks.csv", "verdicts.csv", "summary.json", "REPORT.md"):
        assert (out / name).exists(), name

    cols, tracks = read_csv(out / "tracks.csv")
    assert set(TRACK_COLUMNS) <= set(cols)
    by_track = {r["track"]: r for r in tracks}
    assert len(tracks) == 3
    # track 01: crushed square -> clean sine: DR gap >= 0.3, ours wins
    assert float(by_track["01 - Loud One"]["dr14_ours"]) - float(by_track["01 - Loud One"]["dr14_orig"]) >= 0.3
    # band columns present for every row
    assert float(by_track["01 - Loud One"]["band_delta_mean_db"]) >= 0.0

    vcols, verdicts = read_csv(out / "verdicts.csv")
    assert set(VERDICT_COLUMNS) <= set(vcols)
    assert len(verdicts) == 3
    assert {r["verdict"] for r in verdicts} <= {"ours_better", "orig_better", "comparable", "inconclusive"}
    assert all(r["comparability"] in {"ok", "loudness_mismatch"} for r in verdicts)

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["tracks_measured"] == 3
    assert summary["verdict_overall"] in {"ours_better", "orig_better", "comparable", "inconclusive"}
    assert summary["matched_via"]["number"] == 3
    assert any(s["track"].startswith("07") for s in summary["skipped"]), "bonus track must be reported"
    assert summary["needs_review"] == 0
    for key in ("lufs_gap_median", "dr_change_median", "crest_change_median",
                "band_shift_db_mean", "punch_retention_median", "tp_at_ceiling",
                "dr_regression_tracks",
                "clipped_orig_sum", "clipped_ours_sum", "spikes_orig_sum", "spikes_ours_sum"):
        assert key in summary["systematics"], key
    assert summary["limitations"], "the report must state its own limits"

    report = (out / "REPORT.md").read_text(encoding="utf-8")
    for heading in REPORT_HEADINGS:
        assert heading in report, f"report is missing {heading}"
    assert summary["verdict_overall"] in report


def test_status_lifting_from_report_json(mod, tmp_path):
    src, res = make_album_pair(tmp_path / "work")
    (res / "report.json").write_text(json.dumps([
        {"output": str(res / "01 - Loud One.flac"), "status": "processed_candidate",
         "restoration_state": "candidate_created", "needs_review": False},
        {"output": str(res / "02 - Loud Two.flac"), "status": "qc_fallback_original",
         "restoration_state": "encoded_candidate_rejected", "needs_review": True},
    ]), encoding="utf-8")
    out = tmp_path / "eval"

    proc = run_cli("--src", str(src), "--restored", str(res), "--out", str(out))
    assert proc.returncode == 0, proc.stderr

    _, verdicts = read_csv(out / "verdicts.csv")
    by_track = {r["track"]: r for r in verdicts}
    assert by_track["01 - Loud One"]["status"] == "processed_candidate"
    assert by_track["01 - Loud One"]["needs_review"] == "False"
    assert by_track["02 - Loud Two"]["status"] == "qc_fallback_original"
    assert by_track["02 - Loud Two"]["needs_review"] == "True"
    assert by_track["03 - Quiet Three"]["status"] == "unknown"

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"]["processed_candidate"] == 1
    assert summary["status_counts"]["qc_fallback_original"] == 1
    assert summary["needs_review"] == 1


def test_cli_fails_loud_on_missing_restored_dir(tmp_path):
    src = tmp_path / "work" / "orig"
    src.mkdir(parents=True)
    write_tone(src / "01 - Track One.flac", clipped=True)
    out = tmp_path / "eval"

    proc = run_cli("--src", str(src), "--restored", str(tmp_path / "nope"), "--out", str(out))
    assert proc.returncode == 2, f"expected exit 2, got {proc.returncode}"
    assert "nope" in (proc.stderr + proc.stdout)
    assert not (out / "summary.json").exists(), "a failed run must not leave output behind"
