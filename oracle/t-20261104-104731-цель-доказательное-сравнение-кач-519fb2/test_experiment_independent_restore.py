#!/usr/bin/env python3
"""Oracle: independent-method restoration comparison.

Black-box only: every check drives the CLIs the spec defines and reads their
artifacts. Nothing from `rnext` is imported here, so a worker cannot "pass" by
bending an internal API.

Criteria encoded here (see spec body for the wording):
  * three experiment scripts exist and the restore is independent of the canon
    method (no restore_per_track / rnext.dsp.pipeline);
  * album selection is seeded, deterministic and capped at 5 per artist;
  * the independent restore bounds peak and clipping, never touches sources
    and removes its temp dir;
  * the restore refuses overlapping output and canon destinations;
  * the comparator emits the documented CSV schema, matches audio_measure for
    the ORIG variant, and derives verdicts from its own numbers;
  * the committed report exists with the required sections and is consistent
    with the committed CSV data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

ID = "t-20261104-104731-цель-доказательное-сравнение-кач-519fb2"
DATE = "2026-11-04"

VARIANTS = ("indep", "orig", "ours")
TRACK_COLUMNS = (
    "album", "track", "variant", "rel_path", "dr", "dr_exact", "crest_tt",
    "lufs_i", "tp_dbfs", "peak_dbfs", "rms_db", "clipped_samples",
    "clipped_pct", "spikes_gt_0_9", "band_delta_max_db", "cutoff_hz",
    "hf_floor_db", "sr", "bits", "dur_s",
)
NUMERIC_TRACK_COLUMNS = (
    "dr", "dr_exact", "crest_tt", "peak_dbfs", "rms_db", "clipped_samples",
    "clipped_pct", "spikes_gt_0_9", "band_delta_max_db", "sr", "bits", "dur_s",
)
FORBIDDEN_IN_REF = (
    "restore_per_track",
    "rnext.dsp.pipeline",
    "rnext.dsp import pipeline",
    "restore_track(",
    "Policy(",
)
REPORT_HEADINGS = (
    "## criteria",
    "## per-track",
    "## per-album",
    "## verdict",
    "## counterexamples",
    "## reproducibility",
    "## limitations",
)
REQUESTED_ARTISTS = (
    "City and Colour", "Jelusick", "Perfect Plan", "Lost Society",
    "Papa Roach", "Whom Gods Destroy", "Ross The Boss",
)


# ── paths ─────────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "audio_measure.py").is_file():
            return parent
    raise RuntimeError("repo root not found (scripts/audio_measure.py)")


REPO = _repo_root()
SEL = REPO / "scripts" / "experiment" / "select_albums.py"
REF = REPO / "scripts" / "experiment" / "ref_restore.py"
CMP = REPO / "scripts" / "experiment" / "compare_restore.py"
MEASURE = REPO / "scripts" / "audio_measure.py"
METHOD_DOC = REPO / "docs" / "experiment" / "independent-method.md"
REPORT = REPO / "docs" / f"experiment-independent-restore-{DATE}.md"
DATA_DIR = REPO / "docs" / "experiment" / DATE


# ── process / io helpers ──────────────────────────────────────────────────────

def run(args, timeout: int = 1800) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, *[str(a) for a in args]],
        cwd=str(REPO), capture_output=True, text=True, env=env, timeout=timeout,
    )


def ffmpeg(*args) -> subprocess.CompletedProcess:
    found = any((Path(p) / "ffmpeg").is_file() for p in os.environ.get("PATH", "").split(os.pathsep))
    if not found:
        pytest.fail("ffmpeg is required by this repository and is not on PATH")
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *[str(a) for a in args]],
        capture_output=True, text=True, timeout=300,
    )


def hash_tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def write_track(path: Path, seconds: float = 3.0, freq: float = 440.0,
                headroom_db: float = -6.0, clip: bool = False, sr: int = 44100) -> None:
    """Deterministic synthetic stereo track; `clip=True` hard-clips at full scale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    amp = 10.0 ** (headroom_db / 20.0)
    n = int(sr * seconds)
    buf = bytearray()
    for i in range(n):
        t = i / sr
        s = 0.7 * math.sin(2 * math.pi * freq * t) + 0.3 * math.sin(2 * math.pi * freq * 1.5 * t)
        s *= amp
        if clip:
            s = max(-1.0, min(1.0, s * 2.5))
        v = int(max(-32768, min(32767, round(s * 32767.0))))
        buf += struct.pack("<h", v, v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(buf))


def to_flac(wav: Path) -> Path:
    out = wav.with_suffix(".flac")
    r = ffmpeg("-i", wav, "-c:a", "flac", out)
    assert r.returncode == 0, f"fixture flac encode failed: {r.stderr}"
    wav.unlink()
    return out


def make_restore_lib(root: Path, seconds: float = 3.0, seed: int = 0) -> None:
    """3 albums, 4 tracks: two hard-clipped, one 16-bit WAV left as WAV."""
    jitter = (seed % 7) * 11.0
    a = root / "Alpha - Gamma (2020)"
    write_track(a / "01 - Loud.wav", seconds=seconds, freq=220.0 + jitter,
                headroom_db=-3.0, clip=True)
    to_flac(a / "01 - Loud.wav")
    write_track(a / "02 - Quiet.wav", seconds=seconds, freq=330.0 + jitter,
                headroom_db=-12.0)
    to_flac(a / "02 - Quiet.wav")
    b = root / "Alpha - Delta (2021)"
    write_track(b / "01 - Louder.wav", seconds=seconds, freq=180.0 + jitter,
                headroom_db=-3.0, clip=True)
    c = root / "Beta - Epsilon (2019)"
    write_track(c / "01 - Mid.wav", seconds=seconds, freq=500.0 + jitter,
                headroom_db=-6.0)
    to_flac(c / "01 - Mid.wav")


def make_selection_lib(root: Path) -> None:
    for i in range(1, 8):
        d = root / f"Alpha - Album {i} (20{10 + i})"
        write_track(d / "01 - Track.wav", seconds=0.3, freq=300.0 + i)
    for i in range(1, 3):
        d = root / f"Beta - Rec {i} (20{11 + i})"
        write_track(d / "01 - Track.wav", seconds=0.3, freq=400.0 + i)


def audio_files(root: Path) -> list:
    exts = {".flac", ".wav", ".ape", ".m4a", ".mp3", ".ogg", ".opus", ".aiff"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def measure(root: Path, out_json: Path, quick: bool = True) -> list:
    args = [MEASURE, "--json", out_json]
    if quick:
        args.append("--quick")
    args.append(root)
    r = run(args)
    assert r.returncode == 0, f"audio_measure failed: {r.stderr}"
    return json.loads(out_json.read_text(encoding="utf-8"))


def by_stem(rows: list) -> dict:
    return {Path(row["file"]).stem: row for row in rows}


def read_csv(path: Path) -> list:
    assert path.is_file(), f"missing {path}"
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    return float(s)


def inum(value):
    v = fnum(value)
    return None if v is None else int(round(v))


# ── the documented decision procedure, re-implemented for cross-checking ─────

def track_verdict(ours: dict, indep: dict) -> tuple:
    def disqualified(row: dict) -> bool:
        tp = fnum(row.get("tp_dbfs"))
        return tp is not None and tp > -0.9

    o_dis, i_dis = disqualified(ours), disqualified(indep)
    if o_dis and i_dis:
        verdict, reason = "inconclusive", "ceiling"
    elif o_dis:
        verdict, reason = "indep_better", "ceiling"
    elif i_dis:
        verdict, reason = "ours_better", "ceiling"
    else:
        dr_o, dr_i = fnum(ours.get("dr")), fnum(indep.get("dr"))
        assert dr_o is not None and dr_i is not None
        if abs(dr_o - dr_i) >= 0.3:
            verdict = "ours_better" if dr_o > dr_i else "indep_better"
            reason = "dr"
        else:
            sp_o = inum(ours.get("spikes_gt_0_9"))
            sp_i = inum(indep.get("spikes_gt_0_9"))
            if sp_o != sp_i:
                verdict = "ours_better" if sp_o < sp_i else "indep_better"
                reason = "spikes"
            else:
                verdict, reason = "comparable", "tie"

    lu_o, lu_i = fnum(ours.get("lufs_i")), fnum(indep.get("lufs_i"))
    if lu_o is None or lu_i is None:
        comparability = "unknown"
    elif abs(lu_o - lu_i) > 2.0:
        comparability = "loudness_mismatch"
    else:
        comparability = "ok"
    return verdict, reason, comparability


def album_verdict(track_verdicts: list, lufs_ours, lufs_indep) -> tuple:
    count = {k: sum(1 for v in track_verdicts if v == k)
             for k in ("ours_better", "indep_better", "comparable", "inconclusive")}
    if count["ours_better"] > count["indep_better"]:
        verdict, reason = "ours_better", "majority"
    elif count["indep_better"] > count["ours_better"]:
        verdict, reason = "indep_better", "majority"
    elif count["inconclusive"] > 0:
        verdict, reason = "inconclusive", "tie"
    else:
        verdict, reason = "comparable", "tie"
    a, b = fnum(lufs_ours), fnum(lufs_indep)
    if a is None or b is None:
        comparability = "unknown"
    elif abs(a - b) > 2.0:
        comparability = "loudness_mismatch"
    else:
        comparability = "ok"
    return verdict, reason, comparability


# ── 1. deliverables and independence ─────────────────────────────────────────

def test_scripts_exist_and_restore_is_independent() -> None:
    for path in (SEL, REF, CMP):
        assert path.is_file(), f"missing {path.relative_to(REPO)}"

    ref_src = REF.read_text(encoding="utf-8")
    for token in FORBIDDEN_IN_REF:
        assert token not in ref_src, (
            f"ref_restore.py must not reuse the canon method, found {token!r}"
        )

    cmp_src = CMP.read_text(encoding="utf-8")
    assert "audio_measure" in cmp_src, (
        "compare_restore.py must measure through scripts/audio_measure.py, "
        "not a private re-implementation"
    )


def test_method_is_documented() -> None:
    assert METHOD_DOC.is_file(), "missing docs/experiment/independent-method.md"
    text = METHOD_DOC.read_text(encoding="utf-8").lower()
    assert "declip" in text, "the method doc must name the declipping stage"
    assert "lufs" in text and "-14" in text, "the method doc must state the loudness target"
    assert "-1" in text, "the method doc must state the true-peak ceiling"
    for heading in ("## method", "## parameters", "## why it is independent", "## limitations"):
        assert heading in text, f"method doc missing section {heading!r}"


# ── 2. seeded album selection ────────────────────────────────────────────────

def test_select_five_per_artist_is_deterministic(tmp_path: Path) -> None:
    lib = tmp_path / "release_src"
    make_selection_lib(lib)
    before = hash_tree(lib)
    m1, m2 = tmp_path / "m1.json", tmp_path / "m2.json"

    r = run([SEL, "--src", lib, "--per-artist", "5", "--seed", "7", "--out", m1])
    assert r.returncode == 0, f"select_albums failed: {r.stderr}"
    r = run([SEL, "--src", lib, "--per-artist", "5", "--seed", "7", "--out", m2])
    assert r.returncode == 0, f"select_albums failed: {r.stderr}"
    assert m1.read_bytes() == m2.read_bytes(), (
        "selection must be byte-identical for the same seed (no timestamps)"
    )
    assert hash_tree(lib) == before, "selection must not modify the library"

    manifest = json.loads(m1.read_text(encoding="utf-8"))
    assert set(manifest) == {"src", "per_artist", "seed", "albums"}, manifest.keys()
    assert manifest["per_artist"] == 5
    assert int(manifest["seed"]) == 7

    by_artist: dict = {}
    for entry in manifest["albums"]:
        assert set(entry) == {"artist", "album", "rel_path", "n_tracks"}, entry
        by_artist.setdefault(entry["artist"], []).append(entry)

    assert len(by_artist["Alpha"]) == 5, "7 available, cap is 5"
    assert len(by_artist["Beta"]) == 2, "fewer than the cap means all available"

    available_alpha = {p.name for p in lib.iterdir() if p.name.startswith("Alpha - ")}
    chosen_alpha = {e["album"] for e in by_artist["Alpha"]}
    assert chosen_alpha <= available_alpha, "selection must be a subset of the library"
    for entry in manifest["albums"]:
        assert (lib / entry["rel_path"]).is_dir(), entry["rel_path"]
        assert int(entry["n_tracks"]) == 1


# ── 3. independent restore ───────────────────────────────────────────────────

def test_restore_bounds_peak_and_clipping_and_keeps_sources(tmp_path: Path) -> None:
    src = tmp_path / "release_src"
    make_restore_lib(src)
    out, work = tmp_path / "indep", tmp_path / "work"
    before = hash_tree(src)

    r = run([REF, "--src", src, "--out", out, "--work-dir", work])
    assert r.returncode == 0, f"ref_restore failed: {r.stderr}"

    assert hash_tree(src) == before, "sources must be byte-identical after the run"
    assert not work.exists(), "the temp work dir must be removed"

    inputs = audio_files(src)
    assert len(inputs) == 4, inputs
    for path in inputs:
        expected = out / path.relative_to(src).parent / (path.stem + ".flac")
        assert expected.is_file(), f"missing output {expected.relative_to(out)}"

    rows = measure(out, tmp_path / "indep.json")
    assert len(rows) == len(inputs)
    for row in rows:
        assert "error" not in row, row
        assert row["bits"] == 24, f"independent restore writes 24-bit FLAC: {row}"
        assert row["peak_dbfs"] <= -0.8, f"peak above the -1 dBFS ceiling: {row}"
        assert row["clipped_samples"] == 0, f"restore introduced clipping: {row}"
        lufs = row.get("lufs_i")
        if lufs is not None:
            assert -18.0 <= lufs <= -10.0, f"loudness far from the -14 LUFS target: {row}"

    orig = by_stem(measure(src, tmp_path / "orig.json"))
    restored = by_stem(rows)
    clipped_before = [s for s, row in orig.items()
                      if row.get("error") is None and row.get("clipped_samples", 0) > 0]
    assert len(clipped_before) == 2, f"fixture must contain clipped sources: {clipped_before}"
    for stem in clipped_before:
        assert restored[stem]["clipped_samples"] == 0, (
            f"{stem}: clipping must be reduced to zero, "
            f"got {restored[stem]['clipped_samples']}"
        )


def test_restore_refuses_overlap_and_canon_paths(tmp_path: Path) -> None:
    src = tmp_path / "release_src"
    make_restore_lib(src, seconds=1.0)

    inside = src / "indep"
    r = run([REF, "--src", src, "--out", inside])
    assert r.returncode != 0, "writing into the source tree must be refused"
    assert not inside.exists(), "a refused run must not create the output dir"

    canon = tmp_path / "Some Artist - Some Album (2020) [restored per-track]"
    r = run([REF, "--src", src, "--out", canon])
    assert r.returncode != 0, "the canon marker [restored per-track] must be refused"
    assert not canon.exists(), "a refused run must not create the canon dir"


def test_restore_is_reproducible_metricwise(tmp_path: Path) -> None:
    src = tmp_path / "release_src"
    write_track(src / "Alpha - Zeta (2022)" / "01 - One.wav", seconds=1.5,
                freq=250.0, headroom_db=-3.0, clip=True)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    assert run([REF, "--src", src, "--out", out1]).returncode == 0
    assert run([REF, "--src", src, "--out", out2]).returncode == 0
    a = by_stem(measure(out1, tmp_path / "a.json"))
    b = by_stem(measure(out2, tmp_path / "b.json"))
    assert set(a) == set(b)
    for stem in a:
        for key in ("dr", "dr_exact", "crest_tt", "lufs_i", "peak_dbfs", "clipped_samples"):
            va, vb = fnum(a[stem].get(key)), fnum(b[stem].get(key))
            assert va is not None and vb is not None
            assert abs(va - vb) <= 0.01, f"{stem}:{key} not reproducible ({va} vs {vb})"


# ── 4. neutral comparison ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def compared(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("cmp")
    src = root / "release_src"
    make_restore_lib(src)
    indep = root / "indep"
    r = run([REF, "--src", src, "--out", indep])
    assert r.returncode == 0, f"ref_restore failed: {r.stderr}"

    # OURS: untouched ORIG in one album (ceiling breach -> decisive verdict),
    # the INDEP output everywhere else (identical numbers -> tie branch).
    ours = root / "ours"
    for path in audio_files(src):
        rel = path.relative_to(src)
        target = ours / rel.parent / (rel.stem + ".flac")
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.parts[0].startswith("Alpha - Gamma"):
            target.write_bytes(path.read_bytes())
        else:
            mirror = indep / rel.parent / (rel.stem + ".flac")
            target.write_bytes(mirror.read_bytes())

    data = root / "data"
    r = run([CMP, "--src", src, "--ours", ours, "--indep", indep,
             "--out-dir", data])
    assert r.returncode == 0, f"compare_restore failed: {r.stderr}"
    return {"src": src, "ours": ours, "indep": indep, "data": data, "root": root}


def test_compare_emits_documented_schema(compared: dict) -> None:
    data = compared["data"]
    rows = read_csv(data / "tracks.csv")
    assert rows, "tracks.csv is empty"
    assert tuple(rows[0]) == TRACK_COLUMNS, tuple(rows[0])

    order = [(r["album"], r["track"], r["variant"]) for r in rows]
    assert order == sorted(order), "tracks.csv must be sorted by (album, track, variant)"
    assert {r["variant"] for r in rows} == set(VARIANTS)

    keys = [(r["album"], r["track"], r["variant"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate (album, track, variant) rows"

    for row in rows:
        for column in NUMERIC_TRACK_COLUMNS:
            assert fnum(row[column]) is not None, f"empty metric {column} in {row}"
        assert fnum(row["dr"]) >= 0.0
        assert 0.0 <= fnum(row["clipped_pct"]) <= 100.0
        assert fnum(row["peak_dbfs"]) <= 0.5
        if row["variant"] == "orig":
            assert abs(fnum(row["band_delta_max_db"])) <= 0.05, (
                "band_delta_max_db is measured against ORIG, so ORIG is 0"
            )

    albums = read_csv(data / "albums.csv")
    assert albums, "albums.csv is empty"
    for row in albums:
        assert set(row) >= {"album", "artist", "variant", "n_tracks", "dr_median",
                            "crest_median", "lufs_median", "tp_max",
                            "clipped_pct_sum", "spikes_gt_0_9_sum",
                            "band_delta_max_db"}
        assert int(float(row["n_tracks"])) >= 1
    assert {r["variant"] for r in albums} == set(VARIANTS)

    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) >= {"variants", "albums", "tracks"}
    assert set(summary["variants"]) == set(VARIANTS)
    assert int(summary["tracks"]) == len(rows) // len(VARIANTS)
    assert int(summary["albums"]) == 3

    tables = (data / "tables.md").read_text(encoding="utf-8").lower()
    for heading in ("## per-track", "## per-album", "## verdicts"):
        assert heading in tables, f"tables.md missing {heading!r}"


def test_compare_metrics_match_audio_measure(compared: dict) -> None:
    rows = read_csv(compared["data"] / "tracks.csv")
    orig = by_stem(measure(compared["src"], compared["root"] / "parity.json"))
    checked = 0
    for row in rows:
        if row["variant"] != "orig":
            continue
        ref = orig[row["track"]]
        assert "error" not in ref, ref
        assert fnum(row["dr"]) == pytest.approx(ref["dr"], abs=0.05), row
        assert fnum(row["crest_tt"]) == pytest.approx(ref["crest_tt"], abs=0.05), row
        assert fnum(row["peak_dbfs"]) == pytest.approx(ref["peak_dbfs"], abs=0.05), row
        assert inum(row["clipped_samples"]) == int(ref["clipped_samples"]), row
        checked += 1
    assert checked == 4, checked


def test_verdicts_follow_the_documented_rule(compared: dict) -> None:
    data = compared["data"]
    rows = read_csv(data / "tracks.csv")
    keyed = {(r["album"], r["track"], r["variant"]): r for r in rows}

    verdicts = read_csv(data / "verdicts.csv")
    assert verdicts, "verdicts.csv is empty"
    assert tuple(verdicts[0]) == ("album", "track", "verdict", "reason", "comparability")
    assert {r["verdict"] for r in verdicts} <= {
        "ours_better", "indep_better", "comparable", "inconclusive"}

    written = {(r["album"], r["track"]): r for r in verdicts}
    expected_keys = {(a, t) for (a, t, _v) in keyed}
    assert set(written) == expected_keys, "one verdict row per (album, track)"

    for (album, track), row in written.items():
        want = track_verdict(keyed[(album, track, "ours")], keyed[(album, track, "indep")])
        got = (row["verdict"], row["reason"], row["comparability"])
        assert got == want, f"{album}/{track}: verdict {got} != rule {want}"

    assert any(r["reason"] == "ceiling" for r in verdicts), (
        "the ceiling disqualification branch was never exercised"
    )

    album_rows = read_csv(data / "album_verdicts.csv")
    assert album_rows, "album_verdicts.csv is empty"
    assert tuple(album_rows[0]) == ("album", "artist", "verdict", "reason", "comparability")
    assert {r["album"] for r in album_rows} == {a for (a, _t) in expected_keys}, (
        "every album in tracks.csv needs an album verdict row"
    )

    albums_metrics = {(r["album"], r["variant"]): r
                      for r in read_csv(data / "albums.csv")}
    for row in album_rows:
        album = row["album"]
        tracks = [v["verdict"] for k, v in written.items() if k[0] == album]
        want = album_verdict(tracks,
                             albums_metrics[(album, "ours")]["lufs_median"],
                             albums_metrics[(album, "indep")]["lufs_median"])
        got = (row["verdict"], row["reason"], row["comparability"])
        assert got == want, f"{album}: album verdict {got} != rule {want}"


def test_compare_refuses_incomplete_triple(tmp_path: Path) -> None:
    src = tmp_path / "release_src"
    make_restore_lib(src, seconds=0.6)
    indep, ours = tmp_path / "indep", tmp_path / "ours"
    assert run([REF, "--src", src, "--out", indep]).returncode == 0
    for path in audio_files(indep):
        rel = path.relative_to(indep)
        target = ours / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    missing = sorted(ours.rglob("*.flac"))[0]
    missing.unlink()

    data = tmp_path / "data"
    r = run([CMP, "--src", src, "--ours", ours, "--indep", indep,
             "--out-dir", data])
    assert r.returncode != 0, "a missing counterpart must be a hard error"
    assert r.stderr.strip(), "the error must be explicit on stderr"
    assert not (data / "tracks.csv").exists(), (
        "no partial CSV: a missing counterpart must not turn into a silent row"
    )


# ── 5. committed real-corpus artifacts ───────────────────────────────────────

def test_real_report_and_data_are_consistent() -> None:
    assert REPORT.is_file(), f"missing {REPORT.relative_to(REPO)}"
    text = REPORT.read_text(encoding="utf-8")
    lowered = text.lower()
    for heading in REPORT_HEADINGS:
        assert heading in lowered, f"report missing section {heading!r}"
    assert "TODO" not in text.upper(), "report must not contain stubs"
    assert len(text) >= 3000, "report is too thin to be an analysis"

    tracks = read_csv(DATA_DIR / "tracks.csv")
    albums = read_csv(DATA_DIR / "albums.csv")
    assert read_csv(DATA_DIR / "verdicts.csv"), "verdicts.csv is empty"
    assert read_csv(DATA_DIR / "album_verdicts.csv"), "album_verdicts.csv is empty"
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"src", "per_artist", "seed", "albums"}
    assert manifest["albums"], "manifest lists the selected albums"

    assert tuple(tracks[0]) == TRACK_COLUMNS, tuple(tracks[0])
    assert {r["variant"] for r in tracks} >= set(VARIANTS)
    keys = [(r["album"], r["track"], r["variant"]) for r in tracks]
    assert len(keys) == len(set(keys)), "duplicate rows in the committed data"

    artists = {r["artist"] for r in albums}
    for name in REQUESTED_ARTISTS:
        assert name in artists, f"report corpus is missing artist {name!r}"

    distinct = {(r["artist"], r["album"]) for r in albums}
    assert len(distinct) >= 8, f"too few albums in the experiment: {len(distinct)}"
    per_artist: dict = {}
    for artist, album in distinct:
        per_artist.setdefault(artist, set()).add(album)
    for artist, chosen in per_artist.items():
        assert len(chosen) <= 5, f"{artist}: more than 5 albums selected"

    for artist, album in distinct:
        assert album in text, f"album {album!r} of {artist} is not mentioned in the report"

    assert "scripts/experiment/ref_restore.py" in text, "reproducible commands are required"
    assert "scripts/audio_measure.py" in text, "reproducible commands are required"
    assert "scripts/experiment/select_albums.py" in text
    assert "scripts/experiment/compare_restore.py" in text


# ── 6. seeded property pass ──────────────────────────────────────────────────

def test_properties_invariants_per_seed(tmp_path: Path) -> None:
    """One seed per invocation; the gate runs this 16 times with SEED=0..15."""
    seed = int(os.environ.get("SEED", "0"))
    src = tmp_path / f"lib{seed}"
    make_restore_lib(src, seconds=1.0, seed=seed)
    before = hash_tree(src)

    indep, ours, data = tmp_path / "indep", tmp_path / "ours", tmp_path / "data"
    assert run([REF, "--src", src, "--out", indep]).returncode == 0
    assert hash_tree(src) == before, "sources untouched"

    for path in audio_files(indep):
        rel = path.relative_to(indep)
        target = ours / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    r = run([CMP, "--src", src, "--ours", ours, "--indep", indep, "--out-dir", data])
    assert r.returncode == 0, r.stderr

    rows = read_csv(data / "tracks.csv")
    assert len(rows) == 4 * len(VARIANTS)
    assert len({(r["album"], r["track"], r["variant"]) for r in rows}) == len(rows)
    for row in rows:
        assert fnum(row["peak_dbfs"]) <= 0.5, row
        assert 0 <= fnum(row["clipped_pct"]) <= 100, row
        assert fnum(row["dur_s"]) > 0, row

    verdicts = read_csv(data / "verdicts.csv")
    assert len(verdicts) == 4
    for row in verdicts:
        assert row["verdict"] == "comparable" and row["reason"] == "tie", row
        assert row["comparability"] == "ok", row
