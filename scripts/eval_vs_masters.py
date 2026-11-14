#!/usr/bin/env python3
"""Reproducible eval harness: our per-track restoration vs master restorations.

Turns "how close is our restoration to the masters" into numbers anyone can
reproduce.  Every metric comes from scripts/audio_measure.py (DR, crest, LUFS,
true peak, clipping; plus the spike count and the 1/3-octave band vector from
the same float decode, as in scripts/experiment/compare_restore.py).  A missing
side, an empty side, an unreadable corpus or a metric absent from the
measurement output exits 2 with the offending path on stderr and writes
nothing: tables are rendered only after every release measured clean.

  python3 scripts/eval_vs_masters.py --manifest M.json --out-dir OUT [--report R.md]
  python3 scripts/eval_vs_masters.py --select --corpus A.jsonl [--corpus B.jsonl] \
          --seed 20261104 --per-master 5 --out PICKS.json

Manifest: {"seed": <int|null>, "releases": [{"release": str, "orig": path,
"ours": path, "master": path}]}; relative paths resolve against the manifest's
directory.  Tracks are matched across the three sides by the leading digits of
the file stem; unmatched files are recorded with reason `track_mismatch` and
excluded from every mean (never zero-filled).  Deltas are `side - master` and
the `abs_delta_*` columns are never negative; `band_rms_db` is the exception
documented below.  Outputs in --out-dir: tracks.csv, deltas.csv, albums.csv,
summary.json.

`band_rms_db` is a per-file 30-band vector, so its delta is pairwise: the delta
columns hold the mean absolute band delta versus the master side, while the
`orig`/`ours`/`master` value columns hold that side's mean band level.  LUFS is
a loudness metric: it takes part in the contract's majority vote (summary.json)
but the report presents it separately as comparability, not quality.

`--select` picks releases from rutracker-corpus jsonl files (one JSON object per
line with topic_id, title, seeds): seeds >= 3, single albums preferred over
discographies, per-master picks, explicit --seed required (never defaulted).
The same inputs give byte-identical picks.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import audio_measure  # noqa: E402  (single source of truth for the metrics)

REPO_ROOT = SCRIPTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from rnext import features as F  # noqa: E402


class EvalError(Exception):
    """Bad input or missing metric: abort with exit 2 and write nothing."""


METRICS = ("dr14", "crest_db", "lufs", "true_peak_db",
           "clip_samples", "spikes_over_09", "band_rms_db")
COUNT_METRICS = frozenset({"clip_samples", "spikes_over_09"})


SIDES = ("orig", "ours", "master")
TOL = 0.05
DISCO_RE = re.compile(r"(дискограф|discograph|сборник|collection|anthology|box\s*set)", re.IGNORECASE)


def verdict(abs_ours, abs_orig, tol=TOL):
    """Compare distance to the master: smaller |delta| means closer."""
    if abs_ours < abs_orig * (1.0 - tol):
        return "better"
    if abs_ours > abs_orig * (1.0 + tol):
        return "worse"
    return "comparable"


def overall_verdict(per_metric):
    better = sum(1 for v in per_metric.values() if v == "better")
    worse = sum(1 for v in per_metric.values() if v == "worse")
    if better > worse:
        return "better"
    if worse > better:
        return "worse"
    return "comparable"


def pick_releases(records, seed, per_master=5):
    """Seeded, deterministic pick: seeds>=3 first, single albums before discographies."""
    eligible = [r for r in records if int(r["seeds"]) >= 3]
    eligible.sort(key=lambda r: int(r["topic_id"]))
    single = [r for r in eligible if not DISCO_RE.search(str(r.get("title", "")))]
    rnd = random.Random(seed)
    picks = rnd.sample(single, min(per_master, len(single)))
    taken = {int(r["topic_id"]) for r in picks}
    rest = [r for r in eligible if int(r["topic_id"]) not in taken]
    need = per_master - len(picks)
    if need > 0:
        picks += rnd.sample(rest, min(need, len(rest)))
    return sorted(picks, key=lambda r: int(r["topic_id"]))


# ── measurement ──────────────────────────────────────────────────────────────

TRACK_NUM_RE = re.compile(r"\s*(\d{1,3})")


def _measure_file(path: Path) -> dict:
    """audio_measure metrics for one file plus spikes and the band vector.

    spikes_over_09 counts samples with |x| > 0.9 in the first ~90 s, the same
    definition the independent-restore experiment used; band_rms_db is
    rnext.features.band_rms_db on that decode.  The decode keeps NATIVE
    channels: audio_measure.decode() remixes mono to stereo, and this ffmpeg
    build attenuates that remix by -3 dB, which would hide every full-scale
    spike of a mono take; stereo files (the corpora) decode identically.
    """
    row = audio_measure.analyse(path, quick=False)
    if row.get("error"):
        raise EvalError(f"{path}: does not decode ({row['error']})")
    pairs = (("dr14", "dr"), ("crest_db", "crest_tt"), ("lufs", "lufs_i"),
             ("true_peak_db", "tp_dbfs"), ("clip_samples", "clipped_samples"))
    missing = [name for name, key in pairs if row.get(key) is None]
    if missing:
        raise EvalError(f"{path}: audio_measure returned no {', '.join(missing)}")
    x, sr = F.decode_f32(str(path), 0.0, 90.0, channels=None)
    if x is None or x.size == 0:
        raise EvalError(f"{path}: cannot decode for spikes/bands")
    bands = F.band_rms_db(x, sr)
    if not bands:
        raise EvalError(f"{path}: band_rms_db returned nothing")
    return {
        "dr14": row["dr"],
        "crest_db": row["crest_tt"],
        "lufs": float(row["lufs_i"]),
        "true_peak_db": float(row["tp_dbfs"]),
        "clip_samples": int(row["clipped_samples"]),
        "spikes_over_09": int(np.sum(np.abs(x) > 0.9)),
        "band_rms_db": [float(b) for b in bands],
    }


def _side_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in audio_measure.AUDIO_EXT)


def measure_side(path) -> dict:
    """Aggregate metrics for one side directory.

    Continuous metrics average over files, counts sum, the band vector
    averages per band; a single-file directory yields that file's values.
    """
    root = Path(path)
    if not root.is_dir():
        raise EvalError(f"side directory is missing or not a directory: {root}")
    files = _side_files(root)
    if not files:
        raise EvalError(f"side directory has no audio files: {root}")
    per_file = [_measure_file(p) for p in files]
    agg: dict = {}
    for metric in METRICS:
        values = [m[metric] for m in per_file]
        if metric == "band_rms_db":
            agg[metric] = [float(np.mean(cols)) for cols in zip(*values)]
        elif metric in COUNT_METRICS:
            agg[metric] = int(sum(values))
        else:
            agg[metric] = float(np.mean(values))
    return agg


# ── track matching ───────────────────────────────────────────────────────────


def track_number(stem: str) -> int | None:
    m = TRACK_NUM_RE.match(stem)
    return int(m.group(1)) if m else None


def index_side(root: Path, side: str, release: str) -> dict[int, Path]:
    """Map track number -> file for one side; duplicates fail loudly."""
    files = _side_files(root)
    if not files:
        raise EvalError(f"release {release!r}: side {side} has no audio files: {root}")
    by_num: dict[int, Path] = {}
    for p in files:
        n = track_number(p.stem)
        if n is None:
            continue
        if n in by_num:
            raise EvalError(f"release {release!r}: side {side} has two tracks numbered "
                            f"{n:02d} in {root}: {by_num[n].name} and {p.name}")
        by_num[n] = p
    if not by_num:
        raise EvalError(f"release {release!r}: side {side} has no numbered tracks "
                        f"to match on (leading digits of the stem) in {root}")
    return by_num


# ── measurement run ──────────────────────────────────────────────────────────


def load_manifest(path: Path) -> tuple[int | None, list[dict]]:
    if not path.is_file():
        raise EvalError(f"manifest file is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvalError(f"manifest is not readable JSON: {path} ({exc})")
    releases = data.get("releases")
    if not isinstance(releases, list) or not releases:
        raise EvalError(f"manifest has no non-empty releases list: {path}")
    seed = data.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise EvalError(f"manifest seed must be an integer or null: {path}")
    base = path.resolve().parent
    plans = []
    for rel in releases:
        name = rel.get("release")
        if not isinstance(name, str) or not name.strip():
            raise EvalError(f"manifest release entry without a name: {rel!r} in {path}")
        sides = {}
        for side in SIDES:
            raw = rel.get(side)
            if not isinstance(raw, str) or not raw.strip():
                raise EvalError(f"manifest release {name!r}: missing {side} path")
            p = Path(raw).expanduser()
            sides[side] = p if p.is_absolute() else base / p
        plans.append({"release": name, "sides": sides})
    return seed, plans


def run_measurement(manifest: Path) -> dict:
    """Measure every release; returns the in-memory data behind all outputs."""
    seed, plans = load_manifest(manifest)
    releases_out = []
    track_rows: list[dict] = []   # per matched track: {release, track, sides: {side: metrics}}
    for plan in plans:
        name = plan["release"]
        indexes = {side: index_side(plan["sides"][side], side, name) for side in SIDES}
        common = sorted(set.intersection(*(set(idx) for idx in indexes.values())))
        skipped = []
        for side in SIDES:
            for stem, path in ((p.stem, p) for p in plan["sides"][side].rglob("*")
                               if p.is_file() and p.suffix.lower() in audio_measure.AUDIO_EXT):
                n = track_number(stem)
                if n is None or n not in indexes[side] or n not in set(common):
                    skipped.append({"track": stem, "side": side, "reason": "track_mismatch"})
        if not common:
            releases_out.append({"release": name, "tracks": 0,
                                 "status": "skipped: track_mismatch", "skipped": skipped})
            print(f"eval_vs_masters: {name}: skipped (no track matches all three sides)",
                  flush=True)
            continue
        sides_metrics: dict[int, dict] = {}
        for n in common:
            sides_metrics[n] = {side: _measure_file(indexes[side][n]) for side in SIDES}
        label = {n: indexes["orig"][n].stem for n in common}
        track_rows.append({"release": name, "label": label,
                           "tracks": {n: sides_metrics[n] for n in common}})
        releases_out.append({"release": name, "tracks": len(common),
                             "status": "measured", "skipped": skipped})
        print(f"eval_vs_masters: {name}: {len(common)} tracks matched", flush=True)
    return {"seed": seed, "releases": releases_out, "track_rows": track_rows}


def band_delta(vec_a: list[float], vec_b: list[float], where: str) -> float:
    if len(vec_a) != len(vec_b):
        raise EvalError(f"{where}: band count differs between sides "
                        f"({len(vec_a)} vs {len(vec_b)})")
    return float(np.mean(np.abs(np.asarray(vec_a) - np.asarray(vec_b))))


def build_tables(data: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """deltas.csv rows, albums.csv rows and the per_metric summary block."""
    delta_rows: list[dict] = []
    album_rows: list[dict] = []
    metric_abs_ours: dict[str, list[float]] = {m: [] for m in METRICS}
    metric_abs_orig: dict[str, list[float]] = {m: [] for m in METRICS}
    for entry in data["track_rows"]:
        release = entry["release"]
        per_metric_tracks: dict[str, dict[str, list[float]]] = {
            m: {"orig": [], "ours": [], "master": [],
                "abs_ours": [], "abs_orig": []} for m in METRICS}
        for n in sorted(entry["tracks"]):
            sides = entry["tracks"][n]
            for metric in METRICS:
                o, u, s = (sides["orig"][metric], sides["ours"][metric],
                           sides["master"][metric])
                if metric == "band_rms_db":
                    d_ours = band_delta(u, s, f"{release} track {n:02d}")
                    d_orig = band_delta(o, s, f"{release} track {n:02d}")
                    vo, vu, vs = float(np.mean(o)), float(np.mean(u)), float(np.mean(s))
                else:
                    d_ours = float(u) - float(s)
                    d_orig = float(o) - float(s)
                    vo, vu, vs = float(o), float(u), float(s)
                per_metric_tracks[metric]["orig"].append(vo)
                per_metric_tracks[metric]["ours"].append(vu)
                per_metric_tracks[metric]["master"].append(vs)
                per_metric_tracks[metric]["abs_ours"].append(abs(d_ours))
                per_metric_tracks[metric]["abs_orig"].append(abs(d_orig))
                delta_rows.append({
                    "release": release, "track": entry["label"][n], "metric": metric,
                    "orig": vo, "ours": vu, "master": vs,
                    "delta_ours_master": d_ours, "delta_orig_master": d_orig,
                    "abs_delta_ours_master": abs(d_ours),
                    "abs_delta_orig_master": abs(d_orig),
                })
        for metric in METRICS:
            t = per_metric_tracks[metric]
            abs_ours = float(np.mean(t["abs_ours"]))
            abs_orig = float(np.mean(t["abs_orig"]))
            album_rows.append({
                "release": release, "metric": metric,
                "orig": float(np.mean(t["orig"])),
                "ours": float(np.mean(t["ours"])),
                "master": float(np.mean(t["master"])),
                "abs_delta_ours_master": abs_ours,
                "abs_delta_orig_master": abs_orig,
                "verdict": verdict(abs_ours, abs_orig),
            })
            metric_abs_ours[metric].append(abs_ours)
            metric_abs_orig[metric].append(abs_orig)
    per_metric = {}
    for metric in METRICS:
        abs_ours = float(np.mean(metric_abs_ours[metric]))
        abs_orig = float(np.mean(metric_abs_orig[metric]))
        per_metric[metric] = {"abs_ours": abs_ours, "abs_orig": abs_orig,
                              "verdict": verdict(abs_ours, abs_orig)}
    return delta_rows, album_rows, per_metric


# ── rendering ────────────────────────────────────────────────────────────────

TRACK_COLUMNS = ("release", "track", "side", "dr14", "crest_db", "lufs",
                 "true_peak_db", "clip_samples", "spikes_over_09")
DELTA_COLUMNS = ("release", "track", "metric", "orig", "ours", "master",
                 "delta_ours_master", "delta_orig_master",
                 "abs_delta_ours_master", "abs_delta_orig_master")
ALBUM_COLUMNS = ("release", "metric", "orig", "ours", "master",
                 "abs_delta_ours_master", "abs_delta_orig_master", "verdict")


def fmt(value, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, columns: tuple, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in columns})


def render_report(data: dict, delta_rows: list[dict], album_rows: list[dict],
                  per_metric: dict, manifest: Path) -> str:
    measured = [r for r in data["releases"] if r["status"] == "measured"]
    skipped = [r for r in data["releases"] if r["status"] != "measured"]
    seed = data["seed"]
    lines = ["# Eval: our restoration vs master restorations", ""]
    lines.append(f"Measured {len(measured)} release(s), skipped {len(skipped)}. "
                 f"Selection seed: {seed if seed is not None else 'not set (manifest)'}")
    for rel in skipped:
        why = [f"{s['track']} ({s['side']})" for s in rel["skipped"]] or rel["status"]
        lines.append(f"- skipped: {rel['release']} — {rel['status']}: {', '.join(why)}")
    lines += ["", "## Verdict", "",
              "Distance to the master per metric, averaged over all measured "
              "tracks: |ours − master| against the |orig − master| baseline. "
              "`better` = our gap is smaller beyond the 5% tolerance.", ""]
    lines.append("| metric | abs ours−master | abs orig−master | verdict |")
    lines.append("|---|---|---|---|")
    quality = []
    for metric in METRICS:
        cell = per_metric[metric]
        lines.append(f"| {metric} | {fmt(cell['abs_ours'])} | {fmt(cell['abs_orig'])} "
                     f"| {cell['verdict']} |")
        quality.append(cell["verdict"])
    overall = overall_verdict(per_metric)
    lines += ["", f"verdict_overall: **{overall}** (majority vote over all seven "
              "metrics, per the tool contract).", "",
              "LUFS is a loudness metric: read its row as comparability, not "
              "quality — our pipeline targets a different release loudness than "
              "the masters by design (see docs/eval-vs-masters-run.md).", ""]
    lines += ["## Per track", "",
              "Per-track gaps; for `band_rms_db` the delta is the mean absolute "
              "1/3-octave band delta versus the master side.", ""]
    lines.append("| release | track | metric | orig | ours | master | "
                 "|Δ| ours−master | |Δ| orig−master |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in delta_rows:
        lines.append(
            f"| {row['release']} | {row['track']} | {row['metric']} | {fmt(row['orig'])} "
            f"| {fmt(row['ours'])} | {fmt(row['master'])} "
            f"| {fmt(row['abs_delta_ours_master'])} | {fmt(row['abs_delta_orig_master'])} |")
    lines += ["", "## Per album", ""]
    lines.append("| release | metric | orig | ours | master | |Δ| ours | |Δ| orig | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in album_rows:
        lines.append(
            f"| {row['release']} | {row['metric']} | {fmt(row['orig'])} | {fmt(row['ours'])} "
            f"| {fmt(row['master'])} | {fmt(row['abs_delta_ours_master'])} "
            f"| {fmt(row['abs_delta_orig_master'])} | {row['verdict']} |")
    lines += ["", "## Reproduce", "",
              "```bash", f"python3 scripts/eval_vs_masters.py --manifest {manifest} "
              f"--out-dir <out-dir> --report <report.md>",
              "python3 scripts/eval_vs_masters.py --select --corpus <corpus.jsonl> "
              "--seed <seed> --per-master 5 --out picks.json", "```", "",
              "Measurement: scripts/audio_measure.py (DR/crest/clipping on the first "
              "~90 s, LUFS/TP over the whole file via ffmpeg ebur128); spikes = "
              "samples with |x| > 0.9 in that decode; band vector = "
              "rnext.features.band_rms_db (30 bands, 20 Hz…20 kHz). Track matching "
              "by the leading digits of the file stem; unmatched tracks are "
              "recorded as track_mismatch and excluded from every mean.", ""]
    return "\n".join(lines)


# ── selection mode ───────────────────────────────────────────────────────────


def read_corpus(path: Path) -> list[dict]:
    if not path.is_file():
        raise EvalError(f"corpus file is missing: {path}")
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvalError(f"corpus file is unreadable: {path} ({exc})")
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError as exc:
            raise EvalError(f"{path}:{i}: not valid JSON ({exc})")
        if not isinstance(rec, dict):
            raise EvalError(f"{path}:{i}: corpus record is not a JSON object")
        for field in ("topic_id", "seeds"):
            if field not in rec:
                raise EvalError(f"{path}:{i}: corpus record is missing {field}")
            if isinstance(rec[field], bool) or not isinstance(rec[field], (int, float, str)):
                raise EvalError(f"{path}:{i}: non-numeric {field}: {rec[field]!r}")
            try:
                int(rec[field])
            except (TypeError, ValueError):
                raise EvalError(f"{path}:{i}: non-numeric {field}: {rec[field]!r}")
        records.append(rec)
    if not records:
        raise EvalError(f"corpus file has no records: {path}")
    return records


def run_select(corpus_paths: list[str], seed: int, per_master: int, out: Path) -> None:
    picks = []
    for raw in corpus_paths:
        path = Path(raw).expanduser()
        records = read_corpus(path)
        for rec in pick_releases(records, seed, per_master):
            picks.append({"file": str(path), "topic_id": int(rec["topic_id"]),
                          "title": str(rec.get("title", "")), "seeds": int(rec["seeds"])})
    picks.sort(key=lambda p: p["topic_id"])
    payload = {"seed": int(seed), "per_master": int(per_master), "picks": picks}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"eval_vs_masters: {len(picks)} picks for seed {seed} → {out}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="eval_vs_masters",
        description="Reproducible eval: our restoration vs master restorations.")
    ap.add_argument("--manifest", help="manifest JSON with release/side directories")
    ap.add_argument("--out-dir", help="directory for tracks.csv/deltas.csv/albums.csv/summary.json")
    ap.add_argument("--report", default="", help="optional markdown report path")
    ap.add_argument("--select", action="store_true",
                    help="selection mode: pick releases from corpus jsonl files")
    ap.add_argument("--corpus", action="append", default=[],
                    help="corpus jsonl (one JSON object per line); repeatable")
    ap.add_argument("--seed", type=int, default=None,
                    help="selection seed; required for --select, never defaulted")
    ap.add_argument("--per-master", type=int, default=5,
                    help="picks per corpus file (default 5)")
    ap.add_argument("--out", default="", help="picks.json path for --select")
    args = ap.parse_args(argv)

    try:
        if args.select:
            if args.seed is None:
                raise EvalError("--select requires an explicit --seed (never defaulted)")
            if not args.corpus:
                raise EvalError("--select requires at least one --corpus jsonl file")
            if not args.out:
                raise EvalError("--select requires --out for the picks file")
            run_select(args.corpus, args.seed, args.per_master, Path(args.out))
            return 0

        if not args.manifest or not args.out_dir:
            raise EvalError("measurement mode requires --manifest and --out-dir")
        data = run_measurement(Path(args.manifest).expanduser())
        if not data["track_rows"]:
            raise EvalError("no release matched any track across the three sides; "
                            "nothing to compare")
        delta_rows, album_rows, per_metric = build_tables(data)
        measured = sum(1 for r in data["releases"] if r["status"] == "measured")
        if measured == 0:
            raise EvalError("all releases were skipped; nothing to compare")
        summary = {
            "seed": data["seed"],
            "releases": data["releases"],
            "per_metric": per_metric,
            "verdict_overall": overall_verdict(per_metric),
        }
        track_csv_rows = []
        for entry in data["track_rows"]:
            for n in sorted(entry["tracks"]):
                sides = entry["tracks"][n]
                for side in SIDES:
                    m = sides[side]
                    track_csv_rows.append({
                        "release": entry["release"], "track": entry["label"][n],
                        "side": side, "dr14": str(int(m["dr14"])),
                        "crest_db": fmt(m["crest_db"]), "lufs": fmt(m["lufs"], 1),
                        "true_peak_db": fmt(m["true_peak_db"]),
                        "clip_samples": str(int(m["clip_samples"])),
                        "spikes_over_09": str(int(m["spikes_over_09"]))})
        delta_csv_rows = [{
            "release": r["release"], "track": r["track"], "metric": r["metric"],
            "orig": fmt(r["orig"]), "ours": fmt(r["ours"]), "master": fmt(r["master"]),
            "delta_ours_master": fmt(r["delta_ours_master"]),
            "delta_orig_master": fmt(r["delta_orig_master"]),
            "abs_delta_ours_master": fmt(r["abs_delta_ours_master"]),
            "abs_delta_orig_master": fmt(r["abs_delta_orig_master"]),
        } for r in delta_rows]
        album_csv_rows = [{
            "release": r["release"], "metric": r["metric"],
            "orig": fmt(r["orig"]), "ours": fmt(r["ours"]), "master": fmt(r["master"]),
            "abs_delta_ours_master": fmt(r["abs_delta_ours_master"]),
            "abs_delta_orig_master": fmt(r["abs_delta_orig_master"]),
            "verdict": r["verdict"],
        } for r in album_rows]
        # Every input error has been raised by now: only now touch the disk.
        out_dir = Path(args.out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "tracks.csv", TRACK_COLUMNS, track_csv_rows)
        write_csv(out_dir / "deltas.csv", DELTA_COLUMNS, delta_csv_rows)
        write_csv(out_dir / "albums.csv", ALBUM_COLUMNS, album_csv_rows)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.report:
            report = Path(args.report).expanduser()
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(render_report(data, delta_rows, album_rows, per_metric,
                                            Path(args.manifest).expanduser()),
                              encoding="utf-8")
        print(f"eval_vs_masters: {measured} release(s) measured "
              f"→ {out_dir} (verdict_overall: {summary['verdict_overall']})", flush=True)
        return 0
    except EvalError as exc:
        print(f"eval_vs_masters: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
