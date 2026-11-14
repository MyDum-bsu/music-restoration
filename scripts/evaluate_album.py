#!/usr/bin/env python3
"""Album-level evaluation: our restoration against its own source.

One command answers, for one album, "did the restoration actually turn out
good?": every metric comes from scripts/audio_measure.py::analyse() plus the
spike count and the 1/3-octave band vector from the same float decode
(scripts/eval_vs_masters.measure_file — the single measurement kernel), the
verdict procedure is fixed before the data, and the report separates what the
metrics prove from what they cannot (listening preference).

  python3 scripts/evaluate_album.py --src <original-dir> --restored <restored-dir> \
          --out <dir> [--tp-ceiling -0.9] [--dr-threshold 0.3] [--loudness-mismatch-lu 2.0]

Verdict procedure per track, declared before the data:

  1. OUR output constraint: true peak above --tp-ceiling loses immediately
     (reason=ceiling).  The original is not judged by the ceiling — a hot
     master is expected;
  2. DR gap >= --dr-threshold: the higher DR wins (reason=dr);
  3. otherwise the spike count: more click-like sample jumps lose (reason=spikes);
  4. otherwise comparable/tie.

LUFS never votes: |ours − orig| beyond --loudness-mismatch-lu marks the track
`loudness_mismatch` — comparability, not quality.

Tracks are matched by the leading digits of the file stem first, then by an
exact match of the normalized name (parenthesized segments stripped, so
"Song (Japan bonus track)" matches "Song").  Unmatched files are reported as
skipped with a reason — never zero-filled, never silently dropped.  When the
restored directory carries restore_safe's `report.json`, its per-track
`status`/`needs_review`/`restoration_state` are lifted into the verdicts.

Outputs in --out-dir: tracks.csv, verdicts.csv, summary.json, REPORT.md.
Bad input (missing/empty directory, undecodable file, missing metric) exits 2
with the offending path on stderr and writes nothing.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import audio_measure  # noqa: E402
import eval_vs_masters as EV  # noqa: E402  (the measurement kernel + EvalError)


PEAK_CEILING_DBFS = -0.9
DR_THRESHOLD = 0.3
LOUDNESS_MISMATCH_LU = 2.0
TP_AT_CEILING_TOL = 0.05
DR_REGRESSION_DB = 0.25

LIMITATIONS = [
    "DR, crest, clipping and spikes are measured on the first ~90 s of each "
    "file (audio_measure.decode window); LUFS and true peak cover the whole file.",
    "LUFS gap is release loudness by design, not a quality verdict.",
    "Metrics are engineering proxies: they prove output constraints and "
    "dynamics/spectrum shifts, not listening preference; blind listening "
    "(scripts/listen_pack.py) is a separate loop.",
    "Sample-jump counts are headroom information: the verdict fails only on "
    "growth versus the source. A master legitimately keeps hotter transients "
    "at the same loudness; punch retention (crest change) is scale-invariant, "
    "so it reads the same at any matched loudness.",
    "band_shift_db mixes level change with tonal change; read it as an "
    "indicator, not a measurement of timbre.",
    "The verdict ladder compares against the source take, not against a "
    "master restoration or the original studio intent.",
]


def normalize_name(stem: str) -> str:
    """Lowercased stem without parenthesized segments or punctuation runs."""
    s = stem.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def _name_key(path: Path) -> str:
    """Normalized name WITHOUT the leading track number (the name fallback)."""
    return normalize_name(re.sub(r"^[\s\d.\-–—_]+", "", path.stem))


def match_tracks(src_files: list[Path], res_files: list[Path]):
    """Match source and restored takes: by number first, then by exact name.

    Returns (matched, skipped): matched is a list of
    (label, src_path, res_path, via) with via in {"number", "name"}; skipped
    lists {"track", "side", "reason"} for everything left over.
    """
    src_by_num: dict[int, Path] = {}
    for p in src_files:
        n = EV.track_number(p.stem)
        if n is None:
            continue
        if n in src_by_num:
            raise EV.EvalError(f"src has two tracks numbered {n:02d}: "
                               f"{src_by_num[n].name} and {p.name}")
        src_by_num[n] = p
    res_by_num: dict[int, Path] = {}
    for p in res_files:
        n = EV.track_number(p.stem)
        if n is None:
            continue
        if n in res_by_num:
            raise EV.EvalError(f"restored has two tracks numbered {n:02d}: "
                               f"{res_by_num[n].name} and {p.name}")
        res_by_num[n] = p

    matched = []
    skipped = []
    used_src: set[Path] = set()
    used_res: set[Path] = set()
    for n in sorted(set(src_by_num) & set(res_by_num)):
        s, r = src_by_num[n], res_by_num[n]
        matched.append((s.stem, s, r, "number"))
        used_src.add(s)
        used_res.add(r)

    res_left = [p for p in res_files if p not in used_res]
    by_name: dict[str, list[Path]] = {}
    for p in res_left:
        key = _name_key(p)
        if key:
            by_name.setdefault(key, []).append(p)
    src_left = [p for p in src_files if p not in used_src]
    for p in src_left:
        key = _name_key(p)
        cands = by_name.get(key, []) if key else []
        cands = [c for c in cands if c not in used_res]
        if len(cands) == 1:
            matched.append((p.stem, p, cands[0], "name"))
            used_src.add(p)
            used_res.add(cands[0])
        elif len(cands) > 1:
            skipped.append({"track": p.stem, "side": "src",
                            "reason": "ambiguous_name_match"})
            used_src.add(p)
    for p in src_left:
        if p not in used_src:
            skipped.append({"track": p.stem, "side": "src", "reason": "track_mismatch"})
    for p in res_files:
        if p not in used_res:
            skipped.append({"track": p.stem, "side": "res", "reason": "track_mismatch"})
    return matched, skipped


def track_verdict(ours: dict, orig: dict, tp_ceiling: float = PEAK_CEILING_DBFS,
                  dr_threshold: float = DR_THRESHOLD,
                  loudness_mismatch_lu: float = LOUDNESS_MISMATCH_LU):
    """The declared ladder: ceiling (ours only) -> DR gap -> spikes -> tie."""
    tp = ours.get("true_peak_db")
    if tp is not None and tp > tp_ceiling:
        verdict, reason = "orig_better", "ceiling"
    else:
        dr_o, dr_i = float(ours["dr14"]), float(orig["dr14"])
        if dr_o - dr_i >= dr_threshold:
            verdict, reason = "ours_better", "dr"
        elif dr_i - dr_o >= dr_threshold:
            verdict, reason = "orig_better", "dr"
        else:
            sp_o = int(ours.get("spikes_over_09", 0))
            sp_i = int(orig.get("spikes_over_09", 0))
            if sp_o > sp_i:
                verdict, reason = "orig_better", "spikes"
            elif sp_o < sp_i:
                verdict, reason = "ours_better", "spikes"
            else:
                verdict, reason = "comparable", "tie"
    lu_o, lu_i = ours.get("lufs"), orig.get("lufs")
    if lu_o is None or lu_i is None:
        comparability = "unknown"
    elif abs(float(lu_o) - float(lu_i)) > loudness_mismatch_lu:
        comparability = "loudness_mismatch"
    else:
        comparability = "ok"
    return verdict, reason, comparability


def album_verdict(verdicts):
    """Majority vote over track verdicts; ties are comparable/inconclusive."""
    count = {name: sum(1 for v in verdicts if v == name)
             for name in ("ours_better", "orig_better", "comparable", "inconclusive")}
    if count["ours_better"] > count["orig_better"]:
        return "ours_better", "majority"
    if count["orig_better"] > count["ours_better"]:
        return "orig_better", "majority"
    if count["inconclusive"] > 0:
        return "inconclusive", "tie"
    return "comparable", "tie"


def _audio_files(root: Path, label: str) -> list[Path]:
    if not root.is_dir():
        raise EV.EvalError(f"{label} is not a readable directory: {root}")
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in audio_measure.AUDIO_EXT)
    if not files:
        raise EV.EvalError(f"{label} has no audio files: {root}")
    return files


def _load_runner_report(restored: Path) -> dict[str, dict]:
    """restore_safe's report.json rows keyed by output file name (may be empty)."""
    path = restored / "report.json"
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EV.EvalError(f"runner report is unreadable: {path} ({exc})")
    if not isinstance(rows, list):
        raise EV.EvalError(f"runner report must be a list of rows: {path}")
    keyed = {}
    for row in rows:
        out = row.get("output") or row.get("source")
        if not out:
            continue
        keyed.setdefault(Path(out).name, row)
    return keyed


def measure_pair(matched, runner_report) -> list[dict]:
    """Measure every matched track once per side; attach runner statuses."""
    rows = []
    for label, src_path, res_path, via in matched:
        src_m = EV.measure_file(src_path)
        res_m = EV.measure_file(res_path)
        if len(src_m["band_rms_db"]) != len(res_m["band_rms_db"]):
            raise EV.EvalError(f"{label}: band count differs between sides "
                               f"({len(src_m['band_rms_db'])} vs "
                               f"{len(res_m['band_rms_db'])})")
        bands_src = np.asarray(src_m["band_rms_db"])
        bands_res = np.asarray(res_m["band_rms_db"])
        row = {
            "track": label, "via": via,
            "src": src_m, "res": res_m,
            "band_shift_db": float(np.mean(bands_res - bands_src)),
            "band_delta_mean_db": float(np.mean(np.abs(bands_res - bands_src))),
            "punch_retention_db": float(res_m["crest_db"]) - float(src_m["crest_db"]),
            "status": "unknown", "needs_review": False, "restoration_state": "unknown",
        }
        runner = runner_report.get(res_path.name)
        if runner is not None:
            row["status"] = runner.get("status", "unknown")
            row["needs_review"] = bool(runner.get("needs_review", False))
            row["restoration_state"] = runner.get("restoration_state", "unknown")
        rows.append(row)
    return rows


def build_summary(rows: list[dict], matched_via: dict[str, int], skipped: list[dict],
                  src: Path, restored: Path, tp_ceiling: float,
                  dr_threshold: float) -> dict:
    verdicts = []
    for row in rows:
        v, reason, comparability = track_verdict(
            row["res"], row["src"], tp_ceiling=tp_ceiling, dr_threshold=dr_threshold)
        row["verdict"], row["reason"], row["comparability"] = v, reason, comparability
        verdicts.append(v)
    overall, overall_reason = album_verdict(verdicts)

    def med(key_side, key):
        values = [float(r[key_side][key]) for r in rows]
        return float(statistics.median(values))

    gaps = [float(r["res"]["lufs"]) - float(r["src"]["lufs"]) for r in rows]
    dr_changes = [float(r["res"]["dr14"]) - float(r["src"]["dr14"]) for r in rows]
    crest_changes = [float(r["res"]["crest_db"]) - float(r["src"]["crest_db"]) for r in rows]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    summary = {
        "src": str(src), "restored": str(restored),
        "tp_ceiling": tp_ceiling, "dr_threshold": dr_threshold,
        "tracks_measured": len(rows),
        "matched_via": matched_via,
        "skipped": skipped,
        "verdict_counts": {name: sum(1 for v in verdicts if v == name)
                           for name in ("ours_better", "orig_better",
                                        "comparable", "inconclusive")},
        "verdict_overall": overall,
        "verdict_reason": overall_reason,
        "comparability_counts": {
            name: sum(1 for r in rows if r["comparability"] == name)
            for name in ("ok", "loudness_mismatch", "unknown")},
        "status_counts": status_counts,
        "needs_review": sum(1 for r in rows if r["needs_review"]),
        "systematics": {
            "lufs_gap_median": float(statistics.median(gaps)),
            "dr_change_median": float(statistics.median(dr_changes)),
            "crest_change_median": float(statistics.median(crest_changes)),
            "band_shift_db_mean": float(np.mean([r["band_shift_db"] for r in rows])),
            "punch_retention_median": float(statistics.median(
                [r["punch_retention_db"] for r in rows])),
            "tp_at_ceiling": sum(1 for r in rows
                                 if r["res"]["true_peak_db"] > tp_ceiling - TP_AT_CEILING_TOL),
            "dr_regression_tracks": sum(1 for r in rows
                                        if float(r["res"]["dr14"]) < float(r["src"]["dr14"]) - DR_REGRESSION_DB),
            "clipped_orig_sum": int(sum(r["src"]["clip_samples"] for r in rows)),
            "clipped_ours_sum": int(sum(r["res"]["clip_samples"] for r in rows)),
            "spikes_orig_sum": int(sum(r["src"]["spikes_over_09"] for r in rows)),
            "spikes_ours_sum": int(sum(r["res"]["spikes_over_09"] for r in rows)),
        },
        "limitations": LIMITATIONS,
    }
    return summary


TRACK_COLUMNS = ("track", "dr14_orig", "dr14_ours", "crest_db_orig", "crest_db_ours",
                 "lufs_orig", "lufs_ours", "true_peak_db_orig", "true_peak_db_ours",
                 "clip_samples_orig", "clip_samples_ours",
                 "spikes_over_09_orig", "spikes_over_09_ours",
                 "band_shift_db", "band_delta_mean_db", "punch_retention_db")
VERDICT_COLUMNS = ("track", "verdict", "reason", "comparability",
                   "status", "needs_review", "restoration_state")


def fmt(value, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, columns: tuple, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in columns})


def render_report(summary: dict, rows: list[dict], src: Path, restored: Path) -> str:
    matched_via = summary["matched_via"]
    skipped = summary["skipped"]
    lines = [f"# Album evaluation: {src.name} → {restored.name}", "",
             f"Measured {summary['tracks_measured']} track(s) "
             f"(matched by number: {matched_via.get('number', 0)}, "
             f"by name: {matched_via.get('name', 0)})."]
    if skipped:
        lines.append("Skipped (reported, never dropped):")
        for s in skipped:
            lines.append(f"- {s['track']} ({s['side']}): {s['reason']}")
    lines += ["", "## Verdict", "",
              f"**{summary['verdict_overall']}** ({summary['verdict_reason']}) — "
              f"ours_better {summary['verdict_counts']['ours_better']}, "
              f"orig_better {summary['verdict_counts']['orig_better']}, "
              f"comparable {summary['verdict_counts']['comparable']}, "
              f"inconclusive {summary['verdict_counts']['inconclusive']}.", ""]
    sys_i = summary["systematics"]
    lines += ["| indicator | value |", "|---|---|",
              f"| DR change (median) | {fmt(sys_i['dr_change_median'])} |",
              f"| crest change (median) | {fmt(sys_i['crest_change_median'])} |",
              f"| loudness gap ours−orig (median) | {fmt(sys_i['lufs_gap_median'])} LU |",
              f"| band shift ours−orig (mean) | {fmt(sys_i['band_shift_db_mean'])} dB |",
              f"| tracks at/over TP ceiling | {sys_i['tp_at_ceiling']} |",
              f"| DR regression tracks | {sys_i['dr_regression_tracks']} |",
              f"| clipped samples orig → ours | {sys_i['clipped_orig_sum']} → {sys_i['clipped_ours_sum']} |",
              f"| sample jumps orig → ours | {sys_i['spikes_orig_sum']} → {sys_i['spikes_ours_sum']} |",
              f"| runner statuses | {summary['status_counts']} |",
              f"| needs_review | {summary['needs_review']} |", ""]
    lines += ["## Systematics", ""]
    if sys_i["lufs_gap_median"] < -1.0:
        lines.append(f"- The restored album sits {abs(sys_i['lufs_gap_median']):.1f} LU "
                     "below the source loudness (median) — that is the pipeline's "
                     "loudness design, not damage.")
    elif sys_i["lufs_gap_median"] > 1.0:
        lines.append(f"- The restored album is {sys_i['lufs_gap_median']:.1f} LU above "
                     "the source loudness (median) — check the true-peak headroom.")
    if sys_i["dr_change_median"] >= 0.3:
        lines.append(f"- Dynamics raised: DR +{sys_i['dr_change_median']:.1f}, "
                     f"crest +{sys_i['crest_change_median']:.1f} (median).")
    elif sys_i["dr_change_median"] <= -0.25:
        lines.append(f"- Dynamics regressed: DR {sys_i['dr_change_median']:+.1f} (median) "
                     "— inspect the orig_better/dr tracks below.")
    else:
        lines.append("- Dynamics essentially unchanged (median DR gap below 0.3).")
    if abs(sys_i["band_shift_db_mean"]) >= 0.3:
        direction = "darker" if sys_i["band_shift_db_mean"] < 0 else "brighter"
        lines.append(f"- Spectrum shifted {direction} by "
                     f"{abs(sys_i['band_shift_db_mean']):.1f} dB on average "
                     "(indicator, mixes level with tone).")
    if abs(sys_i["punch_retention_median"]) >= 1.0:
        sign = "+" if sys_i["punch_retention_median"] > 0 else ""
        lines.append(f"- Punch retention: crest {sign}"
                     f"{sys_i['punch_retention_median']:.1f} dB (scale-invariant — "
                     "the same at any matched loudness).")
    if sys_i["clipped_ours_sum"] > sys_i["clipped_orig_sum"]:
        lines.append("- The restoration ADDED clipping — treat as a defect.")
    if sys_i["spikes_ours_sum"] > sys_i["spikes_orig_sum"]:
        lines.append("- The restoration added click-like sample jumps — treat as a defect.")
    if summary["needs_review"]:
        lines.append(f"- {summary['needs_review']} track(s) flagged needs_review by "
                     "the runner: the selection did not resolve the enhancement "
                     "request; do not report them as successfully restored.")
    lines += ["", "## Per track", "",
              "DR/crest/LUFS/TP per side, verdict by the declared ladder "
              "(ceiling → DR → spikes → tie; LUFS never votes).", "",
              "| track | DR o→r | crest o→r | LUFS o→r | TP r | band shift | verdict | reason | status |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['track']} | {fmt(row['src']['dr14'])}→{fmt(row['res']['dr14'])} "
            f"| {fmt(row['src']['crest_db'])}→{fmt(row['res']['crest_db'])} "
            f"| {fmt(row['src']['lufs'], 1)}→{fmt(row['res']['lufs'], 1)} "
            f"| {fmt(row['res']['true_peak_db'])} | {fmt(row['band_shift_db'])} "
            f"| {row['verdict']} | {row['reason']} | {row['status']} |")
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="evaluate_album",
        description="Evaluate a restored album against its own source.")
    ap.add_argument("--src", required=True, help="original (source) album directory")
    ap.add_argument("--restored", required=True, help="restored album directory")
    ap.add_argument("--out", required=True, help="directory for tracks.csv/verdicts.csv/summary.json/REPORT.md")
    ap.add_argument("--tp-ceiling", type=float, default=PEAK_CEILING_DBFS,
                    help=f"true-peak ceiling for OUR output (default {PEAK_CEILING_DBFS})")
    ap.add_argument("--dr-threshold", type=float, default=DR_THRESHOLD,
                    help=f"DR gap that decides a track (default {DR_THRESHOLD})")
    ap.add_argument("--loudness-mismatch-lu", type=float, default=LOUDNESS_MISMATCH_LU,
                    help=f"|LUFS gap| marking loudness_mismatch (default {LOUDNESS_MISMATCH_LU})")
    args = ap.parse_args(argv)

    try:
        src = Path(args.src).expanduser().resolve()
        restored = Path(args.restored).expanduser().resolve()
        src_files = _audio_files(src, "--src")
        res_files = _audio_files(restored, "--restored")
        matched, skipped = match_tracks(src_files, res_files)
        if not matched:
            raise EV.EvalError("no track matched between the source and the "
                               "restored album; nothing to compare")
        matched_via = {"number": sum(1 for m in matched if m[3] == "number"),
                       "name": sum(1 for m in matched if m[3] == "name")}
        runner_report = _load_runner_report(restored)
        rows = measure_pair(matched, runner_report)
        summary = build_summary(rows, matched_via, skipped, src, restored,
                                args.tp_ceiling, args.dr_threshold)
        track_rows = [{
            "track": r["track"],
            "dr14_orig": str(int(r["src"]["dr14"])), "dr14_ours": str(int(r["res"]["dr14"])),
            "crest_db_orig": fmt(r["src"]["crest_db"]), "crest_db_ours": fmt(r["res"]["crest_db"]),
            "lufs_orig": fmt(r["src"]["lufs"], 1), "lufs_ours": fmt(r["res"]["lufs"], 1),
            "true_peak_db_orig": fmt(r["src"]["true_peak_db"]),
            "true_peak_db_ours": fmt(r["res"]["true_peak_db"]),
            "clip_samples_orig": str(int(r["src"]["clip_samples"])),
            "clip_samples_ours": str(int(r["res"]["clip_samples"])),
            "spikes_over_09_orig": str(int(r["src"]["spikes_over_09"])),
            "spikes_over_09_ours": str(int(r["res"]["spikes_over_09"])),
            "band_shift_db": fmt(r["band_shift_db"]),
            "band_delta_mean_db": fmt(r["band_delta_mean_db"]),
            "punch_retention_db": fmt(r["punch_retention_db"]),
        } for r in rows]
        verdict_rows = [{
            "track": r["track"], "verdict": r["verdict"], "reason": r["reason"],
            "comparability": r["comparability"], "status": r["status"],
            "needs_review": str(bool(r["needs_review"])),
            "restoration_state": r["restoration_state"],
        } for r in rows]
        # Every input error has been raised by now: only now touch the disk.
        out_dir = Path(args.out).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "tracks.csv", TRACK_COLUMNS, track_rows)
        write_csv(out_dir / "verdicts.csv", VERDICT_COLUMNS, verdict_rows)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / "REPORT.md").write_text(
            render_report(summary, rows, src, restored), encoding="utf-8")
        print(f"evaluate_album: {summary['tracks_measured']} track(s), "
              f"{len(skipped)} skipped → {out_dir} "
              f"(verdict: {summary['verdict_overall']}/{summary['verdict_reason']})",
              flush=True)
        return 0
    except EV.EvalError as exc:
        print(f"evaluate_album: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
