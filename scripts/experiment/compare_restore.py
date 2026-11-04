#!/usr/bin/env python3
"""Neutral ORIG / OURS / INDEP comparison for the independent-restore experiment.

  python3 scripts/experiment/compare_restore.py --src <root> --ours <dir> \
          --indep <dir> --out-dir <dir>

Albums and tracks come from --src (top-level directory per album, "<artist> -
<album>"); --ours and --indep mirror that album structure.  Every metric is
measured through scripts/audio_measure.py::analyse(); the two extra columns are
spikes_gt_0_9 (samples with |x| > 0.9 in the float decode) and
band_delta_max_db (largest of the eight 1/3-octave band deviations from ORIG,
rnext.features.band_rms_db).

Counterparts are resolved by album and by track key (the path inside the album
directory without the container suffix).  The key must match exactly and must
resolve to exactly one file; a missing counterpart is a hard error and no CSV
is written at all.  Source trees whose names differ from the canonical ones are
normalised beforehand by `scripts/experiment/stage_source.py`, so the
comparator itself never guesses which track is meant.

The decision procedure is declared before the data in
docs/experiment/independent-method.md and docs/experiment-independent-restore-2026-11-04.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
import audio_measure  # noqa: E402  (single source of truth for the metrics)

REPO_ROOT = SCRIPTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from rnext import features as F  # noqa: E402

AUDIO_EXT = set(audio_measure.AUDIO_EXT)
VARIANTS = ("indep", "orig", "ours")
TRACK_COLUMNS = (
    "album", "track", "variant", "rel_path", "dr", "dr_exact", "crest_tt",
    "lufs_i", "tp_dbfs", "peak_dbfs", "rms_db", "clipped_samples",
    "clipped_pct", "spikes_gt_0_9", "band_delta_max_db", "cutoff_hz",
    "hf_floor_db", "sr", "bits", "dur_s",
)
ALBUM_COLUMNS = (
    "album", "artist", "variant", "n_tracks", "dr_median", "crest_median",
    "lufs_median", "tp_max", "clipped_pct_sum", "spikes_gt_0_9_sum",
    "band_delta_max_db",
)
VERDICT_COLUMNS = ("album", "track", "verdict", "reason", "comparability")
ALBUM_VERDICT_COLUMNS = ("album", "artist", "verdict", "reason", "comparability")

SEPARATOR = " - "
# release folders carry a version marker after the album name
VERSION_TAGS = (" TR24", " JP", " FLAC", " 24bit")
CANON_SUFFIX = " [restored per-track]"

DR_THRESHOLD = 0.3
PEAK_DISQUALIFY_ABOVE = -0.9
LOUDNESS_MISMATCH_LU = 2.0


# ── discovery and counterpart resolution ─────────────────────────────────────

def audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)


def track_key(path: Path, root: Path) -> str:
    """Path inside the album directory without the container suffix."""
    return str(path.relative_to(root).with_suffix(""))


def discover_albums(src: Path) -> list[dict]:
    albums = []
    for path in sorted(src.iterdir()):
        if not path.is_dir() or SEPARATOR not in path.name:
            continue
        artist, _rest = path.name.split(SEPARATOR, 1)
        files = audio_files(path)
        if not files:
            continue
        albums.append({"artist": artist.strip(), "album": path.name,
                       "dir": path, "files": files})
    return albums


def album_base(album: str) -> str:
    name = album
    changed = True
    while changed:
        changed = False
        for tag in VERSION_TAGS:
            if name.endswith(tag):
                name = name[: -len(tag)].rstrip()
                changed = True
    return name


def album_candidates(album: str) -> list[str]:
    base = album_base(album)
    names = [album, base, f"{album}{CANON_SUFFIX}", f"{base}{CANON_SUFFIX}"]
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def resolve_album_dir(root: Path, album: str, label: str) -> Path:
    existing = [root / name for name in album_candidates(album) if (root / name).is_dir()]
    if not existing:
        raise SystemExit(
            f"compare_restore: {label}: no directory for album {album!r} under {root} "
            f"(tried {', '.join(album_candidates(album))})")
    if len(existing) > 1:
        raise SystemExit(
            f"compare_restore: {label}: ambiguous directories for album {album!r}: "
            f"{', '.join(str(p) for p in existing)}")
    return existing[0]


def index_tracks(root: Path) -> dict[str, Path]:
    by_key: dict[str, Path] = {}
    for path in audio_files(root):
        key = track_key(path, root)
        if key in by_key:
            raise SystemExit(
                f"compare_restore: duplicate track key {key!r} under {root}: "
                f"{by_key[key]} and {path}")
        by_key[key] = path
    return by_key


def match_variant(album: dict, variant_root: Path, label: str) -> dict[str, Path]:
    """Map every source track key of one album onto a file of the variant."""
    album_dir = resolve_album_dir(variant_root, album["album"], label)
    by_key = index_tracks(album_dir)
    matched: dict[str, Path] = {}
    problems: list[str] = []
    for path in album["files"]:
        key = track_key(path, album["dir"])
        if key in by_key:
            matched[key] = by_key[key]
        else:
            problems.append(f"missing {key!r}")
    if problems:
        raise SystemExit(
            f"compare_restore: {label}: album {album['album']!r} under {variant_root} "
            f"({len(problems)} of {len(album['files'])} tracks without a counterpart):\n  "
            + "\n  ".join(problems))
    return matched


# ── measurement ──────────────────────────────────────────────────────────────

def bits_declared(path: Path, value) -> int:
    """Bit depth of the file; audio_measure leaves it empty for raw PCM/WAV."""
    if value:
        return int(value)
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=bits_per_raw_sample,bits_per_sample", "-of", "json", str(path)],
        capture_output=True, text=True)
    streams = json.loads(r.stdout or "{}").get("streams") or []
    if r.returncode != 0 or not streams:
        raise SystemExit(f"compare_restore: {path}: cannot probe the bit depth")
    for key in ("bits_per_raw_sample", "bits_per_sample"):
        raw = streams[0].get(key)
        if raw:
            return int(raw)
    raise SystemExit(f"compare_restore: {path}: the container declares no bit depth")


def measure(path: Path, orig_bands: list[float] | None) -> tuple[dict, list[float]]:
    """Metrics for one file plus its 1/3-octave bands (for the ORIG delta)."""
    row = audio_measure.analyse(path, quick=False)
    if row.get("error"):
        raise SystemExit(f"compare_restore: cannot measure {path}: {row['error']}")
    for required in ("dr", "dr_exact", "crest_tt", "peak_dbfs", "rms_db",
                     "clipped_samples", "clipped_pct", "cutoff_hz", "hf_floor_db",
                     "sr", "dur_s"):
        if row.get(required) is None:
            raise SystemExit(f"compare_restore: {path}: metric {required} is missing")
    x, sr = audio_measure.decode(path)
    if x is None or x.size == 0:
        raise SystemExit(f"compare_restore: cannot decode {path}")
    bands = F.band_rms_db(x, sr)
    if not bands:
        raise SystemExit(f"compare_restore: band_rms_db returned nothing for {path}")
    if orig_bands is not None and len(bands) != len(orig_bands):
        raise SystemExit(
            f"compare_restore: band count differs for {path}: "
            f"{len(bands)} vs {len(orig_bands)}")
    delta = 0.0
    if orig_bands is not None:
        delta = max(abs(b - o) for b, o in zip(bands, orig_bands))
    metrics = {
        "dr": row["dr"],
        "dr_exact": row["dr_exact"],
        "crest_tt": row["crest_tt"],
        "lufs_i": row.get("lufs_i"),
        "tp_dbfs": row.get("tp_dbfs"),
        "peak_dbfs": row["peak_dbfs"],
        "rms_db": row["rms_db"],
        "clipped_samples": row["clipped_samples"],
        "clipped_pct": row["clipped_pct"],
        "spikes_gt_0_9": int(np.sum(np.abs(x) > 0.9)),
        "band_delta_max_db": delta,
        "cutoff_hz": row["cutoff_hz"],
        "hf_floor_db": row["hf_floor_db"],
        "sr": row["sr"],
        "bits": bits_declared(path, row.get("bits")),
        "dur_s": row["dur_s"],
    }
    return metrics, bands


def fmt_float(value, digits: int) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def csv_row(album: str, key: str, variant: str, rel: str, m: dict) -> dict:
    return {
        "album": album,
        "track": key,
        "variant": variant,
        "rel_path": rel,
        "dr": str(int(m["dr"])),
        "dr_exact": fmt_float(m["dr_exact"], 2),
        "crest_tt": fmt_float(m["crest_tt"], 2),
        "lufs_i": fmt_float(m["lufs_i"], 2),
        "tp_dbfs": fmt_float(m["tp_dbfs"], 2),
        "peak_dbfs": fmt_float(m["peak_dbfs"], 2),
        "rms_db": fmt_float(m["rms_db"], 2),
        "clipped_samples": str(int(m["clipped_samples"])),
        "clipped_pct": fmt_float(m["clipped_pct"], 4),
        "spikes_gt_0_9": str(int(m["spikes_gt_0_9"])),
        "band_delta_max_db": fmt_float(m["band_delta_max_db"], 2),
        "cutoff_hz": str(int(m["cutoff_hz"])),
        "hf_floor_db": fmt_float(m["hf_floor_db"], 1),
        "sr": str(int(m["sr"])),
        "bits": str(int(m["bits"])),
        "dur_s": fmt_float(m["dur_s"], 1),
    }


# ── the declared verdict procedure ───────────────────────────────────────────

def num(value):
    text = "" if value is None else str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return float(text)


def track_verdict(ours: dict, indep: dict) -> tuple[str, str, str]:
    def disqualified(row: dict) -> bool:
        tp = num(row.get("tp_dbfs"))
        return tp is not None and tp > PEAK_DISQUALIFY_ABOVE

    o_dis, i_dis = disqualified(ours), disqualified(indep)
    if o_dis and i_dis:
        verdict, reason = "inconclusive", "ceiling"
    elif o_dis:
        verdict, reason = "indep_better", "ceiling"
    elif i_dis:
        verdict, reason = "ours_better", "ceiling"
    else:
        dr_o, dr_i = num(ours["dr"]), num(indep["dr"])
        if abs(dr_o - dr_i) >= DR_THRESHOLD:
            verdict = "ours_better" if dr_o > dr_i else "indep_better"
            reason = "dr"
        else:
            sp_o, sp_i = int(num(ours["spikes_gt_0_9"])), int(num(indep["spikes_gt_0_9"]))
            if sp_o != sp_i:
                verdict = "ours_better" if sp_o < sp_i else "indep_better"
                reason = "spikes"
            else:
                verdict, reason = "comparable", "tie"
    lu_o, lu_i = num(ours.get("lufs_i")), num(indep.get("lufs_i"))
    if lu_o is None or lu_i is None:
        comparability = "unknown"
    elif abs(lu_o - lu_i) > LOUDNESS_MISMATCH_LU:
        comparability = "loudness_mismatch"
    else:
        comparability = "ok"
    return verdict, reason, comparability


def album_verdict(tracks: list[str], lufs_ours, lufs_indep) -> tuple[str, str, str]:
    count = {name: sum(1 for v in tracks if v == name)
             for name in ("ours_better", "indep_better", "comparable", "inconclusive")}
    if count["ours_better"] > count["indep_better"]:
        verdict, reason = "ours_better", "majority"
    elif count["indep_better"] > count["ours_better"]:
        verdict, reason = "indep_better", "majority"
    elif count["inconclusive"] > 0:
        verdict, reason = "inconclusive", "tie"
    else:
        verdict, reason = "comparable", "tie"
    a, b = num(lufs_ours), num(lufs_indep)
    if a is None or b is None:
        comparability = "unknown"
    elif abs(a - b) > LOUDNESS_MISMATCH_LU:
        comparability = "loudness_mismatch"
    else:
        comparability = "ok"
    return verdict, reason, comparability


# ── aggregation and output ───────────────────────────────────────────────────

def aggregate(album: str, artist: str, variant: str, rows: list[dict]) -> dict:
    def values(column: str) -> list[float]:
        return [v for v in (num(r[column]) for r in rows) if v is not None]

    dr = values("dr")
    crest = values("crest_tt")
    lufs = values("lufs_i")
    tp = values("tp_dbfs")
    clipped = values("clipped_pct")
    spikes = values("spikes_gt_0_9")
    delta = values("band_delta_max_db")
    return {
        "album": album,
        "artist": artist,
        "variant": variant,
        "n_tracks": str(len(rows)),
        "dr_median": fmt_float(statistics.median(dr), 2) if dr else "",
        "crest_median": fmt_float(statistics.median(crest), 2) if crest else "",
        "lufs_median": fmt_float(statistics.median(lufs), 2) if lufs else "",
        "tp_max": fmt_float(max(tp), 2) if tp else "",
        "clipped_pct_sum": fmt_float(sum(clipped), 4) if clipped else "",
        "spikes_gt_0_9_sum": str(int(sum(spikes))) if spikes else "",
        "band_delta_max_db": fmt_float(max(delta), 2) if delta else "",
    }


def write_csv(path: Path, columns: tuple, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def table(columns: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |",
           "|" + "|".join("---" for _ in columns) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


def write_tables(path: Path, tracks: list[dict], albums: list[dict],
                 verdicts: list[dict], album_verdicts: list[dict]) -> None:
    lines = ["# Experiment tables", "",
             "Generated by `scripts/experiment/compare_restore.py`; the prose "
             "analysis lives in `docs/experiment-independent-restore-2026-11-04.md`.",
             "", "## Per-track", "",
             "DR and crest from `scripts/audio_measure.py::analyse()`, LUFS/TP from "
             "ffmpeg ebur128, spikes = samples with |x| > 0.9, band_delta = largest "
             "1/3-octave deviation from ORIG.", ""]
    lines += table(
        ["album", "track", "variant", "DR", "crest", "LUFS", "TP", "spikes",
         "clipped", "band_delta"],
        [[r["album"], r["track"], r["variant"], r["dr"], r["crest_tt"],
          r["lufs_i"] or "-", r["tp_dbfs"] or "-", r["spikes_gt_0_9"],
          r["clipped_pct"], r["band_delta_max_db"]] for r in tracks])
    lines += ["", "## Per-album", ""]
    lines += table(
        ["album", "variant", "n", "DR med", "crest med", "LUFS med", "TP max",
         "clipped % sum", "spikes sum", "band_delta max"],
        [[r["album"], r["variant"], r["n_tracks"], r["dr_median"],
          r["crest_median"], r["lufs_median"] or "-", r["tp_max"] or "-",
          r["clipped_pct_sum"], r["spikes_gt_0_9_sum"],
          r["band_delta_max_db"]] for r in albums])
    lines += ["", "## Verdicts", "",
              "Rule: ceiling disqualification above TP -0.9 dBFS, then a DR gap of "
              "0.3 or more, then the spike count, then a tie.", ""]
    lines += table(["album", "track", "verdict", "reason", "comparability"],
                   [[r["album"], r["track"], r["verdict"], r["reason"],
                     r["comparability"]] for r in verdicts])
    lines += ["", "| album | artist | verdict | reason | comparability |",
              "|---|---|---|---|---|"]
    lines += ["| " + " | ".join([r["album"], r["artist"], r["verdict"], r["reason"],
                                 r["comparability"]]) + " |" for r in album_verdicts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Compare ORIG, OURS and INDEP.")
    ap.add_argument("--src", required=True, help="root of the source releases")
    ap.add_argument("--ours", required=True, help="root of the canonical results")
    ap.add_argument("--indep", required=True, help="root of the independent results")
    ap.add_argument("--out-dir", required=True, help="directory for the CSV/JSON/markdown data")
    ap.add_argument("--append", action="store_true",
                    help="add these albums to an existing data set (same measurement, "
                         "used when the run is split over several invocations)")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    ours_root = Path(args.ours).expanduser().resolve()
    indep_root = Path(args.indep).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    for label, root in (("--src", src), ("--ours", ours_root), ("--indep", indep_root)):
        if not root.is_dir():
            print(f"compare_restore: {label} is not a readable directory: {root}",
                  file=sys.stderr)
            return 2

    albums = discover_albums(src)
    if not albums:
        print(f"compare_restore: no albums with audio files under {src}", file=sys.stderr)
        return 2

    # Resolve every counterpart before measuring anything: no partial data.
    plan: list[dict] = []
    for album in albums:
        matched = {"orig": {track_key(p, album["dir"]): p for p in album["files"]}}
        matched["indep"] = match_variant(album, indep_root, "--indep")
        matched["ours"] = match_variant(album, ours_root, "--ours")
        plan.append({**album, "matches": matched})

    tracks: list[dict] = []
    for album in plan:
        name = album["album"]
        print(f"compare_restore: {name} ({len(album['files'])} tracks)", flush=True)
        for src_file in album["files"]:
            key = track_key(src_file, album["dir"])
            orig, orig_bands = measure(album["matches"]["orig"][key], None)
            for variant in VARIANTS:
                path = album["matches"][variant][key]
                metrics = orig if variant == "orig" else measure(path, orig_bands)[0]
                root = {"orig": src, "ours": ours_root, "indep": indep_root}[variant]
                tracks.append(csv_row(name, key, variant,
                                      str(path.relative_to(root)), metrics))

    artists = {album["album"]: album["artist"] for album in plan}
    if args.append and (out_dir / "tracks.csv").is_file():
        with (out_dir / "tracks.csv").open(encoding="utf-8", newline="") as fh:
            previous = list(csv.DictReader(fh))
        with (out_dir / "albums.csv").open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                artists.setdefault(row["album"], row["artist"])
        known = {(r["album"], r["track"], r["variant"]) for r in previous}
        for row in tracks:
            key = (row["album"], row["track"], row["variant"])
            if key in known:
                print(f"compare_restore: --append would duplicate {key}", file=sys.stderr)
                return 2
        tracks = previous + tracks
        print(f"compare_restore: --append, {len(previous)} existing rows kept", flush=True)

    tracks.sort(key=lambda r: (r["album"], r["track"], r["variant"]))
    duplicates = {(r["album"], r["track"], r["variant"]) for r in tracks}
    if len(duplicates) != len(tracks):
        print("compare_restore: duplicate (album, track, variant) rows", file=sys.stderr)
        return 2
    albums_rows: list[dict] = []
    by_album: dict[str, list[dict]] = {}
    for row in tracks:
        by_album.setdefault(row["album"], []).append(row)
    for name in sorted(by_album):
        rows = by_album[name]
        for variant in VARIANTS:
            variant_rows = [r for r in rows if r["variant"] == variant]
            if len(variant_rows) != len({r["track"] for r in rows}):
                print(f"compare_restore: {name}: incomplete {variant} variant",
                      file=sys.stderr)
                return 2
            albums_rows.append(aggregate(name, artists[name], variant, variant_rows))

    verdicts: list[dict] = []
    verdict_by_album: dict[str, list[str]] = {}
    for name in sorted(by_album):
        keyed = {(r["track"], r["variant"]): r for r in by_album[name]}
        for key in sorted({r["track"] for r in by_album[name]}):
            verdict, reason, comparability = track_verdict(
                keyed[(key, "ours")], keyed[(key, "indep")])
            verdicts.append({"album": name, "track": key, "verdict": verdict,
                             "reason": reason, "comparability": comparability})
            verdict_by_album.setdefault(name, []).append(verdict)

    album_metrics = {(r["album"], r["variant"]): r for r in albums_rows}
    album_verdicts: list[dict] = []
    for name in sorted(by_album):
        verdict, reason, comparability = album_verdict(
            verdict_by_album[name],
            album_metrics[(name, "ours")]["lufs_median"],
            album_metrics[(name, "indep")]["lufs_median"])
        album_verdicts.append({"album": name, "artist": artists[name],
                               "verdict": verdict, "reason": reason,
                               "comparability": comparability})

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "tracks.csv", TRACK_COLUMNS, tracks)
    write_csv(out_dir / "albums.csv", ALBUM_COLUMNS, albums_rows)
    write_csv(out_dir / "verdicts.csv", VERDICT_COLUMNS, verdicts)
    write_csv(out_dir / "album_verdicts.csv", ALBUM_VERDICT_COLUMNS, album_verdicts)
    write_tables(out_dir / "tables.md", tracks, albums_rows, verdicts, album_verdicts)

    summary = {
        "variants": list(VARIANTS),
        "albums": len(by_album),
        "tracks": len({(r["album"], r["track"]) for r in tracks}),
        "rows": len(tracks),
        "artists": sorted(set(artists.values())),
        "src": str(src),
        "ours": str(ours_root),
        "indep": str(indep_root),
        "procedure": {
            "peak_disqualify_above_dbfs": PEAK_DISQUALIFY_ABOVE,
            "dr_threshold": DR_THRESHOLD,
            "loudness_mismatch_lu": LOUDNESS_MISMATCH_LU,
        },
        "verdict_counts": {v: sum(1 for r in verdicts if r["verdict"] == v)
                           for v in ("ours_better", "indep_better", "comparable",
                                     "inconclusive")},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"compare_restore: {len(by_album)} albums, {summary['tracks']} tracks → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
