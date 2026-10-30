#!/usr/bin/env python3
"""DR report in the DR-meter format that masters publish on the tracker.

Generates the classic TT DR text table for a restored album:

    Analyzed: Artist / Album
    ------------------------------------------------------------------
    DR       Peak        RMS    Filename
    ------------------------------------------------------------------
    DR11     -1.39 dB   -13.72 dB      2:08 01. Carpathia
    ...
    ------------------------------------------------------------------
    Number of files:    12
    Official DR value:  DR11

The format matches what `rnext.rttopic.parse_dr_table` parses from the master
topics, so our report can be posted in a release topic as is.
"Official DR value" is computed like in the meter: the blocks of all tracks
are pooled into a single analysis.
"""
import argparse
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rnext import features as F  # noqa: E402
from rnext.features import (  # noqa: E402
    AUDIO_MIN, BLOCK_TIME, CUT_BEST_BINS, MAX_DYNAMIC, dr_rms,
)

LINE = "-" * 66


def track_no(name: str) -> int:
    m = re.match(r"\s*(\d{1,3})", name)
    return int(m.group(1)) if m else 0


def clean(stem: str) -> str:
    s = re.sub(r"^\s*\d{1,3}\s*(?:[-._)]|\s)\s*", "", stem.strip(), count=1)
    return s.strip() or stem


def dur_str(sec: float) -> str:
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def analyzed_line(directory: pathlib.Path, files: list[pathlib.Path]) -> str:
    """Artist / Album from the first track's tags, else the directory name."""
    try:
        from mutagen.flac import FLAC
        f = FLAC(str(files[0]))
        tags = f.tags or {}
        artist = tags.get("albumartist") or tags.get("artist") or ""
        album = tags.get("album") or ""
        if artist or album:
            return f"{artist[0] if artist else ''} / {album[0] if album else ''}".strip(" /")
    except Exception:  # noqa: BLE001
        pass
    return directory.name


def official_dr(rms: np.ndarray, peaks: np.ndarray) -> int:
    """Blocks of all tracks sorted together: the TT DR formula unchanged."""
    rms = np.sort(rms, axis=0)
    peaks = np.sort(peaks, axis=0)
    n_blk = max(int(np.floor(rms.shape[0] * CUT_BEST_BINS)), 1)
    top = rms[rms.shape[0] - n_blk:, :]
    rms_energy_mean = np.sqrt(np.sum(top ** 2, 0) / n_blk)
    ch_dr = -20.0 * np.log10(rms_energy_mean / peaks[peaks.shape[0] - 2, :])
    ch_dr[np.logical_or(np.sum(top ** 2, 0) < AUDIO_MIN, np.abs(ch_dr) > MAX_DYNAMIC)] = 0.0
    return int(round(float(np.mean(ch_dr))))


def analyze(directory: pathlib.Path) -> str:
    files = sorted(p for p in directory.glob("*.flac") if not p.name.startswith("_"))
    if not files:
        raise SystemExit(f"no FLAC files in {directory}")
    rows, rms_all, peaks_all = [], [], []
    for f in files:
        x, sr = F.decode_f32(str(f))
        dr, _drx, _ch, peak_db, rms_db = F.tt_dr(x, sr)
        # blocks of the same size as in tt_dr — for the album-wide Official DR
        delta_fs = 60 if sr == 44100 else 0
        block = BLOCK_TIME * (sr + delta_fs)
        seg_cnt = max(1, int(np.ceil(x.shape[0] / block)))
        rms = np.zeros((seg_cnt, x.shape[1]))
        peaks = np.zeros_like(rms)
        for i in range(seg_cnt):
            seg = x[i * block:(i + 1) * block, :]
            rms[i] = dr_rms(seg)
            peaks[i] = np.max(np.abs(seg), 0)
        rms_all.append(rms)
        peaks_all.append(peaks)
        rows.append((dr, peak_db, rms_db, dur_str(F.ffprobe_audio(str(f))["dur"]),
                     f"{track_no(f.stem):02d}. {clean(f.stem)}"))
    official = official_dr(np.vstack(rms_all), np.vstack(peaks_all))
    text = [f" Analyzed: {analyzed_line(directory, files)}", LINE,
            " DR        Peak        RMS    Filename", LINE, ""]
    for dr, peak, rms, dur, name in rows:
        text.append(f" DR{dr:<2}  {peak:8.2f} dB {rms:8.2f} dB {dur:>7} {name}")
    text += [LINE, "",
             f" Number of files:    {len(rows)}",
             f" Official DR value:  DR{official}", "",
             "=" * 66]
    return "\n".join(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", help="directory with FLAC files (restoration)")
    ap.add_argument("--orig", default="", help="original directory: printed as the first block")
    ap.add_argument("--out", default="", help="report file (default <dir>/DR Report.txt)")
    args = ap.parse_args()

    blocks = []
    if args.orig:
        blocks.append(analyze(pathlib.Path(args.orig)))
    blocks.append(analyze(pathlib.Path(args.directory)))
    report = "\n\n\n".join(blocks) + "\n"
    print(report)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.directory) / "DR Report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"[dr_report] written: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
