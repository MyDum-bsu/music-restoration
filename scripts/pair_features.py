#!/usr/bin/env python3
"""Features for a pair "original (image+cue) → restoration (per-track FLAC)" — into corpus/feature_pairs/.

For each pair of tracks: TT DR, crest, clip stats, 1/3-octave band RMS,
rest-minus-orig deltas. File names: NN. *.flac in the restored dir.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext.cue import parse_cue, track_bounds  # noqa: E402
from rnext import features as F  # noqa: E402
from rnext import workspace  # noqa: E402


def restored_tracks(d: Path) -> list[Path]:
    files = sorted(d.glob("*.flac"))
    files = [f for f in files if re.match(r"\d+\s*\.", f.name)]
    return files


def list_flac(d: Path) -> list[Path]:
    return sorted(d.glob("**/*.flac"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="pair key, directory id (used for saving)")
    ap.add_argument("--restored", required=True)
    ap.add_argument("--orig-image", required=True, help="FLAC image of the original")
    ap.add_argument("--cue", required=True)
    ap.add_argument("--album", default="", help="album label")
    args = ap.parse_args()

    img = Path(args.orig_image)
    cue = Path(args.cue)
    rdir = Path(args.restored)

    meta = F.ffprobe_audio(img)
    total = int(round(meta["dur"] * meta["sr"]))
    file_key = img.name
    tracks = parse_cue(str(cue), file_key=file_key)
    bounds = track_bounds(tracks, total, meta["sr"])

    rfiles = restored_tracks(rdir)
    if len(rfiles) != len(bounds):
        print(f"WARN: restored tracks={len(rfiles)} vs cue={len(bounds)}", sys.stderr)

    # decode the whole original into int16 (~0.7 GiB RAM per CD)
    x16, sr = F.decode_s16(str(img))
    if sr != meta["sr"] or x16.shape[0] < total - 3:
        print(f"WARN: decode len {x16.shape[0]} vs probe {total}", sys.stderr)

    out_dir = workspace.dir_of("corpus") / "feature_pairs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.key}.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for i, rpath in enumerate(rfiles):
            st, en = bounds[i]
            sl = x16[st:en]
            if sl.shape[0] < 8820:
                continue
            rm, rm_ = F.decode_f32(str(rpath))
            om, _ = F.decode_f32(str(img), start_s=st / meta["sr"], dur_s=sl.shape[0] / meta["sr"])
            row = {
                "key": args.key,
                "album": args.album,
                "track_n": i + 1,
                "cue_title": tracks[i]["title"],
                "restored_file": rpath.name,
                "sr": meta["sr"],
                "samples": {"restored": rm.shape[0], "original": sl.shape[0],
                            "delta": abs(rm.shape[0] - sl.shape[0])},
            }
            row["restored"] = {
                "dr": F.tt_dr(rm, rm_)[0], "dr_exact": round(F.tt_dr(rm, rm_)[1], 2),
                "peak_dbfb": round(F.tt_dr(rm, rm_)[3], 2), "rms_db": round(F.tt_dr(rm, rm_)[4], 2),
                "crest_db": round(F.crest_db(rm), 2),
                "clip": F.clip_stats(F.decode_s16(str(rpath))[0]),
                "bands_db": F.band_rms_db(rm, rm_),
            }
            row["original"] = {
                "dr": F.tt_dr(om, meta["sr"])[0], "dr_exact": round(F.tt_dr(om, meta["sr"])[1], 2),
                "peak_dbfb": round(F.tt_dr(om, meta["sr"])[3], 2), "rms_db": round(F.tt_dr(om, meta["sr"])[4], 2),
                "crest_db": round(F.crest_db(om), 2),
                "clip": F.clip_stats(sl),
                "bands_db": F.band_rms_db(om, meta["sr"]),
            }
            row["delta"] = {
                "dr": row["restored"]["dr_exact"] - row["original"]["dr_exact"],
                "crest": row["restored"]["crest_db"] - row["original"]["crest_db"],
                "bands_db": [round(a - b, 3) for a, b in zip(row["restored"]["bands_db"], row["original"]["bands_db"])],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  tr{i+1:>2}: DR {row['original']['dr']:>2} -> {row['restored']['dr']:>2} "
                  f"(Δ{row['delta']['dr']:+5.1f} dB) crest {row['original']['crest_db']:>5.1f} -> "
                  f"{row['restored']['crest_db']:>5.1f} (Δ{row['delta']['crest']:+5.2f}) "
                  f"clip {row['original']['clip']['L']['clipped_samples']:>6}/{row['restored']['clip']['L']['clipped_samples']:>6}")
            del om, rm
    print(f"-> {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
