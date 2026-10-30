#!/usr/bin/env python3
"""ABX set "original vs ours" for blind listening.

- 8-second excerpts taken from the same spot of both sources;
- LOUDNESS MATCHING: every excerpt is normalized to a common RMS
  (−20 dBFS by default) so the level difference (ours is −5…−6 dB below the
  original) does not give the answer away;
- random A/B order + cipher map in abx_map.json.

  python3 scripts/abx_blind.py --ours <dir> --orig <image.flac|cue> --cue <cue> \
      --tracks 1,4,9 --out /media/.../abx/blind-ross
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rnext import features as F  # noqa: E402
from restore_one import encode_flac  # noqa: E402

TARGET_RMS_DB = -20.0


def loudest_window(x: np.ndarray, sr: int, sec: float) -> float:
    """Start of the loudest window of length sec (for the showcase excerpt)."""
    n = x.shape[0]
    win = int(sec * sr)
    if n <= win:
        return 0.0
    step = sr  # 1 s
    starts = list(range(0, n - win, step))
    r = [float(np.mean(x[s:s + win] ** 2)) for s in starts]
    return starts[int(np.argmax(r))] / sr


def norm_rms(y: np.ndarray, target_db: float = TARGET_RMS_DB) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms <= 1e-9:
        return y
    g = 10 ** ((target_db - 20 * np.log10(rms)) / 20.0)
    return y * g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--orig", required=True, help="image FLAC or a directory of tracks")
    ap.add_argument("--cue", default="")
    ap.add_argument("--tracks", default="", help="comma-separated track numbers; empty = all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--spots", default="loud,mid", help="comma-separated loud|mid")
    args = ap.parse_args()

    ours_dir = Path(args.ours)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.orig)

    if src.is_file() and args.cue:
        from rnext.cue import parse_cue, track_bounds
        meta = F.ffprobe_audio(str(src))
        total = int(round(meta["dur"] * meta["sr"]))
        cs = parse_cue(args.cue, file_key=src.name)
        bounds = track_bounds(cs, total, meta["sr"])
        def grab(ti, t0, dur):
            return F.decode_f32(str(src), start_s=t0, dur_s=dur)
        ntr = len(bounds)
        titles = {i + 1: cs[i]["title"] for i in range(len(cs))}
        bounds_of = {i + 1: bounds[i] for i in range(len(bounds))}
    else:
        import os
        fs = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".flac"))
        titles = {i + 1: Path(f).stem for i, f in enumerate(fs)}
        def grab(ti, t0, dur):
            return F.decode_f32(fs[ti - 1], start_s=t0, dur_s=dur)
        ntr = len(fs)
        bounds_of = None

    trks = [int(t) for t in args.tracks.split(",")] if args.tracks else list(range(1, ntr + 1))
    rand = random.Random(0xB11D)
    amap = {}
    for ti in trks:
        o_f = next(ours_dir.glob(f"{ti:02d}*.flac"), None)
        if not o_f:
            print(f"t{ti}: our file not found")
            continue
        # reference point — from the original
        if bounds_of is not None:
            st, en = bounds_of[ti]
            xwhole, sr = grab(ti, st / 44100, (en - st) / 44100)
        else:
            xwhole, sr = F.decode_f32(str(o_f))
        spots = {}
        for spot in args.spots.split(","):
            if spot == "loud":
                t0 = loudest_window(xwhole, sr, args.seconds)
            else:  # mid
                t0 = max(0.0, (xwhole.shape[0] / sr) * 0.5 - args.seconds / 2)
            x, sx = (grab(ti, t0, args.seconds) if bounds_of is not None
                     else F.decode_f32(str(o_f), start_s=t0, dur_s=args.seconds))
            y, sy = F.decode_f32(str(o_f), start_s=t0, dur_s=args.seconds)
            if x is None or y is None:
                continue
            x = norm_rms(x); y = norm_rms(y)
            # reference original (can be auditioned beforehand, not for guessing)
            x16 = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)
            encode_flac(x16, sx, out_dir / f"REF_t{ti:02d}_{spot}.flac", out_dir / "__tmp.wav")
            ours_is_a = rand.random() < 0.5
            seg = {}
            for tag, arr in (("A", x if ours_is_a else y), ("B", y if ours_is_a else x)):
                a16 = np.clip(np.round(arr * 32768.0), -32768, 32767).astype(np.int16)
                name = f"t{ti:02d}_{spot}_{tag}.flac"
                encode_flac(a16, sx, out_dir / name, out_dir / "__tmp.wav")
                seg[tag] = {"file": name, "is_ours": (tag == "A" and ours_is_a) or (tag == "B" and not ours_is_a)}
            amap[f"t{ti:02d}_{spot}"] = {"title": titles.get(ti, "?"), "t0_s": round(t0, 2), "ab": seg}
        print(f"t{ti:>2} {titles.get(ti,'?')[:40]}: excerpts {len(spots)}×{args.seconds:.0f}s", flush=True)
    with open(out_dir / "abx_map.json", "w", encoding="utf-8") as f:
        json.dump(amap, f, ensure_ascii=False, indent=1)
    print(f"-> {out_dir} (map: abx_map.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
