#!/usr/bin/env python3
"""ABX-отрывки: оригинал | наша реставрация | мастер — тройки слепых файлов.

Для каждого трека и каждого окна (10%, 50%, 85% длительности, по 20 с)
пишутся два файла с СЛУЧАЙНОЙ маркировкой A/B (A=ours или master — монетка),
соответствие прячется в abx_map.json. Плюс несжатая копия оригинала
«REF_*» (не для угадывания, для прайминга слуха).

Использование:
  python3 scripts/abx_excerpts.py --ours <dir> --master <dir> \
      --orig-image <img.flac> --cue <cue.cue> --tracks 2,5,11 [--marks 2]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/mydum/Desktop/projects/restore-next")

from rnext import features as F  # noqa: E402
from restore_one import encode_flac  # noqa: E402  (реиспользуем из scripts/)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--orig-image", required=True)
    ap.add_argument("--cue", required=True)
    ap.add_argument("--tracks", default="2,5")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from rnext.cue import parse_cue, track_bounds

    ours_dir, master_dir = Path(args.ours), Path(args.master)
    out_dir = Path(args.out) if args.out else ours_dir / "abx"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Path(args.orig_image)
    meta = F.ffprobe_audio(str(img))
    sr = meta["sr"]
    total = int(round(meta["dur"] * sr))
    cue = parse_cue(args.cue, file_key=img.name)
    bounds = track_bounds(cue, total, sr)

    trks = [int(t) for t in args.tracks.split(",")]
    rand = random.Random(0xBEAD)
    amap = {}
    for tn in trks:
        o_f = next(ours_dir.glob(f"{tn:02d}*.flac"), None)
        m_f = next(master_dir.glob(f"{tn:02d}*.flac"), None)
        if not o_f or not m_f:
            print(f"t{tn}: файл не найден ({o_f}, {m_f})")
            continue
        st, en = bounds[tn - 1]
        dur = (en - st) / sr
        marks = {}
        for seg_i, frac in enumerate((0.10, 0.50, 0.85)):
            t0 = st / sr + max(0.0, min(0.9, frac)) * dur
            x, sr_ = F.decode_f32(str(img), start_s=t0, dur_s=20.0)
            if x is None or x.shape[0] < sr * 10:
                continue
            ref16 = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)
            encode_flac(ref16, sr_, out_dir / f"REF_t{tn:02d}_s{seg_i}.flac", out_dir / "__tmp.wav")
            # A/B: монетка решает, что сегодня «A»
            ours_is_a = rand.random() < 0.5
            for tag, src in (("A", o_f if ours_is_a else m_f),
                             ("B", m_f if ours_is_a else o_f)):
                y, sy = F.decode_f32(str(src), start_s=t0, dur_s=20.0)
                if y is None:
                    continue
                y16 = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
                name = out_dir / f"t{tn:02d}_s{seg_i}_{tag}.flac"
                encode_flac(y16, sy, name, out_dir / "__tmp.wav")
                marks[f"s{seg_i}:{tag}"] = {"file": name.name, "is_ours": tag == "A" and ours_is_a or (tag == "A" and not ours_is_a)}
                marks[f"s{seg_i}:{tag}"] = {"file": name.name, "is_ours": (src == o_f)}
        amap[f"t{tn:02d}"] = {"title": cue[tn - 1]["title"], "segments": marks}

    with open(out_dir / "abx_map.json", "w", encoding="utf-8") as f:
        json.dump(amap, f, ensure_ascii=False, indent=1)
    print(f"\n-> {out_dir} (карта: abx_map.json — не подглядывать до вердикта)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
