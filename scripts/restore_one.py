#!/usr/bin/env python3
"""Реставрация целого альбома оригинала → каталог с FLAC'ами + отчёт.

Без референса: политика (mode/r) обрабатывает все FLAC-треки исходного
каталога (или image+cue). Результат:
  out/NN. Title.flac — 16-бит FLAC (TP-нормализация к пику политики)
  out/restore_report.jsonl — метрики каждого трека (DR/crest/peak/rms до/после)
"""
import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "/home/mydum/Desktop/projects/restore-next")

from rnext import features as F  # noqa: E402
from rnext.dsp.contour import true_peak_dbfs  # noqa: E402
from rnext.dsp.pipeline import Policy, restore_track  # noqa: E402


def encode_flac(y16: np.ndarray, sr: int, flac_path: Path, wav_path: Path) -> None:
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(y16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y16.tobytes())
    subprocess.run(["ffmpeg", "-y", "-threads", "2", "-i", str(wav_path),
                    "-c:a", "flac", "-compression_level", "8", str(flac_path)],
                   check=True, capture_output=True)
    wav_path.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="каталог с FLAC-треками или один image-файл")
    ap.add_argument("--cue", default="", help="cue, если --src это image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="expand", choices=["expand", "flatten"])
    ap.add_argument("--r", type=float, default=1.5)
    ap.add_argument("--k", type=float, default=0.55)
    ap.add_argument("--tp", type=float, default=-0.5, help="target TP, dBFS")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pol = Policy(mode=args.mode, r_expand=args.r, k=args.k, target_tp_dbfs=args.tp)
    src = Path(args.src)

    if src.is_file() and args.cue:
        from rnext.cue import parse_cue, track_bounds
        meta = F.ffprobe_audio(str(src))
        total = int(round(meta["dur"] * meta["sr"]))
        cs = parse_cue(args.cue, file_key=src.name)
        bounds = track_bounds(cs, total, meta["sr"])
        items = [(i + 1, s, e, cs[i]["title"], None) for i, (s, e) in enumerate(bounds)]
    else:
        cands = sorted(src.glob("*.flac"))
        items = [(i + 1, 0, 0, c.stem, c) for i, c in enumerate(cands)]

    ml_ready = False
    if args.ml_snapshot:
        try:
            from rnext.ml import runtime as mlrt
            ml_ready = mlrt.snapshot_info(None if args.ml_snapshot == "auto"
                                          else args.ml_snapshot) is not None
            if not ml_ready:
                print("WARN: снапшот не найден — работаем по авто-политике", flush=True)
        except Exception as e:                             # noqa: BLE001
            print(f"WARN: ML недоступен ({type(e).__name__}) — авто-политика", flush=True)

    report = out_dir / "restore_report.jsonl"
    with report.open("w", encoding="utf-8") as fh:
        for ti, st, en, title, path in items:
            if cue_style := (path is None):
                pass
            if path is None:
                x, sr = F.decode_f32(str(src), start_s=st / float(sr or 44100), dur_s=(en - st) / float(sr or 44100))
            else:
                x, sr = F.decode_f32(str(path))
            if x is None or x.shape[0] < sr * 5:
                print(f"SKIP t{ti} {title}", flush=True)
                continue
            o_ = F.tt_dr(x, sr)
            y = restore_track(x, sr, pol)
            _d = F.tt_dr(y, sr)
            tp = true_peak_dbfs(y, sr)
            y16 = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
            flac_path = out_dir / f"{ti:02d}. {title}.flac"
            encode_flac(y16, sr, flac_path, out_dir / "__tmp.wav")
            rec = {
                "track": ti, "title": title,
                "mode": args.mode, "r": args.r, "tp_target": args.tp, "tp_actual": round(tp, 2),
                "orig": {"dr": o_[0], "peak_dbfs": round(o_[3], 2), "rms_db": round(o_[4], 2)},
                "rest": {"dr": _d[0], "peak_dbfs": round(_d[3], 2), "rms_db": round(_d[4], 2)},
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"t{ti:>2} DR {o_[0]:>2}->{_d[0]:>2}  TP {tp:6.2f}  {title}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
