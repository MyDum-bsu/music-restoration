#!/usr/bin/env python3
"""Реставрация целого альбома оригинала → каталог с FLAC'ами + отчёт.

Вход: image+cue (--src image --cue file.cue) либо каталог с треками (*.flac),
либо один файл. Выход:
  out/NN. Title.flac — 16-бит FLAC (TP-нормализация к потолку политики)
  out/restore_report.jsonl — метрики каждого трека (DR/crest/peak/rms до/после)

Авто-политика включена по умолчанию: класс материала (healthy/sparse/loud)
определяет, включать ли HPF/контур и какую цель Δcrest ставить. Цель можно взять
из модели (`--ml-snapshot auto`), тогда порог healthy согласуется с обучением.

Мьютекс: параллельный запуск двух restore забивал своп, поэтому берём flock на
/media/.../restore/.restore.lock (неблокирующий; --wait — ждать).
"""
import argparse
import dataclasses
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import features as F  # noqa: E402
from rnext import workspace  # noqa: E402
from rnext.dsp.contour import true_peak_dbfs  # noqa: E402
from rnext.dsp.pipeline import Policy, restore_track  # noqa: E402

LOCK = workspace.data_root() / ".restore.lock"


class RestoreBusy(RuntimeError):
    pass


def acquire_lock(wait: bool = False):
    f = open(LOCK, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BlockingIOError as e:
        raise RestoreBusy(f"занят: {LOCK}") from e
    f.seek(0)
    f.truncate()
    f.write(f"pid={os.getpid()}\n")
    f.flush()
    return f


def encode_flac(y16: np.ndarray, sr: int, out_path: Path, tmp_wav: Path) -> None:
    import wave
    with wave.open(str(tmp_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y16.astype("<i2").tobytes())
    subprocess.run(["flac", "-s", "-f", "-8", "-o", str(out_path), str(tmp_wav)], check=True)
    tmp_wav.unlink(missing_ok=True)


def _clean_title(stem: str) -> str:
    """Убрать номер трека из имени файла: «01 - Foo» → «Foo».

    Номер всё равно печатается префиксом при сохранении, иначе получалось
    «01. 01 In the Name of War.flac».
    """
    import re as _re
    s = stem.strip()
    s = _re.sub(r"^\s*\d{1,3}\s*(?:[-._)]|\s)\s*", "", s, count=1)
    return s.strip() or stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="image (с --cue) или каталог/файл с треками")
    ap.add_argument("--cue", default="", help="cue, если --src это image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="expand", choices=["expand", "flatten"])
    ap.add_argument("--r", type=float, default=1.5)
    ap.add_argument("--k", type=float, default=0.55)
    ap.add_argument("--tp", type=float, default=-0.5, help="target TP, dBFS")
    ap.add_argument("--adaptive", action="store_true", help="подгонять r под target Δcrest")
    ap.add_argument("--target", type=float, default=4.6, help="целевой Δcrest (adaptive)")
    ap.add_argument("--auto-policy", dest="auto_policy", action="store_true", default=True,
                    help="класс материала → политика (по умолчанию включено)")
    ap.add_argument("--no-auto", dest="auto_policy", action="store_false",
                    help="выключить авто-политику, работать по ручным --mode/--target/--adaptive")
    ap.add_argument("--eq-curve", default="", help="JSON с кривой тилта (scripts/fit_tilt.py)")
    ap.add_argument("--eq-scale", type=float, default=1.0, help="масштаб кривой тилта")
    ap.add_argument("--eq-match", action="store_true",
                    help="замкнутый EQ: измеряем дельту полос и добираем разницу с кривой")
    ap.add_argument("--eq-match-cap", type=float, default=2.0,
                    help="предел коррекции на полосу, дБ (для --eq-match)")
    ap.add_argument("--ml-snapshot", default="",
                    help="id снапшота цели (или 'auto' — последний); цель берётся из модели")
    ap.add_argument("--wait", action="store_true", help="ждать освобождения лока restore")
    args = ap.parse_args()

    try:
        lock = acquire_lock(wait=args.wait)
    except RestoreBusy as e:
        print(f"BUSY: {e}")
        return 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Авто-политика: режим expand + adaptive включаются сами, цель берёт diagnose().
    mode = "expand" if args.auto_policy else args.mode
    adaptive = args.adaptive or args.auto_policy
    pol = Policy(mode=mode, r_expand=args.r, k=args.k, target_tp_dbfs=args.tp,
                 adaptive=adaptive, target_dcrest=args.target,
                 auto_policy=args.auto_policy)
    eq_tilt = None
    if args.eq_curve:
        import json as _json
        cur = _json.loads(Path(args.eq_curve).read_text(encoding="utf-8"))
        bands = [(float(hz), float(g) * args.eq_scale)
                 for hz, g in zip(cur["band_hz"], cur["gain_db"])]
        if args.eq_match:
            pol = dataclasses.replace(pol, eq_match=True, eq_match_curve=bands,
                                      eq_match_cap_db=args.eq_match_cap,
                                      normalize_tp=True)
        else:
            pol = dataclasses.replace(pol, eq_bands_db=bands)
        eq_tilt = {"curve": str(args.eq_curve), "scale": args.eq_scale,
                   "n_tracks_fit": cur.get("n_tracks"), "tilt_db": cur.get("tilt_out_db")}
        mode_txt = f"замкнутый, cap {args.eq_match_cap} дБ" if args.eq_match else "фиксированный"
        print(f"тилт: {len(bands)} полос из {args.eq_curve} ({mode_txt}), "
              f"низ {bands[3][1]:+.2f} / верх {bands[21][1]:+.2f} дБ", flush=True)
    src = Path(args.src)

    if args.cue and src.is_file():
        from rnext.cue import parse_cue, track_bounds
        meta = F.ffprobe_audio(str(src))
        total = int(round(meta["dur"] * meta["sr"]))
        cs = parse_cue(args.cue, file_key=src.name)
        bounds = track_bounds(cs, total, meta["sr"])
        items = [(i + 1, s, e, cs[i]["title"], None) for i, (s, e) in enumerate(bounds)]
    else:
        cands = sorted(src.glob("*.flac")) if src.is_dir() else [src]
        items = [(i + 1, 0, 0, _clean_title(c.stem), c) for i, c in enumerate(cands)]

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
            if path is None:
                sr0 = F.ffprobe_audio(str(src))["sr"]
                x, sr = F.decode_f32(str(src), start_s=st / sr0, dur_s=(en - st) / sr0)
            else:
                x, sr = F.decode_f32(str(path))
            if x is None or x.shape[0] < sr * 5:
                print(f"SKIP t{ti} {title}", flush=True)
                continue
            o_ = F.tt_dr(x, sr)
            pol_t = pol
            ml_target = None
            if args.ml_snapshot and ml_ready:
                mid = None if args.ml_snapshot == "auto" else args.ml_snapshot
                ml_target, ml_source = mlrt.predict_from_audio_src(x, sr, model_id=mid,
                                                                   n_tracks=len(items))
                if ml_target is not None:
                    # цель из модели + согласование классов: модель училась по
                    # границе healthy ≥ 12, а не ≥ 11
                    pol_t = dataclasses.replace(
                        pol, healthy_dr_min=12,
                        target_dcrest=float(min(ml_target, 8.0) if ml_target > 0 else ml_target))
                    print(f"  t{ti}: цель {ml_target:+.2f} дБ [{ml_source}]", flush=True)
            y, stages = restore_track(x, sr, pol_t, return_stages=True)
            _d = F.tt_dr(y, sr)
            tp = true_peak_dbfs(y, sr)
            # TPDF-дизеринг при 16-битном квантовании: убирает коррелированную
            # ошибку округления (в особенности на тихих хвостах)
            rng = np.random.default_rng(0xC0FFEE + ti)
            dith = rng.random(y.shape) + rng.random(y.shape) - 1.0   # TPDF ±1 LSB
            y16 = np.clip(np.round(y * 32768.0 + dith), -32768, 32767).astype(np.int16)
            flac_path = out_dir / f"{ti:02d}. {title}.flac"
            encode_flac(y16, sr, flac_path, out_dir / "__tmp.wav")
            dx = stages.get("diagnosis", {})
            rec = {
                "track": ti, "title": title,
                "mode": mode, "r": args.r, "adaptive": adaptive,
                "auto_policy": args.auto_policy,
                "class": dx.get("class"),
                "target_dcrest": dx.get("target_dcrest", args.target),
                "ml_snapshot": args.ml_snapshot or None,
                "eq_tilt": eq_tilt,
                "eq_match": (stages.get("eq_match") or None) if args.eq_match else None,
                "ml_target": ml_target,
                "ml_source": ml_source,
                "r_fit": (stages.get("adaptive") or {}).get("r_fit"),
                "tp_target": args.tp, "tp_actual": round(tp, 2),
                "orig": {"dr": o_[0], "dr_exact": round(o_[1], 2),
                         "crest_tt": round(o_[3] - o_[4], 2),
                         "peak_dbfs": round(o_[3], 2), "rms_db": round(o_[4], 2)},
                "rest": {"dr": _d[0], "dr_exact": round(_d[1], 2),
                         "crest_tt": round(_d[3] - _d[4], 2),
                         "peak_dbfs": round(_d[3], 2), "rms_db": round(_d[4], 2)},
                "delta": {"dr": _d[0] - o_[0], "crest": round((_d[3] - _d[4]) - (o_[3] - o_[4]), 2)},
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"t{ti:>2} DR {o_[0]}->{_d[0]}  TP {tp:5.2f}  "
                  f"[{dx.get('class')}] {title}", flush=True)

    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    print(f"готово: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
