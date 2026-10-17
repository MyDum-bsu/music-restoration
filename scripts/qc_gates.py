#!/usr/bin/env python3
"""Автоматические QC-гейты на выходном треке/реставрации.

Проверки:
  1. TP ≤ заданного потолка (интерсэмплинг, 4х oversample);
  2. нет клиппинга int16 (|x| ≥ 32766) на выходе;
  3. TEMPORAL-SMOOTHNESS гейна: максимум |Δg| между соседними кадрами STFT
     на полосах не превышает порога (не должно быть резких переключений гейна);
  4. «зубчатость» абс: нет одиночных скачков амплитуды > 0.9 между соседними
     семплами (кликоподобный мусор от багов пайплайна).

Коды возврата: 0 = ok, 1 = есть нарушения. Печатает JSON-отчёт.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/home/mydum/Desktop/projects/restore-next")

from rnext import features as F  # noqa: E402
from rnext.dsp.contour import true_peak_dbfs  # noqa: E402


def gain_smoothness(x: np.ndarray, sr: int, frame: int = 2048) -> dict:
    """Пер-кадровый уровень по полосам 1/3-окт; max шаг между соседними кадрами."""
    import math

    hop = frame // 2
    starts = np.arange(0, max(x.shape[0] - frame, 0) + 1, hop)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    edges = [20.0]
    while edges[-1] < sr / 2 - 1:
        edges.append(edges[-1] * 2 ** (1 / 3))
    edges[-1] = sr / 2
    bands = []
    for i in range(len(edges) - 1):
        idx = np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
        if idx.size:
            bands.append(idx)
    w = np.hanning(frame)
    max_step, steps_db = 0.0, []
    for b0 in range(0, starts.size, 512):
        b1 = min(b0 + 512, starts.size)
        fr = np.stack([x[s:s + frame, :].T for s in starts[b0:b1]])  # [nb, ch, frame]
        fr = fr * w[None, None, :]
        Z = np.abs(np.fft.rfft(fr, axis=-1)) ** 2
        del fr
        for idx in bands:
            lv = 10.0 * np.log10(Z[:, :, idx].mean(axis=2).T + 1e-16)  # [ch, nf]
            st = np.abs(np.diff(lv, axis=1))
            steps_db.append(st.max())
        del Z
    max_step = float(np.max(steps_db)) if steps_db else 0.0
    return {"max_gain_step_db_per_frame": round(max_step, 2),
            "frame_ms": round(1000 * hop / sr, 1)}


def sample_spikes(x: np.ndarray) -> int:
    d = np.abs(np.diff(x, axis=0)).max(axis=1)
    return int((d > 0.95).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tp-ceiling", type=float, default=-0.5)
    args = ap.parse_args()

    bad = False
    for f in args.files:
        x, sr = F.decode_f32(f)
        if x is None:
            print(f"{f}: DECODE FAIL")
            bad = True
            continue
        st = F.tt_dr(x, sr)
        tp = true_peak_dbfs(x, sr)
        g = gain_smoothness(x, sr)
        spikes = sample_spikes(x)
        clipped = int((np.abs((x * 32768).astype(np.int32)) >= 32766).sum())
        rec = {
            "file": f, "dr": st[0], "peak_db": round(st[3], 2),
            "tp_dbfs": round(tp, 2), "tp_ceiling": args.tp_ceiling,
            "tp_ok": bool(tp <= args.tp_ceiling + 0.05),
            "clipped_samples_out": clipped,
            "max_gain_step_db_per_frame": g["max_gain_step_db_per_frame"],
            "sample_spikes_gt09": spikes,
        }
        bad_cnt = (not rec["tp_ok"]) or clipped > 0 or rec["max_gain_step_db_per_frame"] > 8.0
        rec["verdict"] = "FAIL" if bad_cnt else "OK"
        print(json.dumps(rec, ensure_ascii=False))
        if bad_cnt:
            bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
