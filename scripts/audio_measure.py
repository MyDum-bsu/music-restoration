#!/usr/bin/env python3
"""Измерение качества аудио: DR, громкость, пики и — главное — подлинность.

Отвечает на вопросы, которые возникают при выборе версии релиза:
  * какой это DR (TT-конвенция, как в корпусе) и насколько он убит;
  * не фейковый ли это hi-res (24/96 при пустоте выше 22 кГц);
  * не перекодирован ли FLAC из lossy (обрыв спектра на 16–20 кГц);
  * нет ли клиппинга и сколько сэмплов на пределе;
  * совпадает ли заявленная битность/частота с фактической.

  python3 scripts/audio_measure.py <файл|каталог> [...]        # таблица
  python3 scripts/audio_measure.py --json out.json <каталог>   # + машинный вывод
  python3 scripts/audio_measure.py --quick <каталог>           # без спектра (быстро)

Спектральная проверка считается по самому громкому участку (2 с), поэтому
требует декода; для больших библиотек есть --quick.
"""
import argparse
import json
import math
import statistics as S
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import features as F  # noqa: E402

AUDIO_EXT = {".flac", ".wav", ".ape", ".m4a", ".mp3", ".ogg", ".opus", ".aiff", ".alac"}


def ffprobe(path: Path) -> dict:
    try:
        return F.ffprobe_audio(str(path))
    except Exception:  # noqa: BLE001
        return {}


def decode(path: Path, max_s: float = 90.0):
    """Декод в float32; для длинных файлов — только начало (DR по фрагменту)."""
    try:
        x, sr = F.decode_f32(str(path), 0.0, max_s)
        if x.size == 0:
            x, sr = F.decode_f32(str(path))
        return x, sr
    except Exception:  # noqa: BLE001
        return None, None


def loudest_window(x: np.ndarray, sr: int, win_s: float = 2.0) -> np.ndarray:
    """Самый громкий участок — по нему считаем спектр (тихие места врут)."""
    n = int(win_s * sr)
    if len(x) <= n:
        return x
    mono = np.abs(x).mean(axis=1) if x.ndim > 1 else np.abs(x)
    step = max(1, n // 4)
    best_i, best_e = 0, -1.0
    for i in range(0, len(mono) - n + 1, step):
        e = float(np.mean(mono[i:i + n]))
        if e > best_e:
            best_e, best_i = e, i
    return x[best_i:best_i + n]


def spectral_cutoff(x: np.ndarray, sr: int, drop_db: float = 55.0) -> tuple[float, float, float]:
    """Верхняя граница спектра и крутизна обрыва.

    Возвращает (cutoff_hz, rolloff_db_per_khz). Lossy-транскод виден как
    обрыв на 16–20 кГц с последующим «ничем»; честный CD — до ~21 кГц.
    """
    seg = loudest_window(x, sr)
    if seg.ndim > 1:
        seg = seg.mean(axis=1)
    n = 1 << int(math.floor(math.log2(len(seg))))
    if n < 4096:
        return 0.0, 0.0, 0.0
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(seg[:n] * win))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    # сглаживание по 1/12 октавы, чтобы не поймать узкий провал
    k = 64
    spec = np.convolve(spec, np.ones(k) / k, mode="same")
    peak = spec.max()
    if peak <= 0:
        return 0.0, 0.0
    thresh = peak * 10 ** (-drop_db / 20)
    idx = np.where(spec > thresh)[0]
    if len(idx) == 0:
        return 0.0, 0.0, 0.0
    cut = float(freqs[idx[-1]])
    # крутизна: сколько дБ теряется в последних 2 кГц перед обрывом
    hi = idx[-1]
    lo = np.searchsorted(freqs, max(0.0, cut - 2000))
    if hi > lo + 1 and spec[lo] > 0:
        roll = 20 * math.log10(max(spec[hi], 1e-12) / spec[lo]) / ((cut - freqs[lo]) / 1000)
    else:
        roll = 0.0
    # «пол» выше среза: у lossy-транскода там практически пусто (−80 дБ и ниже),
    # у честного мастера остаётся естественный спад
    passband = spec[(freqs > 1000) & (freqs < max(2000, cut - 1000))]
    above = spec[(freqs > cut + 500) & (freqs < freqs[-1])]
    pb = float(np.median(passband)) if passband.size else 0.0
    hf = float(np.median(above)) if above.size else 0.0
    floor_db = 20 * math.log10(max(hf, 1e-12) / max(pb, 1e-12)) if pb > 0 else 0.0
    return cut, float(roll), float(floor_db)


def lufs_tp(path: Path) -> tuple[float | None, float | None]:  # noqa: D401
    """Интегральная громкость и true peak через ffmpeg ebur128."""
    try:
        r = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path),
                            "-filter:a", "ebur128=peak=true", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, None
    out = r.stderr
    lufs = tp = None
    # итоговая сводка печатается в конце: берём ПОСЛЕДНИЕ блоки
    i = out.rfind("Integrated loudness")
    if i >= 0:
        for line in out[i:].splitlines()[:4]:
            if "I:" in line:
                try:
                    lufs = float(line.split("I:")[1].split("LUFS")[0].strip())
                except ValueError:
                    pass
    j = out.rfind("True peak")
    if j >= 0:
        for line in out[j:].splitlines()[:4]:
            if "Peak:" in line:
                try:
                    tp = float(line.split("Peak:")[1].split("dBFS")[0].strip())
                except ValueError:
                    pass
    return lufs, tp


def analyse(path: Path, quick: bool = False) -> dict:
    info = ffprobe(path)
    sr_decl = info.get("sr")
    bits_decl = info.get("bps") or info.get("bits")
    out = {"file": str(path), "name": path.name, "sr": sr_decl, "bits": bits_decl,
           "dur_s": round(info.get("dur") or 0, 1)}
    x, sr = decode(path)
    if x is None or x.size == 0:
        out["error"] = "не декодируется"
        return out
    dr, drx, _ch, peak, rms = F.tt_dr(x, sr)
    out.update(dr=dr, dr_exact=round(float(drx), 2), crest_tt=round(peak - rms, 2),
               peak_dbfs=round(peak, 2), rms_db=round(rms, 2), sr_actual=sr)
    x16 = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)
    clipped = int(np.sum(np.abs(x16) >= 32767))
    out["clipped_samples"] = clipped
    out["clipped_pct"] = round(100.0 * clipped / max(x16.size, 1), 4)
    if not quick:
        cut, roll, floor_db = spectral_cutoff(x, sr)
        nyq = sr / 2
        out.update(cutoff_hz=round(cut), rolloff_db_per_khz=round(roll, 1),
                   hf_floor_db=round(floor_db, 1), nyquist_hz=round(nyq))
        # фейковый hi-res: заявлено >44.1 кГц, а контента выше 22 кГц нет
        out["fake_hires"] = bool(sr > 48000 and cut < 0.9 * 22050)
        # lossy-транскод = обрыв НИЖЕ 19 кГц И практически пустота выше среза.
        # Одного низкого среза мало: тёмный или сильно лимитированный мастер
        # тоже даёт низкий срез, но не даёт «стены».
        out["lossy_suspect"] = bool(cut and cut < 19000 and floor_db < -75)
    lufs, tp = lufs_tp(path)
    out["lufs_i"] = lufs
    out["tp_dbfs"] = tp
    return out


def verdict(r: dict) -> list[str]:
    v = []
    if r.get("error"):
        return ["не читается"]
    if r.get("fake_hires"):
        v.append("ФЕЙКОВЫЙ HI-RES")
    if r.get("lossy_suspect"):
        v.append("подозрение на lossy-транскод")
    if r.get("clipped_pct", 0) > 0.05:
        v.append(f"клиппинг {r['clipped_pct']:.2f}%")
    if (r.get("dr") or 99) <= 6:
        v.append("сильно убит DR")
    return v or ["ок"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", default="")
    ap.add_argument("--quick", action="store_true", help="без спектра и LUFS")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        pp = Path(p).expanduser()
        if pp.is_dir():
            files += sorted(f for f in pp.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        elif pp.is_file():
            files.append(pp)
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("нет аудиофайлов")
        return 1

    rows = []
    print(f"{'файл':<44}{'формат':>14}{'DR':>4}{'crest':>7}{'LUFS':>7}{'TP':>7}{'срез':>7}  вердикт")
    for f in files:
        r = analyse(f, quick=args.quick)
        rows.append(r)
        fmt = f"{r.get('sr') or '?'}Гц/{r.get('bits') or '?'}бит"
        cut = r.get("cutoff_hz")
        print(f"{f.name[:43]:<44}{fmt:>14}{r.get('dr', '-'):>4}{r.get('crest_tt', 0):>7.2f}"
              f"{(r.get('lufs_i') if r.get('lufs_i') is not None else 0):>7.1f}"
              f"{(r.get('tp_dbfs') if r.get('tp_dbfs') is not None else 0):>7.2f}"
              f"{(cut if cut else 0):>7}  {', '.join(verdict(r))}")
        sys.stdout.flush()

    ok = [r for r in rows if not r.get("error")]
    if ok:
        print(f"\nитог по {len(ok)} файлам: DR медиана {S.median([r['dr'] for r in ok]):.0f} "
              f"(диапазон {min(r['dr'] for r in ok)}…{max(r['dr'] for r in ok)}), "
              f"LUFS медиана {S.median([r['lufs_i'] for r in ok if r.get('lufs_i')] or [0]):.1f}, "
              f"клиппинг есть в {sum(1 for r in ok if r.get('clipped_pct', 0) > 0.05)} файлах, "
              f"фейковый hi-res — {sum(1 for r in ok if r.get('fake_hires'))}, "
              f"lossy-подозрений — {sum(1 for r in ok if r.get('lossy_suspect'))}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"json → {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
