#!/usr/bin/env python3
"""Подгонка «тилта» (тональной кривой) мастера по корпусу → corpus/tilt_curve.json.

Что измеряем: для каждого трека — разницу «мастер минус оригинал» по 1/3-окт
полосам (уже посчитана в фичах как delta.bands_db), из которой снят ОБЩИЙ сдвиг
уровня (иначе разница громкости маскирует тембр). Затем медиана по трекам —
это и есть средняя кривая реставрации (у наших мастеров — мягкий U: саб-бас
+1…2 дБ, ямка в низах-середине, подъём верха).

Кривая обрезается по cap (|гейн полосы| ≤ cap дБ) и умножается на scale —
чтобы не закладывать чужой вкус целиком. Пишется как список (Гц, дБ) для
`Policy.eq_bands_db`, плюс метаданные: n треков, авторы, разброс по полосам.

  python3 scripts/fit_tilt.py                 # подогнать и записать
  python3 scripts/fit_tilt.py --author 5618698 --scale 0.5
  python3 scripts/fit_tilt.py --report        # показать текущую кривую
"""
import argparse
import json
import statistics as S
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import workspace  # noqa: E402

EDGES = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
         800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
         12500, 16000, 20000]
FC = [(EDGES[i] * EDGES[i + 1]) ** 0.5 for i in range(len(EDGES) - 1)]
LOW = slice(3, 9)      # ~40–160 Гц
HIGH = slice(21, 28)   # ~2–8 кГц


def corpus() -> Path:
    return workspace.dir_of("corpus")


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def collect(author: str | None, trusted_only: bool = True) -> tuple[list[list[float]], Counter]:
    plan = {r["key"]: r for r in read_jsonl(corpus() / "dataset_plan.jsonl")}
    qual = {r["key"]: r for r in read_jsonl(dataset_dir() / "reference_quality.jsonl")}
    curves: list[list[float]] = []
    authors = Counter()
    for f in sorted((dataset_dir() / "pairs").glob("*/features.json")):
        key = f.parent.name.replace("__", "/")
        if trusted_only and qual.get(key, {}).get("label") not in ("trusted", "weak_base"):
            continue
        au = str((plan.get(key) or {}).get("author_pid"))
        if author and au != str(author):
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            b = r["delta"]["bands_db"]
            if len(b) != len(FC):
                continue
            m = sum(b) / len(b)                 # снимаем общий сдвиг уровня
            curves.append([x - m for x in b])
            authors[au] += 1
    return curves, authors


def smooth(g: list[float], passes: int = 1) -> list[float]:
    out = list(g)
    for _ in range(passes):
        out = [out[0]] + [(out[i - 1] + 2 * out[i] + out[i + 1]) / 4
                          for i in range(1, len(out) - 1)] + [out[-1]]
    return out


def build(args) -> int:
    curves, authors = collect(args.author)
    if len(curves) < args.min_tracks:
        print(f"мало данных: {len(curves)} треков (нужно ≥{args.min_tracks})")
        return 1
    raw = [S.median([c[i] for c in curves]) for i in range(len(FC))]
    iqr = [S.quantiles([c[i] for c in curves], n=4)[2] - S.quantiles([c[i] for c in curves], n=4)[0]
           for i in range(len(FC))]
    sm = smooth(raw, passes=args.smooth)
    gain = [round(max(-args.cap, min(args.cap, x * args.scale)), 2) for x in sm]
    tilt_raw = sum(raw[LOW]) / 6 - sum(raw[HIGH]) / 7
    tilt_out = sum(gain[LOW]) / 6 - sum(gain[HIGH]) / 7
    doc = {
        "generated": __import__("time").strftime("%Y-%m-%d %H:%M"),
        "n_tracks": len(curves), "n_pairs": len({c for c in authors}), "authors": dict(authors),
        "band_hz": [round(x, 1) for x in FC],
        "gain_db_raw": [round(x, 3) for x in raw],
        "gain_db": gain,
        "spread_iqr_db": [round(x, 2) for x in iqr],
        "cap_db": args.cap, "scale": args.scale, "smooth_passes": args.smooth,
        "tilt_raw_db": round(tilt_raw, 2), "tilt_out_db": round(tilt_out, 2),
        "note": "медиана (мастер − оригинал) по 1/3-окт полосам, общий сдвиг уровня снят; "
                "gain_db обрезан по cap и умножен на scale",
    }
    out = corpus() / "tilt_curve.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"кривая тилта → {out}")
    print(f"треков {len(curves)}, авторы {dict(authors)} | тилт сырой {tilt_raw:+.2f} дБ, "
          f"после cap/scale {tilt_out:+.2f} дБ")
    print(f"{'Гц':>7}{'сырая':>9}{'итог':>9}{'IQR':>8}")
    for i in range(0, len(FC), 2):
        print(f"{FC[i]:>7.0f}{raw[i]:>+9.2f}{gain[i]:>+9.2f}{iqr[i]:>8.2f}")
    return 0


def report() -> int:
    p = corpus() / "tilt_curve.json"
    if not p.is_file():
        print("кривой нет — запустите без --report")
        return 1
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"кривая от {d['generated']}: треков {d['n_tracks']}, авторы {d['authors']}, "
          f"тилт {d['tilt_out_db']:+.2f} дБ (сырой {d['tilt_raw_db']:+.2f}), "
          f"cap {d['cap_db']}, scale {d['scale']}")
    for hz, g, sp in zip(d["band_hz"], d["gain_db"], d["spread_iqr_db"]):
        print(f"  {hz:>7.0f} Гц  {g:>+6.2f} дБ  (IQR {sp:.2f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default=None, help="считать только по этому автору")
    ap.add_argument("--cap", type=float, default=2.0, help="макс. |гейн полосы|, дБ")
    ap.add_argument("--scale", type=float, default=1.0, help="масштаб кривой")
    ap.add_argument("--smooth", type=int, default=1, help="проходов сглаживания")
    ap.add_argument("--min-tracks", type=int, default=50)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    return report() if args.report else build(args)


if __name__ == "__main__":
    sys.exit(main())
