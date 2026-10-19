#!/usr/bin/env python3
"""Сквозная оценка пайплайна на вырезках пар: наш Δcrest против мастерского.

Считаем по каждому треку:
    master.dcrest = Δcrest мастера (crest мастера − crest исходника)
    ours.dcrest   = Δcrest нашего прогона (restore_track с политикой варианта)
    err           = ours − master
Базовая линия «ничего не делать» даёт err = −master.dcrest.

Работаем на ВЫРЕЗКАХ (`pairs/<key>/excerpts/`), а не на сырье. Причины:
  * вырезки постоянны — сырьё вытесняется окном хранения, и пары становятся
    неоцениваемыми;
  * обе стороны вырезаны по одному окну и выровнены по громкости (−20 dBFS RMS),
    поэтому Δcrest считается «яблоко к яблоку»; нормировка громкости crest не
    меняет (умножение на константу), так что сравнение корректно;
  * раньше оценка брала сырьё и индексы треков: после перекачки файлы
    переставлялись, окна разъезжались (DR 5 в фичах против DR 11 в прогоне) —
    и «ошибка» в 3–5 дБ была артефактом сравнения разных окон, а не DSP.

Варианты политики:
    flat  — дефолт: цель 4.6 на всё, что не healthy/sparse
    class — цель по классу исходника (corpus/targets_v0.json)
    model — цель от ML-снапшота (то, что делает рантайм пайплайна)

  python3 scripts/dataset_eval.py --all --variants flat,class,model
  python3 scripts/dataset_eval.py --all --holdout-only      # только не-в-выборке
"""
import argparse
import json
import re
import statistics as S
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_extract as DX  # noqa: E402
from rnext import features as F  # noqa: E402
from rnext import workspace  # noqa: E402
from rnext.dsp.pipeline import Policy, diagnose, restore_track  # noqa: E402

EX_RE = re.compile(r"^t(\d+)_(loud|mid)_(\d+)_(orig|master)\.flac$")


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def state_map() -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(dataset_dir() / "state.jsonl")}


def targets() -> dict:
    p = workspace.dir_of("corpus") / "targets_v0.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"recommendation": {}}


def policy_for(variant: str, rec: dict, dr: int, rms: float,
               ml_target: float | None = None) -> Policy:
    pol = Policy(auto_policy=True, adaptive=True, mode="expand")
    if variant == "class":
        t = rec.get("recommendation", {})
        if dr >= 12:
            pol.healthy_dr_min = 12
            pol.target_dcrest = 0.0
        elif dr >= 9:
            pol.healthy_dr_min = 12
            pol.target_dcrest = t.get("target_dcrest_mid", pol.target_dcrest)
        else:
            pol.target_dcrest = t.get("target_dcrest_crushed", pol.target_dcrest)
    elif variant == "model" and ml_target is not None:
        if dr >= 12:
            pol.healthy_dr_min = 12
        pol.target_dcrest = float(ml_target)
    return pol


def excerpt_sets(pdir: Path) -> dict[tuple[int, str], dict[str, Path]]:
    """{(track, tag): {'orig': path, 'master': path}} — только полные пары."""
    out: dict[tuple[int, str], dict[str, Path]] = {}
    for f in sorted((pdir / "excerpts").glob("*.flac")):
        m = EX_RE.match(f.name)
        if not m:
            continue
        track, tag, _frac, side = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        out.setdefault((track, tag), {})[side] = f
    return {k: v for k, v in out.items() if {"orig", "master"} <= set(v)}


def holdout_keys(seed: int = 17, frac: float = 0.3) -> set[str]:
    """Ключи пар, чьи альбомы НЕ попали в обучение рантайм-снапшота.

    Иначе модель оценивается на своём же обучающем материале — это не оценка.
    """
    try:
        from rnext.ml import dataset as DS
    except Exception:  # noqa: BLE001
        return set()
    t = DS.build_table()
    _tr, ho = DS.split(t, mode="by-album", frac=frac, seed=seed)
    groups = set(t.groups[ho].tolist())
    return {g for g in groups}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--key", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", default="flat,class,model")
    ap.add_argument("--holdout-only", action="store_true",
                    help="только пары, не попавшие в обучение рантайм-модели")
    ap.add_argument("--out", default="eval_results.jsonl")
    ap.add_argument("--summary-out", default="", help="куда записать сводку (json)")
    args = ap.parse_args()

    rec = targets()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    pairs_dir = dataset_dir() / "pairs"
    keys = sorted(p.parent.name.replace("__", "/") for p in pairs_dir.glob("*/features.json"))
    if args.key:
        keys = [k for k in keys if k == args.key]
    elif not args.all:
        ap.error("нужен --all или --key")
    hold = holdout_keys() if args.holdout_only else None
    if hold is not None:
        keys = [k for k in keys if k.replace("/", "__") in hold or k in hold]
    if args.limit:
        keys = keys[:args.limit]

    # цели ML считаем на альбом сразу (модель видит все треки пары)
    ml_by_pair: dict[str, dict[int, tuple[float, str]]] = {}
    if "model" in variants:
        try:
            from rnext.ml import registry as REG, runtime as RT
            mid = REG.latest_snapshot()
            for k in keys:
                pdir = pairs_dir / k.replace("/", "__")
                rows = json.loads((pdir / "features.json").read_text(encoding="utf-8"))
                src: list[str] = []
                tg = RT.predict_targets([r["original"] for r in rows],
                                        n_tracks=len(rows), model_id=mid, sources=src)
                if tg is not None:
                    ml_by_pair[k] = {rows[i]["track"]: (float(tg[i]),
                                                       src[i] if i < len(src) else "")
                                     for i in range(len(rows))}
            print(f"ML-снапшот: {mid}")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] ML-цель недоступна: {e}")

    print(f"пар к оценке: {len(keys)}, варианты {variants}"
          + (" (только не-в-выборке)" if hold is not None else ""))

    out_path = dataset_dir() / args.out
    out_f = open(out_path, "a", encoding="utf-8")
    for key in keys:
        pdir = pairs_dir / key.replace("/", "__")
        rows = json.loads((pdir / "features.json").read_text(encoding="utf-8"))
        by_track = {r["track"]: r for r in rows}
        sets = excerpt_sets(pdir)
        if not sets:
            print(f"  [skip] {key}: нет вырезок")
            continue
        for (track, tag), sides in sorted(sets.items()):
            r = by_track.get(track)
            if r is None:
                continue
            xo, sr = F.decode_f32(sides["orig"], 0.0, None)
            xm, _ = F.decode_f32(sides["master"], 0.0, None)
            if xo.size == 0 or xm.size == 0:
                print(f"  [skip] {key} t{track} {tag}: пустая вырезка")
                continue
            if len(xo) != len(xm):
                n = min(len(xo), len(xm))
                xo, xm = xo[:n], xm[:n]
            # Вырезки нормированы на −20 dBFS RMS. Для сравнения Δcrest это не важно
            # (умножение на константу crest не меняет), но политика смотрит на
            # АБСОЛЮТНУЮ громкость: при −20 dBFS всё уходило в ветку «sparse»
            # (цель 0.8) и все варианты давали одинаковый результат. Возвращаем
            # исходный уровень трека — crest при этом сохраняется.
            rms_now = float(np.sqrt(np.mean(xo.astype(np.float64) ** 2)) + 1e-12)
            want = 10 ** (float(r["original"]["rms_db"]) / 20.0)
            xo = (xo * (want / rms_now)).astype(np.float32)
            fo = DX.features_from_array(xo, sr)
            fm = DX.features_from_array(xm, sr)
            master_dcrest = fm["crest_db"] - fo["crest_db"]
            row = {"key": key, "track": track, "tag": tag, "title": r.get("title"),
                   "src_dr": r["original"]["dr"], "src_rms": r["original"]["rms_db"],
                   "excerpt_s": round(len(xo) / sr, 2),
                   "master_dcrest": round(master_dcrest, 3),
                   "master_ddr": fm["dr"] - fo["dr"]}
            for v in variants:
                tgt, msrc = ml_by_pair.get(key, {}).get(track, (None, ""))
                pol = policy_for(v, rec, r["original"]["dr"], r["original"]["rms_db"],
                                 ml_target=tgt)
                y, stages = restore_track(xo, sr, pol, return_stages=True)
                fy = DX.features_from_array(y, sr)
                row[f"{v}_dcrest"] = round(fy["crest_db"] - fo["crest_db"], 3)
                row[f"{v}_ddr"] = fy["dr"] - fo["dr"]
                row[f"{v}_tp"] = round(fy["peak_db"], 2)
                row[f"{v}_err"] = round(row[f"{v}_dcrest"] - master_dcrest, 3)
                row[f"{v}_r_fit"] = (stages.get("adaptive") or {}).get("r_fit")
                row[f"{v}_class"] = (stages.get("diagnosis") or {}).get("class")
                if v == "model":
                    row["model_target"] = round(tgt, 3) if tgt is not None else None
                    row["model_source"] = msrc
            row["noop_err"] = round(-master_dcrest, 3)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            errs = " ".join(f"{v}:{row[f'{v}_err']:+.2f}" for v in variants)
            print(f"  {key} t{track:<2}{tag:<5} DR{row['src_dr']:>2} маст {master_dcrest:+.2f} "
                  f"наш {errs} (без обработки: {row['noop_err']:+.2f})")
    out_f.close()

    all_rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    seen = set()
    rows = []
    for r in reversed(all_rows):  # последняя запись по (key, track, tag) — свежая
        sig = (r["key"], r["track"], r["tag"])
        if sig in seen:
            continue
        seen.add(sig)
        rows.append(r)
    rows = [r for r in rows if r["key"] in keys]
    if not rows:
        return 0
    summary = {"tracks": len(rows), "pairs": len(set(r["key"] for r in rows)),
               "variants": {}, "holdout_only": bool(args.holdout_only)}
    print(f"\nитог по {len(rows)} вырезкам из {summary['pairs']} пар (файл {out_path}):")
    print(f"{'вариант':<8}{'медиана |err|':>14}{'доля ≤0.5':>11}{'доля ≤1.0':>11}{'смещение':>10}")
    for v in variants + ["noop"]:
        e = [abs(r["noop_err"] if v == "noop" else r[f"{v}_err"]) for r in rows]
        med = float(S.median(e))
        bias = float(S.median([r["noop_err"] if v == "noop" else r[f"{v}_err"] for r in rows]))
        summary["variants"][v] = {
            "median_abs": round(med, 3),
            "share_le_0.5": round(sum(x <= 0.5 for x in e) / len(e), 3),
            "share_le_1.0": round(sum(x <= 1.0 for x in e) / len(e), 3),
            "bias": round(bias, 3),
        }
        print(f"{v:<8}{med:>14.2f}{sum(x <= 0.5 for x in e) / len(e) * 100:>10.0f}%"
              f"{sum(x <= 1.0 for x in e) / len(e) * 100:>10.0f}%{bias:>10.2f}")
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        print(f"сводка → {args.summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
