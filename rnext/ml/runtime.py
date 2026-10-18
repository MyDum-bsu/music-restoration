"""Применение снапшота в пайплайне: предсказание цели Δcrest по треку.

Пайплайн получает цель из модели, если снапшот есть и он совместим по признакам;
иначе — прежняя логика (классовая таблица / константа). Это оставляет
возможность отката: модель — надстройка над физикой, а не её замена.
"""
from pathlib import Path

import numpy as np

from rnext.ml import dataset as ds
from rnext.ml import models as M
from rnext.ml import registry as reg

_cache: dict[str, tuple] = {}


def _load(model_id: str):
    if model_id not in _cache:
        _cache[model_id] = reg.load_snapshot(model_id)
    return _cache[model_id]


def _class_target(dr: float) -> float:
    """Цель по классу исходника (та же таблица, что в проде по умолчанию)."""
    t = v0_targets()
    if dr >= 12:
        return float(t.get("healthy", 0.0))
    if dr >= 9:
        return float(t.get("mid", 2.3))
    return float(t.get("crushed", 5.3))


def v0_targets() -> dict:
    import json as _json
    from rnext import workspace
    p = workspace.dir_of("corpus") / "targets_v0.json"
    if p.is_file():
        return _json.loads(p.read_text(encoding="utf-8")).get("recommendation", {})
    return {}


# Физические признаки «не тот материал»: работают даже без статистик обучения.
# Валидация нашла: на моно-синтетике ridge уходил в +33 дБ — эти правила
# отправляют такой вход на классовую таблицу.
def physical_ood(r: dict) -> str | None:
    side = r.get("side_mid_db")
    if side is not None and np.isfinite(side) and side < -40:
        return "mono (side/mid < -40 дБ)"
    tilt = r.get("tilt_low_high")
    if tilt is not None and np.isfinite(tilt) and (tilt > 35 or tilt < 5):
        return f"экстремальный тилт ({tilt:.0f} дБ)"
    if (r.get("rms_spread") or 0) == 0 and (r.get("attacks_per_s") or 0) == 0:
        return "нет динамики/транзиентов (синтетика или тишина)"
    return None


def ood_z(X: np.ndarray, names: list[str], meta: dict) -> tuple[float, str | None]:
    """Насколько вход выбивается из обучающего распределения (макс |z|)."""
    stats = meta.get("feature_stats") or {}
    worst, worst_name = 0.0, None
    for i, name in enumerate(names):
        st = stats.get(name)
        if not st or not st.get("std"):
            continue
        z = abs((X[0, i] - st["mean"]) / st["std"]) if np.isfinite(X[0, i]) else 0.0
        if z > worst:
            worst, worst_name = z, name
    return worst, worst_name


def predict_targets(feature_rows: list[dict], n_tracks: int,
                    model_id: str | None = None,
                    ood_max_z: float = 6.0,
                    class_margin_db: float = 2.5,
                    sources: list | None = None) -> np.ndarray | None:
    """Цель для треков альбома: общий вход — строки фич исходника.

    Защиты (найдено валидацией: на непохожем материале ridge уходил в +33 дБ):
      * OOD-гейт: если признак выбивается из обучающего распределения (|z| > 6),
        модель не используется — берётся классовая таблица;
      * скобка вокруг классовой цели ±2.5 дБ: даже «своя» модель не имеет права
        заказывать перелёт;
      * физический кап Δcrest ≤ 8 дБ.
    Возвращает Δcrest по треку либо None (тогда работает авто-политика).
    """
    model_id = model_id or reg.latest_snapshot()
    if not model_id:
        return None
    model, meta = _load(model_id)
    names = meta.get("features") or ds.FEATURES
    rows = [ds._row_features({"original": r}, n_tracks) for r in feature_rows]
    X = np.array([[r.get(f, np.nan) for f in names] for r in rows], dtype=float)
    pred = np.asarray(model.predict(X), dtype=float)
    cap = float(meta.get("caps", {}).get("dcrest_max", 8.0))
    out = []
    for i, r in enumerate(rows):
        dr = r.get("dr", 10.0)
        base = _class_target(dr)
        phys = physical_ood(r)
        z, zname = ood_z(X[i:i + 1], names, meta)
        if phys or z > ood_max_z:
            # материал вне обучающего распределения (напр. моно, иной тилт)
            out.append(base)
            if sources is not None:
                sources.append(f"class_fallback ({phys or f'{zname} z={z:.1f}'})")
            continue
        val = min(max(float(pred[i]), base - class_margin_db), base + class_margin_db)
        out.append(val)
        if sources is not None:
            sources.append("model" if val == float(pred[i]) else f"model_clamped({float(pred[i]):+.1f})")
    return np.clip(np.array(out, dtype=float), -1.5, cap)


def predict_from_audio_src(x, sr, model_id: str | None = None, n_tracks: int = 10):
    """Как predict_from_audio, но возвращает (цель, источник решения)."""
    model_id = model_id or reg.latest_snapshot()
    if not model_id:
        return None, None
    from rnext import features as F
    try:
        feats = F.feature_dict(x, sr)
    except Exception:                                     # noqa: BLE001
        return None, None
    src: list = []
    pred = predict_targets([feats], n_tracks, model_id=model_id, sources=src)
    if pred is None:
        return None, None
    return float(pred[0]), (src[0] if src else None)


def predict_from_audio(x, sr, model_id: str | None = None, n_tracks: int = 10) -> float | None:
    """Цель Δcrest для одного трека по его аудио (для restore_one)."""
    model_id = model_id or reg.latest_snapshot()
    if not model_id:
        return None
    from rnext import features as F
    try:
        feats = F.feature_dict(x, sr)
    except Exception:                                     # noqa: BLE001
        return None
    pred = predict_targets([feats], n_tracks, model_id=model_id)
    return None if pred is None else float(pred[0])


def snapshot_info(model_id: str | None = None) -> dict | None:
    model_id = model_id or reg.latest_snapshot()
    if not model_id:
        return None
    try:
        _, meta = _load(model_id)
        return meta
    except Exception:                                     # noqa: BLE001
        return None
