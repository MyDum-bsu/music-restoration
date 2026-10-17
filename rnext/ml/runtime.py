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


def predict_targets(feature_rows: list[dict], n_tracks: int,
                    model_id: str | None = None) -> np.ndarray | None:
    """Цель для треков альбома: общий вход — строки фич исходника.

    Возвращает Δcrest (дБ) по треку либо None, если снапшота нет.
    """
    model_id = model_id or reg.latest_snapshot()
    if not model_id:
        return None
    model, meta = _load(model_id)
    names = meta.get("features") or ds.FEATURES
    X = np.array([[ds._row_features({"original": r}, n_tracks).get(f, np.nan) for f in names]
                  for r in feature_rows], dtype=float)
    pred = np.asarray(model.predict(X), dtype=float)
    # физические скобки: модель не имеет права заказывать перелёт
    cap = float(meta.get("caps", {}).get("dcrest_max", 8.0))
    return np.clip(pred, -1.5, cap)


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
