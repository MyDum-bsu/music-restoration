"""Applying a snapshot in the pipeline: predicting the delta-crest target per track.

The pipeline takes the target from the model if a snapshot exists and is
feature-compatible; otherwise the previous logic applies (class table /
constant). This keeps a rollback path: the model is an add-on over the
physics, not a replacement for it.
"""
from pathlib import Path

import numpy as np

from rnext.ml import dataset as ds
from rnext.ml import models as M
from rnext.ml import registry as reg

_cache: dict[str, tuple] = {}


def _load(model_id: str):
    """Snapshot loading with a per-process cache: without it every track would
    read model.joblib from disk (extra I/O and memory in a parallel run)."""
    if model_id not in _cache:
        _cache[model_id] = reg.load_snapshot(model_id)
    return _cache[model_id]


def _class_target(dr: float) -> float:
    """Target by source class (the same table used by the production default)."""
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


# Physical signs of "wrong material": they work even without training stats.
# Validation found: on mono synthetic material ridge drifted to +33 dB — these
# rules send such input to the class table.
def physical_ood(r: dict) -> str | None:
    side = r.get("side_mid_db")
    if side is not None and np.isfinite(side) and side < -40:
        return "mono (side/mid < -40 dB)"
    tilt = r.get("tilt_low_high")
    if tilt is not None and np.isfinite(tilt) and (tilt > 35 or tilt < 5):
        return f"extreme tilt ({tilt:.0f} dB)"
    if (r.get("rms_spread") or 0) == 0 and (r.get("attacks_per_s") or 0) == 0:
        return "no dynamics/transients (synthetic or silence)"
    return None


def ood_z(X: np.ndarray, names: list[str], meta: dict) -> tuple[float, str | None]:
    """How far the input falls outside the training distribution (max |z|)."""
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
    """Target for the tracks of an album: the common input is source feature rows.

    Safeguards (found by validation: on dissimilar material ridge drifted to +33 dB):
      * OOD gate: if a feature falls outside the training distribution (|z| > 6),
        the model is not used — the class table is used instead;
      * a bracket around the class target of +/-2.5 dB: even the model's "own"
        prediction is not allowed to order an overshoot;
      * a physical cap of delta-crest <= 8 dB.
    Returns the per-track delta-crest or None (then the auto policy applies).
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
            # material outside the training distribution (e.g. mono, different tilt)
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
    """Like predict_from_audio, but returns (target, decision source)."""
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
    """Delta-crest target for one track from its audio (for restore_one)."""
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
