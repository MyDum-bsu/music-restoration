"""Пайплайн реставрации: HPF → [деклип] → [EQ] → контур → TP-лимит.

Политика — простой dataclass, сериализуемый в JSON (в corpus/policy.json).
"""
import json
import math
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import signal

from rnext.dsp import contour, declip, eq as dsp_eq, limit


@dataclass
class Policy:
    # HPF
    hpf_hz: float = 25.0
    # деклип
    declip: bool = True
    # контур
    k: float = 0.75
    bass_scale: float = 0.35
    tau_attack_s: float = 0.25
    tau_release_s: float = 1.8
    gain_min_db: float = -9.0
    gain_max_db: float = 12.0
    bass_hz: float = 160.0
    ramp_to_hz: float = 900.0
    frame: int = 8192
    # eq (опционально: список (hz, dam-ass)) — калибруется, по умолчанию off
    eq_bands_db: list = field(default_factory=list)
    # лимит/вывод
    target_tp_dbfs: float = -0.3
    # контура «прозрачно»: true для healthy-альбомов (не трогать)
    contour_enabled: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(txt: str) -> "Policy":
        d = json.loads(txt)
        return Policy(**d)


def hpf25(x: np.ndarray, sr: int, hz: float = 25.0) -> np.ndarray:
    sos = signal.butter(2, hz, "hp", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0).astype(x.dtype)


def restore_track(x: np.ndarray, sr: int, pol: Policy | None = None,
                  return_stages: bool = False):
    """x [n, 2] float32 [-1,1) → реставрированный [n, 2].

    stages: dict c промежуточными массивами (для дип-приладки/диффа).
    """
    pol = pol or Policy()
    stages = {}
    y = hpf25(x, sr, pol.hpf_hz)
    stages["hpf"] = y.copy()

    need_declip = None
    if pol.declip:
        x16 = np.clip(x[:, 0] * 32767.5, -32768, 32767).astype(np.int16)
        n_clip = declip.clip_positions(np.stack([x16[:, None]] * 2, axis=1) if False else np.column_stack([x16] * 2))
        # аккуратно: ради детекции клипов считаем по L/R одинаково (мастер-выход)
        x16L = np.clip(x[:, 0] * 32767.5, -32768, 32767).astype(np.int16)
        x16R = np.clip(x[:, 1] * 32767.5, -32768, 32767).astype(np.int16)
        xx = np.column_stack([x16L, x16R])
        if declip.clip_positions(xx) > 0:
            from rnext.features import decode_s16  # noqa: F401 (plugin-not-needed)
            y16, n = declip.declip_track(xx)
            y = (y16.astype(np.float64) / 32768.0).astype(np.float32)
            need_declip = n
        stages["declip"] = (y.copy(), need_declip)

    if pol.eq_bands_db:
        y = dsp_eq.apply_eq(y, sr, pol.eq_bands_db, frame=pol.frame)
        stages["eq"] = y.copy()

    if pol.contour_enabled:
        cp = contour.ContourPolicy(k=pol.k, bass_scale=pol.bass_scale,
                                   tau_attack_s=pol.tau_attack_s,
                                   tau_release_s=pol.tau_release_s,
                                   frame=pol.frame,
                                   gain_min_db=pol.gain_min_db,
                                   gain_max_db=pol.gain_max_db,
                                   bass_hz=pol.bass_hz,
                                   ramp_to_hz=pol.ramp_to_hz)
        y = contour.contour_expand(y, sr, cp)
        stages["contour"] = y.copy()

    y = limit.limit_track(y, sr, ceiling_db=pol.target_tp_dbfs)
    stages["limit"] = y.copy()
    return (y, stages) if return_stages else y
