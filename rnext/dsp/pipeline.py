"""Пайплайн реставрации: HPF → [деклип] → [EQ] → контур → TP-лимит.

Политика — простой dataclass, сериализуемый в JSON (в corpus/policy.json).
"""
import json
import math
from dataclasses import dataclass, asdict, field, replace as dc_replace

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
    # режим контура
    mode: str = "flatten"          # "flatten" | "expand"
    r_expand: float = 1.5          # только mode="expand"
    # ── авто-политика по диагнозу (класс материала → обработка) ─────────────
    auto_policy: bool = True
    healthy_dr_min: int = 11
    sparse_rms_below: float = -13.0
    sparse_target_dcrest: float = 0.8
    target_dcrest: float = 4.6
    adaptive: bool = False
    # eq (опционально: список (hz, dam-ass)) — калибруется, по умолчанию off
    eq_bands_db: list = field(default_factory=list)
    # ── замкнутый EQ-этап: подгонка спектра под целевую кривую корпуса ────────
    # Открытый EQ (eq_bands_db) применяет фиксированную кривую и не знает, что
    # уже сделал контур. Замкнутый вариант измеряет фактическую дельту полос
    # относительно оригинала и добирает разницу с целью — то есть компенсирует
    # и собственный бас-дефицит экспандера, и любой разброс по треку.
    eq_match: bool = False
    eq_match_curve: list = field(default_factory=list)   # [(hz, db)] цель
    eq_match_cap_db: float = 2.0          # предел коррекции выше 100 Гц
    eq_match_cap_low_db: float = 4.0      # предел ниже 100 Гц (там ФВЧ срезает низ)
    eq_match_hpf_hz: float = 18.0         # ФВЧ для режима подгонки (25 Гц режет 22–35 Гц)
    eq_match_iters: int = 3
    normalize_tp: bool = False            # дотянуть TP до потолка (после EQ пик падает)
    # лимит/вывод
    target_tp_dbfs: float = -0.5
    # контура «прозрачно»: true для healthy-альбомов (не трогать)
    contour_enabled: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(txt: str) -> "Policy":
        d = json.loads(txt)
        return Policy(**d)


def hpf25(x: np.ndarray, sr: int, hz: float = 25.0) -> np.ndarray:
    """ФВЧ 2-го порядка, zero-phase. padlen подобран по периоду среза: у 25 Гц
    импульсный отклик ~6 мс, и дефолтный padlen=15 сэмплов даёт слышимый
    артефакт на краях (ошибка по RMS ~9 %)."""
    sos = signal.butter(2, hz, "hp", fs=sr, output="sos")
    padlen = int(min(max(3 * sr / hz, 64), x.shape[0] - 1))
    # padtype="even": при "odd" длинный паддинг создаёт DC-полку и фильтр
    # «звонит» на краях (ошибка до 0.5 от амплитуды на первых сэмплах)
    return signal.sosfiltfilt(sos, x, axis=0, padlen=padlen,
                              padtype="even").astype(x.dtype)


def diagnose(x: np.ndarray, sr: int, pol: "Policy") -> dict:
    """Класс материала → план обработки (см. README «Диагноз и авто-политика»).

    Все пороги — в tt-конвенции: crest_tt = peak_dbfs − rms_db (features.tt_dr).
    Классы (проверяются по порядку):
      healthy: orig TT DR ≥ pol.healthy_dr_min (11)
               → HPF/контур/EQ выключены: мастер такой материал не трогает;
      sparse : orig TT RMS ≤ pol.sparse_rms_below
               → цель Δcrest = pol.sparse_target_dcrest;
      loud   : иначе → adaptive-контур с целью pol.target_dcrest.
    """
    from rnext import features as F
    st = F.tt_dr(x, sr)
    dr, drx, _ch, peak, rms = st
    crest = peak - rms
    if not pol.auto_policy:
        return {"class": "manual", "dr": dr, "crest_tt": crest, "rms_tt": rms,
                "target_dcrest": pol.target_dcrest,
                "hpf_on": True, "contour_on": True, "eq_on": True,
                "reason": "auto_policy выключена — ручные флаги"}
    if dr >= pol.healthy_dr_min:
        return {"class": "healthy", "dr": dr, "crest_tt": crest, "rms_tt": rms,
                "target_dcrest": pol.target_dcrest,
                "hpf_on": False, "contour_on": False, "eq_on": False,
                "reason": f"orig TT DR {dr} ≥ {pol.healthy_dr_min}: прозрачно"}
    if rms <= pol.sparse_rms_below:
        return {"class": "sparse", "dr": dr, "crest_tt": crest, "rms_tt": rms,
                "target_dcrest": pol.sparse_target_dcrest,
                "hpf_on": True, "contour_on": True, "eq_on": True,
                "reason": f"orig TT RMS {rms:.1f} ≤ {pol.sparse_rms_below}: "
                          f"цель {pol.sparse_target_dcrest}"}
    return {"class": "loud", "dr": dr, "crest_tt": crest, "rms_tt": rms,
            "target_dcrest": pol.target_dcrest,
            "hpf_on": True, "contour_on": True, "eq_on": True,
            "reason": f"плотный loud: цель {pol.target_dcrest}"}



def tail_stage(y: np.ndarray, sr: int, pol: "Policy", x_ref: np.ndarray) -> tuple:
    """Хвост пайплайна: замкнутая EQ-подгонка → TP-нормализация → лимитер.

    Вынесено отдельно, чтобы АДАПТИВНЫЙ ФИТТЕР мерил цель по ФИНАЛЬНОМУ выходу:
    EQ меняет крест (+~1 дБ), и если целиться до него, итог перелетает.
    """
    info = None
    if pol.eq_match and pol.eq_match_curve and getattr(pol, "eq_on_ref", True):
        from rnext import features as _F
        from rnext.dsp import eq as _eq
        hz = [b[0] for b in pol.eq_match_curve]
        tgt = [b[1] for b in pol.eq_match_curve]
        mt = sum(tgt) / len(tgt)
        tgt = [x - mt for x in tgt]
        bands_in = _F.band_rms_db(x_ref, sr)
        hist = []
        for it in range(max(1, pol.eq_match_iters)):
            y = limit.limit_track(y, sr, ceiling_db=pol.target_tp_dbfs)
            bands_out = _F.band_rms_db(y, sr)
            d = [bands_out[i] - bands_in[i] for i in range(len(tgt))]
            m = sum(d) / len(d)
            d = [v - m for v in d]
            corr = []
            for i in range(len(tgt)):
                cap = pol.eq_match_cap_low_db if hz[i] < 100 else pol.eq_match_cap_db
                corr.append(max(-cap, min(cap, tgt[i] - d[i])))
            hist.append({"iter": it, "measured": [round(v, 2) for v in d],
                         "correction": [round(v, 2) for v in corr],
                         "max_abs": round(max(abs(v) for v in corr), 2)})
            if max(abs(v) for v in corr) < 0.05:
                break
            y = _eq.apply_eq(y, sr, list(zip(hz, corr)), frame=pol.frame)
        info = {"target": [round(v, 2) for v in tgt], "history": hist,
                "cap_db": pol.eq_match_cap_db, "cap_low_db": pol.eq_match_cap_low_db,
                "hpf_hz": pol.eq_match_hpf_hz, "iters": len(hist)}
    y = limit.limit_track(y, sr, ceiling_db=pol.target_tp_dbfs)
    if pol.normalize_tp:
        from rnext.dsp.contour import true_peak_dbfs as _tp
        cur = _tp(y, sr)
        if cur < pol.target_tp_dbfs - 0.01:
            y = (y * (10 ** ((pol.target_tp_dbfs - cur) / 20.0))).astype(y.dtype)
            if info is not None:
                info["tp_normalize"] = {"from": round(cur, 2), "to": pol.target_tp_dbfs}
    return y, info


def restore_track(x: np.ndarray, sr: int, pol: Policy | None = None,
                  return_stages: bool = False):
    """x [n, 2] float32 [-1,1) → реставрированный [n, 2].

    stages: dict c промежуточными массивами (для дип-приладки/диффа).
    """
    pol = pol or Policy()
    stages = {}

    # ── диагноз: класс материала → политика (auto), либо ручные флаги ──
    dx = diagnose(x, sr, pol)
    stages["diagnosis"] = dx
    contour_on = pol.contour_enabled and dx["contour_on"]
    eq_on = bool(pol.eq_bands_db) and dx["eq_on"]

    if dx.get("hpf_on", True):
        hz = pol.eq_match_hpf_hz if pol.eq_match else pol.hpf_hz
        y = hpf25(x, sr, hz)
    else:
        y = x.copy()
    stages["hpf"] = y.copy()

    need_declip = None
    if pol.declip:
        x16 = np.column_stack([
            np.clip(x[:, 0] * 32767.5, -32768, 32767).astype(np.int16),
            np.clip(x[:, 1] * 32767.5, -32768, 32767).astype(np.int16)])
        if declip.clip_positions(x16) > 0:
            y16, n = declip.declip_track(x16)
            y = (y16.astype(np.float64) / 32768.0).astype(np.float32)
            need_declip = n
        stages["declip"] = (y.copy(), need_declip)

    if eq_on:
        y = dsp_eq.apply_eq(y, sr, pol.eq_bands_db, frame=pol.frame)
        stages["eq"] = y.copy()

    if contour_on:
        r_eff = pol.r_expand
        if pol.adaptive and pol.mode == "expand":
            from rnext.dsp.adaptive import fit_r
            crest_ref = dx["crest_tt"]            # Δ меряем от сырого исходника
            eff_target = dx["target_dcrest"]
            base_cp = contour.ContourPolicy(mode=pol.mode, bass_scale=pol.bass_scale,
                                            bass_hz=pol.bass_hz, ramp_to_hz=pol.ramp_to_hz)
            # в пробах хвост считается в 1 итерацию EQ (крест от EQ важен, точность — нет)
            pol_probe = dc_replace(pol, eq_match_iters=1)
            r_eff, predicted = fit_r(
                y, sr, eff_target, base=base_cp,
                post=lambda z, s: tail_stage(z, s, pol_probe, x)[0],
                crest_ref=crest_ref)
            stages["adaptive"] = {"r_fit": r_eff, "predicted_dcrest": predicted}
        cp = contour.ContourPolicy(k=pol.k, bass_scale=pol.bass_scale,
                                   tau_attack_s=pol.tau_attack_s,
                                   tau_release_s=pol.tau_release_s,
                                   frame=pol.frame,
                                   gain_min_db=pol.gain_min_db,
                                   gain_max_db=pol.gain_max_db,
                                   bass_hz=pol.bass_hz,
                                   ramp_to_hz=pol.ramp_to_hz,
                                   mode=pol.mode, r_expand=r_eff)
        y = contour.contour_expand(y, sr, cp)
        stages["contour"] = y.copy()

    # ── хвост: EQ-подгонка → TP-нормализация → лимитер ─────────────────────
    y, eq_info = tail_stage(y, sr, pol, x)
    if eq_info:
        stages["eq_match"] = eq_info
    stages["limit"] = y.copy()

    return (y, stages) if return_stages else y
