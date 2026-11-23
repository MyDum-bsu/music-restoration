#!/usr/bin/env python3
"""E5 Blind Listening Panel Builder (LUFS-matched FLAC) and Statistical Scorer.

Build a ready-to-listen loudness-matched double-blind A/B + BS.1116/MUSHRA panel:

    python3 research/e5_blind_panel.py build \
        --sealed-manifest corpus/sealed-e5-v1.json \
        --out /path/outside/git/music-restoration-e5-blind-panel \
        --seed 2026112399

Score completed listener responses (either JSON or filled-in `listening_sheet.md`):

    python3 research/e5_blind_panel.py score \
        --panel-dir /path/outside/git/music-restoration-e5-blind-panel

Design invariants (spec v1 Section 4 & E4 preregistration):
  - Equal achievable integrated loudness verified `<= 0.2 LU` after FLAC encode
    via `scripts.listen_pack.common_lufs` / `gain_to_lufs` (shared across REF, A, B).
  - Strictly blind filenames (`item_01_A.flac`, `item_01_B.flac`, `item_01_REF.flac`)
    and blind `listening_sheet.md` (zero method names, zero randomization seed).
  - Balanced AB/BA presentation order across comparisons plus hidden repeat items
    for intra-rater reliability checks.
  - Custodian-held `answer_key_DO_NOT_OPEN_BEFORE_LISTENING.json` kept separate.
  - Statistical scoring: exact two-sided binomial sign test + Wilson 95% CI on A/B
    preferences, paired Wilcoxon signed-rank test on quality scores, recording-cluster
    95% bootstrap CI (`declip_metrics.paired_cluster_ci`), listener-cluster summary,
    hidden-repeat consistency, and formal E5 perceptual gate verdict.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import corpus_synth, corpus_v1
from research import declip_metrics as M
from research.e5_sealed_eval import (
    CANDIDATE_ARM,
    DEFAULT_FREEZE_MANIFEST,
    EXPECTED_FREEZE_SHA256,
    FreezeIntegrityError,
    verify_freeze_integrity,
)
from research.run_benchmark import _decode
from rnext import features as F
from rnext.dsp.clip_detector import detect_hard_clip_v2
from rnext.dsp.contour import true_peak_dbfs
from rnext.dsp.declip_methods import (
    apply_headroom_policy,
    declip_aspade,
    declip_ss_pew,
)
from rnext.dsp.repair import select_restoration
from scripts.listen_pack import common_lufs, gain_to_lufs
from scripts.restore_one import encode_flac

COMPARISON_ARMS = (
    ("identity", "Identity (unprocessed input)"),
    ("current_snapshot_auto", "Current production baseline"),
    ("ss_pew_mit__v2_windowed", "External DSP comparator (SS-PEW)"),
)


def _write_temp_flac(path: Path, pcm: np.ndarray, sr: int, *, bits: int = 24, seed: int = 100) -> None:
    rng = np.random.default_rng(seed)
    dith = rng.random(pcm.shape) - rng.random(pcm.shape)
    scale = float(1 << (bits - 1))
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    q = np.clip(np.round(np.clip(pcm, -1.0, 1.0) * scale + dith), lo, hi).astype(np.int32)
    tmp_wav = path.with_suffix(".tmp.wav")
    try:
        encode_flac(q, sr, path, tmp_wav, bits=bits)
    finally:
        tmp_wav.unlink(missing_ok=True)


def _render_arm(arm: str, degraded: np.ndarray, sr: int, bits: int, ceiling_db: float = -1.0) -> np.ndarray:
    x32 = degraded.astype(np.float32)
    if arm == "identity":
        return x32.copy()
    if arm == "current_snapshot_auto":
        out, _rep = select_restoration(degraded, sr, mode="auto", bits=bits)
        return out.astype(np.float32)
    mask, _diag = detect_hard_clip_v2(x32, sr, bits=bits)
    if not mask.any():
        return x32.copy()
    if arm == CANDIDATE_ARM:
        raw, _info = declip_aspade(x32, sr, mask)
    elif arm == "ss_pew_mit__v2_windowed":
        raw, _info = declip_ss_pew(x32, sr, mask)
    else:
        raise ValueError(f"unknown blind-panel arm: {arm}")
    delivered, _head = apply_headroom_policy(raw, x32, sr, mask, ceiling_db=ceiling_db)
    return delivered.astype(np.float32)


def _select_loudest_window(pcm: np.ndarray, sr: int, seconds: float) -> tuple[int, int]:
    """Find the loudest stereo excerpt window of length `seconds` (1 s hop)."""
    n = pcm.shape[0]
    win = min(n, max(sr, round(seconds * sr)))
    if n <= win:
        return 0, n
    step = max(1, sr)
    best_s, best_e = 0, -1.0
    for s in range(0, n - win + 1, step):
        e = float(np.mean(pcm[s:s + win] ** 2))
        if e > best_e:
            best_e = e
            best_s = s
    return best_s, best_s + win


def _collect_real_music_sources(real_music_dir: Path, max_tracks: int = 6, excerpt_s: float = 8.0) -> list[dict]:
    """Load distinct-artist stereo excerpts from a local music folder (local-only, never committed)."""
    sources = []
    if not real_music_dir.is_dir():
        return sources
    seen_artists = set()
    # Reserve the last real-music slot as an unclipped negative control.
    clip_thresholds = (-5.5, -6.5, -5.0, -7.0, -4.5)
    for album_dir in sorted(real_music_dir.iterdir()):
        if not album_dir.is_dir():
            continue
        artist = album_dir.name.split(" - ")[0].strip().lower()
        if artist in seen_artists:
            continue
        flacs = sorted(album_dir.glob("*.flac"))
        if not flacs:
            continue
        # Pick track 1 or 2 with >= 40 s duration.
        chosen = None
        for f in flacs[:3]:
            try:
                info = F.ffprobe_audio(str(f))
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"  [skip unreadable] {f.name}: {type(exc).__name__}", file=sys.stderr)
                continue
            if info.get("ch") == 2 and info.get("sr") in (44100, 48000, 96000) and (info.get("dur") or 0) >= 30.0:
                chosen = (f, info)
                break
        if chosen is None:
            continue
        f, info = chosen
        sr = int(info["sr"])
        bits = int(info["bps"] or 16)
        start_probe = min(35.0, max(5.0, (info["dur"] - excerpt_s) * 0.35))
        seg, sr_dec = F.decode_f32(str(f), start_s=start_probe, dur_s=min(24.0, info["dur"] - start_probe),
                                   strict=True, channels=None)
        if sr_dec != sr or seg.ndim != 2 or seg.shape[1] != 2:
            continue
        w0, w1 = _select_loudest_window(seg, sr, excerpt_s)
        clean = seg[w0:w1].astype(np.float64)
        # Ensure clean excerpt is peak-normalized to -0.5 dBFS before applying controlled hard clip.
        pk = float(np.max(np.abs(clean)))
        if pk < 1e-4:
            continue
        clean = clean * (0.94 / pk)

        idx = len(sources)
        seen_artists.add(artist)
        if idx < min(len(clip_thresholds), max_tracks - 1):
            thr = clip_thresholds[idx]
            deg, mask, _ = corpus_synth.apply_operator(
                clean, sr, {"operator": "hard_clip", "threshold_dbfs": thr, "bits": bits},
                seed=71000 + idx,
            )
            if not mask.any():
                continue
            sources.append({
                "source_id": f"real-clip-{idx + 1:02d}",
                "cluster_id": f"real-artist-{idx + 1:02d}",
                "stratum": "real_music_controlled_hard_clip",
                "profile": f"real_stereo_hard_clip_{thr:g}dBFS",
                "is_defect": True,
                "sr": sr,
                "bits": bits,
                "clean": clean,
                "degraded": deg,
            })
        else:
            sources.append({
                "source_id": f"real-ctrl-{idx + 1:02d}",
                "cluster_id": f"real-artist-{idx + 1:02d}",
                "stratum": "real_music_clean_control",
                "profile": "real_stereo_clean_control",
                "is_defect": False,
                "sr": sr,
                "bits": bits,
                "clean": clean,
                "degraded": clean.copy(),
            })
        if len(sources) >= max_tracks:
            break
    return sources


def _collect_sealed_sources(manifest_path: Path, max_defects: int = 4, max_controls: int = 1) -> list[dict]:
    """Select diverse representative cases from a sealed corpus/v1 manifest."""
    doc = corpus_v1.load(manifest_path)
    defects = [c for c in doc["cases"] if c.get("split") != "control"]
    controls = [c for c in doc["cases"] if c.get("split") == "control"]

    # Pick spread-out profiles deterministically (moderate, heavy, asymmetric, dithered/96k).
    picked = []
    seen_profiles = set()
    for c in defects:
        prof = c.get("profile")
        if prof not in seen_profiles and (c.get("synthesis") or {}).get("seconds", 0) <= 10:
            picked.append(c)
            seen_profiles.add(prof)
        if len(picked) >= max_defects:
            break
    picked.extend(controls[:max_controls])

    sources = []
    with tempfile.TemporaryDirectory(prefix="mr_e5_panel_src_") as tmp:
        tmp_p = Path(tmp)
        for c in picked:
            if c.get("kind") == "paired_synthetic":
                corpus_synth.synthesize_case(c, tmp_p)
            deg_p = corpus_v1.resolve(doc, tmp_p, c["degraded"], allow_custodian=True)
            cln_p = corpus_v1.resolve(doc, tmp_p, c["clean"], allow_custodian=True)
            deg, sr, info = _decode(deg_p)
            cln, _, _ = _decode(cln_p)
            sources.append({
                "source_id": c["id"],
                "cluster_id": (c.get("groups") or {}).get("take") or c["id"],
                "stratum": "sealed_synthetic_holdout",
                "profile": c.get("profile", "unknown"),
                "is_defect": c.get("split") != "control",
                "sr": sr,
                "bits": int(info["bps"] or 16),
                "clean": cln,
                "degraded": deg,
            })
    return sources


def build_blind_panel(
    sources: list[dict],
    out_dir: Path,
    *,
    seed: int = 2026112399,
    target_lufs: float = -18.0,
    comparators: tuple[str, ...] = ("identity", "current_snapshot_auto", "ss_pew_mit__v2_windowed"),
    n_hidden_repeats: int = 2,
    freeze_sha256: str = EXPECTED_FREEZE_SHA256,
) -> dict:
    """Build loudness-matched blind A/B + REF FLAC pack, listening sheet, and custodian answer key."""
    if not sources:
        raise ValueError("no audio sources provided for blind panel")

    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Render all required arm waveforms per source once.
    rendered: dict[str, dict[str, np.ndarray]] = {}
    for src in sources:
        sid = src["source_id"]
        arms_needed = {CANDIDATE_ARM, *comparators}
        rendered[sid] = {"clean_ref": src["clean"].astype(np.float32)}
        for arm in sorted(arms_needed):
            rendered[sid][arm] = _render_arm(arm, src["degraded"], src["sr"], src["bits"])

    # Construct balanced comparison trials across defect sources, plus control trials vs identity.
    defect_srcs = [s for s in sources if s["is_defect"]]
    control_srcs = [s for s in sources if not s["is_defect"]]

    trial_specs: list[dict] = []
    # Balance candidate_is_A deterministically across each comparator.
    for comp in comparators:
        flip_flags = [i % 2 == 0 for i in range(len(defect_srcs))]
        rng.shuffle(flip_flags)
        for src, cand_is_a in zip(defect_srcs, flip_flags):
            trial_specs.append({
                "source": src,
                "comparator": comp,
                "candidate_is_a": cand_is_a,
                "repeat_of_index": None,
            })
    for idx, src in enumerate(control_srcs):
        trial_specs.append({
            "source": src,
            "comparator": "identity",
            "candidate_is_a": (idx % 2 == 0),
            "repeat_of_index": None,
        })

    # Add hidden repeats (flipped AB/BA order!) from distinct defect trials for intra-rater reliability.
    base_defect_indices = [i for i, t in enumerate(trial_specs) if t["source"]["is_defect"]]
    rng.shuffle(base_defect_indices)
    for orig_idx in base_defect_indices[:min(n_hidden_repeats, len(base_defect_indices))]:
        base = trial_specs[orig_idx]
        trial_specs.append({
            "source": base["source"],
            "comparator": base["comparator"],
            "candidate_is_a": not base["candidate_is_a"],
            "repeat_of_index": orig_idx,
        })

    # Shuffle final presentation order so listener cannot guess blocks or repeats.
    order = list(range(len(trial_specs)))
    rng.shuffle(order)
    orig_idx_to_item_id: dict[int, str] = {}
    for pos, spec_idx in enumerate(order, start=1):
        orig_idx_to_item_id[spec_idx] = f"item_{pos:02d}"

    key_items = []
    with tempfile.TemporaryDirectory(prefix="mr_e5_lufs_") as tmp:
        tmp_p = Path(tmp)
        for pos, spec_idx in enumerate(order, start=1):
            item_id = f"item_{pos:02d}"
            spec = trial_specs[spec_idx]
            src = spec["source"]
            sid = src["source_id"]
            comp = spec["comparator"]
            cand_is_a = spec["candidate_is_a"]
            arm_a = CANDIDATE_ARM if cand_is_a else comp
            arm_b = comp if cand_is_a else CANDIDATE_ARM

            # Write pre-normalized FLACs for REF, A, B so `common_lufs` + `gain_to_lufs`
            # enforce one exact achievable loudness across all 3 files (<= 0.2 LU, <= -1.0 dBTP).
            raw_ref = tmp_p / f"{item_id}_raw_ref.flac"
            raw_a = tmp_p / f"{item_id}_raw_a.flac"
            raw_b = tmp_p / f"{item_id}_raw_b.flac"
            _write_temp_flac(raw_ref, rendered[sid]["clean_ref"], src["sr"], bits=src["bits"], seed=pos * 10 + 1)
            _write_temp_flac(raw_a, rendered[sid][arm_a], src["sr"], bits=src["bits"], seed=pos * 10 + 2)
            _write_temp_flac(raw_b, rendered[sid][arm_b], src["sr"], bits=src["bits"], seed=pos * 10 + 3)

            matched_lufs = common_lufs({"REF": raw_ref, "A": raw_a, "B": raw_b}, target_lufs)
            dst_ref = audio_dir / f"{item_id}_REF.flac"
            dst_a = audio_dir / f"{item_id}_A.flac"
            dst_b = audio_dir / f"{item_id}_B.flac"
            norm_ref = gain_to_lufs(raw_ref, dst_ref, matched_lufs)
            norm_a = gain_to_lufs(raw_a, dst_a, matched_lufs)
            norm_b = gain_to_lufs(raw_b, dst_b, matched_lufs)

            # Verify <= 0.2 LU difference between A and B and true peak <= -1.0 + 0.05 dBTP.
            lufs_diff = abs(float(norm_a["measured_lufs"]) - float(norm_b["measured_lufs"]))
            if lufs_diff > 0.2:
                raise ValueError(f"{item_id}: post-encode LUFS mismatch {lufs_diff:.3f} LU > 0.2 LU")
            tp_a = float(true_peak_dbfs(*F.decode_f32(str(dst_a), strict=True, channels=None)))
            tp_b = float(true_peak_dbfs(*F.decode_f32(str(dst_b), strict=True, channels=None)))
            if max(tp_a, tp_b) > -1.0 + 0.05:
                raise ValueError(f"{item_id}: true peak exceeded -1.0 dBTP ({tp_a:.2f}, {tp_b:.2f})")

            repeat_of = (
                orig_idx_to_item_id[spec["repeat_of_index"]]
                if spec["repeat_of_index"] is not None
                else None
            )
            key_items.append({
                "item_id": item_id,
                "source_id": sid,
                "cluster_id": src["cluster_id"],
                "stratum": src["stratum"],
                "profile": src["profile"],
                "is_defect": src["is_defect"],
                "rate": src["sr"],
                "bits": src["bits"],
                "duration_s": round(float(len(src["degraded"]) / src["sr"]), 3),
                "comparison": f"{CANDIDATE_ARM}_vs_{comp}",
                "comparator": comp,
                "candidate_label": "A" if cand_is_a else "B",
                "comparator_label": "B" if cand_is_a else "A",
                "labels": {"A": arm_a, "B": arm_b, "REF": "clean_reference"},
                "repeat_of": repeat_of,
                "matched_lufs": round(float(matched_lufs), 3),
                "lufs_diff_ab": round(lufs_diff, 4),
                "files": {
                    "REF": {"path": f"audio/{dst_ref.name}", "sha256": corpus_v1.digest(dst_ref), **norm_ref},
                    "A": {"path": f"audio/{dst_a.name}", "sha256": corpus_v1.digest(dst_a), "tp_dbfs": round(tp_a, 3), **norm_a},
                    "B": {"path": f"audio/{dst_b.name}", "sha256": corpus_v1.digest(dst_b), "tp_dbfs": round(tp_b, 3), **norm_b},
                },
            })

    answer_key = {
        "schema": "e5-blind-panel-key/v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "e4_freeze_sha256": freeze_sha256,
        "candidate_arm": CANDIDATE_ARM,
        "comparators": list(comparators),
        "randomization_seed": seed,
        "target_lufs": target_lufs,
        "n_items": len(key_items),
        "n_hidden_repeats": sum(1 for it in key_items if it["repeat_of"] is not None),
        "items": key_items,
    }
    key_path = out_dir / "answer_key_DO_NOT_OPEN_BEFORE_LISTENING.json"
    key_path.write_text(json.dumps(answer_key, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # Blind listening sheet (ZERO method names, ZERO seeds, ZERO spoilers, ZERO host paths).
    sheet_lines = [
        "# E5 Blind Listening Panel — Listening Sheet (A/B + BS.1116 / MUSHRA)",
        "",
        "> **ВАЖНО / IMPORTANT:** Не открывайте файл `answer_key_DO_NOT_OPEN_BEFORE_LISTENING.json` до завершения прослушивания!",
        "> Все пары `A` и `B` (и референс `REF`) строго выровнены по интегральной громкости (`<= 0.2 LU`, потолок `<= -1.0 dBTP`).",
        "",
        "## Инструкция / Instructions",
        "",
        "1. Слушайте в наушниках или на мониторах на комфортной постоянной громкости (не крутите ручку громкости между `REF`, `A` и `B`).",
        "2. Для каждой строки `item_XX` прослушайте `audio/item_XX_REF.flac` (чистый ориентир), `audio/item_XX_A.flac` и `audio/item_XX_B.flac`.",
        "3. Заполните столбцы в таблице ниже (либо заполните `responses_template.json` и сохраните как `responses.json`):",
        "   - **Preference:** `A` (лучше A), `B` (лучше B), или `tie` (неотличимы / паритет).",
        "   - **Score A / Score B (MUSHRA 0–100):** `80–100` Отлично (неотличимо от REF), `60–79` Хорошо (лёгкие отличия), `40–59` Удовлетворительно, `20–39` Плохо, `0–19` Очень плохо (явный клиппинг/артефакты).",
        "   - **Artifacts (optional):** например `none`, `clip_distortion`, `pre_echo`, `clicks`, `stereo_shift`, `dullness`.",
        "4. После заполнения запустите автоматический подсчёт статистики:",
        "   ```bash",
        "   python3 research/e5_blind_panel.py score --panel-dir <path-to-this-folder>",
        "   ```",
        "",
        "## Таблица оценок / Rating Table",
        "",
        "| Item | Audio Files | Dur (s) | Rate / Bits | Preference (`A`/`B`/`tie`) | Score A (0-100) | Score B (0-100) | Artifacts / Notes |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for it in key_items:
        rate_str = f"{it['rate']//1000 if it['rate']%1000==0 else it['rate']/1000:g}k/{it['bits']}b"
        sheet_lines.append(
            f"| `{it['item_id']}` | `audio/{it['item_id']}_REF.flac`, `_A.flac`, `_B.flac` | "
            f"{it['duration_s']:.1f} | {rate_str} |  |  |  |  |"
        )
    sheet_lines.append("")
    (out_dir / "listening_sheet.md").write_text("\n".join(sheet_lines), encoding="utf-8")

    # Machine-readable JSON response template (zero spoilers!).
    template = {
        "schema": "e5-blind-panel-responses/v1",
        "listener_id": "owner_01",
        "playback_chain": "headphones / monitors (fill in)",
        "responses": [
            {
                "item_id": it["item_id"],
                "preference": None,
                "score_a": None,
                "score_b": None,
                "artifacts_a": "",
                "artifacts_b": "",
                "notes": "",
            }
            for it in key_items
        ],
    }
    (out_dir / "responses_template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return answer_key


def _wilson_ci(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = wins / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def parse_listening_sheet_md(sheet_path: Path, listener_id: str = "sheet_listener_01") -> dict:
    """Parse a filled-in `listening_sheet.md` rating table into responses JSON format."""
    responses = []
    row_re = re.compile(r"^\|\s*`?(item_\d+)`?\s*\|")
    for line in sheet_path.read_text(encoding="utf-8").splitlines():
        if not row_re.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        item_id = cells[0].strip("`").strip()
        pref = cells[4].strip().lower()
        pref_val = "A" if pref == "a" else ("B" if pref == "b" else ("tie" if pref in ("tie", "=") else None))

        def _num(s: str) -> float | None:
            s = s.strip()
            if not s:
                return None
            return float(s)

        responses.append({
            "item_id": item_id,
            "preference": pref_val,
            "score_a": _num(cells[5]),
            "score_b": _num(cells[6]),
            "artifacts_a": "",
            "artifacts_b": "",
            "notes": cells[7],
        })
    return {
        "schema": "e5-blind-panel-responses/v1",
        "listener_id": listener_id,
        "responses": responses,
    }


def score_blind_panel(answer_key: dict, listener_docs: list[dict]) -> dict:
    """Unblind and compute exact binomial, Wilcoxon, cluster CI, and repeat reliability statistics."""
    key_by_id = {it["item_id"]: it for it in answer_key["items"]}
    comp_names = sorted({it["comparator"] for it in answer_key["items"]})

    # Flatten valid responses across all listeners.
    unblinded_rows = []
    listener_ids = []
    for doc in listener_docs:
        lid = str(doc.get("listener_id") or f"listener_{len(listener_ids) + 1:02d}")
        listener_ids.append(lid)
        for r in doc.get("responses", []):
            iid = r.get("item_id")
            if iid not in key_by_id:
                continue
            pref = r.get("preference")
            sa, sb = r.get("score_a"), r.get("score_b")
            if pref is None and (sa is None or sb is None):
                continue
            kit = key_by_id[iid]
            cand_lbl = kit["candidate_label"]
            comp_lbl = kit["comparator_label"]

            if pref is not None:
                p_norm = str(pref).strip().upper()
                if p_norm == cand_lbl:
                    outcome = "candidate"
                elif p_norm == comp_lbl:
                    outcome = "comparator"
                elif p_norm in ("TIE", "="):
                    outcome = "tie"
                else:
                    raise ValueError(f"invalid preference {pref!r} for {iid}")
            else:
                outcome = "candidate" if float(sa) > float(sb) else ("comparator" if float(sb) > float(sa) else "tie")

            score_cand = (float(sa) if cand_lbl == "A" else float(sb)) if (sa is not None and sb is not None) else None
            score_comp = (float(sb) if cand_lbl == "A" else float(sa)) if (sa is not None and sb is not None) else None
            score_diff = (score_cand - score_comp) if (score_cand is not None and score_comp is not None) else None

            unblinded_rows.append({
                "listener_id": lid,
                "item_id": iid,
                "source_id": kit["source_id"],
                "cluster_id": kit["cluster_id"],
                "is_defect": kit["is_defect"],
                "comparator": kit["comparator"],
                "repeat_of": kit["repeat_of"],
                "outcome": outcome,
                "score_candidate": score_cand,
                "score_comparator": score_comp,
                "score_diff": score_diff,
                "notes": r.get("notes", ""),
            })

    # Per-comparator statistical tests (excluding hidden repeats from primary tests so clusters aren't double-counted).
    comparisons = {}
    for comp in comp_names:
        c_rows = [r for r in unblinded_rows if r["comparator"] == comp and r["repeat_of"] is None and r["is_defect"]]
        wins = sum(1 for r in c_rows if r["outcome"] == "candidate")
        losses = sum(1 for r in c_rows if r["outcome"] == "comparator")
        ties = sum(1 for r in c_rows if r["outcome"] == "tie")
        decided = wins + losses
        binom_p = float(stats.binomtest(wins, decided, 0.5, alternative="two-sided").pvalue) if decided > 0 else None
        win_rate = float(wins / decided) if decided > 0 else None
        w_lo, w_hi = _wilson_ci(wins, decided)

        diffs = [r["score_diff"] for r in c_rows if r["score_diff"] is not None]
        if len(diffs) >= 1 and any(abs(d) > 1e-12 for d in diffs):
            w_stat, w_p = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat, wilcoxon_p = float(w_stat), float(w_p)
        elif len(diffs) >= 1:
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
        else:
            wilcoxon_stat, wilcoxon_p = None, None

        # Recording-cluster collapsed paired 95% CI on score_diff.
        rec_buckets: dict[str, list[float]] = {}
        for r in c_rows:
            if r["score_diff"] is not None:
                rec_buckets.setdefault(r["cluster_id"], []).append(r["score_diff"])
        rec_cluster_diffs = {k: float(np.mean(v)) for k, v in rec_buckets.items()}
        rec_ci = M.paired_cluster_ci(rec_cluster_diffs, confidence=0.95, n_boot=10000, seed=20261123)

        # Listener-cluster collapsed mean score_diff.
        lis_buckets: dict[str, list[float]] = {}
        for r in c_rows:
            if r["score_diff"] is not None:
                lis_buckets.setdefault(r["listener_id"], []).append(r["score_diff"])
        lis_cluster_diffs = {k: float(np.mean(v)) for k, v in lis_buckets.items()}
        lis_ci = M.paired_cluster_ci(lis_cluster_diffs, confidence=0.95, n_boot=10000, seed=20261123)

        comparisons[comp] = {
            "n_primary_ratings": len(c_rows),
            "wins_candidate": wins,
            "losses_comparator": losses,
            "ties": ties,
            "decided": decided,
            "preference_rate_ex_ties": win_rate,
            "preference_wilson_95_ci": [w_lo, w_hi],
            "binomial_two_sided_p": binom_p,
            "n_scored_pairs": len(diffs),
            "mean_score_diff": float(np.mean(diffs)) if diffs else None,
            "median_score_diff": float(np.median(diffs)) if diffs else None,
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_two_sided_p": wilcoxon_p,
            "recording_cluster_ci": rec_ci,
            "listener_cluster_ci": lis_ci,
        }

    # Control items audit (no perceptual harm on unclipped controls).
    ctrl_rows = [r for r in unblinded_rows if not r["is_defect"]]
    ctrl_summary = {
        "n_ratings": len(ctrl_rows),
        "ties": sum(1 for r in ctrl_rows if r["outcome"] == "tie"),
        "candidate_preferred": sum(1 for r in ctrl_rows if r["outcome"] == "candidate"),
        "comparator_preferred": sum(1 for r in ctrl_rows if r["outcome"] == "comparator"),
    }

    # Hidden-repeat intra-rater reliability.
    by_lis_item = {(r["listener_id"], r["item_id"]): r for r in unblinded_rows}
    rep_checked = rep_agree = 0
    rep_score_abs_diffs = []
    for r in unblinded_rows:
        if r["repeat_of"] is None:
            continue
        orig = by_lis_item.get((r["listener_id"], r["repeat_of"]))
        if orig is None:
            continue
        rep_checked += 1
        if orig["outcome"] == r["outcome"]:
            rep_agree += 1
        if orig["score_diff"] is not None and r["score_diff"] is not None:
            rep_score_abs_diffs.append(abs(orig["score_diff"] - r["score_diff"]))
    repeat_reliability = {
        "n_repeat_pairs_checked": rep_checked,
        "preference_agreement_count": rep_agree,
        "preference_agreement_rate": float(rep_agree / rep_checked) if rep_checked else None,
        "mean_abs_score_diff_drift": float(np.mean(rep_score_abs_diffs)) if rep_score_abs_diffs else None,
    }

    # E5 perceptual gate verdict.
    harm = False
    for comp, stats_c in comparisons.items():
        if (
            stats_c["binomial_two_sided_p"] is not None
            and stats_c["binomial_two_sided_p"] < 0.05
            and (stats_c["preference_rate_ex_ties"] or 0.5) < 0.5
        ) or (
            stats_c["wilcoxon_two_sided_p"] is not None
            and stats_c["wilcoxon_two_sided_p"] < 0.05
            and (stats_c["mean_score_diff"] or 0.0) < 0.0
        ):
            harm = True

    id_stats = comparisons.get("identity") or {}
    id_confirmed = bool(
        not harm
        and id_stats.get("n_primary_ratings", 0) >= 4
        and (
            (
                id_stats.get("binomial_two_sided_p") is not None
                and id_stats["binomial_two_sided_p"] < 0.05
                and (id_stats.get("preference_rate_ex_ties") or 0.0) > 0.5
            )
            or (
                id_stats.get("wilcoxon_two_sided_p") is not None
                and id_stats["wilcoxon_two_sided_p"] < 0.05
                and (id_stats.get("mean_score_diff") or 0.0) > 0.0
                and (id_stats.get("recording_cluster_ci", {}).get("bootstrap_ci_low") or 0.0) > 0.0
            )
        )
    )
    if harm:
        verdict = "harm_detected"
    elif id_confirmed:
        verdict = "confirmed_preference"
    else:
        verdict = "inconclusive_insufficient_power"

    return {
        "schema": "e5-blind-panel-score/v1",
        "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidate_arm": answer_key["candidate_arm"],
        "n_listeners": len(set(listener_ids)),
        "n_total_responses": len(unblinded_rows),
        "comparisons": comparisons,
        "controls": ctrl_summary,
        "repeat_reliability": repeat_reliability,
        "verdict": verdict,
    }


def render_score_markdown(score_doc: dict) -> str:
    def _f(v, fmt="{:.3f}"):
        return "-" if v is None else fmt.format(v)

    lines = [
        "# E5 Blind Listening Panel — Statistical Score Report",
        "",
        f"- **Candidate:** `{score_doc['candidate_arm']}`",
        f"- **Listeners:** `{score_doc['n_listeners']}` · **Total scored responses:** `{score_doc['n_total_responses']}`",
        f"- **Perceptual Gate Verdict:** `{score_doc['verdict']}`",
        "",
        "| Comparison (`candidate` vs comparator) | Wins / Losses / Ties | Pref Rate (95% Wilson CI) | Binomial p | Mean ΔScore | Wilcoxon p | Rec-Cluster 95% CI |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for comp, s in score_doc["comparisons"].items():
        wlo, whi = s["preference_wilson_95_ci"]
        pr = f"{_f(s['preference_rate_ex_ties'], '{:.1%}')} [{_f(wlo, '{:.2f}')}, {_f(whi, '{:.2f}')}]"
        rc = s["recording_cluster_ci"]
        rci = f"[{_f(rc['bootstrap_ci_low'], '{:+.2f}')}, {_f(rc['bootstrap_ci_high'], '{:+.2f}')}]"
        lines.append(
            f"| vs `{comp}` | {s['wins_candidate']} / {s['losses_comparator']} / {s['ties']} | "
            f"{pr} | {_f(s['binomial_two_sided_p'], '{:.4f}')} | "
            f"{_f(s['mean_score_diff'], '{:+.2f}')} | {_f(s['wilcoxon_two_sided_p'], '{:.4f}')} | {rci} |"
        )
    rep = score_doc["repeat_reliability"]
    lines += [
        "",
        (
            f"- **Hidden-repeat consistency:** `{rep['preference_agreement_count']}/{rep['n_repeat_pairs_checked']}` "
            f"(`{_f(rep['preference_agreement_rate'], '{:.1%}')}`), mean |Δscore drift| `{_f(rep['mean_abs_score_diff_drift'], '{:.2f}')}`."
        ),
        f"- **Negative controls:** `{score_doc['controls']}`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="build loudness-matched blind listening panel")
    pb.add_argument("--sealed-manifest", default=None, help="sealed corpus/v1 manifest for synthetic holdout items")
    pb.add_argument("--real-music-dir", default=None, help="optional local real-music FLAC folder (local-only)")
    pb.add_argument("--max-sealed-defects", type=int, default=4)
    pb.add_argument("--max-sealed-controls", type=int, default=1)
    pb.add_argument("--max-real-tracks", type=int, default=5)
    pb.add_argument("--excerpt-seconds", type=float, default=8.0)
    pb.add_argument("--comparators", default="identity,current_snapshot_auto,ss_pew_mit__v2_windowed")
    pb.add_argument("--hidden-repeats", type=int, default=2)
    pb.add_argument("--target-lufs", type=float, default=-18.0)
    pb.add_argument("--seed", type=int, default=2026112399)
    pb.add_argument("--freeze-manifest", default=str(DEFAULT_FREEZE_MANIFEST))
    pb.add_argument("--expected-freeze-sha256", default=EXPECTED_FREEZE_SHA256)
    pb.add_argument("--out", required=True)

    ps = sub.add_parser("score", help="score completed blind listening panel responses")
    ps.add_argument("--panel-dir", required=True)
    ps.add_argument("--responses", action="append", default=[],
                    help="response JSON or filled listening_sheet.md path (repeatable for multiple listeners)")
    ps.add_argument("--out-json", default=None)
    ps.add_argument("--out-md", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "build":
        try:
            freeze = verify_freeze_integrity(
                Path(args.freeze_manifest),
                expected_freeze_sha256=args.expected_freeze_sha256,
                repo_root=REPO,
                check_toolchain=False,
            )
        except FreezeIntegrityError as exc:
            print(f"freeze integrity error: {exc}", file=sys.stderr)
            return 2

        sources = []
        if args.sealed_manifest:
            sources += _collect_sealed_sources(
                Path(args.sealed_manifest),
                max_defects=args.max_sealed_defects,
                max_controls=args.max_sealed_controls,
            )
        if args.real_music_dir:
            sources += _collect_real_music_sources(
                Path(args.real_music_dir),
                max_tracks=args.max_real_tracks,
                excerpt_s=args.excerpt_seconds,
            )
        if not sources:
            print("error: provide --sealed-manifest and/or --real-music-dir", file=sys.stderr)
            return 2

        comps = tuple(c.strip() for c in args.comparators.split(",") if c.strip())
        key = build_blind_panel(
            sources,
            Path(args.out),
            seed=args.seed,
            target_lufs=args.target_lufs,
            comparators=comps,
            n_hidden_repeats=args.hidden_repeats,
            freeze_sha256=freeze["freeze_sha256"],
        )
        print(f"e5_blind_panel build: {key['n_items']} blind items (repeats={key['n_hidden_repeats']}) ready")
        return 0

    panel_dir = Path(args.panel_dir)
    key_path = panel_dir / "answer_key_DO_NOT_OPEN_BEFORE_LISTENING.json"
    if not key_path.is_file():
        print("error: answer_key_DO_NOT_OPEN_BEFORE_LISTENING.json not found in panel-dir", file=sys.stderr)
        return 2
    answer_key = json.loads(key_path.read_text(encoding="utf-8"))

    resp_paths = [Path(p) for p in args.responses]
    if not resp_paths:
        default_json = panel_dir / "responses.json"
        default_md = panel_dir / "listening_sheet.md"
        if default_json.is_file():
            resp_paths = [default_json]
        elif default_md.is_file():
            resp_paths = [default_md]
        else:
            print("error: no responses.json or listening_sheet.md found", file=sys.stderr)
            return 2

    listener_docs = []
    for idx, rp in enumerate(resp_paths, start=1):
        if rp.suffix.lower() == ".md":
            listener_docs.append(parse_listening_sheet_md(rp, listener_id=f"listener_{idx:02d}"))
        else:
            listener_docs.append(json.loads(rp.read_text(encoding="utf-8")))

    score_doc = score_blind_panel(answer_key, listener_docs)
    report_md = render_score_markdown(score_doc)

    out_json = Path(args.out_json) if args.out_json else (panel_dir / "panel_score_summary.json")
    out_md = Path(args.out_md) if args.out_md else (panel_dir / "panel_score_report.md")
    out_json.write_text(json.dumps(score_doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    out_md.write_text(report_md, encoding="utf-8")
    print(f"e5_blind_panel score: verdict={score_doc['verdict']} responses={score_doc['n_total_responses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
