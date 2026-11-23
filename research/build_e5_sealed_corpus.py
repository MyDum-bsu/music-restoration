#!/usr/bin/env python3
"""Build the post-freeze powered E5 sealed stereo evaluation manifest (`corpus/sealed-e5-v1.json`).

    python3 research/build_e5_sealed_corpus.py --root corpus

Creates 40 completely fresh, post-freeze independent stereo recording clusters:
  - 30 independent stereo defect clusters (matching the E4 power-registered
    `pilot.sample_size = 30`) spanning 44.1/48/96 kHz, 16/24-bit, dense
    moderate/heavy symmetric clipping, mild sub-full-scale clipping, asymmetric
    positive/negative clipping, rare percussive transient clipping, 16-bit
    TPDF-dithered clipping, single-channel L/R clipping, quiet-tail clipping,
    and late-onset clipping (`defect_gain = 1.0`, no artificial gain step).
  - 10 independent stereo negative controls (clean tonal/percussive/quiet-tail,
    intentional tanh saturation, soft limiter).

Validates strict `corpus/v1` compliance (including the E4-registered `pilot`
block and custodian metadata) and zero cross-manifest group leakage against
`smoke-v1.json`, `dev-v1.json`, `dev-e3-v1.json`, and `sealed-v1.json`.
Never writes audio bytes into tracked paths; regenerates audio into a temporary
directory (or `--audio-root`) strictly to compute deterministic SHA-256 pins.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research import corpus_synth, corpus_v1

RIGHTS_SEALED_E5 = {
    "local_inference": True,
    "colab": False,
    "training": False,
    "listening": True,
    "publication": False,
}

DEFAULT_FREEZE_MANIFEST = REPO / "docs" / "validation" / "e4-freeze-manifest.json"


def _groups(idx: int) -> dict[str, str]:
    return {
        "take": f"t-e5-sealed-{idx:02d}",
        "fingerprint": f"fp-e5-sealed-{idx:02d}",
        "session": f"ses-e5-sealed-{idx:02d}",
        "artist": f"e5-sealed-artist-{idx:02d}",
        "album": f"e5-sealed-album-{idx:02d}",
        "stem_family": f"fam-e5-sealed-{idx:02d}",
    }


def _case(
    cid: str,
    rate: int,
    seconds: int,
    seed: int,
    op: str,
    split: str,
    profile: str,
    bits: int,
    idx: int,
    **kw,
) -> dict:
    return {
        "id": cid,
        "kind": "paired_synthetic",
        "sealed": True,
        "profile": profile,
        "rate": rate,
        "channels": 2,
        "mode": "auto",
        "split": split,
        "rights": dict(RIGHTS_SEALED_E5),
        "groups": _groups(idx),
        "clean": {"path": f"{cid}.clean.wav"},
        "degraded": {"path": f"{cid}.degraded.wav"},
        "synthesis": {
            "generator": corpus_synth.GENERATOR_V2,
            "seed": seed,
            "rate": rate,
            "seconds": seconds,
            "operator": op,
            "bits": bits,
            **kw,
        },
    }


def e5_sealed_cases() -> list[dict]:
    """30 independent stereo defect clusters + 10 independent negative controls."""
    specs = [
        # --- 30 independent defect clusters (fresh seeds 50001..50030) ---
        # Dense symmetric moderate (5 clusters: 44.1/48/96 kHz, 16/24-bit)
        ("e5s-01-dense-mod-44k-16b", 44100, 3, 50001, "hard_clip", "eval",
         "dense_symmetric_moderate", 16, {"threshold_dbfs": -6.0}),
        ("e5s-02-dense-mod-48k-24b", 48000, 3, 50002, "hard_clip", "eval",
         "dense_symmetric_moderate", 24, {"threshold_dbfs": -5.5}),
        ("e5s-03-dense-mod-96k-24b", 96000, 3, 50003, "hard_clip", "eval",
         "dense_symmetric_moderate", 24, {"threshold_dbfs": -6.0}),
        ("e5s-04-dense-mod-44k-24b", 44100, 3, 50004, "hard_clip", "eval",
         "dense_symmetric_moderate", 24, {"threshold_dbfs": -6.5}),
        ("e5s-05-dense-mod-48k-16b", 48000, 3, 50005, "hard_clip", "eval",
         "dense_symmetric_moderate", 16, {"threshold_dbfs": -5.5}),
        # Dense symmetric heavy (4 clusters)
        ("e5s-06-dense-hvy-44k-16b", 44100, 3, 50006, "hard_clip", "eval",
         "dense_symmetric_heavy", 16, {"threshold_dbfs": -8.5}),
        ("e5s-07-dense-hvy-48k-24b", 48000, 3, 50007, "hard_clip", "eval",
         "dense_symmetric_heavy", 24, {"threshold_dbfs": -9.0}),
        ("e5s-08-dense-hvy-96k-24b", 96000, 2, 50008, "hard_clip", "eval",
         "dense_symmetric_heavy", 24, {"threshold_dbfs": -8.0}),
        ("e5s-09-dense-hvy-44k-24b", 44100, 3, 50009, "hard_clip", "eval",
         "dense_symmetric_heavy", 24, {"threshold_dbfs": -8.5}),
        # Mild sub-full-scale (4 clusters)
        ("e5s-10-mild-subfs-44k-24b", 44100, 3, 50010, "hard_clip", "eval",
         "mild_sub_full_scale", 24, {"threshold_dbfs": -3.5}),
        ("e5s-11-mild-subfs-48k-16b", 48000, 3, 50011, "hard_clip", "eval",
         "mild_sub_full_scale", 16, {"threshold_dbfs": -3.0}),
        ("e5s-12-mild-subfs-96k-24b", 96000, 3, 50012, "hard_clip", "eval",
         "mild_sub_full_scale", 24, {"threshold_dbfs": -3.5}),
        ("e5s-13-mild-subfs-48k-24b", 48000, 3, 50013, "hard_clip", "eval",
         "mild_sub_full_scale", 24, {"threshold_dbfs": -4.0}),
        # Asymmetric positive & negative sub-full-scale (4 clusters)
        ("e5s-14-asym-pos-44k-16b", 44100, 3, 50014, "hard_clip", "eval",
         "asymmetric_sub_full_scale", 16, {"threshold_dbfs": -4.5, "pos_scale": 0.65}),
        ("e5s-15-asym-neg-48k-24b", 48000, 3, 50015, "hard_clip", "eval",
         "asymmetric_sub_full_scale", 24, {"threshold_dbfs": -4.5, "neg_scale": 0.62}),
        ("e5s-16-asym-pos-96k-24b", 96000, 3, 50016, "hard_clip", "eval",
         "asymmetric_sub_full_scale", 24, {"threshold_dbfs": -4.0, "pos_scale": 0.70}),
        ("e5s-17-asym-neg-44k-24b", 44100, 3, 50017, "hard_clip", "eval",
         "asymmetric_sub_full_scale", 24, {"threshold_dbfs": -5.0, "neg_scale": 0.68}),
        # Rare percussive transient clipping (3 clusters)
        ("e5s-18-rare-trans-44k-24b", 44100, 3, 50018, "hard_clip", "eval",
         "rare_transient_clip", 24, {"threshold_dbfs": -2.5, "percussive": True}),
        ("e5s-19-rare-trans-48k-24b", 48000, 3, 50019, "hard_clip", "eval",
         "rare_transient_clip", 24, {"threshold_dbfs": -2.5, "percussive": True}),
        ("e5s-20-rare-trans-48k-16b", 48000, 3, 50020, "hard_clip", "eval",
         "rare_transient_clip", 16, {"threshold_dbfs": -3.0, "percussive": True}),
        # 16-bit TPDF-dithered clipping (3 clusters)
        ("e5s-21-dither-44k-16b", 44100, 3, 50021, "hard_clip", "eval",
         "dithered_clip", 16, {"threshold_dbfs": -5.5, "dither": True}),
        ("e5s-22-dither-48k-16b", 48000, 3, 50022, "hard_clip", "eval",
         "dithered_clip", 16, {"threshold_dbfs": -5.0, "dither": True}),
        ("e5s-23-dither-96k-16b", 96000, 3, 50023, "hard_clip", "eval",
         "dithered_clip_96k", 16, {"threshold_dbfs": -5.5, "dither": True}),
        # Single-channel stereo clipping (3 clusters: L-only & R-only)
        ("e5s-24-single-L-44k-24b", 44100, 3, 50024, "hard_clip", "eval",
         "single_channel_stereo", 24, {"threshold_dbfs": -5.5, "channel_mode": "left_only"}),
        ("e5s-25-single-R-48k-16b", 48000, 3, 50025, "hard_clip", "eval",
         "single_channel_stereo", 16, {"threshold_dbfs": -5.5, "channel_mode": "right_only"}),
        ("e5s-26-single-L-96k-24b", 96000, 3, 50026, "hard_clip", "eval",
         "single_channel_stereo", 24, {"threshold_dbfs": -5.5, "channel_mode": "left_only"}),
        # Quiet-tail stereo clipping (2 clusters)
        ("e5s-27-quiet-tail-44k-24b", 44100, 3, 50027, "hard_clip", "eval",
         "quiet_tail", 24, {"threshold_dbfs": -5.0, "quiet_tail": True}),
        ("e5s-28-quiet-tail-96k-24b", 96000, 3, 50028, "hard_clip", "eval",
         "quiet_tail_96k", 24, {"threshold_dbfs": -5.0, "quiet_tail": True}),
        # Late-onset clipping without unrecorded local gain step (`defect_gain=1.0`) (2 clusters)
        ("e5s-29-late-onset-48k-24b", 48000, 6, 50029, "late_clip", "eval",
         "late_onset_clip", 24, {"threshold_dbfs": -5.0, "defect_start_s": 4, "defect_gain": 1.0}),
        ("e5s-30-late-90s-44k-16b", 44100, 92, 50030, "late_clip", "eval",
         "defect_after_90s", 16, {"threshold_dbfs": -5.5, "defect_start_s": 90, "defect_gain": 1.0}),
        # --- 10 independent negative controls (fresh seeds 50031..50040) ---
        ("e5s-31-ctl-clean-44k-24b", 44100, 3, 50031, "clean", "control",
         "clean_control", 24, {}),
        ("e5s-32-ctl-clean-48k-16b", 48000, 3, 50032, "clean", "control",
         "clean_control", 16, {}),
        ("e5s-33-ctl-perc-48k-24b", 48000, 3, 50033, "clean", "control",
         "clean_percussive_control", 24, {"percussive": True}),
        ("e5s-34-ctl-tail-96k-24b", 96000, 3, 50034, "clean", "control",
         "clean_quiet_tail_control", 24, {"quiet_tail": True}),
        ("e5s-35-ctl-sat-44k-16b", 44100, 3, 50035, "intentional_saturation", "control",
         "intentional_control", 16, {"drive": 3.5}),
        ("e5s-36-ctl-sat-48k-24b", 48000, 3, 50036, "intentional_saturation", "control",
         "intentional_control", 24, {"drive": 4.8}),
        ("e5s-37-ctl-sat-44k-24b", 44100, 3, 50037, "intentional_saturation", "control",
         "intentional_control", 24, {"drive": 5.8}),
        ("e5s-38-ctl-softlim-44k-16b", 44100, 3, 50038, "soft_limiter", "control",
         "soft_limiter_control", 16, {"knee": 0.72}),
        ("e5s-39-ctl-softlim-48k-24b", 48000, 3, 50039, "soft_limiter", "control",
         "soft_limiter_control", 24, {"knee": 0.76}),
        ("e5s-40-ctl-softlim-96k-24b", 96000, 3, 50040, "soft_limiter", "control",
         "soft_limiter_control", 24, {"knee": 0.80}),
    ]
    return [
        _case(cid, rate, sec, seed, op, split, prof, bits, idx, **kw)
        for idx, (cid, rate, sec, seed, op, split, prof, bits, kw) in enumerate(specs, 1)
    ]


def build_e5_sealed_manifest(
    root: Path,
    *,
    freeze_manifest_path: Path = DEFAULT_FREEZE_MANIFEST,
    audio_dir: Path | None = None,
    check_existing_manifests: bool = True,
) -> dict:
    """Build and validate `corpus/sealed-e5-v1.json` pinned to E4 pilot registration."""
    freeze = json.loads(freeze_manifest_path.read_text())
    pilot = dict(freeze["e5_preregistration"]["pilot_registration"])
    pilot_errs = corpus_v1.pilot_errors(pilot)
    if pilot_errs:
        raise ValueError("E4 freeze pilot registration invalid: " + "; ".join(pilot_errs))

    cases = e5_sealed_cases()
    stub = {"tier": "sealed", "dataset_version": "sealed-e5-2026-11-23", "cases": cases}
    if check_existing_manifests:
        others = []
        for name in ("smoke-v1.json", "dev-v1.json", "dev-e3-v1.json", "sealed-v1.json"):
            path = root / name
            if path.exists():
                others.append(corpus_v1.load(path))
        leaks = corpus_v1.validate_cross_manifest_leakage([stub, *others])
        if leaks:
            raise ValueError("cross-manifest group leakage: " + "; ".join(leaks))

    def _populate(target_audio: Path) -> dict:
        target_audio.mkdir(parents=True, exist_ok=True)
        files, seen = [], set()
        for case in cases:
            specs = corpus_synth.synthesize_case(case, target_audio)
            for role in ("clean", "degraded", "mask"):
                case.setdefault(role, {}).update(specs[role])
                if specs[role]["path"] not in seen:
                    seen.add(specs[role]["path"])
                    files.append(specs[role])

        doc = {
            "schema": corpus_v1.SCHEMA,
            "tier": "sealed",
            "dataset_version": "sealed-e5-2026-11-23",
            "toolchain": corpus_v1.actual_toolchain(),
            "rng": {"algorithm": "numpy-pcg64", "seed": 2026112305},
            "numeric_tolerance": 1e-6,
            "output_ceiling_dbfs": -1.0,
            "custodian": {
                "name": "sealed custodian (appointed by owner)",
                "appointed_at": "2026-11-23",
                "release": "owner_authorized_single_unseal_e5_2026-11-23",
                "e4_freeze_sha256": freeze["freeze_sha256"],
            },
            "pilot": pilot,
            "files": files,
            "cases": cases,
        }
        errors = corpus_v1.validate_manifest(doc, check_hashes_root=target_audio)
        if errors:
            raise ValueError("manifest validation failed: " + "; ".join(errors))
        return doc

    if audio_dir is not None:
        return _populate(audio_dir)
    with tempfile.TemporaryDirectory(prefix="mr_e5_build_") as tmp:
        return _populate(Path(tmp))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--root", default="corpus")
    ap.add_argument("--freeze-manifest", default=str(DEFAULT_FREEZE_MANIFEST))
    ap.add_argument("--audio-root", default=None,
                    help="optional directory to keep regenerated audio; defaults to temp dir")
    args = ap.parse_args(argv)

    root = Path(args.root)
    audio_dir = Path(args.audio_root) if args.audio_root else None
    doc = build_e5_sealed_manifest(
        root,
        freeze_manifest_path=Path(args.freeze_manifest),
        audio_dir=audio_dir,
    )
    target = root / "sealed-e5-v1.json"
    target.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(
        f"{target}: {len(doc['cases'])} cases "
        f"(defects={sum(1 for c in doc['cases'] if c['split'] == 'eval')}, "
        f"controls={sum(1 for c in doc['cases'] if c['split'] == 'control')}), "
        f"sha256 {corpus_v1.manifest_digest(doc)[:16]}...",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
