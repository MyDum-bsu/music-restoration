#!/usr/bin/env python3
"""Build the E3 extended stereo dev bake-off manifest (`corpus/dev-e3-v1.json`).

    python3 research/build_e3_corpus.py --root corpus

Creates 18 independent stereo cases (12 defect slices + 6 negative controls)
spanning symmetric/asymmetric, sub-full-scale, rare, dithered, single-channel,
quiet-tail, late-onset (>90s), and native 44.1/48/96 kHz 16/24-bit profiles.
Validates strict corpus/v1 compliance and zero cross-manifest group leakage
against smoke-v1, dev-v1, and sealed-v1 before writing any audio bytes.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research import corpus_synth, corpus_v1

RIGHTS_DEV = {"local_inference": True, "colab": True, "training": True,
              "listening": True, "publication": False}


def _groups(idx: int) -> dict:
    return {
        "take": f"t-e3-{idx:02d}",
        "fingerprint": f"fp-e3-{idx:02d}",
        "session": f"ses-e3-{idx:02d}",
        "artist": f"e3-artist-{idx:02d}",
        "album": f"e3-album-{idx:02d}",
        "stem_family": f"fam-e3-{idx:02d}",
    }


def _case(cid, rate, seconds, seed, op, split, profile, bits, idx, **kw):
    return {
        "id": cid,
        "kind": "paired_synthetic",
        "profile": profile,
        "rate": rate,
        "channels": 2,
        "mode": "auto",
        "split": split,
        "rights": dict(RIGHTS_DEV),
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


def e3_dev_cases() -> list[dict]:
    """12 independent stereo defect recordings + 6 negative controls."""
    specs = [
        # Defect slices (12 independent clusters)
        ("e3-dense-sym-44k-16b", 44100, 4, 301, "hard_clip", "eval",
         "dense_symmetric_moderate", 16, {"threshold_dbfs": -6.0}),
        ("e3-dense-heavy-48k-24b", 48000, 4, 302, "hard_clip", "eval",
         "dense_symmetric_heavy", 24, {"threshold_dbfs": -9.0}),
        ("e3-mild-subfs-44k-24b", 44100, 4, 303, "hard_clip", "eval",
         "mild_sub_full_scale", 24, {"threshold_dbfs": -3.5}),
        ("e3-asym-pos-48k-16b", 48000, 4, 304, "hard_clip", "eval",
         "asymmetric_sub_full_scale", 16, {"threshold_dbfs": -4.0, "pos_scale": 0.65}),
        ("e3-asym-neg-44k-24b", 44100, 4, 305, "hard_clip", "eval",
         "asymmetric_sub_full_scale", 24, {"threshold_dbfs": -4.5, "neg_scale": 0.60}),
        ("e3-rare-trans-48k-24b", 48000, 4, 306, "hard_clip", "eval",
         "rare_transient_clip", 24, {"threshold_dbfs": -2.0, "percussive": True}),
        ("e3-dither-44k-16b", 44100, 4, 307, "hard_clip", "eval",
         "dithered_clip", 16, {"threshold_dbfs": -5.5, "dither": True}),
        ("e3-dither-96k-16b", 96000, 3, 308, "hard_clip", "eval",
         "dithered_clip_96k", 16, {"threshold_dbfs": -5.0, "dither": True}),
        ("e3-single-ch-L-48k-24b", 48000, 4, 309, "hard_clip", "eval",
         "single_channel_stereo", 24, {"threshold_dbfs": -5.5, "channel_mode": "left_only"}),
        ("e3-single-ch-R-44k-16b", 44100, 4, 310, "hard_clip", "eval",
         "single_channel_stereo", 16, {"threshold_dbfs": -5.5, "channel_mode": "right_only"}),
        ("e3-quiet-tail-96k-24b", 96000, 3, 311, "hard_clip", "eval",
         "quiet_tail_96k", 24, {"threshold_dbfs": -5.0, "quiet_tail": True}),
        ("e3-late-defect-44k-16b", 44100, 92, 312, "late_clip", "eval",
         "defect_after_90s", 16, {"threshold_dbfs": -3.0, "defect_start_s": 90, "defect_gain": 2.5}),
        # Negative controls (6 independent clusters; zero accepted repairs required)
        ("e3-ctl-clean-tonal-44k-24b", 44100, 4, 321, "clean", "control",
         "clean_control", 24, {}),
        ("e3-ctl-clean-perc-48k-16b", 48000, 4, 322, "clean", "control",
         "clean_percussive_control", 16, {"percussive": True}),
        ("e3-ctl-clean-tail-96k-24b", 96000, 3, 323, "clean", "control",
         "clean_quiet_tail_control", 24, {"quiet_tail": True}),
        ("e3-ctl-intent-sat-44k-16b", 44100, 4, 324, "intentional_saturation", "control",
         "intentional_control", 16, {"drive": 4.0}),
        ("e3-ctl-intent-sat-48k-24b", 48000, 4, 325, "intentional_saturation", "control",
         "intentional_control", 24, {"drive": 5.5}),
        ("e3-ctl-soft-lim-48k-24b", 48000, 4, 326, "soft_limiter", "control",
         "soft_limiter_control", 24, {"knee": 0.72}),
    ]
    return [_case(cid, rate, sec, seed, op, split, prof, bits, idx, **kw)
            for idx, (cid, rate, sec, seed, op, split, prof, bits, kw) in enumerate(specs, 1)]


def build_e3_manifest(root: Path, *, check_existing_manifests: bool = True) -> dict:
    cases = e3_dev_cases()
    stub = {"tier": "dev", "dataset_version": "dev-e3-2026-11-23", "cases": cases}
    if check_existing_manifests:
        others = []
        for name in ("smoke-v1.json", "dev-v1.json", "sealed-v1.json"):
            path = root / name
            if path.exists():
                others.append(corpus_v1.load(path))
        leaks = corpus_v1.validate_cross_manifest_leakage([stub, *others])
        if leaks:
            raise ValueError("cross-manifest group leakage: " + "; ".join(leaks))

    audio = root / "audio" / "dev-e3"
    audio.mkdir(parents=True, exist_ok=True)
    files, seen = [], set()
    for case in cases:
        specs = corpus_synth.synthesize_case(case, audio)
        for role in ("clean", "degraded", "mask"):
            case.setdefault(role, {}).update(specs[role])
            if specs[role]["path"] not in seen:
                seen.add(specs[role]["path"])
                files.append(specs[role])

    doc = {
        "schema": corpus_v1.SCHEMA,
        "tier": "dev",
        "dataset_version": "dev-e3-2026-11-23",
        "toolchain": corpus_v1.actual_toolchain(),
        "rng": {"algorithm": "numpy-pcg64", "seed": 20261123},
        "numeric_tolerance": 1e-6,
        "output_ceiling_dbfs": -1.0,
        "pilot": {
            "sample_size": None,
            "gate_margin_sdr_db": None,
            "gate_margin_vs_external_db": None,
            "min_coverage": None,
            "cpu_p95_s_per_audio_min": None,
            "rss_mb_p95": None,
            "registered_at": None,
        },
        "files": files,
        "cases": cases,
    }
    errors = corpus_v1.validate_manifest(doc, check_hashes_root=audio)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--root", default="corpus")
    args = ap.parse_args()
    root = Path(args.root)
    doc = build_e3_manifest(root)
    target = root / "dev-e3-v1.json"
    target.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(f"{target}: {len(doc['cases'])} cases, sha256 {corpus_v1.manifest_digest(doc)[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
