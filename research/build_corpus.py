#!/usr/bin/env python3
"""Build the corpus/v1 manifests and their deterministic audio.

    python3 research/build_corpus.py --root corpus

Writes corpus/{smoke,dev,sealed}-v1.json plus regenerated audio under
corpus/audio/<tier>/ (gitignored — bytes are reproducible from the pinned
synthesis blocks; the manifest sha256 is what is distributed).
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research import corpus_synth, corpus_v1

TOOLCHAIN = {"python": ">=3.10", "ffmpeg": "pinned-by-environment",
             "numpy": ">=1.24", "scipy": ">=1.10", "bits_wav": 16}
RIGHTS_PUBLIC = {"local_inference": True, "colab": True, "training": True,
                 "listening": True, "publication": True}
RIGHTS_DEV = {"local_inference": True, "colab": True, "training": True,
              "listening": True, "publication": False}
RIGHTS_SEALED = {"local_inference": False, "colab": False, "training": False,
                 "listening": False, "publication": False}


def synth_case(cid, rate, seconds, seed, operator, profile, rights,
               split=None, groups=None, **kw):
    case = {"id": cid, "kind": "paired_synthetic", "profile": profile,
            "rate": rate, "channels": 2, "mode": "auto",
            "rights": rights, "groups": groups or {},
            "clean": {"path": f"{cid}.clean.wav"},
            "degraded": {"path": f"{cid}.degraded.wav"},
            "synthesis": {"generator": corpus_synth.GENERATOR, "seed": seed,
                          "rate": rate, "seconds": seconds,
                          "operator": operator, "bits": 16, **kw}}
    if split:
        case["split"] = split
    return case


def g(**kw):
    base = {"take": None, "fingerprint": None, "session": None,
            "artist": "synth", "album": "synth-corpus", "stem_family": None}
    base.update(kw)
    return base


def build_synth_tier(tier, version, cases, root, pilot=None, custodian=None):
    audio = root / "audio" / tier
    audio.mkdir(parents=True, exist_ok=True)
    files, seen = [], {}
    for case in cases:
        specs = corpus_synth.synthesize_case(case, audio)
        for role in ("clean", "degraded", "mask"):
            case.setdefault(role, {}).update(specs[role])
            if specs[role]["path"] not in seen:
                seen[specs[role]["path"]] = True
                files.append(specs[role])
    doc = {"schema": corpus_v1.SCHEMA, "tier": tier, "dataset_version": version,
           "toolchain": TOOLCHAIN, "rng": {"algorithm": "numpy-pcg64", "seed": 20261123},
           "numeric_tolerance": 1e-6, "output_ceiling_dbfs": -1.0,
           "files": files, "cases": cases}
    if pilot is not None:
        doc["pilot"] = pilot
    if custodian is not None:
        doc["custodian"] = custodian
    return doc


def smoke_cases():
    return [
        synth_case("smoke-clip-rare-asym-44k", 44100, 4, 101, "hard_clip",
                   "rare_asymmetric_sub_full_scale", RIGHTS_PUBLIC,
                   threshold_dbfs=-1.0, pos_scale=.7,
                   groups=g(take="t-smoke-1", fingerprint="fp-smoke-1")),
        synth_case("smoke-clip-dense-sub-48k", 48000, 4, 102, "hard_clip",
                   "dense_sub_full_scale", RIGHTS_PUBLIC,
                   threshold_dbfs=-6.0,
                   groups=g(take="t-smoke-2", fingerprint="fp-smoke-2")),
        synth_case("smoke-clip-dither-96k", 96000, 3, 103, "hard_clip",
                   "dithered_clip", RIGHTS_PUBLIC, threshold_dbfs=-2.0, dither=True,
                   groups=g(take="t-smoke-3", fingerprint="fp-smoke-3")),
        synth_case("smoke-clip-quiet-tail-44k", 44100, 5, 104, "hard_clip",
                   "quiet_tail", RIGHTS_PUBLIC, threshold_dbfs=-2.5, quiet_tail=True,
                   groups=g(take="t-smoke-4", fingerprint="fp-smoke-4")),
        synth_case("smoke-clip-late-90s-44k", 44100, 92, 105, "late_clip",
                   "defect_after_90s", RIGHTS_PUBLIC,
                   threshold_dbfs=-2.0, defect_start_s=90,
                   groups=g(take="t-smoke-5", fingerprint="fp-smoke-5")),
        synth_case("smoke-clean-control-48k", 48000, 4, 106, "clean",
                   "clean_control", RIGHTS_PUBLIC,
                   groups=g(take="t-smoke-6", fingerprint="fp-smoke-6")),
        synth_case("smoke-intentional-sat-44k", 44100, 4, 107,
                   "intentional_saturation", "intentional_control", RIGHTS_PUBLIC,
                   drive=3.0, groups=g(take="t-smoke-7", fingerprint="fp-smoke-7")),
        synth_case("smoke-soft-limiter-48k", 48000, 4, 108, "soft_limiter",
                   "soft_limiter_control", RIGHTS_PUBLIC, knee=.8,
                   groups=g(take="t-smoke-8", fingerprint="fp-smoke-8")),
    ]


def dev_cases():
    specs = [
        ("dev-clip-train-a-44k", 44100, 6, 201, "hard_clip", "train",
         "dense_sub_full_scale", dict(threshold_dbfs=-4.0)),
        ("dev-clip-train-b-48k", 48000, 6, 202, "hard_clip", "train",
         "rare_asymmetric_sub_full_scale",
         dict(threshold_dbfs=-1.5, pos_scale=.75)),
        ("dev-clip-eval-a-44k", 44100, 6, 203, "hard_clip", "eval",
         "dithered_clip", dict(threshold_dbfs=-3.0, dither=True)),
        ("dev-clip-eval-b-96k", 96000, 4, 204, "hard_clip", "eval",
         "quiet_tail", dict(threshold_dbfs=-2.0, quiet_tail=True)),
        ("dev-clean-control-48k", 48000, 5, 205, "clean", "control",
         "clean_control", {}),
        ("dev-intentional-control-44k", 44100, 5, 206, "intentional_saturation",
         "control", "intentional_control", dict(drive=4.0)),
    ]
    cases = []
    for i, (cid, rate, sec, seed, op, split, profile, kw) in enumerate(specs):
        cases.append(synth_case(cid, rate, sec, seed, op, profile, RIGHTS_DEV,
                                split=split, **kw,
                                groups=g(take=f"t-dev-{i}", fingerprint=f"fp-dev-{i}",
                                         session=f"ses-dev-{i}",
                                         artist=f"dev-artist-{i}",
                                         album=f"dev-album-{i // 2}")))
    return cases


def sealed_doc():
    """Custodian-held sealed set: descriptors only, no audio bytes.

    Pilot values left null on purpose: validation must block the sealed tier
    until the E2 pilot registers sample size, gate margins, coverage and CPU
    p95/RSS. Custodian-appointed but not released.
    """
    held = lambda rel: {"path": rel, "sha256": None, "bytes": None,
                        "held_by": "custodian"}
    cases = []
    for i, cid in enumerate(("sealed-real-01", "sealed-real-02")):
        cases.append({"id": cid, "kind": "paired_file", "sealed": True,
                      "profile": "real_capture", "rate": 44100, "channels": 2,
                      "mode": "auto", "split": "eval", "rights": RIGHTS_SEALED,
                      "groups": g(take=f"t-sealed-{i}", fingerprint=f"fp-sealed-{i}",
                                  session=f"ses-sealed-{i}", artist=f"sealed-artist-{i}",
                                  album=f"sealed-album-{i}",
                                  stem_family=f"fam-sealed-{i}"),
                      "clean": held(f"sealed/{cid}.clean.flac"),
                      "degraded": held(f"sealed/{cid}.degraded.flac")})
    return {"schema": corpus_v1.SCHEMA, "tier": "sealed",
            "dataset_version": "sealed-2026-11-23",
            "toolchain": TOOLCHAIN, "rng": {"algorithm": "numpy-pcg64", "seed": None},
            "numeric_tolerance": 1e-6, "output_ceiling_dbfs": -1.0,
            "custodian": {"name": "sealed custodian (appointed by owner)",
                          "appointed_at": "2026-11-23",
                          "release": "withheld until E4 freeze"},
            "pilot": {"sample_size": None, "gate_margin_sdr_db": None,
                      "gate_margin_vs_external_db": None, "min_coverage": None,
                      "cpu_p95_s_per_audio_min": None, "rss_mb_p95": None,
                      "registered_at": None},
            "files": [held(f"sealed/{c['id']}.{r}.flac")
                      for c in cases for r in ("clean", "degraded")],
            "cases": cases}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="corpus")
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    docs = [build_synth_tier("smoke", "smoke-2026-11-23", smoke_cases(), root),
            build_synth_tier("dev", "dev-2026-11-23", dev_cases(), root,
                             pilot={"sample_size": None, "gate_margin_sdr_db": None,
                                    "gate_margin_vs_external_db": None,
                                    "min_coverage": None,
                                    "cpu_p95_s_per_audio_min": None,
                                    "rss_mb_p95": None, "registered_at": None}),
            sealed_doc()]
    for doc in docs:
        target = root / f"{doc['tier']}-v1.json"
        target.write_text(json.dumps(doc, indent=2) + "\n")
        errors = corpus_v1.validate_manifest(doc)
        print(f"{target}: {len(doc['cases'])} cases, "
              f"sha256 {corpus_v1.manifest_digest(doc)[:16]}…, "
              f"validation errors: {len(errors)}")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
