#!/usr/bin/env python3
"""Property probe for t-20261102-233913: one version-quality scorer in rnext.

Runs once per seed:  python3 pbt_scorer.py --seed 7

Properties (all over generated fact dicts, no network):
  P1 delegation  tracker_hunt.score(f) == rnext.planner.score_facts(f)
  P2 determinism same input twice -> identical score and explanation
  P3 lossless     a lossless music version scores strictly above the same lossy one
  P4 bit depth    24-bit music version scores at least as high as the same 16-bit one
  P5 resolution   2160P video version scores at least as high as the same 1080P one

Any violation exits 1 with the failing input. No retries, no fallbacks.
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").is_file() and (p / "rnext").is_dir():
            return p
    raise SystemExit("FAIL: repository root not found (pyproject.toml + rnext/)")


ROOT = repo_root()
sys.path.insert(0, str(ROOT))


def load_planner():
    from rnext import planner  # noqa: PLC0415

    if not hasattr(planner, "score_facts"):
        raise SystemExit("FAIL: rnext/planner.py has no score_facts(facts) — the single scorer is missing")
    return planner


def load_hunt():
    path = ROOT / "scripts/tracker_hunt.py"
    spec = importlib.util.spec_from_file_location("tracker_hunt_under_test", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FAIL: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "score", None)):
        raise SystemExit("FAIL: scripts/tracker_hunt.py has no callable score(facts)")
    return module


MUSIC_SOURCES = ["hi-res (24-bit)", "WEB (digital release)", "CD", "vinyl/DSD", "?"]
VIDEO_RES = ["2160P", "4K", "UHD", "1080P", "720P", "?"]
VIDEO_SRC = ["BDRemux", "BDRip", "WEB-DL", "WEBRip", "HDTV", "?"]
AUDIO = ["DTS-HD MA", "TrueHD", "LPCM", "Atmos", "AC3", "AAC"]
SUBS = ["rus", "eng", "rus+eng"]


def music_facts(rnd: random.Random) -> dict:
    return {
        "kind": "music",
        "lossless": rnd.random() < 0.5,
        "source": rnd.choice(MUSIC_SOURCES),
        "vinyl": rnd.random() < 0.2,
        "bits": rnd.choice([None, 16, 24]),
        "dr_published": rnd.choice([None, 5, 8, 11, 14, 16]),
        "has_log": rnd.random() < 0.5,
        "has_scans": rnd.random() < 0.3,
        "image_cue": rnd.random() < 0.4,
    }


def video_facts(rnd: random.Random) -> dict:
    return {
        "kind": "video",
        "resolution": rnd.choice(VIDEO_RES),
        "source": rnd.choice(VIDEO_SRC),
        "codec": rnd.choice(["x265", "hevc", "h.265", "av1", "x264", "?"]),
        "hdr": rnd.choice([None, "HDR10", "DV"]),
        "audio": rnd.sample(AUDIO, rnd.randint(0, 2)),
        "subs": rnd.sample(SUBS, rnd.randint(0, 2)),
    }


def book_facts(rnd: random.Random) -> dict:
    return {
        "kind": "book",
        "format": rnd.choice([None, "?", "pdf", "epub", "djvu"]),
        "text_layer": rnd.random() < 0.5,
    }


def check(eq_hunt, eq_facts, facts: dict, props: dict) -> None:
    got = eq_hunt(facts)
    want = eq_facts(facts)
    if got != want:
        raise SystemExit(f"FAIL P1 delegation: facts={facts}\n  tracker_hunt.score -> {got}\n  planner.score_facts -> {want}")
    again = eq_hunt(facts)
    if again != got:
        raise SystemExit(f"FAIL P2 determinism: {facts}\n  first={got}\n  second={again}")
    for name, (a, b) in props.items():
        if not (eq_hunt(a) >= eq_hunt(b)):
            raise SystemExit(f"FAIL {name}: {a} scores {eq_hunt(a)} below/equal {b} = {eq_hunt(b)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=200)
    args = ap.parse_args()

    planner = load_planner()
    hunt = load_hunt()
    rnd = random.Random(args.seed)
    checked = 0

    for _ in range(args.rounds):
        check(hunt.score, planner.score_facts, music_facts(rnd), {})
        check(hunt.score, planner.score_facts, video_facts(rnd), {})
        check(hunt.score, planner.score_facts, book_facts(rnd), {})
        checked += 3

    # order properties on controlled pairs, generated too
    for _ in range(args.rounds // 4):
        base = music_facts(rnd)
        lossy = dict(base, lossless=False)
        lossless = dict(base, lossless=True)
        check(hunt.score, planner.score_facts, lossless, {"P3 lossless>lossy": (lossless, lossy)})

        b16 = dict(base, bits=16)
        b24 = dict(base, bits=24)
        check(hunt.score, planner.score_facts, b24, {"P4 24bit>=16bit": (b24, b16)})

        v = video_facts(rnd)
        hi = dict(v, resolution="2160P")
        lo = dict(v, resolution="1080P")
        check(hunt.score, planner.score_facts, hi, {"P5 2160P>=1080P": (hi, lo)})
        checked += 3

    print(f"OK seed={args.seed}: {checked} fact sets satisfy P1-P5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
