"""Property probe for the unified release scorer (item 1 of the review).

Not collected by the contract run (no test_ prefix): this file is driven by the
`properties.command` of the spec, once per seed.

Contract fixed by the spec:
    rnext.planner.score_facts(row: dict, ctx: dict) -> dict  # keys: "score", "flags"
    ctx keys: artist_tokens, album_tokens, year, codes, own_poster
    rnext.planner.choose(cands, threshold=..., kind=None) -> dict | None
    rnext.planner.tokens(s) -> list[str]
    rnext.planner.containment(need, have) -> float
    rnext.planner.title_year(s) -> int | None

The seed is mandatory on purpose: a silently seeded run would make the
"one command per seed" gate meaningless.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

planner = pytest.importorskip("rnext.planner")

SEED = int(os.environ["SEED"])  # KeyError here is the intended explicit failure

WORDS = (
    "Album", "Accept", "Live", "Deluxe", "Discography", "Remaster", "Vinyl", "Instrumental",
    "Bonus", "Edition", "Covers", "The", "Of", "2020", "1994", "24", "96", "FLAC", "APE",
    "MP3", "CD", "Part", "Vol", "Collection", "Heavy", "Metal", "Tribute", "Demo",
)


def random_title(rng: random.Random) -> str:
    words = [rng.choice(WORDS) for _ in range(rng.randint(0, 8))]
    sep = rng.choice([" - ", ", ", " ", " — "])
    title = sep.join(words)
    if rng.random() < 0.4:
        title = rng.choice(["[RM] ", "[CD] ", "[24/96] ", "[LP] "]) + title
    return title.strip()


def random_row(rng: random.Random) -> dict:
    return {
        "title": random_title(rng),
        "seeders": rng.choice([0, 1, 3, 17, 120]),
        "poster_pid": rng.choice([None, 1, 2, 99]),
        "size": rng.randint(0, 5_000_000_000),
    }


def random_ctx(rng: random.Random) -> dict:
    album = [w.lower() for w in rng.sample(WORDS, rng.randint(0, 3))]
    return {
        "artist_tokens": [w.lower() for w in rng.sample(WORDS, rng.randint(0, 3))],
        "album_tokens": album,
        "year": rng.choice([None, 1968, 1994, 2011, 2020, 2022]),
        "codes": rng.choice([[], ["MTR 7562 2"], ["N 0061"]]),
        "own_poster": rng.choice([None, 1, 99]),
    }


def test_seed_is_reported():
    assert 0 <= SEED < 2**31


def test_tokens_is_idempotent_and_drops_years():
    rng = random.Random(SEED)
    for _ in range(200):
        title = random_title(rng)
        once = planner.tokens(title)
        assert once == planner.tokens(" ".join(once))
        assert not [w for w in once if w.isdigit() and len(w) == 4], title


def test_containment_is_a_fraction():
    rng = random.Random(SEED + 1)
    for _ in range(200):
        need = planner.tokens(random_title(rng))
        have = planner.tokens(random_title(rng))
        value = planner.containment(need, have)
        assert 0.0 <= value <= 1.0
        if need:
            assert planner.containment(need, list(need) + have) == 1.0


def test_title_year_within_plausible_range():
    rng = random.Random(SEED + 2)
    for _ in range(200):
        year = planner.title_year(random_title(rng))
        assert year is None or 1900 <= year <= 2099


def test_score_facts_is_deterministic_and_complete():
    rng = random.Random(SEED + 3)
    for _ in range(300):
        row, ctx = random_row(rng), random_ctx(rng)
        first = planner.score_facts(dict(row), dict(ctx))
        second = planner.score_facts(dict(row), dict(ctx))
        assert first == second, (row, ctx)
        assert isinstance(first, dict)
        assert isinstance(first.get("score"), (int, float))
        assert not isinstance(first.get("score"), bool)
        assert first["score"] == first["score"]  # NaN would break ordering
        assert isinstance(first.get("flags"), list)


def test_dead_candidate_is_flagged_or_rejected():
    rng = random.Random(SEED + 4)
    for _ in range(200):
        row = random_row(rng)
        row["seeders"] = 0
        out = planner.score_facts(row, random_ctx(rng))
        if out["score"] > -900:
            assert "dead" in out["flags"], (row, out)


def test_choose_returns_the_best_above_threshold():
    rng = random.Random(SEED + 5)
    for _ in range(200):
        cands = []
        for _ in range(rng.randint(0, 4)):
            out = planner.score_facts(random_row(rng), random_ctx(rng))
            out["seeders"] = rng.choice([0, 5, 50])
            cands.append(out)
        threshold = rng.choice([-1000.0, 0.0, 40.0, 80.0])
        picked = planner.choose([dict(c) for c in cands], threshold=threshold)
        alive = [c for c in cands if c["score"] > -900]
        if not alive or max(c["score"] for c in alive) < threshold:
            assert picked is None, (cands, threshold, picked)
        else:
            assert picked is not None
            assert picked["score"] >= threshold
            assert picked["score"] == max(c["score"] for c in alive)
