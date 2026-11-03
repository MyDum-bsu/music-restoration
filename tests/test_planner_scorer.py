"""One release scorer: rnext.planner.score_facts, everyone else delegates.

The review's first item: `rnext/planner.score_candidate` and
`scripts/tracker_hunt.score` were two independent scorers with different
criteria, so the skill flow and the dataset flow could rank the same topic
differently. These tests pin the single owner and the delegation.
"""

import re
from pathlib import Path

import pytest

from rnext import planner
from scripts import tracker_hunt

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("rnext", "scripts", "research")
# the weight tables themselves: source/label -> points
WEIGHT_MARKERS = ('"vinyl/DSD": 5', '"BDRemux": 30', '"hi-res (24-bit)": 30', '"2160P": 40')

MUSIC = {"kind": "music", "lossless": True, "source": "hi-res (24-bit)", "bits": 24,
         "dr_published": 12, "has_log": True, "has_scans": False, "image_cue": False}
VIDEO = {"kind": "video", "resolution": "2160P", "source": "BDRemux", "codec": "x265",
         "hdr": "HDR10", "audio": ["TrueHD"], "subs": ["rus"]}
BOOK = {"kind": "book", "format": "pdf", "text_layer": True}


def test_tracker_hunt_delegates_to_the_single_scorer():
    for facts in (MUSIC, VIDEO, BOOK):
        assert tracker_hunt.score(dict(facts)) == (
            planner.score_facts(dict(facts))["score"],
            planner.score_facts(dict(facts))["flags"],
        )


def test_weight_tables_live_only_in_the_planner():
    owners = []
    for directory in SCANNED:
        for path in sorted((ROOT / directory).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in WEIGHT_MARKERS):
                owners.append(path.relative_to(ROOT).as_posix())

    assert owners == ["rnext/planner.py"]


def test_the_scorer_answers_with_score_and_flags():
    out = planner.score_facts({"title": "Accept - Balls to the Wall (1983) FLAC",
                               "seeders": 12, "poster_pid": None},
                              {"artist_tokens": ["accept"],
                               "album_tokens": ["balls", "wall"], "year": 1983,
                               "codes": [], "own_poster": 1})

    assert isinstance(out["score"], float)
    assert isinstance(out["flags"], list)
    assert out["kind"] == "album"


def test_a_dead_candidate_is_flagged():
    out = planner.score_facts({"title": "Accept - Balls to the Wall (1983) FLAC",
                               "seeders": 0, "poster_pid": None},
                              {"artist_tokens": ["accept"],
                               "album_tokens": ["balls", "wall"], "year": 1983})

    assert "dead" in out["flags"]
    assert out["score"] < 80, "a dead torrent must not pass the match threshold"


def test_own_restoration_is_rejected():
    row = {"title": "[RM] [restored] Accept - Balls to the Wall (1983) FLAC",
           "seeders": 20, "poster_pid": 99}

    out = planner.score_facts(row, {"artist_tokens": ["accept"],
                                    "album_tokens": ["balls", "wall"],
                                    "own_poster": 99})

    assert out["flags"] == ["own_restoration"]
    assert out["score"] <= -900


def test_choose_takes_the_best_above_the_threshold():
    best = {"score": 95.0, "flags": [], "seeders": 4}
    second = {"score": 90.0, "flags": [], "seeders": 40}

    assert planner.choose([second, best], threshold=80.0) is best
    assert planner.choose([second, best], threshold=96.0) is None
    assert planner.choose([], threshold=0.0) is None
    assert planner.choose([{"score": -1000.0, "flags": [], "kind": "album"}]) is None


def test_version_quality_orders_versions():
    lossless = dict(MUSIC)
    lossy = dict(MUSIC, lossless=False)
    hd = dict(VIDEO)
    sd = dict(VIDEO, resolution="1080P")

    assert planner.score_facts(lossless)["score"] > planner.score_facts(lossy)["score"]
    assert planner.score_facts(hd)["score"] >= planner.score_facts(sd)["score"]


@pytest.mark.parametrize("kind", ["music", "video", "book", "generic"])
def test_every_known_kind_is_scored_without_raising(kind: str):
    out = planner.score_facts({"kind": kind})

    assert isinstance(out["score"], float)
    assert out["kind"] == kind


def test_tokens_and_containment_stay_the_documented_helpers():
    assert planner.tokens("Accept - Balls to the Wall (1983)") == ["accept", "balls", "wall"]
    assert planner.containment(["balls", "wall"], ["balls", "wall", "live"]) == 1.0
    assert planner.containment(["balls"], ["live"]) == 0.0
    assert planner.title_year("Accept - Balls to the Wall (1983)") == 1983
    assert planner.title_year("no year here") is None


def test_no_second_scorer_definition_in_the_repo():
    """`score_facts` must be defined once: the planner owns the weights."""
    pattern = re.compile(r"(?m)^def score_facts\(")
    definers = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in {".git", ".factory", ".venv", "node_modules", "app", "oracle"}:
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
            definers.append(rel.as_posix())

    assert definers == ["rnext/planner.py"]
