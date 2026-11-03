"""Parser tests on saved HTML fixtures (review: agent-critical parsers untested).

The fixtures under tests/fixtures/ are trimmed copies of real rutracker pages:
topic body, [RM] master with spoilers, DR-meter insert. No network, no /media.
"""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from rnext.rttopic import parse_dr_table, parse_topic_page
from scripts.tracker_hunt import parse_music, parse_video

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def html_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def topic_body(name: str) -> tuple[str, str]:
    """(post body text, topic title) — what the tracker parsers see."""
    soup = BeautifulSoup(html_text(name), "html.parser")
    body = soup.select_one(".post_body")
    assert body is not None, f"{name}: fixture has no .post_body"
    title = soup.title.get_text().split("::")[0].strip() if soup.title else ""
    return body.get_text("\n", strip=True), title


def test_parse_music_reads_a_24bit_topic():
    text, title = topic_body("topic_music_24bit.html")
    facts = parse_music(text, title)

    assert facts["kind"] == "music"
    assert facts["lossless"] is True
    assert facts["format"] == "FLAC"
    assert facts["source"] == "hi-res (24-bit)"
    assert facts["bits"] == 24
    assert facts["khz"] == 96.0
    assert facts["vinyl"] is False
    assert facts["has_log"] is True
    assert facts["has_scans"] is True
    # DR meter rows in the topic: the median of the published values
    assert facts["dr_published"] == 12


def test_parse_music_flags_a_lossy_topic():
    facts = parse_music("MP3 320 kbps, no log", "Accept - Album (1994) [MP3]")

    assert facts["lossless"] is False
    assert facts["format"] == "MP3"
    assert facts["kbps"] == 320
    assert "bits" not in facts


def test_parse_video_reads_a_bdremux_topic():
    text, title = topic_body("topic_video_2160p.html")
    facts = parse_video(text, title)

    assert facts["kind"] == "video"
    assert facts["resolution"] == "2160P"
    assert facts["source"] == "BDRemux"
    assert facts["codec"] == "x265"
    assert facts["hdr"] == "HDR"
    assert "TrueHD" in facts["audio"]
    assert "DTS-HD MA" in facts["audio"]
    assert facts["subs"] == ["rus", "eng"]


def test_parse_topic_page_reads_the_release_and_attaches_the_dr_table():
    page = parse_topic_page(html_text("topic_restored_rm.html"))

    assert page["title"] == "[RM] [restored] Accept - Balls to the Wall (1983)"
    assert len(page["releases"]) == 1
    rel = page["releases"][0]
    assert rel["year"] == 1983
    assert rel["album"] == "Balls to the Wall"
    assert rel["source_edition"] == "Noise – N 0061"
    assert rel["processing"].startswith("declip")
    assert [(t["n"], t["dur"]) for t in rel["tracks"]] == [(1, "5:50"), (2, "3:57")]
    # The master publishes its own analysis: the restoration target, no download.
    assert rel["master_dr"]["official_dr"] == 12
    assert [row["dr"] for row in rel["master_dr"]["tracks"]] == [13, 12]


def test_parse_topic_page_reads_metadata_of_a_video_topic():
    page = parse_topic_page(html_text("topic_video_2160p.html"))

    assert page["title"] == "Some Film (2021) BDRemux 2160p"
    assert page["meta"]["Жанр"] == "Drama"


def test_parse_dr_table_reads_every_row():
    soup = BeautifulSoup(html_text("dr_meter_block.html"), "html.parser")
    body = soup.select_one(".sp-body")
    assert body is not None, "fixture has no .sp-body"

    table = parse_dr_table(body.get_text("\n").replace("\xa0", " "))

    assert table is not None
    assert table["analyzed"] == "Accept / Balls to the Wall (1983) [24/96]"
    assert table["official_dr"] == 11
    assert [row["n"] for row in table["tracks"]] == [1, 2, 3]
    assert [row["dr"] for row in table["tracks"]] == [12, 11, 9]
    assert table["tracks"][0]["peak_db"] == -0.5
    assert table["tracks"][0]["rms_db"] == -14.2
    assert table["tracks"][2]["title"] == "Head Over Heels"


def test_parse_dr_table_returns_none_without_a_table():
    assert parse_dr_table("no analysis here") is None


@pytest.mark.parametrize("fixture", ["topic_music_24bit.html", "topic_video_2160p.html",
                                     "topic_restored_rm.html", "dr_meter_block.html"])
def test_every_fixture_is_a_saved_html_page(fixture: str) -> None:
    soup = BeautifulSoup(html_text(fixture), "html.parser")
    assert soup.select_one(".post_body") is not None
