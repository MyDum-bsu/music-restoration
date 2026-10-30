"""Building "restoration -> original" pairs from [RM] master topics.

A pair = (restored_topic_id, release, original_topic_id, match quality).
Output: corpus/pairs.jsonl (append-only, idempotent by the rt/release key).
"""
import json
import re
import time

from rnext.pairsearch import find_originals, split_artist_album
from rnext.rtclient import RTClient
from rnext.rttopic import fetch_topic

PAIRS_FILE = "pairs.jsonl"


def _catalog_bonuses(title: str, source_edition: str | None) -> float:
    """Bonus when the candidate title mentions catalog numbers of the edition."""
    if not source_edition:
        return 0.0
    bonus = 0.0
    # "Victor – VDP-1201", "Noise, N 0061", "IOMLTDCD 608" — catalog numbers; find code patterns with a regex.
    codes = re.findall(r"[A-Z0-9]{2,}[-\s]?[0-9]{3,6}(?:[~-][DC0-9]{1,6})?", source_edition)
    for code in codes:
        if code and code in title:
            bonus += 35
    return bonus


def key_of(rt_id: int, release_idx: int) -> str:
    return f"{rt_id}/rel{release_idx}"


def mine_topic(tid: str, artist_hint: str | None = None, pause: float = 1.5,
               limit_candidates: int = 5, client: RTClient | None = None) -> list[dict]:
    """All releases of a topic -> pair rows with original candidates (no downloads)."""
    c = client or RTClient()
    d = fetch_topic(c, tid)
    if artist_hint:
        artist, _ = split_artist_album(artist_hint)
    else:
        artist, _ = split_artist_album(d["title"].split(")")[-1])
    rows = []
    for i, rel in enumerate(d["releases"]):
        if not rel["album"] or not rel["tracks"]:
            continue
        cands = find_originals(artist, rel["album"], rel["year"], client=c, pages=2, limit=14)
        for k in cands:
            k["score"] += _catalog_bonuses(k["title"], rel["source_edition"])
        cands.sort(key=lambda x: -x["score"])
        rows.append({
            "key": key_of(int(tid), i),
            "restored_topic_id": int(tid),
            "restore_title": d["title"],
            "artist": artist,
            "release": {k: rel[k] for k in ("year", "album", "tracks", "source_edition", "processing")},
            "candidates": [{k: cand[k] for k in ("topic_id", "title", "size_bytes", "size_label", "seeders", "score")}
                           for cand in cands[:limit_candidates]],
            "chosen": cands[0]["topic_id"] if cands and cands[0]["score"] >= 80 else None,
            "status": "pending" if not cands else "scored",
        })
        time.sleep(pause)
    return rows


def load_pairs(workspace_dir) -> dict[str, dict]:
    import pathlib
    p = pathlib.Path(workspace_dir) / "corpus" / PAIRS_FILE
    out = {}
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def save_pairs(workspace_dir, rows: list[dict], append: bool = True) -> int:
    import pathlib
    p = pathlib.Path(workspace_dir) / "corpus"
    p.mkdir(parents=True, exist_ok=True)
    f = p / PAIRS_FILE
    mc = load_pairs(p.parent)
    n = 0
    with open(f, "a" if append else "w", encoding="utf-8") as fh:
        for r in rows:
            if r["key"] in mc:
                continue
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            mc[r["key"]] = r
            n += 1
    return n
