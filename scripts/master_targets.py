#!/usr/bin/env python3
"""Master target summary: DR-meter tables from topics → corpus/master_targets.jsonl.

The master publishes an analysis of their restoration in the topic
(DR/Peak/RMS per track). This is a DR/crest target **without downloading** the
master release: crest_tt = peak − rms.

  python3 scripts/master_targets.py            # build/update the summary
  python3 scripts/master_targets.py --report   # distributions by author/decade
"""
import argparse
import json
import statistics as S
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import planner, workspace  # noqa: E402


def corpus() -> Path:
    return workspace.dir_of("corpus")


def build() -> int:
    rows = []
    for p in sorted((corpus() / "restored").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not planner.OWN_MARK.search(d.get("title") or ""):
            continue
        rt = planner.parse_restore_title(d["title"])
        for idx, rel in enumerate(d.get("releases", [])):
            md = rel.get("master_dr")
            if not md or not md.get("tracks"):
                continue
            tracks = []
            for t in md["tracks"]:
                tracks.append({
                    "n": t.get("n"), "title": t.get("title"), "dr": t["dr"],
                    "peak_db": t["peak_db"], "rms_db": t["rms_db"],
                    "crest_tt": round(t["peak_db"] - t["rms_db"], 2),
                    "dur": t.get("dur"),
                })
            crests = [t["crest_tt"] for t in tracks]
            rows.append({
                "key": f"{d['topic_id']}/rel{idx}",
                "restored_topic_id": d["topic_id"],
                "author_pid": d.get("author_pid"),
                "artist": rt["artist"], "genre": rt["genre"],
                "year": rel.get("year"), "album": rel.get("album"),
                "source_edition": rel.get("source_edition"),
                "processing": rel.get("processing"),
                "n_tracks": len(tracks),
                "official_dr": md.get("official_dr"),
                "dr_mean": round(S.mean(t["dr"] for t in tracks), 2),
                "crest_tt_mean": round(S.mean(crests), 2),
                "crest_tt_sd": round(S.pstdev(crests), 2) if len(crests) > 1 else 0.0,
                "tracks": tracks,
            })
    out = corpus() / "master_targets.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"releases with a target: {len(rows)} → {out}")
    nt = sum(r["n_tracks"] for r in rows)
    print(f"tracks: {nt}; authors: {dict(Counter(r['author_pid'] for r in rows))}")
    return 0


def report() -> int:
    rows = [json.loads(l) for l in (corpus() / "master_targets.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("empty — build the summary first")
        return 1
    tr = [t for r in rows for t in r["tracks"]]
    crest = [t["crest_tt"] for t in tr]
    print(f"releases {len(rows)}, tracks {len(tr)}")
    print(f"crest_tt (master target): median {S.median(crest):.2f}, "
          f"mean {S.mean(crest):.2f}, sd {S.pstdev(crest):.2f}, "
          f"range {min(crest):.2f}..{max(crest):.2f}")
    dr = [r["official_dr"] for r in rows if r.get("official_dr")]
    print(f"release official DR: median {S.median(dr):.0f}, "
          f"range {min(dr)}..{max(dr)} (n={len(dr)})")
    # master "loudness" classes: DR ≤8 / 9–11 / ≥12
    cls = Counter("loud" if d <= 8 else ("mid" if d <= 11 else "healthy") for d in dr)
    print("classes:", dict(cls))
    dec = Counter((r["year"] or 0) // 10 * 10 for r in rows if r.get("year"))
    print("decades:", dict(sorted(dec.items())))
    by_author = defaultdict(list)
    for r in rows:
        by_author[r["author_pid"]].append(r["crest_tt_mean"])
    for a, v in by_author.items():
        print(f"  author {a}: releases {len(v)}, crest_tt median {S.median(v):.2f}, "
              f"sd {S.pstdev(v):.2f}")
    # within-release spread (matters for sample size choice)
    within = [r["crest_tt_sd"] for r in rows if r["n_tracks"] > 3]
    between = [r["crest_tt_mean"] for r in rows]
    print(f"target spread: between albums sd {S.pstdev(between):.2f}, "
          f"within albums median sd {S.median(within):.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    return report() if args.report else build()


if __name__ == "__main__":
    sys.exit(main())
