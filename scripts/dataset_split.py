#!/usr/bin/env python3
"""Splitting the dataset into train / holdout: by albums AND by authors (no leakage).

Rule: holdout is 30% of the albums of EACH author (deterministic by key hash),
plus the `--by-author` mode where one author goes entirely to holdout and the
other to train (transfer check: "did we generalize to another master").

A pair (release) is the split unit: tracks of one album never land in both
parts, otherwise the evaluation is inflated.

  python3 scripts/dataset_split.py                 # write the split
  python3 scripts/dataset_split.py --report
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import workspace  # noqa: E402


def corpus() -> Path:
    return workspace.dir_of("corpus")


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def h01(key: str) -> float:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.3, help="holdout fraction by albums")
    ap.add_argument("--by-author", action="store_true",
                    help="one author entirely in holdout (transfer across masters)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    out_path = dataset_dir() / "split.json"
    if args.report:
        if not out_path.is_file():
            print("no split")
            return 1
        d = json.loads(out_path.read_text(encoding="utf-8"))
        print("train:", len(d["train"]), "holdout:", len(d["holdout"]),
              "| holdout authors:", Counter(v["author_pid"] for v in d["holdout"].values()))
        return 0

    plan = read_jsonl(corpus() / "dataset_plan.jsonl")
    pairs = {r["key"]: r for r in plan if r["status"].startswith("matched")
             and r["release"].get("n_tracks", 0) >= 6}
    print(f"candidate pairs: {len(pairs)}")

    authors = sorted({str(p["author_pid"]) for p in pairs.values()})
    train, hold = {}, {}
    hold_author = authors[0] if (args.by_author and authors) else None
    for key, p in pairs.items():
        au = str(p["author_pid"])
        entry = {"author_pid": p["author_pid"], "artist": p["artist"],
                 "album": p["release"]["album"], "year": p["release"].get("year"),
                 "n_tracks": p["release"].get("n_tracks")}
        if hold_author and au == hold_author:
            hold[key] = entry
        elif h01(key) < args.frac:
            hold[key] = entry
        else:
            train[key] = entry

    out_path.write_text(json.dumps({"train": train, "holdout": hold,
                                    "frac": args.frac, "by_author": bool(hold_author)},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    tr_a = Counter(str(v["author_pid"]) for v in train.values())
    ho_a = Counter(str(v["author_pid"]) for v in hold.values())
    tr_t = sum(v["n_tracks"] for v in train.values())
    ho_t = sum(v["n_tracks"] for v in hold.values())
    print(f"train {len(train)} pairs / ~{tr_t} tracks {dict(tr_a)}")
    print(f"holdout {len(hold)} pairs / ~{ho_t} tracks {dict(ho_a)}")
    print(f"→ {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
