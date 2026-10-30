#!/usr/bin/env python3
"""Picking the first download batch from the plan: stratification by author × decade × crushedness.

Rules:
  * only matched / matched_graphy, seeders ≥ --min-seeders, size ≤ --max-gb per pair;
  * album topics (matched) get priority — verification is cheaper than for a discography;
  * cover both authors and different decades; within a stratum take the lowest master DR
    (more crushed material = larger delta = more useful for tuning);
  * dedup by (artist, album): one album = one pair.

  python3 scripts/dataset_select.py --limit 12 [--per-stratum 2] [--out batch-001]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import planner  # noqa: E402
from rnext import workspace  # noqa: E402


def corpus() -> Path:
    return workspace.dir_of("corpus")


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def decade(y) -> int:
    return (y or 0) // 10 * 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--min-seeders", type=int, default=2)
    ap.add_argument("--min-master-seeders", type=int, default=2,
                    help="minimum seeders on the master topic (otherwise the side is not downloaded)")
    ap.add_argument("--max-gb", type=float, default=2.5, help="max size of the chosen torrent")
    ap.add_argument("--min-tracks", type=int, default=8, help="minimum tracks in a release")
    ap.add_argument("--min-mb", type=float, default=150.0, help="minimum torrent size, MB")
    ap.add_argument("--out", default="batch-001")
    ap.add_argument("--balance-authors", action="store_true", default=True,
                    help="top up batches from the lagging author (needed for cross-author validation)")
    ap.add_argument("--no-balance-authors", dest="balance_authors", action="store_false")
    ap.add_argument("--min-per-stratum", type=int, default=200,
                    help="track target per stratum (author × decade): empty cells are filled first")
    ap.add_argument("--only-author", default=None, help="download only this author (pid)")
    ap.add_argument("--max-master-gb", type=float, default=0.0,
                    help="skip pairs whose master topic is larger than N GB (0 = no limit)")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(corpus() / "dataset_plan.jsonl")
            if r["status"].startswith("matched")]
    plan_index = {r["key"]: r for r in rows}
    print(f"matched in plan: {len(rows)}")

    # seeders and SIZE of the master topic — from the topic parse. Large topics
    # (10+ GB discographies) are dangerous: the wanted-set there is shared by
    # many pairs, the master side easily stalls at 0/N and raw piles up, filling the disk.
    mseed: dict[int, int] = {}
    msize: dict[int, int] = {}
    for p in (corpus() / "restored").glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        mseed[d.get("topic_id")] = d.get("topic_seeders") or 0
        msize[d.get("topic_id")] = d.get("topic_size_bytes") or 0

    # how many pairs are already downloaded/extracted per author: top up the lagging one first
    plan_author = {r["key"]: r.get("author_pid") for r in rows}
    done_by_author: dict[str, int] = {}
    st = workspace.data_root() / "dataset" / "state.jsonl"
    if st.is_file():
        for line in st.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("state") in ("downloading", "done", "extracting"):
                a = str(r.get("author_pid") or plan_author.get(r["key"]))
                done_by_author[a] = done_by_author.get(a, 0) + 1
    author_order = sorted(done_by_author, key=lambda a: done_by_author[a])
    print(f"already in dataset by author: {done_by_author or 'none'}"
          + (f" → topping up {author_order[0]}" if args.balance_authors and author_order else ""))

    # ── what is already in the dataset by strata (author × decade) ────────────────
    have: dict[tuple, int] = defaultdict(int)
    pairs_root = workspace.data_root() / "dataset" / "pairs"
    for d in sorted(pairs_root.glob("*/features.json")):
        try:
            feat_rows = json.loads(d.read_text(encoding="utf-8"))
        except Exception:                                  # noqa: BLE001
            continue
        mp = d.parent / "meta.json"
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
        key = d.parent.name.replace("__", "/")
        plan_rec = plan_index.get(key) or {}
        au = str(meta.get("author_pid") or plan_rec.get("author_pid"))
        yr = meta.get("year") or (plan_rec.get("release") or {}).get("year")
        have[(au, decade(yr))] += len(feat_rows)
    if have:
        print("in the dataset by strata (author, decade) → tracks:")
        for (a, dec), n in sorted(have.items()):
            print(f"   {a} {dec}s: {n}")

    # What is already registered. Skip only TERMINAL states: pairs with RPC/network
    # errors are temporary and must be retried (otherwise they drop out of work forever).
    TERMINAL = {"no_match", "edition_mismatch", "corrupt", "done", "extracting",
                "dead_swarm"}
    done_keys: set[str] = set()
    from rnext import sides as _S
    st_file = workspace.data_root() / "dataset" / "state.jsonl"
    for key, rec in _S.merged_state(st_file).items():
        if rec.get("state") in TERMINAL or rec.get("state") == "downloading":
            done_keys.add(key)
    seen: set[tuple] = set()
    cand: list[dict] = []
    skipped_seed = skipped_alt = skipped_big = 0
    for r in rows:
        ch = r.get("chosen") or {}
        size = ch.get("size_bytes") or 0
        if (ch.get("seeders") or 0) < args.min_seeders or size > args.max_gb * 1e9:
            continue
        if (r["release"].get("n_tracks") or 0) < args.min_tracks:
            continue                    # singles/EPs — too little material for a pair
        if size < args.min_mb * 1e6:
            continue
        if args.only_author and str(r["author_pid"]) != str(args.only_author):
            continue
        # already registered pairs (including rejected) are not taken again
        if r["key"] in done_keys:
            continue
        # editions with different content (acapellas/instrumentals) — not pairs
        title = ch.get("title") or ""
        alt = planner.CONTENT_ALT.search(title)
        if alt and not planner.CONTENT_ALT.search(r["release"]["album"] or ""):
            skipped_alt += 1
            continue
        ms = mseed.get(r["restored_topic_id"], 0)
        if ms < args.min_master_seeders:
            skipped_seed += 1
            continue
        msz = msize.get(r["restored_topic_id"], 0)
        if args.max_master_gb and msz > args.max_master_gb * 1e9:
            skipped_big += 1
            continue
        dedup = (r["artist"].lower(), (r["release"]["album"] or "").lower())
        if dedup in seen:
            continue
        seen.add(dedup)
        md = (r["release"].get("master_dr") or {}).get("official_dr") or 99
        cand.append({**r, "_md": md, "_size": size, "_mse": ms, "_msz": msz,
                     "_graphy": "graphy" in (ch.get("flags") or []),
                     "_dec": decade(r["release"].get("year"))})
    if skipped_seed:
        print(f"dropped due to weak master topic seeders: {skipped_seed}")
    if skipped_alt:
        print(f"dropped as a different edition (acapellas/instrumentals): {skipped_alt}")
    if skipped_big:
        print(f"dropped due to a giant master topic (>{args.max_master_gb:.0f} GB): {skipped_big}")

    # stratum: author × decade × master crushedness class
    def cls(md):
        return "loud" if md <= 10 else ("mid" if md <= 12 else "other")

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in cand:
        strata[(r["author_pid"], r["_dec"], cls(r["_md"]))].append(r)

    def deficit(r) -> int:
        """How many tracks the stratum is short (0 — already filled)."""
        return max(0, args.min_per_stratum - have.get((str(r["author_pid"]), r["_dec"]), 0))

    def rank(r):
        # 1) fill empty strata first (author × decade),
        # 2) album topics with a known target, 3) low master DR (more crushed),
        # 4) more seeders, 5) smaller size
        return (-deficit(r), r["_graphy"], r["_md"] == 99, r["_md"],
                r.get("_msz", 0), -r["_mse"], r["_size"])

    picked = []
    # round-robin over strata: one from each until the limit is reached
    def author_rank(k):
        a = str(k[0])
        return (done_by_author.get(a, 0), -len(strata[k]), k) if args.balance_authors \
            else (-len(strata[k]), k)
    order = sorted(strata, key=author_rank)
    for want in range(1, args.per_stratum + 1):
        for key in order:
            group = sorted(strata[key], key=rank)
            if len(group) >= want and len(picked) < args.limit:
                picked.append(group[want - 1])
    seen_ids = {id(x) for x in picked}
    for r in sorted(cand, key=rank):
        if len(picked) >= args.limit:
            break
        if id(r) not in seen_ids:
            picked.append(r)
            seen_ids.add(id(r))

    out = corpus() / f"dataset_batch_{args.out}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(r["_size"] for r in picked)
    print(f"picked {len(picked)} pairs → {out}  (torrents ~{total / 1e9:.1f} GB, "
          f"both sides ~{2 * total / 1e9:.1f} GB upper estimate)")
    print(f"{'author':>9} {'year':>5} {'DRm':>3} {'ms':>3} {'GB':>5} {'se':>3} {'type':<5} artist — album")
    for r in picked:
        ch = r["chosen"]
        print(f"{r['author_pid']:>9} {str(r['release'].get('year')):>5} {r['_md']:>3} "
              f"{r['_mse']:>3} {r['_size'] / 1e9:>5.2f} {ch.get('seeders'):>3} "
              f"{'graph' if r['_graphy'] else 'album':<5} "
              f"{r['artist'][:22]} — {r['release']['album'][:38]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
