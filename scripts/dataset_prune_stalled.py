#!/usr/bin/env python3
"""Wiping raw of UNFINISHED pairs with low progress (emergency space release).

Needed when the disk is full: we delete raw of pairs downloaded less than
--max-progress and mark them `paused_space`, so the selector puts them back to
work later. Pairs ready for extraction and the last `--keep` extracted ones
are left alone.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import sides as S  # noqa: E402
from rnext.transm import Transm  # noqa: E402
from rnext import workspace  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-progress", type=float, default=0.5)
    ap.add_argument("--peerless", action="store_true",
                    help="drop pairs with NO peers (dead seeds) — they hold the "
                         "in-work cap, blocking new batches")
    ap.add_argument("--keep", type=int, default=6, help="how many recently extracted pairs to leave alone")
    args = ap.parse_args()

    D = workspace.data_root() / "dataset"
    state = S.merged_state(D / "state.jsonl")
    # how many times the pair was already postponed (append-only journal, count entries)
    pauses: dict[str, int] = {}
    for line in (D / "state.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("state") in ("paused_space", "dead_swarm"):
            pauses[r["key"]] = pauses.get(r["key"], 0) + 1
    tmap = S.torrents_by_id()
    if args.peerless:
        print("--peerless mode: dropping pairs without peers")
    have = {p.parent.name.replace("__", "/") for p in (D / "pairs").glob("*/features.json")}
    freed = deleted = kept = 0
    for k, r in state.items():
        if k in have or r.get("state") not in ("downloading", "paused_space"):
            continue
        base = Path(r.get("dir") or "")
        base_exists = base.is_dir()
        done = total = 0
        for side in ("orig", "master"):
            srec = r.get(side) if isinstance(r.get(side), dict) else None
            p = S.side_presence(srec, side, base, tmap=tmap)
            done += p["present"]
            total += p["wanted"]
        prog = (done / total) if total else 0.0

        if args.peerless:
            # in "dead seeds" mode only peer presence matters: 0 peers = the pair
            # will never finish, even with high progress
            peers = 0
            for side in ("orig", "master"):
                srec = r.get(side) if isinstance(r.get(side), dict) else None
                tid = (srec or {}).get("torrent_tid")
                tor = tmap.get(tid) if tid is not None else None
                if tor is None:
                    nm = (srec or {}).get("name")
                    tor = next((x for x in tmap.values() if nm and x.get("name") == nm), None)
                peers += (tor or {}).get("peersConnected") or 0
            if peers > 0:
                kept += 1
                continue
        else:
            # normal mode (space shortage): touch only low-progress pairs
            if not base_exists or prog >= args.max_progress:
                kept += 1
                continue

        sz = sum(f.stat().st_size for f in base.rglob("*") if f.is_file()) if base_exists else 0
        n_pause = pauses.get(k, 0)
        new_state = "dead_swarm" if n_pause >= 2 else "paused_space"
        with open(D / "state.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**r, "state": new_state,
                                "reason": f"peerless, progress {prog:.0%}, pause #{n_pause + 1}",
                                "raw_existed": base_exists},
                               ensure_ascii=False) + "\n")
        if base_exists:
            shutil.rmtree(base, ignore_errors=True)
        freed += sz
        deleted += 1
        print(f"  dropped {k}: no peers, progress {prog:.0%}, {sz / 1e6:.0f} MB"
              + ("" if base_exists else " (raw was already gone)"))

    print(f"removed {deleted} directories (freed {freed / 1e9:.1f} GB), "
          f"kept with progress ≥{args.max_progress:.0%}: {kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
