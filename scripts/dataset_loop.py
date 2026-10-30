#!/usr/bin/env python3
"""Continuous dataset enrichment + auto-training as it grows.

One pass of the loop:
  1. if pair raw data is still downloading — wait;
  2. otherwise: pick a new batch (dataset_select) → download it (dataset_fetch --go);
  3. extract finished pairs (--keep-raw, raw storage window for evaluation);
  4. refresh the reference audit and the target layer (reference_audit, target_model);
  5. recompute the dataset version and run auto-training with evals (ml_auto).

Idempotent: a re-run continues from the same place. Logs go to `dataset/loop.log`.

  python3 scripts/dataset_loop.py --interval 900 --batch-gb 12 --keep-window 12
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import workspace  # noqa: E402
from rnext.transm import Transm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def log(msg: str) -> None:
    line = f"[{time.strftime('%d.%m %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(dataset_dir() / "loop.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], timeout: int = 7200) -> int:
    log("$ " + " ".join(cmd[1:]))
    p = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout)
    return p.returncode


def downloads_state() -> tuple[float, float, float]:
    """(total GB left, speed MB/s, GB left on torrents WITH PEERS).

    The third number is what can actually finish: dead seeds (0 peers) must not
    hold the in-work cap, otherwise one stuck torrent stalls the whole pipeline.
    """
    try:
        t = Transm()
        left = rate = live = 0.0
        for tor in t.torrents(["downloadDir", "leftUntilDone", "rateDownload",
                               "peersConnected"]):
            if "/dataset/raw/" not in (tor["downloadDir"] or ""):
                continue
            left += tor["leftUntilDone"]
            rate += tor["rateDownload"]
            if (tor.get("peersConnected") or 0) > 0:
                live += tor["leftUntilDone"]
        return left / 1e9, rate / 1e6, live / 1e9
    except Exception:                                    # noqa: BLE001
        return 0.0, 0.0, 0.0


STALLS = 0


def batch_counter() -> int:
    n = 0
    for p in (workspace.dir_of("corpus")).glob("dataset_batch_*.jsonl"):
        try:
            n = max(n, int(p.stem.rsplit("-", 1)[-1]))
        except ValueError:
            continue
    return n


def free_gb() -> float:
    import shutil
    try:
        return shutil.disk_usage(str(workspace.data_root())).free / 1e9
    except Exception:                                    # noqa: BLE001
        return 0.0


def one_pass(args) -> None:
    """One pass: cleanup → extraction/audit/training (ALWAYS) → maybe a new batch.

    Previously the pass exited early while downloads ran — finished pairs sat
    unextracted for hours and batches piled up until the disk filled. Now
    processing runs on every pass, and a new batch is taken only while less
    than max_inflight_gb is in flight.
    """
    global STALLS
    free = free_gb()
    if free < args.min_free_gb:
        log(f"free {free:.1f} GB < {args.min_free_gb} GB — pruning raw")
        subprocess.run([sys.executable, "scripts/dataset_extract.py", "--prune",
                        "--prune-keep", str(args.keep_window)], cwd=str(ROOT), check=False)
        if free_gb() < args.min_free_gb:
            # critical: drop UNFINISHED pairs with low progress too
            log("still low on space — dropping unfinished pairs with low progress")
            subprocess.run([sys.executable, "scripts/dataset_prune_stalled.py",
                            "--max-progress", "0.5"], cwd=str(ROOT), check=False)

    # 0) dead seeds must not hold the cap: drop pairs without peers
    subprocess.run([sys.executable, "scripts/dataset_prune_stalled.py", "--peerless"],
                   cwd=str(ROOT), check=False, stdout=subprocess.DEVNULL)
    # ...and immediately recompute wanted-sets: without this, torrents of dropped
    # pairs keep downloading into deleted directories (raw comes back, space leaks)
    subprocess.run([sys.executable, "scripts/dataset_fetch.py", "--repair"],
                   cwd=str(ROOT), check=False, stdout=subprocess.DEVNULL)

    # 1) process what is ready — every pass, regardless of downloads
    run([sys.executable, "scripts/dataset_extract.py", "--all", "--excerpts", "1",
         "--keep-raw", "--keep-raw-window", str(args.keep_window)])
    run([sys.executable, "scripts/reference_audit.py"])
    run([sys.executable, "scripts/target_model.py"])
    run([sys.executable, "scripts/ml_auto.py", "--with-audio"])
    man = dataset_dir() / "manifest.json"
    if man.is_file():
        m = json.loads(man.read_text(encoding="utf-8"))
        log(f"dataset {m['version']}: {m['pairs']} pairs, {m['trusted_tracks']} trusted tracks")

    # 2) decide whether to take a new batch
    left, rate, live = downloads_state()
    if live > args.max_inflight_gb:
        log(f"in flight (with peers) {live:.1f} GB > cap {args.max_inflight_gb} GB — "
            f"not taking a new batch")
        return
    if left > 0.05 and rate >= args.min_rate_mbps:
        log(f"downloading {left:.2f} GB ({rate:.2f} MB/s) — waiting")
        return
    if left > 0.05:
        STALLS += 1
        if STALLS < args.stall_passes:
            log(f"downloading {left:.2f} GB but slowly ({rate:.2f} MB/s; "
                f"{STALLS}/{args.stall_passes}) — waiting")
            return
        log(f"slow tail confirmed ({rate:.2f} MB/s) — taking a new batch")
    STALLS = 0

    n = batch_counter() + 1
    tag = f"batch-{n:03d}"
    if run([sys.executable, "scripts/dataset_select.py", "--limit", str(args.batch_pairs),
            "--min-master-seeders", str(args.min_master_seeders),
            "--max-master-gb", str(args.max_master_gb), "--out", tag]) != 0:
        log("select failed")
        return
    plan = workspace.dir_of("corpus") / f"dataset_batch_{tag}.jsonl"
    run([sys.executable, "scripts/dataset_fetch.py", "--go", "--limit", str(args.batch_pairs),
         "--max-gb", str(args.batch_gb), "--plan", str(plan)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--batch-pairs", type=int, default=14)
    ap.add_argument("--batch-gb", type=float, default=12.0)
    ap.add_argument("--keep-window", type=int, default=12)
    ap.add_argument("--min-master-seeders", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--min-rate-mbps", type=float, default=2.0,
                    help="below this rate a download counts as a slow tail")
    ap.add_argument("--max-inflight-gb", type=float, default=12.0,
                    help="cap on outstanding downloads: above it no new batch is taken")
    ap.add_argument("--max-master-gb", type=float, default=6.0,
                    help="skip pairs whose master topic is larger than this (giant discographies)")
    ap.add_argument("--stall-passes", type=int, default=2,
                    help="how many passes to wait when downloading but at zero speed")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    log(f"enrichment loop: interval {args.interval}s, batch {args.batch_pairs} pairs "
        f"/ {args.batch_gb} GB, raw kept for the {args.keep_window} most recent pairs")
    # Transmission settings (queue/DHT) only live until the client restarts —
    # after a reboot they return to defaults and speed drops ~20x.
    # Applied on every loop start: both in the session and in settings.json.
    try:
        subprocess.run([sys.executable, "scripts/transmission_tune.py", "--apply"],
                       cwd=str(ROOT), check=False, timeout=60)
        log("Transmission settings applied")
    except Exception as e:                                 # noqa: BLE001
        log(f"failed to apply Transmission settings: {type(e).__name__}")
    while True:
        try:
            one_pass(args)
        except Exception as e:                            # noqa: BLE001
            log(f"pass error: {type(e).__name__}: {str(e)[:200]}")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
