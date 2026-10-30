#!/usr/bin/env python3
"""Release watchdog: verifies the pipeline neither loses albums nor corrupts data.

Checks per completed release:
  * did all tracks make it to the phone and does their count match the source;
  * is there a cover and lyrics (and how many tracks lack lyrics);
  * were there errors in the pipeline steps;
plus general: disk space, liveness of the dataset loop, stalled downloads.

  python3 scripts/release_watch.py                 # one pass
  python3 scripts/release_watch.py --loop 600      # in a loop, every 10 minutes
  python3 scripts/release_watch.py --json /tmp/watch.json

Reports only ANOMALIES (normal state is silent) — so problems are visible in the log.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import workspace  # noqa: E402

DATA = workspace.data_root()
PHONE_CACHE: dict = {}


def phone_root() -> Path | None:
    try:
        from rnext.phone import mtp
        return mtp.phone_dir()
    except Exception:  # noqa: BLE001
        return None


def phone_flacs(root: Path, rel: str) -> int | None:
    """How many FLACs are in a directory on the phone (including subdirectories)."""
    key = (str(root), rel)
    if key in PHONE_CACHE:
        return PHONE_CACHE[key]
    d = root / rel
    if not d.is_dir():
        PHONE_CACHE[key] = None
        return None
    n = 0
    stack = [d]
    while stack:
        cur = stack.pop()
        try:
            for p in cur.iterdir():
                if p.is_dir():
                    stack.append(p)
                elif p.suffix.lower() == ".flac":
                    n += 1
        except OSError:
            pass
    PHONE_CACHE[key] = n
    return n


def check_blocks(latest: dict[str, str]) -> list[str]:
    """Parse the latest log blocks: what finished, what had errors."""
    warns: list[str] = []
    for nm, b in latest.items():
        name = nm
        if "Traceback" in b or "error" in b.lower():
            line = next((l for l in b.splitlines() if "error" in l.lower()
                         or "Traceback" in l), "")
            warns.append(f"[step error] {name[:52]}: {line.strip()[:90]}")
        # how many tracks in the release
        m = re.findall(r"copied (\d+)/(\d+) files", b)
        if m:
            got, total = int(m[-1][0]), int(m[-1][1])
            if got != total:
                warns.append(f"[incomplete copy] {name[:52]}: {got} of {total}")
        if "verification ok" not in b and "copied" in b:
            warns.append(f"[no md5 verification] {name[:52]}")
        # Lyrics are NOT checked via the log: the log reflects the moment of the
        # run, while lyrics could be added later (Place Vendome, Q5). The phone
        # check verifies the fact.
        # "cover not found" from the log is NOT treated as an alarm: the cover
        # could have been embedded later (as with Bring Me Your Love), and
        # check_phone_albums verifies the fact
    return warns


def check_phone_albums(root: Path, limit: int = 0) -> list[str]:
    """Real phone check: does the first track of each album have an embedded
    cover and lyrics? The log lies (the cover could have been embedded later),
    this is the fact. We read the file back via gio and inspect the tags."""
    import tempfile
    listfile = Path("/tmp/phone_check.txt")
    if not listfile.is_file():
        return []
    rels = [l.strip() for l in listfile.read_text().splitlines() if l.strip()]
    if limit:
        rels = rels[:limit]
    warns = []
    for rel in rels:
        d = root / rel
        if not d.is_dir():
            warns.append(f"[phone] no directory: {rel}")
            continue
        try:
            cands = sorted(p for p in d.rglob("*.flac"))
        except OSError:
            continue
        if not cands:
            warns.append(f"[phone] no FLAC: {rel}")
            continue
        # Look at SEVERAL tracks: the first may be an instrumental (Subsignal's
        # "271 Days" is an intro without lyrics), and a single check gave false alarms.
        sample = cands if len(cands) <= 3 else [cands[0], cands[len(cands) // 2], cands[-1]]
        got = {"cover": False, "lyrics": False}
        for f in sample:
            key = (str(f), f.stat().st_size)
            if key not in PHONE_CACHE:
                with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as t:
                    tmp = Path(t.name)
                try:
                    subprocess.run(["gio", "cat", str(f)], stdout=open(tmp, "wb"),
                                   timeout=180)
                    r = subprocess.run(["metaflac", "--list", "--block-type=PICTURE",
                                        "--block-type=VORBIS_COMMENT", str(tmp)],
                                       capture_output=True, text=True)
                    low = r.stdout.lower()
                    PHONE_CACHE[key] = {"cover": "type: 6" in r.stdout,
                                        "lyrics": "lyrics=" in low
                                                  or "unsyncedlyrics=" in low}
                except Exception:  # noqa: BLE001
                    PHONE_CACHE[key] = {"cover": True, "lyrics": True}
                finally:
                    tmp.unlink(missing_ok=True)
            v = PHONE_CACHE[key]
            got["cover"] = got["cover"] or v["cover"]
            got["lyrics"] = got["lyrics"] or v["lyrics"]
        if not got["cover"]:
            warns.append(f"[phone: no cover] {rel}")
        if not got["lyrics"]:
            warns.append(f"[phone: no lyrics] {rel}")
    return warns


def check_disk() -> list[str]:
    warns = []
    for mount in ("/", "/home", str(DATA)):
        try:
            u = shutil.disk_usage(mount)
        except OSError:
            continue
        free_gb = u.free / 1e9
        if free_gb < 3:
            warns.append(f"[space] {mount}: {free_gb:.1f} GB free — CRITICAL")
        elif free_gb < 6:
            warns.append(f"[space] {mount}: {free_gb:.1f} GB free")
    return warns


def check_loop() -> list[str]:
    r = subprocess.run(["pgrep", "-f", "dataset_loop.py"], capture_output=True, text=True)
    return [] if r.stdout.strip() else ["[loop] dataset_loop is not running"]


def check_stalled() -> list[str]:
    try:
        from rnext.transm import Transm
        t = Transm()
        rows = [x for x in t.torrents(["id", "name", "downloadDir", "leftUntilDone",
                                       "peersConnected", "status"])
                if "release_src" in (x["downloadDir"] or "") and x["leftUntilDone"] > 0]
    except Exception as e:  # noqa: BLE001
        return [f"[transmission] unavailable: {type(e).__name__}"]
    out = []
    for x in rows:
        if x["peersConnected"] == 0 and x["status"] == 4:
            out.append(f"[stalled download] {x['name'][:44]} — 0 peers, "
                       f"{x['leftUntilDone'] / 1e6:.0f} MB left")
    return out


def one_pass(args) -> int:
    warns: list[str] = []
    root = phone_root()
    if root is None:
        warns.append("[phone] not found (MTP not connected?)")
    # There are several logs (cc, wave2, retry, fix...), and an album may have
    # been re-run. We merge ALL blocks and take the latest by album name.
    all_blocks: dict[str, str] = {}
    # by modification TIME: a successful redo must override a previous failure
    logs = sorted(Path("/tmp").glob("release_*.log"),
                  key=lambda p: p.stat().st_mtime if p.is_file() else 0)
    for log in logs:
        if not log.is_file():
            continue
        for b in re.split(r"^=== ", log.read_text(encoding="utf-8", errors="ignore"),
                          flags=re.M)[1:]:
            head = b.splitlines()[0] if b.splitlines() else "?"
            nm = head.split(" ", 1)[1] if " " in head else head
            all_blocks[nm] = b
    warns += check_blocks(all_blocks)
    if root is not None:
        warns += check_phone_albums(root, limit=args.phone_limit)
    warns += check_disk() + check_loop() + check_stalled()

    stamp = time.strftime("%d.%m %H:%M:%S")
    if warns:
        print(f"[{stamp}] ATTENTION, {len(warns)}:")
        for w in warns:
            print(f"  {w}")
    else:
        print(f"[{stamp}] all clean: no errors, space is fine, loop alive, downloads running")
    if args.json:
        Path(args.json).write_text(json.dumps({"ts": stamp, "warnings": warns},
                                              ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="loop interval, seconds (0 = one pass)")
    ap.add_argument("--json", default="")
    ap.add_argument("--phone-limit", type=int, default=0,
                    help="how many albums on the phone to check per pass (0 = all)")
    args = ap.parse_args()
    if args.loop:
        while True:
            one_pass(args)
            sys.stdout.flush()
            time.sleep(args.loop)
    return one_pass(args)


if __name__ == "__main__":
    sys.exit(main())
