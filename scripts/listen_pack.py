#!/usr/bin/env python3
"""Assemble a listening pack (blind A/B/C comparison of variants).

What it does:
  * takes the pair's excerpts (or cuts them from raw material), loudness-matches
    ALL variants in LUFS (ebur128), not RMS — otherwise a level difference gives
    the answer away and manual comparison has to correct for it every time;
  * shuffles the variants and lays them out into A/B(/C) with blind labeling;
  * writes `pack.json` (what to play, where the original/master/ours is) — the
    player only reveals the answer map via the "show" button.

Pack layout:
    <pack>/audio/<item>_<label>.flac
    <pack>/pack.json

  python3 scripts/listen_pack.py --all --limit 6 --out listen/pack-001
  python3 scripts/listen_pack.py --key 5220623/rel2 --ours-dir out/silver-side-up
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_extract as DX  # noqa: E402
from rnext import features as F  # noqa: E402
from rnext import workspace  # noqa: E402

WINDOW_RE = re.compile(r"^t(\d{2})_(loud|mid)(?:_(\d+))?_(orig|master)\.flac$")
LABELS = ["A", "B", "C", "D"]


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def gain_to_lufs(src: Path, dst: Path, target_lufs: float) -> dict:
    """Bring the file to the target loudness in LUFS (one correction + peak headroom)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = F.lufs(str(src))
    i = info.get("lufs")
    if i is None:
        # loudness not measurable (silence) — just copy
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "flac", str(dst)],
                       check=True)
        return {"lufs": None, "gain_db": 0.0}
    peak = info.get("tp_peak_dbfs")
    gain = target_lufs - float(i)
    if peak is not None and peak + gain > -1.0:
        gain = -1.0 - float(peak)          # don't push into clipping
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src),
           "-af", f"volume={gain:.2f}dB", "-c:a", "flac", "-compression_level", "5", str(dst)]
    subprocess.run(cmd, check=True)
    return {"lufs": round(float(i), 2), "gain_db": round(gain, 2),
            "peak_dbfs": round(peak, 2) if peak is not None else None}


def pair_excerpts(key: str) -> list[tuple[str, str, Path, Path]]:
    """[(item_id, tag, orig_file, master_file)] from already-extracted excerpts."""
    d = dataset_dir() / "pairs" / key.replace("/", "__") / "excerpts"
    if not d.is_dir():
        return []
    by: dict[tuple[str, str], dict[str, Path]] = {}
    for f in sorted(d.glob("*.flac")):
        m = WINDOW_RE.match(f.name)
        if not m:
            continue
        tr, tag, mark, side = m.groups()
        tag = f"{tag}{mark or ''}"
        by.setdefault((tr, tag), {})[side] = f
    out = []
    for (tr, tag), sides in sorted(by.items()):
        if "orig" in sides and "master" in sides:
            out.append((f"t{tr}_{tag}", tag, sides["orig"], sides["master"]))
    return out


def ours_file(ours_dir: Path | None, key: str, track: int, title: str = "") -> Path | None:
    """Our run's file for a track (by number or by title)."""
    if not ours_dir:
        return None
    cands = sorted(ours_dir.glob("*.flac"))
    for c in cands:
        m = re.match(r"^(\d{1,3})[.\s-]", c.name)
        if m and int(m.group(1)) == track:
            return c
    return None


def build_item(item_id: str, variants: dict[str, Path], out_dir: Path, target_lufs: float,
               rng: random.Random, note: str = "") -> dict:
    order = list(variants.keys())
    rng.shuffle(order)                      # blind layout
    labels = {}
    files = {}
    for label, name in zip(LABELS, order):
        src = variants[name]
        dst = out_dir / f"{item_id}_{label}.flac"
        norm = gain_to_lufs(src, dst, target_lufs)
        labels[label] = name
        files[label] = {"file": f"audio/{dst.name}", **norm}
    return {"id": item_id, "labels": labels, "files": files, "note": note}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=6, help="pairs per pack")
    ap.add_argument("--items-per-pair", type=int, default=2)
    ap.add_argument("--ours-dir", default=None, help="directory of our run (restore_one)")
    ap.add_argument("--target-lufs", type=float, default=-20.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0xAB5)
    args = ap.parse_args()

    pairs_root = dataset_dir() / "pairs"
    keys = sorted(p.parent.name.replace("__", "/") for p in pairs_root.glob("*/features.json"))
    if args.key:
        keys = [k for k in keys if k == args.key]
    elif not args.all:
        ap.error("--key or --all is required")
    keys = keys[:args.limit]

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    items = []
    for key in keys:
        ex = pair_excerpts(key)
        if not ex:
            print(f"  [skip] {key}: no excerpts (run dataset_extract --excerpts 1)")
            continue
        meta_file = pairs_root / key.replace("/", "__") / "meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
        for item_id, tag, o, m in ex[:args.items_per_pair]:
            variants = {"original": o, "master": m}
            track_no = int(item_id[1:3])
            ours = ours_file(Path(args.ours_dir) if args.ours_dir else None, key, track_no)
            if ours:
                variants["ours"] = ours
            it = build_item(item_id, variants, out / "audio", args.target_lufs, rng,
                            note=f"{meta.get('artist')} — {meta.get('album')} ({tag})")
            it["key"] = key
            it["artist"] = meta.get("artist")
            it["album"] = meta.get("album")
            items.append(it)
            print(f"  {key} {item_id}: variants {it['labels']}")

    pack = {
        "name": out.name,
        "created": __import__("time").strftime("%Y-%m-%d %H:%M"),
        "target_lufs": args.target_lufs,
        "n_items": len(items),
        "has_ours": any("ours" in it["labels"].values() for it in items),
        "items": items,
    }
    (out / "pack.json").write_text(json.dumps(pack, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    (out / "ratings.json").write_text(json.dumps({"ratings": {}}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\npack: {len(items)} items → {out}")
    print(f"  player: open in the app (see app/) or {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
