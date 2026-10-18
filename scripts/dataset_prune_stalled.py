#!/usr/bin/env python3
"""Снос сырья у НЕЗАВЕРШЁННЫХ пар с низкой готовностью (аварийный выпуск места).

Нужен, когда диск забит: удаляем сырьё пар, которые скачаны меньше чем на
--max-progress, и помечаем их состоянием `paused_space`, чтобы селектор вернул
их в работу позже. Готовые к извлечению пары и последние `--keep` извлечённых
не трогаем.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import sides as S  # noqa: E402
from rnext import workspace  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-progress", type=float, default=0.5)
    ap.add_argument("--keep", type=int, default=6, help="сколько свежих извлечённых пар не трогать")
    args = ap.parse_args()

    D = workspace.data_root() / "dataset"
    state = S.merged_state(D / "state.jsonl")
    tmap = S.torrents_by_id()
    have = {p.parent.name.replace("__", "/") for p in (D / "pairs").glob("*/features.json")}
    freed = deleted = kept = 0
    for k, r in state.items():
        if k in have or r.get("state") not in ("downloading", "paused_space"):
            continue
        base = Path(r.get("dir") or "")
        if not base.is_dir():
            continue
        done = total = 0
        for side in ("orig", "master"):
            srec = r.get(side) if isinstance(r.get(side), dict) else None
            p = S.side_presence(srec, side, base, tmap=tmap)
            done += p["present"]
            total += p["wanted"]
        if total == 0:
            continue
        prog = done / total
        if prog >= args.max_progress:
            kept += 1
            continue
        sz = sum(f.stat().st_size for f in base.rglob("*") if f.is_file())
        with open(D / "state.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**r, "state": "paused_space",
                                "reason": f"raw removed at {prog:.0%} progress"},
                               ensure_ascii=False) + "\n")
        shutil.rmtree(base, ignore_errors=True)
        freed += sz
        deleted += 1
        print(f"  снято {k}: {prog:.0%} готовности, {sz / 1e6:.0f} MB")
    print(f"удалено каталогов {deleted} (освобождено {freed / 1e9:.1f} GB), "
          f"оставлено с прогрессом ≥{args.max_progress:.0%}: {kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
