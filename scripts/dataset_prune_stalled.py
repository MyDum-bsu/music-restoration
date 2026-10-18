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
from rnext.transm import Transm  # noqa: E402
from rnext import workspace  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-progress", type=float, default=0.5)
    ap.add_argument("--peerless", action="store_true",
                    help="снимать пары, у которых НЕТ пиров (мёртвые сиды) — "
                         "они держат кап «в работе», блокируя новые партии")
    ap.add_argument("--keep", type=int, default=6, help="сколько свежих извлечённых пар не трогать")
    args = ap.parse_args()

    D = workspace.data_root() / "dataset"
    state = S.merged_state(D / "state.jsonl")
    # сколько раз пара уже откладывалась (журнал append-only, считаем записи)
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
        print("режим --peerless: снимаем пары без пиров")
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
            # в режиме «мёртвых сидов» решает только наличие пиров: 0 пиров =
            # пара не докачается никогда, даже если прогресс большой
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
            # обычный режим (нехватка места): трогаем только слабо готовые
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
        print(f"  снято {k}: пиров нет, готовность {prog:.0%}, {sz / 1e6:.0f} MB"
              + ("" if base_exists else " (сырья уже не было)"))

    print(f"удалено каталогов {deleted} (освобождено {freed / 1e9:.1f} GB), "
          f"оставлено с прогрессом ≥{args.max_progress:.0%}: {kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
