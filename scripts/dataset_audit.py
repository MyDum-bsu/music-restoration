#!/usr/bin/env python3
"""Аудит датасета: не потерялось ли скачанное по дороге.

Сверяет три источника правды:
  1. журнал загрузок `dataset/state.jsonl` — что заказывали и сколько хотели;
  2. Transmission — сколько реально ушло трафика и что лежит на диске;
  3. `dataset/pairs/*/meta.json` + `features.json` — что извлечено в датасет.

Печатает аномалии:
  * `downloaded_not_extracted` — раздача стояла, а пары в датасете нет;
  * `present_no_state`        — каталог с аудио, не связанный ни с одной парой;
  * `partial`                 — не все нужные файлы на месте;
  * `extracted_no_raw`        — норма (сырьё удалено после извлечения);
  * `no_tracks`/`edition_mismatch`/`no_match` — отбракованные пары (ожидаемо).

  python3 scripts/dataset_audit.py
  python3 scripts/dataset_audit.py --json out.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import sides as S  # noqa: E402
from rnext import sides as S  # noqa: E402
from rnext import workspace  # noqa: E402
from rnext.transm import Transm  # noqa: E402

AUDIO = (".flac", ".ape", ".wv", ".wav", ".m4a", ".alac", ".aiff", ".aif")


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_audio(root: Path) -> tuple[int, float]:
    n = 0
    b = 0
    if not root.is_dir():
        return 0, 0.0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO:
            n += 1
            b += p.stat().st_size
    return n, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    uniq = S.merged_state(dataset_dir() / "state.jsonl")

    # что реально пришло в Transmission
    tors = S.torrents_by_id()

    ours = {tid: x for tid, x in tors.items()
            if "/restore/dataset/raw/" in (x.get("downloadDir") or "")}
    traffic = sum(x.get("downloadedEver") or 0 for x in ours.values())

    pairs_root = dataset_dir() / "pairs"
    extracted = {}
    for d in sorted(pairs_root.glob("*/")):
        meta = d / "meta.json"
        feats = d / "features.json"
        if not feats.is_file():
            continue
        rows = json.loads(feats.read_text(encoding="utf-8"))
        m = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        extracted[d.name.replace("__", "/")] = {"tracks": len(rows), "meta": m}

    # разбор по парам
    rows = []
    used_dirs: set[str] = set()
    for key, r in sorted(uniq.items()):
        sides = {}
        want = 0
        have = 0
        for side in ("orig", "master"):
            s = r.get(side) if isinstance(r.get(side), dict) else None
            if not s:
                continue
            base = Path(r.get("dir") or "")
            pres = S.side_presence(s, side, base, tmap=tors)
            used_dirs.add(pres["dir"])
            w = s.get("wanted_bytes") or 0
            want += w
            have += pres["bytes"]
            sides[side] = {"dir": pres["dir"], "wanted": pres["wanted"],
                           "present": pres["present"], "bytes_wanted": w,
                           "bytes": pres["bytes"]}
        st = r.get("state")
        ex = extracted.get(key)
        if ex:
            status = "extracted"
        elif st in ("no_match", "error"):
            status = "rejected" if st == "no_match" else "error"
        elif want and have < want * 0.98:
            status = "partial"
        elif want:
            status = "downloaded_not_extracted"
        else:
            status = "no_files"
        rows.append({"key": key, "status": status, "artist": r.get("artist"),
                     "album": r.get("album"), "sides": sides,
                     "bytes_wanted": want, "bytes_present": have,
                     "extracted_tracks": ex["tracks"] if ex else 0,
                     "match_fraction": (ex["meta"].get("match_fraction") if ex else None)})

    # каталоги с аудио, которых нет ни в одной записи журнала
    present_dirs = defaultdict(lambda: [0, 0.0])
    raw = dataset_dir() / "raw"
    if raw.is_dir():
        for d in raw.iterdir():
            if d.is_dir():
                n, b = dir_audio(d)
                present_dirs[str(d)] = [n, b]

    orphans = {d: v for d, v in present_dirs.items()
               if not any(str(d).startswith(u) or u.startswith(str(d)) for u in used_dirs if u)}

    # --- отчёт
    by_status = Counter(r["status"] for r in rows)
    total_want = sum(r["bytes_wanted"] for r in rows)
    total_present = sum(r["bytes_present"] for r in rows)
    tracks = sum(v["tracks"] for v in extracted.values())
    print(f"пар в журнале: {len(rows)}   состояние: {dict(by_status)}")
    print(f"извлечено в датасет: {len(extracted)} пар / {tracks} треков")
    print(f"заказано (wanted): {human(total_want)}   на диске сейчас: {human(total_present)}")
    print(f"трафик по нашим раздачам (downloadedEver): {human(traffic)}")
    print(f"раздач в Transmission по датасету: {len(ours)}; "
          f"не завершены: {sum(1 for x in ours.values() if (x.get('leftUntilDone') or 0) > 0)}")

    anom = [r for r in rows if r["status"] in ("downloaded_not_extracted", "partial", "error")]
    if anom:
        print("\nАНОМАЛИИ:")
        for r in anom:
            print(f"  {r['status']:<26} {r['key']:<16} "
                  f"{str(r['artist'])[:20]:<20} {str(r['album'])[:30]:<30} "
                  f"было {human(r['bytes_present'])} из {human(r['bytes_wanted'])}")
    if orphans:
        print("\nКАТАЛОГИ С АУДИО ВНЕ ЖУРНАЛА:")
        for d, (n, b) in orphans.items():
            print(f"  {d} — {n} файлов, {human(b)}")

    rejected = [r for r in rows if r["status"] == "rejected"]
    if rejected:
        print(f"\nОтбраковано при сверке файлов (ожидаемо, данные не нужны): {len(rejected)}")
        for r in rejected[:8]:
            print(f"  {r['key']:<16} {str(r['artist'])[:22]:<22} {str(r['album'])[:30]}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"pairs": rows, "by_status": dict(by_status), "traffic_bytes": traffic,
             "extracted_pairs": len(extracted), "extracted_tracks": tracks,
             "orphan_dirs": {d: v for d, v in orphans.items()}},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\njson → {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
