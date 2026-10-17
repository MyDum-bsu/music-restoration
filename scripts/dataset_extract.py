#!/usr/bin/env python3
"""Извлечение датасета: сырая пара → фичи + вырезки → трёхуровневое хранилище.

Уровни (см. docs/dataset.md):
  1) фичи   dataset/pairs/<key>/features.jsonl + dataset/features/all.jsonl (~1.5 КБ/трек)
  2) вырезки dataset/pairs/<key>/excerpts/*.flac (громкое место + середина)
  3) сырьё  dataset/raw/<key>/ — удаляется после извлечения (--keep-raw оставляет)

Стороны читаются в любом виде: per-track FLAC, образ+cue (в т.ч. несколько
дисков), APE/WV. Треки сопоставляются по названию и длительности.

  python3 scripts/dataset_extract.py --all [--limit N]
  python3 scripts/dataset_extract.py --key 5412248/rel0 [--keep-raw] [--excerpts 0]
  python3 scripts/dataset_extract.py --prune          # снести сырьё у готовых пар
"""
import argparse
import json
import os
import math
import re
import shutil
import subprocess
import sys
from difflib import SequenceMatcher as SM
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import features as F  # noqa: E402
from rnext import workspace  # noqa: E402
from rnext.cue import parse_cue, track_bounds  # noqa: E402

AUDIO_EXT = (".flac", ".ape", ".wv", ".wav", ".alac", ".m4a", ".aiff", ".aif")
IMAGE_MIN = 60_000_000          # >60 МБ — скорее образ, чем трек


# --- пути -------------------------------------------------------------------
def dataset_dir() -> Path:
    p = workspace.data_root() / "dataset"
    for sub in ("raw", "pairs", "features"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def append_lines(p: Path, rows: list[dict]) -> None:
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_flac(x: np.ndarray, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "2",
                    "-i", "-", "-c:a", "flac", "-compression_level", "5", "-y", str(path)],
                   input=x.astype(np.float32).tobytes(), check=True)


# --- разбор стороны ---------------------------------------------------------
class UnreadableAudio(RuntimeError):
    """Файл не читается ffprobe (битый/недокачанный) — пару нельзя извлекать."""

    def __init__(self, msg: str, path: Path | None = None):
        super().__init__(msg)
        self.path = path


def _ffdur(p: Path) -> float:
    try:
        d = F.ffprobe_audio(str(p)).get("dur")
    except Exception as e:                                 # noqa: BLE001
        raise UnreadableAudio(f"{p.name}: {type(e).__name__} {str(e)[:80]}", p) from e
    if not d or d <= 0:
        raise UnreadableAudio(f"{p.name}: нет длительности (битый файл)", p)
    return float(d)


def repair_corrupt(path: Path) -> dict:
    """Удалить битый файл и заставить раздачу перекачать его.

    `torrent-verify` на этом Transmission сам по себе не помогает (resume-данные
    считают куски целыми), поэтому файл удаляем — тогда проверка видит пропажу
    и докачивает именно его.
    """
    from rnext import sides as S
    from rnext.transm import Transm
    out = {"file": str(path), "deleted": False, "torrent": None}
    try:
        if path.is_file():
            path.unlink()
            out["deleted"] = True
    except Exception as e:                                 # noqa: BLE001
        out["error"] = str(e)[:80]
        return out
    tmap = S.torrents_by_id()
    tstr = str(path)
    for tid, tor in tmap.items():
        d = (tor.get("downloadDir") or "").rstrip("/")
        if d and tstr.startswith(d + "/"):
            try:
                Transm()._call("torrent-verify", {"ids": [tid]})
                Transm()._call("torrent-start", {"ids": [tid]})
                out["torrent"] = tid
            except Exception as e:                         # noqa: BLE001
                out["error"] = str(e)[:80]
            break
    return out


def verify_torrent_for(root: Path) -> int | None:
    """Перепроверить раздачу, в чьём каталоге лежит битый файл (докачает кусок)."""
    from rnext import sides as S
    try:
        tmap = S.torrents_by_id()
    except Exception:                                      # noqa: BLE001
        return None
    target = str(root).rstrip("/")
    for tid, tor in tmap.items():
        d = (tor.get("downloadDir") or "").rstrip("/")
        if d and (target == d or target.startswith(d + "/") or d.startswith(target + "/")):
            from rnext.transm import Transm
            try:
                Transm()._call("torrent-verify", {"ids": [tid]})
            except Exception:                              # noqa: BLE001
                return None
            return tid
    return None


def _track_no(name: str) -> int | None:
    m = re.match(r"^\s*(\d{1,3})\s*[-._)]", name)
    if m:
        return int(m.group(1))
    m = re.search(r"^\s*(?:cd|disc|disk)\s*(\d)\D+(\d{1,2})", name, re.IGNORECASE)
    return int(m.group(2)) if m else None


def _clean_title(name: str) -> str:
    t = Path(name).stem
    t = re.sub(r"^\s*(?:\d{1,3}\s*[-._)]\s*)", "", t)
    t = re.sub(r"^\s*(?:cd|disc|disk)\s*\d\s*[-._)]?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[\[\(].*?[\]\)]", " ", t)
    return re.sub(r"[^a-zа-яё0-9]+", " ", t.lower()).strip()


def side_tracks(root: Path) -> list[dict]:
    """Треки стороны: {n, title, file, start_s, dur_s} — файлы или срезы образа."""
    files = [p for p in sorted(root.rglob("*")) if p.is_file()
             and p.suffix.lower() in AUDIO_EXT]
    big = [p for p in files if p.stat().st_size > IMAGE_MIN]
    small = [p for p in files if p not in big]
    out: list[dict] = []

    if big:
        cues = {p.stem.lower(): p for p in root.rglob("*.cue")}
        for img in sorted(big):
            cue = cues.get(img.stem.lower())
            # Длинный трек (10+ мин) может весить >60 МБ. Образом считаем только
            # файл с cue или единственный аудиофайл в своей папке.
            siblings = [q for q in img.parent.glob("*") if q.suffix.lower() in AUDIO_EXT]
            if cue is None and len(siblings) > 1:
                small.append(img)
                continue
            if cue is None:
                others = list(root.rglob("*.cue"))
                cue = others[0] if len(others) == 1 else None
            if cue is None:
                out.append({"n": None, "title": _clean_title(img.name), "file": str(img),
                            "start_s": 0.0, "dur_s": _ffdur(img)})
                continue
            try:
                meta = F.ffprobe_audio(str(img))
            except Exception as e:                         # noqa: BLE001
                raise UnreadableAudio(f"{img.name}: {type(e).__name__} {str(e)[:70]}") from e
            total = int(round(meta["dur"] * meta["sr"]))
            cue_tracks = parse_cue(str(cue), file_key=img.name)
            if not cue_tracks:
                # FILE в cue может не совпадать с именем образа (CDImage.ape и т.п.)
                cue_tracks = parse_cue(str(cue))
            if not cue_tracks:
                out.append({"n": None, "title": _clean_title(img.name), "file": str(img),
                            "start_s": 0.0, "dur_s": _ffdur(img)})
                continue
            for i, (st_, en_) in enumerate(track_bounds(cue_tracks, total, meta["sr"])):
                ct = cue_tracks[i]
                out.append({"n": ct.get("n") or i + 1,
                            "title": _clean_title(ct.get("title") or ""),
                            "file": str(img), "start_s": st_ / meta["sr"],
                            "dur_s": (en_ - st_) / meta["sr"]})
    for p in small:
        out.append({"n": _track_no(p.name), "title": _clean_title(p.name),
                    "file": str(p), "start_s": 0.0, "dur_s": _ffdur(p)})
    # нумерация дисков сквозная: cd2/01 → 101, если в имени есть диск
    return out


def match_sides(a: list[dict], b: list[dict]) -> list[tuple[int, int, float]]:
    """Сопоставление треков двух сторон: (i, j, уверенность)."""
    pairs = []
    for i, x in enumerate(a):
        best, best_s = None, 0.0
        for j, y in enumerate(b):
            if not x["title"] or not y["title"]:
                s = 0.0
            else:
                s = SM(None, x["title"], y["title"]).ratio()
            dd = abs(x["dur_s"] - y["dur_s"])
            if dd <= 2.0:
                s += 0.35
            elif dd <= 5.0:
                s += 0.15
            elif dd > 12.0:
                s -= 0.4
            if x["n"] and y["n"] and x["n"] == y["n"]:
                s += 0.2
            if s > best_s:
                best, best_s = j, s
        pairs.append((i, best, best_s))
    used: set[int] = set()
    out = []
    for i, j, s in sorted(pairs, key=lambda t: -t[2]):
        if j is None or j in used or s < 0.55:
            continue
        used.add(j)
        out.append((i, j, round(s, 3)))
    return sorted(out)


# --- фичи одного трека ------------------------------------------------------
def track_features(t: dict, sr: int | None = None) -> dict:
    x, sr = F.decode_f32(t["file"], t["start_s"], t["dur_s"])
    x16, _ = F.decode_s16(t["file"], t["start_s"], t["dur_s"])
    return features_from_array(x, sr, x16)


def features_from_array(x: np.ndarray, sr: int, x16: np.ndarray | None = None) -> dict:
    """Тот же набор фич, но из массива (для прогона пайплайна в памяти)."""
    return F.feature_dict(x, sr, x16)


def delta_of(o: dict, m: dict) -> dict:
    bands = [round(m["bands_db"][k] - o["bands_db"][k], 3) for k in range(len(o["bands_db"]))]
    d = {
        "dr": m["dr"] - o["dr"],
        "dr_exact": round(m["dr_exact"] - o["dr_exact"], 2),
        "crest": round(m["crest_db"] - o["crest_db"], 3),
        "rms": round(m["rms_db"] - o["rms_db"], 3),
        "peak": round(m["peak_db"] - o["peak_db"], 3),
        "clip_samples": (m["clip"]["L"]["clipped_samples"] + m["clip"]["R"]["clipped_samples"])
        - (o["clip"]["L"]["clipped_samples"] + o["clip"]["R"]["clipped_samples"]),
        "bands_db": bands,
        "spread": (m.get("blocks", {}).get("spread", 0) or 0) - (o.get("blocks", {}).get("spread", 0) or 0),
        "attacks_per_s": round((m.get("transient", {}).get("attacks_per_s", 0) or 0)
                               - (o.get("transient", {}).get("attacks_per_s", 0) or 0), 3),
        "side_mid_db": round((m.get("stereo", {}).get("side_mid_db", 0) or 0)
                             - (o.get("stereo", {}).get("side_mid_db", 0) or 0), 2),
        "tp_overshoot_db": round((m.get("tp_overshoot_db") or 0) - (o.get("tp_overshoot_db") or 0), 2),
    }
    return d


# --- вырезки ----------------------------------------------------------------
def make_excerpt(t_orig: dict, t_mast: dict, out_dir: Path, tag: str,
                 mark_fracs=(0.1, 0.5), win_s: float = 12.0) -> list[str]:
    """Вырезки обеих сторон, выровненные по громкости (−20 dBFS RMS)."""
    made = []
    for frac in mark_fracs:
        for side, t in (("orig", t_orig), ("master", t_mast)):
            dur = min(win_s, t["dur_s"])
            if dur < 4:
                continue
            start = t["start_s"] + max(0.0, (t["dur_s"] - dur) * frac)
            x, sr = F.decode_f32(t["file"], start, dur)
            rms = math.sqrt(float(np.mean(x.astype(np.float64) ** 2)))
            if rms <= 1e-9:
                continue
            x = x * (10 ** (-20.0 / 20.0) / rms)
            x = np.clip(x, -0.999, 0.999)
            p = out_dir / f"{tag}_{int(frac * 100):02d}_{side}.flac"
            write_flac(x, sr, p)
            made.append(p.name)
    return made


# --- обработка пары ---------------------------------------------------------
def side_root(rec: dict, side: str, base: Path) -> Path | None:
    """Каталог стороны: журнал → downloadDir в Transmission → base/<side>."""
    from rnext import sides as S
    srec = rec.get(side) if isinstance(rec.get(side), dict) else None
    p = S.resolve_side_dir(srec, side, base)
    return p if p.is_dir() else None


def wanted_names(srec: dict | None) -> list[str] | None:
    """Имена файлов, нужных этой паре (журнал → список файлов раздачи)."""
    from rnext import sides as S
    names = S.wanted_names(srec)
    return names or None


def side_ready(srec: dict | None, root: Path, base: Path | None = None,
               side: str | None = None) -> bool:
    """Готовность стороны: все wanted-файлы на месте и без *.part.

    Проверяем именно файлы ЭТОЙ пары, а не всю папку темы: у общей дискографии
    в одном каталоге лежат альбомы разных пар, и чужие *.part нас не касаются.
    """
    from rnext import sides as S
    names = S.wanted_names(srec, base=base, side=side)
    if names:
        return all((root / n).is_file() and (root / n).stat().st_size > 0 for n in names)
    # если имён совсем нет — грубая проверка, но только по аудио и не по чужим файлам
    return not any(p.suffix.lower() == ".part" for p in root.rglob("*.part")
                   if p.with_suffix("").suffix.lower() in AUDIO_EXT)


def dir_refcounts() -> dict[str, int]:
    """Сколько пар ссылаются на каждый каталог стороны.

    Каталог общей темы-дискографии один на несколько пар: удалять его после
    извлечения первой пары нельзя — снесём данные остальных.
    """
    refs: dict[str, int] = {}
    from rnext import sides as S
    seen = S.merged_state(dataset_dir() / "state.jsonl")
    for r in seen.values():
        for side in ("orig", "master"):
            srec = r.get(side) if isinstance(r.get(side), dict) else None
            d = (srec or {}).get("dir")
            if d:
                refs[d] = refs.get(d, 0) + 1
    return refs


def cleanup_raw(rec: dict, refs: dict[str, int]) -> None:
    """Удалить сырьё пары, кроме каталогов, нужных другим парам."""
    import shutil as _sh
    for side in ("orig", "master"):
        srec = rec.get(side) if isinstance(rec.get(side), dict) else None
        d = (srec or {}).get("dir")
        if not d:
            continue
        if refs.get(d, 0) > 1:
            continue
        _sh.rmtree(d, ignore_errors=True)
    base = rec.get("dir")
    if base and Path(base).is_dir():
        try:
            next(Path(base).iterdir())
        except StopIteration:
            _sh.rmtree(base, ignore_errors=True)


def extract_pair(rec: dict, excerpts: int, keep_raw: bool, refs: dict[str, int] | None = None) -> dict:
    key = rec["key"]
    base = Path(rec["dir"]) if rec.get("dir") else Path("")
    o_dir = side_root(rec, "orig", base)
    m_dir = side_root(rec, "master", base)
    if o_dir is None or m_dir is None:
        return {"key": key, "state": "not_ready"}
    osrec = rec.get("orig") if isinstance(rec.get("orig"), dict) else None
    msrec = rec.get("master") if isinstance(rec.get("master"), dict) else None
    if not side_ready(osrec, o_dir, base, "orig") or not side_ready(msrec, m_dir, base, "master"):
        return {"key": key, "state": "not_ready"}
    # если альбом лежал внутри дискографии — берём только его папку
    for side, d in (("orig", o_dir), ("master", m_dir)):
        sub = ((rec.get(side) or {}).get("album_dir") if isinstance(rec.get(side), dict) else None)
        if sub:
            cand = base / side / sub
            if cand.is_dir():
                if side == "orig":
                    o_dir = cand
                else:
                    m_dir = cand
    if not o_dir.is_dir():
        return {"key": key, "state": "missing_orig"}
    if not m_dir.is_dir():
        return {"key": key, "state": "missing_master"}
    try:
        o_tr, m_tr = side_tracks(o_dir), side_tracks(m_dir)
    except UnreadableAudio as e:
        fix = repair_corrupt(e.path) if e.path else {}
        return {"key": key, "state": "corrupt", "reason": str(e)[:160], "repair": fix}
    if not o_tr or not m_tr:
        return {"key": key, "state": "no_tracks",
                "orig_tracks": len(o_tr), "master_tracks": len(m_tr)}
    pairs = match_sides(o_tr, m_tr)
    frac = len(pairs) / max(1, min(len(o_tr), len(m_tr)))
    # Проверка издания по длительностям: акапеллы/инструменталы/лайвы и другие
    # альтернативные издания совпадают по названиям, но не по длительности.
    durs = [abs(o_tr[i]["dur_s"] - m_tr[j]["dur_s"]) for i, j, _ in pairs]
    med_ddur = float(np.median(durs)) if durs else 0.0
    if med_ddur > 4.0:
        return {"key": key, "state": "edition_mismatch", "median_ddur_s": round(med_ddur, 1),
                "matched": len(pairs), "of": min(len(o_tr), len(m_tr))}

    out_dir = dataset_dir() / "pairs" / key.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, j, s in pairs:
        o = track_features(o_tr[i])
        m = track_features(m_tr[j])
        rows.append({
            "key": key, "track": o_tr[i]["n"] or i + 1,
            "title": o_tr[i]["title"], "match": s, "oi": i, "mi": j,
            "artist": rec.get("artist"), "album": rec.get("album"),
            "year": rec.get("year"), "author_pid": rec.get("author_pid"),
            "original": o, "restored": m, "delta": delta_of(o, m),
        })
    if excerpts and rows:
        loudest = max(rows, key=lambda r: r["delta"]["crest"])
        mid = rows[len(rows) // 2]
        ex_dir = out_dir / "excerpts"
        for tag, row in (("loud", loudest), ("mid", mid)):
            make_excerpt(o_tr[row["oi"]], m_tr[row["mi"]], ex_dir,
                         f"t{row['track']:02d}_{tag}")
    write_json(out_dir / "features.json", rows)
    append_lines(dataset_dir() / "features" / "all.jsonl", rows)
    meta = {"key": key, "artist": rec.get("artist"), "album": rec.get("album"),
            "year": rec.get("year"), "author_pid": rec.get("author_pid"),
            "orig_topic_id": rec.get("orig_topic_id"), "master_topic_id": rec.get("master_topic_id"),
            "matched_tracks": len(pairs), "orig_tracks": len(o_tr),
            "master_tracks": len(m_tr), "match_fraction": round(frac, 3),
            "master_dr_published": rec.get("master_dr")}
    write_json(out_dir / "meta.json", meta)
    if not keep_raw:
        cleanup_raw(rec, refs or {})
    return {"key": key, "state": "ok", "matched": len(pairs), "of": min(len(o_tr), len(m_tr)),
            "frac": round(frac, 3), "dir": str(out_dir)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--key", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--excerpts", type=int, default=1)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--force", action="store_true", help="переизвлечь уже готовые пары")
    ap.add_argument("--keep-raw-window", type=int, default=0,
                    help="с --keep-raw: держать сырьё только у N последних пар")
    ap.add_argument("--prune", action="store_true", help="снести сырьё у готовых пар")
    args = ap.parse_args()

    if args.prune:
        n = 0
        for meta in (dataset_dir() / "pairs").glob("*/meta.json"):
            raw = dataset_dir() / "raw" / meta.parent.name
            if raw.is_dir():
                shutil.rmtree(raw, ignore_errors=True)
                n += 1
        print(f"prune: удалено сырьё у {n} пар")
        return 0

    from rnext import sides as S
    seen = S.merged_state(dataset_dir() / "state.jsonl")
    rows = [r for r in seen.values() if r.get("state") == "downloading" and r.get("dir")]
    if args.key:
        rows = [r for r in rows if r["key"] == args.key]
    elif not args.all:
        ap.error("нужен --all или --key")
    done = {p.parent.name for p in (dataset_dir() / "pairs").glob("*/meta.json")}
    refs = dir_refcounts()
    if not args.force:
        rows = [r for r in rows if r["key"].replace("/", "__") not in done]
    if args.limit:
        rows = rows[:args.limit]
    print(f"пар к извлечению: {len(rows)}")
    for r in rows:
        try:
            res = extract_pair(r, args.excerpts, args.keep_raw, refs)
        except Exception as e:                          # noqa: BLE001
            res = {"key": r["key"], "state": "error", "reason": f"{type(e).__name__}: {e}"[:200]}
        print(" ", json.dumps(res, ensure_ascii=False))

    if args.keep_raw and args.keep_raw_window > 0:
        prune_raw_window(args.keep_raw_window)
    return 0


def prune_raw_window(window: int) -> None:
    """Оставляем сырьё только у `window` последних извлечённых пар.

    Сырьё нужно для end-to-end оценки (dataset_eval), но 40 GiB не резиновые:
    старые пары очищаются, их место занимают новые.
    """
    metas = sorted((dataset_dir() / "pairs").glob("*/meta.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    state = state_map()
    refs = dir_refcounts()
    dropped = 0
    for m in metas[window:]:
        key = m.parent.name.replace("__", "/")
        rec = state.get(key)
        if not rec:
            continue
        base = Path(rec.get("dir") or "")
        if not base.is_dir():
            continue
        # не трогаем каталоги, которые ещё нужны другим парам
        shared = any(refs.get((rec.get(side) or {}).get("dir", ""), 0) > 1
                     for side in ("orig", "master"))
        if shared:
            continue
        cleanup_raw(rec, refs)
        dropped += 1
    if dropped:
        print(f"  очищено сырьё у {dropped} старых пар (окно {window})")


if __name__ == "__main__":
    sys.exit(main())
