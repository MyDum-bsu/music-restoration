"""Разрешение каталогов сторон пары датасета (единый источник правды).

Проблема, которую это лечит: в журнале у ранних пар не сохранялись ни `dir`,
ни имена нужных файлов, а Transmission при повторных добавлениях выдаёт новые
id. Из-за этого аудит показывал «0 байт на диске» там, где файлы лежали, и
цикл мог заказать повторную загрузку уже скачанного.

Порядок разрешения каталога:
  1) записанный в журнале `srec["dir"]`, если в нём реально есть нужные файлы;
  2) текущий `downloadDir` раздачи в Transmission (по её id);
  3) базовый каталог пары + `/orig|/master`.
"""
from pathlib import Path

from rnext.transm import Transm

AUDIO_EXT = (".flac", ".ape", ".wv", ".wav", ".m4a", ".alac", ".aiff", ".aif")


def merge_sides(older: dict | None, newer: dict | None) -> dict | None:
    """Слить две записи стороны: непустые поля побеждают.

    Ранние писатели журнала теряли `dir`/`wanted_names`, и «последняя запись»
    оказывалась беднее предыдущей. Поэтому журнал надо СЛИВАТЬ, а не замещать.
    """
    if not isinstance(older, dict):
        return newer if isinstance(newer, dict) else None
    if not isinstance(newer, dict):
        return older
    out = dict(older)
    for k, v in newer.items():
        if v not in (None, "", [], {}):
            out[k] = v
    return out


def merged_state(path) -> dict[str, dict]:
    """Журнал загрузок с по-ключевым слиянием записей (не «последняя побеждает»)."""
    import json
    from pathlib import Path as _P
    path = _P(path)
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        k = r.get("key")
        if k not in out:
            out[k] = dict(r)
            continue
        cur = out[k]
        for f, v in r.items():
            if f in ("orig", "master"):
                cur[f] = merge_sides(cur.get(f), v)
            elif v not in (None, "", [], {}):
                cur[f] = v
    return out






def torrents_by_id(t: Transm | None = None) -> dict[int, dict]:
    try:
        t = t or Transm()
        return {x["id"]: x for x in t.torrents(
            ["id", "name", "downloadDir", "files", "wanted", "totalSize",
             "leftUntilDone", "downloadedEver", "percentDone", "status"])}
    except Exception:                                     # noqa: BLE001
        return {}


def resolve_side_dir(srec: dict | None, side: str, base: Path,
                     t: Transm | None = None, tmap: dict[int, dict] | None = None) -> Path:
    if not isinstance(srec, dict):
        return base / side
    d = srec.get("dir")
    if d:
        p = Path(d)
        if p.is_dir():
            return p
    tid = srec.get("torrent_tid")
    if tid is not None:
        tmap = tmap if tmap is not None else torrents_by_id(t)
        tor = tmap.get(tid)
        if tor and tor.get("downloadDir"):
            return Path(tor["downloadDir"])
    return base / side


def wanted_names(srec: dict | None, t: Transm | None = None,
                 tmap: dict[int, dict] | None = None,
                 base: Path | None = None, side: str | None = None) -> list[str]:
    """Имена нужных файлов стороны (журнал → раздача по id → раздача по каталогу).

    Последний шаг нужен потому, что при повторных добавлениях Transmission
    выдаёт НОВЫЙ id: старый в журнале уже не находится, а содержимое раздачи
    (и порядок файлов) то же самое.
    """
    if not isinstance(srec, dict):
        return []
    names = srec.get("wanted_names") or []
    if names:
        return list(names)
    idx = srec.get("wanted") or []
    if not idx:
        return []
    tmap = tmap if tmap is not None else torrents_by_id(t)
    tor = tmap.get(srec.get("torrent_tid"))
    if not tor and base is not None and side:
        target = str(resolve_side_dir(srec, side, base, tmap=tmap)).rstrip("/")
        for x in tmap.values():
            d = (x.get("downloadDir") or "").rstrip("/")
            if d and d == target:
                tor = x
                break
    if not tor:
        return []
    fs = tor.get("files") or []
    return [fs[i]["name"] for i in idx if 0 <= i < len(fs)]


def side_presence(srec: dict | None, side: str, base: Path,
                  t: Transm | None = None, tmap: dict[int, dict] | None = None) -> dict:
    """Сколько из нужных файлов реально на диске и сколько это байт."""
    root = resolve_side_dir(srec, side, base, t, tmap)
    names = wanted_names(srec, t, tmap, base=base, side=side)
    present = []
    for n in names:
        p = root / n
        if p.is_file() and p.stat().st_size > 0:
            present.append(p)
    total = sum(p.stat().st_size for p in present)
    return {"dir": str(root), "wanted": len(names), "present": len(present),
            "bytes": total, "complete": bool(names) and len(present) == len(names)}
