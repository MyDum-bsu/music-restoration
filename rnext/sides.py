"""Resolving the directory of each side of a dataset pair (single source of truth).

The problem this fixes: the journal of early pairs stored neither `dir` nor
the names of the wanted files, and Transmission issues new ids on re-adds.
Because of that the audit reported "0 bytes on disk" where files existed, and
the loop could order a re-download of already downloaded material.

Directory resolution order:
  1) the journaled `srec["dir"]`, if it actually contains the wanted files;
  2) the current `downloadDir` of the torrent in Transmission (by its id);
  3) the pair's base directory + `/orig|/master`.
"""
from pathlib import Path

from rnext.transm import Transm

AUDIO_EXT = (".flac", ".ape", ".wv", ".wav", ".m4a", ".alac", ".aiff", ".aif")


def merge_sides(older: dict | None, newer: dict | None) -> dict | None:
    """Merge two side records: non-empty fields win.

    Early journal writers lost `dir`/`wanted_names`, so the "latest record"
    could end up poorer than the previous one. Hence the journal must be
    MERGED, not replaced.
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
    """Download journal with per-key record merging (not "last one wins")."""
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
             "leftUntilDone", "downloadedEver", "percentDone", "status",
             "peersConnected", "rateDownload"])}
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
    tmap = tmap if tmap is not None else torrents_by_id(t)
    tid = srec.get("torrent_tid")
    tor = tmap.get(tid) if tid is not None else None
    if tor and tor.get("downloadDir"):
        return Path(tor["downloadDir"])
    # the id may be stale (the torrent was removed and re-added) — search by name
    nm = srec.get("name")
    if nm:
        for x in tmap.values():
            d = x.get("downloadDir") or ""
            if x.get("name") == nm and "/dataset/raw/" in d:
                return Path(d)
    return base / side


def wanted_names(srec: dict | None, t: Transm | None = None,
                 tmap: dict[int, dict] | None = None,
                 base: Path | None = None, side: str | None = None) -> list[str]:
    """Names of the side's wanted files (journal -> torrent by id -> torrent by directory).

    The last step is needed because on re-adds Transmission issues a NEW id:
    the old one is no longer found in the journal, while the torrent contents
    (and file order) are the same.
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
    """How many of the wanted files are actually on disk, and their total size in bytes."""
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
