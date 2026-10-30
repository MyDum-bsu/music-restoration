#!/usr/bin/env python3
"""embed_lyrics.py — add lyrics to a music library in chunks (phone or local).

Per chunk (size-capped, memory-friendly): stage from the library -> transplant
lyrics from the local library (normalized artist+title match, no network) ->
resolve the rest via Genius (paced, cached) with lrclib fallback -> write tags
locally -> decode spot-check -> copy back ONLY changed files -> verify each.
Nomatch/instrumentals are never written back.

Deterministic: every ambiguity (AMBIGUOUS / NOMATCH / ERROR) is logged to
stragglers.tsv instead of guessed. Re-run to pick up only what's missing.

Usage:
  python3 embed_lyrics.py                    # whole library, resume-safe
  python3 embed_lyrics.py --dir "Helloween"  # one top-level group
  python3 embed_lyrics.py --force            # re-embed even if lyrics exist
Env/opts: --root (default auto-detected phone), CHUNK_BYTES, WORKERS
(default 1 — Genius rate-limits above ~1 req/s sustained), --skip-dir/--skip-file.
"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = Path(__file__).resolve().parents[3]
if not (_REPO / "rnext").is_dir():
    import os
    _REPO = Path(os.environ.get("MUSIC_RESTORATION_HOME", _REPO))
sys.path.insert(0, str(_REPO))
from rnext.phone import mtp
from rnext.phone.bootstrap import add_skill_scripts
from rnext.phone.config import ensure_dirs, music_dir, state_dir

sys.path.insert(0, str(add_skill_scripts("add-lyrics")))
import genius_lyrics as G

ap = argparse.ArgumentParser()
ap.add_argument("--dir", help="limit to this top-level dir under the music root")
ap.add_argument("--root", help="explicit music root (default: auto-detected phone)")
ap.add_argument("--local", help="local library for transplant (default MUSIC_DIR)")
ap.add_argument("--force", action="store_true", help="re-embed even if lyrics exist")
ap.add_argument("--skip-dir", action="append", default=[], help="top-level dir to skip (repeatable)")
ap.add_argument("--skip-file", action="append", default=[], help="file name to skip (repeatable)")
ap.add_argument("--workers", type=int, default=1, help="keep at 1-2: Genius rate-limits hard")
a = ap.parse_args()

ensure_dirs()
root = Path(a.root) if a.root else mtp.phone_dir()
if root is None or not root.exists():
    sys.exit("no music root found: connect the phone (MTP) or pass --root / set MR_PHONE_DIR")

CHUNK_BYTES = int(__import__("os").environ.get("CHUNK_BYTES", 4 * 1024**3))
STAGE = state_dir() / "stage"
LOCAL = Path(a.local) if a.local else music_dir()
WORKERS = a.workers
STATUS = state_dir() / "status.tsv"
CACHE_FILE = state_dir() / "genius_cache.json"

CACHE = {}


def log(msg):
    with open(STATUS, "a") as fh:
        fh.write(msg + "\n")


def norm(s):
    return G.norm(s)


def cache_load():
    global CACHE
    if CACHE_FILE.exists():
        CACHE = json.loads(CACHE_FILE.read_text())


def cache_save():
    CACHE_FILE.write_text(json.dumps(CACHE, ensure_ascii=False))


def local_lyric_map():
    from mutagen import File as MF
    from mutagen.mp4 import MP4
    m = {}
    files = set(LOCAL.rglob("*")) | set(LOCAL.iterdir())
    for p in files:
        if p.suffix.lower() not in mtp.AUDIO_EXTS or not p.is_file():
            continue
        try:
            f = MF(str(p), easy=True)
            if not f or not f.tags:
                continue
            ar = (f.get("artist") or [""])[0].strip()
            ti = (f.get("title") or [""])[0].strip()
            ly = f.get("lyrics") or f.get("unsyncedlyrics")
            if not ly and p.suffix.lower() in (".m4a", ".m4b", ".mp4"):
                ly = MP4(str(p)).get("\xa9lyr")
            if ar and ti and ly and ly[0].strip():
                m[(norm(ar), norm(ti))] = ly[0]
        except Exception:
            pass
    return m


LYRMAP = {}


def write_lyrics(p, text):
    from mutagen.flac import FLAC
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
    from mutagen.id3 import USLT
    ext = p.suffix.lower()
    if ext == ".flac":
        g = FLAC(str(p)); g["LYRICS"] = text; g["UNSYNCEDLYRICS"] = text; g.save()
    elif ext == ".mp3":
        g = MP3(str(p))
        if g.tags is None: g.add_tags()
        g.tags.setall("USLT", [USLT(encoding=3, lang="eng", desc="", text=text)]); g.save()
    elif ext in (".m4a", ".m4b", ".mp4"):
        g = MP4(str(p)); g["\xa9lyr"] = [text]; g.save()
    else:
        raise RuntimeError(f"unsupported ext {ext}")


def transplant_one(p):
    from mutagen import File as MF
    f = MF(str(p), easy=True)
    if not f or not f.tags:
        return False
    ar = (f.get("artist") or [""])[0].strip()
    ti = (f.get("title") or [""])[0].strip()
    ly = LYRMAP.get((norm(ar), norm(ti)))
    if not ly:
        return False
    write_lyrics(p, ly)
    return True


def has_lyrics(p):
    from mutagen import File as MF
    from mutagen.mp4 import MP4
    try:
        f = MF(str(p), easy=True)
        if not f or not f.tags:
            return False
        ly = f.get("lyrics") or f.get("unsyncedlyrics")
        if ly and ly[0].strip():
            return True
        if p.suffix.lower() in (".m4a", ".m4b", ".mp4") and MP4(str(p)).get("\xa9lyr"):
            return True
    except Exception:
        pass
    return False


def resolve_one(p):
    """Genius first (paced + cached), lrclib fallback. Returns (status, detail)."""
    artist, title, _ = G.read_artist_title(p)
    if not artist or not title:
        return ("ERROR", "missing artist/title tags")
    title = G.clean_search_title(title)
    artist = G.clean_search_title(artist) or artist

    qkey = "search:" + norm(artist + " " + title)
    cands = CACHE.get(qkey)
    if cands is None:
        try:
            cands = G.search_songs(f"{artist} {title}")
        except Exception as e:
            return ("ERROR", str(e))
        CACHE[qkey] = cands
    if not cands:
        fb = G.lrclib_fetch(artist, title)
        return ("OK-FB", fb) if fb else ("NOMATCH", "no genius candidates, lrclib empty")

    status, payload = G.decide(cands, artist, title)
    if status == "NOMATCH":
        fb = G.lrclib_fetch(artist, title)
        return ("OK-FB", fb) if fb else ("NOMATCH", None)
    if status == "AMBIGUOUS":
        return ("AMBIGUOUS", " ;; ".join(c["url"] for c in payload))

    url = payload["url"]
    lkey = "lyrics:" + url
    ly = CACHE.get(lkey)
    if ly is None:
        try:
            ly = G.fetch_lyrics(url)
        except Exception as e:
            return ("ERROR", f"fetch {url}: {e}")
        CACHE[lkey] = ly if len(ly) >= 40 else ""
    if not ly:
        return ("NOMATCH", url)
    return ("OK", ly)


def process_chunk(idx, items):
    name = f"chunk{idx:03d}"
    dst = STAGE / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    t0 = time.time()
    staged, relmap = [], {}
    for p in items:
        rel = p.relative_to(root)
        target = dst / rel
        ok, err = mtp.copy_from_phone(root, p, target, use_gio=True)
        if ok:
            staged.append(target); relmap[target] = rel
        else:
            log(f"COPY-ERR\t{rel}\t{err}")
    log(f"STAGED\t{name}\t{len(staged)}/{len(items)} in {time.time()-t0:.0f}s")
    if not staged:
        shutil.rmtree(dst, ignore_errors=True); return

    changed, strag, tr = [], [], 0
    for p in staged:
        try:
            if transplant_one(p):
                tr += 1; changed.append(p)
        except Exception as e:
            log(f"TRANSPLANT-ERR\t{p.relative_to(dst)}\t{e}")

    todo = [p for p in staged if not has_lyrics(p)]
    if todo:
        t0 = time.time()
        results = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for p, res in zip(todo, ex.map(resolve_one, todo)):
                results[p] = res
        log(f"RESOLVED\t{name}\t{len(todo)} files in {time.time()-t0:.0f}s")
        cache_save()
        for p, (status, detail) in results.items():
            rel = relmap[p]
            if status in ("OK", "OK-FB"):
                try:
                    write_lyrics(p, detail); changed.append(p)
                except Exception as e:
                    log(f"WRITE-ERR\t{rel}\t{e}")
            else:
                strag.append((status, rel, str(detail)))
                log(f"{status}\t{rel}\t{detail}")

    fl = [p for p in changed if p.suffix == ".flac"]
    if fl:
        d = mtp.run(["flac", "-t", "--totally-silent", str(fl[0])], timeout=900)
        log(f"DECODE-SAMPLE\t{name}\t{'OK' if d.returncode == 0 else 'FAIL'}")

    t0 = time.time()
    back = verified = 0
    for p in changed:
        rel = relmap[p]
        ok, err = mtp.copy_to_phone(root, p, rel)
        if not ok:
            log(f"BACK-ERR\t{rel}\t{err}"); continue
        back += 1
        if has_lyrics(root / rel): verified += 1
        else: log(f"BACK-NO-LYRICS\t{rel}")
    log(f"COPIED-BACK\t{name}\t{back}/{len(changed)} verified={verified} in {time.time()-t0:.0f}s")
    if strag:
        with open(STAGE / "stragglers.tsv", "a") as fh:
            for s in strag:
                fh.write("\t".join(s) + "\n")
    shutil.rmtree(dst, ignore_errors=True)


def main():
    global LYRMAP
    STAGE.mkdir(parents=True, exist_ok=True)
    for d in STAGE.glob("chunk*"):
        shutil.rmtree(d, ignore_errors=True)
    skip_dirs, skip_files = set(a.skip_dir), set(a.skip_file)
    items = []
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in mtp.AUDIO_EXTS):
            continue
        rel = p.relative_to(root)
        if rel.parts[0] in skip_dirs or rel.name in skip_files:
            continue
        if a.dir and rel.parts[0] != a.dir:
            continue
        if not a.force and has_lyrics(p):
            continue
        items.append(p)
    cache_load()
    LYRMAP = local_lyric_map()
    log(f"DRIVER-START\tfiles={len(items)} local_map={len(LYRMAP)} workers={WORKERS}")
    chunk, size, idx = [], 0, 0
    for p in items:
        sz = p.stat().st_size
        if chunk and (size + sz > CHUNK_BYTES or len(chunk) >= 120):
            idx += 1
            log(f"CHUNK-BEGIN\t{idx}\t{len(chunk)} files")
            try: process_chunk(idx, chunk)
            except Exception as e: log(f"CHUNK-FAIL\t{idx}\t{e}")
            chunk, size = [], 0
        chunk.append(p); size += sz
    if chunk:
        idx += 1
        log(f"CHUNK-BEGIN\t{idx}\t{len(chunk)} files")
        try: process_chunk(idx, chunk)
        except Exception as e: log(f"CHUNK-FAIL\t{idx}\t{e}")
    log("DRIVER-END")


if __name__ == "__main__":
    main()
