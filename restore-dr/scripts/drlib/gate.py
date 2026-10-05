"""The delivery gate: nothing broken, tagless, coverless, or truncated gets pushed.

Rules (each exists because its violation really happened):
- flac -t integrity on every output file;
- artist/album/title/tracknumber non-empty on every file;
- embedded cover present;
- output duration == input duration (±50 ms) and same sample rate — a 48 kHz
  source once came out 8.8% longer and pitched down while all "loudness"
  metrics looked fine;
- every push is verified by reading the file BACK (gio cat) and comparing md5;
- any failure = loud QUARANTINE of the whole album, never a silent skip.
"""
import re
import shutil
from pathlib import Path

from .audio import duration_and_fs
from .tags import has_cover
from .util import md5f, run

DURATION_TOL_S = 0.05


def flac_ok(f) -> bool:
    return run(["flac", "-t", "-s", str(f)]).returncode == 0


def tags_problems(f) -> list[str]:
    from mutagen.flac import FLAC
    t = FLAC(str(f)).tags or {}
    return [k for k in ("artist", "album", "title", "tracknumber")
            if not t.get(k) or not str(t[k][0]).strip()]


def gate_album(dst_dir: Path, src_dir: Path | None = None) -> list[str]:
    """Return a list of problems; empty list = album may be pushed."""
    problems = []
    for f in sorted(dst_dir.glob("*.flac")):
        if not flac_ok(f):
            problems.append(f"{f.name}: corrupt decode")
            continue
        for k in tags_problems(f):
            problems.append(f"{f.name}: missing {k}")
        if not has_cover(f):
            problems.append(f"{f.name}: no embedded cover")
        dur, fs = duration_and_fs(f)
        if src_dir is not None:
            src = src_dir / f.name
            if src.exists():
                sdur, sfs = duration_and_fs(src)
                if fs != sfs:
                    problems.append(f"{f.name}: samplerate {fs} != source {sfs}")
                elif abs(dur - sdur) > DURATION_TOL_S:
                    problems.append(f"{f.name}: duration {dur:.2f}s != source {sdur:.2f}s")
    return problems


def slugify(rel: str) -> str:
    return re.sub(r"[^\w.-]", "_", rel)[-60:]


def quarantine(album_dir: Path, workroot: Path) -> Path:
    q = workroot / "quarantine" / slugify(album_dir.name)
    q.parent.mkdir(parents=True, exist_ok=True)
    if q.exists():
        shutil.rmtree(q)
    shutil.move(str(album_dir), str(q))
    return q


def sweep_workroot(workroot: Path) -> None:
    """End-of-run cleanup: wipe everything except quarantine/ (gate-failed dst
    is moved there and its source is still pullable, since nothing was pushed)."""
    for p in workroot.iterdir():
        if p.name == "quarantine":
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
