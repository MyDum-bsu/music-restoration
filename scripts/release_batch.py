#!/usr/bin/env python3
"""Batch run of releases: restoration → finalization → phone, one by one.

  python3 scripts/release_batch.py \
      --job "release_src/City and Colour/2005 - Sometimes|City and Colour/2005 - Sometimes|City and Colour - Sometimes (2005)" \
      --job "..." \
      --log /tmp/batch.log --clean-src

Each job: "source directory | path on the phone | release name". Steps:
  1. if the source is image+cue — split into tracks (the image moves to _image/);
  2. restore_one.py with production settings (auto-policy, model, TP −0.5);
  3. finish_release.py — tags, cover, lyrics;
  4. phone_sync.py — replace the album on the phone with md5 verification;
  5. --clean-src — delete the source (free up space; the phone already has it).

Nothing "guesses": if there is no cover, the release ships without one and it
lands in the report; if lyrics weren't found — also in the report.
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

SCRIPTS = Path(__file__).resolve().parent


def _data_root() -> Path:
    """Corpus/release directory: config/workspace.json, else $RESTORE_DATA_ROOT,
    else ~/Music/restorations. The basic run must not fail without the flash drive."""
    try:
        return workspace.data_root()
    except Exception:  # noqa: BLE001
        import os
        return Path(os.environ.get("RESTORE_DATA_ROOT", Path.home() / "Music" / "restorations"))


DATA = _data_root()


def run(cmd: list[str], log, timeout: int = 7200, cwd: Path | None = None) -> tuple[int, str]:
    log.write(f"$ {' '.join(cmd)}\n")
    log.flush()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=str(cwd) if cwd else None)
    out = (r.stdout or "") + (r.stderr or "")
    log.write(out[-4000:] + "\n")
    log.flush()
    return r.returncode, out


def flacs(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*.flac"))


def find_discs(src: Path) -> list[tuple[str, Path]]:
    """Where the FLACs live: the directory itself, a nested release, or discs (CD1/CD2).

    Returns [(disc label, directory)]; the label is empty for a single-disc one.
    """
    if flacs(src):
        return [("", src)]
    # look two levels down for directories with tracks
    cands = []
    for d in sorted(src.rglob("*")):
        if d.is_dir() and flacs(d):
            cands.append(d)
    if not cands:
        return []
    # keep only "leaf" ones (without nested directories with tracks)
    leaves = [d for d in cands if not any(o != d and d in o.parents for o in cands)]
    discs = [d for d in leaves if re.match(r"^(cd|disc|disk)\s*\d+$", d.name, re.I)]
    if discs:
        return [(d.name.upper(), d) for d in sorted(discs)]
    if len(leaves) == 1:
        return [("", leaves[0])]
    # Several subfolders without CD1/CD2 labels (e.g. "Steel The Light" + "mini"):
    # they must NOT be driven into one phone directory — the second would clobber
    # the first. Use the subfolder name as the label so each lands in its own subdirectory.
    return [(d.name, d) for d in sorted(leaves)]


def cue_disc_tags(cue: Path) -> dict:
    """Disk-level TITLE/PERFORMER/REM DATE/REM GENRE from the cue."""
    txt = cue.read_text(encoding="utf-8", errors="ignore")
    top = txt.split("TRACK ", 1)[0]
    def grab(pat: str) -> str:
        m = re.search(pat, top, re.M)
        return m.group(1).strip() if m else ""
    album = grab(r'^\s*TITLE\s+"?([^"\n]+)"?')
    artist = grab(r'^\s*PERFORMER\s+"?([^"\n]+)"?')
    date = grab(r'^\s*REM\s+DATE\s+"?([^"\n]+)"?')
    genre = grab(r'^\s*REM\s+GENRE\s+"?([^"\n]+)"?')
    tags = {}
    if album:
        tags["ALBUM"] = album
    if artist:
        tags["ARTIST"] = artist
        tags["ALBUMARTIST"] = artist
    if date:
        tags["DATE"] = date
    if genre:
        tags["GENRE"] = genre
    return tags


def needs_split(d: Path) -> bool:
    """Is this an image+cue? Signs: a cue exists AND exactly one FLAC (or all FLACs are large).

    Previously split was only called when there was NO FLAC in the directory, so an
    image (one big .flac + .cue) went into restoration as "one track" — and the album
    was silently lost.
    """
    if not list(d.glob("*.cue")):
        return False
    fs = flacs(d)
    if not fs:
        return False
    if len(fs) == 1:
        return True
    return all(f.stat().st_size > 80 * 1024 * 1024 for f in fs) and len(fs) <= 2


def split_image(src: Path, log) -> Path:
    """image+cue → tracks next to it; the image moves to _image/ so it doesn't hit --src."""
    cues = list(src.glob("*.cue"))
    imgs = [p for p in src.glob("*.flac") if p.stat().st_size > 80 * 1024 * 1024]
    if not cues or not imgs:
        return src
    img = imgs[0]
    # cwd=src is required: otherwise shntool drops the tracks into the process's
    # working directory (the repository directory), they never appear in the source,
    # and the image went into restoration whole as "one track"
    code, out = run(["shntool", "split", "-f", str(cues[0]), "-o", "flac",
                     "-t", "%n - %t", str(img)], log, cwd=src)
    made = [p for p in src.glob("[0-9]*.flac") if p != img]
    if not made:
        log.write(f"[warn] split produced no tracks: {out[-300:]}\n")
        return src
    hold = src / "_image"
    hold.mkdir(exist_ok=True)
    for f in [img, cues[0]] + list(src.glob("*.log")) + list(src.glob("*.accurip")):
        if f.is_file() and f.parent == src:
            shutil.move(str(f), str(hold / f.name))
    run(["cuetag", str(hold / cues[0].name)] + [str(p) for p in made], log)
    # cuetag only sets per-track TITLE/PERFORMER. The performer and album name
    # in the cue are often set at the DISK level — without carrying them over,
    # the tracks end up on the phone as <unknown> (as happened with Perfect Plan).
    tags = cue_disc_tags(hold / cues[0].name)
    if tags:
        args = []
        for k, v in tags.items():
            args += [f"--set-tag={k}={v}"]
        for p in made:
            run(["metaflac", *args, str(p)], log)
        log.write(f"disk metadata carried over: {tags}\n")
    log.write(f"split into {len(made)} tracks\n")
    return src


def process(job: str, log, clean_src: bool, skip_phone: bool, jobs: int = 1) -> dict:
    parts = [p.strip() for p in job.split("|")]
    src_rel, phone_rel = parts[0], parts[1] if len(parts) > 1 else ""
    name = parts[2] if len(parts) > 2 else Path(src_rel).name
    src = DATA / src_rel if not Path(src_rel).is_absolute() else Path(src_rel)
    out = DATA / "release" / f"{name} [restored]"
    res = {"job": job, "name": name, "src": str(src), "out": str(out), "steps": {}}

    log.write(f"\n{'=' * 80}\n=== {time.strftime('%H:%M:%S')} {name}\n")
    if not src.is_dir():
        res["error"] = "no source directory"
        log.write(f"[error] {src}\n")
        return res
    discs = find_discs(src)
    if not discs:
        res["error"] = "no FLAC in the source"
        return res
    res["tracks"] = sum(len(flacs(d)) for _, d in discs)
    res["discs"] = [label or "-" for label, _ in discs]
    res["summary"] = []
    res["lyrics"] = []
    res["cover"] = True

    for label, ddir in discs:
        if needs_split(ddir):
            ddir = split_image(ddir, log)
        d_out = out / label if label else out
        d_phone = f"{phone_rel}/{label}" if (label and phone_rel) else phone_rel
        res["steps"][f"restore:{label or '-'}"] = None
        code, _ = run([sys.executable, "-u", str(SCRIPTS / "restore_one.py"),
                       "--src", str(ddir), "--out", str(d_out),
                       "--auto-policy", "--adaptive", "--ml-snapshot", "auto",
                       "--tp", "-0.5", "--wait", "--jobs", str(jobs)], log)
        res["steps"][f"restore:{label or '-'}"] = code
        if code != 0 or not flacs(d_out):
            res["error"] = f"restoration produced no tracks ({label or 'cd1'})"
            return res
        code, out2 = run([sys.executable, "-u", str(SCRIPTS / "finish_release.py"),
                          "--restored", str(d_out), "--source", str(ddir), "--lyrics"], log)
        res["steps"][f"finish:{label or '-'}"] = code
        if "cover not found" in out2:
            res["cover"] = False
        m = re.search(r"lyrics: ([^\n]+)", out2)
        if m:
            res["lyrics"].append(f"{label}:{m.group(1)}" if label else m.group(1))
        m = re.search(r"result: ([^\n]+)", out2)
        if m:
            res["summary"].append(f"{label}: {m.group(1)}" if label else m.group(1))
        if d_phone and not skip_phone:
            code, out3 = run([sys.executable, "-u", str(SCRIPTS / "phone_sync.py"),
                              "--album", str(d_out), "--rel", d_phone], log)
            res["steps"][f"phone:{label or '-'}"] = code
            res["phone_ok"] = res.get("phone_ok", True) and "verification ok" in out3
            if code != 0:
                res["error"] = f"copying to the phone failed verification ({label or 'cd1'})"
                return res
    res["summary"] = "; ".join(res["summary"])
    res["lyrics"] = "; ".join(res["lyrics"])

    if clean_src and res.get("phone_ok"):
        shutil.rmtree(src, ignore_errors=True)
        res["src_removed"] = True
        log.write(f"source removed (already on the phone): {src}\n")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True,
                    help="\"source | path on the phone | release name\"")
    ap.add_argument("--log", default="/tmp/release_batch.log")
    ap.add_argument("--clean-src", action="store_true")
    ap.add_argument("--skip-phone", action="store_true")
    ap.add_argument("--jobs", type=int, default=3,
                    help="tracks in parallel within an album (the result does not depend "
                         "on the thread count; ~0.5 GB RAM per thread)")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    results = []
    with open(args.log, "a", encoding="utf-8") as log:
        for job in args.job:
            try:
                results.append(process(job, log, args.clean_src, args.skip_phone, args.jobs))
            except Exception as e:  # noqa: BLE001
                log.write(f"[exception] {type(e).__name__}: {e}\n")
                results.append({"job": job, "error": f"{type(e).__name__}: {e}"})
    print(f"\n{'=' * 80}\nSUMMARY:")
    for r in results:
        status = "ok" if not r.get("error") else f"ERROR: {r['error']}"
        print(f"  {r.get('name', r['job'])[:44]:<46} {status}")
        if r.get("summary"):
            print(f"      {r['summary']}  | cover: {'yes' if r.get('cover') else 'NO'}, "
                  f"lyrics: {r.get('lyrics')}")
    if args.report:
        Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
