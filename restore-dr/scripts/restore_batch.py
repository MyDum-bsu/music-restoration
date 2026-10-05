#!/usr/bin/env python3
"""restore_batch.py — the canonical album pipeline, one command per album:

  pull -> restore (separate src/dst, v2 chain) -> covers -> fill tags
       -> [lyrics] -> DR.txt + report -> HARD GATE -> push + md5 verify -> cleanup

The gate: a file is pushed only if flac -t passes, artist/album/title/tracknumber
are present, an embedded cover exists, and output duration/samplerate match the
source; every push is verified by reading the file back and comparing md5.
Any failure = loud QUARANTINE, never a silent push. Staging is deleted right
after a verified delivery (quarantine/ survives for inspection).

Sources/destinations are any local dir or a phone Music root over MTP
(auto-detected; see config.env.example).

Usage:
  restore_batch.py "Artist - Album"                    # from/to the phone (or MUSIC_DIR)
  restore_batch.py /path/to/album --out-local ./done   # local file, local result
  restore_batch.py "Album" --mb --lyrics --genre "Prog Rock"
  restore_batch.py "Album" --nd2 1.5 --plots           # gentle + graphs
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drlib import enrich, gate, mtp
from drlib.bootstrap import add_skill_scripts
from drlib.config import ensure_dirs, music_dir, work_dir, work_tmpdir
from drlib.tags import export_cover, import_cover, set_core_tags
from drlib.util import md5f
from master import master_track
from measure import decode, tt_dr


def opts_common(sp):
    sp.add_argument("--nd2", type=float, default=2.5,
                    help="ND2 punch strength: 2.5 crushed masters, 1.5-2.0 delicate; 0 disables")
    sp.add_argument("--slope", type=float, default=1.08)
    sp.add_argument("--offset", type=float, default=-4.6)
    sp.add_argument("--side-db", type=float, default=-3.0)
    sp.add_argument("--tp-target", type=float, default=-0.5)
    sp.add_argument("--depth", type=int, default=16, choices=(16, 24))
    sp.add_argument("--no-reference-eq", action="store_true")
    sp.add_argument("--hpf", type=float, default=None)
    sp.add_argument("--declip", type=lambda v: {"on": True, "off": False, "auto": None}[v], default=None)
    sp.add_argument("--nd2-bass-scale", type=float, default=0.35)
    sp.add_argument("--slew", type=float, default=40.0)
    sp.add_argument("--env-win", type=float, default=3.0)


def find_album(root: Path, name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p.resolve()
    if (root / name).is_dir():
        return root / name
    hits = sorted(d for d in root.rglob(name) if d.is_dir())
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        sys.exit(f"ambiguous album name '{name}' matches {len(hits)} dirs:\n  " + "\n  ".join(map(str, hits[:10])))
    sys.exit(f"album dir not found: '{name}' under {root}")


def stage_album(src: Path, dst: Path) -> list[Path]:
    """Bulk-copy the album locally. rglob, not glob — albums sometimes nest."""
    dst.mkdir(parents=True, exist_ok=True)
    files = []
    for f in sorted(src.rglob("*.flac")):
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
        files.append(target)
    return files


def cover_for(a_dir: Path, b_dir: Path, workroot: Path, artist=None, album=None, use_mb=False):
    """Pick the best cover available. Returns (image_path_or_None, source_string)."""
    cov = workroot / "cover.img"
    b_files = sorted(b_dir.rglob("*.flac"))
    if not b_files:
        return None, "no files"
    src_first = a_dir / b_files[0].relative_to(b_dir)
    if export_cover(src_first, cov):
        return cov, "inherited from source track"
    folder = enrich.pick_folder_cover(a_dir)
    if folder:
        shutil.copy2(folder, cov)
        if cov.stat().st_size > 600 * 1024:  # big scans bloat every track
            small = cov.with_suffix(".small.jpg")
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(cov),
                                "-vf", "scale=1200:-2", "-q:v", "4", str(small)],
                               capture_output=True)
            if r.returncode == 0 and small.exists() and small.stat().st_size > 0:
                cov.unlink()
                small.rename(cov)
        return cov, f"folder image {folder.name}"
    if use_mb and artist and album:
        for mbid in enrich.mb_search_release_ids(artist, album):
            if enrich.caa_cover(mbid, cov):
                return cov, f"Cover Art Archive {mbid}"
    return None, "none found"


def fill_tags(b_dir: Path, src_album: Path, args) -> dict:
    """Guarantee artist/album/title/tracknumber on every file:
    CLI args > existing tags > MusicBrainz (--mb) > filename."""
    files = sorted(b_dir.rglob("*.flac"))
    from mutagen.flac import FLAC
    first = FLAC(str(files[0]))
    artist = args.artist or (first.get("artist") or [None])[0]
    album = args.album or (first.get("album") or [None])[0]
    date = args.date or (first.get("date") or [None])[0]
    trackmap, mb_date = ({}, None)
    if args.mb and artist and album:
        trackmap, mb_date = enrich.mb_trackmap(artist, album, len(files))
        if mb_date and not date:
            date = mb_date
    filled = {"artist": bool(artist), "album": bool(album), "mb_titles": 0, "from_filename": 0}
    if not artist or not album:
        print(f"WARN: no artist/album available (tags empty, no --artist/--album, "
              f"{'MB off' if not args.mb else 'MB found nothing'}) — the gate WILL quarantine this album")
    for f in files:
        fl = FLAC(str(f))
        tno = enrich.parse_trackno(f.name)
        if not fl.get("tracknumber") and tno:
            fl.tags["tracknumber"] = [f"{tno:02d}"]
            filled["from_filename"] += 1
        title = (fl.get("title") or [None])[0]
        if not title and tno and tno in trackmap:
            title = trackmap[tno]
            filled["mb_titles"] += 1
        if not title:
            title = enrich.title_from_filename(f.name)
            fl.tags["title"] = [title]
            filled["from_filename"] += 1
        fl.save()
        set_core_tags(f, artist=artist, album=album, date=date, genre=args.genre,
                      title=None if fl.get("title") else title)  # set_core_tags skips None
    return filled


def lyrics_pass(b_dir: Path) -> tuple[int, int]:
    """genius_lyrics.py tag --dir (Genius with lrclib fallback, sequential, cached)."""
    from drlib.util import python_exe
    script = add_skill_scripts("add-lyrics") / "genius_lyrics.py"
    r = subprocess.run([python_exe(), str(script), "tag", "--dir", str(b_dir)],
                       capture_output=True, text=True, timeout=7200)
    lines = (r.stdout or "").splitlines()
    ok = sum(1 for l in lines if l.startswith("OK"))
    return ok, len([f for f in b_dir.rglob("*.flac")])


def write_reports(a_dir: Path, b_dir: Path, album_name: str, args) -> tuple[list, float, float]:
    rows = []
    for f in sorted(b_dir.glob("*.flac")):
        s = a_dir / f.name
        x, fs = decode(str(s), "s16le")
        d0 = tt_dr(x / 32768.0, fs)[0]
        x, fs = decode(str(f), "s16le")
        d1 = tt_dr(x / 32768.0, fs)[0]
        rows.append((f.name, d0, d1))
    avg0 = sum(r[1] for r in rows) / len(rows)
    avg1 = sum(r[2] for r in rows) / len(rows)
    with open(b_dir / "DR.txt", "w") as t:
        t.write(f"# {album_name} — restored by music-restoration v2 (nd2={args.nd2}, slope={args.slope}, offset={args.offset})\n")
        for name, d0, d1 in rows:
            t.write(f"{name}: DR{d0} -> DR{d1}\n")
        t.write(f"# album DR: {avg0:.1f} -> {avg1:.1f}\n")
    return rows, avg0, avg1


def push_all(b_dir: Path, root: Path, rel: Path, extra_files, workroot: Path) -> list[str]:
    """Push album files + DR.txt (+report), verifying each by reading it back."""
    fails = []
    verify_tmp = workroot / "_verify.bin"
    files = sorted(b_dir.glob("*.flac")) + extra_files
    for f in files:
        rrel = rel / f.relative_to(b_dir)
        ok, err = mtp.copy_to_phone(root, f, rrel)
        if not ok:
            fails.append(f"{f.name}: copy: {err}")
            continue
        got = mtp.cat_md5(root, rrel, verify_tmp)
        if got is None or got != md5f(f):
            fails.append(f"{f.name}: md5 mismatch after push")
    verify_tmp.unlink(missing_ok=True)
    return fails


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("album_dir", help="album dir (path, or name under the source root)")
    p.add_argument("--src-root", help="library to pull from (default: phone if connected, else MUSIC_DIR)")
    p.add_argument("--dst-root", help="library to push to (default: same as source root)")
    p.add_argument("--out-local", help="write the result here instead of pushing")
    p.add_argument("--artist", help="normalize artist/albumartist on all files")
    p.add_argument("--album", help="override album tag")
    p.add_argument("--date")
    p.add_argument("--genre")
    p.add_argument("--mb", action="store_true", help="MusicBrainz/CAA enrichment of missing tags/covers")
    p.add_argument("--lyrics", action="store_true", help="embed lyrics (Genius, lrclib fallback)")
    p.add_argument("--plots", action="store_true", help="before/after PNG per track (slow)")
    p.add_argument("--keep-staging", action="store_true")
    opts_common(p)
    a = p.parse_args()

    ensure_dirs()
    phone = mtp.phone_dir()
    src_root = Path(a.src_root) if a.src_root else (phone if phone and phone.exists() else music_dir())
    src = find_album(src_root, a.album_dir)
    try:
        rel = src.relative_to(src_root)
    except ValueError:  # direct path outside the root: deliver by album name
        rel = Path(src.name)
    album_name = src.name
    ts = time.strftime("%Y%m%d-%H%M%S")
    workroot = work_dir() / f"batch_{ts}"
    a_dir, b_dir = workroot / "src" / album_name, workroot / "dst" / album_name

    print(f"album : {src}\nroot  : {src_root}\nwork  : {workroot}", flush=True)
    if not mtp.is_mtp(src) and src_root == Path(a.src_root or src_root):
        pass  # local roots are fine too
    files = stage_album(src, a_dir)
    if not files:
        sys.exit(f"no FLAC files under {src}")
    print(f"staged {len(files)} tracks", flush=True)

    b_dir.mkdir(parents=True)
    from drlib.audio import duration_and_fs
    t0 = time.time()
    for i, f in enumerate(files, 1):
        steps, prof = master_track(a_dir / f.name, b_dir / f.name,
                                   a.slope, a.offset, a.side_db, a.tp_target, a.depth,
                                   not a.no_reference_eq, a.hpf, a.declip, a.nd2,
                                   a.nd2_bass_scale, a.slew, a.env_win)
        sdur, sfs = duration_and_fs(a_dir / f.name)
        ddur, dfs = duration_and_fs(b_dir / f.name)
        if sfs != dfs or abs(sdur - ddur) > gate.DURATION_TOL_S:
            print(f"FATAL {f.name}: duration/samplerate changed ({sdur:.2f}s/{sfs} -> {ddur:.2f}s/{dfs})", flush=True)
            gate.quarantine(b_dir, workroot)
            sys.exit(2)
        print(f"[{i}/{len(files)}] {f.name}: LUFS {prof['lufs']}, gain {steps['static_gain_db']}dB, "
              f"declip={steps['declip']} hpf={steps['hpf_hz']} ({time.time()-t0:.0f}s)", flush=True)

    cov, cov_src = cover_for(a_dir, b_dir, workroot,
                             artist=a.artist, album=a.album, use_mb=a.mb)
    if cov:
        import_cover(sorted(b_dir.rglob("*.flac")), cov)
        print(f"cover : {cov_src}", flush=True)
    else:
        print("cover : NONE — the gate will quarantine unless a cover appears", flush=True)

    filled = fill_tags(b_dir, src, a)
    print(f"tags  : filled {json.dumps(filled)}", flush=True)

    lyr = (0, 0)
    if a.lyrics:
        lyr = lyrics_pass(b_dir)
        print(f"lyrics: {lyr[0]}/{lyr[1]} embedded (absent sources are reported, never invented)", flush=True)

    rows, avg0, avg1 = write_reports(a_dir, b_dir, album_name, a)
    print(f"DR    : {avg0:.1f} -> {avg1:.1f}", flush=True)

    plots_dir = None
    if a.plots:
        from drlib.util import python_exe
        plots_dir = workroot / "plots"
        for f in sorted(b_dir.glob("*.flac")):
            subprocess.run([python_exe(), str(Path(__file__).parent / "measure.py"), "compare",
                            str(a_dir / f.name), str(f), "--outdir", str(plots_dir),
                            "--label", f.stem], capture_output=True)

    problems = gate.gate_album(b_dir, src_dir=a_dir)
    if problems:
        q = gate.quarantine(b_dir, workroot)
        print("\nQUARANTINE — the album failed the gate, nothing was pushed:", flush=True)
        for pr in problems[:10]:
            print(f"  {pr}", flush=True)
        print(f"dst moved to {q}", flush=True)
        if not a.keep_staging:
            shutil.rmtree(a_dir, ignore_errors=True)
        sys.exit(2)

    if a.out_local:
        out = Path(a.out_local)
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(b_dir.iterdir()):
            shutil.copy2(f, out / f.name)
            if md5f(f) != md5f(out / f.name):
                sys.exit(f"local copy md5 mismatch: {f.name}")
        print(f"delivered to local dir: {out}", flush=True)
        code = 0
    else:
        dst_root = Path(a.dst_root) if a.dst_root else src_root
        mtp.mkdir_phone(dst_root, rel)
        extra = [b_dir / "DR.txt"]
        fails = push_all(b_dir, dst_root, rel, extra, workroot)
        if fails:
            print("\nPUSH FAIL — staging kept for re-push:", flush=True)
            for f_ in fails[:10]:
                print(f"  {f_}", flush=True)
            sys.exit(3)
        print(f"pushed and md5-verified: {len(files) + len(extra)} files -> {dst_root / rel}", flush=True)
        if not a.keep_staging:
            shutil.rmtree(a_dir, ignore_errors=True)
            shutil.rmtree(b_dir, ignore_errors=True)

    gate.sweep_workroot(workroot)
    if workroot.exists() and not any(workroot.iterdir()):
        workroot.rmdir()
    print("DONE — staging cleaned (quarantine/ kept). "
          "If the phone player shows stale info: rescan its library.", flush=True)


if __name__ == "__main__":
    main()
