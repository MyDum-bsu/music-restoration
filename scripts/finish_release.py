#!/usr/bin/env python3
"""Finalize a finished restoration: tags → cover → lyrics → copy into the collection.

`restore_one.py` outputs clean FLACs without tags or pictures (it is about DSP,
not metadata). This step closes the gap and makes the release ready for listening.

  python3 scripts/finish_release.py \
      --restored "release/Album [restored]" \
      --source   "release_src/.../Album" \
      --dest     ~/Music/restorations/"Album [restored]" \
      --lyrics

Order: tags from the source → cover → lyrics (add-lyrics) → copy into --dest.
Tags are carried over by track number; if the source has no tags (image+cue),
they are taken from the cue.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LYRICS_SCRIPT = (Path(__file__).resolve().parents[1]
                 / "skills" / "add-lyrics" / "scripts" / "genius_lyrics.py")
COVER_HINTS = ("cover.jpg", "folder.jpg", "front.jpg", "Cover.jpg", "Folder.jpg",
               "Front.jpg", "cover.png", "folder.png", "front.png")
COVER_BAD = ("back", "cd.", "disc", "inlay", "matrix", "booklet", "1 0", "scan")


def flacs(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*.flac") if not p.name.startswith("_"))


def track_no(p: Path) -> int | None:
    m = re.match(r"\s*(\d{1,3})", p.name)
    return int(m.group(1)) if m else None


def metaflac(*args: str, input_text: str | None = None) -> str:
    r = subprocess.run(["metaflac", *args], capture_output=True, text=True,
                       input=input_text)
    if r.returncode != 0:
        raise RuntimeError(f"metaflac {args[0]}: {r.stderr.strip()[:200]}")
    return r.stdout


def find_cover(*dirs: Path, explicit: Path | None = None) -> Path | None:
    if explicit:
        return explicit if explicit.is_file() else None
    cands: list[Path] = []
    extra = []
    for d in dirs:
        if d and d.is_dir() and d.parent not in dirs:
            extra.append(d.parent)      # torrent root (the cover is sometimes there)
    for d in list(dirs) + extra:
        if not d or not d.is_dir():
            continue
        for name in COVER_HINTS:
            p = d / name
            if p.is_file():
                cands.append(p)
        if not cands:
            for sub in ("Scans", "Covers", "scans", "covers"):
                sd = d / sub
                if sd.is_dir():
                    cands += [p for p in sd.iterdir()
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                              and not any(b in p.name.lower() for b in COVER_BAD)]
    if not cands:
        # No named cover — take any image (Papa Roach's was called
        # "The Paramour Sessions.jpg"), but not a back/inlay/booklet.
        # A cover is nearly square: this cuts off obi strips and gatefolds
        # (for Q5 an obi strip was picked first and shipped to the phone as the cover).
        pool = []
        for d in list(dirs) + extra:
            if not d or not d.is_dir():
                continue
            for p in d.iterdir():
                if (p.suffix.lower() in (".jpg", ".jpeg", ".png")
                        and not any(b in p.name.lower() for b in COVER_BAD)):
                    pool.append(p)
        square = []
        for p in pool:
            try:
                from PIL import Image
                w, h = Image.open(p).size
            except Exception:  # noqa: BLE001
                continue
            if 0.8 <= min(w, h) / max(w, h):
                square.append((w * h, p))
        if square:
            square.sort(reverse=True)      # the largest square one
            cands = [square[0][1]]
        else:
            cands = pool[:1]
    if not cands:
        return None
    # prefer smaller ones: huge scans bloat every track
    cands.sort(key=lambda p: p.stat().st_size)
    return cands[0]


def copy_tags(src: Path, dst: Path) -> int:
    """Carry the source tags into the result, PRESERVING already embedded lyrics.

    metaflac can't do it: it refuses a multi-line value (LYRICS) on import —
    "malformed vorbis comment field" — and the lyrics are lost.
    mutagen carries such fields correctly.
    """
    from mutagen.flac import FLAC
    sf, df = FLAC(str(src)), FLAC(str(dst))
    keep = {k: list(df[k]) for k in ("lyrics", "unsyncedlyrics") if k in df}
    df.delete()
    for k, v in sf.tags or []:
        df[k] = v
    for k, v in keep.items():
        df[k] = v
    df.save()
    return len(sf.tags or [])


def unify_album(rest_files: list[Path]) -> str:
    """One ALBUM/ARTIST/ALBUMARTIST for the whole album + numbering 01..N.

    Some tracks arrive with a different album spelling (Q5: 8 tracks
    "… [VDP-1141]", 2 — without the catalog number), and the player splits the
    album in two: "only 10 and 11 visible, the rest live separately". Numbering
    is fixed too: if the pregap from the cue was dropped, tracks started at 02.
    """
    import re as _re
    from collections import Counter
    from mutagen.flac import FLAC
    if not rest_files:
        return ""
    files = []
    for f in rest_files:
        try:
            files.append((f, FLAC(str(f))))
        except Exception:  # noqa: BLE001
            continue
    if not files:
        return ""
    def vote(key: str) -> str:
        c = Counter()
        for _f, fl in files:
            v = fl.get(key) or fl.get(key.lower()) or [""]
            if v and v[0].strip():
                c[v[0].strip()] += 1
        if not c:
            return ""
        # majority; on a tie — the longer one (usually with the catalog number)
        return max(c.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
    album, artist = vote("ALBUM"), vote("ARTIST")
    changed = []
    for f, fl in files:
        cur = (fl.get("ALBUM") or [""])[0]
        if album and cur != album:
            fl["ALBUM"] = album; changed.append(f.name)
        if artist:
            if (fl.get("ARTIST") or [""])[0] != artist:
                fl["ARTIST"] = artist; changed.append(f.name)
            if (fl.get("ALBUMARTIST") or [""])[0] != artist:
                fl["ALBUMARTIST"] = artist
        fl.save()
    # consecutive numbering if it isn't 1..N
    nums = []
    for f, fl in files:
        m = _re.match(r"^(\d{1,3})\b", (fl.get("TRACKNUMBER") or [""])[0])
        nums.append(int(m.group(1)) if m else None)
    if None in nums or sorted(nums) != list(range(1, len(files) + 1)):
        order = sorted(range(len(files)), key=lambda i: (nums[i] is None, nums[i] or 0))
        for pos, idx in enumerate(order, 1):
            f, fl = files[idx]
            if nums[idx] != pos:
                fl["TRACKNUMBER"] = f"{pos:02d}"; fl.save()
                changed.append(f.name)
        changed.append(f"renumbered 01..{len(files):02d}")
    return f"album=[{album}], artist=[{artist}], {len(set(changed))} changes" if changed else ""


def normalize_titles(rest_files: list[Path]) -> int:
    """Fix SHIFTED tags: when one track's TITLE belongs to another.

    The file name comes from the cue and is therefore correct, but not always
    usable as a tag: it can be truncated ("Strong (Orchestral Version - B…"),
    sanitized ("F_d My Way") or contain "/" replaced with " - " ("Foreplay -
    Long Time"). So the name overrides the tag ONLY when the tag is clearly
    someone else's — it matches another track's title of the same album
    (a sign of a one-position shift), or the tag is missing entirely.
    """
    from mutagen.flac import FLAC
    derived: dict[Path, tuple[int, str]] = {}
    for f in rest_files:
        m = re.match(r"^\s*(\d{1,3})\s*[.\-]\s*(.+)$", f.stem)
        if m:
            derived[f] = (int(m.group(1)), m.group(2).strip())
    if not derived:
        return 0
    # how often each "name from file" occurs — used to catch the shift
    names = {f: n for f, (_num, n) in derived.items()}
    fixed = 0
    for f, (num, name) in derived.items():
        fl = FLAC(str(f))
        title = (fl.get("title") or fl.get("TITLE") or [""])[0].strip()
        trk = (fl.get("tracknumber") or fl.get("TRACKNUMBER") or [""])[0].strip()
        others = {n for g, n in names.items() if g != f}
        shifted = bool(title) and title in others
        if not title or shifted:
            if shifted:
                print(f"  [shift] {f.name[:30]}: was [{title[:30]}] → [{name[:30]}]")
            fl["TITLE"] = name
            fixed += 1
        if (not trk) or (trk.isdigit() and int(trk) != num):
            fl["TRACKNUMBER"] = f"{num:02d}"
        fl.save()
    return fixed


def fill_from_cue(rest_files: list[Path], src: Path | None,
                  rest: Path | None = None) -> str:
    """If the tracks have no ARTIST/ALBUM — take them from the cue next to the source.

    The cue sets the performer at the disk level; cuetag doesn't carry it over,
    and the tracks end up on the phone as <unknown> (Perfect Plan, Place Vendome).
    """
    cue = None
    if src is not None:
        for d in (src, src.parent, src.parent.parent):
            if d.is_dir() and list(d.glob("*.cue")):
                cue = next(iter(d.glob("*.cue")))
                break
    if cue is None:
        # The source is already gone (--clean-src) — parse the release name
        # "Artist - Album (Year) [restored]".
        name = (rest or Path("")).name.replace(" [restored]", "")
        m = re.match(r"^(.*?)\s+-\s+(.*?)\s*\((\d{4})\)$", name)
        if not m:
            return ""
        album, artist, date = m.group(2).strip(), m.group(1).strip(), m.group(3)
        from mutagen.flac import FLAC
        fixed = 0
        for f in rest_files:
            fl = FLAC(str(f))
            changed = False
        # Keys in UPPER CASE: the rest of the library uses uppercase tags,
        # and mixed case looks like different tags to different players.
            for k, v in (("ARTIST", artist), ("ALBUMARTIST", artist),
                         ("ALBUM", album), ("DATE", date)):
                if k.lower() not in {x.lower() for x in fl}:
                    fl[k] = v; changed = True
            if changed:
                fl.save(); fixed += 1
        return f"from name: album={album}, artist={artist}, date={date} → {fixed} tracks" if fixed else ""
    txt = cue.read_text(encoding="utf-8", errors="ignore")
    top = txt.split("TRACK ", 1)[0]
    def grab(pat: str) -> str:
        m = re.search(pat, top, re.M)
        return m.group(1).strip() if m else ""
    album = grab(r'^\s*TITLE\s+"?([^"\n]+)"?')
    artist = grab(r'^\s*PERFORMER\s+"?([^"\n]+)"?')
    date = grab(r'^\s*REM\s+DATE\s+"?([^"\n]+)"?')
    genre = grab(r'^\s*REM\s+GENRE\s+"?([^"\n]+)"?')
    if not (album or artist):
        return ""
    from mutagen.flac import FLAC
    fixed = []
    for f in rest_files:
        fl = FLAC(str(f))
        changed = False
        have = {x.lower() for x in fl}
        for k, v in (("ARTIST", artist), ("ALBUMARTIST", artist), ("ALBUM", album),
                     ("DATE", date), ("GENRE", genre)):
            if v and k.lower() not in have:
                fl[k] = v; changed = True
            if changed:
                fl.save(); fixed.append(f.name[:24])
    if not fixed:
        return ""
    bits = [b for b in (f"album={album}" if album else "", f"artist={artist}" if artist else "",
                        f"date={date}" if date else "") if b]
    return ", ".join(bits) + f" → {len(fixed)} tracks"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restored", required=True)
    ap.add_argument("--source", default="", help="source directory (tags/cover)")
    ap.add_argument("--cover", default="", help="explicit cover")
    ap.add_argument("--dest", default="", help="where to copy (collection)")
    ap.add_argument("--lyrics", action="store_true", help="embed lyrics (genius/lrclib)")
    ap.add_argument("--lyrics-force", action="store_true")
    ap.add_argument("--no-tags", action="store_true")
    ap.add_argument("--skip-spectra", action="store_true",
                    help="do not render waveform/spectrogram comparison pages")
    args = ap.parse_args()

    rest = Path(args.restored).expanduser()
    src = Path(args.source).expanduser() if args.source else None
    if not rest.is_dir():
        print(f"no directory {rest}")
        return 1
    r_files = flacs(rest)
    if not r_files:
        print(f"no FLAC in {rest}")
        return 1

    # --- tags ----------------------------------------------------------------
    tagged = 0
    if not args.no_tags and src and src.is_dir():
        s_files = flacs(src)
        by_no: dict[int, Path] = {}
        for i, p in enumerate(s_files, 1):
            by_no[track_no(p) or i] = p
        for i, rp in enumerate(r_files, 1):
            sp = by_no.get(track_no(rp) or i)
            if sp is None:
                continue
            try:
                n = copy_tags(sp, rp)
                tagged += 1 if n else 0
            except RuntimeError as e:
                print(f"  [warn] tags {rp.name}: {e}")
        print(f"tags carried into {tagged}/{len(r_files)} tracks")

    # --- title normalization from file names ---------------------------------
    fixed_n = normalize_titles(r_files)
    if fixed_n:
        print(f"titles/numbers fixed: {fixed_n} tracks")

    # --- fill missing tags from the cue ---------------------------------------
    missing = fill_from_cue(r_files, src, rest)
    if missing:
        print(f"tags from cue: {missing}")

    # --- album unity -----------------------------------------------------------
    uni = unify_album(r_files)
    if uni:
        print(f"album tags aligned: {uni}")

    # --- cover -----------------------------------------------------------------
    cover = find_cover(rest, src, explicit=Path(args.cover).expanduser() if args.cover else None)
    if cover:
        # big scans (3–5 MB) bloat every track: downscale to ~1200 px
        small = cover
        if cover.stat().st_size > 400 * 1024:
            try:
                from PIL import Image
                im = Image.open(cover).convert("RGB")
                if max(im.size) > 1200:
                    im.thumbnail((1200, 1200))
                small = rest / "cover.jpg"
                im.save(small, quality=88, optimize=True)
                print(f"cover shrunk: {cover.stat().st_size // 1024} KB → "
                      f"{small.stat().st_size // 1024} KB, {im.size[0]}x{im.size[1]}")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] failed to shrink the cover ({type(e).__name__}) — embedding as is")
                small = cover
        for rp in r_files:
            # drop previous pictures: import-picture ADDS a block, it does not replace
            # --dont-shuffle is absent in some metaflac builds (1.4.3 doesn't know it)
            subprocess.run(["metaflac", "--remove", "--block-type=PICTURE", str(rp)],
                           capture_output=True)
            metaflac("--import-picture-from", str(small), str(rp))
        print(f"cover: {small.name} ({small.stat().st_size // 1024} KB) → {len(r_files)} tracks")
        cover = small
    else:
        # Fallback path: WEB releases often have no cover file at all (Subsignal
        # "La Muerta"), and the album went to the phone "naked". Pull one from
        # MusicBrainz using the first track's tags.
        try:
            from mutagen.flac import FLAC as _FLAC
            t = _FLAC(str(r_files[0]))
            art = (t.get("albumartist") or t.get("artist") or [""])[0]
            alb = (t.get("album") or [""])[0]
            if art and alb:
                cand = rest / "_cover_fetch.jpg"
                r = subprocess.run([sys.executable, str(Path(__file__).parent / "fetch_cover.py"),
                                    "--artist", art, "--album", alb, "--out", str(cand)],
                                   capture_output=True, text=True)
                if r.returncode == 0 and cand.is_file():
                    print(r.stdout.strip())
                    for rp in r_files:
                        subprocess.run(["metaflac", "--remove", "--block-type=PICTURE", str(rp)],
                                       capture_output=True)
                        metaflac("--import-picture-from", str(cand), str(rp))
                    cover = cand
                    print(f"cover (from MusicBrainz) → {len(r_files)} tracks")
                else:
                    print("[warn] cover not found in source or MusicBrainz")
            else:
                print("[warn] cover not found, tags empty — nothing to search with")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] cover not found ({type(e).__name__})")

    # --- lyrics ----------------------------------------------------------------
    if args.lyrics:
        if not LYRICS_SCRIPT.is_file():
            print(f"[warn] {LYRICS_SCRIPT} missing — lyrics skipped")
        else:
            cmd = [sys.executable, str(LYRICS_SCRIPT), "tag", "--dir", str(rest)]
            if args.lyrics_force:
                cmd.append("--force")
            r = subprocess.run(cmd, capture_output=True, text=True)
            stat: dict[str, int] = {}
            for line in r.stdout.splitlines():
                st = line.split("\t", 1)[0].strip()
                if st:
                    stat[st] = stat.get(st, 0) + 1
            print("lyrics: " + ", ".join(f"{k}={v}" for k, v in sorted(stat.items())))
            for line in r.stdout.splitlines():
                if line.startswith(("NOMATCH", "AMBIGUOUS", "ERROR")):
                    print("   " + line.replace("\t", " | ")[:150])

    # --- copy into the collection ------------------------------------------------
    if args.dest:
        dst = Path(args.dest).expanduser()
        dst.mkdir(parents=True, exist_ok=True)
        for rp in r_files:
            shutil.copy2(rp, dst / rp.name)
        if cover:
            shutil.copy2(cover, dst / ("cover" + cover.suffix.lower()))
        size = sum(p.stat().st_size for p in dst.iterdir() if p.is_file())
        print(f"copied to {dst}: {len(r_files)} tracks, {size / 1e6:.0f} MB")

    # --- DR report in the meter format (as masters publish on the tracker) --------
    print("DR report (meter):", flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "dr_report.py"),
                        str(rest)], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        print(r.stdout.strip())
        if args.dest:
            shutil.copy2(rest / "DR Report.txt", dst / "DR Report.txt")
    else:
        print(f"[warn] DR report not built: {(r.stderr or '')[-150:]}")

    # --- spectra: waveform/spectrogram pages + animated orig/restored GIFs ------
    if src and src.is_dir() and not args.skip_spectra:
        print("spectra:", flush=True)
        r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "spectrum_report.py"),
                            "--orig", str(src), "--restored", str(rest),
                            "--out", str((dst if args.dest else rest) / "spectra")],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(r.stdout.strip())
        else:
            print(f"[warn] spectra not built: {(r.stderr or '')[-150:]}")

    # --- summary ------------------------------------------------------------------
    report = rest / "restore_report.jsonl"
    if report.is_file():
        rows = [json.loads(l) for l in report.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            import statistics as S
            d = [r["delta"]["dr"] for r in rows]
            c = [r["delta"]["crest"] for r in rows]
            print(f"result: DR +{min(d)}…+{max(d)} (median +{S.median(d):.0f}), "
                  f"Δcrest median {S.median(c):+.2f} dB, TP {rows[0].get('tp_actual')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
