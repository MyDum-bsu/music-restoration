#!/usr/bin/env python3
"""Оформление готовой реставрации: теги → обложка → тексты → копия в коллекцию.

`restore_one.py` отдаёт чистые FLAC без тегов и картинок (он про DSP, не про
метаданные). Этот шаг закрывает разрыв и делает релиз готовым к прослушиванию.

  python3 scripts/finish_release.py \
      --restored "release/Album [restored]" \
      --source   "release_src/.../Album" \
      --dest     ~/Music/restorations/"Album [restored]" \
      --lyrics

Порядок: теги из источника → обложка → тексты (add-lyrics) → копия в --dest.
Теги переносятся по номеру трека; если в источнике нет тегов (image+cue),
берём их из cue.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LYRICS_SCRIPT = Path.home() / ".agents/skills/add-lyrics/scripts/genius_lyrics.py"
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
    for d in dirs:
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
        return None
    # предпочитаем маленькие: огромные сканы раздувают каждый трек
    cands.sort(key=lambda p: p.stat().st_size)
    return cands[0]


def copy_tags(src: Path, dst: Path) -> int:
    """Перенести теги источника в результат, СОХРАНИВ уже вшитые тексты.

    Через metaflac не выходит: многострочное значение (LYRICS) он не принимает
    обратно на импорте — «malformed vorbis comment field», и текст теряется.
    mutagen переносит такие поля корректно.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restored", required=True)
    ap.add_argument("--source", default="", help="каталог источника (теги/обложка)")
    ap.add_argument("--cover", default="", help="явная обложка")
    ap.add_argument("--dest", default="", help="куда скопировать (коллекция)")
    ap.add_argument("--lyrics", action="store_true", help="вшить тексты (genius/lrclib)")
    ap.add_argument("--lyrics-force", action="store_true")
    ap.add_argument("--no-tags", action="store_true")
    args = ap.parse_args()

    rest = Path(args.restored).expanduser()
    src = Path(args.source).expanduser() if args.source else None
    if not rest.is_dir():
        print(f"нет каталога {rest}")
        return 1
    r_files = flacs(rest)
    if not r_files:
        print(f"в {rest} нет FLAC")
        return 1

    # --- теги ---------------------------------------------------------------
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
                print(f"  [warn] теги {rp.name}: {e}")
        print(f"теги перенесены в {tagged}/{len(r_files)} треков")

    # --- обложка ------------------------------------------------------------
    cover = find_cover(rest, src, explicit=Path(args.cover).expanduser() if args.cover else None)
    if cover:
        # большие сканы (3–5 МБ) раздувают каждый трек: ужимаем до ~1200 px
        small = cover
        if cover.stat().st_size > 400 * 1024:
            try:
                from PIL import Image
                im = Image.open(cover).convert("RGB")
                if max(im.size) > 1200:
                    im.thumbnail((1200, 1200))
                small = rest / "cover.jpg"
                im.save(small, quality=88, optimize=True)
                print(f"обложка ужата: {cover.stat().st_size // 1024} KB → "
                      f"{small.stat().st_size // 1024} KB, {im.size[0]}x{im.size[1]}")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] не удалось ужать обложку ({type(e).__name__}) — вшиваю как есть")
                small = cover
        for rp in r_files:
            # снять прежние картинки: import-picture ДОБАВЛЯЕТ блок, а не заменяет
            # --dont-shuffle есть не во всех сборках metaflac (1.4.3 его не знает)
            subprocess.run(["metaflac", "--remove", "--block-type=PICTURE", str(rp)],
                           capture_output=True)
            metaflac("--import-picture-from", str(small), str(rp))
        print(f"обложка: {small.name} ({small.stat().st_size // 1024} KB) → {len(r_files)} треков")
        cover = small
    else:
        # Запасной путь: у WEB-релизов обложки-файла часто нет вовсе (Subsignal
        # «La Muerta»), и альбом уходил на телефон «голым». Тянем из MusicBrainz
        # по тегам первого трека.
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
                    print(f"обложка (из MusicBrainz) → {len(r_files)} треков")
                else:
                    print("[warn] обложка не найдена ни в источнике, ни в MusicBrainz")
            else:
                print("[warn] обложка не найдена, теги пусты — нечем искать")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] обложка не найдена ({type(e).__name__})")

    # --- тексты -------------------------------------------------------------
    if args.lyrics:
        if not LYRICS_SCRIPT.is_file():
            print(f"[warn] нет {LYRICS_SCRIPT} — тексты пропущены")
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
            print("тексты: " + ", ".join(f"{k}={v}" for k, v in sorted(stat.items())))
            for line in r.stdout.splitlines():
                if line.startswith(("NOMATCH", "AMBIGUOUS", "ERROR")):
                    print("   " + line.replace("\t", " | ")[:150])

    # --- копия в коллекцию ---------------------------------------------------
    if args.dest:
        dst = Path(args.dest).expanduser()
        dst.mkdir(parents=True, exist_ok=True)
        for rp in r_files:
            shutil.copy2(rp, dst / rp.name)
        if cover:
            shutil.copy2(cover, dst / ("cover" + cover.suffix.lower()))
        size = sum(p.stat().st_size for p in dst.iterdir() if p.is_file())
        print(f"скопировано в {dst}: {len(r_files)} треков, {size / 1e6:.0f} MB")

    # --- сводка --------------------------------------------------------------
    report = rest / "restore_report.jsonl"
    if report.is_file():
        rows = [json.loads(l) for l in report.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            import statistics as S
            d = [r["delta"]["dr"] for r in rows]
            c = [r["delta"]["crest"] for r in rows]
            print(f"итог: DR +{min(d)}…+{max(d)} (медиана +{S.median(d):.0f}), "
                  f"Δcrest медиана {S.median(c):+.2f} дБ, TP {rows[0].get('tp_actual')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
