#!/usr/bin/env python3
"""Синхронизация альбомов с телефоном (MTP) — замена, копирование, проверка.

Запись на MTP идёт только через `gio copy` (fuse-монтирование фактически
read-only: cp/touch/mutagen-save падают с «Operation not supported»).

  python3 scripts/phone_sync.py --album "<локальный каталог>" --rel "Jelusick/Apocalyptic (2025)"
  python3 scripts/phone_sync.py --file "<локальный файл>" --rel "Telegram/03 - Denied By The Cross.flac"
  python3 scripts/phone_sync.py --list "Jelusick"
  python3 scripts/phone_sync.py --rm "Music/Jelusick/Old"

Проверка обязательна: после копирования читаем файл обратно (`gio cat`) и
сверяем md5 — fuse-метаданные сразу после записи врут.
"""
import argparse
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path("/home/mydum/Desktop/projects/music-restoration/restore-dr/scripts")))

from drlib import mtp  # noqa: E402

AUDIO = {".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".ape"}


def md5f(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(*args, **kw):
    return subprocess.run(list(args), capture_output=True, text=True, **kw)


def rm_phone(root: Path, rel: Path) -> bool:
    """Удалить файл или каталог на телефоне (gio remove/rm)."""
    uri = mtp.uri_for(root, rel)
    for tool in (("gio", "remove", "-f", uri), ("gio", "rm", "-f", uri)):
        if run(*tool).returncode == 0:
            return True
    return False


def album_replace(root: Path, local: Path, rel: Path, verify: bool = True) -> int:
    files = sorted(f for f in local.iterdir() if f.is_file() and f.suffix.lower() in AUDIO)
    extra = sorted(f for f in local.iterdir()
                   if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        print(f"в {local} нет аудио")
        return 1
    target = root / rel
    # 1. снести старое содержимое: игроки кэшируют теги, поэтому заменяем файлы
    #    целиком (свежее событие файла заставляет переиндексировать)
    if target.is_dir():
        for old in target.iterdir():
            if old.is_file() and old.suffix.lower() in (AUDIO | {".jpg", ".jpeg", ".png", ".txt"}):
                rm_phone(root, rel / old.name)
    mtp.mkdir_phone(root, rel)
    # 2. скопировать
    ok = 0
    for f in files + extra:
        good, err = mtp.copy_to_phone(root, f, rel / f.name)
        if good:
            ok += 1
        else:
            print(f"  [ошибка] {f.name}: {err}")
    print(f"скопировано {ok}/{len(files) + len(extra)} файлов → {rel}")
    # 3. проверить чтением обратно
    if verify:
        bad = []
        for f in files[:3] + files[-1:]:
            with tempfile.NamedTemporaryFile(suffix=f.suffix, delete=False) as t:
                tmp = Path(t.name)
            got = mtp.cat_md5(root, rel / f.name, tmp)
            if got != md5f(f):
                bad.append(f.name)
            tmp.unlink(missing_ok=True)
        # список файлов на телефоне
        present = {p.name for p in (root / rel).iterdir()} if (root / rel).is_dir() else set()
        missing = [f.name for f in files if f.name not in present]
        if bad or missing:
            print(f"  [ПРОВЕРКА НЕ ПРОШЛА] md5 не сошёлся: {bad}; нет в списке: {missing}")
            return 1
        print(f"  проверка ок: {len(files)} треков на месте, md5 совпал "
              f"({len(files[:3] + files[-1:])} сверено)")
    return 0


def file_replace(root: Path, local: Path, rel: Path) -> int:
    good, err = mtp.copy_to_phone(root, local, rel)
    if not good:
        print(f"[ошибка] {err}")
        return 1
    with tempfile.NamedTemporaryFile(suffix=local.suffix, delete=False) as t:
        tmp = Path(t.name)
    got = mtp.cat_md5(root, rel, tmp)
    same = got == md5f(local)
    tmp.unlink(missing_ok=True)
    print(f"{'ок' if same else 'ПРОВЕРКА НЕ ПРОШЛА'}: {rel} "
          f"({local.stat().st_size / 1e6:.1f} MB, md5 {'совпал' if same else 'разошёлся'})")
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--album", help="локальный каталог альбома")
    ap.add_argument("--file", help="локальный файл")
    ap.add_argument("--rel", help="путь на телефоне (относительно Music)")
    ap.add_argument("--list", metavar="SUBPATH")
    ap.add_argument("--rm", metavar="SUBPATH")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    root = mtp.phone_dir()
    if root is None:
        print("телефон не найден: подключите в режиме MTP (или проверьте gio mount -l)")
        return 1
    if not mtp.phone_alive(root):
        print(f"корень музыки недоступен: {root}")
        return 1
    print(f"телефон: {root}")

    if args.list:
        d = root / args.list
        if not d.is_dir():
            print(f"нет каталога {args.list}")
            return 1
        for p in sorted(d.iterdir()):
            kind = "dir " if p.is_dir() else f"{p.stat().st_size / 1e6:7.1f}M"
            print(f"  {kind} {p.name}")
        return 0
    if args.rm:
        print("удалено" if rm_phone(root, Path(args.rm)) else "не удалось удалить")
        return 0
    if args.file and args.rel:
        return file_replace(root, Path(args.file).expanduser(), Path(args.rel))
    if args.album and args.rel:
        return album_replace(root, Path(args.album).expanduser(), Path(args.rel),
                             verify=not args.no_verify)
    ap.error("нужен --album/--file с --rel, либо --list/--rm")


if __name__ == "__main__":
    sys.exit(main())
