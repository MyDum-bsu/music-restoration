#!/usr/bin/env python3
"""Пакетный прогон релизов: реставрация → оформление → телефон, по очереди.

  python3 scripts/release_batch.py \
      --job "release_src/City and Colour/2005 - Sometimes|City and Colour/2005 - Sometimes|City and Colour - Sometimes (2005)" \
      --job "..." \
      --log /tmp/batch.log --clean-src

Каждый job: «каталог источника | путь на телефоне | имя релиза». Шаги:
  1. если в источнике image+cue — разбить на треки (образ уводится в _image/);
  2. restore_one.py с продовыми настройками (авто-политика, модель, TP −0.5);
  3. finish_release.py — теги, обложка, тексты;
  4. phone_sync.py — замена альбома на телефоне с проверкой md5;
  5. --clean-src — удалить исходник (освободить место; на телефоне уже есть).

Ничего не «догадывается»: если обложки нет — релиз уходит без неё и это
попадает в отчёт; если тексты не нашлись — тоже в отчёт.
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
DATA = workspace.data_root()


def run(cmd: list[str], log, timeout: int = 7200) -> tuple[int, str]:
    log.write(f"$ {' '.join(cmd)}\n")
    log.flush()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    log.write(out[-4000:] + "\n")
    log.flush()
    return r.returncode, out


def flacs(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*.flac"))


def find_discs(src: Path) -> list[tuple[str, Path]]:
    """Где лежат FLAC: сам каталог, вложенный релиз или диски (CD1/CD2).

    Возвращает [(метка диска, каталог)]; метка пустая для однодискового.
    """
    if flacs(src):
        return [("", src)]
    # ищем на уровень-два вниз каталоги с треками
    cands = []
    for d in sorted(src.rglob("*")):
        if d.is_dir() and flacs(d):
            cands.append(d)
    if not cands:
        return []
    # оставляем только «листовые» (без вложенных каталогов с треками)
    leaves = [d for d in cands if not any(o != d and d in o.parents for o in cands)]
    discs = [d for d in leaves if re.match(r"^(cd|disc|disk)\s*\d+$", d.name, re.I)]
    if discs:
        return [(d.name.upper(), d) for d in sorted(discs)]
    if len(leaves) == 1:
        return [("", leaves[0])]
    # Несколько подпапок без меток CD1/CD2 (например «Steel The Light» + «mini»):
    # их НЕЛЬЗЯ вести в один каталог телефона — второй затрёт первый. Берём имя
    # подпапки как метку, чтобы каждая ушла в свой подкаталог.
    return [(d.name, d) for d in sorted(leaves)]


def split_image(src: Path, log) -> Path:
    """image+cue → треки рядом; образ уводится в _image/, чтобы не попал в --src."""
    cues = list(src.glob("*.cue"))
    imgs = [p for p in src.glob("*.flac") if p.stat().st_size > 80 * 1024 * 1024]
    if not cues or not imgs:
        return src
    img = imgs[0]
    code, out = run(["shntool", "split", "-f", str(cues[0]), "-o", "flac",
                     "-t", "%n - %t", str(img)], log)
    made = [p for p in src.glob("[0-9]*.flac") if p != img]
    if not made:
        log.write(f"[warn] разбиение не дало треков: {out[-300:]}\n")
        return src
    hold = src / "_image"
    hold.mkdir(exist_ok=True)
    for f in [img, cues[0]] + list(src.glob("*.log")) + list(src.glob("*.accurip")):
        if f.is_file() and f.parent == src:
            shutil.move(str(f), str(hold / f.name))
    run(["cuetag", str(hold / cues[0].name)] + [str(p) for p in made], log)
    log.write(f"разбито на {len(made)} треков\n")
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
        res["error"] = "нет каталога источника"
        log.write(f"[error] {src}\n")
        return res
    discs = find_discs(src)
    if not discs:
        res["error"] = "в источнике нет FLAC"
        return res
    res["tracks"] = sum(len(flacs(d)) for _, d in discs)
    res["discs"] = [label or "-" for label, _ in discs]
    res["summary"] = []
    res["lyrics"] = []
    res["cover"] = True

    for label, ddir in discs:
        if not flacs(ddir):
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
            res["error"] = f"реставрация не дала треков ({label or 'cd1'})"
            return res
        code, out2 = run([sys.executable, "-u", str(SCRIPTS / "finish_release.py"),
                          "--restored", str(d_out), "--source", str(ddir), "--lyrics"], log)
        res["steps"][f"finish:{label or '-'}"] = code
        if "не найдена" in out2:
            res["cover"] = False
        m = re.search(r"тексты: ([^\n]+)", out2)
        if m:
            res["lyrics"].append(f"{label}:{m.group(1)}" if label else m.group(1))
        m = re.search(r"итог: ([^\n]+)", out2)
        if m:
            res["summary"].append(f"{label}: {m.group(1)}" if label else m.group(1))
        if d_phone and not skip_phone:
            code, out3 = run([sys.executable, "-u", str(SCRIPTS / "phone_sync.py"),
                              "--album", str(d_out), "--rel", d_phone], log)
            res["steps"][f"phone:{label or '-'}"] = code
            res["phone_ok"] = res.get("phone_ok", True) and "проверка ок" in out3
            if code != 0:
                res["error"] = f"копирование на телефон не прошло проверку ({label or 'cd1'})"
                return res
    res["summary"] = "; ".join(res["summary"])
    res["lyrics"] = "; ".join(res["lyrics"])

    if clean_src and res.get("phone_ok"):
        shutil.rmtree(src, ignore_errors=True)
        res["src_removed"] = True
        log.write(f"источник удалён (на телефоне есть): {src}\n")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True,
                    help="«источник | путь на телефоне | имя релиза»")
    ap.add_argument("--log", default="/tmp/release_batch.log")
    ap.add_argument("--clean-src", action="store_true")
    ap.add_argument("--skip-phone", action="store_true")
    ap.add_argument("--jobs", type=int, default=3,
                    help="треков параллельно внутри альбома (результат не зависит "
                         "от числа потоков; память ~0.5 ГБ на поток)")
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
    print(f"\n{'=' * 80}\nИТОГ:")
    for r in results:
        status = "ок" if not r.get("error") else f"ОШИБКА: {r['error']}"
        print(f"  {r.get('name', r['job'])[:44]:<46} {status}")
        if r.get("summary"):
            print(f"      {r['summary']}  | обложка: {'да' if r.get('cover') else 'НЕТ'}, "
                  f"тексты: {r.get('lyrics')}")
    if args.report:
        Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
