#!/usr/bin/env python3
"""Непрерывное обогащение датасета + авто-обучение по мере роста.

Один проход цикла:
  1. если у пар ещё качается сырьё — ждём;
  2. иначе: выбрать новую партию (dataset_select) → загрузить (dataset_fetch --go);
  3. извлечь готовые пары (--keep-raw, окно хранения сырья под оценку);
  4. обновить аудит референсов и целевой слой (reference_audit, target_model);
  5. пересчитать версию датасета и прогнать авто-обучение с эвалами (ml_auto).

Идемпотентно: повторный запуск продолжает с того же места. Логи — в `dataset/loop.log`.

  python3 scripts/dataset_loop.py --interval 900 --batch-gb 12 --keep-window 12
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import workspace  # noqa: E402
from rnext.transm import Transm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def log(msg: str) -> None:
    line = f"[{time.strftime('%d.%m %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(dataset_dir() / "loop.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], timeout: int = 7200) -> int:
    log("$ " + " ".join(cmd[1:]))
    p = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout)
    return p.returncode


def downloads_state() -> tuple[float, float, float]:
    """(осталось всего GB, скорость MB/s, осталось у РАЗДАЧ С ПИРАМИ).

    Третье число — то, что реально может докачаться: мёртвые сиды (0 пиров)
    не должны блокировать кап «в работе», иначе одна застрявшая раздача
    останавливает весь конвейер.
    """
    try:
        t = Transm()
        left = rate = live = 0.0
        for tor in t.torrents(["downloadDir", "leftUntilDone", "rateDownload",
                               "peersConnected"]):
            if "/dataset/raw/" not in (tor["downloadDir"] or ""):
                continue
            left += tor["leftUntilDone"]
            rate += tor["rateDownload"]
            if (tor.get("peersConnected") or 0) > 0:
                live += tor["leftUntilDone"]
        return left / 1e9, rate / 1e6, live / 1e9
    except Exception:                                    # noqa: BLE001
        return 0.0, 0.0, 0.0


STALLS = 0


def batch_counter() -> int:
    n = 0
    for p in (workspace.dir_of("corpus")).glob("dataset_batch_*.jsonl"):
        try:
            n = max(n, int(p.stem.rsplit("-", 1)[-1]))
        except ValueError:
            continue
    return n


def free_gb() -> float:
    import shutil
    try:
        return shutil.disk_usage(str(workspace.data_root())).free / 1e9
    except Exception:                                    # noqa: BLE001
        return 0.0


def one_pass(args) -> None:
    """Один проход: чистка → извлечение/аудит/обучение (ВСЕГДА) → возможно новая партия.

    Раньше проход выходил рано, пока шла качка, — из-за этого готовые пары
    часами лежали неизвлечёнными, а партии копились, пока диск не кончился.
    Теперь обработка идёт каждую проходку, а новая партия берётся только если
    в работе меньше max_inflight_gb.
    """
    global STALLS
    free = free_gb()
    if free < args.min_free_gb:
        log(f"свободно {free:.1f} GB < {args.min_free_gb} GB — чистим сырьё")
        subprocess.run([sys.executable, "scripts/dataset_extract.py", "--prune",
                        "--prune-keep", str(args.keep_window)], cwd=str(ROOT), check=False)
        if free_gb() < args.min_free_gb:
            # критично: сносим и НЕЗАВЕРШЁННЫЕ пары с малой готовностью
            log("места всё ещё мало — снимаю незавершённые пары с низкой готовностью")
            subprocess.run([sys.executable, "scripts/dataset_prune_stalled.py",
                            "--max-progress", "0.5"], cwd=str(ROOT), check=False)

    # 0) мёртвые сиды не должны держать кап: снимаем пары без пиров
    subprocess.run([sys.executable, "scripts/dataset_prune_stalled.py", "--peerless"],
                   cwd=str(ROOT), check=False, stdout=subprocess.DEVNULL)
    # ...и сразу пересчитываем wanted-наборы: без этого раздачи снятых пар
    # продолжают качать в удалённые каталоги (сырьё возвращается, место утекает)
    subprocess.run([sys.executable, "scripts/dataset_fetch.py", "--repair"],
                   cwd=str(ROOT), check=False, stdout=subprocess.DEVNULL)

    # 1) обработка готового — каждую проходку, независимо от качки
    run([sys.executable, "scripts/dataset_extract.py", "--all", "--excerpts", "1",
         "--keep-raw", "--keep-raw-window", str(args.keep_window)])
    run([sys.executable, "scripts/reference_audit.py"])
    run([sys.executable, "scripts/target_model.py"])
    run([sys.executable, "scripts/ml_auto.py", "--with-audio"])
    man = dataset_dir() / "manifest.json"
    if man.is_file():
        m = json.loads(man.read_text(encoding="utf-8"))
        log(f"датасет {m['version']}: пар {m['pairs']}, trusted-треков {m['trusted_tracks']}")

    # 2) решаем, брать ли новую партию
    left, rate, live = downloads_state()
    if live > args.max_inflight_gb:
        log(f"в работе (с пирами) {live:.1f} GB > капа {args.max_inflight_gb} GB — "
            f"новую партию не берём")
        return
    if left > 0.05 and rate >= args.min_rate_mbps:
        log(f"качается {left:.2f} GB ({rate:.2f} MB/s) — ждём")
        return
    if left > 0.05:
        STALLS += 1
        if STALLS < args.stall_passes:
            log(f"качается {left:.2f} GB, но медленно ({rate:.2f} MB/s; "
                f"{STALLS}/{args.stall_passes}) — ждём")
            return
        log(f"медленный хвост подтверждён ({rate:.2f} MB/s) — берём новую партию")
    STALLS = 0

    n = batch_counter() + 1
    tag = f"batch-{n:03d}"
    if run([sys.executable, "scripts/dataset_select.py", "--limit", str(args.batch_pairs),
            "--min-master-seeders", str(args.min_master_seeders),
            "--max-master-gb", str(args.max_master_gb), "--out", tag]) != 0:
        log("select не сработал")
        return
    plan = workspace.dir_of("corpus") / f"dataset_batch_{tag}.jsonl"
    run([sys.executable, "scripts/dataset_fetch.py", "--go", "--limit", str(args.batch_pairs),
         "--max-gb", str(args.batch_gb), "--plan", str(plan)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--batch-pairs", type=int, default=14)
    ap.add_argument("--batch-gb", type=float, default=12.0)
    ap.add_argument("--keep-window", type=int, default=12)
    ap.add_argument("--min-master-seeders", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--min-rate-mbps", type=float, default=2.0,
                    help="ниже этой скорости качка считается медленным хвостом")
    ap.add_argument("--max-inflight-gb", type=float, default=12.0,
                    help="кап на объём незакрытых загрузок: больше — новую партию не берём")
    ap.add_argument("--max-master-gb", type=float, default=6.0,
                    help="не брать пары, у которых тема мастера больше этого (гигантские дискографии)")
    ap.add_argument("--stall-passes", type=int, default=2,
                    help="сколько проходов ждать, если качается, но скорость 0")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    log(f"цикл обогащения: интервал {args.interval}s, партия {args.batch_pairs} пар "
        f"/ {args.batch_gb} GB, сырьё держим у {args.keep_window} последних пар")
    # Настройки Transmission (очередь/DHT) живут только до перезапуска клиента —
    # после ребута они возвращаются к дефолтам и скорость падает в ~20 раз.
    # Применяем при каждом старте цикла: и в сессии, и в settings.json.
    try:
        subprocess.run([sys.executable, "scripts/transmission_tune.py", "--apply"],
                       cwd=str(ROOT), check=False, timeout=60)
        log("настройки Transmission применены")
    except Exception as e:                                 # noqa: BLE001
        log(f"не удалось применить настройки Transmission: {type(e).__name__}")
    while True:
        try:
            one_pass(args)
        except Exception as e:                            # noqa: BLE001
            log(f"ошибка прохода: {type(e).__name__}: {str(e)[:200]}")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
