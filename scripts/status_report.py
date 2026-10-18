#!/usr/bin/env python3
"""Полный отчёт о состоянии проекта restore-next одной командой.

Секции: флешка и бюджет · датасет · план и журнал · качка · процессы ·
модели · (в --deep: целостность данных, тесты, сквозной прогон пайплайна).

Каждая секция в своём try/except — сбой одной не ломает отчёт.

  python3 scripts/status_report.py            # быстрый отчёт
  python3 scripts/status_report.py --deep     # + целостность, тесты, прогон
"""
import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rnext import workspace  # noqa: E402


def h(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def ok(txt: str) -> None:
    print(f"  \033[32m✓\033[0m {txt}")


def warn(txt: str) -> None:
    print(f"  \033[33m!\033[0m {txt}")


def bad(txt: str) -> None:
    print(f"  \033[31m✗\033[0m {txt}")


def human_gb(n: float) -> str:
    return f"{n:.1f} GB"


# --- секции -----------------------------------------------------------------
def sec_disk() -> None:
    h("1. Флешка и бюджет")
    try:
        root = workspace.data_root()
        # точка монтирования может быть выше data_root (у нас /media/mydum/Data)
        mp = root
        while mp != mp.parent and not mp.is_mount():
            mp = mp.parent
        mounted = mp.is_mount() and root.is_dir()
        if mounted:
            line = subprocess.run(["df", "-h", str(root)], capture_output=True,
                                  text=True).stdout
            ok(f"смонтирована в {mp}: {line.strip().splitlines()[-1]}")
        else:
            bad(f"{root} НЕ смонтирована — scripts/repair_flash.sh")
        used = workspace.usage_bytes()
        budget = workspace.budget_bytes()
        pct = used / budget * 100
        msg = f"бюджет датасета: {used / 2**30:.1f} / {budget / 2**30:.0f} ГиБ ({pct:.0f}%)"
        (ok if pct < 85 else warn)(msg)
        free = subprocess.run(["df", "-BG", "--output=avail", str(root)],
                              capture_output=True, text=True).stdout.strip().splitlines()[-1]
        (ok if int(free.rstrip("G")) > 12 else warn)(f"свободно на разделе: {free}")
    except Exception as e:                                # noqa: BLE001
        bad(f"ошибка: {type(e).__name__}: {e}")


def sec_dataset() -> None:
    h("2. Датасет")
    try:
        D = workspace.data_root() / "dataset"
        pairs = sorted(D.glob("pairs/*/features.json"))
        rows = []
        for f in pairs:
            rows += json.loads(f.read_text(encoding="utf-8"))
        ex = list(D.glob("pairs/*/excerpts/*.flac"))
        cls = Counter("healthy" if r["original"]["dr"] >= 12 else
                      ("mid" if r["original"]["dr"] >= 9 else "crushed") for r in rows)
        man = D / "manifest.json"
        ver = json.loads(man.read_text(encoding="utf-8")).get("version") if man.is_file() else "?"
        ok(f"пар {len(pairs)}, треков {len(rows)}, вырезок {len(ex)} | версия {ver}")
        print(f"    классы исходника: {dict(cls)}")
        for target, name in ((300, "M1 ridge/knn"), (760, "M2 hgb"), (1500, "M3 кросс-автор")):
            left = target - len(rows)
            (ok if left <= 0 else warn)(f"{name}: {'достигнут' if left <= 0 else f'ещё {left} треков'}")
        sz = {k: sum(f.stat().st_size for f in (D / k).rglob("*") if f.is_file())
              for k in ("pairs", "features", "raw")}
        print(f"    фичи {sz['features'] / 1e6:.1f} MB | пары+вырезки {sz['pairs'] / 1e6:.0f} MB | "
              f"сырьё (транзит) {human_gb(sz['raw'] / 1e9)}")
    except Exception as e:                                # noqa: BLE001
        bad(f"ошибка: {type(e).__name__}: {e}")


def sec_plan() -> None:
    h("3. План и журнал")
    try:
        from rnext import sides as S
        C = workspace.dir_of("corpus")
        D = workspace.data_root() / "dataset"
        plan = [json.loads(l) for l in (C / "dataset_plan.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        st = Counter(r["status"] for r in plan)
        ok(f"план: matched {st.get('matched', 0) + st.get('matched_graphy', 0)} "
           f"(альбомных {st.get('matched', 0)}, дискографий {st.get('matched_graphy', 0)}), "
           f"weak {st.get('weak', 0)}, без оригинала {st.get('no_original', 0)}")
        mt = [json.loads(l) for l in (C / "master_targets.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()] if (C / "master_targets.jsonl").is_file() else []
        if mt:
            ok(f"цели мастера без аудио: релизов {len(mt)}, треков {sum(r['n_tracks'] for r in mt)}")
        state = S.merged_state(D / "state.jsonl")
        have = {p.parent.name.replace("__", "/") for p in (D / "pairs").glob("*/features.json")}
        c = Counter("в датасете" if k in have else r.get("state", "?") for k, r in state.items())
        ok(f"журнал: заведено {len(state)} пар — {dict(c)}")
        paused = c.get("paused_space", 0)
        if paused:
            warn(f"{paused} пар отложены по месту/мёртвым сидам — вернутся в работу сами")
    except Exception as e:                                # noqa: BLE001
        bad(f"ошибка: {type(e).__name__}: {e}")


def sec_downloads() -> None:
    h("4. Качка")
    try:
        from rnext.transm import Transm
        t = Transm()
        ours = [x for x in t.torrents(["id", "name", "downloadDir", "leftUntilDone",
                                       "rateDownload", "status", "peersConnected"])
                if "/dataset/raw/" in (x["downloadDir"] or "")]
        active = [x for x in ours if x["status"] == 4]
        left = sum(x["leftUntilDone"] for x in ours)
        live = sum(x["leftUntilDone"] for x in ours if (x.get("peersConnected") or 0) > 0)
        rate = sum(x["rateDownload"] for x in ours)
        dead = [x for x in ours if x["leftUntilDone"] > 0 and not (x.get("peersConnected") or 0)]
        ok(f"раздач {len(ours)}, активных {len(active)}, скорость {rate / 1e6:.1f} MB/s")
        ok(f"в работе {human_gb(left / 1e9)}, из них с пирами {human_gb(live / 1e9)} "
           f"(кап — по живым)")
        if dead:
            warn(f"без пиров {len(dead)} раздач — их снимет чистка мёртвых сидов")
    except Exception as e:                                # noqa: BLE001
        bad(f"ошибка (Transmission RPC): {type(e).__name__}: {str(e)[:80]}")


def sec_processes() -> None:
    h("5. Процессы")
    try:
        for pat, name in (("[d]ataset_loop.py", "цикл обогащения"),
                          ("[w]atchdog.sh", "сторож"),
                          ("[t]ransmission-gtk", "Transmission"),
                          ("restore-next/app/node_modules/[e]lectron/dist", "плеер")):
            n = subprocess.run(["pgrep", "-cf", pat], capture_output=True, text=True).stdout.strip()
            (ok if int(n or 0) > 0 else warn)(f"{name}: {'работает' if int(n or 0) > 0 else 'не запущен'}")
        log = workspace.data_root() / "dataset" / "loop.log"
        if log.is_file():
            tail = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                    if l.strip()][-3:]
            print("    последние строки цикла:")
            for l in tail:
                print(f"      {l[:150]}")
    except Exception as e:                                # noqa: BLE001
        bad(f"ошибка: {type(e).__name__}: {e}")


def sec_models() -> None:
    h("6. Модели")
    try:
        from rnext.ml import registry as reg
        rows = reg.list_snapshots()
        prom = [r for r in rows if r.get("promote")]
        ok(f"снапшотов {len(rows)}, прошедших гейт {len(prom)}")
        for r in prom[-3:]:
            m = r["metrics"]["all"]
            ok(f"PROMOTE {r['model']}/{r['holdout_mode']} ds={r['dataset_version']} "
               f"n={r['n_train']} медиана {m['median_abs']} дБ "
               f"(выигрыш {r.get('gate_gain_db'):+.2f}, CI низ {r.get('gate_ci_low'):+.2f})")
        mid = reg.latest_snapshot()
        if mid:
            _, meta = reg.load_snapshot(mid)
            ok(f"рантайм выберет: {meta['model']} ({mid})")
        else:
            warn("нет продвинутой модели — в проде работает классовая таблица "
                 "(это норма, пока гейт не пройден)")
        lb = ROOT / "models" / "leaderboard.jsonl"
        if lb.is_file():
            last = [json.loads(l) for l in lb.read_text(encoding="utf-8").splitlines() if l.strip()]
            if last:
                print("    последний прогон обучения:")
                for r in last[-4:]:
                    print(f"      {r['holdout_mode']:<9} {r['model']:<6} n={r['n_train']:<4} "
                          f"медиана {r['median_abs']} | базы const46 {r['baseline_const46']}, "
                          f"v0 {r['baseline_v0table']} | promote={r.get('promote')}")
    except Exception as e:                                # noqa: BLE001
        bad(f"ошибка: {type(e).__name__}: {e}")


def sec_deep() -> None:
    h("7. Глубокая проверка (--deep)")
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts/dataset_audit.py"), "--integrity"],
                           capture_output=True, text=True, cwd=str(ROOT), timeout=900)
        for line in (r.stdout or "").strip().splitlines()[-2:]:
            (ok if "проблем нет" in line else warn)(line.strip())
    except Exception as e:                                # noqa: BLE001
        bad(f"целостность: {type(e).__name__}")
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                           capture_output=True, text=True, cwd=str(ROOT), timeout=600)
        line = (r.stdout or "").strip().splitlines()[-1]
        (ok if "passed" in line else bad)(f"тесты DSP: {line}")
    except Exception as e:                                # noqa: BLE001
        bad(f"тесты: {type(e).__name__}")
    # сквозной прогон пайплайна с моделью на синтетическом раздавленном треке
    try:
        import numpy as np
        import wave
        tmp = Path("/tmp/status_smoke")
        tmp.mkdir(parents=True, exist_ok=True)
        sr = 44100
        t = np.arange(sr * 20) / sr
        base = 0.5 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)
        hit = np.exp(-np.arange(400) / 80.0) * np.sin(2 * np.pi * 900 * np.arange(400) / sr) * 0.8
        x = base.copy()
        for i in range(0, sr * 20 - 500, sr // 2):
            x[i:i + 400] += hit
        x = (np.clip(x * 1.0, -1, 1) * 0.98)    # раздавленный, как в корпусе (DR ~5)
        w = wave.open(str(tmp / "tmp.wav"), "wb")
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.stack([x, x], 1) * 32767).astype("<i2").tobytes())
        w.close()
        subprocess.run(["flac", "-s", "-f", "-o", str(tmp / "01. smoke.flac"),
                        str(tmp / "tmp.wav")], check=True)
        out = tmp / "out"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/restore_one.py"),
                            "--src", str(tmp), "--out", str(out), "--ml-snapshot", "auto"],
                           capture_output=True, text=True, cwd=str(ROOT), timeout=1800)
        rep = out / "restore_report.jsonl"
        if rep.is_file():
            rec = json.loads(rep.read_text(encoding="utf-8").splitlines()[0])
            tp_ok = rec["tp_actual"] <= -0.45
            dr_ok = rec["rest"]["dr"] - rec["orig"]["dr"] >= 2
            ml = rec.get("ml_target")
            line = (f"сквозной прогон: класс {rec['class']}, цель {rec['target_dcrest']:.2f}"
                    + (f" ({ml:.2f}, источник: {rec.get('ml_source')})" if ml else "")
                    + f", r_fit {round(rec.get('r_fit') or 0, 2)}, TP {rec['tp_actual']}, "
                      f"DR {rec['orig']['dr']}→{rec['rest']['dr']}")
            (ok if (tp_ok and dr_ok) else bad)(line + ("" if tp_ok and dr_ok else "  ← ПРОВЕРИТЬ"))
        else:
            warn("сквозной прогон не дал отчёта")
    except Exception as e:                                # noqa: BLE001
        bad(f"сквозной прогон: {type(e).__name__}: {str(e)[:80]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true")
    args = ap.parse_args()
    print(f"\033[1mrestore-next — состояние на {time.strftime('%d.%m %H:%M')}\033[0m")
    sec_disk()
    sec_dataset()
    sec_plan()
    sec_downloads()
    sec_processes()
    sec_models()
    if args.deep:
        sec_deep()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
