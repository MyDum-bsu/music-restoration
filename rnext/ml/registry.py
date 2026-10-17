"""Версионирование датасета и снапшоты моделей.

Идея: у каждой модели в метаданных лежит **версия датасета**, на которой она
обучена (`ds-<хэш>` + счётчики), поэтому «какая модель на каких данных» и
«что изменилось между снапшотами» восстанавливается однозначно.

Где что лежит:
  * снапшоты (модель + мета) — на флешке рядом с датасетом:
    `dataset/models/<model_id>/` (чтобы не забивать /home и жить с данными);
  * индекс-реестр (текстовый, маленький) — в репозитории: `models/registry.jsonl`,
    он же источник для лидерборда.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

from rnext import workspace

REGISTRY = Path(__file__).resolve().parents[2] / "models" / "registry.jsonl"


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def models_dir() -> Path:
    """Снапшоты — В РЕПОЗИТОРИИ: откат на любой коммит должен давать рабочую модель."""
    p = Path(__file__).resolve().parents[2] / "models" / "snapshots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifests_dir() -> Path:
    p = Path(__file__).resolve().parents[2] / "dataset_manifests"
    p.mkdir(parents=True, exist_ok=True)
    return p


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return "unknown"


# --- версия датасета --------------------------------------------------------
def _read_plan() -> list[dict]:
    p = workspace.dir_of("corpus") / "dataset_plan.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def compute_dataset_version() -> dict:
    """Хэшируем СОСТАВ датасета (ключи пар + содержимое фич + метки), не mtime."""
    q = dataset_dir() / "reference_quality.jsonl"
    labels = {}
    if q.is_file():
        for line in q.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                labels[r["key"]] = r["label"]

    P = {r["key"]: r for r in (_read_plan())} if True else {}
    items = []
    pairs = 0
    tracks = 0
    trusted_tracks = 0
    by_author: dict[str, int] = {}
    for d in sorted((dataset_dir() / "pairs").glob("*/features.json")):
        key = d.parent.name.replace("__", "/")
        blob = d.read_bytes()
        rows = json.loads(blob)
        lab = labels.get(key, "unlabeled")
        items.append({"key": key, "label": lab, "n": len(rows),
                      "sha1": hashlib.sha1(blob).hexdigest()[:12]})
        pairs += 1
        tracks += len(rows)
        if lab in ("trusted", "weak_base"):
            trusted_tracks += len(rows)
            au = None
            meta = d.parent / "meta.json"
            if meta.is_file():
                au = json.loads(meta.read_text(encoding="utf-8")).get("author_pid")
            if not au:                     # у ранних пар автора нет в мете — берём из плана
                plan = P.get(key) or {}
                au = plan.get("author_pid")
            au = str(au)
            by_author[au] = by_author.get(au, 0) + len(rows)
    payload = json.dumps(items, sort_keys=True, ensure_ascii=False).encode()
    return {
        "version": "ds-" + hashlib.sha1(payload).hexdigest()[:8],
        "pairs": pairs, "tracks": tracks, "trusted_tracks": trusted_tracks,
        "by_author": by_author, "labels": {lab: sum(1 for i in items if i["label"] == lab)
                                           for lab in set(i["label"] for i in items)},
        "items": items,
        "created": int(time.time()),
    }


def write_manifest() -> dict:
    m = compute_dataset_version()
    full = {k: v for k, v in m.items() if k != "items"}
    (dataset_dir() / "manifest.json").write_text(
        json.dumps({**full, "items": m["items"]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    # копию манифеста (с пофайловыми sha1) — в репозиторий: она и есть паспорт
    # версии датасета, по которому воспроизводится любой снапшот
    (manifests_dir() / f"{m['version']}.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    hist = dataset_dir() / "versions.jsonl"
    seen = {json.loads(l)["version"] for l in hist.read_text(encoding="utf-8").splitlines()
            } if hist.is_file() else set()
    if m["version"] not in seen:
        with open(hist, "a", encoding="utf-8") as f:
            f.write(json.dumps(full, ensure_ascii=False) + "\n")
    return full


def current_version() -> dict:
    mf = dataset_dir() / "manifest.json"
    if mf.is_file():
        d = json.loads(mf.read_text(encoding="utf-8"))
        return {k: v for k, v in d.items() if k != "items"}
    return write_manifest()


# --- снапшоты моделей -------------------------------------------------------
def snapshot_id(model: str, target: str, ds_version: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{model}-{target}-{ds_version}-{stamp}"


def save_snapshot(model_obj, meta: dict, model_id: str) -> Path:
    import joblib
    d = models_dir() / model_id
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_obj, d / "model.joblib")
    meta = {**meta, "model_id": model_id, "git": git_commit(), "saved": int(time.time())}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: meta.get(k) for k in
                            ("model_id", "model", "target", "dataset_version",
                             "n_train", "n_hold", "holdout_mode", "metrics",
                             "git", "saved")}, ensure_ascii=False) + "\n")
    return d


def load_snapshot(model_id: str):
    import joblib
    d = models_dir() / model_id
    if not d.is_dir():
        raise FileNotFoundError(f"нет снапшота {model_id}")
    return joblib.load(d / "model.joblib"), json.loads((d / "meta.json").read_text(encoding="utf-8"))


def list_snapshots() -> list[dict]:
    if not REGISTRY.is_file():
        return []
    return [json.loads(l) for l in REGISTRY.read_text(encoding="utf-8").splitlines() if l.strip()]


def latest_snapshot(model: str | None = None, target: str = "dcrest") -> str | None:
    rows = [r for r in list_snapshots() if r["target"] == target
            and (model is None or r["model"] == model)]
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("saved") or 0)
    return rows[-1]["model_id"]


def already_trained(ds_version: str, model: str, target: str = "dcrest") -> bool:
    return any(r["dataset_version"] == ds_version and r["model"] == model
               and r["target"] == target for r in list_snapshots())
