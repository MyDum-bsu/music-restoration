"""Dataset versioning and model snapshots.

Idea: each model's metadata carries the **dataset version** it was trained on
(`ds-<hash>` + counters), so "which model on which data" and "what changed
between snapshots" can be reconstructed unambiguously.

What lives where:
  * snapshots (model + meta) — on the USB drive next to the dataset:
    `dataset/models/<model_id>/` (keeps /home uncluttered, stays with the data);
  * the index registry (text, small) — in the repo: `models/registry.jsonl`,
    also the source for the leaderboard.
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
    """Snapshots live IN THE REPO: a checkout of any commit must yield a working model."""
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


# --- dataset version --------------------------------------------------------
def _read_plan() -> list[dict]:
    p = workspace.dir_of("corpus") / "dataset_plan.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def compute_dataset_version() -> dict:
    """Hash the dataset COMPOSITION (pair keys + feature contents + labels), not mtimes."""
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
            if not au:                     # early pairs have no author in meta — take it from the plan
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
    # copy the manifest (with per-file sha1s) into the repo: it is the passport
    # of the dataset version from which any snapshot can be reproduced
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


# --- model snapshots --------------------------------------------------------
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
        # promote and the gate verdict are REQUIRED in the registry: without
        # them the runtime cannot see gate-passing models on a fresh dataset
        # and rolls back to the old model
        f.write(json.dumps({k: meta.get(k) for k in
                            ("model_id", "model", "target", "dataset_version",
                             "n_train", "n_hold", "holdout_mode", "metrics",
                             "promote", "gate_vs_best", "gate_ci_low", "level",
                             "git", "saved")}, ensure_ascii=False) + "\n")
    return d


def load_snapshot(model_id: str):
    import joblib
    d = models_dir() / model_id
    if not d.is_dir():
        raise FileNotFoundError(f"no snapshot {model_id}")
    return joblib.load(d / "model.joblib"), json.loads((d / "meta.json").read_text(encoding="utf-8"))


def list_snapshots() -> list[dict]:
    if not REGISTRY.is_file():
        return []
    return [json.loads(l) for l in REGISTRY.read_text(encoding="utf-8").splitlines() if l.strip()]


# Preference order for the runtime: stable models first, knn last (it is the
# best on by-album, but that is memorization: train error 0, and on the
# cross-author split it loses the most).
MODEL_PREFERENCE = ("ridge", "hgb", "hgbq", "conformal", "table", "knn")


def latest_snapshot(model: str | None = None, target: str = "dcrest",
                    promoted_only: bool = True) -> str | None:
    """Latest snapshot. By default only GATE-PASSING ones.

    Otherwise the runtime would take the last recorded model, which may be a
    cross-author split the gate did not pass (and which is worse than the
    baseline).
    """
    rows = [r for r in list_snapshots() if r["target"] == target
            and (model is None or r["model"] == model)]
    if promoted_only:
        ok = [r for r in rows if r.get("promote")]
        if not ok:
            return None
        rows = ok
    if not rows:
        return None
    # the runtime serves unseen albums, so prefer the by-album split
    album = [r for r in rows if r.get("holdout_mode") == "by-album"]
    rows = album or rows
    # 1) newest dataset version (more data), 2) best median, 3) model order
    # (knn must not win ties through memorization)
    newest = max(r.get("saved") or 0 for r in rows)
    ver = max((r.get("dataset_version") or "" for r in rows
               if (r.get("saved") or 0) == newest), default="")
    same = [r for r in rows if (r.get("dataset_version") or "") == ver] or rows
    def rank(r):
        med = ((r.get("metrics") or {}).get("all") or {}).get("median_abs")
        m = r.get("model") or ""
        pref = MODEL_PREFERENCE.index(m) if m in MODEL_PREFERENCE else len(MODEL_PREFERENCE)
        return (med if med is not None else 9e9, pref, -(r.get("saved") or 0))
    same.sort(key=rank)
    return same[0]["model_id"]


def already_trained(ds_version: str, model: str, target: str = "dcrest") -> bool:
    return any(r["dataset_version"] == ds_version and r["model"] == model
               and r["target"] == target for r in list_snapshots())
