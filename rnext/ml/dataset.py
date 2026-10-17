"""Обучающая таблица датасета: пары → (признаки, цель, группа, страта).

Единица наблюдения — трек. Группа (альбом) обязательна: без неё любой сплит
течёт, потому что треки одного альбома коррелированы.

Цели:
  * `dcrest` — Δcrest мастера, дБ (основная);
  * `ddr`    — ΔDR мастера;
  * `tilt`   — 30 band-дельт мастера (отдельный многомерный выход).

В таблицу попадают только `trusted`-пары: метки радикальных реставраций
искажены, и учить по ним цель нельзя (см. docs/dataset.md, аудит референсов).
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rnext import workspace

HEALTHY_DR = 12
LOW_BANDS = slice(7, 14)      # ~125–500 Гц
HIGH_BANDS = slice(21, 28)    # ~2–8 кГц

# Признаки: имя → путь в строке фич (original) или производное
BASE_FEATURES = [
    "dr", "dr_exact", "crest_db", "rms_db", "peak_db", "dur_s",
]
DERIVED_SCALARS = [
    "clip_samples", "clip_events", "longest_run",
    "rms_p05", "rms_p50", "rms_p95", "rms_spread",
    "attacks_per_s", "flux_per_s", "rms_range_db",
    "side_mid_db", "side_mid_hf_db", "tp_overshoot_db",
    "tilt_low_high", "tilt_slope",
]
FEATURES = BASE_FEATURES + DERIVED_SCALARS + ["n_tracks"]


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def corpus_dir() -> Path:
    return workspace.dir_of("corpus")


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_quality() -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(dataset_dir() / "reference_quality.jsonl")}


def plan_index() -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(corpus_dir() / "dataset_plan.jsonl")}


def parse_band_freqs() -> np.ndarray:
    edges = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
             800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
             12500, 16000, 20000]
    c = np.sqrt(np.array(edges[:-1]) * np.array(edges[1:]))
    return c


def tilt_features(bands: list[float] | None) -> tuple[float, float]:
    """Наклон спектра: low−high (дБ) и коэффициент линейной регрессии по log-f."""
    if not bands:
        return float("nan"), float("nan")
    b = np.asarray(bands, dtype=float)
    lo = float(np.mean(b[LOW_BANDS]))
    hi = float(np.mean(b[HIGH_BANDS]))
    f = np.log(parse_band_freqs())
    m = np.isfinite(b)
    if m.sum() < 5:
        return lo - hi, float("nan")
    slope = float(np.polyfit(f[m], b[m], 1)[0])
    return lo - hi, slope


def _row_features(row: dict, n_tracks: int) -> dict:
    o = row["original"]
    geom = o.get("clip_geom") or {}
    blocks = o.get("blocks") or {}
    tr = o.get("transient") or {}
    st = o.get("stereo") or {}
    tl, ts = tilt_features(o.get("bands_db"))
    cl = o.get("clip") or {}
    clip_samples = sum((cl.get(k) or {}).get("clipped_samples", 0) for k in ("L", "R"))
    clip_events = sum((cl.get(k) or {}).get("clip_events", 0) for k in ("L", "R"))
    longest = max((geom.get(k) or {}).get("longest_run", 0) for k in ("L", "R")) if geom else 0
    return {
        "dr": o["dr"], "dr_exact": o.get("dr_exact", np.nan),
        "crest_db": o["crest_db"], "rms_db": o["rms_db"], "peak_db": o["peak_db"],
        "dur_s": o.get("dur_s", np.nan),
        "clip_samples": clip_samples, "clip_events": clip_events, "longest_run": longest,
        "rms_p05": blocks.get("p05", np.nan), "rms_p50": blocks.get("p50", np.nan),
        "rms_p95": blocks.get("p95", np.nan), "rms_spread": blocks.get("spread", np.nan),
        "attacks_per_s": tr.get("attacks_per_s", np.nan),
        "flux_per_s": tr.get("flux_per_s", np.nan),
        "rms_range_db": tr.get("rms_range_db", np.nan),
        "side_mid_db": st.get("side_mid_db", np.nan),
        "side_mid_hf_db": st.get("side_mid_hf_db", np.nan),
        "tp_overshoot_db": o.get("tp_overshoot_db", np.nan),
        "tilt_low_high": tl, "tilt_slope": ts,
        "n_tracks": n_tracks,
    }


@dataclass
class Table:
    keys: list[str]
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    authors: np.ndarray
    classes: np.ndarray
    feature_names: list[str]
    y_bands: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.keys)

    def subset(self, idx: np.ndarray) -> "Table":
        return Table(
            keys=[self.keys[i] for i in idx],
            X=self.X[idx], y=self.y[idx], groups=self.groups[idx],
            authors=self.authors[idx], classes=self.classes[idx],
            feature_names=self.feature_names,
            y_bands=None if self.y_bands is None else self.y_bands[idx],
        )


def class_of(dr: float) -> str:
    if dr < 9:
        return "crushed"
    if dr < HEALTHY_DR:
        return "mid"
    return "healthy"


def build_table(target: str = "dcrest", trusted_only: bool = True) -> Table:
    quality = load_quality()
    plan = plan_index()
    keys: list[str] = []
    X: list[list[float]] = []
    y: list[float] = []
    groups: list[str] = []
    authors: list[str] = []
    classes: list[str] = []
    bands: list[list[float]] = []

    for d in sorted((dataset_dir() / "pairs").glob("*/features.json")):
        key = d.parent.name.replace("__", "/")
        q = quality.get(key)
        if trusted_only and q and q["label"] not in ("trusted", "weak_base"):
            continue
        rows = json.loads(d.read_text(encoding="utf-8"))
        if not rows:
            continue
        pi = plan.get(key) or {}
        n_tracks = len(pi.get("release", {}).get("tracks") or []) or len(rows)
        for r in rows:
            if target == "dcrest":
                val = r["delta"]["crest"]
            elif target == "ddr":
                val = r["delta"]["dr"]
            else:
                raise ValueError(f"неизвестная цель: {target}")
            keys.append(f"{key}#t{r['track']}")
            X.append([_row_features(r, n_tracks)[f] for f in FEATURES])
            y.append(float(val))
            groups.append(key)
            authors.append(str(pi.get("author_pid")))
            classes.append(class_of(r["original"]["dr"]))
            bands.append(list(r["delta"]["bands_db"]))
    n = len(keys)
    return Table(keys=keys,
                 X=np.asarray(X, dtype=float).reshape(n, len(FEATURES)),
                 y=np.asarray(y, dtype=float),
                 groups=np.asarray(groups), authors=np.asarray(authors),
                 classes=np.asarray(classes), feature_names=list(FEATURES),
                 y_bands=np.asarray(bands, dtype=float) if bands else None)


def split(table: Table, mode: str = "by-album", frac: float = 0.3,
          seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """train/holdout. `by-album` — 30 % альбомов; `by-author` — один автор целиком."""
    rng = np.random.default_rng(seed)
    if mode == "by-author":
        authors = sorted(set(table.authors))
        if len(authors) < 2:
            raise ValueError("для by-author нужно ≥2 автора в таблице")
        hold_a = authors[int(rng.integers(0, len(authors)))]
        hold = np.where(table.authors == hold_a)[0]
        train = np.where(table.authors != hold_a)[0]
        return train, hold
    albums = sorted(set(table.groups))
    rng.shuffle(albums)
    n_hold = max(1, int(round(len(albums) * frac)))
    hold_albums = set(albums[:n_hold])
    hold = np.array([i for i, g in enumerate(table.groups) if g in hold_albums])
    train = np.array([i for i, g in enumerate(table.groups) if g not in hold_albums])
    return train, hold


def volume() -> dict:
    """Объём датасета для решения «пора обучать уровень N»."""
    q = load_quality()
    pairs = 0
    tracks = 0
    for d in sorted((dataset_dir() / "pairs").glob("*/features.json")):
        key = d.parent.name.replace("__", "/")
        rec = q.get(key)
        if rec and rec["label"] not in ("trusted", "weak_base"):
            continue
        rows = json.loads(d.read_text(encoding="utf-8"))
        pairs += 1
        tracks += len(rows)
    return {"pairs": pairs, "tracks": tracks}
