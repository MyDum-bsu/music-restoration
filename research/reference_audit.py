#!/usr/bin/env python3
"""Reference audit: can the master be trusted as the target for this pair.

Problem: masters are not perfect, some jobs are "radical reworks"
(crest overshoot, EQ skew, collapsed stereo base). Imitating that is wrong,
so each pair gets a label:

    trusted   — deltas within the physical band, no artifacts  → goes into target stats
    suspect   — borderline values, no artifacts                → excluded from the target, kept for listening
    radical   — crest/tilt clearly outside the band            → NEGATIVE example (upper bracket)
    bad_ref   — the master itself added clipping/raised noise  → excluded entirely
    weak_base — few pairs by this author, scored by absolute rules only

Pair metrics (medians over loud tracks, healthy tracks don't count):

    dcrest          master Δcrest relative to the original (the main target)
    peak            master peak, dBFS
    tilt            spectral tilt: high (2…8 kHz) minus low (125…500 Hz), dB
    added_clip      the master's contribution to clipping, samples
    tp_overshoot    intersample peak growth (TP − sample peak), dB
    attack_delta    change in attack density (transientness), 1/s
    side_delta      change in stereo width (side/mid), dB

  python3 scripts/reference_audit.py            # recompute and write
  python3 scripts/reference_audit.py --report   # distributions and flags
"""
import argparse
import json
import statistics as S
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import workspace  # noqa: E402

# 1/3-oct bands from 20 Hz: indices for "low" and "high"
LOW_BANDS = slice(7, 14)      # ~125–500 Hz
HIGH_BANDS = slice(21, 28)    # ~2–8 kHz

DCREST_OK = (2.0, 7.0)        # general band (for crushed material)
DCREST_RADICAL = 8.0
TILT_OK = 3.0
TILT_RADICAL = 5.0
HEALTHY_DR = 12               # such tracks are not taken into the stats

# the expected Δcrest band DEPENDS on the source class: on already dynamic
# material the master barely touches the crest itself (Aerosmith t1: Δ −0.8 at DR 12),
# so "Δcrest ≈ 0" there is normal, not a reason for suspicion.
BAND_BY_CLASS = {
    "crushed": (3.0, 8.0),    # source DR < 9 — expect a noticeable expansion
    "mid": (0.0, 6.0),        # 9–11 — the master's rules are unknown, wide band
    "healthy": (-1.5, 1.5),   # ≥ 12 — the master is transparent
}


def dataset_dir() -> Path:
    return workspace.data_root() / "dataset"


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def plan_index() -> dict[str, dict]:
    """author_pid and year come from the plan: the download journal may lack them."""
    out = {}
    p = workspace.dir_of("corpus") / "dataset_plan.jsonl"
    if p.is_file():
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["key"]] = r
    return out


def load_pairs() -> dict[str, dict]:
    out = {}
    for d in sorted((dataset_dir() / "pairs").glob("*/")):
        fp = d / "features.json"
        if not fp.is_file():
            continue
        rows = json.loads(fp.read_text(encoding="utf-8"))
        if not rows:
            continue
        meta = {}
        mp = d / "meta.json"
        if mp.is_file():
            meta = json.loads(mp.read_text(encoding="utf-8"))
        key = thread_key(d.name)
        pi = PLAN.get(key) or {}
        if pi:
            for k, v in (("author_pid", pi.get("author_pid")), ("artist", pi.get("artist")),
                         ("year", (pi.get("release") or {}).get("year")),
                         ("album", (pi.get("release") or {}).get("album"))):
                if not meta.get(k):
                    meta[k] = v
        out[key] = {"rows": rows, "meta": meta, "dir": str(d)}
    return out


PLAN: dict[str, dict] = {}


def thread_key(dirname: str) -> str:
    tid, rel = dirname.split("__", 1)
    return f"{tid}/{rel}"


def tilt_of(bands: list[float]) -> float:
    lo = S.mean(bands[LOW_BANDS])
    hi = S.mean(bands[HIGH_BANDS])
    return hi - lo


def pair_stats(rows: list[dict]) -> dict:
    loud = [r for r in rows if r["original"]["dr"] < HEALTHY_DR]
    use = loud or rows
    dcrest = S.median([r["delta"]["crest"] for r in use])
    ddr = S.median([r["delta"]["dr"] for r in use])
    peak = S.median([r["restored"]["peak_db"] for r in use])
    tilt = S.median([tilt_of(r["delta"]["bands_db"]) for r in use])
    added_clip = sum(max(0, r["delta"]["clip_samples"]) for r in rows)
    tp_over = [r["delta"].get("tp_overshoot_db") for r in use
               if r["delta"].get("tp_overshoot_db") is not None]
    attack = [r["delta"].get("attacks_per_s") for r in use
              if r["delta"].get("attacks_per_s") is not None]
    side = [r["delta"].get("side_mid_db") for r in use
            if r["delta"].get("side_mid_db") is not None]
    spread = [r["delta"].get("spread") for r in use if r["delta"].get("spread") is not None]
    return {
        "tracks": len(rows), "loud_tracks": len(loud),
        "src_dr": int(S.median([r["original"]["dr"] for r in rows])),
        "src_class": ("crushed" if S.median([r["original"]["dr"] for r in rows]) < 9
                      else ("mid" if S.median([r["original"]["dr"] for r in rows]) < 12
                            else "healthy")),
        "dcrest": round(float(dcrest), 2), "ddr": round(float(ddr), 2),
        "peak": round(float(peak), 2), "tilt": round(float(tilt), 2),
        "added_clip": int(added_clip),
        "tp_overshoot": round(float(S.median(tp_over)), 2) if tp_over else None,
        "attack_delta": round(float(S.median(attack)), 3) if attack else None,
        "side_delta": round(float(S.median(side)), 2) if side else None,
        "spread_delta": round(float(S.median(spread)), 2) if spread else None,
    }


def robust_z(v: float, vals: list[float]) -> float | None:
    if len(vals) < 5:
        return None
    med = S.median(vals)
    mad = S.median([abs(x - med) for x in vals]) or 0.0
    if mad <= 0:
        return None
    return (v - med) / (1.4826 * mad)


def label_of(st: dict, z: float | None, weak_base: bool) -> tuple[str, list[str]]:
    why = []
    if st["added_clip"] > 0:
        why.append(f"added_clip={st['added_clip']}")
    if st["peak"] > -0.2:
        why.append(f"peak={st['peak']}")
    if why:
        return "bad_ref", why
    radical = []
    if st["dcrest"] > DCREST_RADICAL:
        radical.append(f"dcrest={st['dcrest']}")
    if abs(st["tilt"]) > TILT_RADICAL:
        radical.append(f"tilt={st['tilt']}")
    if z is not None and z > 5.0:
        radical.append(f"z={z:.1f}")
    if radical:
        return "radical", radical
    lo, hi = BAND_BY_CLASS[st["src_class"]]
    suspect = []
    if not (lo <= st["dcrest"] <= hi):
        suspect.append(f"dcrest={st['dcrest']} outside the {st['src_class']} band [{lo}..{hi}]")
    if abs(st["tilt"]) > TILT_OK:
        suspect.append(f"tilt={st['tilt']}")
    if z is not None and z > 3.0:
        suspect.append(f"z={z:.1f}")
    if suspect:
        return "suspect", suspect
    return ("weak_base" if weak_base else "trusted"), []


def build() -> int:
    global PLAN
    PLAN = plan_index()
    pairs = load_pairs()
    if not pairs:
        print("no pairs with features — run dataset_extract first")
        return 1
    stats = {k: pair_stats(v["rows"]) for k, v in pairs.items()}
    by_author: dict[str, list[float]] = defaultdict(list)
    for k, st in stats.items():
        by_author[str(pairs[k]["meta"].get("author_pid"))].append(st["dcrest"])

    rows = []
    for k, st in stats.items():
        meta = pairs[k]["meta"]
        au = str(meta.get("author_pid"))
        base = by_author[au]
        z = robust_z(st["dcrest"], [x for x in base if x != st["dcrest"]] or base)
        lab, why = label_of(st, z, weak_base=len(base) < 5)
        rows.append({"key": k, "author_pid": meta.get("author_pid"),
                     "artist": meta.get("artist"), "album": meta.get("album"),
                     "year": meta.get("year"), "label": lab, "reasons": why,
                     "z_author": round(z, 2) if z is not None else None,
                     "author_pairs": len(base), **st})

    out = dataset_dir() / "reference_quality.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"audit: {len(rows)} pairs → {out}")
    print("labels:", dict(Counter(r["label"] for r in rows)))
    for r in rows:
        flags = ",".join(r["reasons"]) if r["reasons"] else ""
        print(f"  {r['label']:<9} DRin {r['src_dr']:>2} dcrest {r['dcrest']:>5} tilt {r['tilt']:>6} "
              f"peak {r['peak']:>6} clip+{r['added_clip']:<3} "
              f"{str(r['artist'])[:18]:<18} {str(r['album'])[:24]:<24} {flags}")
    return 0


def report() -> int:
    rows = read_jsonl(dataset_dir() / "reference_quality.jsonl")
    if not rows:
        print("no audit yet — run without --report")
        return 1
    print(f"pairs: {len(rows)}; labels:", dict(Counter(r["label"] for r in rows)))
    for a in sorted({str(r["author_pid"]) for r in rows}):
        sub = [r for r in rows if str(r["author_pid"]) == a]
        dc = [r["dcrest"] for r in sub]
        til = [r["tilt"] for r in sub]
        print(f"author {a}: pairs {len(sub)} | Δcrest median {S.median(dc):.2f} "
              f"(range {min(dc):.2f}..{max(dc):.2f}) | tilt median {S.median(til):.2f} "
              f"(range {min(til):.2f}..{max(til):.2f})")
        print(f"   trusted {sum(1 for r in sub if r['label'] == 'trusted')}/"
              f"{sum(1 for r in sub if r['label'] != 'bad_ref')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    return report() if args.report else build()


if __name__ == "__main__":
    sys.exit(main())
