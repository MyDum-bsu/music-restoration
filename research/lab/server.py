#!/usr/bin/env python3
"""restore-lab — a local web view: dataset, metrics, spectrograms, pipeline.

Run:  python3 lab/server.py [--port 8765]
Reads the dataset from the flash drive (read-only); all of its own artifacts
(spectrogram cache, runs of "our" pipeline) go to ~/.cache/restore-lab/.
Writes nothing to the flash drive or the repository.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rnext import features as F  # noqa: E402
from rnext.dsp.contour import true_peak_dbfs  # noqa: E402
from rnext.dsp.pipeline import Policy, restore_track  # noqa: E402
from rnext import workspace  # noqa: E402

CACHE = Path.home() / ".cache" / "restore-lab"
SPEC_DIR = CACHE / "spec"
OURS_DIR = CACHE / "ours"
STATIC = Path(__file__).resolve().parent / "static"
BAND_HZ = F._BANDS_HZ  # 31 edge → 30 bands (1/3 octave)

_cache = {"ts": 0.0, "data": {}}  # jsonl loading from the flash drive (NTFS is slow)
_TTL = 30.0


def _read_jsonl(path: Path):
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _dataset():
    """All the heavy dataset files, loaded once, with a TTL cache."""
    now = time.time()
    if now - _cache["ts"] < _TTL:
        return _cache["data"]
    root = workspace.data_root()
    d = {
        "root": str(root),
        "features": _read_jsonl(root / "dataset/features/all.jsonl"),
        "refq": _read_jsonl(root / "dataset/reference_quality.jsonl"),
        "eval": _read_jsonl(root / "dataset/eval_results.jsonl"),
        "audio_evals": _read_jsonl(root / "dataset/audio_evals.jsonl"),
        "manifest": {},
        "pairs_dir": root / "dataset/pairs",
    }
    # manifest of the latest version comes from the repository
    mans = sorted((REPO / "dataset_manifests").glob("ds-*.json"), key=lambda p: p.stat().st_mtime)
    if mans:
        try:
            d["manifest"] = json.loads(mans[-1].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    _cache["ts"], _cache["data"] = now, d
    return d


def _leaderboard():
    rows = _read_jsonl(REPO / "models/leaderboard.jsonl")
    return rows


def _fmt_ver(ts):
    return time.strftime("%d.%m %H:%M", time.localtime(ts)) if ts else "—"


def _agg_features(rows):
    """Per-pair averages over tracks."""
    n = len(rows)
    if not n:
        return None
    mean = lambda vals: round(sum(vals) / len(vals), 2)
    return {
        "n": n,
        "dr_o": mean([r["original"]["dr_exact"] for r in rows]),
        "dr_m": mean([r["restored"]["dr_exact"] for r in rows]),
        "dcrest": mean([r["delta"]["crest"] for r in rows]),
        "ddr": mean([r["delta"]["dr_exact"] for r in rows]),
        "rms_o": mean([r["original"]["rms_db"] for r in rows]),
        "rms_m": mean([r["restored"]["rms_db"] for r in rows]),
    }


_ds_size = {"ts": 0.0, "gb": None}


def _dir_size(root: Path) -> int:
    """Size of the dataset tree; traversal is slow on NTFS — cached for 5 minutes."""
    now = time.time()
    if _ds_size["gb"] is not None and now - _ds_size["ts"] < 300:
        return _ds_size["gb"]
    total = 0
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            try:
                total += (Path(dp) / f).stat().st_size
            except OSError:
                pass
    _ds_size["ts"], _ds_size["gb"] = now, total
    return total


def _lb_latest_ds(lb):
    """Dataset version of the freshest training run (timestamp embedded in model_id)."""
    def ts(r):
        m = re.search(r"-(\d{8})-(\d{6})$", r.get("model_id", ""))
        return m.group(1) + m.group(2) if m else "0"
    best = max(lb, key=ts, default=None)
    return best.get("dataset_version", "") if best else ""


def api_overview(q):
    ds = _dataset()
    root = Path(ds["root"])
    man = ds["manifest"]
    # budget
    budget = {}
    try:
        st = os.statvfs(root)
        free = st.f_bavail * st.f_frsize
        total = st.f_blocks * st.f_frsize
        used = total - free
        budget = {"free_gb": round(free / 2**30, 1), "used_gb": round(used / 2**30, 1),
                  "dataset_gb": round(_dir_size(root) / 2**30, 1),
                  "limit_gb": round(workspace.load_config()["budget_bytes"] / 2**30, 1)}
    except OSError:
        pass
    # source classes and authors
    classes, authors = {}, {}
    for r in ds["refq"]:
        classes[r.get("src_class", "?")] = classes.get(r.get("src_class", "?"), 0) + 1
        a = r.get("author_pid")
        if a:
            authors[str(a)] = authors.get(str(a), 0) + 1
    # models: best by-album on the freshest dataset version + best by-author
    lb = _leaderboard()
    cur_ds = _lb_latest_ds(lb)
    album_rows = [r for r in lb if r.get("dataset_version") == cur_ds and r.get("holdout_mode") == "by-album"]
    best_album = min((r for r in album_rows if r.get("promote")), key=lambda r: r["median_abs"], default=None)
    author_rows = [r for r in lb if r.get("promote") and r.get("holdout_mode") == "by-author"]
    best_author = min(author_rows, key=lambda r: r["median_abs"], default=None) if author_rows else None
    # milestones
    m1 = any(r.get("promote") and r["median_abs"] < 1.0 and r["model"] in ("ridge", "knn")
             and r["holdout_mode"] == "by-album" for r in lb)
    m2 = any(r.get("promote") and r["median_abs"] < 0.8 and r["model"] == "hgb"
             and r["holdout_mode"] == "by-album" for r in lb)
    m3 = any(r.get("promote") and r["median_abs"] < 1.0 and r["holdout_mode"] == "by-author" for r in lb)
    # tail of the loop journal
    loop_tail = []
    lf = root / "dataset/loop.log"
    if lf.is_file():
        try:
            loop_tail = lf.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
        except OSError:
            pass
    # blind residual checks (abx)
    residual = []
    for rj in sorted((root / "abx").glob("*/residual_*.json")):
        try:
            payload = json.loads(rj.read_text(encoding="utf-8"))
            vals = [v.get("residual_db_rel") for v in payload.values() if isinstance(v, dict)]
            arts = sum(v.get("n_prom_gt10_res", 0) for v in payload.values() if isinstance(v, dict))
            if vals:
                residual.append({"name": rj.stem, "tracks": len(vals),
                                 "median_residual_db": round(sorted(vals)[len(vals) // 2], 1),
                                 "artifact_peaks": arts})
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "dataset": {"version": man.get("version"), "pairs": man.get("pairs"),
                    "tracks": man.get("tracks"), "trusted_tracks": man.get("trusted_tracks"),
                    "labels": man.get("labels"), "created": _fmt_ver(man.get("created", 0)),
                    "by_author": man.get("by_author")},
        "budget": budget,
        "classes": classes,
        "authors": authors,
        "model_best": best_album, "model_best_crossauthor": best_author,
        "dataset_version_current": cur_ds,
        "milestones": {"M1_ridge_knn": m1, "M2_hgb": m2, "M3_crossauthor": m3},
        "audio_evals": ds["audio_evals"][-6:],
        "loop_tail": loop_tail,
        "residual": residual,
    }


def api_pairs(q):
    ds = _dataset()
    by_key = {}
    for r in ds["features"]:
        by_key.setdefault(r["key"], []).append(r)
    eval_by_key = {}
    for r in ds["eval"]:
        e = eval_by_key.setdefault(r["key"], {"n": 0, "model_abs": 0.0, "class_abs": 0.0, "flat_abs": 0.0})
        e["n"] += 1
        e["model_abs"] += abs(r.get("model_err", 0))
        e["class_abs"] += abs(r.get("class_err", 0))
        e["flat_abs"] += abs(r.get("flat_err", 0))
    rows = []
    for rq in ds["refq"]:
        key = rq["key"]
        agg = _agg_features(by_key.get(key, []))
        ev = eval_by_key.get(key)
        rows.append({
            "key": key, "artist": rq.get("artist"), "album": rq.get("album"),
            "year": rq.get("year"), "author_pid": rq.get("author_pid"),
            "label": rq.get("label"), "src_class": rq.get("src_class"),
            "src_dr": rq.get("src_dr"), "z_author": rq.get("z_author"),
            "added_clip": rq.get("added_clip"), "tp_overshoot": rq.get("tp_overshoot"),
            "tracks": agg["n"] if agg else rq.get("tracks"),
            "dr_o": agg and agg["dr_o"], "dr_m": agg and agg["dr_m"],
            "dcrest": agg and agg["dcrest"], "ddr": agg and agg["ddr"],
            "rms_o": agg and agg["rms_o"], "rms_m": agg and agg["rms_m"],
            "model_abs": ev and round(ev["model_abs"] / ev["n"], 2),
            "class_abs": ev and round(ev["class_abs"] / ev["n"], 2),
            "n_eval": ev and ev["n"],
        })
    rows.sort(key=lambda r: ((r["artist"] or "").lower(), r["album"] or ""))
    return {"rows": rows}


def api_pair(q):
    key = q.get("key", [""])[0]
    if not re.fullmatch(r"\d+/rel\d+", key):
        sys.stderr.write(f"[api/pair] BAD KEY raw={q.get('key')!r} parsed={key!r}\n")
        sys.stderr.flush()
        return {"error": "bad key"}
    ds = _dataset()
    pdir = ds["pairs_dir"] / key.replace("/", "__")
    meta = {}
    if (pdir / "meta.json").is_file():
        try:
            meta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    tracks = [r for r in ds["features"] if r["key"] == key]
    refq = next((r for r in ds["refq"] if r["key"] == key), None)
    evals = [r for r in ds["eval"] if r["key"] == key]
    excerpts = []
    if pdir.is_dir():
        for f in sorted((pdir / "excerpts").glob("*.flac")):
            m = re.match(r"t(\d+)_(\w+)_(\d+)_(orig|master)\.flac", f.name)
            if m:
                excerpts.append({"file": f.name, "track": int(m.group(1)),
                                 "tag": m.group(2), "len_s": int(m.group(3)), "who": m.group(4)})
    return {"key": key, "meta": meta, "refq": refq, "tracks": tracks,
            "evals": evals, "excerpts": excerpts}


def api_eval(q):
    ds = _dataset()
    rows = ds["eval"]
    variants = ("flat", "class", "model")

    def summar(sel):
        errs = [r.get(f"{v}_err") for r in sel for v in variants if r.get(f"{v}_err") is not None]
        out = {}
        for v in variants:
            vals = [abs(r[f"{v}_err"]) for r in sel if r.get(f"{v}_err") is not None]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            out[v] = {
                "n": len(vals),
                "median_abs": round(vals_sorted[len(vals) // 2], 3),
                "share_le_0.5": round(sum(1 for x in vals if x <= 0.5) / len(vals), 3),
                "share_le_1.0": round(sum(1 for x in vals if x <= 1.0) / len(vals), 3),
                "bias": round(sum(r[f"{v}_err"] for r in sel if r.get(f"{v}_err") is not None) / len(vals), 3),
            }
        return out

    tags = sorted({r.get("tag") for r in rows if r.get("tag")})
    return {"total": len(rows), "all": summar(rows),
            "by_tag": {t: summar([r for r in rows if r.get("tag") == t]) for t in tags},
            "audio_evals": ds["audio_evals"]}


def api_models(q):
    lb = _leaderboard()
    try:
        from rnext.ml import runtime
        info = runtime.snapshot_info(None)
    except Exception:
        info = None
    cur_ds = _lb_latest_ds(lb)
    cur = [r for r in lb if r.get("dataset_version") == cur_ds]
    cur.sort(key=lambda r: (r.get("holdout_mode") != "by-album", r.get("median_abs", 9e9)))
    if info is None and cur:
        r0 = next((r for r in cur if r.get("promote") and r.get("holdout_mode") == "by-album"), cur[0])
        info = {"id": r0["model_id"], "model": r0["model"], "median_abs": r0["median_abs"],
                "note": "from the leaderboard (gate verdict not recorded in the registry)"}
    elif info is not None:
        # snapshot_info knows the model and dataset version but not the id/metric — pulled from the leaderboard
        cand = [r for r in cur if r.get("model") == info.get("model")
                and r.get("dataset_version") == info.get("dataset_version")
                and r.get("holdout_mode") == "by-album"]
        if cand:
            best = min(cand, key=lambda r: r.get("median_abs", 9e9))
            info = dict(info, id=best["model_id"], median_abs=best["median_abs"],
                        holdout_mode=best["holdout_mode"], n_train=best.get("n_train"))
    return {"current_ds": cur_ds, "rows": cur[-40:],
            "runtime_model": info, "n_snapshots": len(lb)}


def api_policy(q):
    pol = Policy()
    return {"defaults": dataclasses.asdict(pol),
            "band_hz": BAND_HZ}


# ------------------------------------------------------------ audio utilities

def _safe_path(rel: str) -> Path | None:
    """A path strictly inside the flash drive or the cache — nothing else."""
    rel = urllib.parse.unquote(rel)
    for base in (workspace.data_root(), CACHE):
        p = (base / rel).resolve()
        try:
            p.relative_to(base.resolve())
        except ValueError:
            continue
        if p.is_file():
            return p
    return None


def _spec_png(path: Path, w: int, h: int, vmin: float, vmax: float) -> Path | None:
    """Spectrogram on an ABSOLUTE dB scale shared by every image in a comparison.

    ffmpeg's showspectrumpic normalizes each picture to its own range, so our pass
    and the reference looked different mainly because of that autoscale. Drawing
    the STFT here with fixed vmin/vmax makes the three panels honestly comparable.
    """
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    st = path.stat()
    key = hashlib.sha1(
        f"{path}|{st.st_mtime_ns}|{st.st_size}|{w}x{h}|{vmin}|{vmax}".encode()).hexdigest()[:20]
    out = SPEC_DIR / f"{key}.png"
    if out.is_file():
        return out
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import signal as sig
    except Exception:
        return None
    x, fs = F.decode_f32(str(path))
    if x is None or len(x) < 2048:
        return None
    m = x.mean(axis=1)
    nper = 2048 if fs >= 40000 else 1024
    f, t, Z = sig.stft(m, fs=fs, nperseg=nper, noverlap=nper * 3 // 4, window="hann")
    mag = 20 * np.log10(np.maximum(np.abs(Z), 1e-12))
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(mag, origin="lower", aspect="auto", cmap="magma",
              vmin=vmin, vmax=vmax,
              extent=[t[0], t[-1], f[0] / 1000, f[-1] / 1000])
    ax.set_axis_off()
    ax.set_ylim(0, 22)
    fig.savefig(out, dpi=100, format="png")
    plt.close(fig)
    return out if out.is_file() else None


def _spec_range(paths) -> tuple[float, float]:
    """Shared colour scale: percentile of absolute magnitude across all panels."""
    import numpy as np
    from scipy import signal as sig
    acc = []
    for p in paths:
        if p is None:
            continue
        x, fs = F.decode_f32(str(p))
        if x is None or len(x) < 2048:
            continue
        nper = 2048 if fs >= 40000 else 1024
        _, _, Z = sig.stft(x.mean(axis=1), fs=fs, nperseg=nper,
                           noverlap=nper * 3 // 4, window="hann")
        acc.append(20 * np.log10(np.maximum(np.abs(Z), 1e-12)).ravel())
    if not acc:
        return -90.0, -20.0
    allv = np.concatenate(acc)
    lo, hi = np.percentile(allv, [8, 99.7])
    return float(lo), float(hi)


def _decode(path: Path):
    return F.decode_f32(str(path))


def api_wave(q, path: Path):
    x, sr = _decode(path)
    if x is None:
        return {"error": "decode"}
    n = x.shape[0]
    win = max(int(0.05 * sr), 1)
    if n / win > 4000:
        win = n // 4000
    nw = n // win
    if nw < 2:
        return {"error": "short"}
    w = x[:nw * win].reshape(nw, win, 2)
    rms = (20 * np.log10(np.maximum(np.sqrt((w ** 2).mean(axis=(1, 2))) * np.sqrt(2), 1e-6))).round(1)
    peak = (20 * np.log10(np.maximum(np.abs(w).max(axis=(1, 2)), 1e-6))).round(1)
    t = ((np.arange(nw) + 0.5) * win / sr).round(2)
    return {"t": t.tolist(), "rms": rms.tolist(), "peak": peak.tolist()}


def api_spectrum(q, path: Path):
    x, sr = _decode(path)
    if x is None:
        return {"error": "decode"}
    bands = F.band_rms_db(x, sr)
    centers = [round((BAND_HZ[i] * BAND_HZ[i + 1]) ** 0.5, 1) for i in range(len(bands))]
    return {"hz": centers, "db": [round(v, 1) for v in bands]}


def api_restore(payload):
    """Run our pipeline on an excerpt. path is relative to the flash drive."""
    rel = payload.get("path", "")
    use_ml = bool(payload.get("ml", True))
    src = _safe_path(rel)
    if src is None:
        return {"error": "file not found"}
    x, sr = _decode(src)
    if x is None or x.shape[0] < sr * 5:
        return {"error": "file too short"}
    ml_target = ml_source = None
    if use_ml:
        try:
            from rnext.ml import runtime as mlrt
            ml_target, ml_source = mlrt.predict_from_audio_src(x, sr, model_id=None, n_tracks=10)
        except Exception as e:
            ml_source = f"model unavailable: {type(e).__name__}"
    pol = Policy(mode="expand", r_expand=1.5, k=0.55, target_tp_dbfs=-0.5,
                 adaptive=True, target_dcrest=4.6, auto_policy=True)
    if ml_target is not None:
        pol = dataclasses.replace(pol, healthy_dr_min=12,
                                  target_dcrest=float(min(ml_target, 8.0) if ml_target > 0 else ml_target))
    t0 = time.time()
    y, stages = restore_track(x, sr, pol, return_stages=True, keep_arrays=False)
    took = round(time.time() - t0, 1)
    # before/after metrics
    o = F.tt_dr(x, sr)
    r = F.tt_dr(y, sr)
    tp_o = true_peak_dbfs(x, sr)
    tp_y = true_peak_dbfs(y, sr)
    x16 = np.clip(x * 32767.5, -32768, 32767).astype(np.int16)
    y16 = np.clip(y * 32767.5, -32768, 32767).astype(np.int16)
    clip_o = F.clip_stats(x16)["L"]["clipped_samples"] + F.clip_stats(x16)["R"]["clipped_samples"]
    clip_y = F.clip_stats(y16)["L"]["clipped_samples"] + F.clip_stats(y16)["R"]["clipped_samples"]
    # save the "ours" variant to the local cache and return the relative path
    key_dir = CACHE_rel_for(src)
    out = OURS_DIR / key_dir / (src.stem + "__ours.flac")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0xC0FFEE)
    dith = rng.random(y.shape) + rng.random(y.shape) - 1.0
    yq = np.clip(np.round(y * 32768.0 + dith), -32768, 32767).astype(np.int16)
    import wave
    wav_tmp = out.with_suffix(".wav")
    with wave.open(str(wav_tmp), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(yq.tobytes())
    subprocess.run(["flac", "-s", "-f", "-8", "-o", str(out), str(wav_tmp)], check=True)
    wav_tmp.unlink(missing_ok=True)
    dx = stages.get("diagnosis", {})
    return {
        "took_s": took,
        "class": dx.get("class"),
        "target_dcrest": dx.get("target_dcrest"),
        "ml_target": ml_target and round(float(ml_target), 2),
        "ml_source": ml_source,
        "r_fit": (stages.get("adaptive") or {}).get("r_fit"),
        "declip_repaired": (stages.get("declip") or (None, None))[1],
        "orig": {"dr": o[0], "dr_exact": round(o[1], 2), "crest": round(o[3] - o[4], 2),
                 "peak": round(o[3], 2), "rms": round(o[4], 2), "tp": round(tp_o, 2), "clip": int(clip_o)},
        "ours": {"dr": r[0], "dr_exact": round(r[1], 2), "crest": round(r[3] - r[4], 2),
                 "peak": round(r[3], 2), "rms": round(r[4], 2), "tp": round(tp_y, 2), "clip": int(clip_y)},
        "delta": {"dr": r[0] - o[0], "crest": round((r[3] - r[4]) - (o[3] - o[4]), 2)},
        "ours_path": str(out.relative_to(CACHE)),
    }


def CACHE_rel_for(src: Path) -> str:
    """Cache subfolder: <pair>/<excerpt name>."""
    parts = src.parts
    pair = next((p for p in reversed(parts) if "__rel" in p), "misc")
    return f"{pair}/{src.stem}"


# ------------------------------------------------------------ http

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, mime: str, cache: bool = True):
        try:
            st = path.stat()
        except OSError:
            self._json({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(st.st_size))
        if cache:
            self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        with path.open("rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        route = u.path
        try:
            if route in ("/", "/index.html"):
                # static version = the freshest mtime among ALL files the page loads;
                # otherwise the browser keeps stale JS/CSS from its cache
                ver = str(int(max((p.stat().st_mtime for p in STATIC.iterdir()
                                   if p.suffix in (".js", ".css")), default=0)))
                html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__V__", ver)
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            elif route.startswith("/static/"):
                name = route[len("/static/"):]
                p = (STATIC / name).resolve()
                if not str(p).startswith(str(STATIC.resolve())) or not p.is_file():
                    self._json({"error": "not found"}, 404)
                    return
                if name.endswith(".css"):
                    mime = "text/css"
                elif name.endswith(".html"):
                    mime = "text/html"
                else:
                    mime = "application/javascript"
                self._file(p, mime + "; charset=utf-8", cache=False)
            elif route == "/api/overview":
                self._json(api_overview(q))
            elif route == "/api/pairs":
                self._json(api_pairs(q))
            elif route == "/api/pair":
                self._json(api_pair(q))
            elif route == "/api/eval":
                self._json(api_eval(q))
            elif route == "/api/models":
                self._json(api_models(q))
            elif route == "/api/policy":
                self._json(api_policy(q))
            elif route == "/api/spectrogram":
                paths = [_safe_path(p) for p in q.get("path", [])]
                paths = [p for p in paths if p is not None]
                if not paths:
                    self._json({"error": "not found"}, 404)
                    return
                w = min(max(int(q.get("w", ["1100"])[0]), 300), 1600)
                h = min(max(int(q.get("h", ["300"])[0]), 120), 800)
                vmin, vmax = _spec_range(paths)
                png = _spec_png(paths[0], w, h, vmin, vmax)
                if png is None:
                    self._json({"error": "render failed"}, 500)
                    return
                self._file(png, "image/png")
            elif route == "/api/wave":
                p = _safe_path(q.get("path", [""])[0])
                self._json(api_wave(q, p) if p else {"error": "not found"})
            elif route == "/api/spectrum":
                p = _safe_path(q.get("path", [""])[0])
                self._json(api_spectrum(q, p) if p else {"error": "not found"})
            elif route == "/api/audio":
                p = _safe_path(q.get("path", [""])[0])
                if p is None:
                    self._json({"error": "not found"}, 404)
                    return
                st = p.stat()
                start, end = 0, st.st_size - 1
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes="):
                    a, _, b = rng[6:].split(",")[0].partition("-")
                    if a.strip():
                        start = int(a)
                    if b.strip():
                        end = min(int(b), st.st_size - 1)
                    if start > end or start >= st.st_size:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{st.st_size}")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{st.st_size}")
                else:
                    self.send_response(200)
                self.send_header("Content-Type", "audio/flac")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                with p.open("rb") as fh:
                    fh.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = fh.read(min(65536, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            else:
                self._json({"error": "no route"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/api/restore":
            self._json({"error": "no route"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            self._json(api_restore(payload))
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser(description="restore-lab — web view of the dataset and the algorithm")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    OURS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"restore-lab: http://127.0.0.1:{a.port}  (data: {workspace.data_root()})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
