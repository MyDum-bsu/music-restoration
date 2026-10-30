#!/usr/bin/env python3
"""Restore a whole original album → directory with FLACs + report.

Input: image+cue (--src image --cue file.cue), a directory with tracks (*.flac),
or a single file. Output:
  out/NN. Title.flac — 16-bit FLAC (TP-normalized to the policy ceiling)
  out/restore_report.jsonl — per-track metrics (DR/crest/peak/rms before/after)

Auto-policy is on by default: the material class (healthy/sparse/loud) decides
whether HPF/contour are enabled and what Δcrest target to set. The target can
come from the model (`--ml-snapshot auto`); then the healthy threshold agrees
with training.

Mutex: running two restores in parallel once swapped the machine to death, so
we take an flock on /media/.../restore/.restore.lock (non-blocking; --wait to wait).
"""
import argparse
import dataclasses
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnext import features as F  # noqa: E402
from rnext.dsp.contour import true_peak_dbfs  # noqa: E402
from rnext.dsp.pipeline import Policy, restore_track  # noqa: E402
from rnext.phone.config import ensure_dirs, state_dir  # noqa: E402

ensure_dirs()
LOCK = state_dir() / "restore.lock"


class RestoreBusy(RuntimeError):
    pass


def acquire_lock(wait: bool = False):
    f = open(LOCK, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BlockingIOError as e:
        raise RestoreBusy(f"busy: {LOCK}") from e
    f.seek(0)
    f.truncate()
    f.write(f"pid={os.getpid()}\n")
    f.flush()
    return f


def encode_flac(y: np.ndarray, sr: int, out_path: Path, tmp_wav: Path,
                bits: int = 16) -> None:
    """float [-1,1] → FLAC. 16 or 24 bits, sample rate unchanged."""
    import wave
    if bits not in (16, 24):
        raise ValueError(f"unsupported bits: {bits}")
    with wave.open(str(tmp_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(bits // 8)
        w.setframerate(sr)
        if bits == 24:
            # WAV 24-bit: the three low bytes of little-endian int32 per sample
            a = y.astype("<i4")
            raw = a.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
            w.writeframes(raw)
        else:
            w.writeframes(y.astype("<i2").tobytes())
    subprocess.run(["flac", "-s", "-f", "-8", "-o", str(out_path), str(tmp_wav)], check=True)
    tmp_wav.unlink(missing_ok=True)


def _clean_title(stem: str) -> str:
    """Strip the track number from the file name: "01 - Foo" → "Foo".

    The number is printed as a prefix on save anyway, otherwise you got
    "01. 01 In the Name of War.flac".
    """
    import re as _re
    s = stem.strip()
    s = _re.sub(r"^\s*\d{1,3}\s*(?:[-._)]|\s)\s*", "", s, count=1)
    return s.strip() or stem



def process_track(job: dict) -> dict | None:
    """Process ONE track — a separate function so tracks can run through a pool.

    Tracks are independent: the model target is computed per album in advance,
    the adaptive fit is per track, dither is seeded by track number. So parallel
    execution does not change the result — it is byte-identical to sequential.
    """
    import dataclasses as _dc
    ti, title, path = job["ti"], job["title"], job["path"]
    args, pol = job["args"], job["pol"]
    mode, adaptive, eq_tilt = job["mode"], job["adaptive"], job["eq_tilt"]
    if path is None:
        sr0 = F.ffprobe_audio(job["src"])["sr"]
        x, sr = F.decode_f32(job["src"], start_s=job["start"] / sr0,
                             dur_s=(job["end"] - job["start"]) / sr0)
    else:
        x, sr = F.decode_f32(str(path))
    if x is None or x.shape[0] < sr * 5:
        return None
    o_ = F.tt_dr(x, sr)
    pol_t, ml_target, ml_source = pol, None, None
    if args.ml_snapshot and job["ml_ready"]:
        from rnext.ml import runtime as mlrt
        mid = None if args.ml_snapshot == "auto" else args.ml_snapshot
        ml_target, ml_source = mlrt.predict_from_audio_src(
            x, sr, model_id=mid, n_tracks=job["n_tracks"])
        if ml_target is not None:
            # the model was trained with the healthy boundary at ≥ 12, not ≥ 11
            pol_t = _dc.replace(pol, healthy_dr_min=12,
                                target_dcrest=float(min(ml_target, 8.0)
                                                    if ml_target > 0 else ml_target))
    # keep_arrays=False: copies of intermediate arrays (~127 MB per track) are of
    # no use to the report, and in a parallel run that is the decisive memory saving
    y, stages = restore_track(x, sr, pol_t, return_stages=True, keep_arrays=False)
    _d = F.tt_dr(y, sr)
    tp = true_peak_dbfs(y, sr)
    want_bits = str(getattr(args, "bits", "auto") or "auto")
    out_bits = (24 if int(job.get("src_bps") or 0) >= 24 else 16) \
        if want_bits == "auto" else int(want_bits)
    if out_bits not in (16, 24):
        out_bits = 16
    rng = np.random.default_rng(0xC0FFEE + ti)
    if out_bits == 24:
        dith = rng.random(y.shape) + rng.random(y.shape) - 1.0   # TPDF ±1 LSB @24
        yq = np.clip(np.round(y * 8388608.0 + dith), -8388608, 8388607).astype(np.int32)
    else:
        dith = rng.random(y.shape) + rng.random(y.shape) - 1.0   # TPDF ±1 LSB @16
        yq = np.clip(np.round(y * 32768.0 + dith), -32768, 32767).astype(np.int16)
    flac_path = job["out_dir"] / f"{ti:02d}. {title}.flac"
    encode_flac(yq, sr, flac_path, job["out_dir"] / f"__tmp{ti}.wav", bits=out_bits)
    dx = stages.get("diagnosis", {})
    rec = {
        "track": ti, "title": title,
        "mode": mode, "r": args.r, "adaptive": adaptive,
        "auto_policy": args.auto_policy,
        "bits": out_bits,
        "class": dx.get("class"),
        "target_dcrest": dx.get("target_dcrest", args.target),
        "ml_snapshot": args.ml_snapshot or None,
        "eq_tilt": eq_tilt,
        "eq_match": (stages.get("eq_match") or None) if args.eq_match else None,
        "ml_target": ml_target, "ml_source": ml_source,
        "r_fit": (stages.get("adaptive") or {}).get("r_fit"),
        "tp_target": args.tp, "tp_actual": round(tp, 2),
        "orig": {"dr": o_[0], "dr_exact": round(o_[1], 2),
                 "crest_tt": round(o_[3] - o_[4], 2),
                 "peak_dbfs": round(o_[3], 2), "rms_db": round(o_[4], 2)},
        "rest": {"dr": _d[0], "dr_exact": round(_d[1], 2),
                 "crest_tt": round(_d[3] - _d[4], 2),
                 "peak_dbfs": round(_d[3], 2), "rms_db": round(_d[4], 2)},
        "delta": {"dr": _d[0] - o_[0],
                  "crest": round((_d[3] - _d[4]) - (o_[3] - o_[4]), 2)},
    }
    msg = (f"t{ti:>2} DR {o_[0]}->{_d[0]}  TP {tp:5.2f}  [{dx.get('class')}] {title}")
    if ml_target is not None:
        msg += f"  (target {ml_target:+.2f} [{ml_source}])"
    return {"rec": rec, "msg": msg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="image (with --cue) or a directory/file of tracks")
    ap.add_argument("--cue", default="", help="cue file, if --src is an image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="expand", choices=["expand", "flatten"])
    ap.add_argument("--r", type=float, default=1.5)
    ap.add_argument("--k", type=float, default=0.55)
    ap.add_argument("--gain-cap", type=float, default=None,
                    help="symmetric contour gain limit, dB (flatten: 6–8)")
    ap.add_argument("--tp", type=float, default=-0.5, help="target TP, dBFS")
    ap.add_argument("--bits", default="auto", choices=["auto", "16", "24"],
                    help="output depth: auto = same as source (24 for 24-bit sources)")
    ap.add_argument("--adaptive", action="store_true", help="fit r to the target Δcrest")
    ap.add_argument("--target", type=float, default=4.6, help="target Δcrest (adaptive)")
    ap.add_argument("--auto-policy", dest="auto_policy", action="store_true", default=True,
                    help="material class → policy (enabled by default)")
    ap.add_argument("--no-auto", dest="auto_policy", action="store_false",
                    help="disable auto-policy, work by manual --mode/--target/--adaptive")
    ap.add_argument("--eq-curve", default="", help="JSON with the tilt curve (scripts/fit_tilt.py)")
    ap.add_argument("--eq-scale", type=float, default=1.0, help="tilt curve scale")
    ap.add_argument("--eq-match", action="store_true",
                    help="closed-loop EQ: measure per-band delta, trim the difference to the curve")
    ap.add_argument("--eq-match-cap", type=float, default=2.0,
                    help="per-band correction limit, dB (for --eq-match)")
    ap.add_argument("--ml-snapshot", default="",
                    help="target snapshot id (or 'auto' = latest); target comes from the model")
    ap.add_argument("--wait", action="store_true", help="wait for the restore lock to be released")
    ap.add_argument("--jobs", type=int, default=1,
                    help="how many tracks to process in parallel (default 1). "
                         "The result does not depend on the number of threads")
    args = ap.parse_args()

    try:
        lock = acquire_lock(wait=args.wait)
    except RestoreBusy as e:
        print(f"BUSY: {e}")
        return 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Auto-policy: expand mode + adaptive switch on by themselves, the target comes from diagnose().
    mode = "expand" if args.auto_policy else args.mode
    adaptive = args.adaptive or args.auto_policy
    gcap = abs(args.gain_cap) if args.gain_cap is not None else None
    pol = Policy(mode=mode, r_expand=args.r, k=args.k, target_tp_dbfs=args.tp,
                 gain_min_db=-(gcap if gcap is not None else 9.0),
                 gain_max_db=(gcap if gcap is not None else 12.0),
                 adaptive=adaptive, target_dcrest=args.target,
                 auto_policy=args.auto_policy)
    eq_tilt = None
    if args.eq_curve:
        import json as _json
        cur = _json.loads(Path(args.eq_curve).read_text(encoding="utf-8"))
        bands = [(float(hz), float(g) * args.eq_scale)
                 for hz, g in zip(cur["band_hz"], cur["gain_db"])]
        if args.eq_match:
            pol = dataclasses.replace(pol, eq_match=True, eq_match_curve=bands,
                                      eq_match_cap_db=args.eq_match_cap,
                                      normalize_tp=True)
        else:
            pol = dataclasses.replace(pol, eq_bands_db=bands)
        eq_tilt = {"curve": str(args.eq_curve), "scale": args.eq_scale,
                   "n_tracks_fit": cur.get("n_tracks"), "tilt_db": cur.get("tilt_out_db")}
        mode_txt = f"closed-loop, cap {args.eq_match_cap} dB" if args.eq_match else "fixed"
        print(f"tilt: {len(bands)} bands from {args.eq_curve} ({mode_txt}), "
              f"low {bands[3][1]:+.2f} / high {bands[21][1]:+.2f} dB", flush=True)
    src = Path(args.src)

    if args.cue and src.is_file():
        from rnext.cue import parse_cue, track_bounds
        meta = F.ffprobe_audio(str(src))
        total = int(round(meta["dur"] * meta["sr"]))
        cs = parse_cue(args.cue, file_key=src.name)
        bounds = track_bounds(cs, total, meta["sr"])
        items = [(i + 1, s, e, cs[i]["title"], None, meta["bps"])
                 for i, (s, e) in enumerate(bounds)]
    else:
        cands = sorted(src.glob("*.flac")) if src.is_dir() else [src]
        items = []
        for i, c in enumerate(cands):
            try:
                bps = F.ffprobe_audio(str(c))["bps"]
            except Exception:                              # noqa: BLE001
                bps = 0
            items.append((i + 1, 0, 0, _clean_title(c.stem), c, bps))

    ml_ready = False
    if args.ml_snapshot:
        try:
            from rnext.ml import runtime as mlrt
            ml_ready = mlrt.snapshot_info(None if args.ml_snapshot == "auto"
                                          else args.ml_snapshot) is not None
            if not ml_ready:
                print("WARN: snapshot not found — using auto-policy", flush=True)
        except Exception as e:                             # noqa: BLE001
            print(f"WARN: ML unavailable ({type(e).__name__}) — auto-policy", flush=True)

    report = out_dir / "restore_report.jsonl"
    jobs = []
    for ti, st, en, title, path, src_bps in items:
        jobs.append({"ti": ti, "title": title, "path": path, "start": st, "end": en,
                     "src": str(src), "src_bps": src_bps, "pol": pol, "args": args,
                     "mode": mode, "adaptive": adaptive, "eq_tilt": eq_tilt,
                     "ml_ready": ml_ready, "n_tracks": len(items), "out_dir": out_dir})

    n_jobs = max(1, int(getattr(args, "jobs", 1) or 1))
    # Memory limit: array copies grow with track length and workers die from OOM
    # (that did happen on a 50-minute image). File size is a crude but available
    # proxy for length.
    sizes = [j["path"].stat().st_size for j in jobs if j.get("path")]
    if sizes:
        biggest = max(sizes)
        cap = 1 if biggest > 250e6 else (2 if biggest > 100e6 else n_jobs)
        if cap < n_jobs:
            print(f"long track ({biggest / 1e6:.0f} MB) — {cap} workers instead of {n_jobs}",
                  flush=True)
            n_jobs = cap
    if n_jobs > 1:
        print(f"{len(jobs)} tracks, {n_jobs} in parallel (report order preserved)",
              flush=True)
    results: list[dict] = []
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        done: set[int] = set()
        broken = False
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futs = {ex.submit(process_track, j): j["ti"] for j in jobs}
                for fut in as_completed(futs):
                    try:
                        res = fut.result()
                    except Exception as e:                  # noqa: BLE001
                        print(f"  [error] track {futs[fut]}: {type(e).__name__}: {e}",
                              flush=True)
                        broken = broken or "BrokenProcessPool" in type(e).__name__
                        continue
                    if res:
                        print("  " + res["msg"], flush=True)
                        results.append(res)
                    done.add(futs[fut])
        except Exception as e:                              # noqa: BLE001
            print(f"  [pool died] {type(e).__name__}: {e} — finishing sequentially",
                  flush=True)
            broken = True
        if broken:
            # Workers die from memory exhaustion (usually on very long tracks or
            # images). Grind the remainder single-threaded — slow is better than
            # losing the album.
            left = [j for j in jobs if j["ti"] not in done]
            print(f"  sequential: {len(left)} tracks", flush=True)
            for j in left:
                res = process_track(j)
                if res:
                    print("  " + res["msg"], flush=True)
                    results.append(res)
    else:
        for j in jobs:
            res = process_track(j)
            if res:
                print("  " + res["msg"], flush=True)
                results.append(res)

    with report.open("w", encoding="utf-8") as fh:
        for res in sorted(results, key=lambda r: r["rec"]["track"]):
            fh.write(json.dumps(res["rec"], ensure_ascii=False) + "\n")
    # remove temporary wavs, if any are left
    for tmp in out_dir.glob("__tmp*.wav"):
        tmp.unlink(missing_ok=True)

    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    print(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
