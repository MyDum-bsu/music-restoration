#!/usr/bin/env python3
"""Independent reference restoration for the comparison experiment.

  python3 scripts/experiment/ref_restore.py --src <root> --out <dir> \
          [--work-dir <dir>] [--jobs N]

Three visible stages, none of them shared with the canonical per-track pipeline:
  1. declipping: every run of samples at 16-bit full scale is replaced by a
     raised-sine reconstruction between the neighbouring clean samples;
  2. loudness: the integrated value from ffmpeg ebur128 (the same meter the
     comparison uses) plus a linear gain toward -14 LUFS;
  3. true peak: an ffmpeg alimiter at -1.0 dBFS.

Output is 24-bit FLAC at the source sample rate, one file per input audio file,
same relative directory and same file name.  Sources are only ever read.
Temporary float buffers live in --work-dir (default <out>/.work) and are removed
after every track and again on exit, including on failure.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

AUDIO_EXT = {".flac", ".wav", ".ape", ".m4a", ".mp3", ".ogg", ".opus", ".aiff",
             ".alac"}

LUFS_TARGET = -14.0
TRUE_PEAK_CEILING = -1.0
LIMITER_LIMIT = 10.0 ** (TRUE_PEAK_CEILING / 20.0)   # 0.891251...
# Same definition audio_measure.py uses for "clipped_samples": 16-bit full scale.
CLIP_LEVEL = 1.0 - 1.0 / 32768.0
CANON_MARKER = "[restored"
EXPERIMENT_ROOT = Path("/media/mydum/Data/Music")


class TrackError(RuntimeError):
    """A single track could not be restored; carries the track name."""


def audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)


# ── guards ───────────────────────────────────────────────────────────────────

def guard_paths(src: Path, out: Path, work: Path) -> None:
    """Refuse destinations that would touch sources, the canon or the library."""
    out_r, work_r = out.resolve(), work.resolve()
    if out_r == src:
        raise SystemExit(f"ref_restore: --out is the source tree itself: {out_r}")
    if src in out_r.parents:
        raise SystemExit(f"ref_restore: --out is inside --src: {out_r}")
    if CANON_MARKER in str(out_r).lower():
        raise SystemExit(
            f"ref_restore: --out carries the canon marker {CANON_MARKER!r} "
            f"and would overwrite canonical results: {out_r}")
    if EXPERIMENT_ROOT in out_r.parents or out_r == EXPERIMENT_ROOT:
        if "experiment" not in str(out_r).lower():
            raise SystemExit(
                f"ref_restore: writing into {EXPERIMENT_ROOT} needs 'experiment' "
                f"in the path: {out_r}")
    if work_r == src or src in work_r.parents:
        raise SystemExit(f"ref_restore: --work-dir is inside the source tree: {work_r}")
    if work_r == out_r:
        raise SystemExit("ref_restore: --work-dir must differ from --out")
    if work_r in out_r.parents:
        raise SystemExit(f"ref_restore: --out is inside --work-dir, which is deleted: {out_r}")


def require_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(f"ref_restore: required tool(s) missing from PATH: {', '.join(missing)}")


# ── audio helpers ────────────────────────────────────────────────────────────

def probe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels,duration,bits_per_raw_sample",
         "-of", "json", str(path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise TrackError(f"ffprobe failed: {r.stderr.strip()}")
    streams = json.loads(r.stdout or "{}").get("streams") or []
    if not streams:
        raise TrackError("no audio stream")
    stream = streams[0]
    try:
        return {"sr": int(stream["sample_rate"]), "ch": int(stream["channels"]),
                "dur": float(stream.get("duration") or 0.0),
                "bits": int(stream.get("bits_per_raw_sample") or 0)}
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackError(f"unreadable stream description: {exc}") from exc


def decode(path: Path, sr: int, ch: int) -> np.ndarray:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(sr), "-ac", str(ch), "-"],
        capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not r.stdout:
        raise TrackError(f"ffmpeg decode failed: {r.stderr.decode(errors='replace').strip()}")
    data = np.frombuffer(r.stdout, dtype="<f4")
    if data.size % ch:
        raise TrackError(f"decode produced {data.size} samples, not a multiple of {ch} channels")
    return data.reshape(-1, ch).astype(np.float32)


# ── stage 1: declipping ──────────────────────────────────────────────────────

def declip_channel(y: np.ndarray) -> None:
    """Replace runs pinned at full scale with a raised-sine arc between neighbours."""
    hot = np.abs(y) >= CLIP_LEVEL
    if not hot.any():
        return
    idx = np.flatnonzero(hot)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1)).astype(int)
    ends = np.concatenate((breaks, [idx.size - 1])).astype(int)
    for start, end in zip(starts, ends):
        i0, i1 = int(idx[start]), int(idx[end])
        left = float(y[i0 - 1]) if i0 > 0 else 0.0
        right = float(y[i1 + 1]) if i1 + 1 < y.size else 0.0
        amp = min(CLIP_LEVEL * 0.999, max(abs(left), abs(right)))
        if abs(left) >= abs(right):
            sign = math.copysign(1.0, left) if left else 0.0
        else:
            sign = math.copysign(1.0, right) if right else 0.0
        if sign == 0.0:
            mean = float(np.mean(y[i0:i1 + 1]))
            sign = math.copysign(1.0, mean) if mean else 1.0
        m = i1 - i0 + 1
        phase = np.arange(1, m + 1, dtype=np.float64) / (m + 1.0)
        base = left + (right - left) * phase
        bump = np.sin(np.pi * phase) * sign * (amp - max(abs(left), abs(right)))
        y[i0:i1 + 1] = base + bump


def declip(x: np.ndarray) -> np.ndarray:
    "Declip every channel in place; `x[:, ch]` is a view, so the buffer is reused."""
    for ch in range(x.shape[1]):
        declip_channel(x[:, ch])
    return x


# ── stage 2 + 3: loudness and true peak ──────────────────────────────────────

def _ffmpeg_input(raw: Path, sr: int, ch: int) -> list[str]:
    return ["-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", str(raw)]


def loudness_measure(raw: Path, sr: int, ch: int) -> dict | None:
    """Integrated loudness and true peak of the declipped buffer, via ffmpeg ebur128.

    ebur128 reports the same integrated value that scripts/audio_measure.py uses
    for `lufs_i`, so the gain computed here is the gain that this very meter
    will show as -14.00 LUFS.  loudnorm is not used: its two-pass mode resamples
    the whole track to 192 kHz to measure a true peak that a limiter guards
    anyway, and that round trip costs memory and quality for nothing.
    """
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-v", "info",
                        *_ffmpeg_input(raw, sr, ch), "-af", "ebur128=peak=true",
                        "-f", "null", "-"],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise TrackError(f"ebur128 measurement failed (exit {r.returncode}): "
                         f"{r.stderr.strip()[-400:]}")
    values: dict[str, float | None] = {"input_i": None, "input_tp": None}
    head = r.stderr.rfind("Integrated loudness")
    if head >= 0:
        for line in r.stderr[head:].splitlines()[:4]:
            if "I:" in line:
                text = line.split("I:")[1].split("LUFS")[0].strip()
                try:
                    values["input_i"] = float(text)
                except ValueError:
                    values["input_i"] = None
    peak = r.stderr.rfind("True peak")
    if peak >= 0:
        for line in r.stderr[peak:].splitlines()[:4]:
            if "Peak:" in line:
                text = line.split("Peak:")[1].split("dBFS")[0].strip()
                try:
                    values["input_tp"] = float(text)
                except ValueError:
                    values["input_tp"] = None
    if values["input_i"] is None and "Integrated loudness" not in r.stderr:
        raise TrackError("ebur128 printed no loudness summary: "
                         f"{r.stderr.strip()[-300:]}")
    if values["input_i"] is None:
        return None   # digital silence: -inf LUFS, nothing to normalise
    return values


def encode_flac(raw: Path, sr: int, ch: int, measured: dict | None, dest: Path) -> None:
    """Apply the linear gain that reaches -14 LUFS, then the true-peak limiter.

    The gain is linear on purpose: integrated loudness is a linear quantity, so
    a volume gain hits the target exactly and leaves the dynamics alone, while
    the limiter catches whatever peaks the gain created.
    """
    gain_db = LUFS_TARGET - measured["input_i"] if measured else 0.0
    limiter = (f"alimiter=limit={LIMITER_LIMIT:.6f}:attack=5:release=50:"
               f"level=false:latency=true")
    filt = f"volume={gain_db:.2f}dB,{limiter}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-v", "error", "-y",
                        *_ffmpeg_input(raw, sr, ch), "-af", filt,
                        "-c:a", "flac", "-sample_fmt", "s32", "-ar", str(sr),
                        "-map_metadata", "-1", str(dest)],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not dest.is_file():
        raise TrackError(f"flac encode failed: {r.stderr.strip()[-400:]}")


def verify_output(dest: Path, sr: int, dur_in: float) -> None:
    info = probe(dest)
    if info["sr"] != sr:
        raise TrackError(f"output sample rate {info['sr']} != source {sr}")
    if info["bits"] != 24:
        raise TrackError(f"output is {info['bits']}-bit, expected 24-bit FLAC")
    if dur_in > 0 and abs(info["dur"] - dur_in) > 0.05:
        raise TrackError(f"output duration {info['dur']:.2f}s != source {dur_in:.2f}s")


# ── one track ────────────────────────────────────────────────────────────────

def process_track(src_file: Path, src_root: Path, out_root: Path, work: Path) -> str:
    rel = src_file.relative_to(src_root)
    dest = out_root / rel.parent / (rel.stem + ".flac")
    try:
        info = probe(src_file)
        sr, ch, dur = info["sr"], info["ch"], info["dur"]
        x = decode(src_file, sr, ch)
        if x.size == 0:
            raise TrackError("decoded to zero samples")
        x = declip(x)
        raw = work / (re.sub(r"[^A-Za-z0-9]+", "_", str(rel)) + ".f32")
        raw.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())
        try:
            measured = loudness_measure(raw, sr, ch)
            encode_flac(raw, sr, ch, measured, dest)
        finally:
            raw.unlink(missing_ok=True)
        verify_output(dest, sr, dur)
    except TrackError as exc:
        raise TrackError(f"{rel}: {exc}") from exc
    except OSError as exc:
        raise TrackError(f"{rel}: {exc}") from exc
    return str(rel)


def _worker(args: tuple[str, str, str, str]) -> str:
    return process_track(*[Path(part) for part in args])


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Independent reference restoration.")
    ap.add_argument("--src", required=True, help="root of source releases")
    ap.add_argument("--out", required=True, help="output root for the restored FLACs")
    ap.add_argument("--work-dir", default="", help="temp dir (default <out>/.work)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel tracks")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser()
    work = Path(args.work_dir).expanduser() if args.work_dir else out / ".work"
    if not src.is_dir():
        print(f"ref_restore: not a readable directory: {src}", file=sys.stderr)
        return 2
    files = audio_files(src)
    if not files:
        print(f"ref_restore: no audio files under {src}", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print(f"ref_restore: --jobs must be >= 1, got {args.jobs}", file=sys.stderr)
        return 2
    try:
        guard_paths(src, out, work)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    require_tools()

    out_resolved = out.resolve()
    work_resolved = work.resolve()
    print(f"ref_restore: {len(files)} tracks from {src} → {out_resolved}")
    try:
        work_resolved.mkdir(parents=True, exist_ok=True)
        out_resolved.mkdir(parents=True, exist_ok=True)
        done = 0
        if args.jobs == 1:
            for path in files:
                rel = process_track(path, src, out_resolved, work_resolved)
                done += 1
                print(f"  [{done}/{len(files)}] {rel}")
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = {pool.submit(_worker, (str(p), str(src), str(out_resolved),
                                                str(work_resolved))): p for p in files}
                for fut in as_completed(futures):
                    rel = fut.result()   # first failure aborts the run
                    done += 1
                    print(f"  [{done}/{len(files)}] {rel}")
    except TrackError as exc:
        print(f"ref_restore: {exc}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(work_resolved, ignore_errors=True)
    print(f"ref_restore: done, {done} tracks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
