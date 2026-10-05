#!/usr/bin/env python3
"""
master.py — the master-like DR-restoration pipeline (v2).

Chain (defaults mimic the reverse-engineered Holocene reference):
  DC fix -> [adeclip if diagnosed] -> [HPF 25 Hz zero-phase]
  -> reference EQ (lowshelf +0.9@150, bell -1.6@315 Q0.7, air +0.45@9k)
  -> CONTOUR EXPANDER (env window + smoothing, gain=(slope-1)*env+offset,
     slew-limited; slope 1.08, offset -4.6)
  -> ND2 punch restoration (Stereo Tool Natural Dynamics 2, optional)
  -> center focus (side -3 dB) -> TP normalize -0.5 dBFS -> TPDF dither
  -> FLAC + original tags + ReplayGain

Usage:
  master.py track IN.flac OUT.flac [opts]
  master.py album INDIR OUTDIR [opts]          # per-track diagnosis + _report.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, lfilter, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose import diagnose
from drlib.audio import decode, encode_wav, get_samplerate
from drlib.config import stereo_tool_bin, sts_dir, work_tmpdir
from drlib.tags import copy_tags
from measure import ebur128  # noqa: F401 (kept for API compatibility)

BIN = stereo_tool_bin()
STS = sts_dir()


# ------------------------------------------------------------ stage helpers

def fix_dc(x):
    return x - x.mean(axis=0, keepdims=True)


def hpf(x, fs, hz=25.0, order=4):
    sos = butter(order, hz / (fs / 2), "highpass", output="sos")
    return sosfiltfilt(sos, x, axis=0).astype(np.float64)  # zero-phase: no start transient


def rbj_peaking(fs, f0, q, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
    return np.array(b) / a[0], np.array(a) / a[0]


def rbj_highshelf(fs, f0, gain_db, s=1.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    c, t = np.cos(w0), 2 * np.sqrt(A) * alpha
    b = [A * ((A + 1) + (A - 1) * c + t), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - t)]
    a = [(A + 1) - (A - 1) * c + t, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - t]
    return np.array(b) / a[0], np.array(a) / a[0]


def rbj_lowshelf(fs, f0, gain_db, s=0.8):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    c, t = np.cos(w0), 2 * np.sqrt(A) * alpha
    b = [A * ((A + 1) - (A - 1) * c + t), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - t)]
    a = [(A + 1) + (A - 1) * c + t, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - t]
    return np.array(b) / a[0], np.array(a) / a[0]


def ref_eq(x, fs):
    """The reference's taste EQ (taken from a measured curve): low shelf +0.9 <150 Hz,
    wide dip -1.6 @315, air +0.45 >9k."""
    for b, a in (rbj_lowshelf(fs, 150, 0.9),
                 rbj_peaking(fs, 315, 0.7, -1.6),
                 rbj_highshelf(fs, 9000, 0.45)):
        x = lfilter(b, a, x, axis=0)
    return x


def slow_envelope_db(m, fs, win_s=0.3, hop_s=0.01, smooth_s=0.5):
    """Centered RMS envelope (dB) of the mono sum, smoothed."""
    win, hop = int(win_s * fs), int(hop_s * fs)
    c = np.concatenate([[0.0], np.cumsum(m ** 2)])
    centers = np.arange(win // 2, len(m) - win // 2, hop)
    rms2 = (c[centers + win // 2] - c[centers - win // 2]) / win
    env = 10 * np.log10(np.maximum(rms2, 1e-12))
    t = centers / fs
    k = max(int(smooth_s / hop_s) // 2 * 2 + 1, 3)
    pad = np.r_[np.full(k // 2, env[0]), env, np.full(k // 2, env[-1])]
    env = np.convolve(pad, np.ones(k) / k, mode="valid")
    return t, env


def slew_limit(g_db, dt, max_slew_db_s):
    out = g_db.copy()
    lo, hi = -max_slew_db_s * dt, max_slew_db_s * dt
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        if d > hi:
            out[i] = out[i - 1] + hi
        elif d < lo:
            out[i] = out[i - 1] + lo
    return out


def contour_expand(x, fs, slope=1.08, offset=-4.6, max_slew=10.0, env_win=0.3):
    """The v2 core: stretch the slow loudness contour.

    gain(t) = (slope*env + offset) - env. Transients keep their crest above the
    contour because the gain moves slowly (detector window + smoothing + slew limit).
    """
    t, env = slow_envelope_db(x.mean(axis=1), fs, win_s=env_win)
    gain = (slope * env + offset) - env  # = (slope-1)*env + offset
    dt = float(np.median(np.diff(t)))
    gain = slew_limit(gain, dt, max_slew)
    n = len(x)
    return x * 10 ** (np.interp(np.arange(n) / fs, t, gain) / 20)[:, None], gain, t


def center_focus(x, side_db=-3.0):
    m = x.mean(axis=1, keepdims=True)
    s = (x[:, [0]] - x[:, [1]]) / 2 * 10 ** (side_db / 20)
    return np.hstack([m + s, m - s])


def true_peak_est(x, fs):
    """4x oversampled TP estimate (fast, for normalization; verification is ebur128)."""
    y = resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(max(float(np.abs(y).max()), 1e-9))


def tp_normalize(x, fs, target_db=-0.5):
    g = target_db - true_peak_est(x, fs)
    return x * 10 ** (g / 20), g


def nd2_stage(wav_in, wav_out, stren, bass_scale=1.0, fs=44100):
    """Stereo Tool Natural Dynamics 2 stage. Only free features are used; the
    unlicensed CLI voice-injects on PAID ones (Declipper/Delossifier) — never
    enable those here, and verify any new .sts section on silence first."""
    if not BIN.exists():
        raise RuntimeError(
            f"Stereo Tool CLI not found: {BIN}\n"
            "Download it free for non-commercial use (see docs/stereo-tool.md) "
            "or run with --nd2 0 to skip the punch stage.")
    sts = STS / "base.sts"
    txt = sts.read_text()
    i = txt.index("[Natural Dynamics 2]")
    j = txt.index("[", txt.index("Enabled=", i))
    block = txt[i:j].replace("Enabled=0", "Enabled=1", 1)
    block = block.replace("Difference=0", "Difference=0.5")
    block = block.replace("Global multiplier=1", f"Global multiplier={stren}")
    if bass_scale != 1.0:  # attenuate the punch effect of the two lowest bands (40/90 Hz)
        for band in (1, 2):
            for key in (f"New method Multiply - band {band}=", f"Punch Multiplier - band {band}=", f"Multiplier - band {band}="):
                lines = block.split("\n")
                for i, l in enumerate(lines):
                    if l.startswith(key):
                        v = float(l.split("=")[1])
                        lines[i] = f"{key}{v * bass_scale:.6f}"
                block = "\n".join(lines)
    tmp = STS / f"_nd2_tmp_{os.getpid()}_{id(wav_in) % 9999}.sts"  # unique: parallel processes
    tmp.write_text(txt[:i] + block + txt[j:])
    subprocess.run([str(BIN), str(wav_in), str(wav_out), "-s", str(tmp), "-b", "32f",
                    "-r", str(fs), "-q"], check=True, capture_output=True)
    tmp.unlink()


def tpdf_dither_quantize(x, bits=16):
    """TPDF dither + quantization. Returns integer codes."""
    levels = 2 ** (bits - 1)
    xn = np.clip(x, -1.0, 1.0 - 1.0 / levels) * levels
    d = (np.random.uniform(-1, 1, x.shape) + np.random.uniform(-1, 1, x.shape))  # TPDF, 2 LSB pk-pk
    return np.round(xn + d * 0.5).astype(np.int32)


# ------------------------------------------------------------ pipeline

def master_track(src, dst, slope=1.08, offset=-4.6, side_db=-3.0, tp_target=-0.5,
                 depth=16, reference_eq=True, hpf_hz=None, declip=None, nd2=0.0,
                 nd2_bass_scale=1.0, slew=10.0, env_win=0.3, label=None, outdir=None, keep_tmp=False):
    """The full chain. declip: None=auto from diagnosis, True/False — forced.
    Returns (steps dict, diagnosis profile). Handles any input sample rate."""
    prof = diagnose(src)
    plan = prof["plan"]
    use_declip = declip if declip is not None else plan["declip"]
    if hpf_hz is not None:
        use_hpf = hpf_hz
    elif reference_eq or plan["hpf_25hz"]:
        use_hpf = 25.0  # the reference always cuts rumble; auto triggers on diagnosis
    else:
        use_hpf = None

    tmp = work_tmpdir("master_")
    steps = {"declip": use_declip, "hpf_hz": use_hpf, "reference_eq": reference_eq,
             "slope": slope, "offset": offset, "side_db": side_db, "nd2": nd2}
    try:
        x, fs = decode(src, "f32le")
        x = fix_dc(x)
        wav = tmp / "0.wav"
        cur = x

        if use_declip:
            encode_wav(cur, wav, fs=fs)
            d = tmp / "1_declip.wav"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                            "-af", "adeclip=window=55:overlap=75:arorder=8",
                            "-c:a", "pcm_f32le", str(d)], check=True, capture_output=True)
            cur, _ = decode(d, "f32le")
            wav = d
        if use_hpf:
            cur = hpf(cur, fs, use_hpf)
        if reference_eq:
            cur = ref_eq(cur, fs)
        if nd2 > 0:
            encode_wav(cur, wav if use_declip else tmp / "2a.wav", fs=fs)
            nd2_in = wav if use_declip else tmp / "2a.wav"
            nd2_out = tmp / "2b_nd2.wav"
            nd2_stage(nd2_in, nd2_out, nd2, bass_scale=nd2_bass_scale, fs=fs)
            cur, _ = decode(nd2_out, "f32le")
        cur, gain_db, t_env = contour_expand(cur, fs, slope, offset, max_slew=slew, env_win=env_win)
        steps["gain_min_max_db"] = [round(float(gain_db.min()), 1), round(float(gain_db.max()), 1)]
        if side_db != 0:
            cur = center_focus(cur, side_db)
        cur, g = tp_normalize(cur, fs, tp_target)
        steps["static_gain_db"] = round(g, 2)

        # quantize + write
        codes = tpdf_dither_quantize(cur, depth)
        y = codes.astype(np.float64) / (2 ** (depth - 1))
        encode_wav(y, tmp / "q.wav", "pcm_s16le" if depth == 16 else "pcm_s24le", fs=fs)
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp / "q.wav"),
                        "-c:a", "flac", "-compression_level", "8", str(dst)],
                       check=True, capture_output=True)
        copy_tags(src, dst)
        return steps, prof
    finally:
        if not keep_tmp:
            subprocess.run(["rm", "-rf", str(tmp)])


def contour_fit(before_file, after_file, fs=None):
    """Fit the contour transfer: after_dB ~ slope*before_dB + offset, 1 s windows."""
    xb, fs = decode(before_file, "f32le")
    xa, _ = decode(after_file, "f32le")
    n = min(len(xb), len(xa))
    win = fs
    nw = n // win
    if nw < 10:
        return {}
    rb = 10 * np.log10(np.maximum(2 * np.mean(xb[:nw * win].reshape(nw, win, 2) ** 2, axis=(1, 2)), 1e-12))
    ra = 10 * np.log10(np.maximum(2 * np.mean(xa[:nw * win].reshape(nw, win, 2) ** 2, axis=(1, 2)), 1e-12))
    A = np.vstack([rb, np.ones_like(rb)]).T
    coef, res, *_ = np.linalg.lstsq(A, ra, rcond=None)
    pred = A @ coef
    return {"fit_slope": round(float(coef[0]), 3), "fit_offset_db": round(float(coef[1]), 2),
            "fit_residual_db": round(float(np.std(ra - pred)), 2)}


def check_duration(src, dst, tol=0.05):
    """Loud duration/samplerate check — a 48k source once got relabeled as 44.1k
    (8.8% longer, pitched down) while every loudness metric looked perfect."""
    from drlib.audio import duration_and_fs
    sdur, sfs = duration_and_fs(src)
    ddur, dfs = duration_and_fs(dst)
    if sfs != dfs:
        return f"SAMPLERATE MISMATCH {sfs} -> {dfs}"
    if abs(sdur - ddur) > tol:
        return f"DURATION MISMATCH {sdur:.2f}s -> {ddur:.2f}s"
    return None


def cmd_track(a):
    steps, prof = master_track(a.src, a.dst, a.slope, a.offset, a.side_db, a.tp_target,
                               a.depth, not a.no_reference_eq, a.hpf, a.declip, a.nd2,
                               a.nd2_bass_scale, a.slew, a.env_win, a.label, a.outdir)
    print(json.dumps(steps, ensure_ascii=False, indent=1))
    fit = contour_fit(a.src, a.dst)
    print("contour fit:", json.dumps(fit))
    bad = check_duration(a.src, a.dst)
    if bad:
        print(f"!! {bad} — do NOT deliver this file", flush=True)
        sys.exit(2)
    if a.outdir:
        from drlib.util import python_exe
        subprocess.run([python_exe(), str(Path(__file__).parent / "measure.py"), "compare",
                        str(a.src), str(a.dst), "--outdir", a.outdir, "--label", a.label or "track"])


def cmd_album(a):
    indir, outdir = Path(a.indir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(indir.glob("*.flac"))
    rows = []
    for f in files:
        dst = outdir / f.name
        steps, prof = master_track(f, dst, a.slope, a.offset, a.side_db, a.tp_target,
                                   a.depth, not a.no_reference_eq, a.hpf, a.declip, a.nd2,
                                   a.nd2_bass_scale, a.slew, a.env_win)
        bad = check_duration(f, dst)
        if bad:
            print(f"!! {f.name}: {bad}", flush=True)
            sys.exit(2)
        mb = diagnose(f)
        ma = diagnose(dst)
        rows.append({"track": f.name, "steps": steps,
                     "before": {k: mb[k] for k in ("lufs", "lra", "true_peak_db")},
                     "after": {k: ma[k] for k in ("lufs", "lra", "true_peak_db")},
                     "dr_before": mb, "dr_after": ma})
        print(f"{f.name}: LUFS {mb['lufs']}->{ma['lufs']}  TP {mb['true_peak_db']}->{ma['true_peak_db']}  "
              f"plan: declip={steps['declip']} hpf={steps['hpf_hz']}")
    (outdir / "_report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--slope", type=float, default=1.08)
        sp.add_argument("--offset", type=float, default=-4.6)
        sp.add_argument("--side-db", type=float, default=-3.0)
        sp.add_argument("--tp-target", type=float, default=-0.5)
        sp.add_argument("--depth", type=int, default=16, choices=(16, 24))
        sp.add_argument("--no-reference-eq", action="store_true")
        sp.add_argument("--hpf", type=float, default=None, help="HPF frequency; default = auto")
        sp.add_argument("--declip", type=lambda v: {"on": True, "off": False, "auto": None}[v],
                        default=None, help="on/off/auto")
        sp.add_argument("--nd2", type=float, default=2.5,
                        help="Natural Dynamics 2 strength (0 disables; needs Stereo Tool CLI)")
        sp.add_argument("--nd2-bass-scale", type=float, default=0.35)
        sp.add_argument("--slew", type=float, default=40.0)
        sp.add_argument("--env-win", type=float, default=3.0)

    t = sub.add_parser("track")
    t.add_argument("src")
    t.add_argument("dst")
    t.add_argument("--label", default="track")
    t.add_argument("--outdir", default="compare")
    common(t)
    t.set_defaults(func=cmd_track)

    al = sub.add_parser("album")
    al.add_argument("indir")
    al.add_argument("outdir")
    common(al)
    al.set_defaults(func=cmd_album)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
