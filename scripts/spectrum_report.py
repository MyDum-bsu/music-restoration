#!/usr/bin/env python3
"""Visual restoration report: MasVis-style pages + animated spectrogram GIFs.

For every selected track renders one PNG page (original vs restored on shared
scales) and one GIF alternating the two spectrograms:

  - waveform envelope (min/max per column);
  - spectrograms (log frequency, one absolute dB scale for both versions);
  - normalized average 1/3-octave spectrum;
  - short-term (1 s) crest factor over time.

Usage:
  python3 scripts/spectrum_report.py --orig "<original dir>" \
      --restored "<restored dir>" --out "<spectra dir>" [--tracks 1,5,9|auto]
"""
import argparse
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy import signal  # noqa: E402
from PIL import Image  # noqa: E402

from rnext import features as F  # noqa: E402

VMIN, VMAX = -100.0, -10.0     # shared spectrogram dB scale (dBFS)
WAVE_COLS = 1600               # waveform envelope columns
SPEC_BINS = 220                # log-spaced spectrogram rows


def track_no(name: str) -> int:
    m = re.match(r"\s*(\d{1,3})", name)
    return int(m.group(1)) if m else 0


def clean(stem: str) -> str:
    s = re.sub(r"^\s*\d{1,3}\s*(?:[-._)]|\s)\s*", "", stem.strip(), count=1)
    return s.strip() or stem


def safe_name(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s).strip("_")[:60]


def pick(files, how: str):
    if not files:
        raise SystemExit("no FLAC files")
    nums = [track_no(f.name) for f in files]
    if how != "auto":
        want = {int(x) for x in re.split(r"[,\s]+", how.strip()) if x.strip()}
        return [f for f, n in zip(files, nums) if n in want]
    first, last = nums[0], nums[-1]
    mid = nums[len(nums) // 2]
    keep = {first, mid, last}
    return [f for f, n in zip(files, nums) if n in keep]


def match_by_no(rest_file, orig_files):
    n = track_no(rest_file.name)
    for f in orig_files:
        if track_no(f.name) == n:
            return f
    return None


def envelope(x, cols=WAVE_COLS):
    """Per-column min/max for a waveform plot."""
    edges = np.linspace(0, x.shape[0], cols + 1, dtype=int)
    lo = np.array([x[a:b, 0].min() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    hi = np.array([x[a:b, 1].max() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    mid = (x[:, 0] + x[:, 1]) / 2.0
    lo = np.minimum(lo, np.array([mid[a:b].min() for a, b in zip(edges[:-1], edges[1:]) if b > a]))
    hi = np.maximum(hi, np.array([mid[a:b].max() for a, b in zip(edges[:-1], edges[1:]) if b > a]))
    return lo, hi


def spectrogram(x, sr, fmax=22000):
    """Mono-mid spectrogram resampled onto log-spaced rows, in dBFS."""
    mid = (x[:, 0] + x[:, 1]) / 2.0
    f, t, S = signal.spectrogram(mid, fs=sr, window="hann",
                                 nperseg=2048, noverlap=1536)
    S = 20.0 * np.log10(S + 1e-12)
    S = np.clip(S, VMIN - 20, 0)
    keep = f <= fmax
    f, S = f[keep], S[keep]
    rows = np.geomspace(max(f[1], 20.0), f[-1], SPEC_BINS)
    idx = np.searchsorted(f, rows)
    idx = np.clip(idx, 0, len(f) - 1)
    return S[idx, :], t, rows


def band_edges(sr):
    edges = [20.0]
    while edges[-1] < sr / 2 - 1:
        edges.append(edges[-1] * 2 ** (1 / 3))
    edges[-1] = sr / 2
    return edges


def band_rms(x, sr):
    """Per-band RMS (dB) on 1/3-octave edges — self-contained so that the
    center-frequency axis always matches the data length."""
    import math
    edges = band_edges(sr)
    f, P = signal.welch((x[:, 0] + x[:, 1]) / 2.0, fs=sr, nperseg=8192, noverlap=4096)
    out = []
    for i in range(len(edges) - 1):
        idx = np.where((f >= edges[i]) & (f < edges[i + 1]))[0]
        if idx.size == 0:
            out.append(np.nan)
            continue
        out.append(10.0 * math.log10(float(np.mean(P[idx])) + 1e-20))
    vals = np.array(out, dtype=np.float64)
    centers = 10 ** ((np.log10(edges[:-1]) + np.log10(edges[1:])) / 2)
    return vals, centers


def short_crest(x, sr, win_s=1.0):
    win = int(win_s * sr)
    n = (x.shape[0] // win) * win
    if n < win:
        return np.array([]), np.array([])
    xx = x[:n].reshape(-1, win, x.shape[1])
    pk = 20 * np.log10(np.abs(xx).max(axis=(1, 2)) + 1e-12)
    rms = 20 * np.log10(np.sqrt((xx.astype(np.float64) ** 2).mean(axis=(1, 2))) + 1e-12)
    t = (np.arange(len(pk)) + 0.5) * win_s
    return t, pk - rms


def fig_page(title, xo, xr, so, sr, sr2, label_o, label_r):
    fig = plt.figure(figsize=(14, 11), dpi=110)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.1, 1.4, 1.4, 1.3], hspace=0.45, wspace=0.18)
    fig.suptitle(title, fontsize=13)
    t = np.arange(xo.shape[0]) / sr
    ax = fig.add_subplot(gs[0, :])
    lo, hi = envelope(xo)
    tw = np.linspace(0, t[-1], len(lo))
    ax.fill_between(tw, lo, hi, color="#1f77b4", alpha=0.55, linewidth=0, label=label_o)
    tr = np.arange(xr.shape[0]) / sr
    lor, hir = envelope(xr)
    twr = np.linspace(0, tr[-1], len(lor))
    ax.fill_between(twr, lor, hir, color="#d62728", alpha=0.45, linewidth=0, label=label_r)
    ax.set_ylim(-1, 1), ax.set_ylabel("amplitude"), ax.legend(loc="lower left", fontsize=8)
    ax.set_title("waveform envelope", fontsize=9, loc="left")
    for row, (S, name) in enumerate(((so["orig"], label_o), (so["rest"], label_r))):
        ax = fig.add_subplot(gs[row + 1, :])
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, so["t"][-1], 0, len(so["rows"])], vmin=VMIN, vmax=VMAX)
        ax.set_yticks([0, len(so["rows"]) // 2, len(so["rows"]) - 1])
        ax.set_yticklabels([f"{so['rows'][0]:.0f}", f"{so['rows'][len(so['rows'])//2]:.0f}",
                            f"{so['rows'][-1]:.0f}"])
        ax.set_ylabel("Hz (log)"), ax.set_title(f"spectrogram — {name}", fontsize=9, loc="left")
        if row == 1:
            ax.set_xlabel("time, s")
    ax = fig.add_subplot(gs[3, 0])
    fo, co = band_rms(xo, sr)
    fr, cr = band_rms(xr, sr2)
    m = np.nanmean(fo)
    ax.semilogx(co, fo - m, label=label_o, color="#1f77b4")
    ax.semilogx(cr, fr - m, label=label_r, color="#d62728")
    ax.grid(True, which="both", alpha=0.3), ax.set_xlabel("Hz"), ax.set_ylabel("dB rel. orig mean")
    ax.set_title("normalized average spectrum (1/3 oct)", fontsize=9, loc="left")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[3, 1])
    tc, co = short_crest(xo, sr)
    _, cr = short_crest(xr, sr)
    ax.plot(tc, co, label=label_o, color="#1f77b4", linewidth=0.8)
    ax.plot(np.arange(len(cr)) + 0.5, cr, label=label_r, color="#d62728", linewidth=0.8)
    ax.grid(True, alpha=0.3), ax.set_xlabel("time, s"), ax.set_ylabel("dB")
    ax.set_title("short-term (1 s) crest factor", fontsize=9, loc="left")
    ax.legend(fontsize=8)
    return fig


def fig_to_frames(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())


def gif(frames, path, ms=800):
    imgs = [Image.fromarray(f[:, :, :3]).convert("P", palette=Image.ADAPTIVE, colors=128)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:] + imgs[:-1][::-1],
                 duration=ms, loop=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orig", required=True)
    ap.add_argument("--restored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracks", default="auto", help='"auto" or comma-separated numbers')
    args = ap.parse_args()
    orig, rest = pathlib.Path(args.orig), pathlib.Path(args.restored)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r_files = [p for p in sorted(rest.glob("*.flac")) if not p.name.startswith("_")]
    o_files = [p for p in sorted(orig.glob("*.flac")) if not p.name.startswith("_")]
    picked = pick(r_files, args.tracks)
    label_o, label_r = "original", "restored"
    lines = ["# Spectra", ""]
    for rf in picked:
        of = match_by_no(rf, o_files)
        if of is None:
            print(f"[skip] no original for {rf.name}")
            continue
        ti = track_no(rf.name)
        title = f"{ti:02d}. {clean(rf.stem)}  —  {label_o} vs {label_r}"
        print(f"[{ti:02d}] {clean(rf.stem)}", flush=True)
        xo, sr = F.decode_f32(str(of))
        xr, sr2 = F.decode_f32(str(rf))
        n = min(len(xo), len(xr))
        xo, xr = xo[:n], xr[:n]
        so_o, t, rows = spectrogram(xo, sr)
        so_r, _, _ = spectrogram(xr, sr2)
        so = {"orig": so_o, "rest": so_r, "t": t, "rows": rows}
        fig = fig_page(title, xo, xr, so, sr, sr2, label_o, label_r)
        png = out / f"{ti:02d}_{safe_name(clean(rf.stem))}.png"
        fig.savefig(png)
        # GIF: alternate the two spectrogram panels on a black page
        spec_frames = []
        for S in (so_o, so_r):
            fg = plt.figure(figsize=(7, 4), dpi=100)
            fg.patch.set_facecolor("black")
            ax = fg.add_axes([0.08, 0.12, 0.86, 0.8])
            ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                      extent=[0, t[-1], 0, len(rows)], vmin=VMIN, vmax=VMAX)
            ax.set_yticks([0, len(rows) // 2, len(rows) - 1])
            ax.set_yticklabels([f"{rows[0]:.0f}", f"{rows[len(rows)//2]:.0f}", f"{rows[-1]:.0f}"])
            ax.set_xlabel("time, s"), ax.set_ylabel("Hz (log)")
            ax.set_title(f"spectrogram — {label_o if S is so_o else label_r}",
                         color="white", fontsize=10)
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_color("white")
            spec_frames.append(fig_to_frames(fg))
            plt.close(fg)
        gif(spec_frames, out / f"{ti:02d}_{safe_name(clean(rf.stem))}.gif")
        lines.append(f"- [{png.name}]({png.name}) / [{png.stem}.gif]({png.stem}.gif) — {title}")
        plt.close(fig)
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"spectra → {out} ({len(lines)-2} tracks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
