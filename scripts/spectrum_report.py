#!/usr/bin/env python3
"""Visual restoration report: waveform/spectrogram pages + animated GIFs.

For every selected track renders one PNG page (original vs restored on shared
scales) and one GIF alternating the two spectrograms:

  - waveform envelope: restored fill + original outline;
  - spectrograms (log frequency, one absolute dB scale for both versions,
    each normalized to its own loudest bin);
  - normalized average 1/3-octave spectrum;
  - short-term (1 s) crest factor over time.

Style presets ("espresso" default): Source Serif Pro text, JetBrains Mono
tick numbers, black & gold duotone spectrograms.

Usage:
  python3 scripts/spectrum_report.py --orig "<original dir>" \
      --restored "<restored dir>" --out "<spectra dir>" \
      [--tracks 1,5,9|auto] [--style espresso|cognac|noir]
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy import signal  # noqa: E402
from PIL import Image  # noqa: E402

from rnext import features as F  # noqa: E402


def _register_fonts() -> None:
    """Make the brand fonts visible to matplotlib even with a cold cache."""
    from matplotlib import font_manager as fm
    patterns = ("SourceSerifPro*", "SourceSerif4*", "JetBrainsMono-*",
                "JetBrainsMono*", "JetBrainsMonoNL*")
    roots = [pathlib.Path.home() / ".local/share/fonts",
             pathlib.Path("/usr/local/share/fonts"),
             pathlib.Path("/usr/share/fonts"),
             pathlib.Path.home() / ".fonts"]
    for root in roots:
        if not root.is_dir():
            continue
        for pat in patterns:
            for f in root.rglob(pat):
                if f.suffix.lower() in (".ttf", ".otf"):
                    try:
                        fm.fontManager.addfont(str(f))
                    except Exception:  # noqa: BLE001
                        pass


_register_fonts()

VMIN, VMAX = -100.0, -10.0     # shared spectrogram dB window (after normalization)
WAVE_COLS = 1600               # waveform envelope columns
SPEC_BINS = 220                # log-spaced spectrogram rows

SERIF = "Source Serif Pro"
MONO = "JetBrains Mono"

PRESETS = {
    "espresso": dict(
        bg="#0B0A08", fg="#F1EADC", muted="#8E8574", grid="#211E17",
        c_o="#B0713E", c_r="#F2C94C", rule="#3A342A",
        spec=["#0B0A08", "#241C10", "#5E4A22", "#B98F35", "#FBE9B8"]),
    "cognac": dict(
        bg="#0D0C0A", fg="#EFE8DA", muted="#97907F", grid="#23201A",
        c_o="#A9986F", c_r="#E9C464", rule="#3A342A",
        spec=["#0D0C0A", "#2E2718", "#6B5526", "#C9A24D", "#F4E3B2"]),
    "noir": dict(
        bg="#0A0908", fg="#ECE6D9", muted="#857D6F", grid="#1E1B15",
        c_o="#8A8172", c_r="#EDBF4E", rule="#3A342A",
        spec=["#0A0908", "#1F1B12", "#4E4120", "#A98A34", "#EFD9A0"]),
}


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
    keep = {nums[0], nums[len(nums) // 2], nums[-1]}
    return [f for f, n in zip(files, nums) if n in keep]


def match_by_no(rest_file, orig_files):
    n = track_no(rest_file.name)
    for f in orig_files:
        if track_no(f.name) == n:
            return f
    return None


# ── computation ──────────────────────────────────────────────────────────────

def envelope(x, cols=WAVE_COLS):
    edges = np.linspace(0, x.shape[0], cols + 1, dtype=int)
    spans = list(zip(edges[:-1], edges[1:]))
    lo = np.array([x[a:b, 0].min() for a, b in spans if b > a])
    hi = np.array([x[a:b, 1].max() for a, b in spans if b > a])
    mid = (x[:, 0] + x[:, 1]) / 2.0
    lo = np.minimum(lo, np.array([mid[a:b].min() for a, b in spans if b > a]))
    hi = np.maximum(hi, np.array([mid[a:b].max() for a, b in spans if b > a]))
    return lo, hi


def spectrogram(x, sr, fmax=22000):
    mid = (x[:, 0] + x[:, 1]) / 2.0
    f, t, S = signal.spectrogram(mid, fs=sr, window="hann",
                                 nperseg=2048, noverlap=1536)
    S = 20.0 * np.log10(S + 1e-12)
    S -= S.max()                       # normalize: full scale = loudest bin
    S = np.clip(S, VMIN - 10, 0)
    keep = f <= fmax
    f, S = f[keep], S[keep]
    rows = np.geomspace(max(f[1], 20.0), f[-1], SPEC_BINS)
    idx = np.clip(np.searchsorted(f, rows), 0, len(f) - 1)
    return S[idx, :], t, rows


def band_edges(sr):
    edges = [20.0]
    while edges[-1] < sr / 2 - 1:
        edges.append(edges[-1] * 2 ** (1 / 3))
    edges[-1] = sr / 2
    return edges


def band_rms(x, sr):
    """Per-band RMS (dB) on 1/3-octave edges; centers always match the data."""
    import math
    edges = band_edges(sr)
    f, P = signal.welch((x[:, 0] + x[:, 1]) / 2.0, fs=sr, nperseg=8192, noverlap=4096)
    out = []
    for i in range(len(edges) - 1):
        idx = np.where((f >= edges[i]) & (f < edges[i + 1]))[0]
        out.append(10.0 * math.log10(float(np.mean(P[idx])) + 1e-20) if idx.size else np.nan)
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
    return (np.arange(len(pk)) + 0.5) * win_s, pk - rms


# ── rendering ────────────────────────────────────────────────────────────────

def glow(ax, x, y, color, label, lw, logx=False):
    plot = ax.semilogx if logx else ax.plot
    plot(x, y, color=color, lw=lw * 3.2, alpha=0.12, label=None)
    plot(x, y, color=color, lw=lw * 1.8, alpha=0.30, label=None)
    plot(x, y, color=color, lw=lw, label=label)


def fig_page(s, cmap, meta, title, xo, xr, so, sr, sr2):
    fig = plt.figure(figsize=(14, 11), dpi=110)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.35, 1.35, 1.25],
                  hspace=0.52, wspace=0.16, left=0.065, right=0.965,
                  top=0.862, bottom=0.065)
    # header: kicker / title / meta / rule — with room to breathe
    fig.text(0.065, 0.968, "SPECTRUM REPORT — ORIGINAL VS RESTORED",
             fontsize=8, color=s["muted"])
    fig.text(0.065, 0.928, title, fontsize=17, color=s["fg"], family=SERIF)
    fig.text(0.065, 0.900, meta, fontsize=9, color=s["muted"], family=SERIF)
    fig.lines.append(plt.Line2D([0.065, 0.935], [0.878, 0.878], transform=fig.transFigure,
                                color=s["rule"], linewidth=1.1))

    def panel(ax, title_):
        ax.set_title(title_, fontsize=8.5, color=s["muted"], loc="left", pad=6,
                     fontfamily=SERIF)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    t = np.arange(xo.shape[0]) / sr
    ax = fig.add_subplot(gs[0, :])
    lor, hir = envelope(xr)
    twr = np.linspace(0, t[-1], len(lor))
    ax.fill_between(twr, lor, hir, color=s["c_r"], alpha=0.5, linewidth=0, label="restored")
    lo, hi = envelope(xo)
    tw = np.linspace(0, t[-1], len(lo))
    ax.plot(tw, lo, color=s["c_o"], lw=0.9, alpha=0.95, label="original")
    ax.plot(tw, hi, color=s["c_o"], lw=0.9, alpha=0.95, label=None)
    ax.set_ylim(-1.02, 1.02), ax.set_yticks([-1, 0, 1])
    panel(ax, "Waveform envelope")
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)

    for row, (S, name) in enumerate(((so["orig"], "original"), (so["rest"], "restored"))):
        ax = fig.add_subplot(gs[row + 1, :])
        ax.imshow(S, origin="lower", aspect="auto", cmap=cmap,
                  extent=[0, so["t"][-1], 0, len(so["rows"])], vmin=VMIN, vmax=VMAX,
                  interpolation="nearest")
        ax.set_yticks([0, len(so["rows"]) // 2, len(so["rows"]) - 1])
        ax.set_yticklabels([f"{so['rows'][0]:.0f}", f"{so['rows'][len(so['rows'])//2]:.0f}",
                            f"{so['rows'][-1]:.0f}"], fontfamily=MONO)
        ax.set_ylabel("Hz (log)")
        panel(ax, f"Spectrogram — {name}")
        if row == 1:
            ax.set_xlabel("time, s")

    ax = fig.add_subplot(gs[3, 0])
    fo, co = band_rms(xo, sr)
    fr, _ = band_rms(xr, sr2)
    m = np.nanmean(fo)
    glow(ax, co, fo - m, s["c_o"], "original", 1.6, logx=True)
    glow(ax, co, fr - m, s["c_r"], "restored", 1.6, logx=True)
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)
    ax.set_xlabel("Hz"), ax.set_ylabel("dB rel. original mean")
    panel(ax, "Average spectrum, 1/3 oct (normalized)")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[3, 1])
    tc, co_st = short_crest(xo, sr)
    _, cr_st = short_crest(xr, sr2)
    glow(ax, tc, co_st, s["c_o"], "original", 1.1)
    glow(ax, np.arange(len(cr_st)) + 0.5, cr_st, s["c_r"], "restored", 1.1)
    ax.grid(True, alpha=0.4), ax.set_xlabel("time, s"), ax.set_ylabel("dB")
    panel(ax, "Short-term (1 s) crest factor")
    ax.legend(fontsize=8, frameon=False)

    for ax in fig.axes:
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontfamily(MONO)
    return fig


def meta_line(directory: pathlib.Path, files: list) -> str:
    """Artist — Album (year) from the tags of the first track."""
    try:
        from mutagen.flac import FLAC
        f = FLAC(str(files[0]))
        tags = f.tags or {}

        def g(k):
            v = tags.get(k) or []
            return v[0] if v else ""

        artist = g("albumartist") or g("artist")
        album, date = g("album"), g("date")[:4]
        bits = [b for b in (f"{artist} — {album}" if artist and album else "",
                            f"({date})" if date else "") if b]
        if bits:
            return " ".join(bits)
    except Exception:  # noqa: BLE001
        pass
    return directory.name


def spec_frames(s, cmap, so, t, rows):
    frames = []
    for S, name in ((so["orig"], "original"), (so["rest"], "restored")):
        fg = plt.figure(figsize=(7, 4), dpi=100)
        fg.patch.set_facecolor(s["bg"])
        ax = fg.add_axes([0.08, 0.12, 0.86, 0.8])
        ax.imshow(S, origin="lower", aspect="auto", cmap=cmap,
                  extent=[0, t[-1], 0, len(rows)], vmin=VMIN, vmax=VMAX)
        ax.set_yticks([0, len(rows) // 2, len(rows) - 1])
        ax.set_yticklabels([f"{rows[0]:.0f}", f"{rows[len(rows)//2]:.0f}", f"{rows[-1]:.0f}"],
                           fontfamily=MONO)
        ax.set_xlabel("time, s"), ax.set_ylabel("Hz (log)")
        ax.set_title(f"spectrogram — {name}", color=s["fg"], fontsize=10, family=SERIF)
        ax.tick_params(colors=s["muted"])
        for sp in ax.spines.values():
            sp.set_color(s["grid"])
        fg.canvas.draw()
        frames.append(np.asarray(fg.canvas.buffer_rgba()))
        plt.close(fg)
    return frames


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
    ap.add_argument("--style", default="espresso", choices=sorted(PRESETS))
    args = ap.parse_args()
    s = PRESETS[args.style]
    cmap = LinearSegmentedColormap.from_list("duo", s["spec"], N=256)
    plt.rcParams.update({
        "figure.facecolor": s["bg"], "axes.facecolor": s["bg"],
        "savefig.facecolor": s["bg"], "text.color": s["fg"],
        "axes.edgecolor": s["muted"], "axes.labelcolor": s["muted"],
        "xtick.color": s["muted"], "ytick.color": s["muted"],
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8.5,
        "grid.color": s["grid"], "grid.linewidth": 0.5,
        "font.family": SERIF, "axes.linewidth": 0.8,
    })
    orig, rest = pathlib.Path(args.orig), pathlib.Path(args.restored)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r_files = [p for p in sorted(rest.glob("*.flac")) if not p.name.startswith("_")]
    o_files = [p for p in sorted(orig.glob("*.flac")) if not p.name.startswith("_")]
    meta = meta_line(rest, r_files)
    lines = ["# Spectra", ""]
    for rf in pick(r_files, args.tracks):
        of = match_by_no(rf, o_files)
        if of is None:
            print(f"[skip] no original for {rf.name}")
            continue
        ti = track_no(rf.name)
        name = clean(rf.stem)
        print(f"[{ti}] {name}", flush=True)
        xo, sr = F.decode_f32(str(of))
        xr, sr2 = F.decode_f32(str(rf))
        n = min(len(xo), len(xr))
        xo, xr = xo[:n], xr[:n]
        so_o, t, rows = spectrogram(xo, sr)
        so_r, _, _ = spectrogram(xr, sr2)
        so = {"orig": so_o, "rest": so_r, "t": t, "rows": rows}
        title = f"{ti}. {name}"
        fig = fig_page(s, cmap, meta, title, xo, xr, so, sr, sr2)
        png = out / f"{ti}_{safe_name(name)}.png"
        fig.savefig(png)
        plt.close(fig)
        gif(spec_frames(s, cmap, so, t, rows),
            out / f"{ti}_{safe_name(name)}.gif")
        lines.append(f"- [{png.name}]({png.name}) / [{png.stem}.gif]({png.stem}.gif) — {title}")
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"spectra → {out} ({len(lines)-2} tracks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
