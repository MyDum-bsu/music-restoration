#!/usr/bin/env python3
"""Render the same orig/restored spectrum page in several premium styles.

Outputs /tmp/spectra-styles/{A,B,C,D}.png + ALL.png contact sheet.
Reuses the computation from scripts/spectrum_report.py.
"""
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from PIL import Image  # noqa: E402

from rnext import features as F  # noqa: E402
from spectrum_report import (  # noqa: E402
    VMIN, VMAX, SPEC_BINS, band_rms, clean, envelope, short_crest, spectrogram,
)

ORIG = "/home/mydum/Music/Helloween/2015 - My God-Given Right [restored]/09. Creatures in Heaven.flac"
REST = "/tmp/spectrum_test/rest/01. Creatures in Heaven.flac"
OUT = pathlib.Path("/tmp/spectra-styles")

STYLES = {
    "A": dict(label="Nightlab — dark studio, amber/cyan",
              bg="#0B0E14", fg="#E9EDF5", muted="#8A93A6", grid="#1D2433",
              c_o="#59C2F0", c_r="#F2B441", spec=["#0B0E14", "#5A3A10", "#C98A1E", "#FFE3AE"],
              font="DejaVu Sans", caps=True),
    "B": dict(label="Gallery — light print, ink & vermilion",
              bg="#FAF7F1", fg="#1A1917", muted="#716C63", grid="#E7E0D4",
              c_o="#37567A", c_r="#C2402A", spec=["#FAF7F1", "#C9A18E", "#6E3B2C", "#1A1917"],
              font="DejaVu Serif", caps=False),
    "C": dict(label="Blueprint — drafting table, cyan on blue",
              bg="#0F2D4E", fg="#DDEAF8", muted="#93B3D1", grid="#24486F",
              c_o="#FFD166", c_r="#8FE3FF", spec=["#0B2440", "#1C4E7E", "#57A5D8", "#CFF0FF"],
              font="DejaVu Sans Mono", caps=True),
    "D": dict(label="Onyx & Gold — black, serif, gold",
              bg="#0A0A0B", fg="#EDEAE2", muted="#8F8A7E", grid="#232320",
              c_o="#B9BEC7", c_r="#D4B36A", spec=["#0A0A0B", "#3F3316", "#9C7F3C", "#F0DCA8"],
              font="DejaVu Serif", caps=False),
    "E": dict(label="3b1b — manim blue vs yellow, soft glow",
              bg="#0E1116", fg="#F2F2F2", muted="#8B93A0", grid="#1E2630",
              c_o="#58C4DD", c_r="#F4D345", spec=["#0E1116", "#123B5E", "#1F6FA3", "#4FC3E8", "#CFF4FF"],
              font="DejaVu Sans", caps=False, glow=True, rule="#2A3140"),
    "G": dict(label="Cognac & Cream — soft black, yellow, bone",
              bg="#0D0C0A", fg="#EFE8DA", muted="#97907F", grid="#23201A",
              c_o="#A9986F", c_r="#E9C464", spec=["#0D0C0A", "#2E2718", "#6B5526", "#C9A24D", "#F4E3B2"],
              font="Source Serif Pro", caps=False, glow=True, rule="#3A342A"),
    "H": dict(label="Espresso & Gold — copper vs gold",
              bg="#0B0A08", fg="#F1EADC", muted="#8E8574", grid="#211E17",
              c_o="#B0713E", c_r="#F2C94C", spec=["#0B0A08", "#241C10", "#5E4A22", "#B98F35", "#FBE9B8"],
              font="Source Serif Pro", caps=False, glow=True, rule="#3A342A"),
    "I": dict(label="Noir Doux — warm gray vs yellow",
              bg="#0A0908", fg="#ECE6D9", muted="#857D6F", grid="#1E1B15",
              c_o="#8A8172", c_r="#EDBF4E", spec=["#0A0908", "#1F1B12", "#4E4120", "#A98A34", "#EFD9A0"],
              font="Source Serif Pro", caps=False, glow=True, rule="#3A342A"),
    "F": dict(label="3b1b — manim blue vs coral, soft glow",
              bg="#0E1116", fg="#F2F2F2", muted="#8B93A0", grid="#1E2630",
              c_o="#58C4DD", c_r="#FF7A6E", spec=["#0E1116", "#2A2547", "#71479B", "#D97BA6", "#FFD9CB"],
              font="DejaVu Sans", caps=False, glow=True, rule="#2A3140"),
}


def spec_cmap(colors):
    return LinearSegmentedColormap.from_list("duo", colors, N=256)


def render(style, data, tag):
    s = STYLES[style]
    xo, xr = data["orig"], data["rest"]
    so, sr_, sr2 = data["so"], data["sr"], data["sr2"]
    co, fo, fr = data["c"]
    tc, cr0, cr1 = data["crest"]
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.facecolor": s["bg"], "axes.facecolor": s["bg"],
        "savefig.facecolor": s["bg"], "text.color": s["fg"],
        "axes.edgecolor": s["muted"], "axes.labelcolor": s["muted"],
        "xtick.color": s["muted"], "ytick.color": s["muted"],
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.labelsize": 8.5, "grid.color": s["grid"],
        "grid.linewidth": 0.5, "font.family": s["font"],
        "axes.linewidth": 0.8,
    })
    cmap = spec_cmap(s["spec"])
    fig = plt.figure(figsize=(14, 11), dpi=110)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.35, 1.35, 1.25],
                  hspace=0.52, wspace=0.16,
                  left=0.065, right=0.965, top=0.872, bottom=0.065)
    # header plate
    kick = "SPECTRUM REPORT — ORIGINAL VS RESTORED"
    fig.text(0.065, 0.965, kick.upper() if s["caps"] else kick, fontsize=8,
             color=s["muted"], family=s["font"])
    fig.text(0.065, 0.925, data["title"], fontsize=17, color=s["fg"], family=s["font"])
    fig.text(0.065, 0.912, data["meta"], fontsize=9, color=s["muted"], family=s["font"])
    fig.lines.append(plt.Line2D([0.065, 0.935], [0.893, 0.893], transform=fig.transFigure,
                                color=s.get("rule", s["c_o"]), linewidth=1.1, alpha=0.9))
    for ax in fig.axes:
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontfamily("JetBrains Mono")

    def panel(ax, title):
        ax.set_title(title, fontsize=8.5, color=s["muted"], loc="left", pad=6,
                     fontfamily=s["font"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)

    t = np.arange(xo.shape[0]) / sr_
    ax = fig.add_subplot(gs[0, :])
    lo, hi = envelope(xo)
    tw = np.linspace(0, t[-1], len(lo))
    ax.fill_between(tw, lo, hi, color=s["c_o"], alpha=0.62, linewidth=0, label="original")
    lor, hir = envelope(xr)
    twr = np.linspace(0, t[-1], len(lor))
    ax.fill_between(twr, lor, hir, color=s["c_r"], alpha=0.5, linewidth=0, label="restored")
    ax.set_ylim(-1.02, 1.02)
    ax.set_yticks([-1, 0, 1])
    panel(ax, "WAVEFORM ENVELOPE" if s["caps"] else "Waveform envelope")
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)

    for row, (S, name) in enumerate(((so["orig"], "original"), (so["rest"], "restored"))):
        ax = fig.add_subplot(gs[row + 1, :])
        ax.imshow(S, origin="lower", aspect="auto", cmap=cmap,
                  extent=[0, so["t"][-1], 0, len(so["rows"])], vmin=VMIN, vmax=VMAX,
                  interpolation="nearest")
        ax.set_yticks([0, len(so["rows"]) // 2, len(so["rows"]) - 1])
        ax.set_yticklabels([f"{so['rows'][0]:.0f}", f"{so['rows'][len(so['rows'])//2]:.0f}",
                            f"{so['rows'][-1]:.0f}"])
        panel(ax, f"SPECTROGRAM — {name.upper()}" if s["caps"] else f"Spectrogram — {name}")
        if row == 1:
            ax.set_xlabel("time, s")

    ax = fig.add_subplot(gs[3, 0])
    def line(ax, x, y, color, label, lw=1.7):
        if s.get("glow"):
            ax.semilogx(x, y, color=color, lw=lw*3.2, alpha=0.12, label=None)
            ax.semilogx(x, y, color=color, lw=lw*1.8, alpha=0.30, label=None)
        ax.semilogx(x, y, color=color, lw=lw, label=label)
    line(ax, co, fo, s["c_o"], "original")
    line(ax, co, fr, s["c_r"], "restored")
    ax.grid(True, which="major", alpha=0.55), ax.grid(True, which="minor", alpha=0.18)
    panel(ax, "AVERAGE SPECTRUM, 1/3 OCT (NORMALIZED)" if s["caps"]
          else "Average spectrum, 1/3 oct (normalized)")
    ax.set_xlabel("Hz"), ax.set_ylabel("dB")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[3, 1])
    def line2(ax, x, y, color, label, lw=1.1):
        if s.get("glow"):
            ax.plot(x, y, color=color, lw=lw*3.0, alpha=0.10, label=None)
            ax.plot(x, y, color=color, lw=lw*1.7, alpha=0.28, label=None)
        ax.plot(x, y, color=color, lw=lw, label=label)
    line2(ax, tc, cr0, s["c_o"], "original")
    line2(ax, np.arange(len(cr1)) + 0.5, cr1, s["c_r"], "restored")
    ax.grid(True, alpha=0.55)
    panel(ax, "SHORT-TERM (1 s) CREST FACTOR" if s["caps"] else "Short-term (1 s) crest factor")
    ax.set_xlabel("time, s"), ax.set_ylabel("dB")
    ax.legend(fontsize=8, frameon=False)

    out = OUT / f"{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xo, sr = F.decode_f32(ORIG)
    xr, sr2 = F.decode_f32(REST)
    n = min(len(xo), len(xr))
    xo, xr = xo[:n], xr[:n]
    so_o, t, rows = spectrogram(xo, sr)
    so_r, _, _ = spectrogram(xr, sr2)
    so = {"orig": so_o, "rest": so_r, "t": t, "rows": rows}
    fo, co = band_rms(xo, sr)
    fr, _ = band_rms(xr, sr2)
    tc, cr0 = short_crest(xo, sr)
    _, cr1 = short_crest(xr, sr2)
    data = {"title": "9. Creatures in Heaven", "orig": xo, "rest": xr, "so": so, "sr": sr,
            "meta": "Helloween — My God-Given Right (2015) · flatten k=0.5 cap=8 · DR 11→11",
            "c": (co, fo, fr), "crest": (tc, cr0, cr1), "sr2": sr2}
    outs = []
    for tag in ("G", "H", "I"):
        p = render(tag, data, tag)
        print("rendered", p)
        outs.append(Image.open(p))
    w, h = outs[0].size
    sheet = Image.new("RGB", (w * 2 + 12, h * 2 + 12), "#2B2B2B")
    for i, im in enumerate(outs):
        sheet.paste(im, ((i % 2) * (w + 12), (i // 2) * (h + 12)))
    sheet.thumbnail((2400, 2400))
    sheet.save(OUT / "ALL.png")
    print("sheet:", OUT / "ALL.png")


if __name__ == "__main__":
    main()
