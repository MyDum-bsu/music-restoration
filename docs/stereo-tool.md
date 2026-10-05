# Stereo Tool CLI — setup and safety

The ND2 punch stage uses the Thimeo Stereo Tool command-line binary
(`stereo_tool_cmd_64`). It is **free for non-commercial use but not
redistributable**, so it is not bundled.

## Install

1. Download the Linux **command line version** from <https://www.stereo-tool.com>.
2. Put the binary in `restore-dr/bin/stereo_tool_cmd_64` (or anywhere) and
   point `MR_STEREO_TOOL_BIN` at it in `~/.config/music-restoration/config.env`.
3. `chmod +x` it.

Without the binary everything else still works: diagnose, measure, scan_dr,
and the contour/EQ/HPF pipeline (run with `--nd2 0`).

## Free vs paid — critical

The unlicensed CLI **voice-injects** (a "voice tracking" watermark) when an
.sts preset enables PAID features — notably the **Declipper** and
**Delossifier**. music-restoration only ever enables **Natural Dynamics 2**
(a free feature) on top of the neutral `sts/base.sts` export. Declipping is
done with ffmpeg's `adeclip` instead, precisely to stay on free features.

Rules:

- Never enable paid sections in an .sts preset here.
- Sanity-check any new/edited preset on **digital silence**: a neutral pass of
  base.sts must stay bit-transparent (max diff 1 code).
- If an output ever sounds "hollow" or picks up strange tones, first suspect
  a paid feature quietly enabled by a preset edit.

## What ND2 is doing here

Natural Dynamics 2 restores per-band punch (transient vs sustain within each
of 9 bands). It is a separate mechanism from slow contour expansion — that
split (contour via numpy, punch via ND2) is what lets the pipeline match a
professional restoration. `master.py` writes a temporary .sts with only the
ND2 section enabled and the requested global multiplier (bass bands scaled by
`--nd2-bass-scale`).
