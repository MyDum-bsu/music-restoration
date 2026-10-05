Stereo Tool CLI goes here.

Download the Linux command-line binary ("command line version") free for
non-commercial use from https://www.stereo-tool.com and put it in this folder
as `stereo_tool_cmd_64` (chmod +x), or set `MR_STEREO_TOOL_BIN` in
~/.config/music-restoration/config.env to any path.

Why not bundled: Stereo Tool is freeware but not redistributable.

Notes:
- Only FREE features are used (Natural Dynamics 2, multiband, EQ, neutral
  pass). The unlicensed binary voice-injects on PAID features (Declipper,
  Delossifier) — never enable those in an .sts preset.
- Sanity check any new preset section on silence: the output must stay
  bit-transparent for a neutral base.sts pass (max diff 1 code).
