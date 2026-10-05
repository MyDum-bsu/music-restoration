---
name: restore-dr
description: DR-restoration pipeline v2 (master-like) — diagnose → fix (DC/HPF/declip) → reference EQ → contour expansion → ND2 punch → center focus → TP normalize → FLAC+RG, with TT DR/LUFS metrics and before/after plots. Use when the user says "реставрируй альбом", "подними DR", "restore dynamic range", or hands you an over-compressed album to restore.
---

# Restore DR v2 (master-like dynamic restoration)

Undo the loudness war on a COPY. Transparency-first: stop when metrics,
artifacts, or blind ABX say quality degrades.

## Requirements

- ffmpeg (adeclip, ebur128), metaflac, flac, python3 + numpy/scipy/matplotlib/mutagen
- Thimeo Stereo Tool CLI (free for non-commercial) in `restore-dr/bin/stereo_tool_cmd_64`.
  CRITICAL: the unlicensed CLI voice-injects on PAID features (Declipper/Delossifier).
  Free: Natural Dynamics 2, multiband, EQ, neutral pass. Verify any new .sts
  section on silence. See docs/stereo-tool.md.
- `sts/base.sts` — neutral export (bit-transparent pass, max diff 1 code)
- Paths are configured in `~/.config/music-restoration/config.env` (phone/music
  dir auto-detected if unset). See config.env.example.

## Pipeline (scripts/master.py, defaults = mimicry of the reverse-engineered reference)

```
fix DC → [adeclip if diagnosed] → [HPF 25 Hz 4th-order zero-phase]
→ ref EQ (lowshelf +0.9@150, bell −1.6@315 Q0.7, air +0.45@9k)
→ CONTOUR EXPANDER (3 s env + 0.5 s smoothing, gain=(slope−1)·env+offset,
   slew 40 dB/s; slope 1.08 offset −4.6)
→ ND2 (Stereo Tool Natural Dynamics 2, global 2.5, bass bands 1-2 ×0.35)
→ center focus (side −3 dB) → TP normalize −0.5 dBFS → TPDF dither → FLAC + tags + RG2.0
```

- `master.py track IN.flac OUT.flac [--slope --offset --side-db --tp-target --depth
  --no-reference-eq --hpf --declip on/off/auto --nd2 --nd2-bass-scale --slew --env-win]`
  — duration/samplerate are checked loudly at the end (exit 2 on mismatch).
- `master.py album INDIR OUTDIR` — per-track diagnosis-driven, writes `_report.json`
- `diagnose.py FILE...` — clip density/flat runs, DC, subsonic <25 Hz, spectral
  tilt/harshness, noise floor, crest percentiles, env3s std, M/S, plan JSON
- `measure.py metrics|compare` — TT DR + LUFS + clips + comparison PNG plots
- `restore_batch.py ALBUM` — the full one-command loop: pull → restore → covers →
  fill tags (optional --mb MusicBrainz/CAA) → [--lyrics] → DR.txt → HARD GATE →
  push + per-file md5 verify → cleanup. Quarantine, never silent skip.
- `scan_dr.py [ROOT]` — TT DR of a whole library, resume-safe, skips `[restored]`
- `acceptance.py --orig D --mine D [--ref D]` — three-way acceptance table

## Verified facts (do not re-derive)

- TT DR algorithm (= dr14_t.meter): 3 s non-overlap RMS blocks (44.1k: 132480
  samples), ×√2 sine-ref, top-20% energy mean, 2nd-highest block peak,
  round(mean across channels). Validated against community values.
- Reference chain (reverse-engineered from a professional restoration): declip
  (often cosmetic), steep HPF ~25 Hz, taste EQ, **slow wideband envelope
  expansion slope ~1.08–1.11 offset ~−4.6 dB (residual ~0.5 dB)**, side −3 dB,
  TP −0.5 dBFS. One batch chain for the whole album. Typical result DR4→DR9.
- Contour gain alone does NOT raise TT DR (peaks ride the same gain): macro
  expansion + transient punch restoration are SEPARATE mechanisms. The split:
  contour (slow gain) + ND2 (per-band punch). Combined ≈ reference.
- env window matters for LRA: 300 ms detector eats post-rock macro swings
  (LRA 17→11); 3 s window preserves them.
- ND2 strength ↔ LUFS trade at fixed TP ceiling: 2.0 → DR8.6/−12.9;
  2.5 → DR9.4/−13.6. ReplayGain makes playback loudness equal anyway.
- ffmpeg gotchas: biquads via lfilter (convolving with b only is WRONG); HPF
  use sosfiltfilt (sosfilt start transient +5 dB); ffprobe csv has a trailing
  comma; ffmpeg can't tag in-place (use metaflac); WAV for segments (flac
  encoder quirk with -ss).
- A 48 kHz source must keep its rate through every stage — once the output was
  relabeled 44.1k (8.8% longer, pitched down) while all loudness metrics looked
  perfect. The duration/samplerate gate exists everywhere for this reason.
- mutagen-based tag copy is multiline-safe (metaflac text export silently
  drops LYRICS continuation lines). copy_tags does NOT copy PICTURE blocks —
  covers are transplanted explicitly.
- Parallel masters need unique STS temp names (master.py uses pid+id).

## Process

1. Copy the album, `flac -t` integrity, note corrupt files.
2. `diagnose.py` on all tracks → JSON (plan triggers).
3. `master.py album` or `restore_batch.py` with chosen policy; `measure.py compare` PNGs.
4. Loudness-matched ABX pair for blind listening; human verdict.
5. Report: before→after table (DR.txt), plots, notes. Graph guide: docs/graphs-guide.md (+ .ru.md).

## Policies

- Transparency-first: default = mimic the reference (TP −0.5, taste EQ, side
  −3 dB). Gentler settings for delicate vocals (nd2 1.5); stronger for crushed
  punch (nd2 2.5–3). Album coherence ±0.5 ND2 across tracks.
- Never restore in place; src/dst separate; quarantine on gate failure.
- Staging is deleted right after a verified delivery (phone/library is the
  archive). The end-of-run sweep wipes the workroot except quarantine/.

## Delivery (MTP/phone)

See docs/mtp-notes.md: fuse mount is read-only, writes via `gio copy -p` with
per-segment URL encoding, verify by reading back (gio cat) + md5, players
cache tags (rescan or move to a new folder to reindex). All of this is baked
into drlib.mtp / drlib.gate / restore_batch.py.
