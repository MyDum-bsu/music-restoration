# Restoration quality: modes, parameters, criteria

## Two contour modes

- **`expand`** (per-band): raises the band level above a smoothed body by a
  factor `r`. Target — Δcrest (default +4.6, from the corpus model when
  available). Gives large DR but touches timbre harder (band delta 1.1–1.5 dB
  RMS) and boosts transients.
- **`flatten`** (per-band, slow modulation around an anchor, `k<1`): quiet
  sections go up, loud ones down, then TP normalization. On crushed masters it
  gives **larger DR with less timbre damage** — depth is limited by
  `--gain-cap` (dB). Verified on a heavily crushed modern album (DR 5–7):
  `flatten k=0.5 cap=8` reached DR 10 at a band delta of 0.92 dB versus
  DR 8 / 1.09 dB for `expand`.

## Why per-track instead of one setting for the album

Tracks inside an album differ wildly (quiet ballad vs crushed rocker), and
**excerpt metrics do not always predict the full track**: a track with wide
sections can land its full-track Δcrest outside the window. That is why
`restore_per_track.py` measures the full result after the winner runs and, if
Δcrest leaves the 5.0–7.5 dB window, retries neighboring caps/targets on the
FULL track and keeps the best by score. Score: |Δcrest − 6.3| + 0.8·(timbre
error) + penalties for under- (<5 dB), over- (>7.5 dB) expansion and DR>12.

## Target numbers

- Δcrest ≈ +6.3 dB — "slightly above reference masters" (their typical shift
  is +4.4…+6). More risks unnatural dynamics.
- DR ≤ 12 — higher only on explicit request.
- Timbre (band delta, RMS) — the lower the more transparent; ≤1.4 dB is good.
- True contour gain step ≤ 8 dB/frame (checked against the gain matrix).
- TP ≤ −0.5 dBFS, clipping 0, sample spikes 0 — mandatory gates (`qc_gates.py`).

## Fake hi-res

If the spectral cutoff is < ~20 kHz inside a 96 kHz container — it is an
upsample with no ultrasonic content. Keep the source container (no
resampling): the file stays compatible and the before/after comparison stays
honest. Downsample only on an explicit request (16/44.1 for a phone/car).

## Listening comparison

The restoration is 4–6 LU quieter than the original. Without loudness
matching the louder original always wins a blind test — match in the player
(ReplayGain/loudness) or build a pack with `scripts/listen_pack.py
--ours-dir ...` (variants are LUFS-aligned, blind A/B/C layout).

## Visual report (like masters post on the tracker)

`scripts/dr_report.py` writes the classic DR-meter text table (`Analyzed:`,
per-track DR/Peak/RMS/duration rows, `Official DR value` computed by pooling
blocks of all tracks, exactly like the meter) — the same format parsed from
master topics by `rnext.rttopic.parse_dr_table`, ready to paste into a release
topic.

`scripts/spectrum_report.py --orig <src> --restored <rest> --out <dir>` adds
the pictures masters attach: per-track PNG pages (waveform envelope, original
vs restored spectrograms on one absolute dB scale, normalized average
spectrum, short-term crest over time) and an animated GIF alternating the two
spectrograms. Default selection: first, middle and last track; `--tracks`
overrides. Style presets: `--style espresso` (default; black & gold, Source
Serif Pro + JetBrains Mono), `cognac`, `noir`. Track numbers carry no leading
zeros ("9. Creatures in Heaven").

## Useful artifacts

- `per_track_report.json` — every candidate and metric per track.
- `PER_TRACK_REPORT.md` — album summary.
- `restore_report.jsonl` (from `restore_one.py`) — before/after per track.
- `_measure.json` (from `audio_measure.py --json`) — independent DR/LUFS/TP.
- `DR Report.txt` — master-style DR-meter table.
- `spectra/*.png`, `spectra/*.gif` — visual comparison pages.
