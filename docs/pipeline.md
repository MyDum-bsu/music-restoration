# The restoration pipeline

Goal: undo loudness-war mastering — restore the dynamics the mastering
engineer intended — with *transparency* as the success criterion, not big DR
numbers. Restoration is not a panacea for a bad recording; it reduces listening
fatigue. Stop when metrics, artifacts, or blind ABX say quality degrades.

## Chain (master.py v2)

```
DC fix → [adeclip if diagnosed] → [HPF 25 Hz zero-phase]
→ reference EQ → CONTOUR EXPANDER → [ND2 punch] → center focus
→ TP normalize −0.5 dBFS → TPDF dither → FLAC + tags + ReplayGain
```

Why each stage exists:

- **DC fix** — some masters carry a DC offset that eats headroom.
- **adeclip** — only when diagnosis shows real flat-top clipping (>= ~20 flat
  runs or >= 100 clipped samples). ffmpeg `adeclip` rebuilds the rounded tops.
  Cosmetic declipping of a true-peak-limited master (0 clips) does nothing.
- **HPF 25 Hz** — 4th-order zero-phase (sosfiltfilt). Subsonic rumble wastes
  limiter headroom and excites speakers. Zero-phase matters: sosfilt leaves a
  +5 dB start transient.
- **Reference EQ** — a gentle "taste" curve reverse-engineered from a
  professional restoration: low shelf +0.9 dB <150 Hz, wide −1.6 dB bell @315,
  air +0.45 dB >9k. Disable with `--no-reference-eq` for your own taste.
- **Contour expander** — the core. Measures a slow (3 s window, smoothed)
  RMS envelope and applies gain = (slope−1)·env + offset with a slew limit
  (40 dB/s). Quiet parts get lifted relative to loud parts → macro dynamics
  return. Transients keep their crest because the gain moves slowly.
- **ND2 (Stereo Tool Natural Dynamics 2)** — per-band punch restoration: a
  separate mechanism from the contour. Key insight measured on reference
  material: contour gain alone does NOT raise TT DR (peaks ride the same
  gain). Strength 1.5–2.0 for delicate vocals, 2.5 for crushed masters,
  bass bands scaled ×0.35 so the punch doesn't boom.
- **Center focus** — side −3 dB (as in the reference) tightens the stereo
  image that loudness processing smears. Skip with `--side-db 0`.
- **TP normalize + TPDF dither** — true-peak −0.5 dBFS headroom, 16-bit
  output with proper dither. ReplayGain tags make playback loudness equal.

Defaults reproduce the reference: slope 1.08, offset −4.6, env 3 s, slew
40 dB/s, side −3, TP −0.5, nd2 2.5 (bass ×0.35), 16-bit.

## Choosing parameters

- `diagnose.py` first: it reports clipping, subsonic energy, crest percentiles
  (punch health), macro envelope std, and proposes a plan.
- Crest median < ~6 and strong macro compression → nd2 2.5; healthy crest
  (9–12) → nd2 0.5–1.5 or none. Delicate vocals → 1.5.
- Keep album coherence: ±0.5 ND2 between tracks, one policy for the album.
- LUFS will drop (typically −9 → −13). That is the point; ReplayGain handles
  playback level. Do not "fix" it with a limiter.

## Verification

- `measure.py metrics` — TT DR (dr14-compatible), LUFS/LRA/TP, clip counts.
- `measure.py compare` — before/after PNG: RMS envelope, crest factor,
  1/3-octave spectrum. Spectrum delta should stay within ±1 dB and monotone.
- `acceptance.py --orig --mine [--ref]` — the three-way table against a
  commercial reference restoration if you have one.
- **Duration gate**: output duration must equal input (±50 ms) at the same
  sample rate. A 48 kHz source once passed every loudness check while being
  8.8% longer and pitched down — the gate exists because ears caught it.
- Blind ABX: loudness-matched pair, human verdict. Metrics lead, ears decide.

## Delivery (restore_batch.py)

One command per album: pull → restore → covers → tags (MusicBrainz optional)
→ lyrics (optional) → `DR.txt` + report → hard gate → push + per-file md5
verify → staging cleanup. Failures quarantine the album loudly. Never restore
in place; staging dies right after a verified delivery.

## Free-software stack

Everything is ffmpeg + numpy/scipy + Stereo Tool (free CLI, non-commercial;
only its free features are used — the unlicensed binary mangles paid ones).
TT DR is reimplemented from the dr14 meter spec. The MasVis-style comparison
plots ship in `measure.py`.
