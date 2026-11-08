---
name: audio-measure
description: Measure audio quality and authenticity — DR (TT convention), LUFS, true peak, clipping, bit depth/sample rate, fake hi-res and lossy-transcoded-to-FLAC. Use when asked to "measure DR", "what's the quality of this file/album", "is this real lossless?", "is it an upconvert?", "compare versions by quality", "check loudness/peaks", or as a step before picking a release version and before restoration. Do NOT use for finding or downloading releases (that's tracker) or for running the restoration itself (release-restore).
---

# audio-measure: audio quality and authenticity

One script, one answer to the question "what is this file, really".

```bash
python3 scripts/audio_measure.py <file|dir> [...]
python3 scripts/audio_measure.py --json out.json <dir>
python3 scripts/audio_measure.py --quick <dir>   # no spectrum/LUFS
```

Prints a table: format (sample rate/bit depth), DR, crest, LUFS, TP, spectral
cutoff and a verdict. `--json` adds machine-readable output (all fields,
including `hf_floor_db`).

## What the fields mean

| field | meaning |
|---|---|
| `dr` | DR in the **TT convention** (peak − RMS per blocks), the same metric as in the restoration corpus. Do not confuse with foobar DR: the numbers are close but not identical |
| `crest_tt` | peak − RMS over the whole file |
| `lufs_i` | integrated loudness (EBU R128) — "how loud" |
| `tp_dbfs` | true peak: **above 0 dBFS means intersample peaks**, common for crushed masterings |
| `cutoff_hz` | the last frequency where the spectrum is still within 55 dB of the peak |
| `hf_floor_db` | median of the spectrum **above** the cutoff relative to the 1–10 kHz band. This is the "wall" detector |
| `clipped_pct` | share of samples at full scale |
| `fake_hires` | claims >44.1 kHz, but there is no content above 22 kHz |
| `lossy_suspect` | cutoff below 19 kHz **and** floor above the cutoff < −75 dB (a real "wall") |

## How to read it (and what not to do)

- **`lossy_suspect` is a flag to verify, not a verdict.** A dark or heavily
  limited master also gives a low cutoff, but no "wall" (the floor above the
  cutoff stays at −30…−40 dB). Even a sharp wall can originate in filtering or
  mastering; the floor does not rule out false positives. Verify acquisition
  provenance and the release edition. A control transcode demonstrates one
  detectable case, not general authenticity detection.
- **DR ≤ 6 — "heavily crushed"**, 7–9 — dense, ≥11 — healthy. This is not a
  "bad/good" judgement. Use it as one clue alongside specific audible defects.
- **TP > 0 dBFS does not mean the file clips**: these are intersample peaks,
  potentially relevant to playback headroom. TP alone cannot establish audible
  harshness. The conservative runner uses a −1 dBFS ceiling on processed outputs.
- **`--quick` skips the spectrum** — for large libraries; DR and
  clipping are always computed.
- DR and spectrum use only the first 90 seconds. This can miss later defects
  and is not full-track acceptance. restore_safe performs full decode/QC.
- `fake_hires` is a bandwidth suspicion flag, not proof of an upsample. Low
  ultrasonic content alone cannot establish the history of a recording.

## Pitfalls

- **Bit depth is taken from ffprobe's `bps`, not from `bits`** — otherwise the output shows `?`.
- **LUFS/TP are parsed from the LAST summary block of ffmpeg**: `Integrated loudness`
  and `True peak` appear in different places in the output, and `Peak:` occurs several times.
- Comparing versions by DR only makes sense on the **same material**:
  different masterings of one album give different DR at the same loudness.
