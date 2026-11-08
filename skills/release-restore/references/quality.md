# Processing and evaluation

## What the conservative runner decides

`restore_safe.py` uses full-source TT DR/RMS to assign a working class: short/silent and DR≥11 retain identity; sparse uses a small target, mid and crushed use stronger class priors. These are heuristics (0/0.8/2.3/5.3 dB), not desired DR numbers or demonstrated perceptual optima. High-DR material with an identified defect needs a separate, explicitly evaluated repair.

Three 15-second windows spread over the track compare identity, gentle/moderate flatten and adaptive expand. Band shape is centered to remove uniform gain. The objective uses block-based TT DR change, not a single global peak; targets are provisional. The full output must preserve length and finite PCM, true peak ≤−1 dBFS (0.05 dB numerical tolerance), no clipped output or additional large sample jumps, band-shape RMS ≤1.5 dB, TT DR loss ≤0.25 dB and boundary peak growth ≤3 dB at matched RMS (first/last 50 ms). It must improve its full-track proxy score over identity. These thresholds are conservative engineering safeguards, not validated audibility thresholds. Encoded FLAC is decoded and rechecked. No automatic upward peak normalization or learned EQ curve is applied.

Shape comparison uses source bands within 60 dB of the strongest band and centers the gain difference over those active bands. This excludes effectively empty bands where a change of dither/noise floor would otherwise look like a large timbre change. This engineering mask also needs validation on quiet, narrow-band and unusual recordings.

The score is a selection aid. The copied-original statuses intentionally preserve source bits and can retain defects. Read violations and listening_verdict rather than inferring a repaired file from the FLAC extension. Source codec provenance must be tracked outside the processing decision.

For detected hard clipping the runner now uses the separate [defect-specific repair branch](declip.md). The historical integer AR routine is excluded from new dynamics candidates; repairing only full-scale int16 samples and clamping reconstructed peaks is insufficient for this task. Its old sine test was not evidence of good performance on complex music.

## Legacy comparison

`restore_per_track.py` sweeps flatten/expand toward a fixed +6.3 dB crest lift and a 5…7.5 dB window, retries full-track neighbors and writes reports. Its flatten policies disable diagnosis. It has no identity candidate. The advertised baseline is reported after selection, not used as an enforced nonregression constraint. Its gain_check uses a hard-coded start of 30 seconds and can fail on short inputs. Its Markdown summary does not replace actual QC.

Keep it for historical reproduction and comparison; record the commit, input hashes, CLI and any benchmark-only adaptation (e.g. changing the hard-coded probe start for a short excerpt). Do not call a patched baseline an untouched historical run.

## Measure different questions separately

- Integrity: successful strict decode; exact frames/channels/sample rate; finite PCM; hash and provenance.
- Output constraints: true peak, clipping, introduced discontinuities, stereo behavior, quiet tails and track boundaries.
- Processing effect: gain-independent band-shape change, crest/DR distribution, transient and time-varying gain behavior. DR/crest are correlated and do not independently vote for quality.
- Known-clean tests: apply a documented synthetic clipping/compression operator to a clean source; compare reconstruction error with gain/time aligned. Retain an untouched clean control and identity baseline.
- Real recordings: original, candidate and a named restorer's version, using identical passages and loudness. A restorer is a preference reference, not known ground truth.

Low high-frequency content cannot prove lossy encoding or fake hi-res. Mark spectral findings as suspected and inspect provenance. Preserve native rate by default; a storage conversion requires its own rationale.

## Listening protocol

`scripts/listen_pack.py` builds existing-dataset packs. Its build_item helper can package explicitly paired local excerpts. It chooses ONE peak-safe achievable loudness for all variants, applies only scalar gain, then measures the encoded files (tolerance 0.2 LU). A silent/unmeasurable variant cannot support a valid loudness comparison.

Use identical passage boundaries; check alignment, speed, sample rate and channels before making waveform comparisons. Randomized A/B labels are blind during playback but pack.json contains the answer map for the player; an evaluator must not inspect that map before voting. Strip metadata that reveals identity when exporting stimuli.

ABX tests audibility, not preference. Ask preference and artifact questions separately, randomize order, include repeats/hidden controls and save ratings. Use multiple excerpts, albums, genres and listeners. For formal small-impairment evaluations consult [ITU-R BS.1116](https://www.itu.int/rec/R-REC-BS.1116-3-201502-I/en). An informal A/B pack is not automatically a standards-conformant test.

Promote a sound-changing algorithm only after a frozen evaluation set shows benefit against both identity and the strongest prior pipeline, with uncertainty and failure cases reported. Split by recording/album/artist and author; eliminate duplicates and training leakage. Five handpicked regressions establish specific fixes, not broad superiority. Keep objective safeguards separate from subjective preference.

## Outputs

- report.json from restore_safe: source hash/format, policy, window scores, full and encoded metrics, fallback/error reasons, listening status.
- Legacy per_track_report.json: candidates and measurements from the old sweep.
- DR Report.txt and spectra: generated by finish_release; useful descriptions, not proof of superiority.
- Listening pack plus ratings: retained with the experiment manifest, code version and exact excerpts.
