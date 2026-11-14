---
name: release-restore
description: Restore musical tracks or albums from local files or discovered releases, with source provenance, conservative candidate selection, encoded QC, metadata and collection delivery.
---

# Restore a release

Work from the music-restoration repository root. Read [quality.md](references/quality.md) when choosing or evaluating processing; read [sources.md](references/sources.md) when obtaining audio or replacing lossy sources.

## Source and workspace

Run `python3 scripts/doctor.py`. A failed tracker check affects network sourcing; use local inputs when their required decoder/encoder dependencies work. Resolve paths with `rnext.workspace`; inspect the mounted filesystem and free space before a batch. Use a dedicated experiment/output directory and reserve space for the largest decoded track and temporary output. Keep originals and provenance. Local music conventions and a task's explicit disk authorization override historic 40 GiB/flash assumptions.

For a name/link without local audio, search via tracker and the additional catalogs in sources.md. Completion means the correct recording/version is downloaded and verified, not merely queued. For image+cue, use split-flac-album and keep the image outside the track directory. Record source URL/topic, edition, codec, hash and any lossy ancestry.

`python3 scripts/audio_measure.py "<source>"` gives diagnostic DR/LUFS/TP and spectral hints. A quiet or high-DR track can still contain damage. Prefer keeping it unchanged unless a particular defect justifies processing. A spectral cutoff alone cannot prove lossy ancestry or fake hi-res.

## Process and verify

Preferred conservative path for mono/stereo FLAC:

```bash
python3 scripts/restore_safe.py --src "<file-or-track-directory>" --out "<empty-output-directory>"
```

Auto mode first checks for repeated extreme plateaus, including a small cluster of quantization/dither levels. For supported hard clipping it uses a float AR proposal only when reliable-context prediction and forward/backward agreement are strong; otherwise it uses consistent sparse reconstruction. Reliable (unmasked) samples stay fixed before a documented common headroom gain. Read [declip.md](references/declip.md) for this branch, its measured evidence, resource use and limitations.

Without that pattern, auto mode retains quiet/sparse and high-DR material; denser material gets identity/gentle/moderate/adaptive dynamics selection using fully processed tracks and full-result QC. Spread windows diagnose those same full-context outputs; isolated-cut processing is not a veto. `--mode declip` requests repair only (no supported pattern → keep original); `--mode dynamics` explicitly allows dynamics candidates for sparse material too. `--clip-threshold <PCM-level>` is an explicit symmetric mask override, not an automatic estimate or quality control; use it only with evidence of that clipping level.

The runner strictly decodes, writes 24-bit PCM FLAC for processed candidates, then decodes and checks that encoded result. It retains the source sample rate. HPF is off unless separately justified; an adaptive dynamics variant also tests equal bass expansion. The legacy integer declipper is excluded from these dynamics candidates. The runner rejects implicit channel conversion. ALAC/WAV can first be losslessly decoded to FLAC with the native rate/channels preserved; keep the original file and provenance. MP3/AAC inputs may be decoded for processing when that is the chosen source, but remain **lossy-derived**.

Inspect `report.json` for EVERY input:
- `processed_candidate`: full-track safeguards and encoded QC passed; perceptual quality remains unevaluated.
- `kept_original`: source copied byte-for-byte; original defects/headroom may remain.
- `qc_fallback_original`: attempted processing rejected, source copied, violations recorded.
- `error`: no deliverable accepted for this track; command returns nonzero.

For dynamics, `restoration_state: no_acceptable_candidate` and `needs_review: true` mean the current search did not resolve the enhancement request. Do not report such a retained dense source as successfully restored or infer that processing is unnecessary. Consider another supported method/source within the task, or report the unresolved case. `processing_need` is not a calibrated classifier verdict. Window warnings and perceptual validation remain distinct from full-file technical acceptance.

For declipping, inspect `clip_diagnosis`, chosen method, raw sample consistency, `repair.output_gain_db` and encoded-waveform checks. The dynamics score is descriptive for this branch: genuine repair need not reach a DR target or preserve the distorted source's spectrum. Detection identifies a plateau pattern; artistic intent cannot be inferred from PCM alone.

Finish only after file count and all statuses agree with the requested tracklist. Retry errors in a new empty directory. Fallback is an explicit outcome, not a successful repair. Keep report.json beside the deliverables. The runner's score is a heuristic; claims about restored sound require listening evidence.

`restore_per_track.py` is the legacy +6.3 dB sweep for reproducibility or an explicitly chosen comparison. It is not the preferred automatic delivery path. It has short-track/reporting limitations; see quality.md. `restore_one.py` is a manual research tool.

For an independent external audit, `qc_gates.py --orig <src-directory> <outputs>` checks peak/clipping/sample jumps. That tool matches by track-number prefix and is not the safe runner's status authority. Use explicitly paired files and matching duration/rate; a copied original can honestly retain a high true peak.

## Evaluate against the source

After checking statuses, one command answers "did this album turn out well?":

```bash
python3 scripts/evaluate_album.py --src "<source-directory>" --restored "<output>" --out "<eval-dir>"
```

It measures both sides with the shared instrument (`audio_measure.py` plus the common spike/band kernel), matches tracks by number then normalized name (a bonus track missing on one side is reported as skipped, never dropped), lifts `report.json` statuses into the verdicts, and writes `tracks.csv`, `verdicts.csv`, `summary.json` and `REPORT.md`. The ladder is fixed before the data: our output's true-peak ceiling first, then a DR gap of 0.3, then click-like sample jumps, then a tie; LUFS never votes, it only marks loudness comparability. `REPORT.md` names the systematic shifts (loudness, tonal band shift) and states its own limitations. Metrics are engineering proxies — they prove output constraints and shifts, not listening preference; see [quality.md](references/quality.md) for what each level of evidence can prove. The full evaluation runbook lives in [docs/evaluation.md](../../docs/evaluation.md).

## Dress and deliver

After checking statuses, use `finish_release.py --restored "<output>" --source "<source-directory>" --dest "<collection-destination>"` for tags, covers, lyrics, DR report and spectra. For a single-file input, pass its parent as source. Recheck tags and track counts after dressing. Copy into the collection, preserve source and report. Use add-lyrics/phone-music for their respective stages.

On requested phone delivery, use phone_sync.py or the project's established replace-in-place path. Verify device-side hashes and rescan the player. Replacing an existing song should preserve its exact path where the user wants player history retained.

Report source provenance (lossless/lossy-derived/unverified), processed/kept/fallback/error counts, before/after measurements, encoded QC and listening status. An increase in DR or a smaller distance to another restorer's metrics does not establish superiority to that restorer.
