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

Preferred conservative path for stereo FLAC:

```bash
python3 scripts/restore_safe.py --src "<file-or-track-directory>" --out "<empty-output-directory>"
```

The runner strictly decodes, selects identity/gentle/moderate/adaptive candidates on spread excerpts, evaluates the full track, writes 24-bit PCM FLAC for processed candidates, then decodes and checks that encoded result. It retains the source sample rate. HPF is off unless separately justified; an adaptive variant also tests equal bass expansion. It rejects implicit channel conversion. ALAC/WAV can first be losslessly decoded to FLAC with the native rate/channels preserved; keep the original file and provenance. MP3/AAC inputs may be decoded for processing when that is the chosen source, but remain **lossy-derived**.

Inspect `report.json` for EVERY input:
- `processed_candidate`: full-track safeguards and encoded QC passed; perceptual quality remains unevaluated.
- `kept_original`: source copied byte-for-byte; original defects/headroom may remain.
- `qc_fallback_original`: attempted processing rejected, source copied, violations recorded.
- `error`: no deliverable accepted for this track; command returns nonzero.

Finish only after file count and all statuses agree with the requested tracklist. Retry errors in a new empty directory. Fallback is an explicit outcome, not a successful repair. Keep report.json beside the deliverables. The runner's score is a heuristic; claims about restored sound require listening evidence.

`restore_per_track.py` is the legacy +6.3 dB sweep for reproducibility or an explicitly chosen comparison. It is not the preferred automatic delivery path. It has short-track/reporting limitations; see quality.md. `restore_one.py` is a manual research tool.

For an independent external audit, `qc_gates.py --orig <src-directory> <outputs>` checks peak/clipping/sample jumps. That tool matches by track-number prefix and is not the safe runner's status authority. Use explicitly paired files and matching duration/rate; a copied original can honestly retain a high true peak.

## Dress and deliver

After checking statuses, use `finish_release.py --restored "<output>" --source "<source-directory>" --dest "<collection-destination>"` for tags, covers, lyrics, DR report and spectra. For a single-file input, pass its parent as source. Recheck tags and track counts after dressing. Copy into the collection, preserve source and report. Use add-lyrics/phone-music for their respective stages.

On requested phone delivery, use phone_sync.py or the project's established replace-in-place path. Verify device-side hashes and rescan the player. Replacing an existing song should preserve its exact path where the user wants player history retained.

Report source provenance (lossless/lossy-derived/unverified), processed/kept/fallback/error counts, before/after measurements, encoded QC and listening status. An increase in DR or a smaller distance to another restorer's metrics does not establish superiority to that restorer.
