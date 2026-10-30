---
name: release-restore
description: Full agent cycle for music restoration — from "restore this" with no file on disk to a finished album in the collection: finds the release, downloads the best quality, measures, restores track-by-track, applies tags/cover/lyrics, generates a master-style DR report with spectra, and reports. Use when the user says "restore", "raise the DR", "save this crushed master", "make a release", "prepare the album for listening", gives a path/link/album name to restore, or asks to take downloaded music all the way to the collection. Do NOT use for measurements without restoration (audio-measure), searching/downloading without restoring (tracker), phone library care (phone-music), or project status (restore-status).
---

# release-restore: "restore this" — from search to collection

One scenario covers everything: the user may not have the file — then the
source is found and downloaded. If the file exists, the search step is skipped.

```bash
S=~/Desktop/projects/music-restoration/scripts
R=/media/mydum/Data/Music/restore          # work flash drive (config/workspace.json)
ALBUM="<Artist> - <Album> (<year>)"        # release name
```

## Steps

1. **Source.** A local path is given → that is the source. A link/name/"this"
   without a file → the **tracker** skill: `search` → `compare` → `fetch` with
   priority high into `$R/release_src/<ALBUM>`. Selection rule — lossless,
   digital master over vinyl, published DR in the topic, image+cue is fine.
   Wait for the download to reach 100% (poll RPC every 30–60 s), then verify
   the files with `audio_measure.py`.
2. **Verify the source.** `python3 $S/audio_measure.py "<source>"` — DR, LUFS,
   TP, clipping, fake hi-res, lossy transcode. If TT DR ≥ 11 (healthy), no
   restoration is needed: say so and stop (or only do the dressing-up step if
   asked).
3. **Split.** image+cue → the **split-flac-album** skill; move the image into
   `_image/`, otherwise it goes into processing as a "single track".
4. **Restore (per-track by default):**

```bash
python3 $S/restore_per_track.py --src "<source>" \
    --out "$R/release/$ALBUM [restored per-track]" \
    [--baseline "<previous version, if any>"]
```

   The script sweeps `flatten`/`expand` candidates on an excerpt of every
   track, scores them by Δcrest/timbre, runs the winner on the FULL track,
   catches a Δcrest outside the 5.0–7.5 dB window and re-processes the track
   with neighboring parameters, then writes `per_track_report.json` and
   `PER_TRACK_REPORT.md`. One track or a quick pass — `restore_one.py` (see
   references/quality.md). If a corpus/models are present,
   `restore_one.py --ml-snapshot auto` refines the target; without them the
   class table works.
5. **QC.** `python3 $S/qc_gates.py --orig "<source>" "$OUT"/*.flac` — TP,
   clipping, sample spikes (12/12 OK is the norm). Then `audio_measure.py` on
   the result: DR up, LUFS down (expected), no clipping.
6. **Dressing.** `python3 $S/finish_release.py --restored "$OUT" --source
   "<source>" --dest "$HOME/Music/$ALBUM [restored per-track]"` — tags, cover,
   lyrics, collection copy. It finishes with `DR Report.txt` — the DR-meter
   table in the same format masters post in their release topics (parsed by
   `rnext.rttopic.parse_dr_table`), plus `spectra/` — per-track PNG pages
   (waveform, spectrograms, average spectrum, crest over time) and animated
   GIFs alternating original/restored.
7. **Delivery.** Phone connected and requested → `python3 $S/phone_sync.py
   --album "<dir>" --rel "<Artist>/<ALBUM>"`. Report to the user: DR/LUFS
   before→after, TP, QC verdict, paths, and a reminder to compare at matched
   loudness.

## Hard rules

- **The flash drive is the work area.** 40 GiB budget, state: `python3
  $S/status_report.py`. Check free space before a big run; if dataset torrents
  are downloading, pause them via the Transmission RPC (filter by `downloadDir`
  inside `$R`) and resume them afterwards.
- **Never touch torrents outside `$R`** — no pausing, no deletion.
- **Transmission RPC down** → start `transmission-gtk`, wait 5 s, re-check with
  `rnext.transm.check()`.
- **Memory:** 96 kHz needs ~2 GB per track — `restore_per_track.py` works
  sequentially, keep `restore_one.py` at `--jobs 1`.
- **`restore_one.py` writes bare FLAC** — tags/cover/lyrics only via
  `finish_release.py` (lyrics via the add-lyrics skill).
- **Copy, never move** into the collection; `$R/release` keeps the working
  version with the reports.
- **Compare at matched loudness**: the restoration is 4–6 LU quieter than the
  original — that is the price of dynamics, not "worse quality".

## Pitfalls

- **Fake hi-res** (cutoff < 0.9×22 kHz at 96 kHz): do not downsample without a
  reason — keep the source container so the A/B stays honest.
- **metaflac drops multiline lyrics** and **adds** PICTURE blocks instead of
  replacing — `finish_release.py` already handles this via mutagen; do not
  work around it by hand.
- **Covers**: cover songs' lyrics are the lyrics of the originals; if the
  original artists are nowhere to be found — ask, do not guess. Instrumentals
  with no lyrics are normal.
- **No front cover among the scans** — a booklet spread is usually the front in
  the right half; crop it and show the user.
- **`/home` disk space** before copying into the collection: `df -h /home`.

Details on modes, parameters and quality criteria — `references/quality.md`.
