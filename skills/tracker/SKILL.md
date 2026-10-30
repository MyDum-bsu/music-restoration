---
name: tracker
description: "Work with torrent trackers: search for releases, break down versions, pick the best by quality, and queue downloads with priority — for music, movies, books and any other material. Use when asked to \"find on the tracker\", \"download an album/movie/book\", \"which version is better\", \"compare releases\", \"is there lossless\", \"what quality should I get\", \"queue the download\", or before a restoration or a listening/viewing session when source material is needed. Do NOT use for measuring the quality of already-downloaded files (that's audio-measure) or for restoration (release-restore)."
---

# tracker: searching, comparing and downloading releases

One engine for every kind of material: common mechanics (search → breakdown →
comparison → download), while the quality rules plug in via a profile per
material type.

```bash
S=~/Desktop/projects/music-restoration/scripts/tracker_hunt.py

python3 $S search "Jelusick"                 # list of releases
python3 $S search "Blade Runner 2049" --facts  # + breakdown of each version
python3 $S info 6415449                      # facts on a single topic
python3 $S compare 6499624 6517503 6795598   # comparison with a recommendation
python3 $S fetch 6415449 --dir "Jelusick/Album [TR24]" --priority high
```

## Quality profiles

| type | what gets parsed | what to look at |
|---|---|---|
| music | format, bit depth, sample rate, source (CD/WEB/hi-res/vinyl), image+cue vs tracks, log, scans, published DR | lossless; a digital master beats vinyl; DR from the topic |
| video | resolution, source (BDRemux/BDRip/WEB-DL/HDTV), codec, HDR/DV, audio, subtitles, 3D | source matters more than resolution; lossless audio; Russian subs |
| books | format, scan vs text, OCR | text layer |
| other | generic breakdown | — |

The profile is picked automatically from the forum section and the topic body
(for movies the rutracker title is often **truncated**, so the markers are
also searched in the body: the Russian section labels for translation,
subtitles and runtime).

## Comparison is Pareto, not "one best"

`compare` prints a table with a score and an **explanation**, then the best
overall pick and an alternative. Versions almost always involve a trade-off:
TR24 24/44.1 vs CD 16/44.1, BDRemux 42 GB vs BDRip 1.4 GB. The skill's job is
to show the difference; the decision stays with the user.

## Workflow

1. **Search** (`search`), with `--facts` if needed.
2. **Comparison** (`compare`) — pick a version with an explanation.
3. **Download** (`fetch`) — priority `high`, a separate directory.
4. **Check after download** — `scripts/audio_measure.py <dir>`: from the
   release page you **cannot tell** whether this is real lossless (fake
   hi-res, lossy transcode). This step is mandatory when quality matters.
5. For music, next come `split-flac-album` (if image+cue) and `release-restore`.

## Pitfalls (each cost time — do not relearn)

- **The tracker goes down for maintenance**: the page answers 200, but it is a
  stub saying the forum is temporarily disabled for maintenance until HH:MM
  MSK. Telltale sign: the page is ~1.5 KB instead of tens of KB. Work from a
  local `.torrent` in `~/Downloads`, and do not treat this as an
  authorization failure.
- **`dl.php` comes as a relative link** (`dl.php?t=…`): without the base
  prefix `https://rutracker.org/forum/` requests fails with `MissingSchema`.
- **A private tracker does not report seeder counts in announce**
  (`seederCount = -1`): do not rely on `trackerStats`, take seeders from the
  search results.
- **Never take the size from the topic text** — small numbers sneak in there
  (cover size, pieces of the tracklist). Searching by id gives the exact size.
- **"TR24" means 24 bits at 44.1 kHz**, not a high sample rate. True hi-res
  is verified after download.
- **Vinyl/DSD (LP, SACD) is a different mastering**, not "higher quality":
  noise, rumble, different dynamics. For restoration a digital master is
  preferable.
- **Do not confuse "the alternative only exists as MP3" with "this release is
  MP3"** — phrase it explicitly, otherwise you get a false conclusion about
  quality (a real mistake from a past session).
- **Keep test downloads out of the dataset raw material**: the enrichment
  loop only cleans `dataset/raw/`, but everything put there it will wipe.
- **Never touch the user's own torrents**: filter any bulk operations (pause,
  priority change, removal) by `downloadDir`.
- **Stalled torrents**: 0 peers → remove the torrent and **clear the wanted
  set**, otherwise the torrent keeps downloading into a deleted directory and
  disk space leaks.
- **image+cue downloads**: the track directory and the image must not be kept
  together if `--src <dir>` will later run on it (the image ends up processed
  as a track).
