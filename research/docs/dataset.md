# Dataset of "original → restoration" pairs (two authors)

This document describes how the training set for the `music-restoration`
pipeline is assembled: why it exists, what it consists of, where it is stored,
and how the DSP core improves on it.

## 1. Two authors are two different restorers

Contrary to the first impression (both catalogs are tagged `[RM]`), these are
**two independent jobs by two different people**:

| | pid 5618698 ("Lisnoy Yozhyg") | pid 3048819 ("Alias Divas") |
|---|---|---|
| topics in catalog | 178 | 500 |
| genre niche | classic rock, hard rock, progressive | heavy genres: death/groove/industrial/metalcore |
| topic meta fields | "Year of release", "Audio codec" | "Edition year", "Audio codec" |
| rip type | usually `image+.cue` | `tracks+.cue` |
| published analysis | textual DR-meter table (62 topics / 515 releases) | less often as text: 18 topics / 122 releases, more often MasVis screenshots |
| artist overlap | — | 3 shared artists out of 176 and 499 (Evanescence, Fear Factory, Leaves' Eyes) |

Different meta fields, different packaging, different genre niches, almost
disjoint catalogs — two authors with their own taste and tooling. This means:

- stratification by "author" **makes sense** (these are different objective
  functions);
- hold-out evaluation must keep 30% of albums of **each** author out of
  tuning, otherwise "universality" is never tested.

A practical consequence for collection: for "Lisnoy Yozhyg" the target DR is
often known in advance from the textual table in the topic; for "Alias Divas"
the master release has to be downloaded to measure the target (and it is needed
for the band deltas anyway).

## 2. What a pair consists of

```
original (untouched edition, CD 16/44)
    └─► features (DR/crest/RMS/peak/clip/30 1/3-oct bands)   ─┐
                                                             ├─► deltas = training row
master restoration [RM] of the same album                     ─┘
    └─► features, excerpts (for listening tests)
```

The pair key is `{restored_topic_id}/rel{idx}`. Mandatory conditions:
the same album and edition (matching catalog number from "Original edition"),
lossless source, not vinyl/SACD/hi-res (a different mastering), and not a
bootleg/live recording standing in for the studio album.

## 3. Assembly pipeline

What is already free (no master-release download): `scripts/master_targets.py`
collects `corpus/master_targets.jsonl` from the topics — **637 releases /
8104 tracks** with a published master analysis (DR/Peak/RMS per track,
crest_tt = peak − rms):

- master crest_tt median 12.60 (sd 1.58, range 8.7…29.7);
- official DR of a release: median 11, range 8…13;
- by author: Lisnoy Yozhyg 515 releases (crest_tt median 12.87), Alias Divas 122 (12.24);
- decades: 70s 20, 80s 70, 90s 87, 2000s 130, 2010s 201, 2020s 127.

That is the "before measuring the source" target; deltas appear once pairs are
assembled.

```
scripts/dataset_plan.py topics   # master topic pages → corpus/restored/{tid}.json
                                 #   releases[] + textual DR table of the master (if present)
scripts/dataset_plan.py match    # candidate pool per artist (1 search instead of 1 per release)
                                 #   → corpus/dataset_plan.jsonl  (matched/weak/no_original)
scripts/dataset_plan.py report   # summary: volume, decades, master DR, share of discographies
scripts/dataset_fetch.py --dry-run   # exactly what will be downloaded (file selection)
scripts/dataset_fetch.py --go        # Transmission: file selection + start
```

Matching details:
- the candidate pool comes from "Source for processing" in the master topic,
  otherwise from the artist name; for misses — a targeted "artist + album"
  search;
- scoring: match of the album part of the title (after the first " - "), year,
  catalog number; penalties for vinyl/DSD/hires/bootleg/live substitution/MP3;
- **the artist's CD discography** is a legitimate candidate
  (`matched_graphy`): the needed disc is picked later from the torrent's file
  list. This is often the only CD source: for Accept, for example, almost all
  albums on rutracker exist only as LP, while the CD editions sit in the
  "Discography (114 CD)" topic;
- the final check is the download itself: matching files against the master
  track list (track number + title tokens), threshold 60%; below it the pair
  is dropped.

## 4. Discography as a source, and "download only what is needed"

Transmission RPC (`rnext/transm.py`) makes it possible to add a torrent
paused, get its file list, unselect the unneeded files (`files-unwanted`), and
only then start. That way, from a discography topic **only the needed
disc/album** is downloaded, while lives, bonuses and scans are not downloaded
at all:

```
torrent-add(paused) → torrent-get(files) → match_tracks() → torrent-set(files-unwanted) → torrent-start
```

Two file-selection modes:
- `tracks` — per-file matching against the master track list (ordinary topics);
- `image` — the image + cue are taken whole and split by the cue after download.

**One topic — several pairs.** A discography topic serves many pairs at once,
while a torrent has a single wanted set. Therefore the set is computed AS THE
UNION of the needed files across all pairs of that topic
(`dataset_fetch.py --repair`); otherwise the second pair would unselect the
files of the first. Same when re-adding: `torrent-duplicate` → union with what
is already selected.

Sanity check on a live torrent: the Anneke van Giersbergen master torrent
(214 files) yielded a wanted set of 25 out of 214 (two albums), Kylie Minogue
(360 files) — 28 out of 360 (two albums) — only what is needed gets
downloaded.

## 5. Storage (hard limit 40 GiB)

```
/media/mydum/Data/Music/restore/dataset/
  plan/…                  not stored: the plan lives in corpus/dataset_plan.jsonl
  raw/<key>/              RAW, transient: download → extract → delete
  pairs/<key>/meta.json   ids, tracks, matched files, master DR, QC flags
  pairs/<key>/features.jsonl   per-track features of both sides (~1.5 KB/track)
  pairs/<key>/excerpts/        8–30 s excerpts (loudest spot + middle), both sides
  features/all.jsonl      merged training table
  state.jsonl             download journal (append-only)
```

Three storage tiers and their cost:

| tier | per track (both sides) | per pair (~12 tracks) | vs raw | fits in 40 GiB |
|---|---|---|---|---|
| raw FLAC (transit) | ~70 MB | ~0.8–1.0 GB | ×1 | ~30 pairs at a time |
| 8–30 s excerpts | ~3 MB | ~36 MB | ×23 | ~900 pairs |
| features | ~1.5 KB | ~18 KB | ×47,000 | practically unlimited |

Policy: **features forever, excerpts forever (they are needed for listening
validation and residual tests), raw only while the task is active.** Rotation
is enforced by `rnext/workspace.py` (40 GiB budget, `remaining_bytes()` before
every download).

## 6. How the pipeline improves on this

1. **`target_dcrest` as a function of the material.** Today it is the constant
   4.6 plus class rules. The dataset will be used to build a model
   `Δcrest ≈ f(class, source DR/crest/RMS, genre, era, author)` — first a
   table of per-stratum means, then a low-dimensional regression with a random
   album effect (important: the ~0.3 dB within-album spread is not explained
   by scalars).
2. **Blind predictor (the main open TODO).** 500+ pairs give ~6000 tracks with
   a known target — enough for the target to stop being a constant.
3. **Spectral matching.** Band deltas over the corpus give the author's
   average "warm" tilt; once enough pairs are collected we can decide whether
   to introduce an EQ layer (currently out of scope for v1) and how to
   parameterize it by genre/era.
4. **Acceptance thresholds and QC gates** are calibrated from distributions
   (not from 3 albums), including the residual test and ABX sets.
5. **Hold-out evaluation**: 30% of albums of each author are held out; a
   baseline of four systems (original, loudness-matched original, our
   pipeline, master), metrics as per-track distributions + blind listening.

## 7. Open questions

- **Volume**: for ±0.15 dB on the album target we need ~20–30 albums per
  stratum; for the per-track predictor — 300–600 tracks (we currently have 37).
- Declipping is not calibrated on genuinely clipped material (1990s era; LP
  rips are out — then it is CD editions with clipping).
- The warm EQ tint of the master was measured on a single album — a band
  dataset is needed.
- A second *independent* restorer is confirmed (Alias Divas), but his niche
  is skewed toward heavy genres; for "classic" coverage there is only Lisnoy
  Yozhyg.
