---
name: restore-status
description: One-command status report of the music-restoration project (audio restoration) — dataset, pair plan, torrent queue, processes, models and their metrics, plus a deep check (data integrity, DSP tests, end-to-end pipeline run). Use when asked "how are things", "what's the dataset status", "status", "how many tracks", "validate the state", "is there progress", "status report", "check that everything works". Do NOT use for restoring a specific album (that's release-restore).
---

# music-restoration: project status

Project: `~/Desktop/projects/music-restoration` — a DR-restoration pipeline
trained on "original → master restoration" pairs from rutracker (two
restoration authors: Lisnoy Yozhyg `pid=5618698`, Alias Divas `pid=3048819`).
The data lives on a flash drive at `/media/mydum/Data/Music/restore`
(40 GiB budget).

## Quick answer

```bash
python3 ~/Desktop/projects/music-restoration/scripts/status_report.py          # report
python3 ~/Desktop/projects/music-restoration/scripts/status_report.py --deep   # + integrity, tests, run
```

The script prints 6 sections (7 with `--deep`) and does not crash if one of
them is unavailable:

| section | what it shows |
|---|---|
| 1. Flash & budget | mounted or not, used/40 GiB limit, free space on the partition |
| 2. Dataset | pairs / tracks / excerpts, source classes, the `ds-…` version, distance to M1/M2/M3 |
| 3. Plan & journal | matched pairs, master targets without audio, statuses of registered pairs |
| 4. Torrents | downloads, active ones, speed, what is "in progress" and how much of it has live peers |
| 5. Processes | enrichment loop, watchdog, Transmission, player + the last lines of the loop log |
| 6. Models | how many snapshots, which passed the gate, what the runtime will pick, the latest leaderboard |
| 7. Deep (--deep) | data integrity, DSP tests, end-to-end pipeline run with a model |

## What matters about the models (verified by measurements)

- **Runtime model selection**: the freshest dataset version → the best median
  (by-album) → only then a fixed model order. The gate verdict (`promote`) is
  written into the snapshot registry — without it a snapshot is invisible to
  the runtime.
- **hgbq** (quantile HGB q10/q50/q90) is implemented and gives its own
  confidence band next to the conformal one.
- **End-to-end evaluation on excerpts** (`dataset_eval.py`, variants
  `flat,class,model`, flag `--holdout-only`): in-sample the model wins (0.86
  vs 0.99 for the class table), but **on holdout the advantage disappears**
  (0.92 vs 0.79–0.85; the difference is within noise, the CI crosses zero).
  But the pipeline vs "do nothing" wins hugely: −2.5 dB, CI [−3.7, −1.6].
- **No model passes the by-author split** (worse than the class table). The
  reason is not volume: **the whole plan has only 2 restorers** (3164 and 1192
  pairs), and on identical material they diverge by 0.2–0.7 dB. More tracks
  from the same two people will not give cross-author generalization —
  different restorers are needed.
- **Production readiness: beta.** The restoration core (DSP + class table +
  physical invariants) is reliable; the ML model is safe (clamped to ±2.5 dB,
  OOD gate), but its benefit outside the training corpus is not yet proven.

## How to read it

- **Volume levels** are tied to model thresholds: **M1 = 300 tracks**
  (ridge/knn/conformal), **M2 = 760** (hgb/hgbq), **M3 = 1500** (meaningful
  cross-author validation).
- **Promotion gate**: a model enters the runtime only if on the holdout it
  beats the STRONGEST baseline (usually `v0table` — the class table) by
  ≥0.10 dB and the lower bound of the 95 % CI of a paired bootstrap > 0.
  Until that happens, the class table of targets runs in production
  (crushed ≈5.3, mid ≈2.3, healthy 0). That is normal, not a breakage.
- **By-author split** (train on one master, validate on the other) — our test
  for "learned the taste of one corpus". No model passes it yet; that is
  expected and reflected in the report.
- **"In progress" vs "with peers"**: the cap on new batches counts downloads
  that have peers. Dead swarms (0 peers) must not block the pipeline —
  `dataset_prune_stalled.py --peerless` removes them.

## If something is off

| symptom in the report | action |
|---|---|
| flash drive reported as not mounted | `scripts/repair_flash.sh` (finds the "Data" partition, clears the dirty flag, mounts; on refusal — `sudo ntfsfix -d /dev/sda5`) |
| free space below 12 GB | `scripts/dataset_prune_stalled.py --max-progress 0.5` and/or `--peerless` |
| loop/watchdog not running | `scripts/resume.sh` (mounting → Transmission tuning → integrity → loop → training); the watchdog `scripts/watchdog.sh` raises the loop by itself |
| no promoted model | do nothing: keep accumulating tracks, the gate will pass one on its own; can be accelerated with `scripts/ml_auto.py --force` |
| Transmission RPC off | `transmission-gtk` is not running: `scripts/resume.sh` will raise it; tuning — `scripts/transmission_tune.py --apply` |
| many `paused_space` entries in the journal | pairs postponed over space/dead swarms — the selector will return them to work; optionally `scripts/dataset_fetch.py --refetch` |
| many `dead_swarm` entries in the journal | pairs with 0 peers after two attempts are marked terminal — that is normal, they do not occupy batch slots |
| stalled torrents in Transmission | `scripts/dataset_prune_stalled.py --peerless`, then `scripts/dataset_fetch.py --repair` (clears wanted on removed pairs and removes zombie torrents) |

## Useful commands (deeper)

```bash
python3 scripts/dataset_audit.py            # what was downloaded/extracted/lost
python3 scripts/dataset_audit.py --integrity
python3 scripts/dataset_select.py --limit 12   # what goes into the next batch
python3 scripts/ml_auto.py --force             # retrain on the current data
python3 scripts/fit_tilt.py --report           # tonal curve of the corpus
python3 scripts/dataset_eval.py --all --variants flat,class,model   # end-to-end evaluation
python3 scripts/dataset_eval.py --all --holdout-only                # honest (not in-sample)

# restoration and release packaging (details — the release-restore skill)
python3 scripts/restore_one.py --src <dir> --out out/ \
        --auto-policy --adaptive --ml-snapshot auto --tp -0.5
python3 scripts/finish_release.py --restored out/ --source <src> \
        --dest ~/Music/restorations/<name> --lyrics

# finding and picking material on the tracker (details — the tracker skill)
python3 scripts/tracker_hunt.py search "Artist" --facts
python3 scripts/tracker_hunt.py compare <tid> <tid>
python3 scripts/fetch_release.py --topic <tid> --dir "Artist/Album" --priority high

# audio quality and authenticity (details — the audio-measure skill)
python3 scripts/audio_measure.py <dir>
```

## Key paths

```
/media/mydum/Data/Music/restore/
  corpus/dataset_plan.jsonl        pair plan (master release → original)
  corpus/restored/{tid}.json       topic breakdowns of masters (+ master DR tables)
  corpus/master_targets.jsonl      master targets without audio (637 releases / 8104 tracks)
  corpus/tilt_curve.json           fitted tonal curve
  dataset/pairs/<key>/             features (features.json), meta.json, excerpts
  dataset/raw/<key>/{orig,master}  raw material (transit, window of fresh pairs)
  dataset/state.jsonl              download journal (merged by key);
                                   statuses: downloading, paused_space, dead_swarm,
                                   no_match, edition_mismatch, error
  dataset/loop.log                 enrichment loop log
  release/                         packaged restorations (tags, lyrics, cover)
  release_src/                     downloaded release sources (bypasses the dataset raw)

~/Music/restorations/              ready for listening (copy from release/)
~/Desktop/projects/music-restoration/skills/   project skills (audio-measure, tracker,
                                   release-restore, restore-status)

~/Desktop/projects/music-restoration/models/registry.jsonl      model snapshots
~/Desktop/projects/music-restoration/models/leaderboard.jsonl   run results
```
