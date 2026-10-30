# Restoration Lab

A local web view over the restoration algorithm: what it is trying to do, how close
it gets, and what the result actually looks like next to the reference.

```bash
cd ~/Desktop/projects/music-restoration
python3 lab/server.py          # → http://127.0.0.1:8765
```

Needs the corpus drive mounted at `/media/mydum/Data/Music/restore`. Nothing is
written to the drive or the repository — spectrogram cache and rendered test
excerpts live in `~/.cache/restore-lab/`.

## Pages

| page | purpose |
|---|---|
| **Overview** | the one number that matters (target accuracy vs. no processing), corpus composition, artifact control |
| **Dataset** | all album pairs in one sortable table; entry point to any album |
| **Models** | accuracy of each candidate model, both holdout schemes, full leaderboard |
| **Pipeline** | the six-stage chain, the gates, live policy defaults |
| **Album** (from Dataset) | metrics, excerpt spectrograms, blind A/B playback, and a live run of the pipeline |

## What the numbers are

`Target accuracy` is the median absolute error, in dB, when predicting how much
crest factor a mastering engineer restored on an album the model never saw. It is
compared against the same measurement with no processing at all, which is the
honest baseline — a model that cannot beat "do nothing" is worthless.

Two holdout schemes are reported. *By album* tests generalisation to new records.
*By engineer* holds out one of the two restoration engineers entirely, so a good
score there means the model learned the task rather than one person's habits.

## Notes on honesty

- The blind audio check runs on a deliberately small sample (one album, five
  excerpts). It is shown as indicative, not as a verdict.
- Each excerpt is 10–12 s of a track, so excerpt metrics are not album metrics.
  Album-level numbers come from the full-track feature set.
- The three spectrograms in an album view are drawn on one shared absolute dB
  scale. If they were auto-normalised per image, the panels would look different
  for reasons that have nothing to do with the audio.
- `sparse` material is routed to the class table instead of the model. That is an
  intentional out-of-distribution guard, and the page says so where it applies.

## Structure

```
lab/
  server.py          # HTTP API, STFT spectrograms, live pipeline runs
  static/index.html  # shell
  static/app.js      # routing, charts (plain canvas), rendering
  static/style.css   # typography-first light theme
```
