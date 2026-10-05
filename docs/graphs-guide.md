# Reading the restoration graphs (plain language)

Every comparison ships a PNG (`measure.py compare`) and a metrics table.
What to look for, and what a good result looks like:

## 1. Loudness over time (RMS, top panel)

Red = original, green = restored.

- A loudness-war master is a nearly horizontal pipe near the top: everything
  equally loud all the time.
- After restoration the green line breathes: verses and pauses sit well below
  choruses. That is the returned "air" of the song.
- Good: green follows the shape of red but with wider swings, sitting 3–6 dB
  lower on average (quieter is expected — see ReplayGain below).

## 2. Crest factor (middle panel)

Peak minus RMS per window — the headroom between peak and loudness.

- A crushed master: low crest (5–8 dB) — a "wall" of sound, no hits.
- After restoration crest grows (10–14 dB) — kick and snare cut through
  again: the punch is back.
- Good: green above red almost everywhere, no dropouts to zero.

## 3. Frequency response (bottom panel, log scale)

1/3-octave averaged spectrum. The lines should lie on top of each other.

- Good: red and green within ~±1 dB everywhere — the timbre is untouched,
  only dynamics changed.
- Bad: a green hump in the bass (ND2 low bands too strong — lower
  `--nd2-bass-scale`), or an overall tilt (something off in the EQ).

## 4. The numbers (DR.txt / measure.py metrics)

| Field | Meaning | What to expect |
|---|---|---|
| DR | TT DR — the standard dynamic range from dr.loudness-war.info | 4–6 = loudness war; 8–12 = healthy; higher is not a goal in itself |
| LUFS | Integrated loudness | Will drop (−9 → −13) — that is normal |
| LRA | Loudness spread between sections | Usually rises slightly |
| TP | True peak | Should be ≤ −0.5 dBTP — headroom against inter-sample clipping |
| clip | Samples at 0 dBFS | Must be 0 after restoration |

- The headline is **DR before → after** (e.g. DR5 → DR9): the album left the
  crushed zone.
- **TP below ceiling and clip = 0** — the signal is technically clean.
- **Lower LUFS is not a defect**: the ReplayGain tag restores playback level
  in compatible players.

## Listening (the final arbiter)

1. Take an original-vs-restored pair loudness-matched (ReplayGain or the
   volume knob).
2. First 30 seconds: just listen — which is more pleasant.
3. Over-restored signs (lower `--nd2`): volume "seesaws", vocals drift, bass
   "breathes".
4. Good restoration signs: less fatigue after 30+ minutes, drums
   distinguishable, quiet details (breaths, reverb tails) audible.
