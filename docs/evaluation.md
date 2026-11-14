# Evaluating a restoration: which command answers which question

"Is the restoration good?" is several different questions. Each level below
answers one of them with a different tool; never read a lower level's answer as
a higher level's claim.

| Level | Question | Tool | What it can prove |
|---|---|---|---|
| 0 | Is the environment sound? | `python3 scripts/doctor.py` | Dependencies, workspace, free disk |
| 1 | Did every track get an explicit outcome? | `report.json` next to the restored album | `processed_candidate` / `kept_original` / `qc_fallback_original` / `error`, `needs_review` |
| 2 | Is the restored album better than its own source? | `python3 scripts/evaluate_album.py --src SRC --restored OUT --out EVAL` | Per-track and album verdict by a ladder declared before the data; systematic shifts; runner statuses |
| 3 | Independent delivery audit | `python3 scripts/qc_gates.py --orig SRC OUT/*.flac` | Peak / clipping / sample-jump gates against the source take |
| 4 | How close are we to master restorations? | `python3 scripts/eval_vs_masters.py --manifest M.json --out-dir OUT` | Seeded, reproducible distance-to-master tables (`better/worse/comparable`) |
| 5 | Is the difference audible and preferred? | `scripts/listen_pack.py` (research loop; product wiring planned) | Blind listening — not covered by metrics |

## Reading the evaluate_album report

`evaluate_album` measures the source and the restored album with the same
instrument (`audio_measure.py` plus the shared spike/band kernel in
`eval_vs_masters.measure_file`), matches tracks by the leading digits of the
stem and then by normalized name (parenthesized segments and the leading
number stripped), and lifts `restore_safe`'s `report.json` statuses into the
verdicts when the file is present.

The per-track verdict ladder is fixed before the data:

1. our output's true peak above the ceiling (default −0.9 dBFS) loses
   immediately (`orig_better/ceiling`); the original is not judged by the
   ceiling — a hot master is expected;
2. a DR gap of ≥ 0.3 decides (`higher DR wins`);
3. otherwise click-like sample jumps (|x| > 0.9) decide (fewer wins);
4. otherwise `comparable/tie`.

LUFS never votes: a gap beyond 2 LU marks `loudness_mismatch` — our pipeline
targets quieter masters by design. `summary.json` keeps the medians (loudness
gap, DR/crest change, band shift), the gate counters (tracks at the ceiling,
DR regressions, added clipping or jumps) and the `needs_review` count. A track
flagged `needs_review` by the runner is unresolved work, not a success.

`REPORT.md` renders the same numbers plus its own limitations. Nothing in it
is typed by hand.

## Corpus-level comparison with masters

`eval_vs_masters.py` takes a manifest of `orig`/`ours`/`master` directory
triples and produces `tracks.csv`, `deltas.csv`, `albums.csv`,
`summary.json` and an optional markdown report. Selection of releases from a
rutracker-corpus jsonl is seeded and reproducible:

```bash
python3 scripts/eval_vs_masters.py --select --corpus catalog.jsonl \
        --seed 20261104 --per-master 5 --out picks.json
python3 scripts/eval_vs_masters.py --manifest manifest.json --out-dir eval-out \
        --report eval-out/EVAL_REPORT.md
```

`better` means our distance to the master is smaller than the original's
beyond a 5% tolerance — closeness to a chosen reference, not superiority over
the restorer. The first documented run of this procedure lives in
[eval-vs-masters-run.md](eval-vs-masters-run.md); its systematic findings
(quieter output, pinned true peak, darker spectrum) are the backlog for the
DSP improvements.

## Known limitations

- DR, crest, clipping and spikes are measured on the first ~90 s of each file;
  LUFS and true peak cover the whole file.
- Metrics are engineering proxies: they prove output constraints and
  dynamics/spectrum shifts, never listening preference. ABX proves
  distinguishability, not preference; both are separate loops.
- Band shift mixes level change with tonal change; read it as an indicator.
- The verdict compares against the source take, not against the original
  studio intent or a named restorer's output (use level 4 for that).
- Handpicked regression albums establish specific fixes, not broad
  superiority; frozen holdout selection by artist/album is future work.

## Reproducing any number

- One cell of `tracks.csv`: `python3 scripts/audio_measure.py --json /tmp/x.json "<file>"`.
- A whole album evaluation: re-run the same `evaluate_album` command — the
  measurement and the ladder are deterministic.
- A corpus run: the seed is recorded in `summary.json`; rebuild the manifest
  and re-run. Unmatched tracks are reported, never filled.
