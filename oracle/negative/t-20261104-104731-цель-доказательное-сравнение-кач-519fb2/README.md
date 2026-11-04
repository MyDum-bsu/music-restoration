# Negative control corpus

Mutant "work" for
`t-20261104-104731-цель-доказательное-сравнение-кач-519fb2`.

`kit.patch` is a new-file patch (three hunks, `-p1`), applied **on top of the
base checkout** before the worker's work exists:

```
scripts/experiment/select_albums.py   no cap, no seed use, timestamp in manifest
scripts/experiment/ref_restore.py     byte-for-byte copy, no declip, no -14 LUFS,
                                      no true-peak ceiling, no path guards
scripts/experiment/compare_restore.py invented numbers, no audio_measure call
```

Nothing else of the deliverable exists in this corpus: no
`docs/experiment/independent-method.md`, no `docs/experiment/2026-11-04/`,
no `docs/experiment-independent-restore-2026-11-04.md`.

A correct oracle must reject it. The tests that must fail (and do, for the
reasons in brackets):

| test | reason |
|---|---|
| `test_method_is_documented` | no method doc |
| `test_select_five_per_artist_is_deterministic` | 7 albums for `Alpha`, manifest keys wrong, bytes differ per run |
| `test_restore_bounds_peak_and_clipping_and_keeps_sources` | unknown `--work-dir` argument, passthrough keeps clipping and 0 dBFS peaks |
| `test_restore_refuses_overlap_and_canon_paths` | writes into `src/indep` and into `[restored per-track]` with exit 0 |
| `test_compare_emits_documented_schema` | 6 columns instead of 20, one variant only |
| `test_compare_metrics_match_audio_measure` | no ORIG rows at all |
| `test_verdicts_follow_the_documented_rule` | no `verdicts.csv` |
| `test_compare_refuses_incomplete_triple` | exits 0 on a missing counterpart |
| `test_real_report_and_data_are_consistent` | no report, no CSVs, no manifest |
| `test_properties_invariants_per_seed` | same as the schema/CSV failures |

`test_scripts_exist_and_restore_is_independent` and
`test_restore_is_reproducible_metricwise` are expected to *pass* on the mutant:
they guard independence and determinism, not quality. If they fail, the oracle
is over-fitted to the reference implementation.
