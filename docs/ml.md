# ML layer: dataset versions, target training, snapshots

The dataset is not "data in general" but a versioned artifact: every model's
metadata records which version it was trained on and with which metrics.

## What we predict

The pipeline (DSP) is fixed; we train the **target estimator**: the master's
`Δcrest` from track features. The target is fed into the physics (`fit_r`
inverts `crest(r)`) and always passes through the invariants (TP ceiling, no
introduced clipping, `Δcrest ≤ 8` brackets). The model is an add-on to the
physics, not a replacement for it.

Targets: `dcrest` (primary), `ddr`, and later the 30 band deltas (tilt).

## Versioning

* `dataset/manifest.json` — composition: `version = ds-<hash>` over (pair keys +
  feature contents + labels) plus counters: pairs, tracks, trusted tracks, by
  author, label distribution. The hash is over **content**, not time.
* `dataset/versions.jsonl` — version history.
* `dataset/models/<model_id>/` — snapshot: `model.joblib` + `meta.json`
  (dataset version, features, split, metrics, hyperparameters, git commit).
* `models/registry.jsonl` — plain-text registry (in the repository): model,
  target, dataset version, metrics, commit. Snapshots are compared against it.
* `models/leaderboard.jsonl` — rows of auto-cycle runs (for the summary).

## The model zoo and the selection rule

| model | minimum tracks | purpose |
|---|---|---|
| `table` | 1 | medians by source class; an interpretable baseline |
| `ridge` | 150 | linear on standardized features |
| `knn` | 150 | diagnostics for "similar material → similar target" |
| `hgb` | 400 + 2 authors | HistGradientBoosting (nonlinearity, NaN support out of the box) |
| `conformal` | 150 | conformal band around the point prediction, 80% coverage |
| `hgbq` | 400 | quantile models for the confidence band |

The rule: `suggest_models(n_train, n_authors)` — complexity grows only once the
volume is reached, and each level must prove its benefit on holdout, otherwise
the previous level stays. `lightgbm/xgboost/statsmodels` are deliberately not
used: they are not installed, and HistGradientBoosting covers the task.

## Metrics and baselines

`rnext/ml/metrics.py`: median of `|err|`, share within ±0.5 and ±1.0 dB, p90,
bias — over all tracks and by source class; coverage of the conformal band.

Baselines are mandatory, otherwise the error number is meaningless:
`noop` (no processing), `const46` (the current default), `v0table`
(`corpus/targets_v0.json`). **Important:** `v0table` is built over all current
pairs, so in reports it is slightly "in sample" — treat it as a reference
point, not as an honest competitor.

## Auto-cycle

```bash
python3 scripts/ml_auto.py                 # one pass: version → training → evals → summary
python3 scripts/ml_auto.py --interval 1800 # the same in a loop
python3 scripts/ml_auto.py --with-audio    # plus end-to-end audio evals of pairs with raw data
python3 scripts/ml_train.py --model ridge --by-author   # manual run/experiment
```

Volume levels (`trusted` tracks): **M1 = 600** (target regression), **M2 = 1500**
(bands + tilt + content gate), **M3 = 3000** (universality). Until the volume is
reached, the auto-cycle prints how much is left and does not start training.

Splits: `by-album` (30% of albums) — primary; `by-author` — checks transfer
between masters, enabled automatically at ≥2 authors.

## Continuous enrichment

```bash
python3 scripts/dataset_loop.py --once              # one pass
python3 scripts/dataset_loop.py --interval 900      # keep running in the background
```

A pass: wait for downloads to finish → pick a new batch → download → extract
(raw data is kept for a `--keep-window` of pairs so there is something to
evaluate) → reference audit → target layer → `ml_auto --with-audio`. Log:
`dataset/loop.log`.

## Runtime: the target from the model in the pipeline

```bash
python3 scripts/restore_one.py --src <img.flac> --cue <cue> --out out/ \
        --ml-snapshot auto          # or a specific id
```

The report gains `ml_snapshot` and `ml_target` per track. When a model is used,
the `healthy` threshold is aligned with the training one (`≥ 12`); otherwise a
DR of 11 would stay transparent even though the model predicts a target for it.
