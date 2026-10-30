"""ML layer of music-restoration: dataset versioning, target training, snapshots.

Separation of responsibilities:
  * `dataset.py`  — training table from dataset pairs (features -> target);
  * `models.py`   — model zoo and selection rule based on sample size;
  * `metrics.py`  — target quality metrics with comparison baselines;
  * `registry.py` — dataset versioning and model snapshots;
  * `runtime.py`  — applying a snapshot in the pipeline (target prediction).
"""
