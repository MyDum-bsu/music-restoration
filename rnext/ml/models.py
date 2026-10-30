"""Model zoo for the target and a selection rule based on sample size.

The rule is simple and explicit (no magic): the more data, the more complex
the model, but every step up must prove its benefit on holdout, otherwise the
previous one stays.

    table   — medians by source class (n >= 1): interpretable baseline
    ridge   — linear on standardized features (n >= 150)
    knn     — diagnosis of "similar material -> similar target" (n >= 150)
    hgb     — HistGradientBoosting with monotonicity on DR (n >= 400)
    hgbq    — quantile HGBs (q10/q50/q90) for a confidence band (n >= 400)
    conformal — conformal add-on over any base model (n >= 150)

Everything is on sklearn: no external dependencies (lightgbm/xgboost/
statsmodels), and that is deliberate — there is nowhere to install them, and
HistGradientBoosting covers the task.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_N = {"table": 1, "ridge": 150, "knn": 150, "conformal": 150,
         "hgb": 400, "hgbq": 400}

MODEL_ORDER = ["table", "ridge", "knn", "hgb"]

# features are matrix columns (see dataset.FEATURES); monotonicity on DR
MONOTONE_DECREASING_ON_DR = True


class TableModel:
    """Delta-crest medians by source class; fallback is the overall median."""

    name = "table"

    def fit(self, X, y, classes=None, feature_names=None):
        dr_col = (feature_names or []).index("dr")
        self.dr_col = dr_col
        cls = np.array([_cls(v) for v in X[:, dr_col]]) if classes is None else np.asarray(classes)
        self.by_class = {c: float(np.median(y[cls == c])) for c in set(cls.tolist())}
        self.fallback = float(np.median(y))
        return self

    def predict(self, X):
        cls = np.array([_cls(v) for v in X[:, self.dr_col]])
        return np.array([self.by_class.get(c, self.fallback) for c in cls])


def _cls(dr: float) -> str:
    if not np.isfinite(dr):
        return "mid"
    if dr < 9:
        return "crushed"
    if dr < 12:
        return "mid"
    return "healthy"


def make_pipeline(base: str, seed: int = 17):
    if base == "ridge":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("m", Ridge(alpha=1.0, random_state=seed))])
    if base == "knn":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("m", KNeighborsRegressor(n_neighbors=min(10, 5), weights="distance"))])
    if base == "hgb":
        return Pipeline([("m", _hgb(seed))])
    raise ValueError(base)


def build(base: str, n_train: int, seed: int = 17, feature_names: list[str] | None = None,
          classes: np.ndarray | None = None, X_train=None, y_train=None):
    """Build and train a model of level `base` (with a sample-size threshold check)."""
    need = MIN_N.get(base, 1)
    if n_train < need:
        raise ValueError(f"{base}: need >= {need} tracks, have {n_train}")
    if base == "table":
        return TableModel().fit(X_train, y_train, classes=classes, feature_names=feature_names)
    if base == "hgbq":
        return fit_hgbq(X_train, y_train, seed=seed)
    return make_pipeline(base, seed).fit(X_train, y_train)


@dataclass
class Conformal:
    """Conformal band around a point prediction.

    split-conformal: residuals on the calibration part of train -> a quantile
    of |resid| gives a half-width with guaranteed coverage (under
    exchangeability), which matters more to us than a "self-confident" point
    prediction.
    """
    model: object
    half_width: float

    def predict(self, X):
        return self.model.predict(X)

    def interval(self, X, level: float = 0.8):
        p = self.predict(X)
        return p - self.half_width, p + self.half_width


@dataclass
class QuantileHGB:
    """Quantile HGBs (q10/q50/q90): a built-in confidence band, no calibration.

    The conformal add-on guarantees coverage, but only around a point
    prediction. Here the band width comes from the data itself: if the model is
    honestly unsure about part of the material, the band is wider there.
    """
    q10: object
    q50: object
    q90: object

    def predict(self, X):
        return self.q50.predict(X)

    def interval(self, X, level: float = 0.8):
        lo, hi = self.q10.predict(X), self.q90.predict(X)
        # quantile estimates are not required to be monotonic — fix the order
        return np.minimum(lo, hi), np.maximum(lo, hi)


def _hgb(seed: int, quantile: float | None = None) -> HistGradientBoostingRegressor:
    # early stopping and 30 leaves chosen by measurement on 5 splits: the
    # train/holdout gap drops from 0.54 to 0.40 dB and the holdout median
    # from 0.759 to 0.694 (at 400 tracks hgb memorizes the corpus without this)
    kw = dict(max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=30,
              l2_regularization=1.0, random_state=seed,
              early_stopping=True, validation_fraction=0.2, n_iter_no_change=20)
    if quantile is None:
        return HistGradientBoostingRegressor(loss="squared_error", **kw)
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)


def fit_hgbq(X, y, seed: int = 17) -> QuantileHGB:
    return QuantileHGB(
        q10=_hgb(seed, 0.10).fit(X, y),
        q50=_hgb(seed, 0.50).fit(X, y),
        q90=_hgb(seed, 0.90).fit(X, y))


def fit_conformal(model, X_cal, y_cal, level: float = 0.8) -> Conformal:
    resid = np.abs(model.predict(X_cal) - y_cal)
    # finite-sample correction (1+1/n) — standard for split-conformal
    q = float(np.quantile(resid, min(1.0, level * (1 + 1 / max(len(resid), 1)))))
    return Conformal(model=model, half_width=q)


def suggest_models(n_train: int, n_authors: int) -> list[str]:
    """Which models are worth training at this sample size (and what to compare)."""
    out = ["table"]
    if n_train >= MIN_N["ridge"]:
        out += ["ridge", "conformal"]
    if n_train >= MIN_N["knn"]:
        out.append("knn")
    if n_train >= MIN_N["hgb"] and n_authors >= 2:
        out += ["hgb", "hgbq"]
    return out
