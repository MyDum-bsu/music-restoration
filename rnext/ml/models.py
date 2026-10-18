"""Зоопарк моделей цели и правило выбора по объёму выборки.

Правило простое и явное (никакой магии): чем больше данных, тем сложнее модель,
но каждый переход должен доказать пользу на holdout, иначе остаётся предыдущая.

    table   — медианы по классу исходника (n ≥ 1): интерпретируемая база
    ridge   — линейная на стандартизованных признаках (n ≥ 150)
    knn     — диагностика «похожий материал → похожая цель» (n ≥ 150)
    hgb     — HistGradientBoosting с монотонностью по DR (n ≥ 400)
    hgbq    — квантильные HGB (q10/q50/q90) для полосы доверия (n ≥ 400)
    conformal — конформная надстройка над любой базовой моделью (n ≥ 150)

Всё на sklearn: внешних зависимостей (lightgbm/xgboost/statsmodels) нет, и это
осознанно — их негде ставить, а HistGradientBoosting закрывает задачу.
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

# признаки — столбцы матрицы (см. dataset.FEATURES); монотонность по DR
MONOTONE_DECREASING_ON_DR = True


class TableModel:
    """Медианы Δcrest по классу исходника; фолбэк — общая медиана."""

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
        # ранняя остановка и листья 30 выбраны замером на 5 сплитах: разрыв
        # обучение/отложено падает с 0.54 до 0.40 дБ, а отложенная медиана
        # с 0.759 до 0.694 (при 400 треках без этого hgb запоминает корпус)
        return Pipeline([("m", HistGradientBoostingRegressor(
            loss="squared_error", max_depth=3, learning_rate=0.05, max_iter=400,
            min_samples_leaf=30, l2_regularization=1.0, random_state=seed,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=20))])
    raise ValueError(base)


def build(base: str, n_train: int, seed: int = 17, feature_names: list[str] | None = None,
          classes: np.ndarray | None = None, X_train=None, y_train=None):
    """Собрать и обучить модель уровня `base` (с проверкой порога объёма)."""
    need = MIN_N.get(base, 1)
    if n_train < need:
        raise ValueError(f"{base}: нужно ≥ {need} треков, есть {n_train}")
    if base == "table":
        return TableModel().fit(X_train, y_train, classes=classes, feature_names=feature_names)
    return make_pipeline(base, seed).fit(X_train, y_train)


@dataclass
class Conformal:
    """Конформная полоса вокруг точечного прогноза.

    split-conformal: остатки на калибровочной части train → квантиль |resid|
    даёт полуширину с гарантированным покрытием (при обменяемости), что для нас
    важнее «самоуверенного» точечного прогноза.
    """
    model: object
    half_width: float

    def predict(self, X):
        return self.model.predict(X)

    def interval(self, X, level: float = 0.8):
        p = self.predict(X)
        return p - self.half_width, p + self.half_width


def fit_conformal(model, X_cal, y_cal, level: float = 0.8) -> Conformal:
    resid = np.abs(model.predict(X_cal) - y_cal)
    # конечная поправка (1+1/n) — стандартная для split-conformal
    q = float(np.quantile(resid, min(1.0, level * (1 + 1 / max(len(resid), 1)))))
    return Conformal(model=model, half_width=q)


def suggest_models(n_train: int, n_authors: int) -> list[str]:
    """Какие модели имеет смысл обучать при таком объёме (и что сравнивать)."""
    out = ["table"]
    if n_train >= MIN_N["ridge"]:
        out += ["ridge", "conformal"]
    if n_train >= MIN_N["knn"]:
        out.append("knn")
    if n_train >= MIN_N["hgb"] and n_authors >= 2:
        out += ["hgb", "hgbq"]
    return out
