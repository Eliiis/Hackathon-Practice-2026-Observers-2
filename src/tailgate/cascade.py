import numpy as np
import xgboost as xgb

from . import shift
from .config import MODEL
from .model import IncomeModel, gate_params, pin_categories

EPSILON = 1e-6
TEMPERATURE_GRID = np.geomspace(0.25, 4.0, 61)
FALLBACK_HOLDOUT = 0.15


def logit(probability):
    probability = np.clip(np.asarray(probability, float), EPSILON, 1 - EPSILON)
    return np.log(probability / (1 - probability))


def calibration_error(probability, outcome, bins=15):
    index = np.clip(np.digitize(probability, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        selected = index == b
        if selected.any():
            error += selected.mean() * abs(outcome[selected].mean() - probability[selected].mean())
    return error


def fit_temperature(probability, outcome):
    z = logit(probability)
    return float(min(TEMPERATURE_GRID,
                     key=lambda t: calibration_error(1 / (1 + np.exp(-z / t)), outcome)))


def apply_temperature(probability, temperature):
    return 1 / (1 + np.exp(-logit(probability) / temperature))


def prior_correct(probability, ratios):
    stacked = np.column_stack([1 - probability, probability]) * np.asarray(ratios, float)
    return (stacked / stacked.sum(axis=1, keepdims=True))[:, 1]


class Gate:
    def __init__(self, categorical, seed=None):
        self.categorical = list(categorical)
        self.seed = seed

    def fit(self, X, y, sample_weight):
        if len(X) == 0 or len(np.unique(y)) < 2:
            raise ValueError("degenerate gate")
        self.features_ = list(X.columns)
        self.categorical_ = [c for c in self.categorical if c in X.columns]
        pinned, self.levels_ = pin_categories(X, self.categorical_)
        self.model_ = xgb.XGBClassifier(**gate_params(self.seed))
        self.model_.fit(pinned, np.asarray(y, int), sample_weight=sample_weight)
        return self

    def predict(self, X):
        pinned, _ = pin_categories(X[self.features_], self.categorical_, self.levels_)
        return self.model_.predict_proba(pinned)[:, 1]


class TailCascade:
    def __init__(self, categorical, dates, seed=None):
        self.categorical = categorical
        self.dates = dates
        self.seed = seed
        self.threshold = float(MODEL["gate_threshold"])
        self.context_from = float(MODEL["expert_context_from"])
        self.context_weight = float(MODEL["expert_context_weight"])

    def _holdout(self, X):
        months = self.dates.loc[X.index].dt.to_period("M")
        holdout = (months == months.max()).to_numpy()
        if 0.0 < holdout.mean() < 1.0:
            return holdout
        return np.random.default_rng(42).random(len(X)) < FALLBACK_HOLDOUT

    def fit(self, X, y, sample_weight):
        y = np.asarray(y, float)
        weight = shift.training_weights(y, np.asarray(sample_weight, float))
        rich = (y >= self.threshold).astype(int)

        holdout = self._holdout(X)
        self.temperature_ = None
        try:
            preliminary = Gate(self.categorical, self.seed).fit(X[~holdout], rich[~holdout],
                                                                weight[~holdout])
            self.temperature_ = fit_temperature(preliminary.predict(X[holdout]),
                                                rich[holdout].astype(float))
        except ValueError:
            pass
        self.gate_ = Gate(self.categorical, self.seed).fit(X, rich, weight)

        context = y >= self.context_from
        expert_weight = np.where(y[context] >= self.threshold, weight[context],
                                 self.context_weight * weight[context])
        self.expert_ = IncomeModel(self.categorical, self.seed).fit(X[context], y[context],
                                                                    expert_weight)
        self.base_ = IncomeModel(self.categorical, self.seed).fit(X[rich == 0], y[rich == 0],
                                                                  weight[rich == 0])
        return self

    def probability(self, X):
        p = self.gate_.predict(X)
        if self.temperature_ is not None:
            p = apply_temperature(p, self.temperature_)
        return prior_correct(p, shift.gate_ratios(self.threshold))

    def predict(self, X):
        p = self.probability(X)
        return p * self.expert_.predict(X) + (1 - p) * self.base_.predict(X)

    def gains(self):
        return self.expert_.gains()


class Blend:
    def __init__(self, categorical, dates, seed=None):
        self.cascade_ = TailCascade(categorical, dates, seed)
        self.solo_ = IncomeModel(categorical, seed)
        self.share = float(MODEL["blend_solo"])

    def fit(self, X, y, sample_weight):
        y = np.asarray(y, float)
        sample_weight = np.asarray(sample_weight, float)
        self.cascade_.fit(X, y, sample_weight)
        self.solo_.fit(X, y, shift.training_weights(y, sample_weight))
        return self

    def predict(self, X):
        return ((1 - self.share) * self.cascade_.predict(X)
                + self.share * self.solo_.predict(X))

    def gains(self):
        return self.cascade_.gains()
