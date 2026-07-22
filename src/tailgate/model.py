import numpy as np
import pandas as pd
import xgboost as xgb

from .config import MODEL


def params(seed=None):
    return dict(
        n_estimators=MODEL["trees"],
        max_bin=MODEL["max_bin"],
        max_leaves=MODEL["max_leaves"],
        learning_rate=MODEL["learning_rate"],
        n_jobs=MODEL["threads"],
        max_depth=0,
        grow_policy="lossguide",
        min_child_weight=60,
        subsample=0.8,
        colsample_bytree=0.7,
        objective="reg:absoluteerror",
        tree_method="hist",
        device="cpu",
        enable_categorical=True,
        random_state=MODEL["seed"] if seed is None else seed,
        verbosity=0,
    )


def gate_params(seed=None):
    return {**params(seed), "objective": "binary:logistic",
            "n_estimators": MODEL["gate_trees"], "learning_rate": MODEL["gate_learning_rate"]}


def pin_categories(frame, categorical, levels=None):
    frame = frame.copy()
    learning = levels is None
    levels = {} if learning else levels
    for name in categorical:
        if learning:
            frame[name] = frame[name].astype("category")
            levels[name] = frame[name].cat.categories
        else:
            frame[name] = pd.Categorical(frame[name], categories=levels[name])
    return frame, levels


class IncomeModel:
    def __init__(self, categorical, seed=None, trees=None):
        self.categorical = list(categorical)
        self.seed = seed
        self.trees = trees

    def fit(self, X, y, sample_weight):
        self.features_ = list(X.columns)
        self.categorical_ = [c for c in self.categorical if c in X.columns]
        pinned, self.levels_ = pin_categories(X, self.categorical_)
        settings = params(self.seed)
        if self.trees:
            settings["n_estimators"] = self.trees
        self.model_ = xgb.XGBRegressor(**settings)
        self.model_.fit(pinned, np.log1p(np.asarray(y, float)),
                        sample_weight=np.asarray(sample_weight, float))
        return self

    def predict(self, X):
        pinned, _ = pin_categories(X[self.features_], self.categorical_, self.levels_)
        return np.expm1(self.model_.predict(pinned))

    def gains(self):
        return self.model_.get_booster().get_score(importance_type="gain")
