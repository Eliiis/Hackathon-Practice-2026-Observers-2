import numpy as np

from . import shift
from .cascade import Blend
from .logs import get
from .metric import format_breakdown, segment_breakdown, wmae

log = get()


def cross_validate(X, y, weight, categorical, dates, folds, clip):
    compass = shift.shifted_weights(y, weight)
    out_of_fold = np.full(len(y), np.nan)
    plain, shifted = [], []

    for number, (train, valid) in enumerate(folds, 1):
        model = Blend(categorical, dates).fit(X.iloc[train], y[train], weight[train])
        prediction = np.clip(model.predict(X.iloc[valid]), *clip)
        out_of_fold[valid] = prediction
        plain.append(wmae(y[valid], prediction, weight[valid]))
        shifted.append(wmae(y[valid], prediction, compass[valid]))
        log.info(f"  фолд {number}/{len(folds)}  трейн {len(train):>6,}  валид {len(valid):>6,}"
                 f"  plain {plain[-1]:>9,.0f}  shifted {shifted[-1]:>9,.0f}")

    return {"out_of_fold": out_of_fold, "plain": float(np.mean(plain)),
            "shifted": float(np.mean(shifted))}


def fit_predict(X, y, weight, categorical, dates, X_test, clip):
    model = Blend(categorical, dates).fit(X, y, weight)
    return np.clip(model.predict(X_test), *clip)


def breakdown_of(result, y, weight):
    covered = ~np.isnan(result["out_of_fold"])
    return segment_breakdown(y[covered], result["out_of_fold"][covered], weight[covered])


def score_out_of_fold(out_of_fold, y, weight, folds):
    compass = shift.shifted_weights(y, weight)
    return {"plain": float(np.mean([wmae(y[v], out_of_fold[v], weight[v]) for _, v in folds])),
            "shifted": float(np.mean([wmae(y[v], out_of_fold[v], compass[v]) for _, v in folds]))}


def report(result, y, weight):
    log.info(f"  plain {result['plain']:,.0f}   SHIFTED {result['shifted']:,.0f}")
    for line in format_breakdown(breakdown_of(result, y, weight)).splitlines():
        log.info(f"  {line}")
