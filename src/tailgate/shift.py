import json

import numpy as np

from .config import ARTIFACTS, SHIFT
from .metric import SEGMENT_EDGES, SEGMENT_NAMES, segment_index

PATH = ARTIFACTS / "shift.json"


def estimate(y_valid, valid_prediction, test_prediction):
    true_bin = segment_index(y_valid)
    predicted_bin = segment_index(valid_prediction)
    test_bin = segment_index(test_prediction)
    n = len(SEGMENT_NAMES)

    confusion = np.zeros((n, n))
    for j in range(n):
        rows = true_bin == j
        if rows.any():
            for i in range(n):
                confusion[i, j] = np.mean(predicted_bin[rows] == i)

    observed = np.array([np.mean(test_bin == i) for i in range(n)])
    train_prior = np.array([np.mean(true_bin == j) for j in range(n)])

    test_prior = np.linalg.lstsq(confusion, observed, rcond=None)[0]
    test_prior = np.clip(test_prior, 1e-4, None)
    test_prior /= test_prior.sum()

    return {"train_prior": train_prior, "test_prior": test_prior,
            "ratios": test_prior / train_prior}


def save(estimated):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps({
        "segments": SEGMENT_NAMES,
        "train_prior": [float(v) for v in estimated["train_prior"]],
        "test_prior": [float(v) for v in estimated["test_prior"]],
    }, indent=2))
    return PATH


def priors():
    return (np.array(SHIFT["train_prior"], float),
            np.array(SHIFT["test_prior"], float))


def ratios():
    train_prior, test_prior = priors()
    return test_prior / train_prior


def gate_ratios(threshold):
    train_prior, test_prior = priors()
    if threshold not in SEGMENT_EDGES:
        raise ValueError(f"threshold {threshold} must be one of {SEGMENT_EDGES}")
    cut = SEGMENT_EDGES.index(threshold)
    return np.array([test_prior[:cut].sum() / train_prior[:cut].sum(),
                     test_prior[cut:].sum() / train_prior[cut:].sum()])


def shifted_weights(y, weight):
    return np.asarray(weight, float) * ratios()[segment_index(y)]


def training_weights(y, weight):
    shifted = shifted_weights(y, weight)
    return shifted * (np.sum(weight) / np.sum(shifted))
