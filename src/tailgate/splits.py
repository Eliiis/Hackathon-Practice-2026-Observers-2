import json
from pathlib import Path

import numpy as np
import pandas as pd


def forward_chaining(dates, folds=5):
    months = pd.to_datetime(pd.Series(np.asarray(dates)).reset_index(drop=True)).dt.to_period("M")
    available = sorted(months.dropna().unique())
    validation_months = available[-folds:] if len(available) > folds else available[1:]

    out = []
    for month in validation_months:
        train = np.where((months < month).to_numpy())[0]
        valid = np.where((months == month).to_numpy())[0]
        if len(train) and len(valid):
            out.append((train, valid))
    return out


def last_month(dates):
    return forward_chaining(dates, folds=1)[-1]


def save(folds, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"train": tr.tolist(), "valid": va.tolist()}
                                for tr, va in folds]))


def load(path):
    payload = json.loads(Path(path).read_text())
    return [(np.array(f["train"]), np.array(f["valid"])) for f in payload]
