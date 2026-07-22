import numpy as np
import pandas as pd

PIVOT = 84017.07996115311

SEGMENT_EDGES = [0.0, 50_000.0, PIVOT, 150_000.0, 300_000.0, np.inf]
SEGMENT_NAMES = ["до 50k", "50–84k", "84–150k", "150–300k", "от 300k"]


def wmae(y_true, y_pred, weight):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    weight = np.asarray(weight, float)
    return float(np.sum(weight * np.abs(y_true - y_pred)) / np.sum(weight))


def segment_index(y):
    return np.asarray(pd.cut(np.asarray(y, float), SEGMENT_EDGES, labels=False), int)


def segment_breakdown(y_true, y_pred, weight):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    weight = np.asarray(weight, float)
    error = weight * np.abs(y_true - y_pred)
    idx = segment_index(y_true)

    rows = []
    for k, name in enumerate(SEGMENT_NAMES):
        m = idx == k
        if not m.any():
            continue
        rows.append({
            "segment": name,
            "rows": int(m.sum()),
            "rows_share": float(m.mean()),
            "metric_share": float(error[m].sum() / error.sum()),
            "wmae": float(error[m].sum() / weight[m].sum()),
            "bias": float(np.median(y_pred[m] - y_true[m])),
        })
    return pd.DataFrame(rows)


def format_breakdown(table):
    lines = [f"{'segment':<11}{'rows':>8}{'rows%':>8}{'metric%':>9}{'wmae':>10}{'bias':>11}"]
    for r in table.itertuples():
        lines.append(f"{r.segment:<11}{r.rows:>8,}{r.rows_share:>7.1%}"
                     f"{r.metric_share:>9.1%}{r.wmae:>10,.0f}{r.bias:>11,.0f}")
    return "\n".join(lines)
