from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA, PROCESSED, ROOT, SUBMISSION

NA_TOKENS = ["", "None", "none", "nan", "NaN", "NAN", "null", "NULL",
             "NA", "N/A", "<NA>", "-"]
NUMERIC_PARSE_RATE = 0.99


@dataclass(frozen=True)
class Schema:
    numeric: list
    categorical: list
    id: str | None
    date: str | None
    target: str | None
    weight: str | None


def to_number(series):
    return pd.to_numeric(series.str.replace(",", ".", regex=False), errors="coerce")


def read_raw(split):
    path = ROOT / DATA[split]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; place the competition csv files in data/raw/")
    return pd.read_csv(path, sep=DATA["sep"], dtype=str, keep_default_na=False,
                       na_values=NA_TOKENS, encoding=DATA["encoding"])


def detect_roles(raw):
    reserved = {DATA["id"], DATA["date"], DATA["target"], DATA["weight"]}
    numeric, categorical = [], []
    for column in raw.columns:
        if column in reserved:
            continue
        values = raw[column]
        present = values.notna().sum()
        unparsed = (to_number(values).isna() & values.notna()).sum()
        parses = present > 0 and unparsed / present <= 1 - NUMERIC_PARSE_RATE
        (numeric if parses else categorical).append(column)
    return numeric, categorical


def cast(raw, numeric, categorical):
    columns = {}
    if DATA["id"] in raw:
        columns[DATA["id"]] = pd.to_numeric(raw[DATA["id"]], errors="coerce").astype("int64")
    if DATA["date"] in raw:
        columns[DATA["date"]] = pd.to_datetime(raw[DATA["date"]], errors="coerce")
    for name in (DATA["target"], DATA["weight"]):
        if name in raw:
            columns[name] = to_number(raw[name]).astype("float64")
    for name in numeric:
        if name in raw:
            columns[name] = to_number(raw[name]).astype("float32")
    for name in categorical:
        if name in raw:
            columns[name] = (raw[name].str.strip()
                             .str.replace(r"\s+", " ", regex=True).astype("category"))
    frame = pd.DataFrame(columns, index=raw.index)
    return frame[[c for c in raw.columns if c in frame.columns]]


def schema_of(frame):
    reserved = {DATA["id"], DATA["target"], DATA["weight"]}
    numeric = [c for c in frame.columns
               if c not in reserved and pd.api.types.is_numeric_dtype(frame[c])
               and c != DATA["date"]]
    categorical = [c for c in frame.columns
                   if isinstance(frame[c].dtype, pd.CategoricalDtype) or frame[c].dtype == object]
    return Schema(
        numeric=numeric,
        categorical=categorical,
        id=DATA["id"] if DATA["id"] in frame else None,
        date=DATA["date"] if DATA["date"] in frame else None,
        target=DATA["target"] if DATA["target"] in frame else None,
        weight=DATA["weight"] if DATA["weight"] in frame else None,
    )


def parquet_path(name):
    return PROCESSED / f"{name}.parquet"


def build_typed(split, train_schema=None):
    raw = read_raw(split)
    if train_schema is None:
        numeric, categorical = detect_roles(raw)
    else:
        numeric, categorical = train_schema.numeric, train_schema.categorical
    frame = cast(raw, numeric, categorical)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path(split))
    return frame


def read(name):
    return pd.read_parquet(parquet_path(name))


def write_submission(ids, predictions, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({SUBMISSION["id"]: np.asarray(ids),
                          SUBMISSION["prediction"]: np.rint(predictions).astype("int64")})
    frame.set_index(SUBMISSION["id"]).to_csv(path, sep=SUBMISSION["sep"])
    return path
