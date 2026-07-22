import json
import time
from contextlib import contextmanager

import numpy as np

from . import analysis, data, evaluate, features, figures, shift, splits
from .config import ARTIFACTS, DATA, SPLIT, SUBMISSIONS
from .logs import get
from .metric import SEGMENT_NAMES
from .model import IncomeModel

log = get()

FOLDS_PATH = ARTIFACTS / "folds.json"
OUT_OF_FOLD_PATH = ARTIFACTS / "out_of_fold.npy"
TEST_PREDICTION_PATH = ARTIFACTS / "test_prediction.npy"
SUBMISSION_PATH = SUBMISSIONS / "submission.csv"


@contextmanager
def stage(name):
    log.info(f"[{name}]")
    started = time.time()
    yield
    log.info(f"[{name}] за {time.time() - started:,.0f}s")


def training_set():
    schema = data.schema_of(data.read("train"))
    frame = data.read("train_features")
    selected, categorical = features.select(frame, schema)
    return (frame, frame[DATA["target"]].to_numpy(), frame[DATA["weight"]].to_numpy(),
            selected, categorical)


def parse():
    with stage("parse"):
        train = data.build_typed("train")
        schema = data.schema_of(train)
        test = data.build_typed("test", train_schema=schema)
        log.info(f"  трейн {train.shape}  тест {test.shape}")
        log.info(f"  числовых {len(schema.numeric)}  категориальных {len(schema.categorical)}")
        log.info(f"  трейн {train[DATA['date']].min():%Y-%m}..{train[DATA['date']].max():%Y-%m}"
                 f"  тест {test[DATA['date']].min():%Y-%m}..{test[DATA['date']].max():%Y-%m}")


def build_features():
    with stage("features"):
        train = data.read("train")
        schema = data.schema_of(train)
        constants = features.proxy_constants(train, train[DATA["target"]])
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "income_proxies.json").write_text(json.dumps(
            {name: {"scale": scale, "sigma": sigma}
             for name, (scale, sigma) in constants.items()}, indent=2))
        for name, (scale, sigma) in constants.items():
            log.info(f"  прокси {name[:44]:<44} k={scale:>7.3f}  sigma={sigma:.3f}")

        fitted, cross = features.fit_proxy_sigma(train, train[DATA["target"]], constants,
                                                 schema.numeric, schema.categorical)
        log.info(f"  контекстная сигма обучена для {len(fitted)} источников")

        engineered = features.engineer(train, numeric=schema.numeric, constants=constants,
                                       sigma=cross)
        train_joined = train.join(engineered)

        test = data.read("test")
        test_sigma = features.apply_proxy_sigma(test, fitted, schema.numeric, schema.categorical)
        test_engineered = features.engineer(test, numeric=schema.numeric,
                                            keep=list(engineered.columns), constants=constants,
                                            sigma=test_sigma)
        test_joined = test.join(test_engineered)

        selected, categorical = features.select(train_joined, schema)
        layout = features.channel_layout(selected)
        channels, train_arena = features.fit_channels(
            train_joined, train[DATA["target"]],
            train[DATA["date"]].dt.to_period("M").astype(str), categorical, layout)
        test_arena = features.apply_channels(test_joined, channels, layout)
        for name, columns in layout.items():
            log.info(f"  канал {name:<10} колонок {len(columns):>4}")

        train_joined.join(train_arena).to_parquet(data.parquet_path("train_features"))
        test_joined.join(test_arena).to_parquet(data.parquet_path("test_features"))
        log.info(f"  построено {engineered.shape[1] + train_arena.shape[1]} признаков")


def folds():
    with stage("folds"):
        frame = data.read("train_features")
        generated = splits.forward_chaining(frame[DATA["date"]], folds=SPLIT["folds"])
        splits.save(generated, FOLDS_PATH)
        log.info(f"  {len(generated)} out-of-time фолдов, "
                 f"размеры валидации {[len(v) for _, v in generated]}")


def estimate_shift():
    with stage("shift"):
        frame, y, weight, selected, categorical = training_set()
        test = data.read("test_features")
        train_index, valid_index = splits.last_month(frame[DATA["date"]])

        model = IncomeModel(categorical).fit(frame[selected].iloc[train_index],
                                             y[train_index], weight[train_index])
        estimated = shift.estimate(y[valid_index],
                                   model.predict(frame[selected].iloc[valid_index]),
                                   model.predict(test[selected]))
        path = shift.save(estimated)
        frozen = shift.priors()[1]
        for name, test_prior, fixed in zip(SEGMENT_NAMES, estimated["test_prior"], frozen):
            log.info(f"  {name:<10} свежая {test_prior:>7.2%}  в конфиге {fixed:>7.2%}"
                     f"  расхождение {test_prior - fixed:>+8.4%}")
        log.info(f"  свежая оценка -> {path}; чтобы принять её, перенесите значения"
                 f" в config.yaml")


def validate():
    with stage("validate"):
        frame, y, weight, selected, categorical = training_set()
        result = evaluate.cross_validate(frame[selected], y, weight, categorical,
                                         frame[DATA["date"]], splits.load(FOLDS_PATH),
                                         clip=(y.min(), y.max()))
        np.save(OUT_OF_FOLD_PATH, result["out_of_fold"])
        evaluate.report(result, y, weight)


def analyze():
    with stage("analysis"):
        frame, y, weight, selected, categorical = training_set()
        every_fold = splits.load(FOLDS_PATH)
        used = every_fold[-analysis.ABLATION_FOLDS:]

        out_of_fold = np.load(OUT_OF_FOLD_PATH)
        validation = evaluate.score_out_of_fold(out_of_fold, y, weight, every_fold)
        validation["folds"] = len(every_fold)

        table, model, engineered = analysis.ablation(
            frame, y, weight, selected, categorical, frame[DATA["date"]], used,
            clip=(y.min(), y.max()))
        gains = analysis.feature_gains(model, engineered)
        breakdown = evaluate.breakdown_of({"out_of_fold": out_of_fold}, y, weight)
        path = analysis.write_report(table, gains, breakdown, validation, y, len(used),
                                     engineered)
        log.info(f"  отчёт -> {path}")
        for drawn in figures.draw(breakdown, table, gains, out_of_fold, y, weight,
                                  every_fold):
            log.info(f"  график -> {drawn}")


def submit():
    with stage("submit"):
        frame, y, weight, selected, categorical = training_set()
        test = data.read("test_features")
        prediction = evaluate.fit_predict(frame[selected], y, weight, categorical,
                                          frame[DATA["date"]], test[selected],
                                          clip=(y.min(), y.max()))
        np.save(TEST_PREDICTION_PATH, prediction)
        path = data.write_submission(test[DATA["id"]], prediction, SUBMISSION_PATH)
        log.info(f"  медиана {np.median(prediction):,.0f}"
                 f"  доля >=300k {np.mean(prediction >= 300000):.2%}"
                 f"  ожидание {shift.priors()[1][-1]:.2%}")
        log.info(f"  сабмит -> {path}")
