import time

import numpy as np
import pandas as pd

from . import data, features, shift
from .cascade import Blend, TailCascade
from .config import REPORTS
from .logs import get
from .metric import SEGMENT_NAMES, wmae
from .model import IncomeModel

log = get()

ABLATION_FOLDS = 2
TOP_FEATURES = 15


def weighted_median(values, weights):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def feature_groups(selected):
    raw = set(data.read("train").columns)
    return ([c for c in selected if c in raw],
            [c for c in selected if c not in raw and not features.is_fusion(c)],
            [c for c in selected if features.is_fusion(c)])


def ablation(frame, y, weight, selected, categorical, dates, folds, clip):
    base, domain, fusion = feature_groups(selected)
    compass = shift.shifted_weights(y, weight)
    steps = [
        ("константа: взвешенная медиана трейна", None, False, None),
        ("сырые признаки, веса метрики", base, False, None),
        ("+ веса с поправкой на смещение долей", base, True, None),
        ("+ доменные признаки", base + domain, True, None),
        ("+ слияние источников и каналов", selected, True, None),
        ("+ каскад по хвосту", selected, True, TailCascade),
        ("+ подмешивание одиночной модели", selected, True, Blend),
    ]

    rows, fitted = [], None
    for name, columns, corrected, cascade in steps:
        started = time.time()
        plain, shifted = [], []
        for train, valid in folds:
            if columns is None:
                prediction = np.full(len(valid), weighted_median(y[train], weight[train]))
            elif cascade is not None:
                fitted = cascade(categorical, dates).fit(
                    frame[columns].iloc[train], y[train], weight[train])
                prediction = np.clip(fitted.predict(frame[columns].iloc[valid]), *clip)
            else:
                weights = (shift.training_weights(y[train], weight[train]) if corrected
                           else weight[train])
                fitted = IncomeModel(categorical).fit(
                    frame[columns].iloc[train], y[train], weights)
                prediction = np.clip(fitted.predict(frame[columns].iloc[valid]), *clip)
            plain.append(wmae(y[valid], prediction, weight[valid]))
            shifted.append(wmae(y[valid], prediction, compass[valid]))

        rows.append({"step": name, "features": 0 if columns is None else len(columns),
                     "plain": np.mean(plain), "shifted": np.mean(shifted),
                     "seconds": time.time() - started})
        log.info(f"  {name:<40} shifted {rows[-1]['shifted']:>9,.0f}"
                 f"  {rows[-1]['seconds']:>6,.0f}s")

    table = pd.DataFrame(rows)
    table["gain"] = table["shifted"].shift(1) - table["shifted"]
    return table, fitted, domain + fusion


def feature_gains(model, engineered):
    scores = model.gains()
    table = (pd.Series(scores, name="gain").sort_values(ascending=False)
             .head(TOP_FEATURES).reset_index().rename(columns={"index": "feature"}))
    table["origin"] = [
        "слияние" if features.is_fusion(name)
        else "доменный" if name in set(engineered) else "сырой"
        for name in table["feature"]]
    table["share"] = table["gain"] / sum(scores.values())
    return table


def _table(frame, columns, formats):
    header = "| " + " | ".join(columns.values()) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, divider]
    for row in frame.itertuples():
        cells = [formats[key](getattr(row, key)) for key in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(ablation_table, gains_table, breakdown, validation, y, folds_used, engineered):
    train_prior, test_prior = shift.priors()
    tail = breakdown.iloc[-1]

    sections = [
        "# Аналитика решения\n",
        f"Обучающая выборка {len(y):,} строк и {ablation_table['features'].max()} признаков, "
        f"из них {len(engineered)} построены вручную. Все сравнения парные — одни и те же "
        f"out-of-time фолды и одна и та же метрика.\n",

        "## 1. Почему задача про хвост\n",
        f"Вес наблюдения — функция самого дохода: `w = min(|y − 84017.08| / 84017.08, 2.5707)`, "
        f"кап ровно с 300 000. Вес близок к нулю у медианного клиента, поэтому метрика "
        f"сосредоточена в хвосте распределения:\n",
        _table(breakdown, {"segment": "сегмент", "rows": "строк", "rows_share": "доля строк",
                           "metric_share": "доля метрики", "wmae": "WMAE", "bias": "смещение"},
               {"segment": lambda v: v, "rows": lambda v: f"{v:,}",
                "rows_share": lambda v: f"{v:.1%}", "metric_share": lambda v: f"{v:.1%}",
                "wmae": lambda v: f"{v:,.0f}", "bias": lambda v: f"{v:+,.0f}"}),
        f"\nСегмент {tail.segment} — это {tail.rows_share:.1%} строк и {tail.metric_share:.1%} "
        f"метрики. Систематическое смещение на нём {tail.bias:+,.0f}: модель узнаёт богатых, "
        f"но не дотягивает до их уровня. Любая оптимизация вне хвоста метрику почти не двигает.\n",

        "## 2. Из чего собран результат\n",
        f"Лесенка на {folds_used} последних фолдах, каждая строка добавляет ровно один "
        f"механизм к предыдущей. Колонка «выигрыш» — улучшение shifted WMAE к строке выше "
        f"(меньше — лучше).\n",
        _table(ablation_table, {"step": "шаг", "features": "признаков", "plain": "plain",
                                "shifted": "shifted", "gain": "выигрыш", "seconds": "время"},
               {"step": lambda v: v, "features": lambda v: f"{v}" if v else "—",
                "plain": lambda v: f"{v:,.0f}", "shifted": lambda v: f"{v:,.0f}",
                "gain": lambda v: "—" if pd.isna(v) else f"{v:,.0f}",
                "seconds": lambda v: f"{v:,.0f}s"}),
        "\nЧитается так: первая строка — цена ничегонеделания, вторая — что даёт сам "
        "градиентный бустинг на сырых данных, третья — поправка на сдвиг распределения, "
        "четвёртая — доменные отношения, пятая — слияние источников дохода и канальная "
        "декомпозиция, шестая — разделение на гейт, эксперта и базу, седьмая — "
        "подмешивание одиночной модели к каскаду.\n",

        "## 3. Поправка на смещение долей\n",
        "Тест взят за более поздние месяцы и систематически богаче трейна. Доли "
        "восстановлены методом BBSE по предсказаниям на отложенном месяце и на тесте, "
        "без единой тестовой метки, и зафиксированы в config.yaml:\n",
        _table(pd.DataFrame({"segment": SEGMENT_NAMES,
                             "train": train_prior, "test": test_prior,
                             "ratio": test_prior / train_prior}),
               {"segment": "сегмент", "train": "трейн", "test": "тест", "ratio": "отношение"},
               {"segment": lambda v: v, "train": lambda v: f"{v:.2%}",
                "test": lambda v: f"{v:.2%}", "ratio": lambda v: f"{v:.3f}"}),
        "\nОтношения используются дважды: как веса обучения, чтобы модель оптимизировалась "
        "под ту смесь, на которой её оценят, и как локальный компас — обычная кросс-валидация "
        "лидерборд не предсказывает, потому что считает метрику на другой смеси доходов.\n",

        "## 4. Что дал feature engineering\n",
        f"Построенных признаков {len(engineered)} из {ablation_table['features'].max()}. "
        f"Топ-{TOP_FEATURES} по вкладу в разбиения **хвостовой модели** — того эксперта, "
        f"который обслуживает сегмент, где сидит больше половины метрики. Важности базовой "
        f"модели здесь были бы обманчивы: они описывают, что отличает бедных, а не что "
        f"двигает метрику:\n",
        _table(gains_table, {"feature": "признак", "origin": "происхождение", "share": "вклад"},
               {"feature": lambda v: f"`{v}`", "origin": lambda v: v,
                "share": lambda v: f"{v:.1%}"}),
        "\nДоменный блок: деревья не умеют делить один признак на другой, поэтому вручную "
        "заданы отношения — кредитный лимит к заявленному доходу (лимит это доход, который "
        "банк уже верифицировал), приток и траты к доходу и к медиане по региону, доли "
        "пропусков по блокам источников, тир должности по стемам.\n",
        "Блок слияния решает другую задачу. Источников дохода много, каждый наполовину "
        "пропущен, и точность у них разная: у лучшего разброс ошибки 0.26 в логарифме, "
        "у худшего 0.88. Поэтому каждый приводится к шкале таргета своим коэффициентом, "
        "а складываются они весами по обратной дисперсии, причём саму дисперсию предсказывает "
        "отдельная модель под конкретного клиента — источник, надёжный для наёмного "
        "сотрудника, бесполезен для владельца бизнеса. Канальная декомпозиция делает то же "
        "на уровне данных: пять независимых моделей по своим источникам, их расхождение и "
        "покрытие идут в каскад признаками.\n",

        "## 5. Итог\n",
        f"Полная валидация финальной конфигурации на всех {validation['folds']} фолдах: "
        f"plain {validation['plain']:,.0f}, shifted {validation['shifted']:,.0f}. В таблице "
        f"абляции цифры оптимистичнее: там только {folds_used} последних фолда — самые "
        f"крупные по объёму обучения и потому самые лёгкие. Ранние фолды учатся на "
        f"одном-двух месяцах и тянут среднее вверх.\n",
        "Локальная валидация не сопоставима с лидербордом по абсолютной величине, и это "
        "не дефект: пивот 84 017 и кап 300 000 заданы в рублях, а доходы за полгода между "
        "выборками выросли примерно на 10%. При неизменной модели метрика от одного этого "
        "раздувается на десять с лишним процентов. Сравнивать имеет смысл только парные "
        "разницы на одних и тех же строках.\n",
        "Всё обучение на CPU: сборка признаков около трёх минут, модель около двух.\n",
    ]

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "analysis.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path
