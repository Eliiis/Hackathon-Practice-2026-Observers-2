import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import REPORTS
from .metric import wmae

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
GREEN = "#008300"
MAGENTA = "#e87ba4"
YELLOW = "#eda100"
RED = "#e34948"

FIGURES = REPORTS / "figures"
DPI = 200

LEADERBOARD = [
    ("каскад без объединения", 78059.75),
    ("+ калибровка источников", 76827.41),
    ("+ контекстное рассеяние", 76675.17),
    ("+ смешивание", 76572.11),
    ("+ поканальные оценки", 75910.88),
]


def canvas(width, height):
    figure, axes = plt.subplots(figsize=(width, height), dpi=DPI)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=MUTED, labelsize=8, length=0)
    return figure, axes


def label(axes, title, subtitle):
    axes.set_title(title, color=INK, fontsize=12, fontweight="600", loc="left", pad=18)
    axes.text(0, 1.015, subtitle, transform=axes.transAxes, color=SECONDARY,
              fontsize=8.5, va="bottom")


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.28)
    plt.close(figure)
    return path


def segments(breakdown):
    figure, axes = canvas(7.4, 4.0)
    names = list(breakdown["segment"])
    position = np.arange(len(names))
    rows = breakdown["rows_share"].to_numpy() * 100
    metric = breakdown["metric_share"].to_numpy() * 100

    axes.barh(position + 0.20, rows, height=0.34, color=BLUE, label="доля наблюдений")
    axes.barh(position - 0.20, metric, height=0.34, color=GREEN, label="доля оценки")
    for y, value in zip(position + 0.20, rows):
        axes.text(value + 0.8, y, f"{value:.1f}%", va="center", fontsize=8, color=SECONDARY)
    for y, value in zip(position - 0.20, metric):
        axes.text(value + 0.8, y, f"{value:.1f}%", va="center", fontsize=8, color=SECONDARY)

    axes.set_yticks(position, names)
    axes.set_xlim(0, max(metric.max(), rows.max()) * 1.18)
    axes.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY, loc="lower right")
    label(axes, "Оценка качества сосредоточена в верхнем хвосте",
          "весовая функция обращается в ноль у клиента с доходом 84 017")
    return save(figure, "segments.png")


def bias(breakdown):
    figure, axes = canvas(7.4, 3.6)
    names = list(breakdown["segment"])
    values = breakdown["bias"].to_numpy() / 1000
    position = np.arange(len(names))
    colours = [BLUE if v >= 0 else RED for v in values]

    axes.barh(position, values, height=0.5, color=colours)
    for y, value in zip(position, values):
        offset = 2.5 if value >= 0 else -2.5
        axes.text(value + offset, y, f"{value:+,.0f}", va="center",
                  ha="left" if value >= 0 else "right", fontsize=8, color=SECONDARY)

    axes.set_yticks(position, names)
    axes.axvline(0, color=BASELINE, linewidth=1.0)
    span = max(abs(values.min()), abs(values.max())) * 1.30
    axes.set_xlim(-span, span)
    axes.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    axes.set_xlabel("тысяч рублей", color=MUTED, fontsize=8.5)
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    label(axes, "Систематическое смещение прогноза по сегментам",
          "в верхнем сегменте модель узнаёт состоятельных клиентов, но не дотягивает до их уровня")
    return save(figure, "bias.png")


def ladder(ablation_table):
    steps = ablation_table.iloc[2:].copy()
    figure, axes = canvas(7.4, 3.8)
    names = [s.removeprefix("+ ") for s in steps["step"]]
    values = steps["gain"].to_numpy()
    position = np.arange(len(names))[::-1]

    axes.barh(position, values, height=0.5, color=BLUE)
    for y, value in zip(position, values):
        axes.text(value + values.max() * 0.02, y, f"{value:,.0f}", va="center",
                  fontsize=8.5, color=SECONDARY)

    axes.set_yticks(position, names)
    axes.set_xlim(0, values.max() * 1.16)
    axes.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    axes.set_xlabel("улучшение оценки, единиц", color=MUTED, fontsize=8.5)
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    base = ablation_table.iloc[1]
    label(axes, "Вклад механизмов, добавляемых последовательно",
          f"градиентный бустинг на сырых признаках даёт {base['gain']:,.0f} до этих шагов")
    return save(figure, "ladder.png")


def leaderboard():
    figure, axes = canvas(7.4, 3.8)
    names = [n for n, _ in LEADERBOARD]
    values = [v for _, v in LEADERBOARD]
    position = np.arange(len(values))

    axes.plot(position, values, color=BLUE, linewidth=2.0, marker="o",
              markersize=8, markerfacecolor=BLUE, markeredgecolor=SURFACE,
              markeredgewidth=2.0)
    for x, value in zip(position, values):
        axes.text(x, value + 190, f"{value:,.2f}", ha="center", fontsize=8.5, color=SECONDARY)

    axes.set_xticks(position, names, fontsize=8)
    plt.setp(axes.get_xticklabels(), rotation=18, ha="right")
    axes.set_ylim(min(values) - 500, max(values) + 700)
    axes.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    axes.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.set_ylabel("публичная оценка", color=MUTED, fontsize=8.5)
    label(axes, "Публичная оценка по мере внедрения механизмов",
          f"меньше — лучше; суммарное улучшение {values[0] - values[-1]:,.0f} единиц")
    return save(figure, "leaderboard.png")


def importance(gains_table):
    figure, axes = canvas(7.4, 4.6)
    origins = {"слияние": BLUE, "доменный": GREEN, "сырой": MAGENTA}
    table = gains_table.iloc[::-1]
    position = np.arange(len(table))
    values = table["share"].to_numpy() * 100
    colours = [origins.get(o, MUTED) for o in table["origin"]]

    axes.barh(position, values, height=0.62, color=colours)
    for y, value in zip(position, values):
        axes.text(value + values.max() * 0.02, y, f"{value:.1f}%", va="center",
                  fontsize=8, color=SECONDARY)

    axes.set_yticks(position, table["feature"], fontsize=8)
    axes.set_xlim(0, values.max() * 1.14)
    axes.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in origins.values()]
    axes.legend(handles, origins.keys(), frameon=False, fontsize=8.5,
                labelcolor=SECONDARY, loc="lower right")
    share = (table["origin"] == "слияние").mean() * 100
    label(axes, "Значимость признаков для модели верхнего сегмента",
          f"на блок объединения источников приходится {share:.0f}% верхних позиций")
    return save(figure, "importance.png")


def folds(out_of_fold, y, weight, blocks):
    figure, axes = canvas(7.4, 3.8)
    scores = [wmae(y[valid], out_of_fold[valid], weight[valid]) for _, valid in blocks]
    sizes = [len(train) for train, _ in blocks]
    position = np.arange(len(scores))

    axes.bar(position, scores, width=0.52, color=BLUE)
    for x, (score, size) in enumerate(zip(scores, sizes)):
        axes.text(x, score + 900, f"{score:,.0f}", ha="center", fontsize=8.5, color=SECONDARY)
        axes.text(x, 1600, f"{size:,}", ha="center", fontsize=7.5, color=SURFACE)

    axes.set_xticks(position, [f"блок {i + 1}" for i in position])
    axes.set_ylim(0, max(scores) * 1.16)
    axes.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    axes.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.set_ylabel("WMAE", color=MUTED, fontsize=8.5)
    label(axes, "Проверка с соблюдением хронологии",
          "белым внутри столбца — объём обучающей выборки блока")
    return save(figure, "folds.png")


def draw(breakdown, ablation_table, gains_table, out_of_fold, y, weight, blocks):
    return [segments(breakdown), bias(breakdown), ladder(ablation_table),
            leaderboard(), importance(gains_table),
            folds(out_of_fold, y, weight, blocks)]
