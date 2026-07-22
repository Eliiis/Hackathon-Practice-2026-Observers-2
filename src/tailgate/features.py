import numpy as np
import pandas as pd
import xgboost as xgb

from .config import DATA
from .model import IncomeModel

INCOME_SOURCES = [
    "salary_6to12m_avg", "dp_ils_avg_salary_1y", "dp_ils_paymentssum_avg_12m",
    "dp_payoutincomedata_payout_avg_6_month", "incomeValue",
]

PROXY_SOURCES = [
    "salary_6to12m_avg", "dp_payoutincomedata_payout_avg_6_month",
    "dp_payoutincomedata_payout_max_6_month", "dp_payoutincomedata_payout_avg_3_month",
    "dp_ils_paymentssum_month_avg", "dp_payoutincomedata_payout_avg_prev_year",
    "incomeValue", "dp_ils_paymentssum_avg_12m", "dp_ils_avg_salary_1y",
    "dp_ils_avg_salary_2y", "dp_ils_accpayment_avg_12m", "dp_ils_paymentssum_avg_6m",
    "dp_ils_ipkcurrentyear_currentyearpensfactor", "incomeValueCategory",
]

PROXY_TURNOVER = ["turn_cur_cr_avg_act_v2", "turn_cur_cr_avg_v2", "avg_credit_turn_rur"]

CHANNELS = {
    "official": ("dp_ils_", "dp_payout", "salary_", "incomeValue"),
    "bki": ("hdb_", "bki_", "loan", "vert_pil"),
    "flow": ("turn_", "avg_balance", "curr_rur", "dda_", "total_rur", "avg_cur", "avg_credit",
             "avg_debet", "diff_avg"),
    "tx": ("by_category__", "avg_by_category__", "amount_by_category", "avg_3m_", "avg_6m_",
           "summarur", "avg_amount_daily"),
    "behav": ("vert_", "sms", "cntVoice", "mob_", "device", "winback", "cntRegion", "period_"),
}

CHANNEL_CORE = ["gender", "age", "adminarea", "addrref", "per_capita_income_rur_amt",
                "salary_median_in_gex_r1", "incomeValueCategory"]

CHANNEL_TREES = 150
MIN_CHANNEL_ROWS = 3000
MIN_PROXY_ROWS = 500
MIN_SIGMA_ROWS = 400
SIGMA_FOLDS = 3
SIGMA_CONTEXT = 120
SIGMA_FLOOR, SIGMA_CEIL = 0.08, 2.0
SIGMA_PARAMS = dict(n_estimators=80, max_depth=4, learning_rate=0.15, subsample=0.8,
                    colsample_bytree=0.3, tree_method="hist", device="cpu",
                    enable_categorical=True, n_jobs=-1, verbosity=0, random_state=42)

DEAD_IN_TEST = ["first_salary_income"]

TIME_SHORTCUTS = [
    "dp_ils_days_from_last_doc", "days_to_last_transaction", "tz_msk_timedelta",
    "winback_cnt", "days_after_last_request", "vert_ghost_close_dpay3_last_days",
    "dp_ils_days_multiple_job_share_2y", "hdb_bki_total_pil_last_days",
    "hdb_bki_last_product_days", "dp_ils_days_multiple_job_cnt_5y", "period_last_act_ad",
]

POSITION_COLUMN = "dp_ewb_last_employment_position"

POSITION_TIERS = [
    ("position_worker", 1.0, ["водител", "слесар", "монтаж", "оператор", "продавец",
                              "кассир", "охран", "уборщ", "грузчик", "рабоч", "повар",
                              "официант", "кладовщик", "комплектовщ", "курьер"]),
    ("position_professional", 2.0, ["врач", "юрист", "бухгалтер", "экономист",
                                    "преподавател", "учител"]),
    ("position_engineer", 3.0, ["инженер", "программист", "разработ", "аналитик",
                                "архитектор"]),
    ("position_senior", 3.0, ["ведущ", "старш"]),
    ("position_chief", 4.0, ["главн"]),
    ("position_management", 5.0, ["руководител", "начальник", "заведующ", "заместител"]),
    ("position_executive", 6.0, ["генеральн", "директор", "председатель", "управляющ",
                                 "учредител", "владел"]),
]

MISSINGNESS_BLOCKS = ["dp_ils_", "dp_payout", "hdb_bki_", "turn_", "by_category"]

LIMIT_COLUMNS = [
    "hdb_bki_total_max_limit", "hdb_bki_total_cc_max_limit", "hdb_bki_active_cc_max_limit",
    "hdb_bki_total_pil_max_limit", "hdb_bki_active_pil_max_limit",
    "bki_total_max_limit", "bki_total_il_max_limit",
]

LUXURY_MARKERS = ("oteli", "hotel", "restaurant", "puteshestv", "zarubezh")

MAX_CATEGORY_LEVELS = 200


def column(frame, name):
    if name in frame.columns:
        return frame[name]
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def columns(frame, names):
    return pd.concat([column(frame, n) for n in names], axis=1)


def ratio(numerator, denominator):
    return numerator / denominator.where(denominator > 0)


def declared_income(frame):
    return columns(frame, INCOME_SOURCES).max(1)


def income_evidence(frame):
    sources = columns(frame, INCOME_SOURCES)
    return pd.DataFrame({
        "income_signal_count": sources.notna().sum(1).astype("float64"),
    }, index=frame.index)


def proxy_constants(frame, target):
    target = np.asarray(target, float)
    constants = {}
    for name in PROXY_SOURCES:
        if name not in frame.columns:
            continue
        values = frame[name].to_numpy(dtype="float64")
        present = np.isfinite(values) & (values > 0)
        if present.sum() < MIN_PROXY_ROWS:
            continue
        residual = np.log(target[present] / values[present])
        constants[name] = (float(np.exp(np.median(residual))), float(np.std(residual)))
    return dict(sorted(constants.items(), key=lambda kv: kv[1][1]))


def sigma_context(frame, numeric, categorical):
    chosen = [c for c in numeric if c in frame.columns][:SIGMA_CONTEXT]
    chosen += [c for c in categorical if c in frame.columns]
    return frame[chosen], [c for c in categorical if c in frame.columns]


def fit_proxy_sigma(frame, target, constants, numeric, categorical):
    from .model import pin_categories

    target = np.asarray(target, float)
    context, context_categorical = sigma_context(frame, numeric, categorical)
    fitted, cross = {}, {}
    for name, (scale, sigma) in constants.items():
        values = frame[name].to_numpy(dtype="float64")
        present = np.where(np.isfinite(values) & (values > 0))[0]
        if len(present) < MIN_SIGMA_ROWS:
            continue
        error = np.abs(np.log(target[present] / (values[present] * scale)))
        held = np.full(len(frame), np.nan)
        order = np.random.default_rng(42).permutation(len(present))
        for fold in range(SIGMA_FOLDS):
            out = order[fold::SIGMA_FOLDS]
            rest = np.setdiff1d(order, out)
            pinned, levels = pin_categories(context.iloc[present[rest]], context_categorical)
            model = xgb.XGBRegressor(**SIGMA_PARAMS).fit(pinned, error[rest])
            scored, _ = pin_categories(context.iloc[present[out]], context_categorical, levels)
            held[present[out]] = model.predict(scored)
        pinned, levels = pin_categories(context.iloc[present], context_categorical)
        fitted[name] = (xgb.XGBRegressor(**SIGMA_PARAMS).fit(pinned, error), levels)
        cross[name] = np.clip(held, SIGMA_FLOOR, SIGMA_CEIL)
    return fitted, cross


def apply_proxy_sigma(frame, fitted, numeric, categorical):
    from .model import pin_categories

    context, context_categorical = sigma_context(frame, numeric, categorical)
    out = {}
    for name, (model, levels) in fitted.items():
        scored, _ = pin_categories(context, context_categorical, levels)
        out[name] = np.clip(model.predict(scored), SIGMA_FLOOR, SIGMA_CEIL)
    return out


def income_pool(frame, constants, sigma=None):
    estimates, precisions = [], []
    for name, (scale, fallback) in constants.items():
        values = frame[name].to_numpy(dtype="float64")
        ok = np.isfinite(values) & (values > 0)
        estimates.append(np.where(ok, np.log1p(np.where(ok, values, 1.0) * scale), np.nan))
        spread = fallback if sigma is None or name not in sigma else sigma[name]
        spread = np.where(np.isfinite(spread), spread, fallback)
        precisions.append(np.where(ok, 1.0 / np.asarray(spread, float) ** 2, 0.0))

    estimates, precisions = np.array(estimates), np.array(precisions)
    available = precisions > 0
    any_source = available.any(axis=0)
    total = precisions.sum(axis=0)
    index = available.argmax(axis=0)
    position = np.arange(estimates.shape[1])

    pooled = np.where(total > 0,
                      np.nansum(np.nan_to_num(estimates) * precisions, axis=0)
                      / np.where(total > 0, total, 1.0), np.nan)
    best = np.where(any_source, estimates[index, position], np.nan)
    rank = np.where(any_source, index, np.nan)
    with np.errstate(invalid="ignore"):
        disagreement = np.where(available.sum(axis=0) > 1, np.nanstd(estimates, axis=0), np.nan)

    built = {
        "income_best": np.expm1(best),
        "income_best_rank": rank,
        "income_best_precise": np.where(rank == 0, np.expm1(best), np.nan),
        "income_pooled": np.expm1(pooled),
        "income_pooled_log": pooled,
        "income_precision": total,
        "income_sigma_best": np.where(
            any_source, np.reciprocal(np.sqrt(np.where(precisions > 0, precisions, np.nan)))[
                index, position], np.nan),
        "income_sources_n": available.sum(axis=0).astype(float),
        "income_disagreement": disagreement,
    }
    for name in PROXY_TURNOVER:
        if name in frame.columns:
            turnover = frame[name].to_numpy(dtype="float64")
            built[f"income_to_{name}"] = np.expm1(pooled) / np.where(turnover > 0, turnover, np.nan)
    return pd.DataFrame(built, index=frame.index)


def coverage_pattern(frame, numeric):
    out = {}
    for prefix in MISSINGNESS_BLOCKS:
        block = [c for c in frame.columns if c.startswith(prefix)]
        if block:
            out[f"missing_{prefix.strip('_')}"] = frame[block].isna().mean(1)
    if numeric:
        out["missing_overall"] = frame[numeric].isna().mean(1)
    return pd.DataFrame(out, index=frame.index)


def salary_trajectory(frame):
    salary = columns(frame, [f"dp_ils_avg_salary_{h}" for h in ("1y", "2y", "3y")])
    return pd.DataFrame({
        "salary_trend": column(frame, "dp_ils_avg_salary_1y") / column(frame, "dp_ils_avg_salary_3y"),
        "salary_volatility": salary.std(1) / (salary.mean(1).abs() + 1),
        "payment_acceleration": (column(frame, "dp_ils_accpayment_avg_3m")
                                 / column(frame, "dp_ils_accpayment_avg_12m")),
    }, index=frame.index)


def cash_flow(frame):
    credit = column(frame, "turn_cur_cr_sum_v2")
    debit = column(frame, "turn_cur_db_sum_v2")
    declared = declared_income(frame)
    out = {
        "flow_net": credit - debit,
        "flow_credit_to_debit": credit / (debit.abs() + 1),
        "flow_turnover_to_income": credit / (declared + 1),
        "savings_to_income": column(frame, "curr_rur_amt_cm_avg") / (declared + 1),
    }
    transactions = [c for c in frame.columns
                    if c.startswith(("by_category__", "avg_by_category__"))
                    and pd.api.types.is_numeric_dtype(frame[c])]
    if transactions:
        block = frame[transactions]
        out["transaction_volume"] = block.abs().sum(1, min_count=1)
        out["transaction_variety"] = (block.fillna(0) != 0).sum(1).astype("float64")
    return pd.DataFrame(out, index=frame.index)


def credit_capacity(frame):
    declared = declared_income(frame)
    granted = columns(frame, LIMIT_COLUMNS).max(1)
    all_limits = [c for c in frame.columns
                  if "bki" in c.lower() and "limit" in c and pd.api.types.is_numeric_dtype(frame[c])]
    total = frame[all_limits].sum(1, min_count=1) if all_limits else pd.Series(np.nan, index=frame.index)
    return pd.DataFrame({
        "limit_best": granted,
        "limit_total": total,
        "limit_mortgage": column(frame, "hdb_bki_total_ip_max_limit"),
        "limit_auto": column(frame, "hdb_bki_total_auto_max_limit"),
        "limit_to_income": ratio(granted, declared),
        "limit_utilisation": column(frame, "avg_balance_rur_amt_1m_af") / (total + 1),
        "loan_request_to_income": ratio(column(frame, "loan_cur_amt"), declared),
    }, index=frame.index)


def wealth_signals(frame):
    declared = declared_income(frame)
    region = column(frame, "salary_median_in_gex_r1").fillna(column(frame, "per_capita_income_rur_amt"))
    inflow = columns(frame, ["turn_cur_cr_avg_v2", "avg_credit_turn_rur"]).max(1)
    spend = pd.concat([column(frame, "summarur_1m_purch"),
                       column(frame, "avg_amount_daily_transactions_90d") * 30], axis=1).max(1)
    luxury_columns = [c for c in frame.columns
                      if any(k in c for k in LUXURY_MARKERS)
                      and pd.api.types.is_numeric_dtype(frame[c])]
    luxury = (frame[luxury_columns].abs().sum(1, min_count=1) if luxury_columns
              else pd.Series(np.nan, index=frame.index))
    return pd.DataFrame({
        "inflow_monthly": inflow,
        "inflow_to_income": ratio(inflow, declared),
        "inflow_to_region": ratio(inflow, region),
        "spend_monthly": spend,
        "spend_to_income": ratio(spend, declared),
        "luxury_spend": luxury,
        "luxury_share": ratio(luxury, spend),
        "own_funds_to_income": ratio(column(frame, "total_rur_amt_cm_avg"), declared),
        "income_to_region": ratio(declared, region),
        "limit_to_region": ratio(columns(frame, LIMIT_COLUMNS).max(1), region),
    }, index=frame.index)


def job_position(frame):
    if POSITION_COLUMN not in frame.columns:
        return pd.DataFrame(index=frame.index)
    known = frame[POSITION_COLUMN].notna()
    title = frame[POSITION_COLUMN].astype(str).str.lower().where(known, "")
    out = {"position_known": known.astype(float)}
    tier = pd.Series(np.nan, index=frame.index)
    tier[known] = 0.0
    for name, rank, stems in POSITION_TIERS:
        matched = title.str.contains("|".join(stems), regex=True)
        out[name] = matched.astype(float)
        tier = tier.where(~matched, np.maximum(tier.fillna(0.0), rank))
    out["position_tier"] = tier
    return pd.DataFrame(out, index=frame.index)


FUSION_PREFIXES = ("arena_", "income_best", "income_pooled", "income_precision",
                   "income_sigma", "income_sources", "income_disagreement",
                   "income_to_turn", "income_to_avg")


def is_fusion(name):
    return name.startswith(FUSION_PREFIXES)


def channel_layout(columns):
    layout = {}
    for name, prefixes in CHANNELS.items():
        picked = [c for c in columns if any(c.startswith(p) or p in c for p in prefixes)]
        if picked:
            core = [c for c in CHANNEL_CORE if c in columns and c not in picked]
            layout[name] = picked + core
    return layout


def fit_channels(frame, target, months, categorical, layout):
    target = np.asarray(target, float)
    months = np.asarray(months)
    fitted, estimates, coverage = {}, {}, {}
    for name, columns in layout.items():
        estimate = np.full(len(frame), np.nan)
        for held in np.unique(months):
            inner, outer = months != held, months == held
            if inner.sum() < MIN_CHANNEL_ROWS or not outer.any():
                continue
            model = IncomeModel(categorical, 42, CHANNEL_TREES).fit(
                frame[columns][inner], target[inner], frame[DATA["weight"]][inner])
            estimate[outer] = model.predict(frame[columns][outer])
        fitted[name] = IncomeModel(categorical, 42, CHANNEL_TREES).fit(
            frame[columns], target, frame[DATA["weight"]])
        estimates[f"arena_{name}"] = estimate
        coverage[f"arena_cov_{name}"] = frame[columns].notna().mean(axis=1).to_numpy()
    return fitted, arena_block(frame, estimates, coverage)


def apply_channels(frame, fitted, layout):
    estimates, coverage = {}, {}
    for name, model in fitted.items():
        estimates[f"arena_{name}"] = model.predict(frame[layout[name]])
        coverage[f"arena_cov_{name}"] = frame[layout[name]].notna().mean(axis=1).to_numpy()
    return arena_block(frame, estimates, coverage)


def arena_block(frame, estimates, coverage):
    stacked = np.log1p(np.clip(np.array(list(estimates.values())), 1.0, None))
    built = dict(estimates)
    with np.errstate(invalid="ignore"):
        built["arena_spread"] = np.nanstd(stacked, axis=0)
        built["arena_range"] = np.nanmax(stacked, axis=0) - np.nanmin(stacked, axis=0)
        built["arena_mean"] = np.expm1(np.nanmean(stacked, axis=0))
        built["arena_max"] = np.expm1(np.nanmax(stacked, axis=0))
    built.update(coverage)
    return pd.DataFrame(built, index=frame.index).astype("float32")


def engineer(frame, numeric=None, keep=None, constants=None, sigma=None):
    blocks = [
        income_evidence(frame),
        income_pool(frame, constants, sigma) if constants else pd.DataFrame(index=frame.index),
        coverage_pattern(frame, numeric),
        salary_trajectory(frame),
        cash_flow(frame),
        credit_capacity(frame),
        wealth_signals(frame),
        job_position(frame),
    ]
    built = pd.concat([b for b in blocks if b.shape[1]], axis=1)
    built = built.loc[:, ~built.columns.duplicated()]
    built = built.replace([np.inf, -np.inf], np.nan).astype("float32")

    if keep is not None:
        for name in keep:
            if name not in built.columns:
                built[name] = np.float32("nan")
        return built[list(keep)]

    empty = [c for c in built.columns
             if built[c].isna().all() or built[c].nunique(dropna=True) <= 1]
    return built.drop(columns=empty)


def select(frame, schema):
    excluded = set(TIME_SHORTCUTS) | set(DEAD_IN_TEST)
    numeric = [c for c in schema.numeric if c not in excluded]
    categorical = [c for c in schema.categorical
                   if c not in excluded and c != schema.date
                   and frame[c].nunique() <= MAX_CATEGORY_LEVELS]
    known = set(schema.numeric) | set(schema.categorical) | {
        schema.id, schema.date, schema.target, schema.weight}
    engineered = [c for c in frame.columns if c not in known]
    return list(dict.fromkeys(numeric + categorical + engineered)), categorical
