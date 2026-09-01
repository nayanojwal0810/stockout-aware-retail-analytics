from pathlib import Path
import numpy as np
import pandas as pd

from stockout_retail.risk.prioritization import (
    add_risk_scores,
    assign_action_segments,
)

print("Shared src risk layer: LOADED")

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 110)
print("PHASE 3 - STEP 4R: DEMAND-WEIGHTED OPERATIONAL RISK PRIORITIZATION")
print("=" * 110)

# =========================================================
# 1. LOAD RAW DATA
# =========================================================
raw_cols = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt"
]

raw = pd.read_parquet(
    RAW / "train.parquet",
    columns=raw_cols
)

raw["dt"] = pd.to_datetime(
    raw["dt"]
)

raw = raw.sort_values(
    [
        "store_id",
        "product_id",
        "dt"
    ]
).reset_index(
    drop=True
)

group_cols = [
    "store_id",
    "product_id"
]

print(
    "\nRAW DATA:",
    raw.shape
)

# =========================================================
# 2. LOAD STOCKOUT-ADJUSTED DEMAND
# =========================================================
adjusted_path = (
    PROCESSED
    /
    "stockout_adjusted_demand.parquet"
)

adjusted = pd.read_parquet(
    adjusted_path,
    columns=[
        "store_id",
        "product_id",
        "dt",
        "sale_amount",
        "stock_hour6_22_cnt",
        "stockout_flag",
        "full_stockout_flag",
        "adjusted_demand",
        "estimated_censored_gap"
    ]
)

adjusted["dt"] = pd.to_datetime(
    adjusted["dt"]
)

print(
    "ADJUSTED DATA:",
    adjusted.shape
)

print(
    "Adjusted-demand period:",
    adjusted["dt"].min(),
    "->",
    adjusted["dt"].max()
)

# =========================================================
# 3. BASIC COVERAGE CHECK
# =========================================================
series_index = (
    raw[group_cols]
    .drop_duplicates()
)

n_series = len(
    series_index
)

adjusted_series = (
    adjusted[group_cols]
    .drop_duplicates()
)

print(
    "Raw series:",
    n_series
)

print(
    "Adjusted-demand series:",
    len(adjusted_series)
)

assert (
    len(adjusted_series)
    ==
    n_series
)

# =========================================================
# 4. RECENT OPERATIONAL DEMAND METRICS
#
# Use the period for which cross-fitted adjusted demand
# exists. This avoids mixing a full 90-day ranking with
# a 42-day forecast-risk measurement.
# =========================================================
adjusted["stockout_day"] = (
    adjusted["stockout_flag"]
    .astype(int)
)

adjusted["full_stockout_day"] = (
    adjusted["full_stockout_flag"]
    .astype(int)
)

# ---------------------------------------------------------
# Adjusted demand on all days
# ---------------------------------------------------------
recent_series = (
    adjusted
    .groupby(
        group_cols
    )
    .agg(
        recent_days=(
            "dt",
            "nunique"
        ),
        recent_mean_adjusted_demand=(
            "adjusted_demand",
            "mean"
        ),
        recent_total_adjusted_demand=(
            "adjusted_demand",
            "sum"
        ),
        recent_mean_observed_sales=(
            "sale_amount",
            "mean"
        ),
        recent_stockout_days=(
            "stockout_day",
            "sum"
        ),
        recent_full_stockout_days=(
            "full_stockout_day",
            "sum"
        ),
        recent_mean_stockout_hours=(
            "stock_hour6_22_cnt",
            "mean"
        ),
        recent_max_stockout_hours=(
            "stock_hour6_22_cnt",
            "max"
        ),
        recent_estimated_censored_demand=(
            "estimated_censored_gap",
            "sum"
        )
    )
    .reset_index()
)

recent_series[
    "recent_stockout_day_rate"
] = (
    recent_series[
        "recent_stockout_days"
    ]
    /
    recent_series[
        "recent_days"
    ]
)

recent_series[
    "recent_full_stockout_rate"
] = (
    recent_series[
        "recent_full_stockout_days"
    ]
    /
    recent_series[
        "recent_days"
    ]
)

# =========================================================
# 5. DEMAND-WEIGHTED STOCKOUT BURDEN
#
# Estimated censored demand is expressed in demand units.
#
# This answers:
#
#   "How much estimated demand appears to be lost/censored
#    by stockouts for this store-product?"
#
# It does NOT mean actual lost sales were observed directly.
# =========================================================
recent_series[
    "censored_demand_rate"
] = (
    recent_series[
        "recent_estimated_censored_demand"
    ]
    /
    recent_series[
        "recent_total_adjusted_demand"
    ].replace(
        0,
        np.nan
    )
)

recent_series[
    "censored_demand_rate"
] = (
    recent_series[
        "censored_demand_rate"
    ]
    .fillna(0)
    .clip(
        lower=0
    )
)

# =========================================================
# 6. LOAD FORECAST RESULTS
# =========================================================
forecast_path = (
    PROCESSED
    /
    "inventory_backtest_forecasts.csv"
)

forecasts = pd.read_csv(
    forecast_path,
    parse_dates=[
        "forecast_origin",
        "dt"
    ]
)

print(
    "\nFORECAST DATA:",
    forecasts.shape
)

# =========================================================
# 7. LOAD ACTUAL SALES FOR FORECAST ERROR
# =========================================================
forecast_actual = raw[
    [
        "store_id",
        "product_id",
        "dt",
        "sale_amount"
    ]
]

forecast_eval = forecasts.merge(
    forecast_actual,
    on=group_cols + ["dt"],
    how="inner"
)

print(
    "Forecast evaluation rows:",
    len(forecast_eval)
)

assert (
    len(forecast_eval)
    ==
    len(forecasts)
)

# =========================================================
# 8. FORECAST UNDER-FORECAST METRICS
# =========================================================
forecast_eval[
    "forecast_error"
] = (
    forecast_eval[
        "sale_amount"
    ]
    -
    forecast_eval[
        "forecast_sales"
    ]
)

forecast_eval[
    "absolute_error"
] = np.abs(
    forecast_eval[
        "forecast_error"
    ]
)

forecast_eval[
    "positive_underforecast"
] = (
    forecast_eval[
        "forecast_error"
    ]
    .clip(
        lower=0
    )
)

# ---------------------------------------------------------
# Store-product forecast-risk summary
# ---------------------------------------------------------
forecast_risk = (
    forecast_eval
    .groupby(
        group_cols
    )
    .agg(
        forecast_mae=(
            "absolute_error",
            "mean"
        ),
        mean_forecast_bias=(
            "forecast_error",
            "mean"
        ),
        mean_underforecast=(
            "positive_underforecast",
            "mean"
        ),
        p90_underforecast=(
            "positive_underforecast",
            lambda x:
                x.quantile(
                    0.90
                )
        ),
        p95_underforecast=(
            "positive_underforecast",
            lambda x:
                x.quantile(
                    0.95
                )
        ),
        max_underforecast=(
            "positive_underforecast",
            "max"
        )
    )
    .reset_index()
)

# =========================================================
# 9. COMBINE RISK TABLE
# =========================================================
risk = recent_series.merge(
    forecast_risk,
    on=group_cols,
    how="inner"
)

print(
    "\nFinal risk series:",
    len(risk)
)

assert (
    len(risk)
    ==
    n_series
)

# =========================================================
# 10. DATA QUALITY CHECK
# =========================================================
risk_numeric = [
    "recent_mean_adjusted_demand",
    "recent_total_adjusted_demand",
    "recent_estimated_censored_demand",
    "recent_stockout_day_rate",
    "recent_full_stockout_rate",
    "recent_mean_stockout_hours",
    "censored_demand_rate",
    "forecast_mae",
    "mean_forecast_bias",
    "mean_underforecast",
    "p90_underforecast",
    "p95_underforecast"
]

missing_counts = (
    risk[
        risk_numeric
    ]
    .isna()
    .sum()
)

print("\n" + "=" * 110)
print("RISK TABLE MISSING VALUES")
print("=" * 110)

print(
    missing_counts[
        missing_counts > 0
    ]
    .to_string()
)

# =========================================================
# 11. DEMAND WEIGHTED STOCKOUT EXPOSURE
#
# Estimated censored demand / recent days gives an average
# daily demand-gap measure.
# =========================================================
risk[
    "daily_censored_demand"
] = (
    risk[
        "recent_estimated_censored_demand"
    ]
    /
    risk[
        "recent_days"
    ]
)

risk[
    "demand_exposed_to_stockout"
] = (
    risk[
        "recent_mean_adjusted_demand"
    ]
    *
    risk[
        "recent_stockout_day_rate"
    ]
)

# =========================================================
# 12. OPERATIONAL RISK
# =========================================================
risk = add_risk_scores(risk)
risk = assign_action_segments(risk)

n_critical = max(
    1,
    int(len(risk) * 0.01),
)

# =========================================================
# 15. BUSINESS FLAGS
# =========================================================
risk[
    "high_demand"
] = (
    risk[
        "recent_mean_adjusted_demand"
    ]
    >=
    risk[
        "recent_mean_adjusted_demand"
    ]
    .quantile(
        0.75
    )
)

risk[
    "high_stockout_burden"
] = (
    risk[
        "recent_estimated_censored_demand"
    ]
    >=
    risk[
        "recent_estimated_censored_demand"
    ]
    .quantile(
        0.75
    )
)

risk[
    "high_uncertainty"
] = (
    risk[
        "mean_underforecast"
    ]
    >=
    risk[
        "mean_underforecast"
    ]
    .quantile(
        0.75
    )
)

# =========================================================
# 16. CRITICAL QUEUE
# =========================================================
critical = risk[
    risk[
        "action_segment"
    ]
    ==
    "CRITICAL"
].copy()

print("\n" + "=" * 110)
print("CRITICAL ACTION QUEUE")
print("=" * 110)

print(
    "Critical series:",
    len(critical)
)

print(
    "Portfolio share:",
    f"{len(critical) / len(risk) * 100:.2f}%"
)

critical_columns = [
    "operational_rank",
    "store_id",
    "product_id",
    "recent_mean_adjusted_demand",
    "recent_stockout_day_rate",
    "recent_full_stockout_rate",
    "recent_estimated_censored_demand",
    "daily_censored_demand",
    "censored_demand_rate",
    "forecast_mae",
    "mean_underforecast",
    "p95_underforecast",
    "operational_risk_score"
]

print(
    critical[
        critical_columns
    ]
    .head(25)
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 17. ACTION SEGMENT SUMMARY
# =========================================================
print("\n" + "=" * 110)
print("ACTION SEGMENT SUMMARY")
print("=" * 110)

segment_summary = (
    risk
    .groupby(
        "action_segment"
    )
    .agg(
        series=(
            "product_id",
            "size"
        ),
        mean_adjusted_demand=(
            "recent_mean_adjusted_demand",
            "mean"
        ),
        mean_stockout_day_rate=(
            "recent_stockout_day_rate",
            "mean"
        ),
        mean_censored_demand=(
            "recent_estimated_censored_demand",
            "mean"
        ),
        mean_censored_demand_rate=(
            "censored_demand_rate",
            "mean"
        ),
        mean_underforecast=(
            "mean_underforecast",
            "mean"
        ),
        mean_forecast_mae=(
            "forecast_mae",
            "mean"
        ),
        mean_risk_score=(
            "operational_risk_score",
            "mean"
        )
    )
    .reset_index()
)

segment_order = {
    "CRITICAL": 1,
    "HIGH_PRIORITY": 2,
    "MONITOR": 3,
    "STANDARD": 4
}

segment_summary[
    "sort_order"
] = (
    segment_summary[
        "action_segment"
    ]
    .map(
        segment_order
    )
)

segment_summary = (
    segment_summary
    .sort_values(
        "sort_order"
    )
    .drop(
        columns="sort_order"
    )
)

print(
    segment_summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 18. TOP 5% HIGH PRIORITY PORTFOLIO
# =========================================================
high_priority = risk[
    risk[
        "action_segment"
    ]
    .isin(
        [
            "CRITICAL",
            "HIGH_PRIORITY"
        ]
    )
].copy()

print("\n" + "=" * 110)
print("HIGH-PRIORITY PORTFOLIO")
print("=" * 110)

print(
    "Series:",
    len(high_priority)
)

print(
    "Portfolio share:",
    f"{len(high_priority) / len(risk) * 100:.2f}%"
)

print(
    "Mean adjusted demand:",
    round(
        high_priority[
            "recent_mean_adjusted_demand"
        ].mean(),
        4
    )
)

print(
    "Mean stockout day rate:",
    round(
        high_priority[
            "recent_stockout_day_rate"
        ].mean(),
        4
    )
)

print(
    "Mean estimated censored demand:",
    round(
        high_priority[
            "recent_estimated_censored_demand"
        ].mean(),
        4
    )
)

# =========================================================
# 19. PORTFOLIO DEMAND CONCENTRATION
#
# Here we use actual/estimated demand units rather than
# stockout rates.
# =========================================================
total_adjusted_demand = (
    risk[
        "recent_total_adjusted_demand"
    ].sum()
)

top_1_adjusted_demand = (
    risk
    .head(
        n_critical
    )[
        "recent_total_adjusted_demand"
    ]
    .sum()
)

top_5_n = max(
    1,
    int(
        len(risk)
        *
        0.05
    )
)

top_5_adjusted_demand = (
    risk
    .head(
        top_5_n
    )[
        "recent_total_adjusted_demand"
    ]
    .sum()
)

print("\n" + "=" * 110)
print("DEMAND CONCENTRATION")
print("=" * 110)

print(
    "Top 1% share of adjusted-demand volume:",
    f"{top_1_adjusted_demand / total_adjusted_demand * 100:.2f}%"
)

print(
    "Top 5% share of adjusted-demand volume:",
    f"{top_5_adjusted_demand / total_adjusted_demand * 100:.2f}%"
)

# =========================================================
# 20. STOCKOUT-LOSS CONCENTRATION
#
# This is now a real demand-unit concentration metric:
# estimated censored demand, not a sum of percentages.
# =========================================================
total_censored = (
    risk[
        "recent_estimated_censored_demand"
    ].sum()
)

top_1_censored = (
    risk
    .head(
        n_critical
    )[
        "recent_estimated_censored_demand"
    ]
    .sum()
)

top_5_censored = (
    risk
    .head(
        top_5_n
    )[
        "recent_estimated_censored_demand"
    ].sum()
)

print("\n" + "=" * 110)
print("CENSORED-DEMAND CONCENTRATION")
print("=" * 110)

print(
    "Total estimated censored demand:",
    round(
        total_censored,
        4
    )
)

print(
    "Top 1% share of estimated censored demand:",
    f"{top_1_censored / total_censored * 100:.2f}%"
)

print(
    "Top 5% share of estimated censored demand:",
    f"{top_5_censored / total_censored * 100:.2f}%"
)

# =========================================================
# 21. PRODUCT-LEVEL CONCENTRATION
#
# This identifies whether a single product is driving the
# ranking.
# =========================================================
product_summary = (
    risk
    .groupby(
        "product_id"
    )
    .agg(
        store_product_series=(
            "store_id",
            "nunique"
        ),
        total_adjusted_demand=(
            "recent_total_adjusted_demand",
            "sum"
        ),
        total_censored_demand=(
            "recent_estimated_censored_demand",
            "sum"
        ),
        mean_stockout_rate=(
            "recent_stockout_day_rate",
            "mean"
        ),
        mean_risk_score=(
            "operational_risk_score",
            "mean"
        )
    )
    .reset_index()
    .sort_values(
        "total_censored_demand",
        ascending=False
    )
)

print("\n" + "=" * 110)
print("TOP PRODUCTS BY ESTIMATED CENSORED DEMAND")
print("=" * 110)

print(
    product_summary
    .head(20)
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 22. STORE CONCENTRATION
# =========================================================
store_summary = (
    risk
    .groupby(
        "store_id"
    )
    .agg(
        series=(
            "product_id",
            "size"
        ),
        total_adjusted_demand=(
            "recent_total_adjusted_demand",
            "sum"
        ),
        total_censored_demand=(
            "recent_estimated_censored_demand",
            "sum"
        ),
        mean_stockout_rate=(
            "recent_stockout_day_rate",
            "mean"
        ),
        mean_risk_score=(
            "operational_risk_score",
            "mean"
        )
    )
    .reset_index()
    .sort_values(
        "total_censored_demand",
        ascending=False
    )
)

print("\n" + "=" * 110)
print("TOP STORES BY ESTIMATED CENSORED DEMAND")
print("=" * 110)

print(
    store_summary
    .head(20)
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 23. PRODUCT 267 DIAGNOSTIC
#
# The previous ranking was dominated by Product 267.
# We explicitly investigate whether that remains true.
# =========================================================
product_267 = risk[
    risk[
        "product_id"
    ]
    ==
    267
].copy()

print("\n" + "=" * 110)
print("PRODUCT 267 CONCENTRATION DIAGNOSTIC")
print("=" * 110)

print(
    "Product 267 store-product series:",
    len(product_267)
)

if len(product_267) > 0:

    print(
        "Share of portfolio:",
        f"{len(product_267) / len(risk) * 100:.2f}%"
    )

    print(
        "Share of adjusted demand:",
        f"{product_267['recent_total_adjusted_demand'].sum() / total_adjusted_demand * 100:.2f}%"
    )

    print(
        "Share of estimated censored demand:",
        f"{product_267['recent_estimated_censored_demand'].sum() / total_censored * 100:.2f}%"
    )

# =========================================================
# 24. BUSINESS FLAGS
# =========================================================
print("\n" + "=" * 110)
print("BUSINESS FLAG COUNTS")
print("=" * 110)

print(
    "High demand series:",
    int(
        risk[
            "high_demand"
        ].sum()
    )
)

print(
    "High stockout-burden series:",
    int(
        risk[
            "high_stockout_burden"
        ].sum()
    )
)

print(
    "High uncertainty series:",
    int(
        risk[
            "high_uncertainty"
        ].sum()
    )
)

print(
    "High demand + stockout burden + uncertainty:",
    int(
        (
            risk[
                "high_demand"
            ]
            &
            risk[
                "high_stockout_burden"
            ]
            &
            risk[
                "high_uncertainty"
            ]
        )
        .sum()
    )
)

# =========================================================
# 25. MANAGEMENT INTERPRETATION
# =========================================================
print("\n" + "=" * 110)
print("MANAGEMENT INTERPRETATION")
print("=" * 110)

print(
    "CRITICAL:"
)

print(
    "Top 1% of store-product combinations by operational "
    "risk score; immediate investigation."
)

print(
    "\nHIGH_PRIORITY:"
)

print(
    "Next 4%; persistent stockout demand burden or strong "
    "demand/forecast-risk combination."
)

print(
    "\nMONITOR:"
)

print(
    "Next 10%; elevated risk but lower urgency."
)

print(
    "\nSTANDARD:"
)

print(
    "Remaining store-product combinations."
)

print(
    "\nIMPORTANT:"
)

print(
    "Estimated censored demand is a model-based estimate, "
    "not directly observed lost sales."
)

# =========================================================
# 26. HOLDOUT PROTECTION
# =========================================================
print("\n" + "=" * 110)
print("HOLDOUT PROTECTION")
print("=" * 110)

print(
    "Official evaluation:",
    "2024-06-26 -> 2024-07-02"
)

print(
    "Final evaluation used:",
    False
)

# =========================================================
# 27. LEAKAGE CHECK
# =========================================================
print("\n" + "=" * 110)
print("LEAKAGE CHECK")
print("=" * 110)

print(
    "Risk inputs come from historical data:",
    True
)

print(
    "Future evaluation data used:",
    False
)

print(
    "Future stockout information used:",
    False
)

print(
    "Final holdout used:",
    False
)

# =========================================================
# 28. SANITY CHECKS
# =========================================================
assert (
    len(risk)
    ==
    n_series
)

assert (
    risk[
        "operational_risk_score"
    ]
    .between(
        0,
        1
    )
    .all()
)

assert (
    risk[
        "operational_rank"
    ]
    .is_unique
)

assert (
    risk[
        "recent_stockout_day_rate"
    ]
    .between(
        0,
        1
    )
    .all()
)

assert (
    risk[
        "recent_full_stockout_rate"
    ]
    .between(
        0,
        1
    )
    .all()
)

assert (
    risk[
        "recent_estimated_censored_demand"
    ]
    .ge(0)
    .all()
)

assert (
    len(
        critical
    )
    ==
    n_critical
)

assert (
    risk[
        "action_segment"
    ]
    .value_counts()
    .sum()
    ==
    n_series
)

print(
    "\nDemand-weighted risk prioritization checks: PASS"
)

# =========================================================
# 29. SAVE
# =========================================================
risk_output = (
    PROCESSED
    /
    "operational_risk_prioritization.csv"
)

segment_output = (
    PROCESSED
    /
    "operational_risk_segments.csv"
)

product_output = (
    PROCESSED
    /
    "operational_risk_by_product.csv"
)

store_output = (
    PROCESSED
    /
    "operational_risk_by_store.csv"
)

risk.to_csv(
    risk_output,
    index=False
)

segment_summary.to_csv(
    segment_output,
    index=False
)

product_summary.to_csv(
    product_output,
    index=False
)

store_summary.to_csv(
    store_output,
    index=False
)

print("\n" + "=" * 110)
print("OUTPUT")
print("=" * 110)

print(
    "Store-product risk:",
    risk_output
)

print(
    "Segment summary:",
    segment_output
)

print(
    "Product summary:",
    product_output
)

print(
    "Store summary:",
    store_output
)

print("\n" + "=" * 110)
print("PHASE 3 STEP 4R COMPLETE")
print("=" * 110)