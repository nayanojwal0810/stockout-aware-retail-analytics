from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 100)
print("PHASE 2 - STEP 4: STOCKOUT-AFFECTED DEMAND RECONSTRUCTION AUDIT")
print("=" * 100)

# =========================================================
# 1. LOAD TRAIN DATA
# =========================================================
cols = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "activity_flag"
]

df = pd.read_parquet(
    RAW / "train.parquet",
    columns=cols
)

df["dt"] = pd.to_datetime(
    df["dt"]
)

df = df.sort_values(
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
    "\nInput shape:",
    df.shape
)

# =========================================================
# 2. DEFINE STOCKOUT STATES
# =========================================================
df["stockout_state"] = np.select(
    [
        df["stock_hour6_22_cnt"] == 0,
        df["stock_hour6_22_cnt"].between(1, 15),
        df["stock_hour6_22_cnt"] == 16
    ],
    [
        "NORMAL",
        "PARTIAL_STOCKOUT",
        "FULL_STOCKOUT"
    ],
    default="UNKNOWN"
)

# =========================================================
# 3. OVERALL STATE DISTRIBUTION
# =========================================================
print("\n" + "=" * 100)
print("STOCKOUT STATE DISTRIBUTION")
print("=" * 100)

state_summary = (
    df.groupby("stockout_state")["sale_amount"]
    .agg(
        observations="size",
        mean_sales="mean",
        median_sales="median",
        zero_rate=lambda x: (
            x == 0
        ).mean()
    )
)

state_summary["share"] = (
    state_summary["observations"]
    /
    len(df)
)

print(
    state_summary[
        [
            "observations",
            "share",
            "mean_sales",
            "median_sales",
            "zero_rate"
        ]
    ]
    .round(4)
    .to_string()
)

# =========================================================
# 4. SERIES-LEVEL NORMAL SALES BENCHMARK
#
# Estimate typical observed sales for each series using
# ONLY non-stockout days.
# =========================================================
normal = df[
    df["stockout_state"] == "NORMAL"
].copy()

normal_series = (
    normal.groupby(group_cols)["sale_amount"]
    .agg(
        normal_days="size",
        normal_mean="mean",
        normal_median="median",
        normal_std="std",
        normal_p90=lambda x: x.quantile(0.90)
    )
)

print("\n" + "=" * 100)
print("NORMAL-DAY SERIES COVERAGE")
print("=" * 100)

print(
    normal_series["normal_days"]
    .describe()
    .round(2)
    .to_string()
)

print(
    "\nSeries with no normal days:",
    int(
        (
            normal_series["normal_days"] == 0
        ).sum()
    )
)

# =========================================================
# 5. MERGE NORMAL BENCHMARK
# =========================================================
df = df.merge(
    normal_series,
    on=group_cols,
    how="left"
)

# =========================================================
# 6. IDENTIFY CANDIDATE CENSORED OBSERVATIONS
#
# High-confidence candidates:
# - full stockout
# - series has enough normal history
# - normal median > 0
#
# We are NOT claiming these are exact lost sales.
# =========================================================
df["candidate_censored"] = (
    (
        df["stockout_state"]
        != "NORMAL"
    )
    &
    (
        df["normal_days"] >= 10
    )
    &
    (
        df["normal_median"] > 0
    )
)

df["high_confidence_censored"] = (
    (
        df["stockout_state"]
        == "FULL_STOCKOUT"
    )
    &
    (
        df["normal_days"] >= 10
    )
    &
    (
        df["normal_median"] > 0
    )
)

print("\n" + "=" * 100)
print("CANDIDATE CENSORED OBSERVATIONS")
print("=" * 100)

print(
    "All non-normal stockout observations:",
    int(
        (
            df["stockout_state"]
            != "NORMAL"
        ).sum()
    )
)

print(
    "Candidate censored observations:",
    int(
        df["candidate_censored"].sum()
    )
)

print(
    "High-confidence full-stockout observations:",
    int(
        df["high_confidence_censored"].sum()
    )
)

# =========================================================
# 7. IMPLIED DEMAND PRESSURE
#
# Simple benchmark:
# expected demand proxy = normal median
#
# This is NOT true demand.
# It is an empirical counterfactual proxy.
# =========================================================
df["demand_proxy_median"] = (
    df["normal_median"]
)

df["proxy_gap"] = (
    df["demand_proxy_median"]
    -
    df["sale_amount"]
)

df["proxy_gap"] = (
    df["proxy_gap"]
    .clip(lower=0)
)

# =========================================================
# 8. CENSORING MAGNITUDE
# =========================================================
print("\n" + "=" * 100)
print("CENSORING MAGNITUDE")
print("=" * 100)

candidate = df[
    df["candidate_censored"]
].copy()

if len(candidate) > 0:

    print(
        "Candidate observations:",
        len(candidate)
    )

    print(
        "Mean observed sales:",
        round(
            candidate["sale_amount"].mean(),
            4
        )
    )

    print(
        "Mean proxy demand:",
        round(
            candidate["demand_proxy_median"].mean(),
            4
        )
    )

    print(
        "Mean proxy gap:",
        round(
            candidate["proxy_gap"].mean(),
            4
        )
    )

    print(
        "Median proxy gap:",
        round(
            candidate["proxy_gap"].median(),
            4
        )
    )

    positive_gap = (
        candidate["proxy_gap"] > 0
    )

    print(
        "Positive proxy gap:",
        int(
            positive_gap.sum()
        ),
        f"({positive_gap.mean() * 100:.2f}%)"
    )

# =========================================================
# 9. FULL STOCKOUT ANALYSIS
# =========================================================
print("\n" + "=" * 100)
print("FULL STOCKOUT ANALYSIS")
print("=" * 100)

full = df[
    df["high_confidence_censored"]
].copy()

if len(full) > 0:

    print(
        "High-confidence full-stockout observations:",
        len(full)
    )

    print(
        "Mean observed sales:",
        round(
            full["sale_amount"].mean(),
            4
        )
    )

    print(
        "Mean normal-series median:",
        round(
            full["normal_median"].mean(),
            4
        )
    )

    print(
        "Mean proxy gap:",
        round(
            full["proxy_gap"].mean(),
            4
        )
    )

    print(
        "Median proxy gap:",
        round(
            full["proxy_gap"].median(),
            4
        )
    )

# =========================================================
# 10. STOCKOUT INTENSITY VS PROXY GAP
# =========================================================
print("\n" + "=" * 100)
print("STOCKOUT INTENSITY VS DEMAND PROXY GAP")
print("=" * 100)

gap_by_hours = (
    candidate.groupby(
        "stock_hour6_22_cnt"
    )
    .agg(
        observations=("sale_amount", "size"),
        mean_observed_sales=("sale_amount", "mean"),
        mean_normal_median=("normal_median", "mean"),
        mean_proxy_gap=("proxy_gap", "mean"),
        median_proxy_gap=("proxy_gap", "median")
    )
)

print(
    gap_by_hours
    .round(4)
    .to_string()
)

# =========================================================
# 11. SERIES-LEVEL CENSORING EXPOSURE
# =========================================================
series_censor = (
    df.groupby(group_cols)
    .agg(
        total_days=("dt", "nunique"),
        stockout_days=(
            "candidate_censored",
            "sum"
        ),
        full_stockout_days=(
            "high_confidence_censored",
            "sum"
        ),
        avg_stockout_hours=(
            "stock_hour6_22_cnt",
            "mean"
        ),
        mean_proxy_gap=(
            "proxy_gap",
            "mean"
        )
    )
)

series_censor["stockout_day_rate"] = (
    series_censor["stockout_days"]
    /
    series_censor["total_days"]
)

series_censor["full_stockout_day_rate"] = (
    series_censor["full_stockout_days"]
    /
    series_censor["total_days"]
)

print("\n" + "=" * 100)
print("SERIES-LEVEL CENSORING EXPOSURE")
print("=" * 100)

print(
    series_censor[
        [
            "stockout_day_rate",
            "full_stockout_day_rate",
            "avg_stockout_hours",
            "mean_proxy_gap"
        ]
    ]
    .describe()
    .round(4)
    .to_string()
)

# =========================================================
# 12. DATA SUPPORT FOR DEMAND RECOVERY
# =========================================================
print("\n" + "=" * 100)
print("DEMAND RECOVERY DATA SUPPORT")
print("=" * 100)

series_with_normal_history = (
    normal_series["normal_days"] >= 10
)

print(
    "Series with >=10 normal days:",
    int(
        series_with_normal_history.sum()
    ),
    "/",
    len(normal_series)
)

print(
    "Coverage:",
    f"{series_with_normal_history.mean() * 100:.2f}%"
)

# =========================================================
# 13. POTENTIAL BIAS CHECK
#
# Compare candidate stockout observations with their
# corresponding normal history on discount / calendar.
# =========================================================
print("\n" + "=" * 100)
print("CONTEXT COMPARISON")
print("=" * 100)

context = (
    df[
        df["candidate_censored"]
    ]
    .groupby("stockout_state")
    .agg(
        observations=("sale_amount", "size"),
        mean_discount=("discount", "mean"),
        mean_holiday=("holiday_flag", "mean"),
        mean_activity=("activity_flag", "mean"),
        mean_stockout_hours=(
            "stock_hour6_22_cnt",
            "mean"
        )
    )
)

print(
    context.round(4)
    .to_string()
)

# =========================================================
# 14. SAVE AUDIT OUTPUT
# =========================================================
audit_cols = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "stockout_state",
    "normal_days",
    "normal_median",
    "normal_mean",
    "candidate_censored",
    "high_confidence_censored",
    "demand_proxy_median",
    "proxy_gap",
    "discount",
    "holiday_flag",
    "activity_flag"
]

audit_path = (
    PROCESSED
    /
    "demand_censoring_audit.parquet"
)

df[audit_cols].to_parquet(
    audit_path,
    index=False
)

print("\n" + "=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    "Saved:",
    audit_path
)

print(
    "Rows:",
    len(df)
)

print("\n" + "=" * 100)
print("DEMAND CENSORING AUDIT COMPLETE")
print("=" * 100)