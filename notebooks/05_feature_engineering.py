from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

PROCESSED.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 100)
print("PHASE 2 - STEP 1: LEAKAGE-SAFE FEATURE ENGINEERING")
print("=" * 100)

# =========================================================
# 1. LOAD TRAIN DATA
# =========================================================
train = pd.read_parquet(
    RAW / "train.parquet"
)

train["dt"] = pd.to_datetime(train["dt"])

train = train.sort_values(
    [
        "store_id",
        "product_id",
        "dt"
    ]
).reset_index(drop=True)

print("\nInput shape:", train.shape)

# =========================================================
# 2. FORECASTING SERIES
# =========================================================
group_cols = [
    "store_id",
    "product_id"
]

group = train.groupby(
    group_cols,
    sort=False
)

# =========================================================
# 3. STOCKOUT FEATURES
# =========================================================
train["stockout_flag"] = (
    train["stock_hour6_22_cnt"] > 0
).astype(int)

train["full_stockout_flag"] = (
    train["stock_hour6_22_cnt"] == 16
).astype(int)

# =========================================================
# 4. LAGGED SALES
# =========================================================
train["lag_1_sales"] = (
    group["sale_amount"]
    .shift(1)
)

train["lag_2_sales"] = (
    group["sale_amount"]
    .shift(2)
)

train["lag_7_sales"] = (
    group["sale_amount"]
    .shift(7)
)

train["lag_14_sales"] = (
    group["sale_amount"]
    .shift(14)
)

# =========================================================
# 5. LAGGED STOCKOUT FEATURES
# =========================================================
train["lag_1_stockout_hours"] = (
    group["stock_hour6_22_cnt"]
    .shift(1)
)

train["lag_7_stockout_hours"] = (
    group["stock_hour6_22_cnt"]
    .shift(7)
)

train["lag_1_stockout_flag"] = (
    group["stockout_flag"]
    .shift(1)
)

# =========================================================
# 6. LAGGED DISCOUNT
# =========================================================
train["lag_1_discount"] = (
    group["discount"]
    .shift(1)
)

train["lag_7_discount"] = (
    group["discount"]
    .shift(7)
)

# =========================================================
# 7. LAGGED WEATHER
# =========================================================
weather_cols = [
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level"
]

for col in weather_cols:
    train[f"lag_1_{col}"] = (
        group[col]
        .shift(1)
    )

# =========================================================
# 8. SAFE ROLLING FEATURES
#
# We first create shifted series and then rolling values
# within each store-product series.
# =========================================================
train["rolling_7_mean_sales"] = (
    train.groupby(group_cols)["sale_amount"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                window=7,
                min_periods=3
            )
            .mean()
    )
)

train["rolling_14_mean_sales"] = (
    train.groupby(group_cols)["sale_amount"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                window=14,
                min_periods=7
            )
            .mean()
    )
)

train["rolling_28_mean_sales"] = (
    train.groupby(group_cols)["sale_amount"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                window=28,
                min_periods=14
            )
            .mean()
    )
)

train["rolling_7_std_sales"] = (
    train.groupby(group_cols)["sale_amount"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                window=7,
                min_periods=3
            )
            .std()
    )
)

# =========================================================
# 9. ROLLING STOCKOUT FEATURES
# =========================================================
train["rolling_7_stockout_hours"] = (
    train.groupby(group_cols)["stock_hour6_22_cnt"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                window=7,
                min_periods=3
            )
            .mean()
    )
)

train["rolling_14_stockout_hours"] = (
    train.groupby(group_cols)["stock_hour6_22_cnt"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                window=14,
                min_periods=7
            )
            .mean()
    )
)

train["rolling_7_stockout_days"] = (
    train.groupby(group_cols)["stockout_flag"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                window=7,
                min_periods=3
            )
            .sum()
    )
)

train["rolling_14_stockout_days"] = (
    train.groupby(group_cols)["stockout_flag"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                window=14,
                min_periods=7
            )
            .sum()
    )
)

# =========================================================
# 10. DEMAND TREND FEATURES
# =========================================================
train["recent_7_vs_14_ratio"] = (
    train["rolling_7_mean_sales"]
    /
    train["rolling_14_mean_sales"].replace(
        0,
        np.nan
    )
)

train["lag_1_vs_lag_7_ratio"] = (
    train["lag_1_sales"]
    /
    train["lag_7_sales"].replace(
        0,
        np.nan
    )
)

train["rolling_7_cv_sales"] = (
    train["rolling_7_std_sales"]
    /
    train["rolling_7_mean_sales"].replace(
        0,
        np.nan
    )
)

# =========================================================
# 11. CALENDAR FEATURES
# =========================================================
train["day_of_week"] = (
    train["dt"].dt.dayofweek
)

train["day_of_month"] = (
    train["dt"].dt.day
)

train["week_of_year"] = (
    train["dt"].dt.isocalendar().week
    .astype(int)
)

train["month"] = (
    train["dt"].dt.month
)

train["is_weekend"] = (
    train["day_of_week"] >= 5
).astype(int)

train["holiday_flag"] = (
    train["holiday_flag"]
    .astype(int)
)

train["activity_flag"] = (
    train["activity_flag"]
    .astype(int)
)

# =========================================================
# 12. SERIES AGE
# =========================================================
train["series_day_number"] = (
    train.groupby(group_cols)
    .cumcount()
    + 1
)

# =========================================================
# 13. FEATURE LIST
# =========================================================
feature_cols = [
    "lag_1_sales",
    "lag_2_sales",
    "lag_7_sales",
    "lag_14_sales",
    "rolling_7_mean_sales",
    "rolling_14_mean_sales",
    "rolling_28_mean_sales",
    "rolling_7_std_sales",
    "rolling_7_cv_sales",
    "lag_1_stockout_hours",
    "lag_7_stockout_hours",
    "lag_1_stockout_flag",
    "rolling_7_stockout_hours",
    "rolling_14_stockout_hours",
    "rolling_7_stockout_days",
    "rolling_14_stockout_days",
    "lag_1_discount",
    "lag_7_discount",
    "recent_7_vs_14_ratio",
    "lag_1_vs_lag_7_ratio",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "is_weekend",
    "holiday_flag",
    "activity_flag",
    "lag_1_precpt",
    "lag_1_avg_temperature",
    "lag_1_avg_humidity",
    "lag_1_avg_wind_level",
    "series_day_number"
]

# =========================================================
# 14. FEATURE COVERAGE
# =========================================================
print("\n" + "=" * 100)
print("FEATURE COVERAGE")
print("=" * 100)

for col in feature_cols:

    missing = train[col].isna().sum()
    pct = (
        missing
        /
        len(train)
        *
        100
    )

    print(
        f"{col:<35}"
        f"missing={missing:>10,}"
        f"pct={pct:>7.2f}%"
    )

# =========================================================
# 15. TEMPORAL ALIGNMENT CHECK
# =========================================================
print("\n" + "=" * 100)
print("TEMPORAL ALIGNMENT CHECK")
print("=" * 100)

example_series = (
    train[
        [
            "store_id",
            "product_id"
        ]
    ]
    .drop_duplicates()
    .iloc[0]
)

example = train[
    (
        train["store_id"]
        ==
        example_series["store_id"]
    )
    &
    (
        train["product_id"]
        ==
        example_series["product_id"]
    )
].head(35)

print(
    example[
        [
            "dt",
            "sale_amount",
            "lag_1_sales",
            "lag_7_sales",
            "rolling_7_mean_sales",
            "rolling_14_mean_sales",
            "lag_1_stockout_hours",
            "rolling_7_stockout_hours",
            "lag_1_discount"
        ]
    ]
    .to_string(index=False)
)

# =========================================================
# 16. LEAKAGE CHECK
# =========================================================
print("\n" + "=" * 100)
print("LEAKAGE CHECK")
print("=" * 100)

same_day_fields = [
    "sale_amount",
    "hours_sale",
    "stock_hour6_22_cnt",
    "hours_stock_status"
]

print(
    "Same-day fields excluded from model features:"
)

for col in same_day_fields:

    print(
        f"{col:<25}"
        f"{col not in feature_cols}"
    )

# =========================================================
# 17. FUTURE REALIZED WEATHER CHECK
# =========================================================
print(
    "\nFuture realized weather fields excluded directly:"
)

for col in weather_cols:

    print(
        f"{col:<25}"
        f"{col not in feature_cols}"
    )

# =========================================================
# 18. FEATURE / TARGET ASSOCIATION
# =========================================================
print("\n" + "=" * 100)
print("FEATURE RELATIONSHIP WITH TARGET")
print("=" * 100)

correlations = []

for col in feature_cols:

    subset = train[
        [
            col,
            "sale_amount"
        ]
    ].dropna()

    if len(subset) == 0:
        continue

    corr = (
        subset
        .corr(
            method="spearman"
        )
        .iloc[0, 1]
    )

    correlations.append(
        {
            "feature": col,
            "spearman_corr": corr,
            "abs_corr": abs(corr)
        }
    )

correlations = pd.DataFrame(
    correlations
).sort_values(
    "abs_corr",
    ascending=False
)

print(
    correlations
    .head(20)
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 19. FEATURE SANITY
# =========================================================
print("\n" + "=" * 100)
print("FEATURE SANITY")
print("=" * 100)

print(
    train[
        feature_cols
    ]
    .describe()
    .T[
        [
            "count",
            "mean",
            "std",
            "min",
            "50%",
            "max"
        ]
    ]
    .round(4)
    .to_string()
)

# =========================================================
# 20. SAVE FEATURE TABLE
# =========================================================
save_cols = (
    group_cols
    +
    [
        "dt",
        "sale_amount"
    ]
    +
    feature_cols
)

feature_table = train[
    save_cols
].copy()

output_path = (
    PROCESSED
    /
    "train_features.parquet"
)

feature_table.to_parquet(
    output_path,
    index=False
)

print("\n" + "=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    "Feature table shape:",
    feature_table.shape
)

print(
    "Output file:",
    output_path
)

print(
    "Output exists:",
    output_path.exists()
)

print("\n" + "=" * 100)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 100)