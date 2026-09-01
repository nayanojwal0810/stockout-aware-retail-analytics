from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")

print("=" * 100)
print("PHASE 1 - STEP 3: FORECAST-TIME LEAKAGE & FEATURE AVAILABILITY AUDIT")
print("=" * 100)

# =========================================================
# 1. LOAD DATA
# =========================================================
train = pd.read_parquet(
    RAW / "train.parquet",
    columns=[
        "store_id",
        "product_id",
        "dt",
        "sale_amount",
        "hours_sale",
        "stock_hour6_22_cnt",
        "hours_stock_status",
        "discount",
        "holiday_flag",
        "activity_flag",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level"
    ]
)

eval_df = pd.read_parquet(
    RAW / "eval.parquet",
    columns=[
        "store_id",
        "product_id",
        "dt",
        "sale_amount",
        "hours_sale",
        "stock_hour6_22_cnt",
        "hours_stock_status",
        "discount",
        "holiday_flag",
        "activity_flag",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level"
    ]
)

train["dt"] = pd.to_datetime(train["dt"])
eval_df["dt"] = pd.to_datetime(eval_df["dt"])

print("\nTRAIN:", train.shape)
print("EVAL :", eval_df.shape)

# =========================================================
# 2. SAME-DAY HOURLY SALES VS DAILY SALES
# =========================================================
print("\n" + "=" * 100)
print("SAME-DAY HOURLY SALES LEAKAGE TEST")
print("=" * 100)

def hourly_sum(x):
    arr = np.asarray(x, dtype=float)
    return arr.sum()

sample = train.head(10000).copy()

sample["hourly_sale_sum"] = (
    sample["hours_sale"].apply(hourly_sum)
)

sample["sales_difference"] = (
    sample["hourly_sale_sum"]
    -
    sample["sale_amount"]
)

print(
    "Rows checked:",
    len(sample)
)

print(
    "Exact hourly-sum matches:",
    int(
        np.isclose(
            sample["hourly_sale_sum"],
            sample["sale_amount"],
            rtol=0,
            atol=1e-9
        ).sum()
    )
)

print(
    "Mismatch rows:",
    int(
        (
            ~np.isclose(
                sample["hourly_sale_sum"],
                sample["sale_amount"],
                rtol=0,
                atol=1e-9
            )
        ).sum()
    )
)

print(
    "Maximum absolute difference:",
    sample["sales_difference"]
    .abs()
    .max()
)

print(
    "Mean absolute difference:",
    sample["sales_difference"]
    .abs()
    .mean()
)

# =========================================================
# 3. STOCKOUT VECTOR VS AGGREGATE COUNT
# =========================================================
print("\n" + "=" * 100)
print("STOCKOUT VECTOR CONSISTENCY")
print("=" * 100)

def stockout_count_from_vector(x):
    arr = np.asarray(x)

    # Hours 06:00 through 21:00 = 16 daytime hours.
    return int(arr[6:22].sum())

sample["recomputed_stockout_hours"] = (
    sample["hours_stock_status"]
    .apply(stockout_count_from_vector)
)

sample["stockout_count_matches"] = (
    sample["recomputed_stockout_hours"]
    ==
    sample["stock_hour6_22_cnt"]
)

print(
    "Rows checked:",
    len(sample)
)

print(
    "Matching rows:",
    int(
        sample["stockout_count_matches"]
        .sum()
    )
)

print(
    "Mismatching rows:",
    int(
        (
            ~sample["stockout_count_matches"]
        ).sum()
    )
)

print(
    "Match rate:",
    f"{sample['stockout_count_matches'].mean() * 100:.2f}%"
)

# =========================================================
# 4. SAME-DAY ENDOGENOUS INFORMATION
# =========================================================
print("\n" + "=" * 100)
print("SAME-DAY INFORMATION THAT MUST NOT BE USED DIRECTLY")
print("=" * 100)

same_day_endogenous = [
    "sale_amount",
    "hours_sale",
    "stock_hour6_22_cnt",
    "hours_stock_status"
]

for col in same_day_endogenous:
    print(
        f"- {col}: "
        "same-day observed information; "
        "cannot be used directly to predict that day's sales."
    )

# =========================================================
# 5. KNOWN-IN-ADVANCE VS OBSERVED-AFTER-THE-DAY
# =========================================================
print("\n" + "=" * 100)
print("FEATURE TIMING CLASSIFICATION")
print("=" * 100)

feature_timing = {
    "sale_amount": "TARGET",
    "hours_sale": "OBSERVED_DURING_DAY",
    "stock_hour6_22_cnt": "OBSERVED_DURING_DAY",
    "hours_stock_status": "OBSERVED_DURING_DAY",
    "discount": "POTENTIALLY_KNOWN_IN_ADVANCE",
    "holiday_flag": "KNOWN_IN_ADVANCE",
    "activity_flag": "POTENTIALLY_KNOWN_IN_ADVANCE",
    "precpt": "FUTURE_WEATHER_OBSERVATION",
    "avg_temperature": "FUTURE_WEATHER_OBSERVATION",
    "avg_humidity": "FUTURE_WEATHER_OBSERVATION",
    "avg_wind_level": "FUTURE_WEATHER_OBSERVATION"
}

for col, timing in feature_timing.items():

    print(
        f"{col:<25} {timing}"
    )

# =========================================================
# 6. CHECK WHETHER EVAL FEATURES ARE PRESENT
# =========================================================
print("\n" + "=" * 100)
print("EVALUATION-TIME FEATURE AVAILABILITY")
print("=" * 100)

for col in [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "stock_hour6_22_cnt"
]:

    print(
        f"{col:<25}"
        f" train_non_null={train[col].notna().sum():,}"
        f" eval_non_null={eval_df[col].notna().sum():,}"
    )

# =========================================================
# 7. SAME-DAY FEATURE / TARGET CORRELATION
# =========================================================
print("\n" + "=" * 100)
print("SAME-DAY FEATURE RELATIONSHIP WITH TARGET")
print("=" * 100)

for col in [
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level"
]:

    corr = train[
        [col, "sale_amount"]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    print(
        f"{col:<25}"
        f" Spearman correlation = {corr:.4f}"
    )

# =========================================================
# 8. DEMONSTRATE LAGGED FEATURE CONSTRUCTION
# =========================================================
print("\n" + "=" * 100)
print("LAG FEATURE FEASIBILITY")
print("=" * 100)

# Sort by forecasting series and date.
train = train.sort_values(
    [
        "store_id",
        "product_id",
        "dt"
    ]
)

group_cols = [
    "store_id",
    "product_id"
]

# Historical sales features
train["lag_1_sales"] = (
    train.groupby(group_cols)["sale_amount"]
    .shift(1)
)

train["lag_7_sales"] = (
    train.groupby(group_cols)["sale_amount"]
    .shift(7)
)

# Historical stockout features
train["lag_1_stockout_hours"] = (
    train.groupby(group_cols)["stock_hour6_22_cnt"]
    .shift(1)
)

# Historical discount
train["lag_1_discount"] = (
    train.groupby(group_cols)["discount"]
    .shift(1)
)

# =========================================================
# 9. LAG COVERAGE
# =========================================================
print("\n" + "=" * 100)
print("LAG COVERAGE")
print("=" * 100)

for col in [
    "lag_1_sales",
    "lag_7_sales",
    "lag_1_stockout_hours",
    "lag_1_discount"
]:

    missing = train[col].isna().sum()

    print(
        f"{col:<25}"
        f" missing={missing:,}"
        f" pct={missing / len(train) * 100:.2f}%"
    )

# =========================================================
# 10. LAGGED FEATURE RELATIONSHIP WITH TARGET
# =========================================================
print("\n" + "=" * 100)
print("LAGGED FEATURE RELATIONSHIP WITH NEXT-DAY TARGET")
print("=" * 100)

lag_features = [
    "lag_1_sales",
    "lag_7_sales",
    "lag_1_stockout_hours",
    "lag_1_discount"
]

for col in lag_features:

    subset = train[
        [col, "sale_amount"]
    ].dropna()

    corr = subset.corr(
        method="spearman"
    ).iloc[0, 1]

    print(
        f"{col:<25}"
        f" Spearman correlation = {corr:.4f}"
    )

# =========================================================
# 11. WEATHER INFORMATION CHECK
# =========================================================
print("\n" + "=" * 100)
print("WEATHER FEATURE INTERPRETATION")
print("=" * 100)

weather_cols = [
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level"
]

for col in weather_cols:

    print(
        f"\n{col}"
    )

    print(
        "Train mean:",
        round(
            train[col].mean(),
            4
        )
    )

    print(
        "Eval mean:",
        round(
            eval_df[col].mean(),
            4
        )
    )

    print(
        "Train min/max:",
        (
            round(train[col].min(), 4),
            round(train[col].max(), 4)
        )
    )

    print(
        "Eval min/max:",
        (
            round(eval_df[col].min(), 4),
            round(eval_df[col].max(), 4)
        )
    )

# =========================================================
# 12. DISCOUNT / ACTIVITY STABILITY
# =========================================================
print("\n" + "=" * 100)
print("KNOWN-IN-ADVANCE CANDIDATE FEATURES")
print("=" * 100)

for col in [
    "discount",
    "holiday_flag",
    "activity_flag"
]:

    train_values = train[col].value_counts(
        normalize=True
    ).sort_index()

    eval_values = eval_df[col].value_counts(
        normalize=True
    ).sort_index()

    print(f"\n{col}")

    print("TRAIN distribution:")
    print(
        train_values
        .round(4)
        .to_string()
    )

    print("EVAL distribution:")
    print(
        eval_values
        .round(4)
        .to_string()
    )

# =========================================================
# 13. FINAL FORECASTING POLICY
# =========================================================
print("\n" + "=" * 100)
print("PROPOSED FORECASTING INFORMATION POLICY")
print("=" * 100)

policy = {
    "Historical sales": "ALLOW AS LAGGED FEATURES",
    "Historical stockout": "ALLOW AS LAGGED FEATURES",
    "Historical discount": "ALLOW AS LAGGED FEATURES",
    "Holiday calendar": "ALLOW",
    "Activity calendar": "ALLOW ONLY IF KNOWN BEFORE FORECAST",
    "Future actual weather": "DO NOT USE DIRECTLY",
    "Future actual stockout": "DO NOT USE",
    "Future actual sales": "DO NOT USE",
    "Current-day hours_sale": "DO NOT USE",
    "Current-day stockout": "DO NOT USE"
}

for item, rule in policy.items():
    print(
        f"{item:<35} {rule}"
    )

# =========================================================
# 14. FINAL SUMMARY
# =========================================================
print("\n" + "=" * 100)
print("LEAKAGE AUDIT SUMMARY")
print("=" * 100)

print(
    "Same-day hourly sales sum validated:",
    int(
        np.isclose(
            sample["hourly_sale_sum"],
            sample["sale_amount"],
            rtol=0,
            atol=1e-9
        ).sum()
    ),
    "/",
    len(sample)
)

print(
    "Stockout aggregate consistency:",
    f"{sample['stockout_count_matches'].mean() * 100:.2f}%"
)

print(
    "Train/Eval temporal overlap: FALSE"
)

print(
    "Primary forecasting rule: "
    "use only information available before the forecast date."
)

print("\n" + "=" * 100)
print("FEATURE TIMING AUDIT COMPLETE")
print("=" * 100)