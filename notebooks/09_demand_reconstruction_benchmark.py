from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 100)
print("PHASE 2 - STEP 5: DEMAND RECONSTRUCTION BENCHMARK")
print("=" * 100)

# =========================================================
# 1. LOAD DATA
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
).reset_index(drop=True)

group_cols = [
    "store_id",
    "product_id"
]

print(
    "\nInput shape:",
    df.shape
)

# =========================================================
# 2. STOCKOUT STATES
# =========================================================
df["stockout_flag"] = (
    df["stock_hour6_22_cnt"] > 0
).astype(int)

df["normal_day"] = (
    df["stock_hour6_22_cnt"] == 0
)

# =========================================================
# 3. TEMPORAL FEATURES
# =========================================================
group = df.groupby(
    group_cols,
    sort=False
)

# Historical sales features
df["lag_1_sales"] = (
    group["sale_amount"]
    .shift(1)
)

df["lag_7_sales"] = (
    group["sale_amount"]
    .shift(7)
)

df["rolling_7_sales"] = (
    df.groupby(group_cols)["sale_amount"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                7,
                min_periods=3
            )
            .mean()
    )
)

df["rolling_14_sales"] = (
    df.groupby(group_cols)["sale_amount"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                14,
                min_periods=7
            )
            .mean()
    )
)

# Historical stockout history
df["lag_1_stockout"] = (
    group["stockout_flag"]
    .shift(1)
)

df["rolling_7_stockout_hours"] = (
    df.groupby(group_cols)["stock_hour6_22_cnt"]
    .transform(
        lambda s:
            s.shift(1)
            .rolling(
                7,
                min_periods=3
            )
            .mean()
    )
)

# Calendar
df["day_of_week"] = (
    df["dt"].dt.dayofweek
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# =========================================================
# 4. CONTEXT FEATURES
# =========================================================
feature_cols = [
    "lag_1_sales",
    "lag_7_sales",
    "rolling_7_sales",
    "rolling_14_sales",
    "lag_1_stockout",
    "rolling_7_stockout_hours",
    "day_of_week",
    "is_weekend",
    "discount",
    "holiday_flag",
    "activity_flag"
]

# =========================================================
# 5. PSEUDO-CENSORING VALIDATION PERIOD
#
# Use later normal observations as "fake stockout" days.
# Earlier normal days are used to learn reconstruction.
# =========================================================
TRAIN_END = pd.Timestamp(
    "2024-06-04"
)

VALID_START = pd.Timestamp(
    "2024-06-05"
)

VALID_END = pd.Timestamp(
    "2024-06-18"
)

print("\n" + "=" * 100)
print("PSEUDO-CENSORING DESIGN")
print("=" * 100)

print(
    "Reconstruction training period:",
    f"{TRAIN_END.date()}"
)

print(
    "Pseudo-censored validation:",
    f"{VALID_START.date()} -> {VALID_END.date()}"
)

print(
    "Only genuinely NORMAL validation days are used."
)

# =========================================================
# 6. TRAIN DATA = NORMAL HISTORICAL OBSERVATIONS
# =========================================================
reconstruction_train = df[
    (df["dt"] <= TRAIN_END)
    &
    (df["normal_day"])
].copy()

# =========================================================
# 7. VALIDATION DATA = FUTURE NORMAL DAYS
# =========================================================
reconstruction_valid = df[
    df["dt"].between(
        VALID_START,
        VALID_END
    )
    &
    df["normal_day"]
].copy()

print(
    "\nTraining normal observations:",
    len(reconstruction_train)
)

print(
    "Validation normal observations:",
    len(reconstruction_valid)
)

# =========================================================
# 8. SERIES MEDIAN BENCHMARK
# =========================================================
series_median = (
    reconstruction_train
    .groupby(group_cols)["sale_amount"]
    .median()
    .rename(
        "series_median_prediction"
    )
)

reconstruction_valid = (
    reconstruction_valid.merge(
        series_median,
        on=group_cols,
        how="left"
    )
)

# Fallback to global median only if a series
# has no historical normal observations.
global_median = (
    reconstruction_train["sale_amount"]
    .median()
)

reconstruction_valid[
    "series_median_prediction"
] = (
    reconstruction_valid[
        "series_median_prediction"
    ]
    .fillna(global_median)
)

# =========================================================
# 9. WEEKDAY-SPECIFIC MEDIAN BENCHMARK
# =========================================================
weekday_median = (
    reconstruction_train
    .groupby(
        group_cols
        +
        ["day_of_week"]
    )["sale_amount"]
    .median()
    .rename(
        "weekday_median_prediction"
    )
    .reset_index()
)

reconstruction_valid = (
    reconstruction_valid.merge(
        weekday_median,
        on=group_cols + ["day_of_week"],
        how="left"
    )
)

# Fallback hierarchy:
# weekday-specific → series median → global median
reconstruction_valid[
    "weekday_median_prediction"
] = (
    reconstruction_valid[
        "weekday_median_prediction"
    ]
    .fillna(
        reconstruction_valid[
            "series_median_prediction"
        ]
    )
    .fillna(
        global_median
    )
)

# =========================================================
# 10. CONTEXT-AWARE MODEL
# =========================================================
train_model = reconstruction_train[
    feature_cols
    +
    ["sale_amount"]
].copy()

valid_model = reconstruction_valid[
    feature_cols
    +
    ["sale_amount"]
].copy()

train_model = train_model[
    train_model["sale_amount"].notna()
]

model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.08,
    max_iter=150,
    max_leaf_nodes=31,
    min_samples_leaf=100,
    l2_regularization=1.0,
    random_state=42,
    early_stopping=False
)

X_train = train_model[
    feature_cols
].to_numpy(
    dtype=np.float32
)

y_train = train_model[
    "sale_amount"
].to_numpy(
    dtype=np.float32
)

X_valid = valid_model[
    feature_cols
].to_numpy(
    dtype=np.float32
)

model.fit(
    X_train,
    y_train
)

context_prediction = model.predict(
    X_valid
)

context_prediction = np.maximum(
    context_prediction,
    0
)

reconstruction_valid[
    "context_model_prediction"
] = context_prediction

# =========================================================
# 11. METRICS
# =========================================================
def mae(
    actual,
    predicted
):
    return np.mean(
        np.abs(
            actual - predicted
        )
    )


def wape(
    actual,
    predicted
):
    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return np.nan

    return (
        np.sum(
            np.abs(
                actual - predicted
            )
        )
        /
        denominator
    )


def rmse(
    actual,
    predicted
):
    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


# =========================================================
# 12. EVALUATE RECONSTRUCTION METHODS
# =========================================================
actual = (
    reconstruction_valid[
        "sale_amount"
    ]
    .to_numpy(
        dtype=np.float32
    )
)

predictions = {
    "Series_Median":
        reconstruction_valid[
            "series_median_prediction"
        ].to_numpy(
            dtype=np.float32
        ),

    "Weekday_Median":
        reconstruction_valid[
            "weekday_median_prediction"
        ].to_numpy(
            dtype=np.float32
        ),

    "Context_Model":
        reconstruction_valid[
            "context_model_prediction"
        ].to_numpy(
            dtype=np.float32
        )
}

results = []

for model_name, prediction in predictions.items():

    prediction = np.maximum(
        prediction,
        0
    )

    results.append({
        "method": model_name,
        "observations": len(actual),
        "MAE": mae(
            actual,
            prediction
        ),
        "WAPE": wape(
            actual,
            prediction
        ),
        "RMSE": rmse(
            actual,
            prediction
        )
    })

results_df = pd.DataFrame(
    results
).sort_values(
    "WAPE"
)

# =========================================================
# 13. PRINT RESULTS
# =========================================================
print("\n" + "=" * 100)
print("PSEUDO-CENSORING RECONSTRUCTION RESULTS")
print("=" * 100)

print(
    results_df
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 14. IMPROVEMENT OVER SIMPLE BENCHMARK
# =========================================================
series_wape = (
    results_df.loc[
        results_df["method"]
        ==
        "Series_Median",
        "WAPE"
    ].iloc[0]
)

weekday_wape = (
    results_df.loc[
        results_df["method"]
        ==
        "Weekday_Median",
        "WAPE"
    ].iloc[0]
)

context_wape = (
    results_df.loc[
        results_df["method"]
        ==
        "Context_Model",
        "WAPE"
    ].iloc[0]
)

print("\n" + "=" * 100)
print("RECONSTRUCTION IMPROVEMENT")
print("=" * 100)

print(
    "Weekday median improvement vs series median:",
    f"{(series_wape - weekday_wape) / series_wape * 100:.2f}%"
)

print(
    "Context model improvement vs series median:",
    f"{(series_wape - context_wape) / series_wape * 100:.2f}%"
)

print(
    "Context model improvement vs weekday median:",
    f"{(weekday_wape - context_wape) / weekday_wape * 100:.2f}%"
)

# =========================================================
# 15. ERROR BY TRUE SALES LEVEL
# =========================================================
reconstruction_valid[
    "context_abs_error"
] = (
    np.abs(
        reconstruction_valid[
            "sale_amount"
        ]
        -
        reconstruction_valid[
            "context_model_prediction"
        ]
    )
)

reconstruction_valid[
    "actual_sales_bucket"
] = pd.qcut(
    reconstruction_valid[
        "sale_amount"
    ],
    q=5,
    duplicates="drop"
)

bucket_results = (
    reconstruction_valid
    .groupby(
        "actual_sales_bucket",
        observed=True
    )["context_abs_error"]
    .agg(
        observations="size",
        mean_abs_error="mean",
        median_abs_error="median"
    )
)

print("\n" + "=" * 100)
print("CONTEXT MODEL ERROR BY ACTUAL SALES")
print("=" * 100)

print(
    bucket_results
    .round(4)
    .to_string()
)

# =========================================================
# 16. VALIDATION COVERAGE
# =========================================================
print("\n" + "=" * 100)
print("VALIDATION COVERAGE")
print("=" * 100)

print(
    "Validation observations:",
    len(reconstruction_valid)
)

print(
    "Missing context predictions:",
    reconstruction_valid[
        "context_model_prediction"
    ].isna().sum()
)

print(
    "Missing weekday median predictions:",
    reconstruction_valid[
        "weekday_median_prediction"
    ].isna().sum()
)

# =========================================================
# 17. ACTUAL STOCKOUT DESCRIPTIVE CHECK
# =========================================================
actual_stockout = df[
    df["stockout_flag"] == 1
]

print("\n" + "=" * 100)
print("ACTUAL STOCKOUT REFERENCE")
print("=" * 100)

print(
    "Actual stockout observations:",
    len(actual_stockout)
)

print(
    "Mean observed sales:",
    round(
        actual_stockout[
            "sale_amount"
        ].mean(),
        4
    )
)

print(
    "Median observed sales:",
    round(
        actual_stockout[
            "sale_amount"
        ].median(),
        4
    )
)

# =========================================================
# 18. SAVE VALIDATION RESULTS
# =========================================================
results_path = (
    PROCESSED
    /
    "demand_reconstruction_benchmark.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

detail_path = (
    PROCESSED
    /
    "pseudo_censored_validation.parquet"
)

reconstruction_valid[
    [
        "store_id",
        "product_id",
        "dt",
        "sale_amount",
        "series_median_prediction",
        "weekday_median_prediction",
        "context_model_prediction",
        "context_abs_error"
    ]
].to_parquet(
    detail_path,
    index=False
)

print("\n" + "=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    "Summary saved:",
    results_path
)

print(
    "Validation detail saved:",
    detail_path
)

print("\n" + "=" * 100)
print("DEMAND RECONSTRUCTION BENCHMARK COMPLETE")
print("=" * 100)