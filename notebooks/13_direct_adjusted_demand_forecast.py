from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 110)
print("PHASE 3 - STEP 2: DIRECT 7-DAY FORECAST OF STOCKOUT-ADJUSTED DEMAND")
print("=" * 110)

# =========================================================
# 1. LOAD RAW DATA
# =========================================================
raw_cols = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "activity_flag"
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

series_index = (
    raw[group_cols]
    .drop_duplicates()
    .sort_values(group_cols)
    .set_index(group_cols)
    .index
)

n_series = len(series_index)

print(
    "\nRAW DATA:",
    raw.shape
)

print(
    "SERIES:",
    n_series
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
    adjusted_path
)

adjusted["dt"] = pd.to_datetime(
    adjusted["dt"]
)

adjusted = adjusted[
    [
        "store_id",
        "product_id",
        "dt",
        "adjusted_demand"
    ]
]

print(
    "ADJUSTED DATA:",
    adjusted.shape
)

print(
    "ADJUSTED DATE RANGE:",
    adjusted["dt"].min(),
    "->",
    adjusted["dt"].max()
)

# =========================================================
# 3. MERGE
# =========================================================
df = raw.merge(
    adjusted,
    on=[
        "store_id",
        "product_id",
        "dt"
    ],
    how="left"
)

print(
    "MERGED DATA:",
    df.shape
)

print(
    "Adjusted target coverage:",
    f"{df['adjusted_demand'].notna().mean() * 100:.2f}%"
)

# =========================================================
# 4. CREATE LEAKAGE-SAFE ADJUSTED-DEMAND FEATURES
# =========================================================
def create_features(
    data
):

    data = data.sort_values(
        group_cols + ["dt"]
    ).copy()

    g = data.groupby(
        group_cols,
        sort=False
    )

    # -----------------------------------------------------
    # Adjusted-demand lags
    # -----------------------------------------------------
    data["lag_1"] = (
        g["adjusted_demand"]
        .shift(1)
    )

    data["lag_2"] = (
        g["adjusted_demand"]
        .shift(2)
    )

    data["lag_7"] = (
        g["adjusted_demand"]
        .shift(7)
    )

    data["lag_14"] = (
        g["adjusted_demand"]
        .shift(14)
    )

    # -----------------------------------------------------
    # Rolling adjusted-demand features
    # -----------------------------------------------------
    data["rolling_7_mean"] = (
        data.groupby(
            group_cols
        )["adjusted_demand"]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(
                    7,
                    min_periods=7
                )
                .mean()
        )
    )

    data["rolling_14_mean"] = (
        data.groupby(
            group_cols
        )["adjusted_demand"]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(
                    14,
                    min_periods=14
                )
                .mean()
        )
    )

    data["rolling_7_std"] = (
        data.groupby(
            group_cols
        )["adjusted_demand"]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(
                    7,
                    min_periods=7
                )
                .std()
        )
    )

    # -----------------------------------------------------
    # Historical stockout information
    #
    # Stockout is never taken from the future target date.
    # -----------------------------------------------------
    data["lag_1_stockout_hours"] = (
        g["stock_hour6_22_cnt"]
        .shift(1)
    )

    data["lag_7_stockout_hours"] = (
        g["stock_hour6_22_cnt"]
        .shift(7)
    )

    data["rolling_7_stockout_hours"] = (
        data.groupby(
            group_cols
        )["stock_hour6_22_cnt"]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(
                    7,
                    min_periods=7
                )
                .mean()
        )
    )

    # -----------------------------------------------------
    # Historical discount
    # -----------------------------------------------------
    data["lag_1_discount"] = (
        g["discount"]
        .shift(1)
    )

    data["lag_7_discount"] = (
        g["discount"]
        .shift(7)
    )

    # -----------------------------------------------------
    # Calendar
    # -----------------------------------------------------
    data["day_of_week"] = (
        data["dt"].dt.dayofweek
    )

    data["day_of_month"] = (
        data["dt"].dt.day
    )

    data["week_of_year"] = (
        data["dt"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["holiday_flag"] = (
        data["holiday_flag"]
        .astype(int)
    )

    data["activity_flag"] = (
        data["activity_flag"]
        .astype(int)
    )

    return data


model_data = create_features(
    df[
        [
            "store_id",
            "product_id",
            "dt",
            "adjusted_demand",
            "stock_hour6_22_cnt",
            "discount",
            "holiday_flag",
            "activity_flag"
        ]
    ].copy()
)

# =========================================================
# 5. FEATURES
# =========================================================
features = [
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "rolling_7_mean",
    "rolling_14_mean",
    "rolling_7_std",
    "lag_1_stockout_hours",
    "lag_7_stockout_hours",
    "rolling_7_stockout_hours",
    "lag_1_discount",
    "lag_7_discount",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "is_weekend",
    "holiday_flag",
    "activity_flag"
]

print(
    "\nFEATURE COUNT:",
    len(features)
)

assert len(features) == 18

# =========================================================
# 6. FOLDS
#
# Adjusted demand starts on 2024-05-15.
# Therefore only folds 3, 4, 5 are eligible.
# =========================================================
folds = [
    {
        "fold": 3,
        "train_end": "2024-05-14",
        "valid_start": "2024-05-15",
        "valid_end": "2024-05-21"
    },
    {
        "fold": 4,
        "train_end": "2024-05-21",
        "valid_start": "2024-05-22",
        "valid_end": "2024-05-28"
    },
    {
        "fold": 5,
        "train_end": "2024-05-28",
        "valid_start": "2024-05-29",
        "valid_end": "2024-06-04"
    }
]

# =========================================================
# 7. METRICS
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
            (
                actual
                -
                predicted
            ) ** 2
        )
    )


# =========================================================
# 8. MODEL
# =========================================================
def build_model():

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=False
    )


# =========================================================
# 9. RESULTS
# =========================================================
results = []

# =========================================================
# 10. DIRECT MULTI-HORIZON FORECASTING
# =========================================================
for fold_info in folds:

    fold = fold_info["fold"]

    train_end = pd.Timestamp(
        fold_info["train_end"]
    )

    valid_start = pd.Timestamp(
        fold_info["valid_start"]
    )

    valid_end = pd.Timestamp(
        fold_info["valid_end"]
    )

    print("\n" + "=" * 110)
    print(
        f"FOLD {fold}"
    )
    print(
        f"Forecast origin: {train_end.date()}"
    )
    print(
        f"Forecast window: "
        f"{valid_start.date()} -> {valid_end.date()}"
    )
    print("=" * 110)

    # -----------------------------------------------------
    # Origin features
    # -----------------------------------------------------
    origin = model_data[
        model_data["dt"] == train_end
    ].copy()

    print(
        "Origin rows:",
        len(origin)
    )

    assert (
        len(origin)
        ==
        n_series
    )

    X_origin = (
        origin[
            features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    assert (
        X_origin.shape
        ==
        (
            n_series,
            len(features)
        )
    )

    # -----------------------------------------------------
    # Loop through horizons
    # -----------------------------------------------------
    for horizon in range(1, 8):

        target_date = (
            train_end
            +
            pd.Timedelta(
                days=horizon
            )
        )

        print(
            f"\nHORIZON +{horizon}: "
            f"{target_date.date()}"
        )

        # -------------------------------------------------
        # Direct target:
        #
        # adjusted demand at t+h
        #
        # The origin remains at t, so no future predictor
        # information is used.
        # -------------------------------------------------
        training = model_data.copy()

        training[
            "direct_target"
        ] = (
            training
            .groupby(
                group_cols
            )[
                "adjusted_demand"
            ]
            .shift(
                -horizon
            )
        )

        training = training[
            (
                training["dt"]
                <=
                train_end
            )
            &
            (
                training[
                    "adjusted_demand"
                ]
                .notna()
            )
            &
            (
                training[
                    "direct_target"
                ]
                .notna()
            )
        ].copy()

        training = training.dropna(
            subset=features
        )

        print(
            "Training rows:",
            len(training)
        )

        if len(training) == 0:

            print(
                "SKIPPED: no valid training observations."
            )

            continue

        # -------------------------------------------------
        # Train horizon-specific model
        # -------------------------------------------------
        model = build_model()

        X_train = (
            training[
                features
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        y_train = (
            training[
                "direct_target"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        model.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # Predict all series at once
        # -------------------------------------------------
        prediction = model.predict(
            X_origin
        )

        prediction = np.maximum(
            prediction,
            0
        ).astype(
            np.float32
        )

        # -------------------------------------------------
        # Actual adjusted demand at target date
        # -------------------------------------------------
        actual = model_data[
            model_data["dt"] == target_date
        ][
            group_cols
            +
            [
                "adjusted_demand"
            ]
        ]

        print(
            "Expected actual rows:",
            n_series
        )

        print(
            "Actual rows:",
            len(actual)
        )

        assert (
            len(actual)
            ==
            n_series
        )

        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------
        actual_values = (
            actual[
                "adjusted_demand"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        mae_value = mae(
            actual_values,
            prediction
        )

        wape_value = wape(
            actual_values,
            prediction
        )

        rmse_value = rmse(
            actual_values,
            prediction
        )

        print(
            f"MAE={mae_value:.4f} "
            f"WAPE={wape_value:.4f} "
            f"RMSE={rmse_value:.4f}"
        )

        results.append(
            {
                "fold": fold,
                "horizon": horizon,
                "target_date": target_date,
                "model": "GB_Direct_Adjusted_Demand",
                "target": "adjusted_demand",
                "rows": len(actual),
                "MAE": mae_value,
                "WAPE": wape_value,
                "RMSE": rmse_value
            }
        )

# =========================================================
# 11. RESULTS TABLE
# =========================================================
results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 110)
print("DIRECT ADJUSTED-DEMAND RESULTS")
print("=" * 110)

print(
    results_df
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 12. CROSS-FOLD / HORIZON SUMMARY
# =========================================================
print("\n" + "=" * 110)
print("ADJUSTED-DEMAND SUMMARY BY HORIZON")
print("=" * 110)

summary = (
    results_df
    .groupby(
        "horizon"
    )
    .agg(
        folds=("fold", "nunique"),
        mean_mae=("MAE", "mean"),
        std_mae=("MAE", "std"),
        mean_wape=("WAPE", "mean"),
        std_wape=("WAPE", "std"),
        mean_rmse=("RMSE", "mean")
    )
)

print(
    summary
    .round(4)
    .to_string()
)

# =========================================================
# 13. OVERALL SUMMARY
# =========================================================
print("\n" + "=" * 110)
print("OVERALL ADJUSTED-DEMAND FORECAST")
print("=" * 110)

print(
    "Mean MAE:",
    round(
        results_df["MAE"].mean(),
        4
    )
)

print(
    "Mean WAPE:",
    round(
        results_df["WAPE"].mean(),
        4
    )
)

print(
    "Mean RMSE:",
    round(
        results_df["RMSE"].mean(),
        4
    )
)

# =========================================================
# 14. FOLD SUMMARY
# =========================================================
print("\n" + "=" * 110)
print("FOLD SUMMARY")
print("=" * 110)

fold_summary = (
    results_df
    .groupby(
        "fold"
    )
    .agg(
        mean_mae=("MAE", "mean"),
        mean_wape=("WAPE", "mean"),
        mean_rmse=("RMSE", "mean")
    )
)

print(
    fold_summary
    .round(4)
    .to_string()
)

# =========================================================
# 15. WAPE BY FOLD/HORIZON
# =========================================================
print("\n" + "=" * 110)
print("WAPE BY FOLD AND HORIZON")
print("=" * 110)

wape_table = (
    results_df
    .pivot(
        index="fold",
        columns="horizon",
        values="WAPE"
    )
)

print(
    wape_table
    .round(4)
    .to_string()
)

# =========================================================
# 16. COMPARE WITH RAW DIRECT FORECAST
#
# Raw result already exists from Step 1B.
# We compare only the common folds 3-5 and summarize
# separately.
# =========================================================
raw_path = (
    PROCESSED
    /
    "direct_7day_forecast_results.csv"
)

if raw_path.exists():

    raw_results = pd.read_csv(
        raw_path
    )

    common_raw = raw_results[
        raw_results["fold"].isin(
            folds_df
            if False
            else [3, 4, 5]
        )
    ].copy()

    raw_common_wape = (
        common_raw[
            "WAPE"
        ].mean()
    )

    adjusted_wape = (
        results_df[
            "WAPE"
        ].mean()
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "RAW VS ADJUSTED-DEMAND FORECAST"
    )

    print(
        "=" * 110
    )

    print(
        "Raw observed-sales direct WAPE "
        "(common folds 3-5):",
        round(
            raw_common_wape,
            4
        )
    )

    print(
        "Adjusted-demand direct WAPE "
        "(folds 3-5):",
        round(
            adjusted_wape,
            4
        )
    )

    print(
        "Important:",
        "These are different targets and should NOT "
        "be interpreted as a head-to-head accuracy contest."
    )

else:

    print(
        "\nRaw direct forecast result file not found."
    )

# =========================================================
# 17. LEAKAGE CHECK
# =========================================================
print("\n" + "=" * 110)
print("LEAKAGE CHECK")
print("=" * 110)

print(
    "Origin features only:",
    True
)

print(
    "Future actual adjusted demand used as predictors:",
    False
)

print(
    "Future actual sales used as predictors:",
    False
)

print(
    "Future actual stockout used as predictors:",
    False
)

print(
    "Recursive predictions used:",
    False
)

print(
    "Final evaluation set used:",
    False
)

# =========================================================
# 18. COVERAGE CHECK
# =========================================================
print("\n" + "=" * 110)
print("COVERAGE CHECK")
print("=" * 110)

print(
    "Expected folds:",
    3
)

print(
    "Actual folds:",
    results_df[
        "fold"
    ].nunique()
)

print(
    "Expected horizons:",
    7
)

print(
    "Actual horizons:",
    results_df[
        "horizon"
    ].nunique()
)

print(
    "Expected model results:",
    3 * 7
)

print(
    "Actual model results:",
    len(results_df)
)

# =========================================================
# 19. SANITY CHECKS
# =========================================================
assert (
    results_df[
        "fold"
    ].nunique()
    ==
    3
)

assert (
    results_df[
        "horizon"
    ].nunique()
    ==
    7
)

assert (
    len(results_df)
    ==
    21
)

assert (
    results_df["rows"]
    ==
    n_series
).all()

assert (
    results_df[
        [
            "MAE",
            "WAPE",
            "RMSE"
        ]
    ]
    .isna()
    .any()
    .any()
    ==
    False
)

assert (
    results_df["MAE"] >= 0
).all()

assert (
    results_df["WAPE"] >= 0
).all()

assert (
    results_df["RMSE"] >= 0
).all()

print(
    "\nAll adjusted-demand direct forecast checks: PASS"
)

# =========================================================
# 20. SAVE
# =========================================================
output_path = (
    PROCESSED
    /
    "direct_7day_adjusted_demand_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print(
    "\nSaved:",
    output_path
)

print("\n" + "=" * 110)
print("PHASE 3 STEP 2 COMPLETE")
print("=" * 110)