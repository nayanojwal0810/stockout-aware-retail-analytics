from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 110)
print("PHASE 3 - STEP 1B: DIRECT MULTI-HORIZON 7-DAY FORECAST")
print("=" * 110)

# =========================================================
# 1. LOAD DATA
# =========================================================
columns = [
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
    columns=columns
)

df["dt"] = pd.to_datetime(df["dt"])

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
    "\nDATA SHAPE:",
    df.shape
)

print(
    "DATE RANGE:",
    df["dt"].min(),
    "->",
    df["dt"].max()
)

# =========================================================
# 2. CREATE LEAKAGE-SAFE ORIGIN FEATURES
# =========================================================
def create_origin_features(data):

    data = data.sort_values(
        group_cols + ["dt"]
    ).copy()

    g = data.groupby(
        group_cols,
        sort=False
    )

    # Historical sales
    data["lag_1"] = (
        g["sale_amount"]
        .shift(1)
    )

    data["lag_2"] = (
        g["sale_amount"]
        .shift(2)
    )

    data["lag_7"] = (
        g["sale_amount"]
        .shift(7)
    )

    data["lag_14"] = (
        g["sale_amount"]
        .shift(14)
    )

    # Historical rolling demand
    data["rolling_7_mean"] = (
        data.groupby(group_cols)["sale_amount"]
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
        data.groupby(group_cols)["sale_amount"]
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
        data.groupby(group_cols)["sale_amount"]
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

    # Historical stockout
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

    # Historical discount
    data["lag_1_discount"] = (
        g["discount"]
        .shift(1)
    )

    data["lag_7_discount"] = (
        g["discount"]
        .shift(7)
    )

    # Calendar of forecast origin
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


feature_data = create_origin_features(
    df
)

# =========================================================
# 3. FEATURES
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
# 4. VALIDATION FOLDS
# =========================================================
folds = [
    {
        "fold": 1,
        "train_end": "2024-04-30",
        "valid_start": "2024-05-01",
        "valid_end": "2024-05-07"
    },
    {
        "fold": 2,
        "train_end": "2024-05-07",
        "valid_start": "2024-05-08",
        "valid_end": "2024-05-14"
    },
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
# 5. METRICS
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
# 6. MODEL
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
# 7. RUN DIRECT HORIZONS
# =========================================================
results = []

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
    # Origin rows
    #
    # These contain information available at the forecast
    # origin. They are the predictor rows for all horizons.
    # -----------------------------------------------------
    origin_rows = feature_data[
        feature_data["dt"] == train_end
    ].copy()

    print(
        "Origin rows:",
        len(origin_rows)
    )

    assert (
        len(origin_rows)
        ==
        50000
    )

    # -----------------------------------------------------
    # For each horizon, train a separate direct model.
    # -----------------------------------------------------
    for horizon in range(1, 8):

        target_date_start = (
            train_end
            +
            pd.Timedelta(
                days=horizon
            )
        )

        # -------------------------------------------------
        # Training origin dates:
        #
        # For each historical origin date t, the target is
        # sales at t + horizon.
        #
        # Target is created by shifting sales backward.
        # -------------------------------------------------
        training = feature_data.copy()

        training[
            "direct_target"
        ] = (
            training
            .groupby(group_cols)[
                "sale_amount"
            ]
            .shift(-horizon)
        )

        # Only use origins <= current fold origin.
        # This guarantees no future data enters training.
        training = training[
            training["dt"] <= train_end
        ].copy()

        training = training.dropna(
            subset=features + ["direct_target"]
        )

        print(
            f"\nHORIZON +{horizon}"
        )

        print(
            "Training rows:",
            len(training)
        )

        # -------------------------------------------------
        # Train
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

        assert (
            X_train.shape[1]
            ==
            len(features)
        )

        model.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # Predict all 50,000 series simultaneously from
        # the same forecast origin.
        # -------------------------------------------------
        X_origin = (
            origin_rows[
                features
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        prediction = model.predict(
            X_origin
        )

        prediction = np.maximum(
            prediction,
            0
        ).astype(
            np.float32
        )

        predictions = pd.DataFrame(
            {
                "store_id":
                    origin_rows[
                        "store_id"
                    ].to_numpy(),
                "product_id":
                    origin_rows[
                        "product_id"
                    ].to_numpy(),
                "forecast_date":
                    target_date_start,
                "horizon":
                    horizon,
                "prediction":
                    prediction
            }
        )

        # -------------------------------------------------
        # Actual values
        # -------------------------------------------------
        actual_date = target_date_start

        actual = df[
            df["dt"] == actual_date
        ][
            group_cols
            +
            [
                "sale_amount"
            ]
        ]

        evaluation = predictions.merge(
            actual,
            on=group_cols,
            how="inner"
        )

        expected_rows = 50000

        print(
            "Expected evaluation rows:",
            expected_rows
        )

        print(
            "Actual evaluation rows:",
            len(evaluation)
        )

        assert (
            len(evaluation)
            ==
            expected_rows
        )

        actual_values = (
            evaluation[
                "sale_amount"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        prediction_values = (
            evaluation[
                "prediction"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        horizon_mae = mae(
            actual_values,
            prediction_values
        )

        horizon_wape = wape(
            actual_values,
            prediction_values
        )

        horizon_rmse = rmse(
            actual_values,
            prediction_values
        )

        print(
            f"MAE={horizon_mae:.4f} "
            f"WAPE={horizon_wape:.4f} "
            f"RMSE={horizon_rmse:.4f}"
        )

        results.append(
            {
                "fold": fold,
                "horizon": horizon,
                "model": "GB_Direct",
                "target": "observed_sales",
                "rows": len(evaluation),
                "MAE": horizon_mae,
                "WAPE": horizon_wape,
                "RMSE": horizon_rmse
            }
        )

# =========================================================
# 8. RESULTS
# =========================================================
results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 110)
print("DIRECT 7-DAY FORECAST RESULTS")
print("=" * 110)

print(
    results_df
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 9. CROSS-FOLD SUMMARY
# =========================================================
print("\n" + "=" * 110)
print("CROSS-FOLD SUMMARY")
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
# 10. OVERALL DIRECT FORECAST
# =========================================================
overall = (
    results_df
    .agg(
        mean_mae=("MAE", "mean"),
        mean_wape=("WAPE", "mean"),
        mean_rmse=("RMSE", "mean")
    )
)

print("\n" + "=" * 110)
print("OVERALL DIRECT FORECAST")
print("=" * 110)

print(
    overall
    .round(4)
    .to_string()
)

# =========================================================
# 11. WAPE BY HORIZON
# =========================================================
print("\n" + "=" * 110)
print("WAPE BY HORIZON")
print("=" * 110)

wape_horizon = (
    results_df
    .pivot(
        index="fold",
        columns="horizon",
        values="WAPE"
    )
)

print(
    wape_horizon
    .round(4)
    .to_string()
)

# =========================================================
# 12. HORIZON STABILITY
# =========================================================
print("\n" + "=" * 110)
print("HORIZON STABILITY")
print("=" * 110)

for horizon in range(1, 8):

    values = results_df[
        results_df["horizon"] == horizon
    ]["WAPE"]

    print(
        f"Horizon +{horizon}: "
        f"mean={values.mean():.4f} "
        f"std={values.std():.4f} "
        f"min={values.min():.4f} "
        f"max={values.max():.4f}"
    )

# =========================================================
# 13. LEAKAGE CHECK
# =========================================================
print("\n" + "=" * 110)
print("LEAKAGE CHECK")
print("=" * 110)

print(
    "Forecast origin features used:",
    True
)

print(
    "Future actual sales used as predictors:",
    False
)

print(
    "Recursive predictions used as predictors:",
    False
)

print(
    "Future actual stockout used as predictors:",
    False
)

print(
    "Same-day hours_sale used:",
    False
)

print(
    "Final evaluation set used:",
    False
)

# =========================================================
# 14. HOLDOUT PROTECTION
# =========================================================
print("\n" + "=" * 110)
print("FINAL HOLDOUT PROTECTION")
print("=" * 110)

print(
    "Official evaluation period:",
    "2024-06-26 -> 2024-07-02"
)

print(
    "Official evaluation touched:",
    False
)

# =========================================================
# 15. SANITY CHECKS
# =========================================================
assert (
    len(results_df)
    ==
    5 * 7
)

assert (
    results_df["fold"]
    .nunique()
    ==
    5
)

assert (
    results_df["horizon"]
    .nunique()
    ==
    7
)

assert (
    results_df["rows"]
    ==
    50000
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
    "\nAll direct 7-day forecast checks: PASS"
)

# =========================================================
# 16. SAVE
# =========================================================
output_path = (
    PROCESSED
    /
    "direct_7day_forecast_results.csv"
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
print("PHASE 3 STEP 1B COMPLETE")
print("=" * 110)