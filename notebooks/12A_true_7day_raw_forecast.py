from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 110)
print("PHASE 3 - STEP 1A: TRUE 7-DAY OBSERVED-SALES FORECAST")
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

series_index = (
    df[group_cols]
    .drop_duplicates()
    .sort_values(group_cols)
    .set_index(group_cols)
    .index
)

n_series = len(series_index)

print("\nDATA SHAPE:", df.shape)
print("SERIES:", n_series)
print("DATE RANGE:", df["dt"].min(), "->", df["dt"].max())

# =========================================================
# 2. LEAKAGE-SAFE FEATURE CONSTRUCTION
# =========================================================
def create_features(data):

    data = data.sort_values(
        group_cols + ["dt"]
    ).copy()

    grouped = data.groupby(
        group_cols,
        sort=False
    )

    # Historical sales
    data["lag_1"] = (
        grouped["sale_amount"]
        .shift(1)
    )

    data["lag_2"] = (
        grouped["sale_amount"]
        .shift(2)
    )

    data["lag_7"] = (
        grouped["sale_amount"]
        .shift(7)
    )

    data["lag_14"] = (
        grouped["sale_amount"]
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
        grouped["stock_hour6_22_cnt"]
        .shift(1)
    )

    data["lag_7_stockout_hours"] = (
        grouped["stock_hour6_22_cnt"]
        .shift(7)
    )

    data["rolling_7_stockout_hours"] = (
        data.groupby(group_cols)["stock_hour6_22_cnt"]
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
        grouped["discount"]
        .shift(1)
    )

    data["lag_7_discount"] = (
        grouped["discount"]
        .shift(7)
    )

    # Calendar
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
    df
)

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
# 3. MODEL
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
# 4. METRICS
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
# 5. VALIDATION FOLDS
#
# Genuine 7-day recursive forecasts.
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
# 6. VECTORIZED RECURSIVE FORECAST
# =========================================================
def recursive_forecast(
    model,
    history,
    forecast_dates
):

    history = history.sort_values(
        group_cols + ["dt"]
    ).copy()

    # -----------------------------------------------------
    # History matrices
    # -----------------------------------------------------
    history_dates = (
        np.array(
            sorted(
                history["dt"].unique()
            )
        )
    )

    target_matrix = (
        history
        .pivot(
            index=group_cols,
            columns="dt",
            values="sale_amount"
        )
        .reindex(
            series_index,
            columns=history_dates
        )
    )

    stockout_matrix = (
        history
        .pivot(
            index=group_cols,
            columns="dt",
            values="stock_hour6_22_cnt"
        )
        .reindex(
            series_index,
            columns=history_dates
        )
    )

    discount_matrix = (
        history
        .pivot(
            index=group_cols,
            columns="dt",
            values="discount"
        )
        .reindex(
            series_index,
            columns=history_dates
        )
    )

    target_values = (
        target_matrix
        .to_numpy(
            dtype=np.float32
        )
    )

    stockout_values = (
        stockout_matrix
        .to_numpy(
            dtype=np.float32
        )
    )

    discount_values = (
        discount_matrix
        .to_numpy(
            dtype=np.float32
        )
    )

    # -----------------------------------------------------
    # All series should have complete history.
    # -----------------------------------------------------
    valid_series = (
        np.sum(
            ~np.isnan(
                target_values
            ),
            axis=1
        )
        >=
        14
    )

    if not valid_series.all():

        print(
            "Series with <14 valid observations:",
            int(
                (~valid_series).sum()
            )
        )

    target_values = (
        target_values[
            valid_series
        ]
    )

    stockout_values = (
        stockout_values[
            valid_series
        ]
    )

    discount_values = (
        discount_values[
            valid_series
        ]
    )

    forecast_index = (
        series_index[
            valid_series
        ]
    )

    # The raw dataset has complete daily observations.
    # Remaining NaN values should only come from malformed
    # input, so fail rather than silently impute.
    if np.isnan(target_values).any():

        raise ValueError(
            "Unexpected missing historical sales detected."
        )

    target_values = target_values.astype(
        np.float32
    )

    stockout_values = np.nan_to_num(
        stockout_values,
        nan=0.0
    ).astype(
        np.float32
    )

    discount_values = np.nan_to_num(
        discount_values,
        nan=0.0
    ).astype(
        np.float32
    )

    predictions = []

    # =====================================================
    # TRUE RECURSIVE FORECAST
    # =====================================================
    for step, forecast_date in enumerate(
        forecast_dates,
        start=1
    ):

        print(
            f"      Day {step}/{len(forecast_dates)} "
            f"-> {forecast_date.date()}"
        )

        # -------------------------------------------------
        # Lagged demand
        # -------------------------------------------------
        lag_1 = (
            target_values[:, -1]
        )

        lag_2 = (
            target_values[:, -2]
        )

        lag_7 = (
            target_values[:, -7]
        )

        lag_14 = (
            target_values[:, -14]
        )

        # -------------------------------------------------
        # Rolling demand
        # -------------------------------------------------
        rolling_7_mean = (
            np.mean(
                target_values[:, -7:],
                axis=1
            )
        )

        rolling_14_mean = (
            np.mean(
                target_values[:, -14:],
                axis=1
            )
        )

        rolling_7_std = (
            np.std(
                target_values[:, -7:],
                axis=1,
                ddof=1
            )
        )

        # -------------------------------------------------
        # Historical stockout
        # -------------------------------------------------
        lag_1_stockout = (
            stockout_values[:, -1]
        )

        lag_7_stockout = (
            stockout_values[:, -7]
        )

        rolling_7_stockout = (
            np.mean(
                stockout_values[:, -7:],
                axis=1
            )
        )

        # -------------------------------------------------
        # Historical discount
        # -------------------------------------------------
        lag_1_discount = (
            discount_values[:, -1]
        )

        lag_7_discount = (
            discount_values[:, -7]
        )

        # -------------------------------------------------
        # Forecast-date calendar
        # -------------------------------------------------
        day_of_week = (
            forecast_date.dayofweek
        )

        day_of_month = (
            forecast_date.day
        )

        week_of_year = int(
            forecast_date
            .isocalendar()
            .week
        )

        is_weekend = int(
            forecast_date.dayofweek >= 5
        )

        # -------------------------------------------------
        # Calendar values known for the forecast date
        # -------------------------------------------------
        context = (
            df[
                df["dt"] == forecast_date
            ][
                group_cols
                +
                [
                    "holiday_flag",
                    "activity_flag"
                ]
            ]
            .drop_duplicates(
                subset=group_cols
            )
            .set_index(
                group_cols
            )
            .reindex(
                forecast_index
            )
        )

        holiday = (
            context[
                "holiday_flag"
            ]
            .fillna(0)
            .to_numpy(
                dtype=np.float32
            )
        )

        activity = (
            context[
                "activity_flag"
            ]
            .fillna(0)
            .to_numpy(
                dtype=np.float32
            )
        )

        # -------------------------------------------------
        # Feature matrix
        # -------------------------------------------------
        X_future = np.column_stack(
            [
                lag_1,
                lag_2,
                lag_7,
                lag_14,
                rolling_7_mean,
                rolling_14_mean,
                rolling_7_std,
                lag_1_stockout,
                lag_7_stockout,
                rolling_7_stockout,
                lag_1_discount,
                lag_7_discount,
                np.full(
                    len(forecast_index),
                    day_of_week,
                    dtype=np.float32
                ),
                np.full(
                    len(forecast_index),
                    day_of_month,
                    dtype=np.float32
                ),
                np.full(
                    len(forecast_index),
                    week_of_year,
                    dtype=np.float32
                ),
                np.full(
                    len(forecast_index),
                    is_weekend,
                    dtype=np.float32
                ),
                holiday,
                activity
            ]
        ).astype(
            np.float32
        )

        # -------------------------------------------------
        # Defensive shape check
        # -------------------------------------------------
        assert (
            X_future.shape
            ==
            (
                len(forecast_index),
                len(features)
            )
        )

        # -------------------------------------------------
        # Predict all series together
        # -------------------------------------------------
        prediction = model.predict(
            X_future
        )

        prediction = np.maximum(
            prediction,
            0
        ).astype(
            np.float32
        )

        predictions.append(
            pd.DataFrame(
                {
                    "store_id":
                        forecast_index
                        .get_level_values(
                            "store_id"
                        ),
                    "product_id":
                        forecast_index
                        .get_level_values(
                            "product_id"
                        ),
                    "dt":
                        forecast_date,
                    "prediction":
                        prediction
                }
            )
        )

        # -------------------------------------------------
        # Recursive state update
        #
        # We do NOT append actual future sales.
        # The previous prediction becomes the next day's
        # lagged demand.
        #
        # Future stockout is unknown.
        # Therefore stockout history for predicted days
        # is set to zero.
        #
        # Future discount is not used directly.
        # Its future lagged value therefore becomes zero.
        # -------------------------------------------------
        target_values = np.column_stack(
            [
                target_values,
                prediction
            ]
        )

        stockout_values = np.column_stack(
            [
                stockout_values,
                np.zeros(
                    len(forecast_index),
                    dtype=np.float32
                )
            ]
        )

        discount_values = np.column_stack(
            [
                discount_values,
                np.zeros(
                    len(forecast_index),
                    dtype=np.float32
                )
            ]
        )

    return pd.concat(
        predictions,
        ignore_index=True
    )


# =========================================================
# 7. RUN FIVE TRUE 7-DAY FOLDS
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

    forecast_dates = pd.date_range(
        valid_start,
        valid_end,
        freq="D"
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
    # Training data only
    # -----------------------------------------------------
    train_part = model_data[
        model_data["dt"] <= train_end
    ].copy()

    train_part = train_part.dropna(
        subset=features
    )

    print(
        "\nTraining rows:",
        len(train_part)
    )

    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------
    model = build_model()

    X_train = (
        train_part[
            features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_train = (
        train_part[
            "sale_amount"
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

    print(
        "Training model..."
    )

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Recursive forecasting
    # -----------------------------------------------------
    history = df[
        df["dt"] <= train_end
    ].copy()

    predictions = recursive_forecast(
        model,
        history,
        forecast_dates
    )

    # -----------------------------------------------------
    # Actual future observations
    # -----------------------------------------------------
    actual = df[
        df["dt"].between(
            valid_start,
            valid_end
        )
    ][
        group_cols
        +
        [
            "dt",
            "sale_amount"
        ]
    ]

    # -----------------------------------------------------
    # Merge forecast + actual
    # -----------------------------------------------------
    evaluation = actual.merge(
        predictions,
        on=group_cols + ["dt"],
        how="inner"
    )

    evaluation = evaluation.dropna(
        subset=["prediction"]
    )

    expected_rows = (
        n_series
        *
        len(forecast_dates)
    )

    print(
        "\nExpected forecast rows:",
        expected_rows
    )

    print(
        "Actual forecast rows:",
        len(evaluation)
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    actual_values = (
        evaluation[
            "sale_amount"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    predicted_values = (
        evaluation[
            "prediction"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    fold_mae = mae(
        actual_values,
        predicted_values
    )

    fold_wape = wape(
        actual_values,
        predicted_values
    )

    fold_rmse = rmse(
        actual_values,
        predicted_values
    )

    print(
        "\nTRUE 7-DAY RAW-SALES FORECAST"
    )

    print(
        f"MAE={fold_mae:.4f} "
        f"WAPE={fold_wape:.4f} "
        f"RMSE={fold_rmse:.4f}"
    )

    # -----------------------------------------------------
    # Horizon-specific error
    # -----------------------------------------------------
    evaluation["horizon"] = (
        evaluation["dt"]
        -
        valid_start
    ).dt.days + 1

    horizon_results = []

    for horizon in sorted(
        evaluation["horizon"].unique()
    ):

        subset = evaluation[
            evaluation["horizon"]
            ==
            horizon
        ]

        a = subset[
            "sale_amount"
        ].to_numpy(
            dtype=np.float32
        )

        p = subset[
            "prediction"
        ].to_numpy(
            dtype=np.float32
        )

        horizon_results.append(
            {
                "horizon": int(horizon),
                "MAE": mae(a, p),
                "WAPE": wape(a, p),
                "RMSE": rmse(a, p)
            }
        )

    horizon_results = pd.DataFrame(
        horizon_results
    )

    print(
        "\nHORIZON ERROR"
    )

    print(
        horizon_results
        .round(4)
        .to_string(index=False)
    )

    results.append(
        {
            "fold": fold,
            "model": "GB_Raw_Sales",
            "target": "observed_sales",
            "rows": len(evaluation),
            "MAE": fold_mae,
            "WAPE": fold_wape,
            "RMSE": fold_rmse
        }
    )

# =========================================================
# 8. CROSS-FOLD SUMMARY
# =========================================================
results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 110)
print("CROSS-FOLD SUMMARY")
print("=" * 110)

summary = (
    results_df
    .groupby(
        [
            "model",
            "target"
        ]
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
# 9. FOLD-BY-FOLD WAPE
# =========================================================
print("\n" + "=" * 110)
print("WAPE BY FOLD")
print("=" * 110)

wape_table = (
    results_df
    .pivot(
        index="fold",
        columns="model",
        values="WAPE"
    )
)

print(
    wape_table
    .round(4)
    .to_string()
)

# =========================================================
# 10. VALIDATION STABILITY
# =========================================================
print("\n" + "=" * 110)
print("VALIDATION STABILITY")
print("=" * 110)

print(
    "WAPE min:",
    round(
        results_df["WAPE"].min(),
        4
    )
)

print(
    "WAPE max:",
    round(
        results_df["WAPE"].max(),
        4
    )
)

print(
    "WAPE range:",
    round(
        results_df["WAPE"].max()
        -
        results_df["WAPE"].min(),
        4
    )
)

# =========================================================
# 11. HOLDOUT PROTECTION
# =========================================================
print("\n" + "=" * 110)
print("HOLDOUT PROTECTION")
print("=" * 110)

print(
    "Official evaluation used:",
    False
)

print(
    "Official evaluation period:",
    "2024-06-26 -> 2024-07-02"
)

# =========================================================
# 12. METHODOLOGY CHECK
# =========================================================
print("\n" + "=" * 110)
print("METHODOLOGY CHECK")
print("=" * 110)

print(
    "Forecast horizon:",
    "7 days"
)

print(
    "Recursive forecasting:",
    True
)

print(
    "Future actual sales used:",
    False
)

print(
    "Future actual stockout used:",
    False
)

print(
    "Same-day hours_sale used:",
    False
)

print(
    "Final holdout touched:",
    False
)

print(
    "Prediction matrix:",
    "vectorized"
)

# =========================================================
# 13. SANITY CHECKS
# =========================================================
assert len(results_df) == 5

assert (
    results_df["rows"]
    ==
    n_series * 7
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
    (results_df["MAE"] >= 0).all()
)

assert (
    (results_df["WAPE"] >= 0).all()
)

assert (
    (results_df["RMSE"] >= 0).all()
)

print(
    "\nAll true 7-day forecast checks: PASS"
)

# =========================================================
# 14. SAVE
# =========================================================
output_path = (
    PROCESSED
    /
    "true_7day_raw_forecast_results.csv"
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
print("PHASE 3 STEP 1A COMPLETE")
print("=" * 110)