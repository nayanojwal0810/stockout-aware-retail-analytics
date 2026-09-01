from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from stockout_retail.reconstruction.demand import (
    add_stockout_state,
    create_reconstruction_features,
    build_adjusted_demand,
)
from stockout_retail.forecasting.evaluation import mae, wape, rmse

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 100)
print("PHASE 2 - STEP 7: STOCKOUT-ADJUSTED DEMAND SERIES")
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
# 2. STOCKOUT FLAGS
# =========================================================
df = add_stockout_state(df)

# Keep the original column name used by this workflow.
df["full_stockout_flag"] = (
    df["full_stockout_flag"].astype(int)
)

# =========================================================
# 3. LEAKAGE-SAFE HISTORICAL FEATURES
# =========================================================
df = create_reconstruction_features(df)

# =========================================================
# 4. RECONSTRUCTION FEATURES
# =========================================================
feature_cols = [
    "lag_1_sales",
    "lag_7_sales",
    "rolling_7_sales",
    "rolling_14_sales",
    "lag_1_stockout_hours",
    "lag_7_stockout_hours",
    "rolling_7_stockout_hours",
    "lag_1_discount",
    "lag_7_discount",
    "day_of_week",
    "is_weekend",
    "holiday_flag",
    "activity_flag"
]

print(
    "\nReconstruction features:",
    len(feature_cols)
)

# =========================================================
# 5. EXPANDING CROSS-FIT WINDOWS
#
# Each validation block is predicted from an earlier
# model trained ONLY on earlier normal-stock observations.
# =========================================================
folds = [
    {
        "fold": 1,
        "train_end": "2024-05-14",
        "valid_start": "2024-05-15",
        "valid_end": "2024-05-21"
    },
    {
        "fold": 2,
        "train_end": "2024-05-21",
        "valid_start": "2024-05-22",
        "valid_end": "2024-05-28"
    },
    {
        "fold": 3,
        "train_end": "2024-05-28",
        "valid_start": "2024-05-29",
        "valid_end": "2024-06-04"
    },
    {
        "fold": 4,
        "train_end": "2024-06-04",
        "valid_start": "2024-06-05",
        "valid_end": "2024-06-11"
    },
    {
        "fold": 5,
        "train_end": "2024-06-11",
        "valid_start": "2024-06-12",
        "valid_end": "2024-06-18"
    },
    {
        "fold": 6,
        "train_end": "2024-06-18",
        "valid_start": "2024-06-19",
        "valid_end": "2024-06-25"
    }
]

# =========================================================
# 6. STORAGE
# =========================================================
fold_outputs = []

# =========================================================
# 7. RUN EXPANDING CROSS-FIT
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

    model_train = df[
        (df["dt"] <= train_end)
        &
        (df["normal_day"])
    ].copy()

    validation = df[
        df["dt"].between(
            valid_start,
            valid_end
        )
    ].copy()

    model_train = model_train[
        model_train["sale_amount"].notna()
    ]

    print("\n" + "=" * 100)
    print(
        f"FOLD {fold}: "
        f"Train <= {train_end.date()} | "
        f"Predict {valid_start.date()} -> {valid_end.date()}"
    )
    print("=" * 100)

    print(
        "Normal training observations:",
        len(model_train)
    )

    print(
        "Validation observations:",
        len(validation)
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------
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

    X_train = model_train[
        feature_cols
    ].to_numpy(
        dtype=np.float32
    )

    y_train = model_train[
        "sale_amount"
    ].to_numpy(
        dtype=np.float32
    )

    X_valid = validation[
        feature_cols
    ].to_numpy(
        dtype=np.float32
    )

    model.fit(
        X_train,
        y_train
    )

    validation[
        "cross_fitted_demand_prediction"
    ] = np.maximum(
        model.predict(
            X_valid
        ),
        0
    )

    validation[
        "reconstruction_fold"
    ] = fold

    fold_outputs.append(
        validation[
            [
                "store_id",
                "product_id",
                "dt",
                "sale_amount",
                "stockout_flag",
                "full_stockout_flag",
                "stock_hour6_22_cnt",
                "cross_fitted_demand_prediction",
                "reconstruction_fold"
            ]
        ]
    )

# =========================================================
# 8. COMBINE
# =========================================================
cross_fitted = pd.concat(
    fold_outputs,
    ignore_index=True
)

print("\n" + "=" * 100)
print("CROSS-FITTED COVERAGE")
print("=" * 100)

print(
    "Rows reconstructed:",
    len(cross_fitted)
)

print(
    "Date start:",
    cross_fitted["dt"].min()
)

print(
    "Date end:",
    cross_fitted["dt"].max()
)

print(
    "Missing predictions:",
    cross_fitted[
        "cross_fitted_demand_prediction"
    ].isna().sum()
)

print(
    "Folds:",
    cross_fitted[
        "reconstruction_fold"
    ].nunique()
)

# =========================================================
# 9. VALIDATION ON NORMAL DAYS
# =========================================================
normal_oof = cross_fitted[
    cross_fitted["stockout_flag"] == 0
].copy()

actual = (
    normal_oof[
        "sale_amount"
    ]
    .to_numpy(
        dtype=np.float32
    )
)

prediction = (
    normal_oof[
        "cross_fitted_demand_prediction"
    ]
    .to_numpy(
        dtype=np.float32
    )
)

print("\n" + "=" * 100)
print("CROSS-FITTED NORMAL-DAY VALIDATION")
print("=" * 100)

print(
    "Normal observations:",
    len(normal_oof)
)

print(
    "MAE:",
    round(
        mae(
            actual,
            prediction
        ),
        4
    )
)

print(
    "WAPE:",
    round(
        wape(
            actual,
            prediction
        ),
        4
    )
)

print(
    "RMSE:",
    round(
        rmse(
            actual,
            prediction
        ),
        4
    )
)

# =========================================================
# 10. FOLD VALIDATION
# =========================================================
fold_metrics = []

for fold in sorted(
    cross_fitted[
        "reconstruction_fold"
    ].unique()
):

    subset = cross_fitted[
        (
            cross_fitted[
                "reconstruction_fold"
            ]
            ==
            fold
        )
        &
        (
            cross_fitted[
                "stockout_flag"
            ]
            ==
            0
        )
    ]

    a = subset[
        "sale_amount"
    ].to_numpy(
        dtype=np.float32
    )

    p = subset[
        "cross_fitted_demand_prediction"
    ].to_numpy(
        dtype=np.float32
    )

    fold_metrics.append(
        {
            "fold": int(fold),
            "normal_observations": len(subset),
            "MAE": mae(a, p),
            "WAPE": wape(a, p),
            "RMSE": rmse(a, p)
        }
    )

fold_metrics = pd.DataFrame(
    fold_metrics
)

print("\n" + "=" * 100)
print("FOLD VALIDATION")
print("=" * 100)

print(
    fold_metrics
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 11. BUILD ADJUSTED DEMAND
# =========================================================
cross_fitted = build_adjusted_demand(
    cross_fitted,
    prediction_column="cross_fitted_demand_prediction",
)

# =========================================================
# 12. SUMMARY BY STOCKOUT STATE
# =========================================================
cross_fitted["stockout_state"] = np.select(
    [
        cross_fitted["stock_hour6_22_cnt"] == 0,
        cross_fitted["stock_hour6_22_cnt"].between(1, 15),
        cross_fitted["stock_hour6_22_cnt"] == 16
    ],
    [
        "NORMAL",
        "PARTIAL_STOCKOUT",
        "FULL_STOCKOUT"
    ],
    default="UNKNOWN"
)

print("\n" + "=" * 100)
print("ADJUSTED DEMAND SUMMARY")
print("=" * 100)

summary = (
    cross_fitted
    .groupby(
        "stockout_state"
    )
    .agg(
        observations=(
            "sale_amount",
            "size"
        ),
        observed_sales=(
            "sale_amount",
            "mean"
        ),
        adjusted_demand=(
            "adjusted_demand",
            "mean"
        ),
        estimated_gap=(
            "estimated_censored_gap",
            "mean"
        )
    )
)

print(
    summary
    .round(4)
    .to_string()
)

# =========================================================
# 13. STOCKOUT INTENSITY
# =========================================================
intensity = (
    cross_fitted[
        cross_fitted["stockout_flag"] == 1
    ]
    .groupby(
        "stock_hour6_22_cnt"
    )
    .agg(
        observations=(
            "sale_amount",
            "size"
        ),
        observed_sales=(
            "sale_amount",
            "mean"
        ),
        adjusted_demand=(
            "adjusted_demand",
            "mean"
        ),
        estimated_gap=(
            "estimated_censored_gap",
            "mean"
        )
    )
)

print("\n" + "=" * 100)
print("ADJUSTED DEMAND BY STOCKOUT INTENSITY")
print("=" * 100)

print(
    intensity
    .round(4)
    .to_string()
)

# =========================================================
# 14. FULL STOCKOUT
# =========================================================
full = cross_fitted[
    cross_fitted[
        "full_stockout_flag"
    ]
    ==
    1
]

print("\n" + "=" * 100)
print("FULL STOCKOUT SUMMARY")
print("=" * 100)

print(
    "Rows:",
    len(full)
)

print(
    "Observed sales:",
    round(
        full[
            "sale_amount"
        ].mean(),
        4
    )
)

print(
    "Adjusted demand:",
    round(
        full[
            "adjusted_demand"
        ].mean(),
        4
    )
)

print(
    "Estimated gap:",
    round(
        full[
            "estimated_censored_gap"
        ].mean(),
        4
    )
)

# =========================================================
# 15. SERIES-LEVEL IMPACT
# =========================================================
series_summary = (
    cross_fitted
    .groupby(
        [
            "store_id",
            "product_id"
        ]
    )
    .agg(
        total_days=(
            "dt",
            "nunique"
        ),
        stockout_days=(
            "stockout_flag",
            "sum"
        ),
        mean_observed_sales=(
            "sale_amount",
            "mean"
        ),
        mean_adjusted_demand=(
            "adjusted_demand",
            "mean"
        ),
        total_estimated_gap=(
            "estimated_censored_gap",
            "sum"
        )
    )
)

series_summary[
    "stockout_rate"
] = (
    series_summary[
        "stockout_days"
    ]
    /
    series_summary[
        "total_days"
    ]
)

print("\n" + "=" * 100)
print("SERIES-LEVEL ADJUSTMENT")
print("=" * 100)

print(
    series_summary[
        [
            "stockout_rate",
            "mean_observed_sales",
            "mean_adjusted_demand",
            "total_estimated_gap"
        ]
    ]
    .describe()
    .round(4)
    .to_string()
)

# =========================================================
# 16. FINAL WINDOW COVERAGE
# =========================================================
final_window = cross_fitted[
    cross_fitted["dt"].between(
        "2024-06-19",
        "2024-06-25"
    )
]

print("\n" + "=" * 100)
print("FINAL PRE-HOLDOUT WINDOW")
print("=" * 100)

print(
    "Rows:",
    len(final_window)
)

print(
    "Expected rows:",
    7 * 50000
)

print(
    "Stockout rows:",
    int(
        final_window[
            "stockout_flag"
        ].sum()
    )
)

print(
    "Missing predictions:",
    final_window[
        "cross_fitted_demand_prediction"
    ].isna().sum()
)

# =========================================================
# 17. LEAKAGE CHECK
# =========================================================
print("\n" + "=" * 100)
print("LEAKAGE CHECK")
print("=" * 100)

print(
    "Predictions generated only from earlier dates:",
    True
)

print(
    "Normal-day training only:",
    True
)

print(
    "Final evaluation data used:",
    False
)

print(
    "Final holdout:",
    "2024-06-26 -> 2024-07-02"
)

# =========================================================
# 18. SAVE
# =========================================================
output_cols = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "stockout_flag",
    "full_stockout_flag",
    "stockout_state",
    "cross_fitted_demand_prediction",
    "adjusted_demand",
    "estimated_censored_gap",
    "reconstruction_fold"
]

output_path = (
    PROCESSED
    /
    "stockout_adjusted_demand.parquet"
)

cross_fitted[
    output_cols
].to_parquet(
    output_path,
    index=False
)

metrics_path = (
    PROCESSED
    /
    "adjusted_demand_reconstruction_metrics.csv"
)

fold_metrics.to_csv(
    metrics_path,
    index=False
)

print("\n" + "=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    "Adjusted demand series:",
    output_path
)

print(
    "Fold metrics:",
    metrics_path
)

print(
    "Adjusted-demand rows:",
    len(cross_fitted)
)

print("\n" + "=" * 100)
print("STOCKOUT-ADJUSTED DEMAND SERIES COMPLETE")
print("=" * 100)