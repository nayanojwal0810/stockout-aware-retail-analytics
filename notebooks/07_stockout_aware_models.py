from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 100)
print("PHASE 2 - STEP 3: STOCKOUT-AWARE MODEL EVALUATION")
print("=" * 100)

# =========================================================
# 1. LOAD FEATURE TABLE
# =========================================================
feature_path = (
    PROCESSED
    / "train_features.parquet"
)

df = pd.read_parquet(
    feature_path
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

print("\nFeature table:", df.shape)

# =========================================================
# 2. MODEL FEATURE SETS
# =========================================================

# Historical demand information
demand_features = [
    "lag_1_sales",
    "lag_2_sales",
    "lag_7_sales",
    "lag_14_sales",
    "rolling_7_mean_sales",
    "rolling_14_mean_sales",
    "rolling_28_mean_sales",
    "rolling_7_std_sales",
    "rolling_7_cv_sales",
    "recent_7_vs_14_ratio"
]

# Historical stockout information
stockout_features = [
    "lag_1_stockout_hours",
    "lag_7_stockout_hours",
    "lag_1_stockout_flag",
    "rolling_7_stockout_hours",
    "rolling_14_stockout_hours",
    "rolling_7_stockout_days",
    "rolling_14_stockout_days"
]

# Calendar information known at forecast time
calendar_features = [
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "is_weekend",
    "holiday_flag",
    "activity_flag"
]

# Historical discount
promotion_features = [
    "lag_1_discount",
    "lag_7_discount"
]

demand_only_features = (
    demand_features
    +
    calendar_features
    +
    promotion_features
)

stockout_aware_features = (
    demand_features
    +
    stockout_features
    +
    calendar_features
    +
    promotion_features
)

print("\nDemand-only features:", len(
    demand_only_features
))

print(
    "Stockout-aware features:",
    len(stockout_aware_features)
)

# =========================================================
# 3. VALIDATION FOLDS
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
    }
]

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
            (actual - predicted) ** 2
        )
    )


# =========================================================
# 5. MODEL FACTORY
# =========================================================
def build_model():

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=False
    )


# =========================================================
# 6. RESULTS STORAGE
# =========================================================
results = []

# =========================================================
# 7. ROLLING VALIDATION
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

    train_part = df[
        df["dt"] <= train_end
    ].copy()

    valid_part = df[
        df["dt"].between(
            valid_start,
            valid_end
        )
    ].copy()

    print("\n" + "=" * 100)
    print(
        f"FOLD {fold}: "
        f"Train <= {train_end.date()} | "
        f"Valid {valid_start.date()} -> {valid_end.date()}"
    )
    print("=" * 100)

    print(
        "Train rows:",
        len(train_part)
    )

    print(
        "Validation rows:",
        len(valid_part)
    )

    actual = (
        valid_part["sale_amount"]
        .to_numpy(
            dtype=np.float32
        )
    )

    # -----------------------------------------------------
    # BASELINE: 7-DAY MOVING AVERAGE
    # -----------------------------------------------------
    baseline_valid = valid_part[
        [
            "sale_amount",
            "rolling_7_mean_sales"
        ]
    ].dropna()

    baseline_actual = (
        baseline_valid["sale_amount"]
        .to_numpy(
            dtype=np.float32
        )
    )

    baseline_pred = np.maximum(
        baseline_valid[
            "rolling_7_mean_sales"
        ].to_numpy(
            dtype=np.float32
        ),
        0
    )

    baseline_mae = mae(
        baseline_actual,
        baseline_pred
    )

    baseline_wape = wape(
        baseline_actual,
        baseline_pred
    )

    baseline_rmse = rmse(
        baseline_actual,
        baseline_pred
    )

    print(
        "\n7-Day Moving Average"
    )

    print(
        f"MAE={baseline_mae:.4f} "
        f"WAPE={baseline_wape:.4f} "
        f"RMSE={baseline_rmse:.4f}"
    )

    results.append({
        "fold": fold,
        "model": "Moving_Average_7",
        "MAE": baseline_mae,
        "WAPE": baseline_wape,
        "RMSE": baseline_rmse
    })

    # -----------------------------------------------------
    # DEMAND-ONLY MODEL
    # -----------------------------------------------------
    demand_train = train_part[
        demand_only_features
        +
        ["sale_amount"]
    ]

    demand_valid = valid_part[
        demand_only_features
        +
        ["sale_amount"]
    ]

    train_mask = (
        demand_train["sale_amount"]
        .notna()
    )

    # HGB can handle NaNs in predictors,
    # but target itself must be valid.
    demand_train = demand_train[
        train_mask
    ]

    model = build_model()

    X_train = demand_train[
        demand_only_features
    ].to_numpy(
        dtype=np.float32
    )

    y_train = demand_train[
        "sale_amount"
    ].to_numpy(
        dtype=np.float32
    )

    X_valid = demand_valid[
        demand_only_features
    ].to_numpy(
        dtype=np.float32
    )

    y_valid = demand_valid[
        "sale_amount"
    ].to_numpy(
        dtype=np.float32
    )

    model.fit(
        X_train,
        y_train
    )

    demand_pred = model.predict(
        X_valid
    )

    demand_pred = np.maximum(
        demand_pred,
        0
    )

    demand_mae = mae(
        y_valid,
        demand_pred
    )

    demand_wape = wape(
        y_valid,
        demand_pred
    )

    demand_rmse = rmse(
        y_valid,
        demand_pred
    )

    print(
        "\nDemand-Only Gradient Boosting"
    )

    print(
        f"MAE={demand_mae:.4f} "
        f"WAPE={demand_wape:.4f} "
        f"RMSE={demand_rmse:.4f}"
    )

    results.append({
        "fold": fold,
        "model": "GB_Demand_Only",
        "MAE": demand_mae,
        "WAPE": demand_wape,
        "RMSE": demand_rmse
    })

    # -----------------------------------------------------
    # STOCKOUT-AWARE MODEL
    # -----------------------------------------------------
    stockout_train = train_part[
        stockout_aware_features
        +
        ["sale_amount"]
    ]

    stockout_valid = valid_part[
        stockout_aware_features
        +
        ["sale_amount"]
    ]

    train_mask = (
        stockout_train["sale_amount"]
        .notna()
    )

    stockout_train = stockout_train[
        train_mask
    ]

    model = build_model()

    X_train = stockout_train[
        stockout_aware_features
    ].to_numpy(
        dtype=np.float32
    )

    y_train = stockout_train[
        "sale_amount"
    ].to_numpy(
        dtype=np.float32
    )

    X_valid = stockout_valid[
        stockout_aware_features
    ].to_numpy(
        dtype=np.float32
    )

    y_valid = stockout_valid[
        "sale_amount"
    ].to_numpy(
        dtype=np.float32
    )

    model.fit(
        X_train,
        y_train
    )

    stockout_pred = model.predict(
        X_valid
    )

    stockout_pred = np.maximum(
        stockout_pred,
        0
    )

    stockout_mae = mae(
        y_valid,
        stockout_pred
    )

    stockout_wape = wape(
        y_valid,
        stockout_pred
    )

    stockout_rmse = rmse(
        y_valid,
        stockout_pred
    )

    print(
        "\nStockout-Aware Gradient Boosting"
    )

    print(
        f"MAE={stockout_mae:.4f} "
        f"WAPE={stockout_wape:.4f} "
        f"RMSE={stockout_rmse:.4f}"
    )

    results.append({
        "fold": fold,
        "model": "GB_Stockout_Aware",
        "MAE": stockout_mae,
        "WAPE": stockout_wape,
        "RMSE": stockout_rmse
    })

    # -----------------------------------------------------
    # STOCKOUT MODEL IMPROVEMENT
    # -----------------------------------------------------
    wape_improvement_vs_demand = (
        (
            demand_wape
            -
            stockout_wape
        )
        /
        demand_wape
        *
        100
    )

    wape_improvement_vs_baseline = (
        (
            baseline_wape
            -
            stockout_wape
        )
        /
        baseline_wape
        *
        100
    )

    print(
        "\nStockout-value comparison:"
    )

    print(
        "WAPE improvement vs demand-only:",
        f"{wape_improvement_vs_demand:.2f}%"
    )

    print(
        "WAPE improvement vs MA-7:",
        f"{wape_improvement_vs_baseline:.2f}%"
    )

# =========================================================
# 8. CROSS-FOLD SUMMARY
# =========================================================
results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 100)
print("CROSS-FOLD MODEL SUMMARY")
print("=" * 100)

summary = (
    results_df
    .groupby("model")
    .agg(
        folds=("fold", "nunique"),
        mean_mae=("MAE", "mean"),
        std_mae=("MAE", "std"),
        mean_wape=("WAPE", "mean"),
        std_wape=("WAPE", "std"),
        mean_rmse=("RMSE", "mean")
    )
    .sort_values(
        "mean_wape"
    )
)

print(
    summary.round(4)
    .to_string()
)

# =========================================================
# 9. WAPE BY FOLD
# =========================================================
print("\n" + "=" * 100)
print("WAPE BY FOLD")
print("=" * 100)

wape_table = (
    results_df
    .pivot(
        index="fold",
        columns="model",
        values="WAPE"
    )
)

print(
    wape_table.round(4)
    .to_string()
)

# =========================================================
# 10. PAIRED STOCKOUT IMPROVEMENT
# =========================================================
paired = (
    results_df[
        results_df["model"].isin(
            [
                "GB_Demand_Only",
                "GB_Stockout_Aware"
            ]
        )
    ]
    .pivot(
        index="fold",
        columns="model",
        values="WAPE"
    )
)

paired["wape_improvement_pct"] = (
    (
        paired["GB_Demand_Only"]
        -
        paired["GB_Stockout_Aware"]
    )
    /
    paired["GB_Demand_Only"]
    *
    100
)

print("\n" + "=" * 100)
print("STOCKOUT FEATURE VALUE")
print("=" * 100)

print(
    paired.round(4)
    .to_string()
)

print(
    "\nMean WAPE improvement from adding stockout history:",
    f"{paired['wape_improvement_pct'].mean():.2f}%"
)

print(
    "Folds where stockout-aware model improved:",
    int(
        (
            paired["wape_improvement_pct"]
            > 0
        ).sum()
    ),
        "/",
        len(paired)
)

# =========================================================
# 11. BEST MODEL
# =========================================================
best_model = (
    summary
    .sort_values(
        "mean_wape"
    )
    .index[0]
)

print("\n" + "=" * 100)
print("MODEL DECISION")
print("=" * 100)

print(
    "Best model by mean WAPE:",
    best_model
)

print(
    "Mean WAPE:",
    round(
        summary.loc[
            best_model,
            "mean_wape"
        ],
        4
    )
)

print(
    "Mean MAE:",
    round(
        summary.loc[
            best_model,
            "mean_mae"
        ],
        4
    )
)

# =========================================================
# 12. SAVE RESULTS
# =========================================================
output_path = (
    PROCESSED
    /
    "model_validation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    "Results saved to:",
    output_path
)

print(
    "Output exists:",
    output_path.exists()
)

# =========================================================
# 13. HOLDOUT PROTECTION
# =========================================================
print("\n" + "=" * 100)
print("HOLDOUT PROTECTION")
print("=" * 100)

print(
    "Official evaluation set used:",
    False
)

print(
    "Official evaluation period:",
    "2024-06-26 to 2024-07-02"
)

print("\n" + "=" * 100)
print("STOCKOUT-AWARE MODEL EVALUATION COMPLETE")
print("=" * 100)