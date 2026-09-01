from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from stockout_retail.reconstruction.demand import (
    add_stockout_state,
    create_reconstruction_features,
)
from stockout_retail.forecasting.evaluation import mae, wape, rmse

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 100)
print("PHASE 2 - STEP 6: CROSS-FITTED DEMAND RECONSTRUCTION")
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

print("\nInput shape:", df.shape)

# =========================================================
# 2. STOCKOUT STATE
# =========================================================
df = add_stockout_state(df)

df["full_stockout"] = (
    df["full_stockout_flag"] == 1
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
    "rolling_7_stockout_hours",
    "day_of_week",
    "is_weekend",
    "lag_1_discount",
    "holiday_flag",
    "activity_flag"
]
print(
    "\nReconstruction features:",
    len(feature_cols)
)

# =========================================================
# 5. CROSS-FITTING WINDOWS
#
# Model for each validation block uses only NORMAL
# observations occurring before that block.
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
# 6. STORAGE FOR OOF PREDICTIONS
# =========================================================
oof_predictions = []

# =========================================================
# 7. CROSS-FITTED PREDICTION
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

    # Need target only for normal days during
    # reconstruction-model training.
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
    # FIT RECONSTRUCTION MODEL
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

    prediction = model.predict(
        X_valid
    )

    prediction = np.maximum(
        prediction,
        0
    )

    validation["cross_fitted_demand"] = (
        prediction
    )

    validation["reconstruction_fold"] = (
        fold
    )

    oof_predictions.append(
        validation[
            [
                "store_id",
                "product_id",
                "dt",
                "sale_amount",
                "stockout_flag",
                "full_stockout",
                "stock_hour6_22_cnt",
                "cross_fitted_demand",
                "reconstruction_fold"
            ]
        ]
    )

# =========================================================
# 8. COMBINE OOF PREDICTIONS
# =========================================================
oof = pd.concat(
    oof_predictions,
    ignore_index=True
)

print("\n" + "=" * 100)
print("CROSS-FITTED OUTPUT")
print("=" * 100)

print(
    "OOF observations:",
    len(oof)
)

print(
    "OOF dates:",
    oof["dt"].min(),
    "->",
    oof["dt"].max()
)

print(
    "Missing predictions:",
    oof["cross_fitted_demand"].isna().sum()
)

# =========================================================
# 9. EVALUATE RECONSTRUCTION ON KNOWN NORMAL DAYS
#
# These days have observed sales, so they provide
# genuine ground truth for evaluating reconstruction.
# =========================================================
normal_oof = oof[
    oof["stockout_flag"] == 0
].copy()

# =========================================================
# 10. METRICS

actual = (
    normal_oof["sale_amount"]
    .to_numpy(
        dtype=np.float32
    )
)

predicted = (
    normal_oof["cross_fitted_demand"]
    .to_numpy(
        dtype=np.float32
    )
)

print("\n" + "=" * 100)
print("CROSS-FITTED RECONSTRUCTION VALIDATION")
print("=" * 100)

print(
    "Known normal observations:",
    len(normal_oof)
)

print(
    "MAE:",
    round(
        mae(
            actual,
            predicted
        ),
        4
    )
)

print(
    "WAPE:",
    round(
        wape(
            actual,
            predicted
        ),
        4
    )
)

print(
    "RMSE:",
    round(
        rmse(
            actual,
            predicted
        ),
        4
    )
)

# =========================================================
# 11. OOF ERROR BY FOLD
# =========================================================
fold_results = []

for fold in sorted(
    normal_oof[
        "reconstruction_fold"
    ].unique()
):

    subset = normal_oof[
        normal_oof[
            "reconstruction_fold"
        ]
        ==
        fold
    ]

    a = subset[
        "sale_amount"
    ].to_numpy(
        dtype=np.float32
    )

    p = subset[
        "cross_fitted_demand"
    ].to_numpy(
        dtype=np.float32
    )

    fold_results.append(
        {
            "fold": int(fold),
            "observations": len(subset),
            "MAE": mae(a, p),
            "WAPE": wape(a, p),
            "RMSE": rmse(a, p)
        }
    )

fold_results = pd.DataFrame(
    fold_results
)

print("\n" + "=" * 100)
print("CROSS-FITTED ERROR BY FOLD")
print("=" * 100)

print(
    fold_results
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 12. APPLY OOF DEMAND ESTIMATE TO STOCKOUT DAYS
#
# IMPORTANT:
# These are predictions, NOT observed truth.
# =========================================================
stockout_oof = oof[
    oof["stockout_flag"] == 1
].copy()

stockout_oof[
    "estimated_censored_demand"
] = (
    stockout_oof[
        "cross_fitted_demand"
    ]
)

stockout_oof[
    "estimated_censored_gap"
] = (
    stockout_oof[
        "estimated_censored_demand"
    ]
    -
    stockout_oof[
        "sale_amount"
    ]
).clip(
    lower=0
)

print("\n" + "=" * 100)
print("CROSS-FITTED STOCKOUT ESTIMATION")
print("=" * 100)

print(
    "Stockout observations with OOF predictions:",
    len(stockout_oof)
)

print(
    "Mean observed sales:",
    round(
        stockout_oof[
            "sale_amount"
        ].mean(),
        4
    )
)

print(
    "Mean estimated demand:",
    round(
        stockout_oof[
            "estimated_censored_demand"
        ].mean(),
        4
    )
)

print(
    "Mean estimated censored gap:",
    round(
        stockout_oof[
            "estimated_censored_gap"
        ].mean(),
        4
    )
)

# =========================================================
# 13. FULL STOCKOUT ESTIMATION
# =========================================================
full_oof = stockout_oof[
    stockout_oof["full_stockout"]
].copy()

print("\n" + "=" * 100)
print("FULL STOCKOUT CROSS-FITTED ESTIMATION")
print("=" * 100)

print(
    "Full-stockout observations:",
    len(full_oof)
)

if len(full_oof) > 0:

    print(
        "Observed sales mean:",
        round(
            full_oof[
                "sale_amount"
            ].mean(),
            4
        )
    )

    print(
        "Estimated demand mean:",
        round(
            full_oof[
                "estimated_censored_demand"
            ].mean(),
            4
        )
    )

    print(
        "Estimated gap mean:",
        round(
            full_oof[
                "estimated_censored_gap"
            ].mean(),
            4
        )
    )

# =========================================================
# 14. STOCKOUT INTENSITY
# =========================================================
intensity = (
    stockout_oof
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
        estimated_demand=(
            "estimated_censored_demand",
            "mean"
        ),
        estimated_gap=(
            "estimated_censored_gap",
            "mean"
        )
    )
)

print("\n" + "=" * 100)
print("CROSS-FITTED DEMAND BY STOCKOUT INTENSITY")
print("=" * 100)

print(
    intensity
    .round(4)
    .to_string()
)

# =========================================================
# 15. SANITY CHECK
#
# Estimated demand should never be negative.
# =========================================================
negative_predictions = (
    oof[
        "cross_fitted_demand"
    ]
    <
    0
).sum()

print("\n" + "=" * 100)
print("SANITY CHECKS")
print("=" * 100)

print(
    "Negative demand predictions:",
    int(
        negative_predictions
    )
)

print(
    "Missing OOF predictions:",
    int(
        oof[
            "cross_fitted_demand"
        ]
        .isna()
        .sum()
    )
)

print(
    "Future data used for each fold:",
    "NO"
)

# =========================================================
# 16. SAVE OOF RESULTS
# =========================================================
oof_path = (
    PROCESSED
    /
    "cross_fitted_demand_predictions.parquet"
)

oof.to_parquet(
    oof_path,
    index=False
)

fold_path = (
    PROCESSED
    /
    "cross_fitted_reconstruction_metrics.csv"
)

fold_results.to_csv(
    fold_path,
    index=False
)

print("\n" + "=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    "OOF predictions saved:",
    oof_path
)

print(
    "Fold metrics saved:",
    fold_path
)

print(
    "OOF prediction rows:",
    len(oof)
)

print("\n" + "=" * 100)
print("CROSS-FITTED DEMAND RECONSTRUCTION COMPLETE")
print("=" * 100)