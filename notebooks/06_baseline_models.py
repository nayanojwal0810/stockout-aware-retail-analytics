from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")

print("=" * 100)
print("PHASE 2 - STEP 2: BASELINE FORECAST MODELS")
print("=" * 100)

# =========================================================
# 1. LOAD TRAIN DATA
# =========================================================
train = pd.read_parquet(
    RAW / "train.parquet",
    columns=[
        "store_id",
        "product_id",
        "dt",
        "sale_amount"
    ]
)

train["dt"] = pd.to_datetime(
    train["dt"]
)

train = train.sort_values(
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
    "\nInput:",
    train.shape
)

# =========================================================
# 2. CREATE HISTORICAL LAGS
# =========================================================
group = train.groupby(
    group_cols,
    sort=False
)

train["lag_1"] = (
    group["sale_amount"]
    .shift(1)
)

train["lag_7"] = (
    group["sale_amount"]
    .shift(7)
)

train["rolling_7"] = (
    train.groupby(group_cols)["sale_amount"]
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

train["rolling_14"] = (
    train.groupby(group_cols)["sale_amount"]
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
def mae(actual, predicted):
    return np.mean(
        np.abs(
            actual - predicted
        )
    )


def wape(actual, predicted):

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


def rmse(actual, predicted):

    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


def mase(
    actual,
    predicted,
    seasonal_naive_error
):

    denominator = np.mean(
        np.abs(
            seasonal_naive_error
        )
    )

    if denominator == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                actual - predicted
            )
        )
        /
        denominator
    )


# =========================================================
# 5. MODEL DICTIONARY
# =========================================================
model_columns = {
    "Naive_1": "lag_1",
    "Seasonal_Naive_7": "lag_7",
    "Moving_Average_7": "rolling_7",
    "Moving_Average_14": "rolling_14"
}

results = []

# =========================================================
# 6. ROLLING VALIDATION
# =========================================================
for fold_info in folds:

    fold = fold_info["fold"]

    valid_start = pd.Timestamp(
        fold_info["valid_start"]
    )

    valid_end = pd.Timestamp(
        fold_info["valid_end"]
    )

    validation = train[
        train["dt"].between(
            valid_start,
            valid_end
        )
    ].copy()

    # ---------------------------------------------
    # Seasonal naive errors used for MASE scaling
    # ---------------------------------------------
    seasonal_naive_error = (
        validation["sale_amount"]
        -
        validation["lag_7"]
    )

    print("\n" + "=" * 100)
    print(
        f"FOLD {fold}: "
        f"{valid_start.date()} -> {valid_end.date()}"
    )
    print("=" * 100)

    print(
        "Validation rows:",
        len(validation)
    )

    for model_name, prediction_col in model_columns.items():

        subset = validation[
            [
                "sale_amount",
                prediction_col
            ]
        ].dropna()

        if len(subset) == 0:
            continue

        actual = (
            subset["sale_amount"]
            .to_numpy()
        )

        predicted = (
            subset[prediction_col]
            .to_numpy()
        )

        # Sales forecast cannot be negative
        predicted = np.maximum(
            predicted,
            0
        )

        fold_wape = wape(
            actual,
            predicted
        )

        fold_mae = mae(
            actual,
            predicted
        )

        fold_rmse = rmse(
            actual,
            predicted
        )

        # Align MASE denominator with rows
        seasonal_errors = (
            subset["sale_amount"]
            -
            validation.loc[
                subset.index,
                "lag_7"
            ]
        )

        fold_mase = mase(
            actual,
            predicted,
            seasonal_errors
        )

        results.append({
            "fold": fold,
            "model": model_name,
            "rows": len(subset),
            "MAE": fold_mae,
            "WAPE": fold_wape,
            "RMSE": fold_rmse,
            "MASE": fold_mase
        })

        print(
            f"{model_name:<20}"
            f" rows={len(subset):>8,}"
            f" MAE={fold_mae:.4f}"
            f" WAPE={fold_wape:.4f}"
            f" RMSE={fold_rmse:.4f}"
            f" MASE={fold_mase:.4f}"
        )

results = pd.DataFrame(
    results
)

# =========================================================
# 7. OVERALL VALIDATION SUMMARY
# =========================================================
print("\n" + "=" * 100)
print("CROSS-FOLD BASELINE SUMMARY")
print("=" * 100)

summary = (
    results
    .groupby("model")
    .agg(
        folds=("fold", "nunique"),
        mean_mae=("MAE", "mean"),
        std_mae=("MAE", "std"),
        mean_wape=("WAPE", "mean"),
        std_wape=("WAPE", "std"),
        mean_rmse=("RMSE", "mean"),
        mean_mase=("MASE", "mean")
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
# 8. FOLD-TO-FOLD STABILITY
# =========================================================
print("\n" + "=" * 100)
print("FOLD-TO-FOLD WAPE")
print("=" * 100)

wape_table = (
    results
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
# 9. BEST BASELINE
# =========================================================
best_model = (
    summary
    .sort_values(
        "mean_wape"
    )
    .index[0]
)

best_wape = (
    summary
    .loc[
        best_model,
        "mean_wape"
    ]
)

best_mae = (
    summary
    .loc[
        best_model,
        "mean_mae"
    ]
)

print("\n" + "=" * 100)
print("BASELINE WINNER")
print("=" * 100)

print(
    "Best model by mean WAPE:",
    best_model
)

print(
    "Mean WAPE:",
    round(
        best_wape,
        4
    )
)

print(
    "Mean MAE:",
    round(
        best_mae,
        4
    )
)

# =========================================================
# 10. VALIDATION CONSISTENCY
# =========================================================
print("\n" + "=" * 100)
print("VALIDATION CONSISTENCY")
print("=" * 100)

print(
    "Validation folds:",
    results["fold"].nunique()
)

print(
    "Models tested:",
    results["model"].nunique()
)

print(
    "Final evaluation touched:",
    False
)

print(
    "Final evaluation period:",
    "2024-06-26 to 2024-07-02"
)

# =========================================================
# 11. SANITY CHECKS
# =========================================================
assert results["fold"].nunique() == 5

assert (
    results["model"].nunique()
    == 4
)

assert (
    train["dt"].max()
    <
    pd.Timestamp("2024-06-26")
)

assert (
    best_wape >= 0
)

print(
    "\nBaseline validation checks: PASS"
)

print("\n" + "=" * 100)
print("BASELINE MODEL EVALUATION COMPLETE")
print("=" * 100)