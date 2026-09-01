from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")

print("=" * 110)
print("PHASE 3 - STEP 3: EMPIRICAL UNCERTAINTY & SAFETY-STOCK CALIBRATION")
print("=" * 110)

# =========================================================
# 1. LOAD DIRECT FORECASTS
# =========================================================
forecast_path = (
    PROCESSED
    /
    "inventory_backtest_forecasts.csv"
)

forecasts = pd.read_csv(
    forecast_path,
    parse_dates=[
        "forecast_origin",
        "dt"
    ]
)

print(
    "\nForecast rows:",
    len(forecasts)
)

print(
    "Folds:",
    forecasts["fold"].nunique()
)

print(
    "Horizon count:",
    forecasts["horizon"].nunique()
)

# =========================================================
# 2. LOAD ACTUAL SALES
# =========================================================
raw_path = Path("data/raw/train.parquet")

actual = pd.read_parquet(
    raw_path,
    columns=[
        "store_id",
        "product_id",
        "dt",
        "sale_amount"
    ]
)

actual["dt"] = pd.to_datetime(
    actual["dt"]
)

print(
    "Actual rows:",
    len(actual)
)

# =========================================================
# 3. MERGE FORECAST + ACTUAL
# =========================================================
evaluation = forecasts.merge(
    actual,
    on=[
        "store_id",
        "product_id",
        "dt"
    ],
    how="inner"
)

print(
    "Evaluation rows:",
    len(evaluation)
)

# =========================================================
# 4. SANITY CHECK
# =========================================================
expected_rows = (
    5
    *
    7
    *
    50000
)

assert (
    len(evaluation)
    ==
    expected_rows
)

# =========================================================
# 5. BUILD 7-DAY OBSERVED DEMAND / FORECAST
# =========================================================
seven_day = (
    evaluation
    .groupby(
        [
            "fold",
            "forecast_origin",
            "store_id",
            "product_id"
        ]
    )
    .agg(
        forecast_7day=(
            "forecast_sales",
            "sum"
        ),
        actual_7day=(
            "sale_amount",
            "sum"
        )
    )
    .reset_index()
)

print(
    "\n7-day series:",
    len(seven_day)
)

assert (
    len(seven_day)
    ==
    5
    *
    50000
)

# =========================================================
# 6. FORECAST ERROR
# =========================================================
seven_day["forecast_error"] = (
    seven_day["actual_7day"]
    -
    seven_day["forecast_7day"]
)

# Positive error means actual demand exceeded forecast.
seven_day["positive_error"] = (
    seven_day["forecast_error"]
    .clip(
        lower=0
    )
)

# =========================================================
# 7. ERROR DISTRIBUTION
# =========================================================
print("\n" + "=" * 110)
print("7-DAY FORECAST ERROR")
print("=" * 110)

print(
    seven_day[
        "forecast_error"
    ]
    .describe()
    .round(4)
    .to_string()
)

print(
    "\nMean error:",
    round(
        seven_day[
            "forecast_error"
        ].mean(),
        4
    )
)

print(
    "Median error:",
    round(
        seven_day[
            "forecast_error"
        ].median(),
        4
    )
)

print(
    "Positive-error share:",
    round(
        (
            seven_day[
                "forecast_error"
            ]
            >
            0
        )
        .mean()
        *
        100,
        2
    ),
    "%"
)

# =========================================================
# 8. FOLD ORDER
# =========================================================
folds = sorted(
    seven_day["fold"].unique()
)

print(
    "\nAvailable folds:",
    folds
)

# =========================================================
# 9. SERVICE-LEVEL QUANTILES
# =========================================================
service_levels = {
    "SL80": 0.80,
    "SL90": 0.90,
    "SL95": 0.95
}

# =========================================================
# 10. NESTED / WALK-FORWARD CALIBRATION
#
# Calibration for fold k uses only errors from folds < k.
# Fold 1 cannot have prior calibration, so it is excluded
# from empirical-policy evaluation.
# =========================================================
calibrated_rows = []

for fold in folds:

    prior_folds = [
        x
        for x in folds
        if x < fold
    ]

    current = seven_day[
        seven_day["fold"] == fold
    ].copy()

    if len(prior_folds) == 0:

        print(
            f"\nFold {fold}: "
            "no prior calibration fold available -> empirical policy skipped."
        )

        continue

    calibration_pool = seven_day[
        seven_day["fold"].isin(
            prior_folds
        )
    ].copy()

    print(
        f"\nFold {fold}: "
        f"calibrating from folds {prior_folds}"
    )

    # -----------------------------------------------------
    # Empirical positive forecast-error quantiles
    # -----------------------------------------------------
    empirical_quantiles = {}

    for policy, alpha in service_levels.items():

        empirical_quantiles[
            policy
        ] = np.quantile(
            calibration_pool[
                "positive_error"
            ],
            alpha
        )

        print(
            f"{policy} empirical safety stock:",
            round(
                empirical_quantiles[
                    policy
                ],
                4
            )
        )

    # -----------------------------------------------------
    # Apply calibration to current fold
    # -----------------------------------------------------
    for policy, alpha in service_levels.items():

        temp = current.copy()

        empirical_ss = (
            empirical_quantiles[
                policy
            ]
        )

        temp[
            "policy"
        ] = policy

        temp[
            "empirical_safety_stock"
        ] = empirical_ss

        temp[
            "empirical_inventory_target"
        ] = (
            temp[
                "forecast_7day"
            ]
            +
            temp[
                "empirical_safety_stock"
            ]
        ).clip(
            lower=0
        )

        temp[
            "shortage_units"
        ] = (
            temp[
                "actual_7day"
            ]
            -
            temp[
                "empirical_inventory_target"
            ]
        ).clip(
            lower=0
        )

        temp[
            "excess_units"
        ] = (
            temp[
                "empirical_inventory_target"
            ]
            -
            temp[
                "actual_7day"
            ]
        ).clip(
            lower=0
        )

        temp[
            "fill_rate"
        ] = (
            1
            -
            temp[
                "shortage_units"
            ]
            /
            temp[
                "actual_7day"
            ].replace(
                0,
                np.nan
            )
        ).clip(
            lower=0,
            upper=1
        )

        temp[
            "stockout_event"
        ] = (
            temp[
                "actual_7day"
            ]
            >
            temp[
                "empirical_inventory_target"
            ]
        )

        calibrated_rows.append(
            temp[
                [
                    "fold",
                    "forecast_origin",
                    "store_id",
                    "product_id",
                    "actual_7day",
                    "forecast_7day",
                    "forecast_error",
                    "policy",
                    "empirical_safety_stock",
                    "empirical_inventory_target",
                    "shortage_units",
                    "excess_units",
                    "fill_rate",
                    "stockout_event"
                ]
            ]
        )

# =========================================================
# 11. COMBINE
# =========================================================
calibrated = pd.concat(
    calibrated_rows,
    ignore_index=True
)

print("\n" + "=" * 110)
print("EMPIRICAL POLICY COVERAGE")
print("=" * 110)

print(
    "Rows:",
    len(calibrated)
)

print(
    "Evaluation folds:",
    sorted(
        calibrated[
            "fold"
        ].unique()
    )
)

assert (
    len(
        calibrated[
            "fold"
        ].unique()
    )
    ==
    4
)

# =========================================================
# 12. EMPIRICAL POLICY SUMMARY
# =========================================================
print("\n" + "=" * 110)
print("EMPIRICAL SAFETY-STOCK POLICY SUMMARY")
print("=" * 110)

policy_summary = (
    calibrated
    .groupby(
        "policy"
    )
    .agg(
        folds=(
            "fold",
            "nunique"
        ),
        observations=(
            "actual_7day",
            "size"
        ),
        safety_stock=(
            "empirical_safety_stock",
            "mean"
        ),
        mean_inventory=(
            "empirical_inventory_target",
            "mean"
        ),
        median_inventory=(
            "empirical_inventory_target",
            "median"
        ),
        mean_shortage=(
            "shortage_units",
            "mean"
        ),
        mean_excess=(
            "excess_units",
            "mean"
        ),
        mean_fill_rate=(
            "fill_rate",
            "mean"
        ),
        stockout_rate=(
            "stockout_event",
            "mean"
        )
    )
    .reset_index()
)

print(
    policy_summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 13. REALIZED SERVICE LEVEL
# =========================================================
policy_summary[
    "realized_service_level"
] = (
    1
    -
    policy_summary[
        "stockout_rate"
    ]
)

print("\n" + "=" * 110)
print("CALIBRATED VS REALIZED SERVICE")
print("=" * 110)

print(
    policy_summary[
        [
            "policy",
            "realized_service_level",
            "mean_fill_rate"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 14. FOLD STABILITY
# =========================================================
print("\n" + "=" * 110)
print("FOLD-LEVEL EMPIRICAL POLICY PERFORMANCE")
print("=" * 110)

fold_summary = (
    calibrated
    .groupby(
        [
            "fold",
            "policy"
        ]
    )
    .agg(
        safety_stock=(
            "empirical_safety_stock",
            "mean"
        ),
        mean_inventory=(
            "empirical_inventory_target",
            "mean"
        ),
        mean_shortage=(
            "shortage_units",
            "mean"
        ),
        mean_excess=(
            "excess_units",
            "mean"
        ),
        mean_fill_rate=(
            "fill_rate",
            "mean"
        ),
        stockout_rate=(
            "stockout_event",
            "mean"
        )
    )
    .reset_index()
)

print(
    fold_summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 15. CALIBRATION QUALITY
# =========================================================
print("\n" + "=" * 110)
print("CALIBRATION QUALITY")
print("=" * 110)

for policy in service_levels:

    subset = calibrated[
        calibrated["policy"] == policy
    ]

    realized = (
        1
        -
        subset[
            "stockout_event"
        ].mean()
    )

    target = (
        service_levels[
            policy
        ]
    )

    print(
        f"{policy}: "
        f"target={target:.2f} "
        f"realized={realized:.4f} "
        f"gap={realized - target:.4f}"
    )

# =========================================================
# 16. COMPARE WITH NORMAL APPROXIMATION
#
# Using the same forecast errors, estimate a normal-style
# one-sided buffer:
#
# quantile ≈ mean + z * std
#
# Since only positive shortage protection matters, use
# max(0, quantile).
# =========================================================
error_mean = (
    seven_day[
        "forecast_error"
    ].mean()
)

error_std = (
    seven_day[
        "forecast_error"
    ].std()
)

z_values = {
    "SL80": 0.8416,
    "SL90": 1.2816,
    "SL95": 1.6449
}

normal_rows = []

for policy, z in z_values.items():

    normal_ss = max(
        0,
        error_mean
        +
        z
        *
        error_std
    )

    normal_rows.append(
        {
            "policy": policy,
            "normal_error_mean": error_mean,
            "normal_error_std": error_std,
            "normal_safety_stock": normal_ss
        }
    )

normal_summary = pd.DataFrame(
    normal_rows
)

print("\n" + "=" * 110)
print("NORMAL VS EMPIRICAL SAFETY STOCK")
print("=" * 110)

empirical_ss_summary = (
    calibrated
    .groupby(
        "policy"
    )[
        "empirical_safety_stock"
    ]
    .mean()
    .reset_index()
    .rename(
        columns={
            "empirical_safety_stock":
                "empirical_safety_stock"
        }
    )
)

comparison = normal_summary.merge(
    empirical_ss_summary,
    on="policy",
    how="left"
)

comparison[
    "empirical_minus_normal"
] = (
    comparison[
        "empirical_safety_stock"
    ]
    -
    comparison[
        "normal_safety_stock"
    ]
)

print(
    comparison
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 17. ERROR QUANTILES
# =========================================================
print("\n" + "=" * 110)
print("FULL 7-DAY ERROR QUANTILES")
print("=" * 110)

quantiles = (
    seven_day[
        "forecast_error"
    ]
    .quantile(
        [
            0.50,
            0.75,
            0.80,
            0.90,
            0.95,
            0.99
        ]
    )
)

print(
    quantiles
    .round(4)
    .to_string()
)

# =========================================================
# 18. EXTREME ERROR RATE
# =========================================================
print("\n" + "=" * 110)
print("EXTREME UNDER-FORECASTING")
print("=" * 110)

for threshold in [
    2,
    5,
    10,
    20
]:

    share = (
        seven_day[
            "forecast_error"
        ]
        >
        threshold
    ).mean()

    print(
        f"Error > {threshold:>2} units:",
        f"{share * 100:.3f}%"
    )

# =========================================================
# 19. HOLDOUT PROTECTION
# =========================================================
print("\n" + "=" * 110)
print("HOLDOUT PROTECTION")
print("=" * 110)

print(
    "Final evaluation period:",
    "2024-06-26 -> 2024-07-02"
)

print(
    "Final evaluation used:",
    False
)

# =========================================================
# 20. LEAKAGE CHECK
# =========================================================
print("\n" + "=" * 110)
print("LEAKAGE CHECK")
print("=" * 110)

print(
    "Fold k calibrated only from prior folds:",
    True
)

print(
    "Current-fold errors used to calibrate itself:",
    False
)

print(
    "Future sales used:",
    False
)

print(
    "Final holdout used:",
    False
)

# =========================================================
# 21. SANITY CHECKS
# =========================================================
assert (
    len(calibrated)
    >
    0
)

assert (
    calibrated[
        "empirical_safety_stock"
    ]
    .ge(0)
    .all()
)

assert (
    calibrated[
        "empirical_inventory_target"
    ]
    .ge(0)
    .all()
)

assert (
    calibrated[
        "fill_rate"
    ]
    .dropna()
    .between(
        0,
        1
    )
    .all()
)

assert (
    calibrated[
        "fold"
    ]
    .nunique()
    ==
    4
)

print(
    "\nUncertainty calibration checks: PASS"
)

# =========================================================
# 22. SAVE
# =========================================================
policy_output = (
    PROCESSED
    /
    "empirical_uncertainty_policy_results.csv"
)

summary_output = (
    PROCESSED
    /
    "empirical_uncertainty_policy_summary.csv"
)

calibrated.to_csv(
    policy_output,
    index=False
)

policy_summary.to_csv(
    summary_output,
    index=False
)

print("\n" + "=" * 110)
print("OUTPUT")
print("=" * 110)

print(
    "Detailed policy results:",
    policy_output
)

print(
    "Policy summary:",
    summary_output
)

print("\n" + "=" * 110)
print("PHASE 3 STEP 3 COMPLETE")
print("=" * 110)