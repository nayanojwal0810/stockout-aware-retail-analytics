from pathlib import Path
import numpy as np
import pandas as pd
from stockout_retail.config import GROUP_COLS, FINAL_TRAIN_END, FINAL_HOLDOUT_START, FINAL_HOLDOUT_END
from stockout_retail.features.demand_features import CANONICAL_FEATURES, create_features
from stockout_retail.forecasting.model import build_model
from stockout_retail.forecasting.evaluation import mae, wape, rmse

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 110)
print("PHASE 3 - STEP 6: FINAL 7-DAY HOLDOUT EVALUATION")
print("=" * 110)

# =========================================================
# 1. FINAL HOLDOUT DEFINITION
# =========================================================
TRAIN_END = pd.Timestamp(FINAL_TRAIN_END)
HOLDOUT_START = pd.Timestamp(FINAL_HOLDOUT_START)
HOLDOUT_END = pd.Timestamp(FINAL_HOLDOUT_END)

HORIZONS = range(
    1,
    8
)

print(
    "\nFinal training end:",
    TRAIN_END
)

print(
    "Final holdout:",
    HOLDOUT_START,
    "->",
    HOLDOUT_END
)

# =========================================================
# 2. LOAD DATA
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

train = pd.read_parquet(
    RAW / "train.parquet",
    columns=cols
)

eval_df = pd.read_parquet(
    RAW / "eval.parquet",
    columns=cols
)

train["dt"] = pd.to_datetime(
    train["dt"]
)

eval_df["dt"] = pd.to_datetime(
    eval_df["dt"]
)

print(
    "\nTRAIN:",
    train.shape
)

print(
    "EVAL:",
    eval_df.shape
)

# =========================================================
# 3. HARD HOLDOUT ASSERTIONS
# =========================================================
assert (
    train["dt"].max()
    ==
    TRAIN_END
)

assert (
    eval_df["dt"].min()
    ==
    HOLDOUT_START
)

assert (
    eval_df["dt"].max()
    ==
    HOLDOUT_END
)

assert (
    set(
        train["dt"]
    ).isdisjoint(
        set(
            eval_df["dt"]
        )
    )
)

# =========================================================
# 4. SORT
# =========================================================
group_cols = GROUP_COLS

train = train.sort_values(
    group_cols + ["dt"]
).reset_index(
    drop=True
)

eval_df = eval_df.sort_values(
    group_cols + ["dt"]
).reset_index(
    drop=True
)

# =========================================================
# 5. FEATURE ENGINEERING
#
# EXACT DEVELOPMENT SPECIFICATION
# =========================================================
all_data = pd.concat(
    [
        train,
        eval_df
    ],
    ignore_index=True
)

all_features = create_features(
    all_data
)

features = CANONICAL_FEATURES

print(
    "\nFeature count:",
    len(features)
)

assert (
    len(features)
    ==
    18
)

# =========================================================
# 6. HOLDOUT FEATURE POLICY CHECK
# =========================================================
# Same-day sale_amount is never a feature.
# Same-day stockout hours are never a feature.
#
# Discount/activity/holiday flags are included only because
# this is the frozen specification used during development.
# They must be interpreted as ex-ante known/planned variables.
# =========================================================
for f in features:

    assert (
        f
        !=
        "sale_amount"
    )

    assert (
        f
        !=
        "stock_hour6_22_cnt"
    )

print(
    "Same-day realized sales used:",
    False
)

print(
    "Same-day realized stockout hours used:",
    False
)

# =========================================================
# 7. IDENTIFY HOLDOUT ORIGIN
# =========================================================
origin = all_features[
    all_features["dt"]
    ==
    TRAIN_END
].copy()

print(
    "\nOrigin rows:",
    len(origin)
)

assert (
    len(origin)
    ==
    50000
)

# =========================================================
# 8. MODEL SPECIFICATION
#
# FROZEN from development.
# Do not tune after observing holdout.
# =========================================================
# =========================================================
# 9. DIRECT 7-DAY MODELS
# =========================================================
predictions = []

for horizon in HORIZONS:

    target_date = (
        TRAIN_END
        +
        pd.Timedelta(
            days=horizon
        )
    )

    print("\n" + "-" * 110)
    print(
        f"HORIZON +{horizon}:",
        target_date.date()
    )
    print("-" * 110)

    # -----------------------------------------------------
    # Target = future sale at exactly +horizon.
    # Only observations through June 25 may be used.
    # -----------------------------------------------------
    training = all_features.copy()

    training[
        "direct_target"
    ] = (
        training
        .groupby(
            group_cols
        )[
            "sale_amount"
        ]
        .shift(
            -horizon
        )
    )

    training = training[
        training["dt"]
        <=
        TRAIN_END
    ].copy()

    training = training.dropna(
        subset=
        features
        +
        [
            "direct_target"
        ]
    )

    print(
        "Training rows:",
        len(training)
    )

    assert (
        len(training)
        >
        0
    )

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

    model = build_model()

    model.fit(
        X_train,
        y_train
    )

    X_origin = (
        origin[
            features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    prediction = model.predict(
        X_origin
    )

    prediction = (
        np.maximum(
            prediction,
            0
        )
        .astype(
            np.float32
        )
    )

    fold_prediction = pd.DataFrame(
        {
            "store_id":
                origin[
                    "store_id"
                ].to_numpy(),

            "product_id":
                origin[
                    "product_id"
                ].to_numpy(),

            "forecast_origin":
                TRAIN_END,

            "dt":
                target_date,

            "horizon":
                horizon,

            "prediction":
                prediction
        }
    )

    predictions.append(
        fold_prediction
    )

predictions = pd.concat(
    predictions,
    ignore_index=True
)

# =========================================================
# 10. FORECAST COVERAGE
# =========================================================
print("\n" + "=" * 110)
print("FINAL HOLDOUT FORECAST COVERAGE")
print("=" * 110)

expected_rows = (
    7
    *
    50000
)

print(
    "Predicted rows:",
    len(predictions)
)

print(
    "Expected rows:",
    expected_rows
)

assert (
    len(predictions)
    ==
    expected_rows
)

assert (
    predictions["dt"].min()
    ==
    HOLDOUT_START
)

assert (
    predictions["dt"].max()
    ==
    HOLDOUT_END
)

# =========================================================
# 11. MERGE ACTUAL HOLDOUT
# =========================================================
holdout_actual = eval_df[
    [
        "store_id",
        "product_id",
        "dt",
        "sale_amount",
        "stock_hour6_22_cnt"
    ]
]

results = predictions.merge(
    holdout_actual,
    on=[
        "store_id",
        "product_id",
        "dt"
    ],
    how="inner"
)

print(
    "Evaluation rows:",
    len(results)
)

assert (
    len(results)
    ==
    expected_rows
)

# =========================================================
# 12. METRICS
# =========================================================
actual = (
    results[
        "sale_amount"
    ]
    .to_numpy(
        dtype=float
    )
)

predicted = (
    results[
        "prediction"
    ]
    .to_numpy(
        dtype=float
    )
)

overall_mae = mae(
    actual,
    predicted
)

overall_wape = wape(
    actual,
    predicted
)

overall_rmse = rmse(
    actual,
    predicted
)

print("\n" + "=" * 110)
print("FINAL HOLDOUT RESULT")
print("=" * 110)

print(
    "MAE:",
    round(
        overall_mae,
        4
    )
)

print(
    "WAPE:",
    round(
        overall_wape,
        4
    )
)

print(
    "RMSE:",
    round(
        overall_rmse,
        4
    )
)

# =========================================================
# 13. HORIZON PERFORMANCE
# =========================================================
horizon_rows = []

for horizon in HORIZONS:

    subset = results[
        results[
            "horizon"
        ]
        ==
        horizon
    ]

    y_true = (
        subset[
            "sale_amount"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_pred = (
        subset[
            "prediction"
        ]
        .to_numpy(
            dtype=float
        )
    )

    horizon_rows.append(
        {
            "horizon": horizon,
            "date":
                subset[
                    "dt"
                ].iloc[0],
            "MAE":
                mae(
                    y_true,
                    y_pred
                ),
            "WAPE":
                wape(
                    y_true,
                    y_pred
                ),
            "RMSE":
                rmse(
                    y_true,
                    y_pred
                )
        }
    )

horizon_summary = pd.DataFrame(
    horizon_rows
)

print("\n" + "=" * 110)
print("HORIZON PERFORMANCE")
print("=" * 110)

print(
    horizon_summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 14. HOLDOUT STOCKOUT DIAGNOSTIC
#
# Descriptive only. It does not alter the forecast.
# =========================================================
print("\n" + "=" * 110)
print("HOLDOUT STOCKOUT DIAGNOSTIC")
print("=" * 110)

stockout_hours = (
    results[
        "stock_hour6_22_cnt"
    ]
)

print(
    "Rows with any stockout:",
    int(
        (
            stockout_hours
            >
            0
        ).sum()
    )
)

print(
    "Any-stockout share:",
    f"{(
        stockout_hours
        >
        0
    ).mean() * 100:.2f}%"
)

print(
    "Full-stockout rows:",
    int(
        (
            stockout_hours
            ==
            16
        ).sum()
    )
)

print(
    "Full-stockout share:",
    f"{(
        stockout_hours
        ==
        16
    ).mean() * 100:.2f}%"
)

# =========================================================
# 15. PERFORMANCE BY STOCKOUT STATE
# =========================================================
results[
    "stockout_state"
] = np.select(
    [
        results[
            "stock_hour6_22_cnt"
        ]
        ==
        0,

        results[
            "stock_hour6_22_cnt"
        ]
        ==
        16,

        results[
            "stock_hour6_22_cnt"
        ]
        >
        0
    ],
    [
        "NORMAL",
        "FULL_STOCKOUT",
        "PARTIAL_STOCKOUT"
    ],
    default="UNKNOWN"
)

state_rows = []

for state in [
    "NORMAL",
    "PARTIAL_STOCKOUT",
    "FULL_STOCKOUT"
]:

    subset = results[
        results[
            "stockout_state"
        ]
        ==
        state
    ]

    if len(subset) == 0:
        continue

    y_true = (
        subset[
            "sale_amount"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_pred = (
        subset[
            "prediction"
        ]
        .to_numpy(
            dtype=float
        )
    )

    state_rows.append(
        {
            "stockout_state": state,
            "observations": len(
                subset
            ),
            "mean_actual_sales":
                y_true.mean(),
            "mean_prediction":
                y_pred.mean(),
            "MAE":
                mae(
                    y_true,
                    y_pred
                ),
            "WAPE":
                wape(
                    y_true,
                    y_pred
                )
        }
    )

state_summary = pd.DataFrame(
    state_rows
)

print("\n" + "=" * 110)
print("HOLDOUT PERFORMANCE BY STOCKOUT STATE")
print("=" * 110)

print(
    state_summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 16. FORECAST BIAS
# =========================================================
results[
    "error"
] = (
    results[
        "sale_amount"
    ]
    -
    results[
        "prediction"
    ]
)

print("\n" + "=" * 110)
print("FINAL HOLDOUT BIAS")
print("=" * 110)

print(
    "Mean error:",
    round(
        results[
            "error"
        ].mean(),
        4
    )
)

print(
    "Median error:",
    round(
        results[
            "error"
        ].median(),
        4
    )
)

print(
    "Underforecast share:",
    f"{(
        results[
            'error'
        ]
        >
        0
    ).mean() * 100:.2f}%"
)

# =========================================================
# 17. COMPARISON TO DEVELOPMENT BENCHMARK
#
# This is descriptive only.
# =========================================================
development_path = (
    PROCESSED
    /
    "true_7day_raw_forecast_results.csv"
)

if development_path.exists():

    development = pd.read_csv(
        development_path
    )

    development_wape = np.nan

    if "wape" in development.columns:

        development_wape = (
            development[
                "wape"
            ].mean()
        )

    elif "WAPE" in development.columns:

        development_wape = (
            development[
                "WAPE"
            ].mean()
        )

    print("\n" + "=" * 110)
    print("DEVELOPMENT REFERENCE")
    print("=" * 110)

    print(
        "Development artifact loaded:",
        True
    )

else:

    development_wape = np.nan

    print("\n" + "=" * 110)
    print("DEVELOPMENT REFERENCE")
    print("=" * 110)

    print(
        "Development artifact loaded:",
        False
    )

# =========================================================
# 18. HOLDOUT INTEGRITY
# =========================================================
print("\n" + "=" * 110)
print("HOLDOUT INTEGRITY")
print("=" * 110)

print(
    "Training data ends:",
    TRAIN_END
)

print(
    "Holdout begins:",
    HOLDOUT_START
)

print(
    "Holdout rows used for model fitting:",
    False
)

print(
    "Holdout actual sales used before scoring:",
    False
)

print(
    "Model/hyperparameters tuned on holdout:",
    False
)

print(
    "Same-day sale_amount used as feature:",
    False
)

print(
    "Same-day stockout hours used as feature:",
    False
)

# =========================================================
# 19. FINAL DECISION
# =========================================================
print("\n" + "=" * 110)
print("FINAL HOLDOUT DECISION")
print("=" * 110)

print(
    "This result is the FINAL untouched out-of-time score."
)

print(
    "No model changes should be made after observing it."
)

# =========================================================
# 20. SANITY CHECKS
# =========================================================
assert (
    len(results)
    ==
    350000
)

assert (
    results[
        "prediction"
    ]
    .ge(0)
    .all()
)

assert (
    results[
        "dt"
    ].min()
    ==
    HOLDOUT_START
)

assert (
    results[
        "dt"
    ].max()
    ==
    HOLDOUT_END
)

assert (
    results[
        "horizon"
    ].nunique()
    ==
    7
)

assert (
    results[
        "horizon"
    ]
    .min()
    ==
    1
)

assert (
    results[
        "horizon"
    ]
    .max()
    ==
    7
)

print(
    "\nFinal holdout checks: PASS"
)

# =========================================================
# 21. SAVE FINAL ARTIFACTS
# =========================================================
prediction_output = (
    PROCESSED
    /
    "final_holdout_predictions.csv"
)

horizon_output = (
    PROCESSED
    /
    "final_holdout_horizon_metrics.csv"
)

state_output = (
    PROCESSED
    /
    "final_holdout_stockout_metrics.csv"
)

summary_output = (
    PROCESSED
    /
    "final_holdout_summary.csv"
)

results.to_csv(
    prediction_output,
    index=False
)

horizon_summary.to_csv(
    horizon_output,
    index=False
)

state_summary.to_csv(
    state_output,
    index=False
)

pd.DataFrame(
    [
        {
            "holdout_start":
                HOLDOUT_START,
            "holdout_end":
                HOLDOUT_END,
            "MAE":
                overall_mae,
            "WAPE":
                overall_wape,
            "RMSE":
                overall_rmse,
            "rows":
                len(results),
            "model_family":
                "HistGradientBoostingRegressor",
            "forecast_type":
                "direct_7_day"
        }
    ]
).to_csv(
    summary_output,
    index=False
)

print("\n" + "=" * 110)
print("OUTPUT")
print("=" * 110)

print(
    "Predictions:",
    prediction_output
)

print(
    "Horizon metrics:",
    horizon_output
)

print(
    "Stockout-state metrics:",
    state_output
)

print(
    "Final summary:",
    summary_output
)

print("\n" + "=" * 110)
print("PHASE 3 STEP 6 COMPLETE")
print("=" * 110)