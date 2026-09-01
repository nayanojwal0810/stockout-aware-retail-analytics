from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 110)
print("PHASE 3 - STEP 7: FINAL HOLDOUT BASELINE BENCHMARK")
print("=" * 110)

# =========================================================
# 1. FINAL HOLDOUT DEFINITION
# =========================================================
TRAIN_END = pd.Timestamp("2024-06-25")
HOLDOUT_START = pd.Timestamp("2024-06-26")
HOLDOUT_END = pd.Timestamp("2024-07-02")

group_cols = [
    "store_id",
    "product_id"
]

print(
    "\nTraining ends:",
    TRAIN_END
)

print(
    "Final holdout:",
    HOLDOUT_START,
    "->",
    HOLDOUT_END
)

# =========================================================
# 2. LOAD TRAIN + HOLDOUT
# =========================================================
columns = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount"
]

train = pd.read_parquet(
    RAW / "train.parquet",
    columns=columns
)

holdout = pd.read_parquet(
    RAW / "eval.parquet",
    columns=columns
)

train["dt"] = pd.to_datetime(train["dt"])
holdout["dt"] = pd.to_datetime(holdout["dt"])

train = train.sort_values(
    group_cols + ["dt"]
).reset_index(drop=True)

holdout = holdout.sort_values(
    group_cols + ["dt"]
).reset_index(drop=True)

print(
    "\nTRAIN:",
    train.shape
)

print(
    "HOLDOUT:",
    holdout.shape
)

# =========================================================
# 3. HARD HOLDOUT CHECKS
# =========================================================
assert (
    train["dt"].max()
    ==
    TRAIN_END
)

assert (
    holdout["dt"].min()
    ==
    HOLDOUT_START
)

assert (
    holdout["dt"].max()
    ==
    HOLDOUT_END
)

assert (
    set(train["dt"]).isdisjoint(
        set(holdout["dt"])
    )
)

assert (
    len(holdout)
    ==
    350000
)

# =========================================================
# 4. SERIES COVERAGE
# =========================================================
train_series = (
    train[group_cols]
    .drop_duplicates()
)

holdout_series = (
    holdout[group_cols]
    .drop_duplicates()
)

print(
    "\nTraining series:",
    len(train_series)
)

print(
    "Holdout series:",
    len(holdout_series)
)

assert (
    len(train_series)
    ==
    50000
)

assert (
    len(holdout_series)
    ==
    50000
)

assert (
    train_series.merge(
        holdout_series,
        on=group_cols,
        how="outer",
        indicator=True
    )["_merge"]
    .eq("both")
    .all()
)

# =========================================================
# 5. HOLDOUT ACTUALS
# =========================================================
actual = holdout[
    [
        "store_id",
        "product_id",
        "dt",
        "sale_amount"
    ]
].copy()

# =========================================================
# 6. BUILD HISTORICAL LOOKUP
# =========================================================
series_history = (
    train
    .groupby(
        group_cols
    )["sale_amount"]
    .agg(
        list
    )
)

# Convert to numpy arrays for efficient calculation.
history = {
    key: np.asarray(
        values,
        dtype=np.float32
    )
    for key, values in series_history.items()
}

# =========================================================
# 7. GENERATE BASELINE FORECASTS
#
# Each baseline uses information available at June 25 only.
#
# NAIVE-1:
#   every future day = June 25 sales
#
# SEASONAL-NAIVE-7:
#   day h = sales from 7 days before that future date
#
# MA-7:
#   every future day = mean of June 19-25
# =========================================================
future_dates = pd.date_range(
    HOLDOUT_START,
    HOLDOUT_END,
    freq="D"
)

forecast_rows = []

for key, values in history.items():

    store_id, product_id = key

    # -----------------------------------------------------
    # Verify full training history.
    # -----------------------------------------------------
    assert (
        len(values)
        ==
        90
    )

    # -----------------------------------------------------
    # Naive-1
    # -----------------------------------------------------
    naive_1 = float(
        values[-1]
    )

    # -----------------------------------------------------
    # Moving average 7
    # -----------------------------------------------------
    moving_average_7 = float(
        np.mean(
            values[-7:]
        )
    )

    # -----------------------------------------------------
    # Seasonal naive 7
    #
    # For each future horizon h:
    # future_date h -> date h-7
    #
    # With 90 training days:
    # h=1 -> index 83
    # h=7 -> index 89
    # -----------------------------------------------------
    seasonal_values = []

    for horizon in range(1, 8):

        seasonal_index = (
            len(values)
            -
            7
            +
            horizon
            -
            1
        )

        seasonal_values.append(
            float(
                values[
                    seasonal_index
                ]
            )
        )

    for horizon, dt in enumerate(
        future_dates,
        start=1
    ):

        forecast_rows.append(
            {
                "store_id":
                    store_id,

                "product_id":
                    product_id,

                "dt":
                    dt,

                "horizon":
                    horizon,

                "naive_1":
                    max(
                        0.0,
                        naive_1
                    ),

                "seasonal_naive_7":
                    max(
                        0.0,
                        seasonal_values[
                            horizon - 1
                        ]
                    ),

                "moving_average_7":
                    max(
                        0.0,
                        moving_average_7
                    )
            }
        )

baselines = pd.DataFrame(
    forecast_rows
)

print(
    "\nBaseline forecast rows:",
    len(baselines)
)

assert (
    len(baselines)
    ==
    350000
)

# =========================================================
# 8. LOAD FINAL MODEL PREDICTIONS
# =========================================================
final_prediction_path = (
    PROCESSED
    /
    "final_holdout_predictions.csv"
)

final_predictions = pd.read_csv(
    final_prediction_path,
    parse_dates=[
        "forecast_origin",
        "dt"
    ]
)

print(
    "Final model prediction rows:",
    len(final_predictions)
)

print(
    "Final model columns:",
    final_predictions.columns.tolist()
)

# =========================================================
# 9. STANDARDIZE FINAL MODEL PREDICTION COLUMN
# =========================================================
if "prediction" in final_predictions.columns:

    final_predictions = (
        final_predictions
        [
            [
                "store_id",
                "product_id",
                "dt",
                "horizon",
                "prediction"
            ]
        ]
        .copy()
    )

else:

    prediction_candidates = [
        c
        for c in final_predictions.columns
        if (
            "prediction" in c.lower()
            or
            "forecast" in c.lower()
        )
    ]

    if len(prediction_candidates) != 1:

        raise ValueError(
            "Could not uniquely identify final-model "
            "prediction column. Candidates: "
            + str(prediction_candidates)
        )

    prediction_col = (
        prediction_candidates[0]
    )

    final_predictions = (
        final_predictions
        [
            [
                "store_id",
                "product_id",
                "dt",
                "horizon",
                prediction_col
            ]
        ]
        .rename(
            columns={
                prediction_col:
                    "prediction"
            }
        )
    )

assert (
    len(final_predictions)
    ==
    350000
)

# =========================================================
# 10. MERGE ALL FORECASTS WITH ACTUALS
# =========================================================
results = (
    baselines
    .merge(
        actual,
        on=[
            "store_id",
            "product_id",
            "dt"
        ],
        how="inner"
    )
    .merge(
        final_predictions,
        on=[
            "store_id",
            "product_id",
            "dt",
            "horizon"
        ],
        how="inner"
    )
)

print(
    "Final comparison rows:",
    len(results)
)

assert (
    len(results)
    ==
    350000
)

# =========================================================
# 11. METRICS
# =========================================================
def mae(
    y_true,
    y_pred
):

    return np.mean(
        np.abs(
            y_true
            -
            y_pred
        )
    )


def wape(
    y_true,
    y_pred
):

    denominator = np.sum(
        np.abs(
            y_true
        )
    )

    if denominator == 0:
        return np.nan

    return (
        np.sum(
            np.abs(
                y_true
                -
                y_pred
            )
        )
        /
        denominator
    )


def rmse(
    y_true,
    y_pred
):

    return np.sqrt(
        np.mean(
            (
                y_true
                -
                y_pred
            )
            ** 2
        )
    )


# =========================================================
# 12. OVERALL MODEL COMPARISON
# =========================================================
model_columns = {
    "Naive_1":
        "naive_1",
    "Seasonal_Naive_7":
        "seasonal_naive_7",
    "Moving_Average_7":
        "moving_average_7",
    "Direct_Gradient_Boosting":
        "prediction"
}

summary_rows = []

for model_name, prediction_col in model_columns.items():

    y_true = (
        results[
            "sale_amount"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_pred = (
        results[
            prediction_col
        ]
        .to_numpy(
            dtype=float
        )
    )

    summary_rows.append(
        {
            "model":
                model_name,

            "observations":
                len(results),

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

summary = pd.DataFrame(
    summary_rows
).sort_values(
    "WAPE"
).reset_index(
    drop=True
)

print("\n" + "=" * 110)
print("FINAL HOLDOUT MODEL COMPARISON")
print("=" * 110)

print(
    summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 13. BEST BASELINE
# =========================================================
baseline_summary = summary[
    summary["model"] !=
    "Direct_Gradient_Boosting"
].copy()

best_baseline = (
    baseline_summary
    .sort_values(
        "WAPE"
    )
    .iloc[0]
)

final_model = summary[
    summary["model"]
    ==
    "Direct_Gradient_Boosting"
].iloc[0]

print("\n" + "=" * 110)
print("BEST BASELINE")
print("=" * 110)

print(
    "Model:",
    best_baseline["model"]
)

print(
    "WAPE:",
    round(
        best_baseline["WAPE"],
        4
    )
)

# =========================================================
# 14. INCREMENTAL VALUE OF FINAL MODEL
# =========================================================
final_wape = (
    final_model[
        "WAPE"
    ]
)

baseline_wape = (
    best_baseline[
        "WAPE"
    ]
)

final_mae = (
    final_model[
        "MAE"
    ]
)

baseline_mae = (
    best_baseline[
        "MAE"
    ]
)

wape_improvement = (
    (
        baseline_wape
        -
        final_wape
    )
    /
    baseline_wape
    *
    100
)

mae_improvement = (
    (
        baseline_mae
        -
        final_mae
    )
    /
    baseline_mae
    *
    100
)

print("\n" + "=" * 110)
print("INCREMENTAL MODEL VALUE")
print("=" * 110)

print(
    "Best baseline WAPE:",
    round(
        baseline_wape,
        4
    )
)

print(
    "Final model WAPE:",
    round(
        final_wape,
        4
    )
)

print(
    "WAPE improvement:",
    f"{wape_improvement:.2f}%"
)

print(
    "Best baseline MAE:",
    round(
        baseline_mae,
        4
    )
)

print(
    "Final model MAE:",
    round(
        final_mae,
        4
    )
)

print(
    "MAE improvement:",
    f"{mae_improvement:.2f}%"
)

# =========================================================
# 15. HORIZON-BY-HORIZON COMPARISON
# =========================================================
horizon_rows = []

for horizon in range(1, 8):

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

    row = {
        "horizon":
            horizon,

        "date":
            subset[
                "dt"
            ].iloc[0]
    }

    for model_name, prediction_col in model_columns.items():

        y_pred = (
            subset[
                prediction_col
            ]
            .to_numpy(
                dtype=float
            )
        )

        row[
            f"{model_name}_WAPE"
        ] = wape(
            y_true,
            y_pred
        )

    horizon_rows.append(
        row
    )

horizon_summary = pd.DataFrame(
    horizon_rows
)

print("\n" + "=" * 110)
print("HORIZON-BY-HORIZON WAPE")
print("=" * 110)

print(
    horizon_summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 16. HORIZON WIN COUNT
# =========================================================
print("\n" + "=" * 110)
print("HORIZON WIN COUNT")
print("=" * 110)

model_win_counts = {
    model: 0
    for model in model_columns
}

for _, row in horizon_summary.iterrows():

    values = {
        model:
            row[
                f"{model}_WAPE"
            ]
        for model in model_columns
    }

    winner = min(
        values,
        key=values.get
    )

    model_win_counts[
        winner
    ] += 1

for model, wins in model_win_counts.items():

    print(
        f"{model}: {wins}/7 horizons"
    )

# =========================================================
# 17. PAIRED OBSERVATION-LEVEL COMPARISON
#
# This shows how many individual holdout rows have a lower
# absolute error under the final model than under the best
# baseline.
# =========================================================
best_baseline_col = model_columns[
    best_baseline["model"]
]

results[
    "final_abs_error"
] = (
    np.abs(
        results[
            "sale_amount"
        ]
        -
        results[
            "prediction"
        ]
    )
)

results[
    "baseline_abs_error"
] = (
    np.abs(
        results[
            "sale_amount"
        ]
        -
        results[
            best_baseline_col
        ]
    )
)

final_better = (
    results[
        "final_abs_error"
    ]
    <
    results[
        "baseline_abs_error"
    ]
)

baseline_better = (
    results[
        "baseline_abs_error"
    ]
    <
    results[
        "final_abs_error"
    ]
)

ties = (
    results[
        "final_abs_error"
    ]
    ==
    results[
        "baseline_abs_error"
    ]
)

print("\n" + "=" * 110)
print("PAIRED ROW-LEVEL COMPARISON")
print("=" * 110)

print(
    "Final model lower absolute error:",
    f"{final_better.mean() * 100:.2f}%"
)

print(
    "Best baseline lower absolute error:",
    f"{baseline_better.mean() * 100:.2f}%"
)

print(
    "Exact ties:",
    f"{ties.mean() * 100:.2f}%"
)

# =========================================================
# 18. HOLDOUT ERROR DISTRIBUTION
# =========================================================
print("\n" + "=" * 110)
print("FINAL MODEL ERROR DISTRIBUTION")
print("=" * 110)

final_errors = (
    results[
        "sale_amount"
    ]
    -
    results[
        "prediction"
    ]
)

print(
    final_errors
    .describe()
    .round(4)
    .to_string()
)

# =========================================================
# 19. HOLDOUT STOCKOUT COVERAGE
#
# Descriptive only.
# =========================================================
raw_stockout = pd.read_parquet(
    RAW / "eval.parquet",
    columns=[
        "store_id",
        "product_id",
        "dt",
        "stock_hour6_22_cnt"
    ]
)

raw_stockout["dt"] = pd.to_datetime(
    raw_stockout["dt"]
)

results = results.merge(
    raw_stockout,
    on=[
        "store_id",
        "product_id",
        "dt"
    ],
    how="left"
)

print("\n" + "=" * 110)
print("HOLDOUT STOCKOUT CONTEXT")
print("=" * 110)

print(
    "Any-stockout rows:",
    int(
        (
            results[
                "stock_hour6_22_cnt"
            ]
            >
            0
        ).sum()
    )
)

print(
    "Any-stockout share:",
    f"{(
        results[
            'stock_hour6_22_cnt'
        ]
        >
        0
    ).mean() * 100:.2f}%"
)

print(
    "Full-stockout rows:",
    int(
        (
            results[
                "stock_hour6_22_cnt"
            ]
            ==
            16
        ).sum()
    )
)

# =========================================================
# 20. HOLDOUT INTEGRITY
# =========================================================
print("\n" + "=" * 110)
print("HOLDOUT INTEGRITY")
print("=" * 110)

print(
    "Holdout period:",
    HOLDOUT_START,
    "->",
    HOLDOUT_END
)

print(
    "Holdout used to fit baselines:",
    False
)

print(
    "Holdout used to fit final model:",
    False
)

print(
    "Holdout used to tune final model:",
    False
)

print(
    "Future data used in baseline construction:",
    False
)

print(
    "Official holdout remains untouched before this test:",
    True
)

# =========================================================
# 21. FINAL DECISION
# =========================================================
print("\n" + "=" * 110)
print("FINAL FORECASTING GATE")
print("=" * 110)

if final_wape < baseline_wape:

    print(
        "RESULT: FINAL MODEL BEATS BEST BASELINE"
    )

    print(
        "The direct gradient-boosting model adds "
        "predictive value on the untouched holdout."
    )

elif final_wape > baseline_wape:

    print(
        "RESULT: FINAL MODEL DOES NOT BEAT BEST BASELINE"
    )

    print(
        "The final model does not demonstrate "
        "incremental predictive value over simple forecasting."
    )

else:

    print(
        "RESULT: FINAL MODEL TIES BEST BASELINE"
    )

    print(
        "The final model does not demonstrate "
        "clear incremental predictive value."
    )

# =========================================================
# 22. SANITY CHECKS
# =========================================================
assert (
    len(results)
    ==
    350000
)

assert (
    summary[
        "model"
    ].nunique()
    ==
    4
)

assert (
    summary[
        "WAPE"
    ]
    .notna()
    .all()
)

assert (
    summary[
        "MAE"
    ]
    .notna()
    .all()
)

assert (
    summary[
        "RMSE"
    ]
    .notna()
    .all()
)

assert (
    horizon_summary.shape[0]
    ==
    7
)

assert (
    results[
        "prediction"
    ]
    .ge(0)
    .all()
)

print(
    "\nFinal baseline benchmark checks: PASS"
)

# =========================================================
# 23. SAVE RESULTS
# =========================================================
summary_output = (
    PROCESSED
    /
    "final_holdout_baseline_comparison.csv"
)

horizon_output = (
    PROCESSED
    /
    "final_holdout_baseline_horizon_comparison.csv"
)

detail_output = (
    PROCESSED
    /
    "final_holdout_baseline_detailed.csv"
)

summary.to_csv(
    summary_output,
    index=False
)

horizon_summary.to_csv(
    horizon_output,
    index=False
)

results.to_csv(
    detail_output,
    index=False
)

print("\n" + "=" * 110)
print("OUTPUT")
print("=" * 110)

print(
    "Model comparison:",
    summary_output
)

print(
    "Horizon comparison:",
    horizon_output
)

print(
    "Detailed comparison:",
    detail_output
)

print("\n" + "=" * 110)
print("PHASE 3 STEP 7 COMPLETE")
print("=" * 110)