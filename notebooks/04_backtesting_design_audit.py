from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw")

print("=" * 100)
print("PHASE 1 - STEP 4: BACKTESTING DESIGN AUDIT")
print("=" * 100)

# =========================================================
# 1. LOAD ONLY DATE / KEY COLUMNS
# =========================================================
train = pd.read_parquet(
    RAW / "train.parquet",
    columns=[
        "store_id",
        "product_id",
        "dt"
    ]
)

eval_df = pd.read_parquet(
    RAW / "eval.parquet",
    columns=[
        "store_id",
        "product_id",
        "dt"
    ]
)

train["dt"] = pd.to_datetime(train["dt"])
eval_df["dt"] = pd.to_datetime(eval_df["dt"])

train_dates = np.array(
    sorted(train["dt"].unique())
)

eval_dates = np.array(
    sorted(eval_df["dt"].unique())
)

print("\nDATA PERIODS")
print("Train start:", train_dates.min())
print("Train end  :", train_dates.max())
print("Eval start :", eval_dates.min())
print("Eval end   :", eval_dates.max())

# =========================================================
# 2. FORECAST HORIZON
# =========================================================
print("\n" + "=" * 100)
print("FORECAST HORIZON")
print("=" * 100)

print(
    "Final evaluation horizon:",
    len(eval_dates),
    "days"
)

print(
    "Recommendation:",
    "7-day ahead forecasting"
)

print(
    "Reason:",
    "The official evaluation set contains exactly "
    "7 consecutive future dates."
)

# =========================================================
# 3. CANDIDATE ROLLING VALIDATION WINDOWS
# =========================================================
print("\n" + "=" * 100)
print("CANDIDATE ROLLING-ORIGIN VALIDATION")
print("=" * 100)

# We reserve the first 28 days as history so that
# lag-7 and rolling features are available.
#
# Each validation fold predicts the next 7 days.

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

fold_df = pd.DataFrame(folds)

fold_df["train_end"] = pd.to_datetime(
    fold_df["train_end"]
)

fold_df["valid_start"] = pd.to_datetime(
    fold_df["valid_start"]
)

fold_df["valid_end"] = pd.to_datetime(
    fold_df["valid_end"]
)

print(
    fold_df.to_string(index=False)
)

# =========================================================
# 4. VALIDATE FOLD DATES AGAINST TRAIN DATA
# =========================================================
print("\n" + "=" * 100)
print("FOLD DATE VALIDATION")
print("=" * 100)

for _, row in fold_df.iterrows():

    train_end = row["train_end"]
    valid_start = row["valid_start"]
    valid_end = row["valid_end"]

    train_available = train[
        train["dt"] <= train_end
    ]

    validation_available = train[
        train["dt"].between(
            valid_start,
            valid_end
        )
    ]

    train_days = (
        train_available["dt"]
        .nunique()
    )

    valid_days = (
        validation_available["dt"]
        .nunique()
    )

    leakage = (
        valid_start
        <=
        train_end
    )

    print(
        f"Fold {int(row['fold'])}: "
        f"train_days={train_days:>3} "
        f"valid_days={valid_days:>2} "
        f"leakage={leakage}"
    )

# =========================================================
# 5. SERIES COVERAGE PER FOLD
# =========================================================
print("\n" + "=" * 100)
print("STORE-PRODUCT COVERAGE PER FOLD")
print("=" * 100)

full_pairs = (
    train[
        ["store_id", "product_id"]
    ]
    .drop_duplicates()
)

print(
    "Total store-product series:",
    len(full_pairs)
)

for _, row in fold_df.iterrows():

    train_end = row["train_end"]
    valid_start = row["valid_start"]
    valid_end = row["valid_end"]

    valid_pairs = (
        train[
            train["dt"].between(
                valid_start,
                valid_end
            )
        ]
        [
            ["store_id", "product_id"]
        ]
        .drop_duplicates()
    )

    valid_pair_count = len(valid_pairs)

    print(
        f"Fold {int(row['fold'])}: "
        f"validation series={valid_pair_count:,}"
    )

# =========================================================
# 6. LAG FEASIBILITY
# =========================================================
print("\n" + "=" * 100)
print("LAG / ROLLING FEATURE FEASIBILITY")
print("=" * 100)

candidate_features = {
    "lag_1": "previous day",
    "lag_2": "two days earlier",
    "lag_7": "same weekday previous week",
    "rolling_7_mean": "previous 7 days",
    "rolling_14_mean": "previous 14 days",
    "rolling_28_mean": "previous 28 days",
    "rolling_7_std": "previous 7 days variability"
}

for feature, definition in candidate_features.items():

    print(
        f"{feature:<20} {definition}"
    )

print(
    "\nImportant:"
)

print(
    "Rolling features must be shifted so that "
    "the forecast date itself is never included."
)

# =========================================================
# 7. DEMONSTRATE LEAKAGE-SAFE FEATURE LOGIC
# =========================================================
print("\n" + "=" * 100)
print("LEAKAGE-SAFE FEATURE EXAMPLE")
print("=" * 100)

example = pd.DataFrame(
    {
        "dt": pd.date_range(
            "2024-04-01",
            periods=10,
            freq="D"
        ),
        "sale_amount": np.arange(
            10,
            20
        )
    }
)

example["lag_1"] = (
    example["sale_amount"]
    .shift(1)
)

example["rolling_7_mean"] = (
    example["sale_amount"]
    .shift(1)
    .rolling(7)
    .mean()
)

print(
    example.to_string(
        index=False
    )
)

print(
    "\nThe shift(1) is mandatory because "
    "today's target cannot be used in today's predictors."
)

# =========================================================
# 8. WEATHER POLICY
# =========================================================
print("\n" + "=" * 100)
print("WEATHER MODEL POLICY")
print("=" * 100)

print(
    "Primary model:",
    "DO NOT use future realized weather."
)

print(
    "Reason:",
    "Realized future weather is not necessarily available "
    "when the forecast is generated."
)

print(
    "Historical weather:",
    "May be used as lagged features."
)

print(
    "Future weather forecast:",
    "Potential secondary scenario if a forecast source exists."
)

# =========================================================
# 9. PROMOTION / DISCOUNT POLICY
# =========================================================
print("\n" + "=" * 100)
print("PROMOTION / DISCOUNT POLICY")
print("=" * 100)

print(
    "Historical discount:",
    "ALLOW as lagged information."
)

print(
    "Future discount:",
    "ALLOW only if demonstrably known before forecast creation."
)

print(
    "Same-day realized discount:",
    "DO NOT use to predict same-day sales."
)

# =========================================================
# 10. FINAL EVALUATION HOLDOUT
# =========================================================
print("\n" + "=" * 100)
print("FINAL HOLDOUT POLICY")
print("=" * 100)

print(
    "Final evaluation period:",
    f"{eval_dates.min().date()} to {eval_dates.max().date()}"
)

print(
    "Final evaluation days:",
    len(eval_dates)
)

print(
    "Final evaluation data will remain untouched "
    "until model selection is complete."
)

# =========================================================
# 11. PROPOSED VALIDATION DESIGN
# =========================================================
print("\n" + "=" * 100)
print("PROPOSED VALIDATION DESIGN")
print("=" * 100)

print(
    "Method:",
    "rolling-origin 7-day forecasting"
)

print(
    "Validation folds:",
    len(fold_df)
)

print(
    "Selection metric:",
    "weighted MAE + WAPE"
)

print(
    "Secondary metrics:",
    "RMSE, MASE, pinball loss, interval coverage"
)

print(
    "Final test:",
    "official 7-day evaluation set"
)

# =========================================================
# 12. SANITY CHECK
# =========================================================
print("\n" + "=" * 100)
print("SANITY CHECKS")
print("=" * 100)

assert not train_dates[-1] >= eval_dates[0]

assert len(
    eval_dates
) == 7

assert len(
    fold_df
) == 5

assert all(
    fold_df["train_end"]
    <
    fold_df["valid_start"]
)

assert all(
    fold_df["valid_start"]
    <=
    fold_df["valid_end"]
)

print(
    "All backtesting design checks: PASS"
)

print("\n" + "=" * 100)
print("BACKTESTING DESIGN AUDIT COMPLETE")
print("=" * 100)