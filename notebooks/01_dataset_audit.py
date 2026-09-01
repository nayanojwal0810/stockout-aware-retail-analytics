from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw")

print("=" * 100)
print("PHASE 1 - STEP 1: DATASET & LEAKAGE AUDIT")
print("=" * 100)

# =========================================================
# 1. LOAD DATA
# =========================================================
train = pd.read_parquet(
    RAW / "train.parquet"
)

eval_df = pd.read_parquet(
    RAW / "eval.parquet"
)

print("\n" + "=" * 100)
print("DATASET SHAPES")
print("=" * 100)
print("Train:", train.shape)
print("Eval :", eval_df.shape)

# =========================================================
# 2. SCHEMA
# =========================================================
print("\n" + "=" * 100)
print("SCHEMA")
print("=" * 100)
print(train.dtypes.to_string())

# =========================================================
# 3. DATE COVERAGE
# =========================================================
print("\n" + "=" * 100)
print("DATE COVERAGE")
print("=" * 100)

train["dt"] = pd.to_datetime(train["dt"])
eval_df["dt"] = pd.to_datetime(eval_df["dt"])

print("Train start:", train["dt"].min())
print("Train end  :", train["dt"].max())
print("Eval start :", eval_df["dt"].min())
print("Eval end   :", eval_df["dt"].max())

temporal_overlap = (
    max(train["dt"].min(), eval_df["dt"].min())
    <=
    min(train["dt"].max(), eval_df["dt"].max())
)

print(
    "Train/Eval temporal overlap:",
    temporal_overlap
)

# =========================================================
# 4. ENTITY COVERAGE
# =========================================================
print("\n" + "=" * 100)
print("ENTITY COVERAGE")
print("=" * 100)

for name, df in [
    ("TRAIN", train),
    ("EVAL", eval_df)
]:
    print(f"\n{name}")

    print("Cities:", df["city_id"].nunique())
    print("Stores:", df["store_id"].nunique())
    print("Products:", df["product_id"].nunique())

    print(
        "Store-product pairs:",
        df[
            ["store_id", "product_id"]
        ].drop_duplicates().shape[0]
    )

    print("Dates:", df["dt"].nunique())

# =========================================================
# 5. MISSING VALUES
# =========================================================
print("\n" + "=" * 100)
print("MISSING VALUES")
print("=" * 100)

for name, df in [
    ("TRAIN", train),
    ("EVAL", eval_df)
]:
    print(f"\n{name}")

    missing = df.isna().sum()

    if missing.sum() == 0:
        print("No missing values.")
    else:
        for col, count in missing.items():
            if count > 0:
                pct = count / len(df) * 100

                print(
                    f"{col:<25}"
                    f"missing={count:>10,}"
                    f"pct={pct:>8.3f}%"
                )

# =========================================================
# 6. DUPLICATE & GRAIN AUDIT
# =========================================================
print("\n" + "=" * 100)
print("DUPLICATE & GRAIN AUDIT")
print("=" * 100)

key_cols = [
    "store_id",
    "product_id",
    "dt"
]

train_duplicate_keys = train.duplicated(
    subset=key_cols
).sum()

eval_duplicate_keys = eval_df.duplicated(
    subset=key_cols
).sum()

print(
    "Train duplicate store-product-date keys:",
    train_duplicate_keys
)

print(
    "Eval duplicate store-product-date keys:",
    eval_duplicate_keys
)

# Array-like columns cannot be safely hashed by pandas.
scalar_cols = [
    col
    for col in train.columns
    if col not in [
        "hours_sale",
        "hours_stock_status"
    ]
]

print(
    "Train duplicate scalar-field rows:",
    train.duplicated(
        subset=scalar_cols
    ).sum()
)

print(
    "Eval duplicate scalar-field rows:",
    eval_df.duplicated(
        subset=scalar_cols
    ).sum()
)

# =========================================================
# 7. SERIES LENGTH
# =========================================================
print("\n" + "=" * 100)
print("SERIES LENGTH")
print("=" * 100)

train_series_length = (
    train.groupby(
        ["store_id", "product_id"]
    )["dt"]
    .nunique()
)

eval_series_length = (
    eval_df.groupby(
        ["store_id", "product_id"]
    )["dt"]
    .nunique()
)

print("TRAIN")
print(
    train_series_length
    .describe()
    .round(2)
    .to_string()
)

print("\nEVAL")
print(
    eval_series_length
    .describe()
    .round(2)
    .to_string()
)

# =========================================================
# 8. TARGET SANITY
# =========================================================
print("\n" + "=" * 100)
print("TARGET SANITY")
print("=" * 100)

for name, df in [
    ("TRAIN", train),
    ("EVAL", eval_df)
]:
    sales = df["sale_amount"]

    print(f"\n{name}")
    print("min   :", sales.min())
    print("p50   :", sales.quantile(0.50))
    print("p90   :", sales.quantile(0.90))
    print("p95   :", sales.quantile(0.95))
    print("p99   :", sales.quantile(0.99))
    print("max   :", sales.max())
    print("mean  :", sales.mean())

    zeros = (sales == 0).sum()

    print(
        "zeros :",
        zeros,
        f"({zeros / len(df) * 100:.2f}%)"
    )

    print(
        "negative:",
        (sales < 0).sum()
    )

# =========================================================
# 9. STOCKOUT NUMERIC FIELD
# =========================================================
print("\n" + "=" * 100)
print("STOCKOUT COUNT FIELD")
print("=" * 100)

for name, df in [
    ("TRAIN", train),
    ("EVAL", eval_df)
]:
    col = "stock_hour6_22_cnt"

    print(f"\n{name} - {col}")

    print("dtype :", df[col].dtype)
    print("unique:", df[col].nunique())
    print("min   :", df[col].min())
    print("median:", df[col].median())
    print("max   :", df[col].max())

    zero_count = (
        df[col] == 0
    ).sum()

    print(
        "zero  :",
        zero_count,
        f"({zero_count / len(df) * 100:.2f}%)"
    )

    print("\nValue counts:")
    print(
        df[col]
        .value_counts()
        .sort_index()
        .to_string()
    )

# =========================================================
# 10. ARRAY FIELD INSPECTION
# =========================================================
print("\n" + "=" * 100)
print("ARRAY FIELD INSPECTION")
print("=" * 100)

array_cols = [
    "hours_sale",
    "hours_stock_status"
]

for col in array_cols:

    print(f"\nFIELD: {col}")

    sample = train[col].head(5)

    for i, value in enumerate(
        sample,
        start=1
    ):
        print(
            f"Sample {i}:"
        )
        print(
            "  Python type:",
            type(value)
        )
        print(
            "  Value:",
            value
        )

        if isinstance(
            value,
            np.ndarray
        ):
            print(
                "  Shape:",
                value.shape
            )
            print(
                "  Length:",
                len(value)
            )
            print(
                "  Dtype:",
                value.dtype
            )

# =========================================================
# 11. ARRAY LENGTH CONSISTENCY
# =========================================================
print("\n" + "=" * 100)
print("ARRAY LENGTH CONSISTENCY")
print("=" * 100)

for col in array_cols:

    lengths = train[col].apply(
        lambda x: len(x)
        if isinstance(x, (list, np.ndarray))
        else np.nan
    )

    print(f"\n{col}")

    print(
        lengths
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

# =========================================================
# 12. ARRAY NULL / VALUE CHECK
# =========================================================
print("\n" + "=" * 100)
print("ARRAY VALUE INSPECTION")
print("=" * 100)

for col in array_cols:

    sample_values = []

    for value in train[col].head(100):

        if isinstance(
            value,
            (list, np.ndarray)
        ):
            sample_values.extend(
                np.asarray(value).flatten().tolist()
            )

    print(f"\n{col}")

    if sample_values:

        sample_series = pd.Series(
            sample_values
        )

        print(
            "Flattened sample count:",
            len(sample_series)
        )

        print(
            "Unique sample values:",
            sample_series.nunique()
        )

        print(
            "Top sample values:"
        )

        print(
            sample_series
            .value_counts()
            .head(15)
            .to_string()
        )

# =========================================================
# 13. STOCKOUT / SALES RELATIONSHIP
# =========================================================
print("\n" + "=" * 100)
print("SALES VS STOCKOUT COUNT")
print("=" * 100)

train["stockout_candidate"] = (
    train["stock_hour6_22_cnt"] > 0
)

eval_df["stockout_candidate"] = (
    eval_df["stock_hour6_22_cnt"] > 0
)

for name, df in [
    ("TRAIN", train),
    ("EVAL", eval_df)
]:

    summary = (
        df.groupby(
            "stockout_candidate"
        )["sale_amount"]
        .agg(
            rows="size",
            mean="mean",
            median="median",
            p90=lambda x:
                x.quantile(0.90)
        )
    )

    print(f"\n{name}")

    print(
        summary.round(3)
        .to_string()
    )

# =========================================================
# 14. OTHER FEATURE DISTRIBUTIONS
# =========================================================
print("\n" + "=" * 100)
print("NUMERIC FEATURE SUMMARY")
print("=" * 100)

numeric_features = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level"
]

print(
    train[
        numeric_features
    ]
    .describe()
    .T
    .round(3)
    .to_string()
)

# =========================================================
# 15. POTENTIAL LEAKAGE CANDIDATES
# =========================================================
print("\n" + "=" * 100)
print("POTENTIAL LEAKAGE REVIEW")
print("=" * 100)

print(
    "Target:",
    "sale_amount"
)

print(
    "\nPotentially sales-derived fields:"
)

for col in train.columns:

    if any(
        term in col.lower()
        for term in [
            "sale",
            "sales",
            "target",
            "revenue"
        ]
    ):
        print(
            "-",
            col
        )

print(
    "\nFields requiring timing validation:"
)

for col in [
    "hours_sale",
    "stock_hour6_22_cnt",
    "hours_stock_status",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level"
]:
    print(
        "-",
        col
    )

# =========================================================
# 16. TRAIN / EVAL STORE-PRODUCT OVERLAP
# =========================================================
print("\n" + "=" * 100)
print("TRAIN / EVAL STORE-PRODUCT OVERLAP")
print("=" * 100)

train_pairs = set(
    zip(
        train["store_id"],
        train["product_id"]
    )
)

eval_pairs = set(
    zip(
        eval_df["store_id"],
        eval_df["product_id"]
    )
)

print(
    "Train pairs:",
    len(train_pairs)
)

print(
    "Eval pairs:",
    len(eval_pairs)
)

print(
    "Shared pairs:",
    len(
        train_pairs
        &
        eval_pairs
    )
)

print(
    "Eval-only pairs:",
    len(
        eval_pairs
        -
        train_pairs
    )
)

# =========================================================
# 17. DAILY ROW COUNTS
# =========================================================
print("\n" + "=" * 100)
print("DAILY OBSERVATION COUNTS")
print("=" * 100)

train_daily = (
    train.groupby("dt")
    .size()
)

eval_daily = (
    eval_df.groupby("dt")
    .size()
)

print("TRAIN")
print(
    train_daily
    .describe()
    .round(2)
    .to_string()
)

print("\nEVAL")
print(
    eval_daily
    .describe()
    .round(2)
    .to_string()
)

# =========================================================
# 18. EXPECTED FULL GRID
# =========================================================
print("\n" + "=" * 100)
print("DATASET GRID CHECK")
print("=" * 100)

train_expected = (
    50000 * 90
)

eval_expected = (
    50000 * 7
)

print(
    "Expected train rows:",
    train_expected
)

print(
    "Actual train rows:",
    len(train)
)

print(
    "Expected eval rows:",
    eval_expected
)

print(
    "Actual eval rows:",
    len(eval_df)
)

# =========================================================
# 19. SERIES DATE CONTINUITY
# =========================================================
print("\n" + "=" * 100)
print("SERIES DATE CONTINUITY")
print("=" * 100)

train_series = (
    train.groupby(
        [
            "store_id",
            "product_id"
        ]
    )["dt"]
    .agg(
        start="min",
        end="max",
        days="nunique"
    )
)

eval_series = (
    eval_df.groupby(
        [
            "store_id",
            "product_id"
        ]
    )["dt"]
    .agg(
        start="min",
        end="max",
        days="nunique"
    )
)

print("TRAIN")
print(
    train_series
    .describe()
    .round(2)
    .to_string()
)

print("\nEVAL")
print(
    eval_series
    .describe()
    .round(2)
    .to_string()
)

# =========================================================
# 20. FINAL AUDIT SUMMARY
# =========================================================
print("\n" + "=" * 100)
print("AUDIT COMPLETE")
print("=" * 100)

print(
    "Temporal overlap:",
    temporal_overlap
)

print(
    "Train duplicate keys:",
    train_duplicate_keys
)

print(
    "Eval duplicate keys:",
    eval_duplicate_keys
)

print(
    "Train missing values:",
    int(
        train.isna()
        .sum()
        .sum()
    )
)

print(
    "Eval missing values:",
    int(
        eval_df.isna()
        .sum()
        .sum()
    )
)

print(
    "Train store-product pairs:",
    len(train_pairs)
)

print(
    "Eval store-product pairs:",
    len(eval_pairs)
)

print("\nDo not define the final stockout rule until")
print("the array fields and dataset documentation are reviewed.")