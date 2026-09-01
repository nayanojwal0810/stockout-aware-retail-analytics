from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")

print("=" * 100)
print("PHASE 1 - STEP 2: STOCKOUT DIAGNOSTICS")
print("=" * 100)

train = pd.read_parquet(
    RAW / "train.parquet",
    columns=[
        "store_id",
        "product_id",
        "dt",
        "sale_amount",
        "hours_sale",
        "stock_hour6_22_cnt",
        "hours_stock_status",
        "discount",
        "holiday_flag",
        "activity_flag"
    ]
)

train["dt"] = pd.to_datetime(train["dt"])

# ---------------------------------------------------------
# 1. Basic stockout intensity
# ---------------------------------------------------------
train["stockout_flag"] = (
    train["stock_hour6_22_cnt"] > 0
)

train["full_stockout_flag"] = (
    train["stock_hour6_22_cnt"] >= 16
)

print("\n" + "=" * 100)
print("STOCKOUT FREQUENCY")
print("=" * 100)

print(
    "Any stockout:",
    int(train["stockout_flag"].sum()),
    f"({train['stockout_flag'].mean() * 100:.2f}%)"
)

print(
    "Full 16-hour stockout:",
    int(train["full_stockout_flag"].sum()),
    f"({train['full_stockout_flag'].mean() * 100:.2f}%)"
)

# ---------------------------------------------------------
# 2. Sales by stockout intensity
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("SALES BY STOCKOUT HOURS")
print("=" * 100)

summary = (
    train.groupby("stock_hour6_22_cnt")["sale_amount"]
    .agg(
        observations="size",
        mean="mean",
        median="median",
        p90=lambda x: x.quantile(0.90),
        zero_rate=lambda x: (x == 0).mean()
    )
)

print(
    summary.round(3).to_string()
)

# ---------------------------------------------------------
# 3. Zero sales by stockout intensity
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("ZERO-SALES RATE BY STOCKOUT HOURS")
print("=" * 100)

zero_summary = (
    train.groupby("stock_hour6_22_cnt")["sale_amount"]
    .agg(
        observations="size",
        zero_sales=lambda x: (x == 0).sum(),
        zero_rate=lambda x: (x == 0).mean()
    )
)

print(
    zero_summary.round(4).to_string()
)

# ---------------------------------------------------------
# 4. Hourly stock status interpretation
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("HOURLY STOCK STATUS EXAMPLES")
print("=" * 100)

for i, value in enumerate(
    train["hours_stock_status"].head(10),
    start=1
):
    arr = np.asarray(value)

    print(
        f"Row {i}: length={len(arr)}, "
        f"sum={arr.sum()}, "
        f"unique={np.unique(arr).tolist()}, "
        f"values={arr.tolist()}"
    )

# ---------------------------------------------------------
# 5. Validate aggregate count against hourly vector
# ---------------------------------------------------------
def stock_count_from_hourly(arr):
    arr = np.asarray(arr)

    # Dataset's aggregate field is for 06:00-22:00.
    # Python positions 6 through 21 represent those 16 hours.
    return int(arr[6:22].sum())


sample = train.head(10000).copy()

sample["hourly_recomputed_stockout"] = (
    sample["hours_stock_status"]
    .apply(stock_count_from_hourly)
)

sample["count_matches"] = (
    sample["hourly_recomputed_stockout"]
    ==
    sample["stock_hour6_22_cnt"]
)

print("\n" + "=" * 100)
print("AGGREGATE STOCKOUT CONSISTENCY")
print("=" * 100)

print(
    "Rows checked:",
    len(sample)
)

print(
    "Matching rows:",
    int(sample["count_matches"].sum())
)

print(
    "Mismatching rows:",
    int((~sample["count_matches"]).sum())
)

print(
    "Match rate:",
    f"{sample['count_matches'].mean() * 100:.2f}%"
)

# ---------------------------------------------------------
# 6. Stockout intensity distribution
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("STOCKOUT INTENSITY DISTRIBUTION")
print("=" * 100)

bins = [-1, 0, 3, 7, 11, 15, 16]
labels = [
    "0 hours",
    "1-3 hours",
    "4-7 hours",
    "8-11 hours",
    "12-15 hours",
    "16 hours"
]

train["stockout_bucket"] = pd.cut(
    train["stock_hour6_22_cnt"],
    bins=bins,
    labels=labels
)

bucket_summary = (
    train.groupby(
        "stockout_bucket",
        observed=False
    )["sale_amount"]
    .agg(
        observations="size",
        mean="mean",
        median="median",
        zero_rate=lambda x: (x == 0).mean()
    )
)

print(
    bucket_summary.round(3).to_string()
)

# ---------------------------------------------------------
# 7. Stockout rate by store-product series
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("SERIES-LEVEL STOCKOUT EXPOSURE")
print("=" * 100)

series = (
    train.groupby(
        ["store_id", "product_id"]
    )
    .agg(
        days=("dt", "nunique"),
        stockout_days=("stockout_flag", "sum"),
        full_stockout_days=("full_stockout_flag", "sum"),
        avg_stockout_hours=("stock_hour6_22_cnt", "mean"),
        avg_sales=("sale_amount", "mean"),
        zero_sales_rate=("sale_amount", lambda x: (x == 0).mean())
    )
)

print(
    series[
        [
            "stockout_days",
            "full_stockout_days",
            "avg_stockout_hours",
            "avg_sales",
            "zero_sales_rate"
        ]
    ]
    .describe()
    .round(3)
    .to_string()
)

print("\n" + "=" * 100)
print("STOCKOUT DIAGNOSTICS COMPLETE")
print("=" * 100)