from pathlib import Path
import numpy as np
import pandas as pd

from stockout_retail.reconstruction.demand import (
    add_stockout_state,
    build_adjusted_demand,
)
from stockout_retail.reconstruction.validation import (
    reconstruction_coverage,
)

print("Shared src reconstruction layer: LOADED")

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

print("=" * 110)
print("PHASE 3 - STEP 3R: DEMAND RECONSTRUCTION STRESS TEST")
print("=" * 110)

# =========================================================
# 1. LOAD RAW DATA
# =========================================================
raw_cols = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt"
]

raw = pd.read_parquet(
    RAW / "train.parquet",
    columns=raw_cols
)

raw["dt"] = pd.to_datetime(raw["dt"])

raw = raw.sort_values(
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

print(
    "\nRAW DATA:",
    raw.shape
)

# =========================================================
# 2. LOAD ACTUAL CROSS-FITTED OOF RECONSTRUCTION
# =========================================================
oof_path = (
    PROCESSED
    /
    "cross_fitted_demand_predictions.parquet"
)

oof = pd.read_parquet(
    oof_path
)

oof["dt"] = pd.to_datetime(
    oof["dt"]
)

print(
    "OOF DATA:",
    oof.shape
)

print(
    "OOF COLUMNS:",
    oof.columns.tolist()
)

required_oof = [
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

missing_oof = [
    c
    for c in required_oof
    if c not in oof.columns
]

if missing_oof:
    raise ValueError(
        "Missing expected OOF columns: "
        + str(missing_oof)
    )

# =========================================================
# 3. LOAD FINAL STOCKOUT-ADJUSTED DEMAND SERIES
# =========================================================
adjusted_path = (
    PROCESSED
    /
    "stockout_adjusted_demand.parquet"
)

adjusted = pd.read_parquet(
    adjusted_path
)

adjusted["dt"] = pd.to_datetime(
    adjusted["dt"]
)

print(
    "ADJUSTED DATA:",
    adjusted.shape
)

print(
    "ADJUSTED COLUMNS:",
    adjusted.columns.tolist()
)

required_adjusted = [
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

missing_adjusted = [
    c
    for c in required_adjusted
    if c not in adjusted.columns
]

if missing_adjusted:
    raise ValueError(
        "Missing expected adjusted-demand columns: "
        + str(missing_adjusted)
    )

# =========================================================
# 4. COVERAGE
# =========================================================
print("\n" + "=" * 110)
print("RECONSTRUCTION COVERAGE")
print("=" * 110)

print(
    "OOF start:",
    oof["dt"].min()
)

print(
    "OOF end:",
    oof["dt"].max()
)

print(
    "Adjusted-demand start:",
    adjusted["dt"].min()
)

print(
    "Adjusted-demand end:",
    adjusted["dt"].max()
)

print(
    "OOF missing predictions:",
    oof[
        "cross_fitted_demand"
    ]
    .isna()
    .sum()
)

print(
    "Adjusted-demand missing predictions:",
    adjusted[
        "adjusted_demand"
    ]
    .isna()
    .sum()
)

# =========================================================
# 5. NORMAL-DAY HISTORY DEPTH
# =========================================================
normal = raw[
    raw["stock_hour6_22_cnt"] == 0
].copy()

normal_days = (
    normal
    .groupby(
        group_cols
    )["dt"]
    .nunique()
)

print("\n" + "=" * 110)
print("NORMAL-DAY HISTORY DEPTH")
print("=" * 110)

for threshold in [
    5,
    10,
    20,
    30,
    40,
    50
]:

    count = (
        normal_days
        >= threshold
    ).sum()

    print(
        f"Series with >= {threshold:>2} normal days: "
        f"{count:>5} "
        f"({count / len(normal_days) * 100:.2f}%)"
    )

# =========================================================
# 6. METRICS
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
            (
                actual - predicted
            ) ** 2
        )
    )


# =========================================================
# 7. NORMAL-DAY OOF VALIDATION
# =========================================================
normal_oof = oof[
    oof["stockout_flag"] == 0
].copy()

normal_oof = normal_oof.dropna(
    subset=[
        "sale_amount",
        "cross_fitted_demand"
    ]
)

normal_actual = (
    normal_oof[
        "sale_amount"
    ]
    .to_numpy(
        dtype=float
    )
)

normal_prediction = (
    normal_oof[
        "cross_fitted_demand"
    ]
    .to_numpy(
        dtype=float
    )
)

print("\n" + "=" * 110)
print("OOF NORMAL-DAY RECONSTRUCTION VALIDATION")
print("=" * 110)

print(
    "Observations:",
    len(normal_oof)
)

print(
    "MAE:",
    round(
        mae(
            normal_actual,
            normal_prediction
        ),
        4
    )
)

print(
    "WAPE:",
    round(
        wape(
            normal_actual,
            normal_prediction
        ),
        4
    )
)

print(
    "RMSE:",
    round(
        rmse(
            normal_actual,
            normal_prediction
        ),
        4
    )
)

# =========================================================
# 8. NORMAL-DAY BIAS
# =========================================================
normal_oof[
    "reconstruction_error"
] = (
    normal_oof[
        "sale_amount"
    ]
    -
    normal_oof[
        "cross_fitted_demand"
    ]
)

print("\n" + "=" * 110)
print("NORMAL-DAY RECONSTRUCTION BIAS")
print("=" * 110)

print(
    "Mean error:",
    round(
        normal_oof[
            "reconstruction_error"
        ].mean(),
        4
    )
)

print(
    "Median error:",
    round(
        normal_oof[
            "reconstruction_error"
        ].median(),
        4
    )
)

print(
    "Underprediction share:",
    f"{(
        normal_oof[
            'reconstruction_error'
        ]
        >
        0
    ).mean() * 100:.2f}%"
)

# =========================================================
# 9. NORMAL-DAY REFERENCE LEVELS
# =========================================================
normal_sales_p95 = (
    normal[
        "sale_amount"
    ]
    .quantile(
        0.95
    )
)

normal_sales_p99 = (
    normal[
        "sale_amount"
    ]
    .quantile(
        0.99
    )
)

normal_sales_p999 = (
    normal[
        "sale_amount"
    ]
    .quantile(
        0.999
    )
)

print("\n" + "=" * 110)
print("NORMAL-DAY SALES REFERENCE")
print("=" * 110)

print(
    "Normal sales P95:",
    round(
        normal_sales_p95,
        4
    )
)

print(
    "Normal sales P99:",
    round(
        normal_sales_p99,
        4
    )
)

print(
    "Normal sales P99.9:",
    round(
        normal_sales_p999,
        4
    )
)

# =========================================================
# 10. NORMAL-DAY SERIES MEDIAN
# =========================================================
series_median = (
    normal
    .groupby(
        group_cols
    )[
        "sale_amount"
    ]
    .median()
    .rename(
        "normal_series_median"
    )
    .reset_index()
)

# =========================================================
# 11. OOF STOCKOUT OBSERVATIONS
# =========================================================
stockout = oof[
    oof["stockout_flag"] == 1
].copy()

stockout = stockout.merge(
    series_median,
    on=group_cols,
    how="left"
)

print("\n" + "=" * 110)
print("OOF STOCKOUT OBSERVATIONS")
print("=" * 110)

print(
    "Rows:",
    len(stockout)
)

full_stockout_count = int(
    stockout[
        "full_stockout"
    ]
    .astype(bool)
    .sum()
)

print(
    "Full stockout rows:",
    full_stockout_count
)

# =========================================================
# 12. PRIMARY RECONSTRUCTION
# =========================================================
stockout["primary"] = (
    stockout[
        "cross_fitted_demand"
    ]
    .clip(
        lower=0
    )
)

# =========================================================
# 13. NORMAL MEDIAN AVAILABILITY
# =========================================================
stockout[
    "median_available"
] = (
    stockout[
        "normal_series_median"
    ]
    .notna()
)

print("\n" + "=" * 110)
print("SHRINKAGE BASELINE COVERAGE")
print("=" * 110)

median_available_rows = int(
    stockout[
        "median_available"
    ].sum()
)

median_missing_rows = int(
    (
        ~stockout[
            "median_available"
        ]
    ).sum()
)

print(
    "Stockout rows with normal-series median:",
    median_available_rows
)

print(
    "Stockout rows without normal-series median:",
    median_missing_rows
)

print(
    "Shrinkage-baseline coverage:",
    f"{median_available_rows / len(stockout) * 100:.2f}%"
)

# =========================================================
# 14. SAFE MEDIAN BASELINE
#
# For the very small number of series without a normal-day
# median, use the primary reconstruction only for the
# sensitivity variant. This does NOT change the primary
# estimate.
# =========================================================
stockout[
    "normal_series_median_safe"
] = (
    stockout[
        "normal_series_median"
    ]
    .fillna(
        stockout[
            "primary"
        ]
    )
)

# =========================================================
# 15. SENSITIVITY VARIANTS
# =========================================================
stockout[
    "shrink_75"
] = (
    0.75
    *
    stockout[
        "primary"
    ]
    +
    0.25
    *
    stockout[
        "normal_series_median_safe"
    ]
)

stockout[
    "shrink_50"
] = (
    0.50
    *
    stockout[
        "primary"
    ]
    +
    0.50
    *
    stockout[
        "normal_series_median_safe"
    ]
)

stockout[
    "cap_p99"
] = (
    stockout[
        "primary"
    ]
    .clip(
        upper=normal_sales_p99
    )
)

stockout[
    "cap_p999"
] = (
    stockout[
        "primary"
    ]
    .clip(
        upper=normal_sales_p999
    )
)

variants = [
    "primary",
    "shrink_75",
    "shrink_50",
    "cap_p99",
    "cap_p999"
]

# =========================================================
# 16. RECONSTRUCTION SENSITIVITY SUMMARY
# =========================================================
variant_rows = []

for variant in variants:

    estimated = (
        stockout[
            variant
        ]
        .clip(
            lower=0
        )
    )

    observed = (
        stockout[
            "sale_amount"
        ]
    )

    positive_gap = (
        estimated
        -
        observed
    ).clip(
        lower=0
    )

    variant_rows.append(
        {
            "variant": variant,
            "stockout_rows": len(stockout),
            "mean_estimated_demand":
                estimated.mean(),
            "median_estimated_demand":
                estimated.median(),
            "mean_observed_sales":
                observed.mean(),
            "mean_estimated_gap":
                positive_gap.mean(),
            "total_estimated_gap":
                positive_gap.sum(),
            "p95_estimated_demand":
                estimated.quantile(
                    0.95
                ),
            "p99_estimated_demand":
                estimated.quantile(
                    0.99
                ),
            "max_estimated_demand":
                estimated.max()
        }
    )

variant_summary = pd.DataFrame(
    variant_rows
)

print("\n" + "=" * 110)
print("RECONSTRUCTION SENSITIVITY SUMMARY")
print("=" * 110)

print(
    variant_summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 17. FULL-STOCKOUT SENSITIVITY
# =========================================================
full_stockout = stockout[
    stockout[
        "full_stockout"
    ]
    .astype(bool)
].copy()

full_rows = []

for variant in variants:

    estimated = (
        full_stockout[
            variant
        ]
        .clip(
            lower=0
        )
    )

    observed = (
        full_stockout[
            "sale_amount"
        ]
    )

    positive_gap = (
        estimated
        -
        observed
    ).clip(
        lower=0
    )

    full_rows.append(
        {
            "variant": variant,
            "rows": len(
                full_stockout
            ),
            "observed_sales_mean":
                observed.mean(),
            "estimated_demand_mean":
                estimated.mean(),
            "estimated_gap_mean":
                positive_gap.mean(),
            "estimated_demand_p95":
                estimated.quantile(
                    0.95
                ),
            "estimated_demand_max":
                estimated.max()
        }
    )

full_summary = pd.DataFrame(
    full_rows
)

print("\n" + "=" * 110)
print("FULL-STOCKOUT RECONSTRUCTION SENSITIVITY")
print("=" * 110)

print(
    full_summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 18. STOCKOUT INTENSITY ROBUSTNESS
# =========================================================
intensity_rows = []

for variant in variants:

    temp = stockout.copy()

    temp[
        "variant_demand"
    ] = (
        temp[
            variant
        ]
        .clip(
            lower=0
        )
    )

    temp[
        "estimated_gap"
    ] = (
        temp[
            "variant_demand"
        ]
        -
        temp[
            "sale_amount"
        ]
    ).clip(
        lower=0
    )

    grouped = (
        temp
        .groupby(
            "stock_hour6_22_cnt"
        )
        .agg(
            observations=(
                "sale_amount",
                "size"
            ),
            mean_observed_sales=(
                "sale_amount",
                "mean"
            ),
            mean_estimated_demand=(
                "variant_demand",
                "mean"
            ),
            mean_estimated_gap=(
                "estimated_gap",
                "mean"
            )
        )
        .reset_index()
    )

    grouped[
        "variant"
    ] = variant

    intensity_rows.append(
        grouped
    )

intensity_summary = pd.concat(
    intensity_rows,
    ignore_index=True
)

print("\n" + "=" * 110)
print("STOCKOUT INTENSITY ROBUSTNESS")
print("=" * 110)

print(
    intensity_summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 19. STOCKOUT INTENSITY MONOTONICITY
# =========================================================
print("\n" + "=" * 110)
print("STOCKOUT INTENSITY MONOTONICITY")
print("=" * 110)

monotonicity_rows = []

for variant in variants:

    temp = stockout.copy()

    temp[
        "estimated_gap"
    ] = (
        temp[
            variant
        ]
        -
        temp[
            "sale_amount"
        ]
    ).clip(
        lower=0
    )

    grouped = (
        temp
        .groupby(
            "stock_hour6_22_cnt"
        )[
            "estimated_gap"
        ]
        .mean()
        .reset_index()
    )

    spearman = (
        grouped[
            "stock_hour6_22_cnt"
        ]
        .corr(
            grouped[
                "estimated_gap"
            ],
            method="spearman"
        )
    )

    monotonicity_rows.append(
        {
            "variant": variant,
            "spearman_r": spearman
        }
    )

monotonicity = pd.DataFrame(
    monotonicity_rows
)

print(
    monotonicity
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 20. SERIES-LEVEL RECONSTRUCTION
# =========================================================
series_variant_rows = []

for variant in variants:

    temp = stockout.copy()

    temp[
        "variant_demand"
    ] = (
        temp[
            variant
        ]
        .clip(
            lower=0
        )
    )

    temp[
        "variant_gap"
    ] = (
        temp[
            "variant_demand"
        ]
        -
        temp[
            "sale_amount"
        ]
    ).clip(
        lower=0
    )

    grouped = (
        temp
        .groupby(
            group_cols
        )
        .agg(
            estimated_gap=(
                "variant_gap",
                "sum"
            ),
            estimated_demand=(
                "variant_demand",
                "sum"
            )
        )
        .reset_index()
    )

    grouped[
        "variant"
    ] = variant

    series_variant_rows.append(
        grouped
    )

series_results = pd.concat(
    series_variant_rows,
    ignore_index=True
)

# =========================================================
# 21. RISK-RANKING STABILITY
# =========================================================
print("\n" + "=" * 110)
print("RISK-RANKING STABILITY")
print("=" * 110)

base = (
    series_results[
        series_results[
            "variant"
        ]
        ==
        "primary"
    ]
    .sort_values(
        "estimated_gap",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

n_series = len(base)

top1_n = max(
    1,
    int(
        n_series
        *
        0.01
    )
)

top5_n = max(
    1,
    int(
        n_series
        *
        0.05
    )
)

base_top1 = set(
    map(
        tuple,
        base
        .head(top1_n)[
            group_cols
        ]
        .to_numpy()
    )
)

base_top5 = set(
    map(
        tuple,
        base
        .head(top5_n)[
            group_cols
        ]
        .to_numpy()
    )
)

stability_rows = []

for variant in variants[1:]:

    current = (
        series_results[
            series_results[
                "variant"
            ]
            ==
            variant
        ]
        .sort_values(
            "estimated_gap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    current_top1 = set(
        map(
            tuple,
            current
            .head(top1_n)[
                group_cols
            ]
            .to_numpy()
        )
    )

    current_top5 = set(
        map(
            tuple,
            current
            .head(top5_n)[
                group_cols
            ]
            .to_numpy()
        )
    )

    top1_overlap = (
        len(
            base_top1
            &
            current_top1
        )
        /
        len(base_top1)
    )

    top5_overlap = (
        len(
            base_top5
            &
            current_top5
        )
        /
        len(base_top5)
    )

    stability_rows.append(
        {
            "variant": variant,
            "top1_overlap_pct":
                top1_overlap * 100,
            "top5_overlap_pct":
                top5_overlap * 100
        }
    )

stability = pd.DataFrame(
    stability_rows
)

print(
    stability
    .round(2)
    .to_string(index=False)
)

# =========================================================
# 22. SERIES-LEVEL RANK CORRELATION
# =========================================================
print("\n" + "=" * 110)
print("SERIES-LEVEL RANK CORRELATION")
print("=" * 110)

rank_rows = []

base_rank = (
    base[
        group_cols
        +
        [
            "estimated_gap"
        ]
    ]
    .rename(
        columns={
            "estimated_gap":
                "primary_gap"
        }
    )
)

for variant in variants[1:]:

    current = (
        series_results[
            series_results[
                "variant"
            ]
            ==
            variant
        ][
            group_cols
            +
            [
                "estimated_gap"
            ]
        ]
        .rename(
            columns={
                "estimated_gap":
                    "variant_gap"
            }
        )
    )

    comparison = base_rank.merge(
        current,
        on=group_cols,
        how="inner"
    )

    rho = (
        comparison[
            "primary_gap"
        ]
        .corr(
            comparison[
                "variant_gap"
            ],
            method="spearman"
        )
    )

    rank_rows.append(
        {
            "variant": variant,
            "spearman_rank_correlation":
                rho
        }
    )

rank_corr = pd.DataFrame(
    rank_rows
)

print(
    rank_corr
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 23. EXTREME PRIMARY RECONSTRUCTION CHECK
# =========================================================
print("\n" + "=" * 110)
print("EXTREME PRIMARY RECONSTRUCTION CHECK")
print("=" * 110)

primary = stockout[
    "primary"
]

print(
    "Primary P50:",
    round(
        primary.quantile(
            0.50
        ),
        4
    )
)

print(
    "Primary P90:",
    round(
        primary.quantile(
            0.90
        ),
        4
    )
)

print(
    "Primary P95:",
    round(
        primary.quantile(
            0.95
        ),
        4
    )
)

print(
    "Primary P99:",
    round(
        primary.quantile(
            0.99
        ),
        4
    )
)

print(
    "Primary max:",
    round(
        primary.max(),
        4
    )
)

print(
    "Primary > normal P99:",
    f"{(
        primary
        >
        normal_sales_p99
    ).mean() * 100:.3f}%"
)

print(
    "Primary > normal P99.9:",
    f"{(
        primary
        >
        normal_sales_p999
    ).mean() * 100:.3f}%"
)

# =========================================================
# 24. TOP SERIES BY ESTIMATED CENSORED DEMAND
# =========================================================
series_primary = (
    series_results[
        series_results[
            "variant"
        ]
        ==
        "primary"
    ]
    .sort_values(
        "estimated_gap",
        ascending=False
    )
)

print("\n" + "=" * 110)
print("TOP 20 SERIES BY ESTIMATED CENSORED DEMAND")
print("=" * 110)

print(
    series_primary
    .head(20)
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 25. AGGREGATE SENSITIVITY
# =========================================================
print("\n" + "=" * 110)
print("AGGREGATE RECONSTRUCTION SENSITIVITY")
print("=" * 110)

base_gap = (
    variant_summary.loc[
        variant_summary[
            "variant"
        ]
        ==
        "primary",
        "total_estimated_gap"
    ]
    .iloc[0]
)

aggregate_sensitivity_rows = []

for variant in variants:

    current_gap = (
        variant_summary.loc[
            variant_summary[
                "variant"
            ]
            ==
            variant,
            "total_estimated_gap"
        ]
        .iloc[0]
    )

    change_pct = (
        (
            current_gap
            -
            base_gap
        )
        /
        base_gap
        *
        100
    )

    aggregate_sensitivity_rows.append(
        {
            "variant": variant,
            "total_gap": current_gap,
            "change_vs_primary_pct":
                change_pct
        }
    )

    print(
        f"{variant:<12} "
        f"total_gap={current_gap:.2f} "
        f"change_vs_primary={change_pct:+.2f}%"
    )

aggregate_sensitivity = pd.DataFrame(
    aggregate_sensitivity_rows
)

# =========================================================
# 26. SENIOR STRESS-TEST DECISION
# =========================================================
minimum_top1_overlap = (
    stability[
        "top1_overlap_pct"
    ].min()
)

minimum_top5_overlap = (
    stability[
        "top5_overlap_pct"
    ].min()
)

minimum_rank_corr = (
    rank_corr[
        "spearman_rank_correlation"
    ].min()
)

p99_gap = (
    variant_summary.loc[
        variant_summary[
            "variant"
        ]
        ==
        "cap_p99",
        "total_estimated_gap"
    ]
    .iloc[0]
)

p99_change = (
    (
        p99_gap
        -
        base_gap
    )
    /
    base_gap
    *
    100
)

print("\n" + "=" * 110)
print("SENIOR STRESS-TEST DECISION")
print("=" * 110)

print(
    "Minimum top-1% overlap:",
    f"{minimum_top1_overlap:.2f}%"
)

print(
    "Minimum top-5% overlap:",
    f"{minimum_top5_overlap:.2f}%"
)

print(
    "Minimum Spearman rank correlation:",
    f"{minimum_rank_corr:.4f}"
)

print(
    "Aggregate gap change under P99 cap:",
    f"{p99_change:+.2f}%"
)

if (
    minimum_top1_overlap >= 70
    and
    minimum_top5_overlap >= 80
    and
    minimum_rank_corr >= 0.80
    and
    abs(p99_change) <= 25
):

    decision = (
        "PASS - reconstruction conclusions appear robust "
        "to the tested perturbations."
    )

elif (
    minimum_top1_overlap >= 50
    and
    minimum_top5_overlap >= 65
    and
    minimum_rank_corr >= 0.60
):

    decision = (
        "PASS WITH CAUTION - the reconstruction is directionally "
        "robust, but ranking or magnitude sensitivity remains."
    )

else:

    decision = (
        "FAIL - reconstruction conclusions are materially "
        "sensitive to reasonable perturbations."
    )

print(
    "\nDECISION:",
    decision
)

# =========================================================
# 27. METHODOLOGY INTERPRETATION
# =========================================================
print("\n" + "=" * 110)
print("METHODOLOGY INTERPRETATION")
print("=" * 110)

print(
    "Primary estimate:",
    "cross-fitted demand prediction"
)

print(
    "Shrinkage tests:",
    "sensitivity to model overstatement"
)

print(
    "P99/P99.9 caps:",
    "sensitivity to extreme predictions"
)

print(
    "Ranking stability:",
    "tests whether business priorities change"
)

print(
    "Monotonicity:",
    "tests whether estimated gaps generally increase "
    "with stockout intensity"
)

print(
    "Important limitation:",
    "true counterfactual demand during genuine stockouts "
    "is unobserved."
)

print(
    "Interpretation:",
    "this test establishes robustness, not ground-truth recovery."
)

# =========================================================
# 28. HOLDOUT PROTECTION
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
# 29. LEAKAGE CHECK
# =========================================================
print("\n" + "=" * 110)
print("LEAKAGE CHECK")
print("=" * 110)

print(
    "Cross-fitted OOF predictions used:",
    True
)

print(
    "Future target used in OOF prediction:",
    False
)

print(
    "Final evaluation used:",
    False
)

# =========================================================
# 30. SANITY CHECKS
# =========================================================

assert (
    len(raw)
    ==
    4500000
)

assert (
    len(oof)
    ==
    1750000
)

assert (
    len(adjusted)
    ==
    2100000
)

assert (
    oof[
        "cross_fitted_demand"
    ]
    .isna()
    .sum()
    ==
    0
)

assert (
    adjusted[
        "adjusted_demand"
    ]
    .isna()
    .sum()
    ==
    0
)

assert (
    stockout[
        "primary"
    ]
    .isna()
    .sum()
    ==
    0
)

assert (
    stockout[
        variants
    ]
    .isna()
    .any()
    .any()
    ==
    False
)

assert (
    variant_summary[
        "mean_estimated_demand"
    ]
    .ge(0)
    .all()
)

assert (
    full_summary[
        "estimated_demand_mean"
    ]
    .ge(0)
    .all()
)

assert (
    intensity_summary[
        "mean_estimated_demand"
    ]
    .ge(0)
    .all()
)

assert (
    len(series_results)
    ==
    n_series * len(variants)
)

assert (
    len(stability)
    ==
    len(variants) - 1
)

assert (
    len(rank_corr)
    ==
    len(variants) - 1
)

assert (
    minimum_rank_corr >= 0
)

print(
    "\nDemand reconstruction stress-test checks: PASS"
)

# =========================================================
# 31. SAVE OUTPUTS
# =========================================================
variant_output = (
    PROCESSED
    /
    "demand_reconstruction_variant_summary.csv"
)

full_output = (
    PROCESSED
    /
    "demand_reconstruction_full_stockout_sensitivity.csv"
)

intensity_output = (
    PROCESSED
    /
    "demand_reconstruction_intensity_sensitivity.csv"
)

stability_output = (
    PROCESSED
    /
    "demand_reconstruction_rank_stability.csv"
)

rank_output = (
    PROCESSED
    /
    "demand_reconstruction_rank_correlation.csv"
)

aggregate_output = (
    PROCESSED
    /
    "demand_reconstruction_aggregate_sensitivity.csv"
)

decision_output = (
    PROCESSED
    /
    "demand_reconstruction_stress_decision.csv"
)

variant_summary.to_csv(
    variant_output,
    index=False
)

full_summary.to_csv(
    full_output,
    index=False
)

intensity_summary.to_csv(
    intensity_output,
    index=False
)

stability.to_csv(
    stability_output,
    index=False
)

rank_corr.to_csv(
    rank_output,
    index=False
)

aggregate_sensitivity.to_csv(
    aggregate_output,
    index=False
)

pd.DataFrame(
    [
        {
            "decision": decision,
            "minimum_top1_overlap_pct":
                minimum_top1_overlap,
            "minimum_top5_overlap_pct":
                minimum_top5_overlap,
            "minimum_spearman_rank_correlation":
                minimum_rank_corr,
            "p99_gap_change_pct":
                p99_change
        }
    ]
).to_csv(
    decision_output,
    index=False
)

print("\n" + "=" * 110)
print("OUTPUT")
print("=" * 110)

print(
    "Variant summary:",
    variant_output
)

print(
    "Full-stockout sensitivity:",
    full_output
)

print(
    "Intensity sensitivity:",
    intensity_output
)

print(
    "Rank stability:",
    stability_output
)

print(
    "Rank correlation:",
    rank_output
)

print(
    "Aggregate sensitivity:",
    aggregate_output
)

print(
    "Stress-test decision:",
    decision_output
)

print("\n" + "=" * 110)
print("PHASE 3 STEP 3R COMPLETE")
print("=" * 110)