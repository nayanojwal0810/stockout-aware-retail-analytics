from pathlib import Path
import numpy as np
import pandas as pd

from stockout_retail.inventory.policy import (
    service_level_safety_stock,
    risk_adjusted_inventory_target,
)
from stockout_retail.inventory.sensitivity import risk_band_impact

print("Shared src inventory layer: LOADED")

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

print("=" * 110)
print("PHASE 3 - STEP 5B: POLICY SENSITIVITY BY STOCKOUT-RISK INTENSITY")
print("=" * 110)

# =========================================================
# 1. LOAD DIRECT FORECASTS
# =========================================================
forecast_path = (
    PROCESSED
    / "inventory_backtest_forecasts.csv"
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

# =========================================================
# 2. LOAD RAW ACTUAL SALES + STOCKOUT HISTORY
# =========================================================
raw = pd.read_parquet(
    RAW / "train.parquet",
    columns=[
        "store_id",
        "product_id",
        "dt",
        "sale_amount",
        "stock_hour6_22_cnt"
    ]
)

raw["dt"] = pd.to_datetime(
    raw["dt"]
)

print(
    "Raw rows:",
    len(raw)
)

group_cols = [
    "store_id",
    "product_id"
]

# =========================================================
# 3. LOAD CROSS-FITTED ADJUSTED DEMAND
# =========================================================
adjusted_path = (
    PROCESSED
    /
    "stockout_adjusted_demand.parquet"
)

adjusted = pd.read_parquet(
    adjusted_path,
    columns=[
        "store_id",
        "product_id",
        "dt",
        "adjusted_demand",
        "stockout_flag",
        "stock_hour6_22_cnt"
    ]
)

adjusted["dt"] = pd.to_datetime(
    adjusted["dt"]
)

print(
    "Adjusted-demand rows:",
    len(adjusted)
)

# =========================================================
# 4. EVALUATION FOLDS
#
# Adjusted demand begins on 2024-05-15, therefore only
# folds 3-5 have a valid adjusted-demand target.
# =========================================================
evaluation_folds = [
    3,
    4,
    5
]

fold_origins = {
    3: pd.Timestamp("2024-05-14"),
    4: pd.Timestamp("2024-05-21"),
    5: pd.Timestamp("2024-05-28")
}

forecasts = forecasts[
    forecasts["fold"].isin(
        evaluation_folds
    )
].copy()

# =========================================================
# 5. MERGE FORECAST WITH ADJUSTED DEMAND
# =========================================================
evaluation = forecasts.merge(
    adjusted,
    on=[
        "store_id",
        "product_id",
        "dt"
    ],
    how="inner"
)

expected_rows = (
    3
    *
    7
    *
    50000
)

print(
    "\nEvaluation rows:",
    len(evaluation)
)

print(
    "Expected rows:",
    expected_rows
)

assert (
    len(evaluation)
    ==
    expected_rows
)

# =========================================================
# 6. MERGE OBSERVED SALES EXPLICITLY
# =========================================================
evaluation = evaluation.merge(
    raw[
        [
            "store_id",
            "product_id",
            "dt",
            "sale_amount"
        ]
    ],
    on=[
        "store_id",
        "product_id",
        "dt"
    ],
    how="inner"
)

print(
    "Evaluation rows after observed-sales merge:",
    len(evaluation)
)

assert (
    len(evaluation)
    ==
    expected_rows
)

# =========================================================
# 7. BUILD 7-DAY DECISION UNIT
# =========================================================
seven_day = (
    evaluation
    .groupby(
        [
            "fold",
            "forecast_origin",
            "store_id",
            "product_id"
        ],
        as_index=False
    )
    .agg(
        forecast_7day=(
            "forecast_sales",
            "sum"
        ),
        adjusted_7day_demand=(
            "adjusted_demand",
            "sum"
        ),
        observed_7day_sales=(
            "sale_amount",
            "sum"
        )
    )
)

print(
    "\n7-day decision rows:",
    len(seven_day)
)

assert (
    len(seven_day)
    ==
    3
    *
    50000
)

# =========================================================
# 8. PRE-FORECAST STOCKOUT EXPOSURE
#
# Use ONLY the seven completed days immediately before the
# forecast origin.
#
# No future stockout information enters the risk band.
# =========================================================
risk_frames = []

for fold, origin in fold_origins.items():

    window_start = (
        origin
        -
        pd.Timedelta(
            days=6
        )
    )

    window_end = origin

    historical = raw[
        raw["dt"].between(
            window_start,
            window_end
        )
    ].copy()

    summary = (
        historical
        .groupby(
            group_cols,
            as_index=False
        )
        .agg(
            pre7_stockout_hours=(
                "stock_hour6_22_cnt",
                "mean"
            ),
            pre7_total_stockout_hours=(
                "stock_hour6_22_cnt",
                "sum"
            ),
            pre7_stockout_days=(
                "stock_hour6_22_cnt",
                lambda x:
                    int(
                        (
                            x > 0
                        ).sum()
                    )
            ),
            pre7_full_stockout_days=(
                "stock_hour6_22_cnt",
                lambda x:
                    int(
                        (
                            x == 16
                        ).sum()
                    )
            )
        )
    )

    summary["fold"] = fold
    summary["forecast_origin"] = origin

    risk_frames.append(
        summary
    )

pre_forecast_risk = pd.concat(
    risk_frames,
    ignore_index=True
)

print("\n" + "=" * 110)
print("PRE-FORECAST STOCKOUT EXPOSURE")
print("=" * 110)

print(
    "Risk rows:",
    len(pre_forecast_risk)
)

assert (
    len(pre_forecast_risk)
    ==
    3
    *
    50000
)

# =========================================================
# 9. MERGE RISK EXPOSURE
# =========================================================
seven_day = seven_day.merge(
    pre_forecast_risk,
    on=[
        "fold",
        "forecast_origin",
        "store_id",
        "product_id"
    ],
    how="inner"
)

print(
    "Decision rows after risk merge:",
    len(seven_day)
)

assert (
    len(seven_day)
    ==
    3
    *
    50000
)

# =========================================================
# 10. RISK BAND CUTPOINTS
#
# Cutpoints are calculated over the evaluation decision
# population. They are used only to define descriptive
# risk bands, NOT to tune the final holdout.
# =========================================================
risk_q25 = (
    seven_day[
        "pre7_stockout_hours"
    ]
    .quantile(
        0.25
    )
)

risk_q50 = (
    seven_day[
        "pre7_stockout_hours"
    ]
    .quantile(
        0.50
    )
)

risk_q75 = (
    seven_day[
        "pre7_stockout_hours"
    ]
    .quantile(
        0.75
    )
)

print("\n" + "=" * 110)
print("PRE-FORECAST STOCKOUT-RISK CUTPOINTS")
print("=" * 110)

print(
    "Q25:",
    round(
        risk_q25,
        4
    )
)

print(
    "Q50:",
    round(
        risk_q50,
        4
    )
)

print(
    "Q75:",
    round(
        risk_q75,
        4
    )
)

# =========================================================
# 11. ASSIGN RISK BAND
# =========================================================
def assign_risk_band(
    value
):

    if value <= risk_q25:
        return "LOW"

    if value <= risk_q50:
        return "MEDIUM"

    if value <= risk_q75:
        return "HIGH"

    return "VERY_HIGH"


seven_day[
    "risk_band"
] = (
    seven_day[
        "pre7_stockout_hours"
    ]
    .apply(
        assign_risk_band
    )
)

print("\n" + "=" * 110)
print("RISK-BAND DISTRIBUTION")
print("=" * 110)

print(
    seven_day[
        "risk_band"
    ]
    .value_counts(
        sort=False
    )
    .to_string()
)

# =========================================================
# 12. WALK-FORWARD SAFETY-STOCK CALIBRATION
#
# Fold 4 uses Fold 3 errors.
# Fold 5 uses Folds 3 + 4 errors.
#
# Fold 3 has no earlier adjusted-demand fold, so it is
# excluded from the policy-impact evaluation.
# =========================================================
service_levels = {
    "SL80": 0.80,
    "SL90": 0.90,
    "SL95": 0.95
}

calibration_frames = []

for fold in [
    4,
    5
]:

    current = seven_day[
        seven_day["fold"] == fold
    ].copy()

    previous = seven_day[
        seven_day["fold"] < fold
    ].copy()

    previous[
        "adjusted_forecast_error"
    ] = (
        previous[
            "adjusted_7day_demand"
        ]
        -
        previous[
            "forecast_7day"
        ]
    )

    previous[
        "positive_adjusted_error"
    ] = (
        previous[
            "adjusted_forecast_error"
        ]
        .clip(
            lower=0
        )
    )

    print(
        f"\nFold {fold} calibration folds:",
        sorted(
            previous[
                "fold"
            ].unique()
        )
    )

    for policy, service_level in service_levels.items():

        safety_stock = service_level_safety_stock(
            previous[
                "positive_adjusted_error"
            ],
            service_level,
        )

        temp = current.copy()

        temp[
            "service_policy"
        ] = policy

        temp[
            "safety_stock"
        ] = safety_stock

        calibration_frames.append(
            temp
        )

        print(
            f"{policy} safety stock:",
            round(
                safety_stock,
                4
            )
        )

calibrated = pd.concat(
    calibration_frames,
    ignore_index=True
)

print(
    "\nCalibrated rows:",
    len(calibrated)
)

assert (
    len(calibrated)
    ==
    2
    *
    3
    *
    50000
)

# =========================================================
# 13. RISK MULTIPLIER SCENARIOS
# =========================================================
multipliers = [
    0.00,
    0.10,
    0.20,
    0.30
]

policy_frames = []

for multiplier in multipliers:

    temp = calibrated.copy()

    temp[
        "risk_multiplier"
    ] = multiplier

    # -----------------------------------------------------
    # Base inventory target
    # -----------------------------------------------------
    temp[
        "base_inventory_target"
    ] = (
        temp[
            "forecast_7day"
        ]
        +
        temp[
            "safety_stock"
        ]
    ).clip(
        lower=0
    )

    # -----------------------------------------------------
    # Stockout-risk-aware inventory target
    #
    # Historical exposure only.
    # -----------------------------------------------------
    temp[
        "inventory_target"
    ] = risk_adjusted_inventory_target(
        temp[
            "forecast_7day"
        ].to_numpy(),
        temp[
            "safety_stock"
        ].to_numpy(),
        (
            temp[
                "pre7_stockout_hours"
            ]
            /
            16.0
        ).to_numpy(),
        multiplier,
    )

    # -----------------------------------------------------
    # Shortage against adjusted-demand proxy
    # -----------------------------------------------------
    temp[
        "shortage"
    ] = (
        temp[
            "adjusted_7day_demand"
        ]
        -
        temp[
            "inventory_target"
        ]
    ).clip(
        lower=0
    )

    # -----------------------------------------------------
    # Excess inventory proxy
    # -----------------------------------------------------
    temp[
        "excess"
    ] = (
        temp[
            "inventory_target"
        ]
        -
        temp[
            "adjusted_7day_demand"
        ]
    ).clip(
        lower=0
    )

    # -----------------------------------------------------
    # Fill rate
    # -----------------------------------------------------
    temp[
        "fill_rate"
    ] = (
        1
        -
        temp[
            "shortage"
        ]
        /
        temp[
            "adjusted_7day_demand"
        ].replace(
            0,
            np.nan
        )
    ).clip(
        lower=0,
        upper=1
    )

    temp[
        "demand_coverage"
    ] = (
        temp[
            "inventory_target"
        ]
        /
        temp[
            "adjusted_7day_demand"
        ].replace(
            0,
            np.nan
        )
    )

    policy_frames.append(
        temp
    )

policies = pd.concat(
    policy_frames,
    ignore_index=True
)

# =========================================================
# 14. RISK-BAND SUMMARY
# =========================================================
band_summary = (
    policies
    .groupby(
        [
            "risk_band",
            "service_policy",
            "risk_multiplier"
        ],
        as_index=False
    )
    .agg(
        observations=(
            "adjusted_7day_demand",
            "size"
        ),
        mean_pre7_stockout_hours=(
            "pre7_stockout_hours",
            "mean"
        ),
        mean_forecast=(
            "forecast_7day",
            "mean"
        ),
        mean_adjusted_demand=(
            "adjusted_7day_demand",
            "mean"
        ),
        mean_safety_stock=(
            "safety_stock",
            "mean"
        ),
        mean_inventory_target=(
            "inventory_target",
            "mean"
        ),
        mean_shortage=(
            "shortage",
            "mean"
        ),
        mean_excess=(
            "excess",
            "mean"
        ),
        mean_fill_rate=(
            "fill_rate",
            "mean"
        ),
        mean_coverage=(
            "demand_coverage",
            "mean"
        )
    )
)

print("\n" + "=" * 110)
print("POLICY PERFORMANCE BY STOCKOUT-RISK BAND")
print("=" * 110)

print(
    band_summary
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 15. IMPACT VS ZERO UPLIFT
# =========================================================
band_impact = risk_band_impact(
    band_summary
)

print("\n" + "=" * 110)
print("RISK-BAND POLICY IMPACT")
print("=" * 110)

print(
    band_impact[
        [
            "risk_band",
            "service_policy",
            "risk_multiplier",
            "inventory_increase_pct",
            "shortage_reduction_pct",
            "fill_rate_improvement_pp"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 16. BASELINE RISK-BAND CHARACTERISTICS
# =========================================================
baseline_risk = (
    seven_day
    .groupby(
        "risk_band",
        as_index=False
    )
    .agg(
        observations=(
            "adjusted_7day_demand",
            "size"
        ),
        mean_pre7_stockout_hours=(
            "pre7_stockout_hours",
            "mean"
        ),
        mean_stockout_days=(
            "pre7_stockout_days",
            "mean"
        ),
        mean_adjusted_demand=(
            "adjusted_7day_demand",
            "mean"
        ),
        mean_observed_sales=(
            "observed_7day_sales",
            "mean"
        )
    )
)

print("\n" + "=" * 110)
print("RISK-BAND BUSINESS PROFILE")
print("=" * 110)

print(
    baseline_risk
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 17. BENEFIT MONOTONICITY
#
# Does the same risk-aware policy have greater shortage
# reduction as historical stockout risk rises?
# =========================================================
print("\n" + "=" * 110)
print("RISK-BAND BENEFIT MONOTONICITY")
print("=" * 110)

band_order = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "VERY_HIGH": 4
}

monotonicity_rows = []

for policy in service_levels:

    for multiplier in [
        0.10,
        0.20,
        0.30
    ]:

        subset = band_impact[
            (
                band_impact[
                    "service_policy"
                ]
                ==
                policy
            )
            &
            (
                band_impact[
                    "risk_multiplier"
                ]
                ==
                multiplier
            )
        ].copy()

        subset[
            "band_order"
        ] = (
            subset[
                "risk_band"
            ]
            .map(
                band_order
            )
        )

        subset = subset.dropna(
            subset=[
                "shortage_reduction_pct"
            ]
        )

        if len(subset) >= 2:

            rho = (
                subset[
                    "band_order"
                ]
                .corr(
                    subset[
                        "shortage_reduction_pct"
                    ],
                    method="spearman"
                )
            )

        else:

            rho = np.nan

        monotonicity_rows.append(
            {
                "service_policy": policy,
                "risk_multiplier": multiplier,
                "spearman_band_benefit":
                    rho
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
# 18. FOLD × RISK-BAND STABILITY
# =========================================================
fold_band = (
    policies
    .groupby(
        [
            "fold",
            "risk_band",
            "service_policy",
            "risk_multiplier"
        ],
        as_index=False
    )
    .agg(
        mean_inventory=(
            "inventory_target",
            "mean"
        ),
        mean_shortage=(
            "shortage",
            "mean"
        ),
        mean_excess=(
            "excess",
            "mean"
        ),
        mean_fill_rate=(
            "fill_rate",
            "mean"
        )
    )
)

print("\n" + "=" * 110)
print("FOLD × RISK-BAND STABILITY")
print("=" * 110)

print(
    fold_band
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 19. SL90 FOCUS
# =========================================================
sl90 = band_impact[
    band_impact[
        "service_policy"
    ]
    ==
    "SL90"
].copy()

print("\n" + "=" * 110)
print("SL90 STOCKOUT-RISK POLICY COMPARISON")
print("=" * 110)

print(
    sl90[
        [
            "risk_band",
            "risk_multiplier",
            "inventory_increase_pct",
            "shortage_reduction_pct",
            "fill_rate_improvement_pp"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# =========================================================
# 20. HIGH-RISK VS LOW-RISK BENEFIT
# =========================================================
print("\n" + "=" * 110)
print("HIGH-RISK VS LOW-RISK POLICY BENEFIT")
print("=" * 110)

for multiplier in [
    0.10,
    0.20,
    0.30
]:

    subset = sl90[
        sl90[
            "risk_multiplier"
        ]
        ==
        multiplier
    ]

    low = subset[
        subset[
            "risk_band"
        ]
        ==
        "LOW"
    ]

    very_high = subset[
        subset[
            "risk_band"
        ]
        ==
        "VERY_HIGH"
    ]

    if (
        len(low)
        ==
        1
        and
        len(very_high)
        ==
        1
    ):

        low_benefit = (
            low[
                "shortage_reduction_pct"
            ].iloc[0]
        )

        very_high_benefit = (
            very_high[
                "shortage_reduction_pct"
            ].iloc[0]
        )

        print(
            f"Multiplier {multiplier:.2f}: "
            f"LOW={low_benefit:.2f}% "
            f"VERY_HIGH={very_high_benefit:.2f}%"
        )

# =========================================================
# 21. HOLDOUT PROTECTION
# =========================================================
print("\n" + "=" * 110)
print("HOLDOUT PROTECTION")
print("=" * 110)

print(
    "Official evaluation:",
    "2024-06-26 -> 2024-07-02"
)

print(
    "Final evaluation used:",
    False
)

# =========================================================
# 22. LEAKAGE CHECK
# =========================================================
print("\n" + "=" * 110)
print("LEAKAGE CHECK")
print("=" * 110)

print(
    "Risk bands based only on pre-forecast stockout history:",
    True
)

print(
    "Future stockout used to assign risk:",
    False
)

print(
    "Future adjusted demand used to assign risk:",
    False
)

print(
    "Fold k safety stock uses earlier folds only:",
    True
)

print(
    "Current fold used to calibrate itself:",
    False
)

print(
    "Final holdout used:",
    False
)

# =========================================================
# 23. SANITY CHECKS
# =========================================================
assert (
    len(evaluation)
    ==
    1050000
)

assert (
    len(seven_day)
    ==
    150000
)

assert (
    len(pre_forecast_risk)
    ==
    150000
)

assert (
    len(calibrated)
    ==
    300000
)

assert (
    len(policies)
    ==
    1200000
)

assert (
    set(
        seven_day[
            "risk_band"
        ].unique()
    )
    ==
    {
        "LOW",
        "MEDIUM",
        "HIGH",
        "VERY_HIGH"
    }
)

assert (
    policies[
        "inventory_target"
    ]
    .ge(0)
    .all()
)

assert (
    policies[
        "shortage"
    ]
    .ge(0)
    .all()
)

assert (
    policies[
        "excess"
    ]
    .ge(0)
    .all()
)

assert (
    policies[
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
    band_summary[
        "observations"
    ]
    .gt(0)
    .all()
)

print(
    "\nPolicy-by-risk stress-test checks: PASS"
)

# =========================================================
# 24. SAVE
# =========================================================
detail_output = (
    PROCESSED
    /
    "policy_risk_band_detail.csv"
)

summary_output = (
    PROCESSED
    /
    "policy_risk_band_summary.csv"
)

impact_output = (
    PROCESSED
    /
    "policy_risk_band_impact.csv"
)

fold_output = (
    PROCESSED
    /
    "policy_risk_band_fold_results.csv"
)

monotonicity_output = (
    PROCESSED
    /
    "policy_risk_band_monotonicity.csv"
)

baseline_output = (
    PROCESSED
    /
    "policy_risk_band_business_profile.csv"
)

policies.to_csv(
    detail_output,
    index=False
)

band_summary.to_csv(
    summary_output,
    index=False
)

band_impact.to_csv(
    impact_output,
    index=False
)

fold_band.to_csv(
    fold_output,
    index=False
)

monotonicity.to_csv(
    monotonicity_output,
    index=False
)

baseline_risk.to_csv(
    baseline_output,
    index=False
)

print("\n" + "=" * 110)
print("OUTPUT")
print("=" * 110)

print(
    "Detailed policy results:",
    detail_output
)

print(
    "Risk-band summary:",
    summary_output
)

print(
    "Risk-band impact:",
    impact_output
)

print(
    "Fold × risk-band:",
    fold_output
)

print(
    "Benefit monotonicity:",
    monotonicity_output
)

print(
    "Business profile:",
    baseline_output
)

print("\n" + "=" * 110)
print("PHASE 3 STEP 5B COMPLETE")
print("=" * 110)