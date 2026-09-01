"""Stockout-aware demand reconstruction."""

import numpy as np
import pandas as pd

from stockout_retail.config import (
    GROUP_COLS,
    DATE_COL,
    TARGET_COL,
    STOCKOUT_HOURS_COL,
)
from stockout_retail.data.validation import require_columns


RECONSTRUCTION_FEATURES = [
    "lag_1_sales",
    "lag_7_sales",
    "rolling_7_sales",
    "rolling_14_sales",
    "lag_1_stockout_hours",
    "lag_7_stockout_hours",
    "rolling_7_stockout_hours",
    "lag_1_discount",
    "lag_7_discount",
    "day_of_week",
    "is_weekend",
    "holiday_flag",
    "activity_flag",
]

RECONSTRUCTION_INPUTS = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "activity_flag",
]


def create_reconstruction_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the same 13 features used by the reconstruction workflow."""
    require_columns(data, RECONSTRUCTION_INPUTS)

    data = data.sort_values(
        GROUP_COLS + [DATE_COL]
    ).copy()

    group = data.groupby(
        GROUP_COLS,
        sort=False,
    )

    data["lag_1_sales"] = group[TARGET_COL].shift(1)
    data["lag_7_sales"] = group[TARGET_COL].shift(7)

    data["rolling_7_sales"] = (
        data.groupby(GROUP_COLS)[TARGET_COL]
        .transform(
            lambda s: s.shift(1).rolling(
                7,
                min_periods=3,
            ).mean()
        )
    )

    data["rolling_14_sales"] = (
        data.groupby(GROUP_COLS)[TARGET_COL]
        .transform(
            lambda s: s.shift(1).rolling(
                14,
                min_periods=7,
            ).mean()
        )
    )

    data["lag_1_stockout_hours"] = (
        group[STOCKOUT_HOURS_COL].shift(1)
    )

    data["lag_7_stockout_hours"] = (
        group[STOCKOUT_HOURS_COL].shift(7)
    )

    data["rolling_7_stockout_hours"] = (
        data.groupby(GROUP_COLS)[STOCKOUT_HOURS_COL]
        .transform(
            lambda s: s.shift(1).rolling(
                7,
                min_periods=3,
            ).mean()
        )
    )

    data["day_of_week"] = data[DATE_COL].dt.dayofweek
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["lag_1_discount"] = group["discount"].shift(1)
    data["lag_7_discount"] = group["discount"].shift(7)

    return data


def add_stockout_state(data: pd.DataFrame) -> pd.DataFrame:
    """Add the stockout flags used by the reconstruction workflow."""
    require_columns(
        data,
        [STOCKOUT_HOURS_COL],
    )

    data = data.copy()

    data["stockout_flag"] = (
        data[STOCKOUT_HOURS_COL] > 0
    ).astype(int)

    data["full_stockout_flag"] = (
        data[STOCKOUT_HOURS_COL] == 16
    ).astype(int)

    data["normal_day"] = (
        data[STOCKOUT_HOURS_COL] == 0
    )

    data["stockout_state"] = np.select(
        [
            data[STOCKOUT_HOURS_COL] == 0,
            data[STOCKOUT_HOURS_COL].between(1, 15),
            data[STOCKOUT_HOURS_COL] == 16,
        ],
        [
            "NORMAL",
            "PARTIAL_STOCKOUT",
            "FULL_STOCKOUT",
        ],
        default="UNKNOWN",
    )

    return data


def build_adjusted_demand(
    data: pd.DataFrame,
    prediction_column: str = "cross_fitted_demand_prediction",
) -> pd.DataFrame:
    """Build adjusted demand from observed sales and OOF predictions."""
    require_columns(
        data,
        [
            TARGET_COL,
            "stockout_flag",
            prediction_column,
        ],
    )

    data = data.copy()

    data["adjusted_demand"] = np.where(
        data["stockout_flag"] == 1,
        data[prediction_column],
        data[TARGET_COL],
    )

    data["estimated_censored_gap"] = (
        data["adjusted_demand"]
        - data[TARGET_COL]
    ).clip(lower=0)

    return data