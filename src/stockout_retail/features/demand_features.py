import pandas as pd

from stockout_retail.config import (
    GROUP_COLS,
    DATE_COL,
    TARGET_COL,
    STOCKOUT_HOURS_COL,
)
from stockout_retail.data.validation import require_columns


CANONICAL_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "rolling_7_mean",
    "rolling_14_mean",
    "rolling_7_std",
    "lag_1_stockout_hours",
    "lag_7_stockout_hours",
    "rolling_7_stockout_hours",
    "lag_1_discount",
    "lag_7_discount",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "is_weekend",
    "holiday_flag",
    "activity_flag",
]


REQUIRED_COLUMNS = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "activity_flag",
]


def create_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    require_columns(
        data,
        REQUIRED_COLUMNS,
    )

    data = data.sort_values(
        GROUP_COLS + [DATE_COL]
    ).copy()

    group = data.groupby(
        GROUP_COLS,
        sort=False,
    )

    data["lag_1"] = (
        group[TARGET_COL]
        .shift(1)
    )

    data["lag_2"] = (
        group[TARGET_COL]
        .shift(2)
    )

    data["lag_7"] = (
        group[TARGET_COL]
        .shift(7)
    )

    data["lag_14"] = (
        group[TARGET_COL]
        .shift(14)
    )

    data["rolling_7_mean"] = (
        data.groupby(GROUP_COLS)[TARGET_COL]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(
                    7,
                    min_periods=7,
                )
                .mean()
        )
    )

    data["rolling_14_mean"] = (
        data.groupby(GROUP_COLS)[TARGET_COL]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(
                    14,
                    min_periods=14,
                )
                .mean()
        )
    )

    data["rolling_7_std"] = (
        data.groupby(GROUP_COLS)[TARGET_COL]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(
                    7,
                    min_periods=7,
                )
                .std()
        )
    )

    data["lag_1_stockout_hours"] = (
        group[STOCKOUT_HOURS_COL]
        .shift(1)
    )

    data["lag_7_stockout_hours"] = (
        group[STOCKOUT_HOURS_COL]
        .shift(7)
    )

    data["rolling_7_stockout_hours"] = (
        data.groupby(GROUP_COLS)[STOCKOUT_HOURS_COL]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(
                    7,
                    min_periods=7,
                )
                .mean()
        )
    )

    data["lag_1_discount"] = (
        group["discount"]
        .shift(1)
    )

    data["lag_7_discount"] = (
        group["discount"]
        .shift(7)
    )

    data["day_of_week"] = (
        data[DATE_COL]
        .dt
        .dayofweek
    )

    data["day_of_month"] = (
        data[DATE_COL]
        .dt
        .day
    )

    data["week_of_year"] = (
        data[DATE_COL]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["holiday_flag"] = (
        data["holiday_flag"]
        .astype(int)
    )

    data["activity_flag"] = (
        data["activity_flag"]
        .astype(int)
    )

    return data