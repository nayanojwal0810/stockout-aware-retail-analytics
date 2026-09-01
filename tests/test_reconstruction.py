import math

import pandas as pd

from stockout_retail.reconstruction.demand import (
    RECONSTRUCTION_FEATURES,
    add_stockout_state,
    build_adjusted_demand,
    create_reconstruction_features,
)


def make_data():
    dates = pd.date_range(
        "2024-01-01",
        periods=20,
        freq="D",
    )

    return pd.DataFrame(
        {
            "store_id": [1] * 20,
            "product_id": [10] * 20,
            "dt": dates,
            "sale_amount": [
                float(i + 1)
                for i in range(20)
            ],
            "stock_hour6_22_cnt": (
                [0] * 15
                + [4, 16, 0, 0, 0]
            ),
            "discount": [0.1] * 20,
            "holiday_flag": [0] * 20,
            "activity_flag": [1] * 20,
        }
    )


def test_feature_contract():
    data = create_reconstruction_features(
        make_data()
    )

    assert len(RECONSTRUCTION_FEATURES) == 13
    assert all(
        c in data.columns
        for c in RECONSTRUCTION_FEATURES
    )


def test_features_use_only_past_values():
    data = create_reconstruction_features(
        make_data()
    )
    row = data.sort_values("dt").iloc[10]

    assert row["lag_1_sales"] == 10.0
    assert row["lag_7_sales"] == 4.0
    assert math.isclose(
        row["rolling_7_sales"],
        7.0,
    )
    assert pd.isna(row["lag_14_sales"]) if "lag_14_sales" in row else True


def test_stockout_state():
    data = add_stockout_state(
        make_data()
    )

    assert data.loc[
        0, "stockout_flag"
    ] == 0

    assert data.loc[
        15, "stockout_flag"
    ] == 1

    assert data.loc[
        16, "full_stockout_flag"
    ] == 1

    assert data.loc[
        15, "stockout_state"
    ] == "PARTIAL_STOCKOUT"

    assert data.loc[
        16, "stockout_state"
    ] == "FULL_STOCKOUT"


def test_adjusted_demand():
    data = add_stockout_state(
        make_data()
    )
    data[
        "cross_fitted_demand_prediction"
    ] = data["sale_amount"] + 1.0

    out = build_adjusted_demand(data)

    assert out.loc[
        0, "adjusted_demand"
    ] == 1.0

    assert out.loc[
        15, "adjusted_demand"
    ] == 17.0

    assert out.loc[
        16, "adjusted_demand"
    ] == 18.0

    assert out.loc[
        16, "estimated_censored_gap"
    ] == 1.0