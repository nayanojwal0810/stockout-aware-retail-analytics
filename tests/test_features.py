import pandas as pd

from stockout_retail.features.demand_features import (
    CANONICAL_FEATURES,
    create_features,
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
            "sale_amount": [float(i + 1) for i in range(20)],
            "stock_hour6_22_cnt": [0] * 20,
            "discount": [0.1] * 20,
            "holiday_flag": [0] * 20,
            "activity_flag": [1] * 20,
        }
    )


def test_feature_count():
    data = create_features(make_data())

    assert len(CANONICAL_FEATURES) == 18
    assert all(
        column in data.columns
        for column in CANONICAL_FEATURES
    )

def test_lags_use_previous_values():
    data = create_features(make_data())
    row = data.sort_values("dt").iloc[10]

    assert row["lag_1"] == 10.0
    assert row["lag_7"] == 4.0
    assert pd.isna(row["lag_14"])


def test_rolling_features_use_past_values():
    data = create_features(make_data())
    row = data.sort_values("dt").iloc[10]

    assert row["rolling_7_mean"] == 7.0
    assert pd.isna(row["rolling_14_mean"])


def test_feature_engineering_does_not_use_same_day_sales():
    data = make_data()
    original = create_features(data)

    changed = data.copy()
    changed.loc[
        changed["dt"] == pd.Timestamp("2024-01-11"),
        "sale_amount",
    ] = 9999.0

    changed_features = create_features(changed)

    current_row = (
        changed_features["dt"]
        == pd.Timestamp("2024-01-11")
    )

    assert original.loc[
        current_row,
        "lag_1",
    ].iloc[0] == changed_features.loc[
        current_row,
        "lag_1",
    ].iloc[0]

    assert original.loc[
        current_row,
        "rolling_7_mean",
    ].iloc[0] == changed_features.loc[
        current_row,
        "rolling_7_mean",
    ].iloc[0]