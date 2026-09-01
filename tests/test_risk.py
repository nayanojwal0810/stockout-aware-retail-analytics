import math

import pandas as pd

from stockout_retail.risk.prioritization import (
    add_risk_scores,
    assign_action_segments,
    rank_series,
)


def make_data(n=100):
    return pd.DataFrame(
        {
            "store_id": range(n),
            "product_id": [1] * n,
            "recent_mean_adjusted_demand": [
                float(i + 1)
                for i in range(n)
            ],
            "recent_estimated_censored_demand": [
                float(i + 1)
                for i in range(n)
            ],
            "mean_underforecast": [
                float(i + 1)
                for i in range(n)
            ],
        }
    )


def test_risk_score():
    data = add_risk_scores(make_data(10))

    assert "operational_risk_score" in data.columns
    assert data["operational_risk_score"].between(
        0,
        1,
    ).all()

    assert math.isclose(
        data["operational_risk_score"].min(),
        0.1,
    )
    assert math.isclose(
        data["operational_risk_score"].max(),
        1.0,
    )


def test_action_segments():
    data = add_risk_scores(make_data(100))
    result = assign_action_segments(data)

    assert (
        result["action_segment"]
        .value_counts()["CRITICAL"]
        == 1
    )

    assert (
        result["action_segment"]
        .value_counts()["HIGH_PRIORITY"]
        == 4
    )

    assert (
        result["action_segment"]
        .value_counts()["MONITOR"]
        == 10
    )

    assert (
        result["action_segment"]
        .value_counts()["STANDARD"]
        == 85
    )


def test_ranking_is_descending():
    result = rank_series(
        make_data(100)
    )

    assert result.iloc[0][
        "operational_risk_score"
    ] >= result.iloc[-1][
        "operational_risk_score"
    ]

    assert result.iloc[0][
        "operational_rank"
    ] == 1
