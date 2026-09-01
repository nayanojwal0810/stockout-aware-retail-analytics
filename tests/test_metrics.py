import math

from stockout_retail.forecasting.evaluation import (
    mae,
    rmse,
    wape,
)


def test_mae():
    assert math.isclose(
        mae([1, 2, 3], [1, 1, 4]),
        2 / 3,
    )


def test_wape():
    assert math.isclose(
        wape([1, 2, 3], [1, 1, 4]),
        2 / 6,
    )


def test_rmse():
    assert math.isclose(
        rmse([1, 2, 3], [1, 1, 4]),
        math.sqrt(2 / 3),
    )


def test_wape_zero_actuals():
    assert math.isnan(
        wape([0, 0], [0, 1])
    )
