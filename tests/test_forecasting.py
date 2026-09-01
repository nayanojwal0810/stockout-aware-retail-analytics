import math

import numpy as np
import pandas as pd

from stockout_retail.forecasting.baselines import (
    naive_1,
    seasonal_naive_7,
    moving_average_7,
)
from stockout_retail.forecasting.evaluation import (
    mae,
    wape,
    rmse,
    evaluate_forecast,
)
from stockout_retail.forecasting.model import build_model


def test_baselines():
    series = pd.Series(np.arange(1.0, 11.0))

    assert naive_1(series).iloc[1] == 1.0
    assert seasonal_naive_7(series).iloc[7] == 1.0
    assert math.isclose(
        moving_average_7(series).iloc[7],
        4.0,
    )


def test_metrics():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 1.0, 4.0])

    assert math.isclose(
        mae(actual, predicted),
        2 / 3,
    )
    assert math.isclose(
        wape(actual, predicted),
        2 / 6,
    )
    assert math.isclose(
        rmse(actual, predicted),
        math.sqrt(2 / 3),
    )


def test_model_configuration():
    model = build_model()

    assert model.learning_rate == 0.08
    assert model.max_iter == 120
    assert model.max_leaf_nodes == 31
    assert model.min_samples_leaf == 100
    assert model.l2_regularization == 1.0
    assert model.early_stopping is False


def test_evaluate_forecast():
    result = evaluate_forecast(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 4.0],
    )

    assert set(result) == {"MAE", "WAPE", "RMSE"}
    assert result["MAE"] > 0
    assert result["WAPE"] > 0
    assert result["RMSE"] > 0
