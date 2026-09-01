"""Forecast evaluation metrics."""

import numpy as np


def mae(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def wape(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = np.abs(actual).sum()

    if denominator == 0:
        return np.nan

    return float(
        np.abs(actual - predicted).sum() / denominator
    )


def rmse(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def evaluate_forecast(actual, predicted) -> dict[str, float]:
    """Return the three project forecast metrics."""
    return {
        "MAE": mae(actual, predicted),
        "WAPE": wape(actual, predicted),
        "RMSE": rmse(actual, predicted),
    }
