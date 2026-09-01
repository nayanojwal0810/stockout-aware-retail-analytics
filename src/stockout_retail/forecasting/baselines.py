"""Simple forecasting baselines."""

import pandas as pd


def naive_1(series: pd.Series) -> pd.Series:
    """Use the previous day as the forecast."""
    return series.shift(1).clip(lower=0)


def seasonal_naive_7(series: pd.Series) -> pd.Series:
    """Use the value from seven days earlier."""
    return series.shift(7).clip(lower=0)


def moving_average_7(series: pd.Series) -> pd.Series:
    """Use the mean of the previous seven days."""
    return (
        series.shift(1)
        .rolling(7, min_periods=7)
        .mean()
        .clip(lower=0)
    )
