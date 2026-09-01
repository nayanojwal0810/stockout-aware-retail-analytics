"""Operational risk prioritization."""

import numpy as np
import pandas as pd


def add_risk_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentile-based operational risk."""
    required = [
        "recent_mean_adjusted_demand",
        "recent_estimated_censored_demand",
        "mean_underforecast",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = data.copy()

    data["demand_percentile"] = (
        data["recent_mean_adjusted_demand"]
        .rank(
            pct=True,
            method="average",
        )
    )

    data["stockout_burden_percentile"] = (
        data["recent_estimated_censored_demand"]
        .rank(
            pct=True,
            method="average",
        )
    )

    data["uncertainty_percentile"] = (
        data["mean_underforecast"]
        .rank(
            pct=True,
            method="average",
        )
    )

    data["operational_risk_score"] = (
        0.45
        * data["stockout_burden_percentile"]
        + 0.35
        * data["demand_percentile"]
        + 0.20
        * data["uncertainty_percentile"]
    )

    return data


def assign_action_segments(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Assign CRITICAL, HIGH_PRIORITY, MONITOR and STANDARD segments."""
    if "operational_risk_score" not in data.columns:
        raise ValueError(
            "operational_risk_score is required."
        )

    data = data.sort_values(
        [
            "operational_risk_score",
            "recent_mean_adjusted_demand",
        ],
        ascending=False,
    ).reset_index(drop=True)

    n = len(data)

    n_critical = max(
        1,
        int(n * 0.01),
    )

    n_high = max(
        1,
        int(n * 0.05),
    )

    n_monitor = max(
        1,
        int(n * 0.15),
    )

    data["operational_rank"] = np.arange(
        1,
        n + 1,
    )

    data["action_segment"] = "STANDARD"

    data.loc[
        data["operational_rank"] <= n_monitor,
        "action_segment",
    ] = "MONITOR"

    data.loc[
        data["operational_rank"] <= n_high,
        "action_segment",
    ] = "HIGH_PRIORITY"

    data.loc[
        data["operational_rank"] <= n_critical,
        "action_segment",
    ] = "CRITICAL"

    return data


def rank_series(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate risk score and final operational ranking."""
    result = add_risk_scores(data)
    return assign_action_segments(result)