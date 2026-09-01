"""Validation helpers for demand reconstruction."""

import pandas as pd

from stockout_retail.config import DATE_COL, TARGET_COL


def reconstruction_coverage(
    oof: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> dict:
    """Return basic prediction coverage information."""
    return {
        "oof_start": oof[DATE_COL].min(),
        "oof_end": oof[DATE_COL].max(),
        "adjusted_start": adjusted[DATE_COL].min(),
        "adjusted_end": adjusted[DATE_COL].max(),
        "oof_missing": int(
            oof["cross_fitted_demand"].isna().sum()
        ),
        "adjusted_missing": int(
            adjusted[
                "cross_fitted_demand_prediction"
            ].isna().sum()
        ),
    }


def stockout_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize adjusted demand by stockout state."""
    required = [
        TARGET_COL,
        "adjusted_demand",
        "estimated_censored_gap",
        "stockout_state",
    ]

    missing = [
        c for c in required
        if c not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return (
        data.groupby("stockout_state")
        .agg(
            observations=(TARGET_COL, "size"),
            observed_sales=(TARGET_COL, "mean"),
            adjusted_demand=("adjusted_demand", "mean"),
            estimated_gap=(
                "estimated_censored_gap",
                "mean",
            ),
        )
        .reset_index()
    )