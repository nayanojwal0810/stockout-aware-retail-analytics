"""Inventory policy sensitivity calculations."""

import pandas as pd


def policy_impact(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each policy scenario with its zero-risk baseline."""
    required = [
        "service_policy",
        "risk_multiplier",
        "mean_inventory_target",
        "mean_shortage",
        "mean_fill_rate",
    ]

    missing = [
        c for c in required
        if c not in summary.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    baseline = (
        summary[
            summary["risk_multiplier"] == 0
        ][
            [
                "service_policy",
                "mean_inventory_target",
                "mean_shortage",
                "mean_fill_rate",
            ]
        ]
        .rename(
            columns={
                "mean_inventory_target":
                    "baseline_inventory",
                "mean_shortage":
                    "baseline_shortage",
                "mean_fill_rate":
                    "baseline_fill_rate",
            }
        )
    )

    result = summary.merge(
        baseline,
        on="service_policy",
        how="left",
    )

    result["inventory_increase_pct"] = (
        (
            result["mean_inventory_target"]
            - result["baseline_inventory"]
        )
        / result["baseline_inventory"]
        .replace(0, float("nan"))
        * 100
    )

    result["shortage_reduction_pct"] = (
        (
            result["baseline_shortage"]
            - result["mean_shortage"]
        )
        / result["baseline_shortage"]
        .replace(0, float("nan"))
        * 100
    )

    result["fill_rate_improvement_pp"] = (
        (
            result["mean_fill_rate"]
            - result["baseline_fill_rate"]
        )
        * 100
    )

    return result


def risk_band_impact(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate policy impact separately for each risk band."""
    required = [
        "risk_band",
        "service_policy",
        "risk_multiplier",
        "mean_inventory_target",
        "mean_shortage",
        "mean_fill_rate",
    ]

    missing = [
        c for c in required
        if c not in detail.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    baseline = (
        detail[
            detail["risk_multiplier"] == 0
        ][
            [
                "risk_band",
                "service_policy",
                "mean_inventory_target",
                "mean_shortage",
                "mean_fill_rate",
            ]
        ]
        .rename(
            columns={
                "mean_inventory_target":
                    "baseline_inventory",
                "mean_shortage":
                    "baseline_shortage",
                "mean_fill_rate":
                    "baseline_fill_rate",
            }
        )
    )

    result = detail.merge(
        baseline,
        on=[
            "risk_band",
            "service_policy",
        ],
        how="left",
    )

    result["inventory_increase_pct"] = (
        (
            result["mean_inventory_target"]
            - result["baseline_inventory"]
        )
        / result["baseline_inventory"]
        .replace(0, float("nan"))
        * 100
    )

    result["shortage_reduction_pct"] = (
        (
            result["baseline_shortage"]
            - result["mean_shortage"]
        )
        / result["baseline_shortage"]
        .replace(0, float("nan"))
        * 100
    )

    result["fill_rate_improvement_pp"] = (
        (
            result["mean_fill_rate"]
            - result["baseline_fill_rate"]
        )
        * 100
    )

    return result
