"""Inventory policy calculations."""

import numpy as np


def service_level_safety_stock(
    positive_errors,
    service_level: float,
) -> float:
    """Estimate safety stock from historical positive forecast errors."""
    errors = np.asarray(
        positive_errors,
        dtype=float,
    )

    if errors.size == 0:
        raise ValueError(
            "positive_errors cannot be empty"
        )

    if not 0 < service_level <= 1:
        raise ValueError(
            "service_level must be between 0 and 1"
        )

    return float(
        np.quantile(
            errors,
            service_level,
        )
    )


def inventory_target(
    forecast,
    safety_stock,
) -> np.ndarray:
    """Base inventory target."""
    return np.maximum(
        np.asarray(forecast, dtype=float)
        + np.asarray(safety_stock, dtype=float),
        0,
    )


def risk_adjusted_inventory_target(
    forecast,
    safety_stock,
    stockout_intensity,
    risk_multiplier: float,
) -> np.ndarray:
    """Increase inventory protection with historical stockout intensity."""
    base = inventory_target(
        forecast,
        safety_stock,
    )

    intensity = np.asarray(
        stockout_intensity,
        dtype=float,
    )

    if risk_multiplier < 0:
        raise ValueError(
            "risk_multiplier cannot be negative"
        )

    return np.maximum(
        base
        * (
            1
            + risk_multiplier
            * np.clip(
                intensity,
                0,
                1,
            )
        ),
        0,
    )


def shortage_units(
    demand,
    inventory,
) -> np.ndarray:
    """Demand that cannot be covered by inventory."""
    return np.maximum(
        np.asarray(demand, dtype=float)
        - np.asarray(inventory, dtype=float),
        0,
    )


def excess_units(
    demand,
    inventory,
) -> np.ndarray:
    """Inventory remaining after demand."""
    return np.maximum(
        np.asarray(inventory, dtype=float)
        - np.asarray(demand, dtype=float),
        0,
    )


def fill_rate(
    demand,
    inventory,
) -> np.ndarray:
    """Unit fill rate."""
    demand = np.asarray(
        demand,
        dtype=float,
    )
    shortage = shortage_units(
        demand,
        inventory,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        result = (
            1
            - shortage / demand
        )

    return np.clip(
        result,
        0,
        1,
    )


def demand_coverage(
    demand,
    inventory,
) -> np.ndarray:
    """Inventory divided by demand."""
    demand = np.asarray(
        demand,
        dtype=float,
    )
    inventory = np.asarray(
        inventory,
        dtype=float,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        return inventory / demand