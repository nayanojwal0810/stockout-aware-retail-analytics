import math

import numpy as np
import pandas as pd

from stockout_retail.inventory.policy import (
    service_level_safety_stock,
    inventory_target,
    risk_adjusted_inventory_target,
    shortage_units,
    excess_units,
    fill_rate,
    demand_coverage,
)
from stockout_retail.inventory.sensitivity import (
    policy_impact,
)


def test_safety_stock():
    result = service_level_safety_stock(
        [0, 1, 2, 3, 4],
        0.8,
    )

    assert math.isclose(
        result,
        3.2,
    )


def test_inventory_target():
    result = inventory_target(
        [7, 8],
        2,
    )

    assert np.allclose(
        result,
        [9, 10],
    )


def test_risk_adjusted_inventory():
    result = risk_adjusted_inventory_target(
        [10],
        2,
        [0.5],
        0.2,
    )

    assert math.isclose(
        result[0],
        13.2,
    )


def test_shortage_excess_fill_rate():
    demand = [10, 5]
    inventory = [8, 7]

    assert np.allclose(
        shortage_units(demand, inventory),
        [2, 0],
    )
    assert np.allclose(
        excess_units(demand, inventory),
        [0, 2],
    )
    assert np.allclose(
        fill_rate(demand, inventory),
        [0.8, 1.0],
    )


def test_demand_coverage():
    result = demand_coverage(
        [10, 5],
        [8, 7],
    )

    assert np.allclose(
        result,
        [0.8, 1.4],
    )


def test_policy_impact():
    summary = pd.DataFrame(
        {
            "service_policy": [
                "SL90",
                "SL90",
            ],
            "risk_multiplier": [
                0.0,
                0.2,
            ],
            "mean_inventory_target": [
                10.0,
                10.5,
            ],
            "mean_shortage": [
                2.0,
                1.5,
            ],
            "mean_fill_rate": [
                0.8,
                0.85,
            ],
        }
    )

    result = policy_impact(summary)

    row = result[
        result["risk_multiplier"] == 0.2
    ].iloc[0]

    assert math.isclose(
        row["inventory_increase_pct"],
        5.0,
    )
    assert math.isclose(
        row["shortage_reduction_pct"],
        25.0,
    )
    assert math.isclose(
        row["fill_rate_improvement_pp"],
        5.0,
    )