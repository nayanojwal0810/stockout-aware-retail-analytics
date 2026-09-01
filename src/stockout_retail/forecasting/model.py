"""Final forecasting model."""

from sklearn.ensemble import HistGradientBoostingRegressor

from stockout_retail.config import RANDOM_STATE


def build_model() -> HistGradientBoostingRegressor:
    """Build the model used for final holdout evaluation."""
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
        early_stopping=False,
    )