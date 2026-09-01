from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GROUP_COLS = [
    "store_id",
    "product_id",
]

DATE_COL = "dt"
TARGET_COL = "sale_amount"
STOCKOUT_HOURS_COL = "stock_hour6_22_cnt"

FINAL_TRAIN_END = "2024-06-25"
FINAL_HOLDOUT_START = "2024-06-26"
FINAL_HOLDOUT_END = "2024-07-02"

FORECAST_HORIZONS = range(1, 8)

RANDOM_STATE = 42