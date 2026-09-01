from pathlib import Path

import pandas as pd

from stockout_retail.config import RAW_DIR, DATE_COL


def load_raw_data(
    filename: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    data = pd.read_parquet(
        path,
        columns=columns,
    )

    if DATE_COL in data.columns:
        data[DATE_COL] = pd.to_datetime(
            data[DATE_COL]
        )

    return data