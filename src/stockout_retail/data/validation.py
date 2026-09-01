import pandas as pd


def require_columns(
    data: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def validate_unique_series_date(
    data: pd.DataFrame,
    group_cols: list[str],
    date_col: str,
) -> None:
    duplicates = data.duplicated(
        subset=group_cols + [date_col]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates:,} duplicate series-date rows."
        )


def validate_non_negative(
    data: pd.DataFrame,
    columns: list[str],
) -> None:
    for column in columns:
        if column not in data.columns:
            continue

        if (data[column] < 0).any():
            raise ValueError(
                f"Negative values found in: {column}"
            )