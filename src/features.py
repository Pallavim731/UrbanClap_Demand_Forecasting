import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    target_column: str = "demand",
) -> pd.DataFrame:
    """Create historical demand features."""
    data = df.copy()

    data["lag_1"] = data[target_column].shift(1)
    data["lag_7"] = data[target_column].shift(7)
    data["lag_14"] = data[target_column].shift(14)
    data["lag_28"] = data[target_column].shift(28)

    return data


def add_rolling_features(
    df: pd.DataFrame,
    target_column: str = "demand",
) -> pd.DataFrame:
    """Create rolling demand statistics."""
    data = df.copy()

    data["rolling_mean_7"] = (
        data[target_column].rolling(window=7).mean()
    )

    data["rolling_mean_14"] = (
        data[target_column].rolling(window=14).mean()
    )

    data["rolling_mean_28"] = (
        data[target_column].rolling(window=28).mean()
    )

    return data