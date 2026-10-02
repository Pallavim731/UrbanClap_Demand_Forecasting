import numpy as np
import pandas as pd


def create_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "demand",
    input_window: int = 28,
    forecast_horizon: int = 7,
):
    """Create sequences for multi-step time-series forecasting."""
    features = df[feature_columns].to_numpy(dtype=np.float32)
    target = df[target_column].to_numpy(dtype=np.float32)

    X = []
    y = []

    max_start = len(df) - input_window - forecast_horizon + 1

    for start in range(max_start):
        end = start + input_window
        target_end = end + forecast_horizon

        X.append(features[start:end])
        y.append(target[end:target_end])

    return np.array(X), np.array(y)