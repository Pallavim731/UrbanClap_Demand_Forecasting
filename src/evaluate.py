import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate Mean Absolute Percentage Error."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    non_zero = actual != 0

    if not np.any(non_zero):
        return 0.0

    return float(
        np.mean(
            np.abs(
                (actual[non_zero] - predicted[non_zero])
                / actual[non_zero]
            )
        )
        * 100
    )


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate forecasting evaluation metrics."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mape = calculate_mape(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    return {
        "MAPE": float(mape),
        "MAE": float(mae),
        "RMSE": float(rmse),
    }