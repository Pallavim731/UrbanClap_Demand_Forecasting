import pandas as pd


def clean_booking_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean basic booking data before feature engineering."""
    data = df.copy()

    data = data.drop_duplicates()
    data = data.dropna()

    return data


def add_datetime_features(
    df: pd.DataFrame, datetime_column: str = "datetime"
) -> pd.DataFrame:
    """Create basic calendar features from a datetime column."""
    data = df.copy()

    data[datetime_column] = pd.to_datetime(
        data[datetime_column],
        errors="coerce",
    )

    data = data.dropna(subset=[datetime_column])

    data["day_of_week"] = data[datetime_column].dt.dayofweek
    data["month"] = data[datetime_column].dt.month
    data["hour"] = data[datetime_column].dt.hour

    return data
