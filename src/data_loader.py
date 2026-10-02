from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")


def load_booking_data(filename: str) -> pd.DataFrame:
    """Load a booking CSV file from the raw data directory."""
    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path)