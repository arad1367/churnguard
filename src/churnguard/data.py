"""Data loading utilities for ChurnGuard."""
from pathlib import Path

import pandas as pd

# data.py lives in src/churnguard/, so the project root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn CSV exactly as downloaded (no cleaning)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Download it first (see Phase 1.3)."
        )
    return pd.read_csv(path)