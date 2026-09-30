"""Data loading utilities for ChurnGuard."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
CLEAN_SQL_PATH = PROJECT_ROOT / "sql" / "customers_clean.sql"

# Explicit raw schema: must match RAW.TELCO_CUSTOMERS in Snowflake.
# We never let tools guess types (DuckDB would read "Yes"/"No" as booleans).
RAW_SCHEMA = {
    "customerID": "VARCHAR", "gender": "VARCHAR", "SeniorCitizen": "INTEGER",
    "Partner": "VARCHAR", "Dependents": "VARCHAR", "tenure": "INTEGER",
    "PhoneService": "VARCHAR", "MultipleLines": "VARCHAR", "InternetService": "VARCHAR",
    "OnlineSecurity": "VARCHAR", "OnlineBackup": "VARCHAR", "DeviceProtection": "VARCHAR",
    "TechSupport": "VARCHAR", "StreamingTV": "VARCHAR", "StreamingMovies": "VARCHAR",
    "Contract": "VARCHAR", "PaperlessBilling": "VARCHAR", "PaymentMethod": "VARCHAR",
    "MonthlyCharges": "DOUBLE", "TotalCharges": "VARCHAR", "Churn": "VARCHAR",
}

load_dotenv(PROJECT_ROOT / ".env")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn CSV exactly as downloaded (no cleaning)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Download it first (see Phase 1.3)."
        )
    return pd.read_csv(path)


def load_clean_data(source: str | None = None) -> pd.DataFrame:
    """Load type-cleaned customer data from 'local' (DuckDB) or 'snowflake'.

    Both sources apply the SAME SQL transformation (sql/customers_clean.sql)
    and the SAME raw schema, so they return identical data.
    """
    source = source or os.getenv("DATA_SOURCE", "local")

    if source == "snowflake":
        from churnguard.snowflake_conn import read_sql  # imported only when needed
        df = read_sql("SELECT * FROM ANALYTICS.CUSTOMERS_CLEAN")
    elif source == "local":
        import duckdb
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Raw data not found at {RAW_DATA_PATH}. Download it first (see Phase 1.3)."
            )
        csv = RAW_DATA_PATH.as_posix()
        columns = ", ".join(f"'{name}': '{dtype}'" for name, dtype in RAW_SCHEMA.items())
        duck_source = f"read_csv('{csv}', header = true, columns = {{{columns}}})"
        query = CLEAN_SQL_PATH.read_text().format(source=duck_source)
        df = duckdb.sql(query).df()
    else:
        raise ValueError(f"Unknown DATA_SOURCE: {source!r} (use 'local' or 'snowflake')")

    df.columns = df.columns.str.lower()  # Snowflake returns UPPERCASE names
    return df