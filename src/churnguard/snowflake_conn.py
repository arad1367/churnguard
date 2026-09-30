"""Snowflake connection helpers (key-pair auth, config from .env)."""
import os

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

from churnguard.data import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def get_connection(schema: str = "RAW"):
    """Open a Snowflake connection as the service user."""
    key_path = PROJECT_ROOT / os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"]
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        authenticator="SNOWFLAKE_JWT",
        private_key_file=str(key_path),
        role=os.environ["SNOWFLAKE_ROLE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=schema,
    )


def read_sql(query: str) -> pd.DataFrame:
    """Run a query and return the result as a DataFrame."""
    with get_connection() as conn:
        return conn.cursor().execute(query).fetch_pandas_all()