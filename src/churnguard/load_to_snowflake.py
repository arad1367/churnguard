"""ELT pipeline: load raw CSV into Snowflake RAW, build clean view in ANALYTICS.

Idempotent: running it again gives the same end state (no duplicates).
"""
from churnguard.data import PROJECT_ROOT, RAW_DATA_PATH
from churnguard.snowflake_conn import get_connection

CREATE_RAW_TABLE = """
CREATE OR REPLACE TABLE RAW.TELCO_CUSTOMERS (
    customerID VARCHAR, gender VARCHAR, SeniorCitizen INTEGER,
    Partner VARCHAR, Dependents VARCHAR, tenure INTEGER,
    PhoneService VARCHAR, MultipleLines VARCHAR, InternetService VARCHAR,
    OnlineSecurity VARCHAR, OnlineBackup VARCHAR, DeviceProtection VARCHAR,
    TechSupport VARCHAR, StreamingTV VARCHAR, StreamingMovies VARCHAR,
    Contract VARCHAR, PaperlessBilling VARCHAR, PaymentMethod VARCHAR,
    MonthlyCharges FLOAT, TotalCharges VARCHAR, Churn VARCHAR
)
"""

CREATE_FILE_FORMAT = """
CREATE OR REPLACE FILE FORMAT RAW.CSV_FMT
    TYPE = CSV
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
"""

CREATE_STAGE = "CREATE OR REPLACE STAGE RAW.CHURN_STAGE FILE_FORMAT = RAW.CSV_FMT"


def main() -> None:
    csv_path = RAW_DATA_PATH.as_posix()  # forward slashes, required by PUT
    clean_sql = (PROJECT_ROOT / "sql" / "customers_clean.sql").read_text()

    with get_connection(schema="RAW") as conn:
        cur = conn.cursor()

        print("1/5 Creating raw table...")
        cur.execute(CREATE_RAW_TABLE)

        print("2/5 Creating file format and stage...")
        cur.execute(CREATE_FILE_FORMAT)
        cur.execute(CREATE_STAGE)

        print("3/5 Uploading CSV to stage...")
        cur.execute(f"PUT 'file://{csv_path}' @RAW.CHURN_STAGE OVERWRITE = TRUE")

        print("4/5 Copying into table...")
        cur.execute(
            "COPY INTO RAW.TELCO_CUSTOMERS FROM @RAW.CHURN_STAGE "
            "ON_ERROR = 'ABORT_STATEMENT'"
        )
        rows = cur.execute("SELECT COUNT(*) FROM RAW.TELCO_CUSTOMERS").fetchone()[0]
        print(f"      Loaded {rows} rows")

        print("5/5 Building clean view...")
        view_sql = clean_sql.format(source="RAW.TELCO_CUSTOMERS")
        cur.execute(f"CREATE OR REPLACE VIEW ANALYTICS.CUSTOMERS_CLEAN AS {view_sql}")

    print("Done.")


if __name__ == "__main__":
    main()