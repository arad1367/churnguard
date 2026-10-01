"""Create a reproducible, stratified train/test split and save it to disk.

Run once. Every later phase reads these files, so all experiments
use exactly the same split. The test set stays untouched until final evaluation.
"""
from sklearn.model_selection import train_test_split

from churnguard.data import PROJECT_ROOT, load_clean_data

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
TARGET = "churned"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    df = load_clean_data()

    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],        # keep the same churn rate in both sets
        random_state=RANDOM_STATE,  # same split every time
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)

    print(f"Train: {len(train)} rows, churn rate {train[TARGET].mean():.3f}")
    print(f"Test:  {len(test)} rows, churn rate {test[TARGET].mean():.3f}")


if __name__ == "__main__":
    main()