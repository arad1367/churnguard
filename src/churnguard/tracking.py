"""Shared MLflow configuration."""
import os

import mlflow
from dotenv import load_dotenv

from churnguard.data import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

EXPERIMENT_NAME = "churnguard"
MODEL_NAME = "churnguard-churn-model"


def setup_mlflow() -> None:
    """Point MLflow at the tracking server and select our experiment."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)