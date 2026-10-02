"""One-time final evaluation of the 'candidate' model on the untouched test set.

Includes a fairness check and a promotion gate:
if the model passes, it becomes 'champion' (the version the API will serve).

    python -m churnguard.evaluate
"""
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from churnguard.evaluation import metrics_at_threshold
from churnguard.features import split_X_y
from churnguard.split import TEST_PATH
from churnguard.tracking import MODEL_NAME, setup_mlflow

MIN_TEST_PR_AUC = 0.55        # must clearly beat the ~0.265 baseline
MAX_RECALL_GAP = 0.10         # fairness: recall difference between groups


def main() -> None:
    setup_mlflow()
    client = MlflowClient()

    mv = client.get_model_version_by_alias(MODEL_NAME, "candidate")
    threshold = float(mv.tags["threshold"])
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@candidate")
    print(f"Evaluating {MODEL_NAME} v{mv.version} ({mv.tags['model_type']}), "
          f"threshold {threshold}")

    X_test, y_test = split_X_y(pd.read_parquet(TEST_PATH))
    proba = model.predict_proba(X_test)[:, 1]
    overall = metrics_at_threshold(y_test, proba, threshold)

    # Fairness: compare metrics for seniors vs. non-seniors
    groups = {}
    for value, label in [(0, "non_senior"), (1, "senior")]:
        mask = (X_test["senior_citizen"] == value).to_numpy()
        groups[label] = metrics_at_threshold(y_test[mask], proba[mask], threshold)
        groups[label]["churn_rate"] = float(y_test[mask].mean())
        groups[label]["n"] = int(mask.sum())
    recall_gap = abs(groups["senior"]["recall"] - groups["non_senior"]["recall"])

    with mlflow.start_run(run_name=f"test_eval_v{mv.version}"):
        mlflow.set_tag("stage", "test_evaluation")
        mlflow.set_tag("model_version", mv.version)
        mlflow.log_param("threshold", threshold)
        mlflow.log_metrics({f"test_{k}": v for k, v in overall.items()})
        for label, m in groups.items():
            mlflow.log_metrics({f"test_{label}_{k}": v for k, v in m.items()})
        mlflow.log_metric("test_recall_gap_senior", recall_gap)

    print("\n=== Test set results ===")
    print(pd.Series(overall).round(3).to_string())
    print("\n=== Fairness (by senior_citizen) ===")
    print(pd.DataFrame(groups).T[["n", "churn_rate", "recall", "precision", "contact_rate"]]
          .round(3).to_string())
    print(f"Recall gap: {recall_gap:.3f}")

    client.set_model_version_tag(MODEL_NAME, mv.version, "test_pr_auc", f"{overall['pr_auc']:.3f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "test_profit", f"{overall['profit']:.0f}")

    passed = overall["pr_auc"] >= MIN_TEST_PR_AUC and overall["profit"] > 0
    if passed:
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
        print(f"\n✅ PASSED gate → v{mv.version} is now 'champion'")
    else:
        print("\n❌ FAILED gate → model NOT promoted")

    if recall_gap > MAX_RECALL_GAP:
        print(f"⚠️  Recall gap {recall_gap:.3f} > {MAX_RECALL_GAP}: needs human fairness review")


if __name__ == "__main__":
    main()