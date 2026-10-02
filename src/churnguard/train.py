"""Train, compare, tune, and register churn models with MLflow.

Run from the project root, with the MLflow server running:
    python -m churnguard.train
"""
import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import skops.io as sio
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churnguard.data import PROJECT_ROOT
from churnguard.evaluation import (
    CUSTOMER_VALUE,
    OFFER_COST,
    OFFER_SUCCESS_RATE,
    expected_profit,
    find_best_threshold,
    metrics_at_threshold,
)
from churnguard.features import build_preprocessor, split_X_y
from churnguard.split import RANDOM_STATE, TRAIN_PATH
from churnguard.tracking import MODEL_NAME, setup_mlflow

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "brier": "neg_brier_score",  # sklearn maximizes scores, so Brier is negated
}

# Simplest first. If models perform about equally, we prefer the simpler one.
COMPLEXITY_ORDER = [
    "logreg_unweighted", "logreg_balanced", "random_forest", "xgboost", "xgboost_tuned",
]
PR_AUC_TOLERANCE = 0.01

# Allowlist for model serialization: only types from these packages may be
# trusted when saving/loading the model with skops (a safer format than pickle).
TRUSTED_PREFIXES = ("churnguard.", "numpy.", "sklearn.", "xgboost.", "scipy.")


def make_pipeline(model, scale_numeric: bool = True) -> Pipeline:
    """Preprocessing + model in ONE object (prevents training-serving skew)."""
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
        ("model", model),
    ])


def get_candidates(pos_weight: float) -> dict[str, Pipeline]:
    return {
        "baseline_dummy": make_pipeline(DummyClassifier(strategy="prior")),
        "logreg_unweighted": make_pipeline(LogisticRegression(max_iter=1000)),
        "logreg_balanced": make_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=False,  # trees don't need scaling
        ),
        "xgboost": make_pipeline(
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_numeric=False,
        ),
    }


def log_cv_run(name: str, pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Cross-validate one pipeline and log everything to MLflow as one run."""
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("stage", "cv_comparison")
        mlflow.log_param("model", name)
        mlflow.log_params(
            {f"model__{k}": v for k, v in pipe.named_steps["model"].get_params().items()}
        )

        scores = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=1)

        summary = {}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            if metric == "brier":
                values = -values  # undo sklearn's negation
            mlflow.log_metric(f"cv_{metric}_mean", values.mean())
            mlflow.log_metric(f"cv_{metric}_std", values.std())
            summary[metric] = values.mean()

    print(
        f"{name:18s} PR-AUC {summary['pr_auc']:.3f} | ROC-AUC {summary['roc_auc']:.3f} | "
        f"recall@0.5 {summary['recall']:.3f} | precision@0.5 {summary['precision']:.3f} | "
        f"Brier {summary['brier']:.3f}"
    )
    return summary


def tune_xgboost(X: pd.DataFrame, y: pd.Series, pos_weight: float) -> RandomizedSearchCV:
    """Random search over XGBoost hyperparameters, optimizing PR-AUC."""
    pipe = make_pipeline(
        XGBClassifier(
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scale_numeric=False,
    )
    param_distributions = {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, 5],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_lambda": [0.5, 1.0, 5.0, 10.0],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions,
        n_iter=25,
        scoring="average_precision",
        cv=CV,
        random_state=RANDOM_STATE,
        n_jobs=1,  # XGBoost already uses all CPU cores
        verbose=1,
    )
    search.fit(X, y)
    return search


def get_trusted_types(model) -> list[str]:
    """Find the types skops won't load by default and trust them ONLY if they
    come from allowlisted packages. Anything unexpected stops the pipeline."""
    untrusted = sio.get_untrusted_types(data=sio.dumps(model))
    unexpected = [t for t in untrusted if not t.startswith(TRUSTED_PREFIXES)]
    if unexpected:
        raise ValueError(f"Refusing to trust unexpected types in model: {unexpected}")
    print(f"      Trusting {len(untrusted)} reviewed types: {untrusted}")
    return untrusted


def plot_profit_curve(thresholds, profits, best: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, profits)
    ax.axvline(best, ls="--", color="red", label=f"best threshold = {best:.2f}")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected profit ($, out-of-fold)")
    ax.set_title("Profit vs. threshold")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    setup_mlflow()

    train = pd.read_parquet(TRAIN_PATH)
    X, y = split_X_y(train)
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    print(f"Training rows: {len(X)} | scale_pos_weight: {pos_weight:.2f}\n")

    # 1. Compare candidate models with cross-validation
    print("Step 1/5: Cross-validating candidate models...")
    pipelines = get_candidates(pos_weight)
    results = {name: log_cv_run(name, pipe, X, y) for name, pipe in pipelines.items()}

    # 2. Tune gradient boosting
    print("\nStep 2/5: Tuning XGBoost (a few minutes)...")
    search = tune_xgboost(X, y, pos_weight)
    pipelines["xgboost_tuned"] = search.best_estimator_
    results["xgboost_tuned"] = log_cv_run("xgboost_tuned", search.best_estimator_, X, y)

    # 3. Select: simplest model within tolerance of the best PR-AUC
    best_score = max(results[n]["pr_auc"] for n in COMPLEXITY_ORDER)
    best_name = next(
        n for n in COMPLEXITY_ORDER if results[n]["pr_auc"] >= best_score - PR_AUC_TOLERANCE
    )
    print(f"\nStep 3/5: Best PR-AUC {best_score:.3f} → selected '{best_name}' "
          f"(simplest within {PR_AUC_TOLERANCE})")

    # 4. Choose the threshold on OUT-OF-FOLD predictions (never on the test set)
    print("Step 4/5: Choosing profit-maximizing threshold...")
    final_pipe = clone(pipelines[best_name])
    oof_proba = cross_val_predict(final_pipe, X, y, cv=CV, method="predict_proba")[:, 1]
    threshold, thresholds, profits = find_best_threshold(y, oof_proba)
    oof_metrics = metrics_at_threshold(y, oof_proba, threshold)
    print(f"      threshold = {threshold:.2f} | OOF profit = ${oof_metrics['profit']:,.0f} | "
          f"recall = {oof_metrics['recall']:.3f} | precision = {oof_metrics['precision']:.3f}")

    # 5. Fit on ALL training data, log, register
    print("Step 5/5: Fitting final model and registering it...")
    final_pipe.fit(X, y)

    with mlflow.start_run(run_name=f"final_{best_name}"):
        mlflow.set_tag("stage", "final_training")
        mlflow.log_params({
            "model": best_name,
            "threshold": threshold,
            "customer_value": CUSTOMER_VALUE,
            "offer_cost": OFFER_COST,
            "offer_success_rate": OFFER_SUCCESS_RATE,
            "train_rows": len(X),
        })
        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})
        mlflow.log_metric(
            "oof_profit_contact_everyone", expected_profit(y, np.ones(len(y), dtype=int))
        )

        fig = plot_profit_curve(thresholds, profits, threshold)
        mlflow.log_figure(fig, "profit_curve.png")
        plt.close(fig)

        signature = infer_signature(X, final_pipe.predict(X))
        model_info = mlflow.sklearn.log_model(
            sk_model=final_pipe,
            name="model",
            signature=signature,
            input_example=X.head(3),
            code_paths=[str(PROJECT_ROOT / "src" / "churnguard")],
            skops_trusted_types=get_trusted_types(final_pipe),
        )

    client = MlflowClient()
    version = mlflow.register_model(model_info.model_uri, MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, version.version, "threshold", f"{threshold:.2f}")
    client.set_model_version_tag(MODEL_NAME, version.version, "model_type", best_name)
    client.set_registered_model_alias(MODEL_NAME, "candidate", version.version)

    print(f"\nRegistered {MODEL_NAME} version {version.version} with alias 'candidate'.")
    print("Next: analyze it in notebooks/05_model_analysis.ipynb, then run "
          "python -m churnguard.evaluate")


if __name__ == "__main__":
    main()