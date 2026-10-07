"""Train PyTorch MLPs with the SAME cross-validation protocol as Phase 4
and compare them with the existing models in MLflow.

    python -m churnguard.train_nn
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.model_selection import cross_validate

from churnguard.features import split_X_y
from churnguard.nn_model import TorchMLPClassifier
from churnguard.split import TRAIN_PATH
from churnguard.tracking import EXPERIMENT_NAME, setup_mlflow
from churnguard.train import CV, SCORING, make_pipeline

CONFIGS = {
    # Deliberately over-sized and unregularized: shows overfitting in the curves
    "mlp_overfit_demo": dict(
        hidden_sizes=(256, 128), dropout=0.0, weight_decay=0.0,
        lr=1e-3, max_epochs=150, patience=None,
    ),
    # Small network + dropout + weight decay + early stopping
    "mlp_regularized": dict(
        hidden_sizes=(64, 32), dropout=0.3, weight_decay=1e-3,
        lr=1e-3, max_epochs=300, patience=20,
    ),
}


def plot_learning_curves(history: dict, best_epoch: int, title: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="train loss")
    ax.plot(epochs, history["val_loss"], label="validation loss")
    ax.axvline(best_epoch, ls="--", color="red", label=f"best epoch = {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def run_config(name: str, params: dict, X: pd.DataFrame, y: pd.Series) -> dict:
    pipe = make_pipeline(TorchMLPClassifier(**params), scale_numeric=True)

    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"stage": "cv_comparison", "framework": "pytorch"})
        mlflow.log_param("model", name)
        mlflow.log_params(
            {f"model__{k}": v for k, v in pipe.named_steps["model"].get_params().items()}
        )

        # 1. Same 5-fold CV and metrics as Phase 4 → fair comparison
        scores = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=1)
        summary = {}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            if metric == "brier":
                values = -values
            mlflow.log_metric(f"cv_{metric}_mean", values.mean())
            mlflow.log_metric(f"cv_{metric}_std", values.std())
            summary[metric] = values.mean()

        # 2. One fit on all training data to record learning curves
        pipe.fit(X, y)
        net = pipe.named_steps["model"]
        for epoch, (tl, vl, vp) in enumerate(
            zip(net.history_["train_loss"], net.history_["val_loss"], net.history_["val_pr_auc"]),
            start=1,
        ):
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl, "val_pr_auc": vp}, step=epoch)
        mlflow.log_metrics({"epochs_trained": net.n_epochs_, "best_epoch": net.best_epoch_})

        fig = plot_learning_curves(net.history_, net.best_epoch_, name)
        mlflow.log_figure(fig, "learning_curve.png")
        plt.close(fig)

    print(
        f"{name:18s} PR-AUC {summary['pr_auc']:.3f} | ROC-AUC {summary['roc_auc']:.3f} | "
        f"Brier {summary['brier']:.3f} | epochs {net.n_epochs_} (best {net.best_epoch_})"
    )
    return summary


def main() -> None:
    setup_mlflow()
    X, y = split_X_y(pd.read_parquet(TRAIN_PATH))

    print("Training PyTorch MLPs with 5-fold CV (several minutes)...\n")
    for name, params in CONFIGS.items():
        run_config(name, params, X, y)

    # Compare with every model from Phase 4
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.stage = 'cv_comparison'",
    )
    table = (
        runs[["tags.mlflow.runName", "metrics.cv_pr_auc_mean", "metrics.cv_pr_auc_std",
              "metrics.cv_roc_auc_mean", "metrics.cv_brier_mean"]]
        .rename(columns=lambda c: c.split(".")[-1])
        .sort_values("cv_pr_auc_mean", ascending=False)
        .round(3)
    )
    print("\n=== All models (5-fold CV on training data) ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()