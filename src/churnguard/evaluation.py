"""Business-aware evaluation: profit, threshold selection, metrics."""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Business assumptions (docs/problem_statement.md)
CUSTOMER_VALUE = 500
OFFER_COST = 50
OFFER_SUCCESS_RATE = 0.30

# Value of each decision relative to "contact nobody" ($0)
VALUE_TRUE_POSITIVE = OFFER_SUCCESS_RATE * CUSTOMER_VALUE - OFFER_COST  # +100
VALUE_FALSE_POSITIVE = -OFFER_COST                                       # -50

# If probabilities were perfectly calibrated, contact when p > this value
THEORETICAL_THRESHOLD = OFFER_COST / (OFFER_SUCCESS_RATE * CUSTOMER_VALUE)  # 0.333


def expected_profit(y_true, y_pred) -> float:
    """Profit vs. doing nothing. Missed churners = $0 (no gain, no extra cost)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return float(tp * VALUE_TRUE_POSITIVE + fp * VALUE_FALSE_POSITIVE)


def find_best_threshold(y_true, y_proba, thresholds=None):
    """Try many thresholds; return the most profitable one plus the full curve."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    y_proba = np.asarray(y_proba)
    profits = np.array(
        [expected_profit(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    )
    best = float(thresholds[profits.argmax()])
    return best, thresholds, profits


def metrics_at_threshold(y_true, y_proba, threshold: float) -> dict:
    """All metrics we track, for given probabilities and a decision threshold."""
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "profit": expected_profit(y_true, y_pred),
        "contact_rate": float(y_pred.mean()),
    }