# Model Card: churnguard-churn-model v\_\_

## Purpose

Ranks telecom customers by churn risk so the retention team can decide whom to contact.
Not intended for: pricing decisions, credit decisions, or any automated action without human review.

## Training data

IBM Telco Customer Churn, 5634 training rows (80% stratified split), churn rate 26.5%.
Snapshot data; no time dimension, so drift over time is untested.

## Model

Type: **_ (selected over _** because \_\_\_)
Features: see src/churnguard/features.py; gender excluded (no signal + sensitive).

## Performance

| Metric                    | CV (train, mean ± std) | Test   |
| ------------------------- | ---------------------- | ------ |
| PR-AUC                    | **_ ± _**              | \_\_\_ |
| ROC-AUC                   | \_\_\_                 | \_\_\_ |
| Recall @ threshold        | \_\_\_                 | \_\_\_ |
| Precision @ threshold     | \_\_\_                 | \_\_\_ |
| Profit vs. contact nobody | \_\_\_                 | \_\_\_ |

Threshold: \_\_\_ (profit-maximizing on out-of-fold predictions; theoretical 0.33 if calibrated)

## Fairness

Recall seniors: **_ | non-seniors: _** | gap: **_
Interpretation: _**

## Limitations

- Business values ($500, $50, 30%) are assumptions; profit estimates depend on them.
- Synthetic/public dataset; real performance must be re-validated on company data.
- Probabilities from class-weighted models are not calibrated (if applicable).

## Explainability

Top global drivers (SHAP): **_, _**, \_\_\_
