# ChurnGuard: Problem Statement

## Business problem

[Telecom loses customers; retention team has limited budget...]

## Decision supported

[Which customers to contact with a retention offer each month]

## Target definition

Churn = "Yes" if the customer left within the last month.

## Cost assumptions

- Value of retained customer: $500
- Offer cost: $50
- Offer success rate: 30%

## Success metrics

- ML: PR-AUC, recall at precision ≥ X
- Business: expected profit vs. "contact nobody" and "contact everyone"

## Baseline to beat

Always predict "No churn" → ~73.5% accuracy, 0% recall, $0 profit.

## Risks

- Class imbalance
- Data leakage (features only known after a customer churns)
- Dataset is a snapshot, so real data will drift over time
