# EDA Findings (training set only, n = 5634)

## Key findings

1. Contract is the strongest categorical signal (Cramér's V = **_).
   Month-to-month churn ≈ _** vs two-year ≈ \_\_\_.
2. Tenure: median **_ months for churners vs _** for stayers.
   Churn concentrates in the first year.
3. Fiber optic + month-to-month is the highest-risk combination (\_\_\_).
4. Gender shows no relationship with churn (p = \_\_\_, V ≈ 0).
5. total_charges ≈ tenure × monthly_charges (corr = \_\_\_).

## Decision log

| #   | Decision                                                     | Evidence                                                                   | Alternatives considered                                                                       |
| --- | ------------------------------------------------------------ | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 1   | Fill missing total_charges with tenure × monthly_charges     | All missing rows have tenure = 0 (structural missingness); formula gives 0 | Median (wrong: gives new customers ~$1,400), drop rows (loses data, fails at prediction time) |
| 2   | Replace "No internet service" / "No phone service" with "No" | Fully redundant with internet_service / phone_service                      | Keep as separate category (redundant columns)                                                 |
| 3   | Drop gender                                                  | No predictive signal + sensitive attribute (fairness/legal risk)           | Keep and let the model decide                                                                 |
| 4   | Keep senior_citizen, but check fairness in Phase 4           | Has signal, but age is a protected attribute                               | Drop it                                                                                       |
| 5   | Encode contract as ordinal (0 < 1 < 2)                       | Natural order: longer commitment → lower churn                             | One-hot (loses order information)                                                             |
| 6   | Add engineered features (see features.py)                    | Domain reasoning, validated in notebook 04                                 | —                                                                                             |
