| Column       | Type            | Meaning            | Notes / issues                          |
| ------------ | --------------- | ------------------ | --------------------------------------- |
| customerID   | string          | Unique ID          | Not a feature, identifier only          |
| tenure       | int             | Months as customer | 0 = new customer                        |
| TotalCharges | should be float | Total billed       | Stored as text; 11 blanks when tenure=0 |
| Churn        | Yes/No          | Target             | ~26.5% Yes                              |
| ...          |                 |                    |                                         |
