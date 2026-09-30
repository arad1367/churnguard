-- =========================================================
-- ChurnGuard: Exploratory SQL analysis (Snowflake)
-- Source: CHURNGUARD.ANALYTICS.CUSTOMERS_CLEAN
-- Run each query separately (cursor on query + Ctrl+Enter)
-- =========================================================

USE ROLE CHURN_ROLE;
USE WAREHOUSE CHURN_WH;
USE SCHEMA CHURNGUARD.ANALYTICS;


-- ---------------------------------------------------------
-- 1. GROUP BY: churn rate by contract type
-- Finding: month-to-month ~42% churn vs. two-year ~3%
-- ---------------------------------------------------------
SELECT contract,
       COUNT(*)               AS customers,
       ROUND(AVG(churned), 3) AS churn_rate
FROM CUSTOMERS_CLEAN
GROUP BY contract
ORDER BY churn_rate DESC;


-- ---------------------------------------------------------
-- 2. CTE + CASE WHEN: churn rate by tenure bucket
-- A CTE (WITH ...) is a named temporary result for readability
-- ---------------------------------------------------------
WITH bucketed AS (
    SELECT churned,
           CASE WHEN tenure < 12 THEN '0-1 yr'
                WHEN tenure < 24 THEN '1-2 yr'
                WHEN tenure < 48 THEN '2-4 yr'
                ELSE '4+ yr' END AS tenure_bucket
    FROM CUSTOMERS_CLEAN
)
SELECT tenure_bucket,
       COUNT(*)               AS customers,
       ROUND(AVG(churned), 3) AS churn_rate
FROM bucketed
GROUP BY tenure_bucket
ORDER BY tenure_bucket;


-- ---------------------------------------------------------
-- 3. Window function (RANK): riskiest payment method per contract
-- PARTITION BY restarts the ranking for each contract;
-- unlike GROUP BY, window functions keep every row
-- ---------------------------------------------------------
WITH rates AS (
    SELECT contract,
           payment_method,
           COUNT(*)     AS customers,
           AVG(churned) AS churn_rate
    FROM CUSTOMERS_CLEAN
    GROUP BY contract, payment_method
)
SELECT *,
       RANK() OVER (PARTITION BY contract ORDER BY churn_rate DESC) AS risk_rank
FROM rates
ORDER BY contract, risk_rank;


-- ---------------------------------------------------------
-- 4. NTILE: churn rate by monthly-charge decile
-- NTILE(10) splits customers into 10 equal-sized groups
-- ---------------------------------------------------------
WITH d AS (
    SELECT churned,
           NTILE(10) OVER (ORDER BY monthly_charges) AS decile
    FROM CUSTOMERS_CLEAN
)
SELECT decile,
       COUNT(*)               AS customers,
       ROUND(AVG(churned), 3) AS churn_rate
FROM d
GROUP BY decile
ORDER BY decile;


-- ---------------------------------------------------------
-- 5. Running total: cumulative churners by tenure
-- Inner SUM = group aggregate; outer SUM ... OVER = running window
-- ---------------------------------------------------------
SELECT tenure,
       SUM(churned)                             AS churners,
       SUM(SUM(churned)) OVER (ORDER BY tenure) AS cumulative_churners
FROM CUSTOMERS_CLEAN
GROUP BY tenure
ORDER BY tenure;


-- ---------------------------------------------------------
-- 6. QUALIFY: top 3 highest-paying customers per contract
-- QUALIFY filters on window results (like HAVING for aggregates).
-- Not all databases support it; elsewhere use a CTE + WHERE.
-- ---------------------------------------------------------
SELECT contract,
       customer_id,
       monthly_charges
FROM CUSTOMERS_CLEAN
QUALIFY ROW_NUMBER() OVER (PARTITION BY contract ORDER BY monthly_charges DESC) <= 3;


-- ---------------------------------------------------------
-- 7. JOIN: retention offers per contract + estimated discount cost
-- These offers reappear in the RAG policy documents (Phase 6)
-- ---------------------------------------------------------
CREATE OR REPLACE TABLE RETENTION_OFFERS (
    contract         VARCHAR,
    offer_name       VARCHAR,
    max_discount_pct INTEGER
);

INSERT INTO RETENTION_OFFERS VALUES
    ('Month-to-month', 'Upgrade to 1-year plan with discount',   20),
    ('One year',       'Free premium tech support for 6 months', 10),
    ('Two year',       'Loyalty thank-you credit',                5);

SELECT c.contract,
       o.offer_name,
       o.max_discount_pct,
       COUNT(*)                                                    AS churners,
       ROUND(SUM(c.monthly_charges * o.max_discount_pct / 100), 2) AS monthly_discount_cost
FROM CUSTOMERS_CLEAN c
JOIN RETENTION_OFFERS o
  ON c.contract = o.contract
WHERE c.churned = 1
GROUP BY c.contract, o.offer_name, o.max_discount_pct;