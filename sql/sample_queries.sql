-- Overall KPI snapshot
SELECT
    COUNT(*) AS total_customers,
    ROUND(AVG(Churn), 4) AS churn_rate,
    ROUND(AVG(Credit_Limit), 2) AS avg_credit_limit,
    ROUND(AVG(Months_on_book), 2) AS avg_tenure_months
FROM customers;

-- Churn rate by product band
SELECT
    CASE
        WHEN Total_Relationship_Count <= 2 THEN '1-2'
        WHEN Total_Relationship_Count <= 4 THEN '3-4'
        ELSE '5-6'
    END AS product_band,
    COUNT(*) AS customers,
    ROUND(AVG(Churn), 4) AS churn_rate
FROM customers
GROUP BY product_band
ORDER BY churn_rate DESC;

-- High-value but at-risk customers
SELECT
    Retention_Priority,
    COUNT(*) AS customers,
    ROUND(AVG(churn_probability), 4) AS avg_churn_probability,
    ROUND(AVG(Credit_Limit), 2) AS avg_credit_limit
FROM customers
GROUP BY Retention_Priority
ORDER BY avg_churn_probability DESC;

-- Cross-sell opportunities
SELECT
    Cross_Sell_Opportunity,
    COUNT(*) AS customers,
    ROUND(AVG(Churn), 4) AS churn_rate,
    ROUND(AVG(Total_Relationship_Count), 2) AS avg_products
FROM customers
GROUP BY Cross_Sell_Opportunity;
