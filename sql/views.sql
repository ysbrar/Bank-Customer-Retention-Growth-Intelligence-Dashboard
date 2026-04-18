CREATE VIEW IF NOT EXISTS vw_high_risk_customers AS
SELECT
    CLIENTNUM,
    Card_Category,
    Income_Category,
    Credit_Limit,
    Total_Relationship_Count,
    churn_probability,
    risk_band,
    Retention_Priority
FROM customers
WHERE risk_band = 'High'
ORDER BY churn_probability DESC, Credit_Limit DESC;

CREATE VIEW IF NOT EXISTS vw_cross_sell_targets AS
SELECT
    CLIENTNUM,
    Card_Category,
    Income_Category,
    Total_Relationship_Count,
    Credit_Limit,
    Cross_Sell_Opportunity,
    churn_probability
FROM customers
WHERE Cross_Sell_Opportunity IN ('Good Candidate', 'High Potential')
ORDER BY Credit_Limit DESC, churn_probability ASC;
