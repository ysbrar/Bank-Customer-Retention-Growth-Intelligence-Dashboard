DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS segment_summary;
DROP TABLE IF EXISTS card_summary;

CREATE TABLE customers (
    CLIENTNUM INTEGER PRIMARY KEY,
    Attrition_Flag TEXT,
    Churn INTEGER,
    Customer_Age INTEGER,
    Gender TEXT,
    Dependent_count INTEGER,
    Education_Level TEXT,
    Marital_Status TEXT,
    Income_Category TEXT,
    Card_Category TEXT,
    Months_on_book INTEGER,
    Total_Relationship_Count INTEGER,
    Months_Inactive_12_mon INTEGER,
    Contacts_Count_12_mon INTEGER,
    Credit_Limit REAL,
    Total_Revolving_Bal REAL,
    Avg_Open_To_Buy REAL,
    Total_Amt_Chng_Q4_Q1 REAL,
    Total_Trans_Amt REAL,
    Total_Trans_Ct INTEGER,
    Total_Ct_Chng_Q4_Q1 REAL,
    Avg_Utilization_Ratio REAL,
    Engagement_Score REAL,
    Value_Score REAL,
    Retention_Priority TEXT,
    Cross_Sell_Opportunity TEXT,
    churn_probability REAL,
    risk_band TEXT
);

CREATE TABLE segment_summary (
    Age_Band TEXT,
    Tenure_Band TEXT,
    Product_Band TEXT,
    customer_count INTEGER,
    churn_rate REAL
);

CREATE TABLE card_summary (
    Card_Category TEXT,
    customer_count INTEGER,
    churn_rate REAL,
    avg_credit_limit REAL,
    avg_transactions REAL
);
