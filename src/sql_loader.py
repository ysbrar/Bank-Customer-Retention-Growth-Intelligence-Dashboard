import sqlite3
import pandas as pd

from src.config import PROCESSED_DATA_PATH, SQLITE_DB_PATH

def load_into_sqlite() -> None:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    conn = sqlite3.connect(SQLITE_DB_PATH)

    customers = df[[
        "CLIENTNUM", "Attrition_Flag", "Churn", "Customer_Age", "Gender", "Dependent_count",
        "Education_Level", "Marital_Status", "Income_Category", "Card_Category",
        "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon",
        "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio", "Engagement_Score", "Value_Score", "Retention_Priority",
        "Cross_Sell_Opportunity", "churn_probability", "risk_band"
    ]].copy()
    customers.to_sql("customers", conn, if_exists="replace", index=False)

    segment_summary = (
        df.groupby(["Age_Band", "Tenure_Band", "Product_Band"], dropna=False)["Churn"]
        .agg(customer_count="count", churn_rate="mean")
        .reset_index()
    )
    segment_summary.to_sql("segment_summary", conn, if_exists="replace", index=False)

    card_summary = (
        df.groupby("Card_Category", dropna=False)
        .agg(
            customer_count=("CLIENTNUM", "count"),
            churn_rate=("Churn", "mean"),
            avg_credit_limit=("Credit_Limit", "mean"),
            avg_transactions=("Total_Trans_Ct", "mean"),
        )
        .reset_index()
    )
    card_summary.to_sql("card_summary", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    print(f"SQLite database created at {SQLITE_DB_PATH}")

if __name__ == "__main__":
    load_into_sqlite()
