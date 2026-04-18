import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    return df

def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Churn"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
    df["Has_Revolving_Balance"] = (df["Total_Revolving_Bal"] > 0).astype(int)

    df["Engagement_Score"] = (
        0.35 * (df["Total_Trans_Ct"] / df["Total_Trans_Ct"].max())
        + 0.35 * (df["Total_Trans_Amt"] / df["Total_Trans_Amt"].max())
        + 0.20 * (1 - df["Months_Inactive_12_mon"] / max(df["Months_Inactive_12_mon"].max(), 1))
        + 0.10 * (1 - df["Contacts_Count_12_mon"] / max(df["Contacts_Count_12_mon"].max(), 1))
    ).round(4)

    df["Value_Score"] = (
        0.45 * (df["Credit_Limit"] / df["Credit_Limit"].max())
        + 0.30 * (df["Total_Trans_Amt"] / df["Total_Trans_Amt"].max())
        + 0.15 * (df["Total_Relationship_Count"] / df["Total_Relationship_Count"].max())
        + 0.10 * (df["Months_on_book"] / df["Months_on_book"].max())
    ).round(4)

    df["Engagement_Band"] = pd.qcut(df["Engagement_Score"], q=3, labels=["Low", "Medium", "High"])
    df["Value_Band"] = pd.qcut(df["Value_Score"], q=3, labels=["Low", "Medium", "High"])
    df["Tenure_Band"] = pd.cut(
        df["Months_on_book"], bins=[0, 24, 36, 48, 120],
        labels=["0-24", "25-36", "37-48", "49+"], include_lowest=True
    )
    df["Age_Band"] = pd.cut(
        df["Customer_Age"], bins=[0, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"], include_lowest=True
    )
    df["Balance_Band"] = pd.cut(
        df["Credit_Limit"], bins=[-1, 2500, 5000, 10000, 20000, 1e9],
        labels=["<2.5k", "2.5k-5k", "5k-10k", "10k-20k", "20k+"]
    )
    df["Product_Band"] = pd.cut(
        df["Total_Relationship_Count"], bins=[0, 2, 4, 6, 20],
        labels=["1-2", "3-4", "5-6", "7+"], include_lowest=True
    )

    df["Retention_Priority"] = np.select(
        [
            (df["Value_Band"].astype(str) == "High") & (df["Engagement_Band"].astype(str) == "Low"),
            (df["Value_Band"].astype(str) == "High") & (df["Engagement_Band"].astype(str).isin(["Medium", "High"])),
            (df["Value_Band"].astype(str) == "Medium") & (df["Engagement_Band"].astype(str) == "Low"),
        ],
        ["High Value, At Risk", "Loyal / Upsell", "Monitor"],
        default="Lower Priority",
    )

    df["Cross_Sell_Opportunity"] = np.select(
        [
            (df["Total_Relationship_Count"] <= 2) & (df["Engagement_Band"].astype(str).isin(["Medium", "High"])) & (df["Churn"] == 0),
            (df["Total_Relationship_Count"] <= 2) & (df["Value_Band"].astype(str) == "High"),
        ],
        ["Good Candidate", "High Potential"],
        default="Low Priority",
    )

    return df

def build_model_dataset() -> pd.DataFrame:
    df = load_raw_data()
    df = add_business_features(df)
    return df

if __name__ == "__main__":
    dataset = build_model_dataset()
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed dataset to {PROCESSED_DATA_PATH}")
