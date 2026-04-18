import json
import pandas as pd
from src.config import FEATURE_IMPORTANCE_PATH, METRICS_PATH, PROCESSED_DATA_PATH

def load_project_assets():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    return df, feature_importance, metrics

def top_segment_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col, dropna=False)["Churn"]
        .agg(customer_count="count", churn_rate="mean")
        .sort_values("churn_rate", ascending=False)
        .reset_index()
    )
