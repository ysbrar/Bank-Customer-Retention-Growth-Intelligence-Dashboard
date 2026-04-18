from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH        = ROOT_DIR / "data" / "raw" / "BankChurners.csv"
PROCESSED_DATA_PATH  = ROOT_DIR / "data" / "processed" / "customer_scored.csv"
MODEL_PATH           = ROOT_DIR / "models" / "churn_model.joblib"
METRICS_PATH         = ROOT_DIR / "outputs" / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "outputs" / "feature_importance.csv"
ROC_CURVES_PATH      = ROOT_DIR / "outputs" / "roc_curves.json"
COMPARISON_PATH      = ROOT_DIR / "outputs" / "model_comparison.json"
SQLITE_DB_PATH       = ROOT_DIR / "bank_churn.db"
