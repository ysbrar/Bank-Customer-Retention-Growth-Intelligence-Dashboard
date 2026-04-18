import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    COMPARISON_PATH, FEATURE_IMPORTANCE_PATH, METRICS_PATH,
    MODEL_PATH, PROCESSED_DATA_PATH, ROC_CURVES_PATH,
)
from src.data_prep import build_model_dataset


def _build_preprocessor(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    categorical_cols = list(dict.fromkeys(categorical_cols))
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    return (
        ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), numeric_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), categorical_cols),
            ]
        ),
        numeric_cols,
        categorical_cols,
    )


def _make_pipeline(estimator, preprocessor) -> Pipeline:
    return Pipeline(steps=[("prep", preprocessor), ("model", estimator)])


def _evaluate(pipe, X_test, y_test) -> dict:
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    return {
        "metrics": {
            "accuracy":  float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall":    float(recall_score(y_test, preds)),
            "f1_score":  float(f1_score(y_test, preds)),
            "roc_auc":   float(roc_auc_score(y_test, probs)),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        },
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc_score(y_test, probs)),
        },
    }


def train_and_save_model() -> None:
    df = build_model_dataset()
    feature_cols = [c for c in df.columns if c not in ["CLIENTNUM", "Attrition_Flag", "Churn"]]
    X = df[feature_cols]
    y = df["Churn"]

    preprocessor, numeric_cols, _ = _build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    }

    results = {}
    roc_curves_out = {}
    best_name, best_pipe, best_auc = None, None, -1.0

    for name, estimator in candidates.items():
        pipe = _make_pipeline(estimator, _build_preprocessor(X)[0])
        pipe.fit(X_train, y_train)
        eval_result = _evaluate(pipe, X_test, y_test)

        results[name] = eval_result["metrics"]
        roc_curves_out[name] = eval_result["roc"]

        if eval_result["metrics"]["roc_auc"] > best_auc:
            best_auc = eval_result["metrics"]["roc_auc"]
            best_name = name
            best_pipe = pipe

    print(f"Best model: {best_name} (ROC AUC = {best_auc:.4f})")

    # Feature importance from the best model (on numeric features via permutation)
    fi_result = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc"
    )
    importance_df = (
        pd.DataFrame({"feature": X.columns, "importance": fi_result.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Score the full dataset with the best model
    df["churn_probability"] = best_pipe.predict_proba(X)[:, 1]
    df["risk_band"] = pd.cut(
        df["churn_probability"], bins=[-0.001, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"]
    )
    df["best_model"] = best_name

    # Comparison table (drop confusion matrix for readability)
    comparison = {
        name: {k: v for k, v in m.items() if k != "confusion_matrix"}
        for name, m in results.items()
    }

    for path in [MODEL_PATH, FEATURE_IMPORTANCE_PATH, METRICS_PATH, ROC_CURVES_PATH, COMPARISON_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipe, MODEL_PATH)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results[best_name], f, indent=2)

    with open(ROC_CURVES_PATH, "w", encoding="utf-8") as f:
        json.dump(roc_curves_out, f, indent=2)

    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("Model, metrics, ROC curves, comparison, feature importance, and scored data saved.")


if __name__ == "__main__":
    train_and_save_model()
