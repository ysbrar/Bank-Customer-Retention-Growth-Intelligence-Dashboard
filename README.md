# Bank Customer Retention & Growth Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44+-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.24+-blueviolet?logo=plotly&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-embedded-green?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end machine learning analytics project that answers three real business questions about bank customer churn:

- **Which customers are most likely to leave?** — Predictive model with ROC AUC 0.944+
- **Which segments should be prioritized?** — Multi-dimensional segmentation by age, tenure, engagement, and value
- **What should the business do?** — Actionable retention and cross-sell recommendations backed by model scores

Built with Python, scikit-learn, SQLite, Streamlit, and Plotly.

---

## Dashboard Preview

| Executive Overview | Customer Segmentation |
|---|---|
| KPI metrics, churn by product/credit band, risk distribution | Churn breakdowns across 4 segment dimensions |

| Model Insights | Recommendations |
|---|---|
| ROC curves (3 models), feature importance, confusion matrix | Priority-ranked action queues for retention and cross-sell |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.12 |
| Data | pandas, NumPy |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| Visualization | Plotly, Streamlit |
| Database | SQLite |
| Serialization | joblib |

---

## Project Structure

```
bank-churn-dashboard/
├── app.py                          # Streamlit dashboard (4 tabs)
├── requirements.txt
├── data/
│   ├── raw/BankChurners.csv        # 10,127 customer records
│   └── processed/customer_scored.csv
├── models/
│   └── churn_model.joblib          # Best trained pipeline
├── outputs/
│   ├── model_metrics.json          # Accuracy, precision, recall, F1, ROC AUC
│   ├── model_comparison.json       # All 3 models side-by-side
│   ├── roc_curves.json             # fpr/tpr/auc for each model
│   └── feature_importance.csv      # Permutation importance scores
├── sql/
│   ├── schema.sql                  # Table definitions
│   ├── views.sql                   # High-risk and cross-sell views
│   └── sample_queries.sql          # 4 analysis queries
└── src/
    ├── config.py                   # All file paths in one place
    ├── data_prep.py                # Feature engineering pipeline
    ├── train_model.py              # Multi-model training + evaluation
    ├── sql_loader.py               # SQLite loader
    └── insights.py                 # Shared data helpers
```

---

## Engineered Features

| Feature | Description |
|---|---|
| `Engagement_Score` | Weighted composite of transaction frequency, volume, inactivity, and contact count |
| `Value_Score` | Weighted composite of credit limit, spend, product depth, and tenure |
| `Age_Band` | 5-level age segmentation |
| `Tenure_Band` | 4-level customer lifetime segmentation |
| `Balance_Band` | 5-level credit limit segmentation |
| `Product_Band` | 4-level product count segmentation |
| `Engagement_Band` | Low / Medium / High from Engagement_Score quartiles |
| `Value_Band` | Low / Medium / High from Value_Score quartiles |
| `Retention_Priority` | Business rule combining Value_Band + Engagement_Band |
| `Cross_Sell_Opportunity` | Identifies warm, low-product customers for product expansion |
| `churn_probability` | Model output — continuous score 0–1 |
| `risk_band` | Low / Medium / High bucketed from churn_probability |

---

## Model Performance

Three models were trained and compared on an 80/20 stratified train-test split:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.887 | 0.608 | 0.834 | 0.703 | 0.944 |
| Random Forest | 0.941 | 0.902 | 0.708 | 0.793 | 0.983 |
| Gradient Boosting | 0.968 | 0.952 | 0.846 | 0.896 | 0.992 |


The best model by ROC AUC is saved to `models/churn_model.joblib` and used to score all customers.

**Top churn predictors (permutation importance):**

1. Total_Trans_Ct
2. Total_Trans_Amt
3. Total_Ct_Chng_Q4_Q1
4. Product_Band
5. Engagement_Band

---

## Key Insights

- **Overall churn rate:** 16.1%
- **High-risk customers:** 1,874
- **Highest-churn segment:** Low engagement (36.3% churn rate)
- **1–2 product customers:** 26.9% churn — highest among product bands
- **<$2,500 credit limit band:** 20.9% churn — highest among balance bands

---

## Quick Start

```bash
# 1. Clone the repo and open the folder
git clone https://github.com/ysbrar/bank-churn-dashboard.git
cd bank-churn-dashboard

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models and generate scored data
python -m src.train_model

# 5. (Optional) Load data into SQLite
python -m src.sql_loader

# 6. Launch the dashboard
streamlit run app.py
```

The dashboard auto-trains the model on first launch if outputs are missing.

---

## SQL Analytics Layer

The project includes a SQLite database with production-style SQL artifacts:

- **3 tables:** `customers`, `segment_summary`, `card_summary`
- **2 views:** `vw_high_risk_customers`, `vw_cross_sell_targets`
- **4 starter queries** in `sql/sample_queries.sql`

```sql
-- Example: churn rate by product band
SELECT Product_Band, COUNT(*) AS customers, ROUND(AVG(Churn), 4) AS churn_rate
FROM customers
GROUP BY Product_Band
ORDER BY churn_rate DESC;
```

---


## Dataset

Uses the publicly available `BankChurners.csv` dataset (Kaggle). Included under `data/raw/`.
