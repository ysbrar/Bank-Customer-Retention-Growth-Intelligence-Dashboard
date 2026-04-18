import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    COMPARISON_PATH, FEATURE_IMPORTANCE_PATH,
    METRICS_PATH, PROCESSED_DATA_PATH, ROC_CURVES_PATH,
)
from src.train_model import train_and_save_model

st.set_page_config(
    page_title="Bank Churn Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
  [data-testid="metric-container"] {
    background: #f7f9fc;
    border: 1px solid #dde3ef;
    border-radius: 8px;
    padding: 12px 16px;
  }
  [data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700; }
  .stTabs [data-baseweb="tab"] {
    background: #f0f2f6;
    border-radius: 6px 6px 0 0;
    padding: 6px 20px;
    font-weight: 500;
    color: #1a1a1a !important;
  }
  .stTabs [aria-selected="true"] {
    background: white;
    border-bottom: 3px solid #1f77b4;
    color: #1f77b4 !important;
  }
  h1 { font-size: 1.75rem !important; font-weight: 700; }
  h3 { color: #1a3557; margin-top: 0.5rem; }
  .insight-box {
    background: #f0f6ff;
    border-left: 4px solid #1f77b4;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 8px;
    color: #1a1a1a !important;
  }
  .insight-box * { color: #1a1a1a !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────
CHART_BG = "rgba(0,0,0,0)"
HEATMAP_SCALE = ["#aec7e8", "#1f77b4"]
CHURN_SCALE = ["#aec7e8", "#d62728"]
RISK_COLORS = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}

BAND_ORDER = {
    "Age_Band":        ["18-30", "31-40", "41-50", "51-60", "60+"],
    "Tenure_Band":     ["0-24", "25-36", "37-48", "49+"],
    "Balance_Band":    ["<2.5k", "2.5k-5k", "5k-10k", "10k-20k", "20k+"],
    "Product_Band":    ["1-2", "3-4", "5-6", "7+"],
    "Engagement_Band": ["Low", "Medium", "High"],
    "Value_Band":      ["Low", "Medium", "High"],
    "risk_band":       ["Low", "Medium", "High"],
    "Retention_Priority": ["High Value, At Risk", "Monitor", "Loyal / Upsell", "Lower Priority"],
}


# ── Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_assets():
    needs_train = (
        not PROCESSED_DATA_PATH.exists()
        or not FEATURE_IMPORTANCE_PATH.exists()
        or not METRICS_PATH.exists()
    )
    if needs_train:
        train_and_save_model()

    df = pd.read_csv(PROCESSED_DATA_PATH)
    fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    rocs = None
    if ROC_CURVES_PATH.exists():
        with open(ROC_CURVES_PATH, encoding="utf-8") as f:
            rocs = json.load(f)
    comparison = None
    if COMPARISON_PATH.exists():
        with open(COMPARISON_PATH, encoding="utf-8") as f:
            comparison = json.load(f)
    return df, fi, metrics, rocs, comparison


df, fi, metrics, rocs, comparison = load_assets()


# ── Helpers ───────────────────────────────────────────────────────────────
def churn_bar(data_df, col, title, order_key=None):
    agg = (
        data_df.groupby(col, dropna=False)["Churn"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Churn Rate", "count": "Customers"})
    )
    if order_key and order_key in BAND_ORDER:
        agg[col] = pd.Categorical(agg[col], categories=BAND_ORDER[order_key], ordered=True)
        agg = agg.sort_values(col)
    agg["label"] = agg["Churn Rate"].map("{:.1%}".format)
    fig = px.bar(
        agg, x=col, y="Churn Rate", text="label", title=title,
        color="Churn Rate", color_continuous_scale=CHURN_SCALE,
        hover_data={"Customers": True, "Churn Rate": ":.1%"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        plot_bgcolor=CHART_BG, yaxis_tickformat=".0%",
        margin=dict(t=40, b=20, l=0, r=0),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    selected_gender = st.multiselect(
        "Gender",
        sorted(df["Gender"].dropna().unique()),
        default=sorted(df["Gender"].dropna().unique()),
    )
    selected_card = st.multiselect(
        "Card Category",
        sorted(df["Card_Category"].dropna().unique()),
        default=sorted(df["Card_Category"].dropna().unique()),
    )
    selected_risk = st.multiselect(
        "Risk Band", ["Low", "Medium", "High"], default=["Low", "Medium", "High"]
    )
    st.divider()
    best_model_name = df["best_model"].iloc[0] if "best_model" in df.columns else "Logistic Regression"
    st.markdown(f"""
**Dataset:** 10,127 customers
**Best Model:** {best_model_name}
**ROC AUC:** {metrics.get('roc_auc', 0):.3f}
**Churn Rate:** {df['Churn'].mean():.1%}
    """)

filtered_df = df[
    df["Gender"].isin(selected_gender)
    & df["Card_Category"].isin(selected_card)
    & df["risk_band"].isin(selected_risk)
].copy()

# ── Header ────────────────────────────────────────────────────────────────
st.title("Bank Customer Retention & Growth Intelligence")
st.caption(
    f"End-to-end churn analytics across **{len(df):,} customers** · "
    "Python · Scikit-learn · SQL · Streamlit · Plotly"
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview", "Customer Segmentation", "Model Insights", "Recommendations"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Overview
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    total          = len(filtered_df)
    churn_rate     = filtered_df["Churn"].mean()
    avg_credit     = filtered_df["Credit_Limit"].mean()
    avg_tenure     = filtered_df["Months_on_book"].mean()
    high_risk      = int((filtered_df["risk_band"] == "High").sum())
    cross_sell_ct  = int((filtered_df["Cross_Sell_Opportunity"] == "Good Candidate").sum())
    base_churn     = df["Churn"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers",      f"{total:,}")
    c2.metric(
        "Churn Rate", f"{churn_rate:.1%}",
        delta=f"{churn_rate - base_churn:+.1%}" if total < len(df) else None,
        delta_color="inverse",
    )
    c3.metric("Avg Credit Limit",     f"${avg_credit:,.0f}")
    c4.metric("Avg Tenure",           f"{avg_tenure:.1f} mo")
    c5.metric("High-Risk Customers",  f"{high_risk:,}")
    c6.metric("Cross-Sell Candidates", f"{cross_sell_ct:,}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            churn_bar(filtered_df, "Product_Band", "Churn Rate by Product Count", "Product_Band"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            churn_bar(filtered_df, "Balance_Band", "Churn Rate by Credit Limit Band", "Balance_Band"),
            use_container_width=True,
        )

    risk_dist = (
        filtered_df["risk_band"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .reset_index()
    )
    risk_dist.columns = ["Risk Band", "Customers"]
    fig_pie = px.pie(
        risk_dist, names="Risk Band", values="Customers",
        title="Customer Risk Band Distribution",
        color="Risk Band", color_discrete_map=RISK_COLORS,
        hole=0.42,
    )
    fig_pie.update_traces(textinfo="percent+label", pull=[0, 0, 0.05])
    fig_pie.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Customer Segmentation
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            churn_bar(filtered_df, "Age_Band",        "Churn Rate by Age Band",        "Age_Band"),
            use_container_width=True,
        )
        st.plotly_chart(
            churn_bar(filtered_df, "Engagement_Band", "Churn Rate by Engagement Level", "Engagement_Band"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            churn_bar(filtered_df, "Tenure_Band",     "Churn Rate by Tenure Band",     "Tenure_Band"),
            use_container_width=True,
        )
        st.plotly_chart(
            churn_bar(filtered_df, "Value_Band",      "Churn Rate by Customer Value",  "Value_Band"),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Top 20 Retention Targets")
    st.caption("Customers sorted by churn probability then credit limit — highest-value at-risk accounts first.")

    display_cols = [
        "CLIENTNUM", "Card_Category", "Income_Category",
        "Total_Relationship_Count", "Credit_Limit", "Total_Trans_Ct",
        "Retention_Priority", "Cross_Sell_Opportunity", "churn_probability", "risk_band",
    ]
    top_targets = filtered_df[display_cols].sort_values(
        ["churn_probability", "Credit_Limit"], ascending=[False, False]
    ).head(20).copy()
    top_targets["churn_probability"] = top_targets["churn_probability"].map("{:.1%}".format)
    top_targets["Credit_Limit"]      = top_targets["Credit_Limit"].map("${:,.0f}".format)
    st.dataframe(top_targets, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{metrics['accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")
    c3.metric("Recall",    f"{metrics['recall']:.3f}")
    c4.metric("F1 Score",  f"{metrics['f1_score']:.3f}")
    c5.metric("ROC AUC",   f"{metrics['roc_auc']:.3f}")

    st.divider()
    col1, col2 = st.columns([3, 2])

    with col1:
        fi_plot = fi.head(12).sort_values("importance")
        fig_fi = px.bar(
            fi_plot, x="importance", y="feature", orientation="h",
            title="Top 12 Churn Predictors (Permutation Importance)",
            color="importance", color_continuous_scale=HEATMAP_SCALE,
        )
        fig_fi.update_layout(
            coloraxis_showscale=False, plot_bgcolor=CHART_BG,
            yaxis_title="", margin=dict(t=40, b=20, l=0, r=0),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col2:
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            x=["Predicted Stay", "Predicted Churn"],
            y=["Actual Stay", "Actual Churn"],
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            title="Confusion Matrix",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_cm.update_layout(
            plot_bgcolor=CHART_BG, margin=dict(t=40, b=20, l=0, r=0)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curves
    if rocs:
        st.divider()
        col_roc, col_cmp = st.columns([3, 2])

        with col_roc:
            st.subheader("ROC Curves — Model Comparison")
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            fig_roc = go.Figure()
            fig_roc.add_shape(
                type="line", x0=0, x1=1, y0=0, y1=1,
                line=dict(dash="dot", color="#aaa", width=1),
            )
            for i, (name, curve) in enumerate(rocs.items()):
                fig_roc.add_trace(go.Scatter(
                    x=curve["fpr"], y=curve["tpr"],
                    name=f"{name} (AUC = {curve['auc']:.3f})",
                    mode="lines",
                    line=dict(color=palette[i % len(palette)], width=2.5),
                ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.55, y=0.08, bgcolor="rgba(255,255,255,0.8)"),
                plot_bgcolor=CHART_BG,
                xaxis=dict(range=[0, 1], showgrid=True, gridcolor="#eee"),
                yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#eee"),
                margin=dict(t=20, b=20, l=0, r=0),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_cmp:
            if comparison:
                st.subheader("Model Comparison")
                comp_df = pd.DataFrame(comparison).T.reset_index()
                comp_df.columns = ["Model"] + list(comp_df.columns[1:])
                for col in comp_df.columns[1:]:
                    comp_df[col] = comp_df[col].astype(float).map("{:.3f}".format)
                comp_df.columns = [
                    c.replace("_", " ").title() for c in comp_df.columns
                ]
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Recommendations
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Business Recommendations")

        insights = [
            ("Retain low-product customers first",
             "Customers with <strong>1–2 products</strong> churn at ~27% — the highest of any segment. "
             "Prioritize onboarding campaigns that encourage a second product within the first 6 months."),
            ("Intervene before engagement drops",
             "Low-engagement customers churn at <strong>36%+</strong>. Proactive outreach — rewards nudges, "
             "check-in calls — works best before customers go fully silent."),
            ("Protect high-value at-risk accounts",
             "The <strong>'High Value, At Risk'</strong> cohort has strong credit limits but declining engagement. "
             "Assign relationship-manager outreach here for the highest retention ROI."),
            ("Cross-sell to warm, engaged customers",
             "Customers with ≤2 products and medium/high engagement are the lowest-cost "
             "cross-sell targets — they are already active and receptive."),
            ("Use model scores to build a weekly action queue",
             "The churn_probability and risk_band fields can power a prioritized weekly list "
             "for retention teams — highest probability, highest credit limit first."),
        ]

        for i, (title, body) in enumerate(insights, 1):
            st.markdown(
                f'<div class="insight-box"><strong>{i}. {title}</strong><br>{body}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.subheader("Retention Priority Breakdown")
        rec = filtered_df.groupby("Retention_Priority", dropna=False).agg(
            Customers=("CLIENTNUM", "count"),
            Churn_Rate=("Churn", "mean"),
            Avg_Credit_Limit=("Credit_Limit", "mean"),
            Avg_Churn_Prob=("churn_probability", "mean"),
        ).reset_index()
        priority_order = BAND_ORDER["Retention_Priority"]
        rec["Retention_Priority"] = pd.Categorical(
            rec["Retention_Priority"], categories=priority_order, ordered=True
        )
        rec = rec.sort_values("Retention_Priority")
        rec["Churn_Rate"]       = rec["Churn_Rate"].map("{:.1%}".format)
        rec["Avg_Credit_Limit"] = rec["Avg_Credit_Limit"].map("${:,.0f}".format)
        rec["Avg_Churn_Prob"]   = rec["Avg_Churn_Prob"].map("{:.1%}".format)
        st.dataframe(rec, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Cross-Sell Opportunity Breakdown")
    cross = filtered_df.groupby("Cross_Sell_Opportunity", dropna=False).agg(
        Customers=("CLIENTNUM", "count"),
        Churn_Rate=("Churn", "mean"),
        Avg_Products=("Total_Relationship_Count", "mean"),
        Avg_Credit_Limit=("Credit_Limit", "mean"),
    ).reset_index()
    cross["Churn_Rate"]       = cross["Churn_Rate"].map("{:.1%}".format)
    cross["Avg_Products"]     = cross["Avg_Products"].map("{:.2f}".format)
    cross["Avg_Credit_Limit"] = cross["Avg_Credit_Limit"].map("${:,.0f}".format)
    st.dataframe(cross, use_container_width=True, hide_index=True)
