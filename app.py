import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# Page Configurations & Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Helix Bank | AI Churn Prediction Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Design System & Styling: Midnight Navy (#0B1120), Electric Blue (#3B82F6), Cyan (#22D3EE)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Base Reset & Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0B1120 !important;
        color: #F8FAFC !important;
    }
    
    /* Top Header Bar */
    .top-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #151F32;
        border: 1px solid #293548;
        border-radius: 12px;
        padding: 14px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px -2px rgba(11, 17, 32, 0.5);
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .brand-icon {
        font-size: 1.6rem;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 6px 10px;
    }
    
    .brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    
    .brand-subtitle {
        font-size: 0.8rem;
        color: #94A3B8;
        margin: 0;
    }
    
    .nav-status {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #10B981;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 20px;
    }
    
    .status-pill-cyan {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(34, 211, 238, 0.1);
        border: 1px solid rgba(34, 211, 238, 0.25);
        color: #22D3EE;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 20px;
    }
    
    /* Page Titles */
    .view-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    
    .view-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }
    
    /* Metrics Summary Cards Container */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    
    @media (max-width: 900px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    @media (max-width: 600px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Standard Card Styling */
    .saas-card {
        background-color: #151F32;
        border: 1px solid #293548;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 16px -2px rgba(11, 17, 32, 0.4);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    
    .saas-card:hover {
        border-color: #3B82F6;
    }
    
    .card-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .card-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    
    .card-meta {
        font-size: 0.8rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .meta-blue { color: #3B82F6; }
    .meta-cyan { color: #22D3EE; }
    .meta-green { color: #10B981; }
    .meta-yellow { color: #F59E0B; }
    .meta-red { color: #EF4444; }
    
    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .badge-safe {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-info {
        background-color: rgba(34, 211, 238, 0.15);
        color: #22D3EE;
        border: 1px solid rgba(34, 211, 238, 0.3);
    }
    
    /* Recommendation & Action Blocks */
    .action-block {
        background-color: #151F32;
        border: 1px solid #293548;
        border-left: 4px solid #3B82F6;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .action-block-cyan {
        border-left-color: #22D3EE;
    }
    
    .action-block-green {
        border-left-color: #10B981;
    }
    
    .action-block-red {
        border-left-color: #EF4444;
    }
    
    .action-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    
    .action-desc {
        font-size: 0.85rem;
        color: #94A3B8;
        line-height: 1.45;
        margin: 0;
    }
    
    /* Script / Code Callout Box */
    .script-box {
        background-color: #0E1626;
        border: 1px solid #293548;
        border-radius: 10px;
        padding: 14px 18px;
        color: #E2E8F0;
        font-size: 0.88rem;
        line-height: 1.5;
        font-style: italic;
        margin: 12px 0;
    }
    
    /* Sidebar Overrides */
    section[data-testid="stSidebar"] {
        background-color: #151F32 !important;
        border-right: 1px solid #293548 !important;
    }
    
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }
    
    /* Streamlit Widget Styling (Inputs, Sliders, Selectboxes) */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background-color: #0E1626 !important;
        border-color: #293548 !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }
    
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {
        color: #F8FAFC !important;
    }
    
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }
    
    /* Streamlit Buttons */
    div.stButton > button {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 10px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    }
    
    div.stButton > button:hover {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        transform: translateY(-1px) !important;
    }
    
    div.stDownloadButton > button {
        background-color: #151F32 !important;
        color: #22D3EE !important;
        border: 1px solid #293548 !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
    }
    
    div.stDownloadButton > button:hover {
        background-color: rgba(34, 211, 238, 0.08) !important;
        border-color: #22D3EE !important;
    }
    
    /* Form Labels */
    label, p, span {
        color: #F8FAFC;
    }
    
    div[data-testid="stWidgetLabel"] p {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #CBD5E1 !important;
    }
    
    /* Slider Customization */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #3B82F6 !important;
        border: 2px solid #FFFFFF !important;
    }
    
    /* Radio Option List */
    div[role="radiogroup"] label {
        background-color: rgba(14, 22, 38, 0.6);
        border: 1px solid #293548;
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 6px;
        transition: all 0.15s ease;
    }
    
    div[role="radiogroup"] label:hover {
        border-color: #3B82F6;
        background-color: rgba(59, 130, 246, 0.08);
    }
    
    /* Dataframe Container */
    div[data-testid="stDataFrame"] {
        background-color: #151F32 !important;
        border: 1px solid #293548 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* Progress Bar */
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #3B82F6 !important;
    }
    
    /* Clean Divider */
    hr {
        border-color: #293548 !important;
        margin: 20px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data & Model Loader
# ---------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "churn_model.joblib")
METRICS_PATH = os.path.join(PROJECT_DIR, "model_metrics.json")
DEMO_DATA_PATH = os.path.join(PROJECT_DIR, "demo_customers.csv")

@st.cache_resource
def load_model_assets():
    """Loads the model pipeline and metrics. Generates them on the fly if missing."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
        with st.spinner("Model assets not found. Training model pipeline now on realistic banking data..."):
            from train_model import generate_synthetic_data, train_and_save_pipeline
            df = generate_synthetic_data(10000)
            train_and_save_pipeline(df, PROJECT_DIR)
            
    pipeline = joblib.load(MODEL_PATH)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        
    return pipeline, metrics

try:
    pipeline, metrics = load_model_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

@st.cache_data
def get_demo_data():
    if os.path.exists(DEMO_DATA_PATH):
        return pd.read_csv(DEMO_DATA_PATH)
    else:
        from train_model import generate_synthetic_data
        return generate_synthetic_data(200)

demo_df = get_demo_data()

# ---------------------------------------------------------
# Reusable Plotly Theme Helper (Midnight Navy & Electric Blue)
# ---------------------------------------------------------
def apply_saas_theme(fig, height=320):
    fig.update_layout(
        paper_bgcolor="#151F32",
        plot_bgcolor="#151F32",
        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12),
        height=height,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#94A3B8")
        )
    )
    fig.update_xaxes(
        gridcolor="#293548",
        zerolinecolor="#293548",
        tickfont=dict(color="#94A3B8")
    )
    fig.update_yaxes(
        gridcolor="#293548",
        zerolinecolor="#293548",
        tickfont=dict(color="#94A3B8")
    )
    return fig

# ---------------------------------------------------------
# Sidebar Navigation & System Specs
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #293548; margin-bottom: 18px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 8px; padding: 6px 10px; font-size: 1.3rem;">🏦</div>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">Helix Bank</h3>
                <p style="margin: 0; font-size: 0.76rem; color: #22D3EE; font-weight: 500;">Predictive Retention AI</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>Navigation Menu</p>", unsafe_allow_html=True)
    
    app_mode = st.radio(
        "Select Panel",
        [
            "📊 Portfolio Overview",
            "🔮 Churn Simulator",
            "🚀 Batch Predictor",
            "🧠 Model Performance"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model Specs Sidebar Widget
    acc_val = metrics["accuracy"] * 100
    auc_val = metrics["roc_curve"]["auc"]
    st.markdown(f"""
    <div class="saas-card" style="padding: 14px; margin-top: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">Active Pipeline</span>
            <span class="status-pill" style="font-size: 0.7rem; padding: 2px 8px;">Production</span>
        </div>
        <div style="font-size: 0.88rem; font-weight: 600; color: #F8FAFC; margin-bottom: 4px;">Random Forest Ensemble</div>
        <div style="font-size: 0.78rem; color: #94A3B8;">Accuracy: <strong style="color:#10B981;">{acc_val:.2f}%</strong></div>
        <div style="font-size: 0.78rem; color: #94A3B8;">ROC AUC: <strong style="color:#22D3EE;">{auc_val:.4f}</strong></div>
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 8px;">BCA Minor Project • 2026</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Global Top Header Bar
# ---------------------------------------------------------
st.markdown(f"""
<div class="top-navbar">
    <div class="nav-brand">
        <div class="brand-icon">🏦</div>
        <div>
            <h1 class="brand-title">Customer Churn Intelligence Platform</h1>
            <p class="brand-subtitle">Automated Early Warning & Behavioral Retention System for Retail Banking</p>
        </div>
    </div>
    <div class="nav-status">
        <div class="status-pill">
            <span style="width: 7px; height: 7px; background-color: #10B981; border-radius: 50%;"></span>
            System Online
        </div>
        <div class="status-pill-cyan">
            Accuracy: {metrics['accuracy']*100:.1f}%
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Feature Panel 1: Executive Portfolio Overview
# =========================================================
if app_mode == "📊 Portfolio Overview":
    st.markdown("<h2 class='view-title'>Executive Portfolio Risk Monitor</h2>", unsafe_allow_html=True)
    st.markdown("<p class='view-subtitle'>High-level visibility into customer retention dynamics, account exposures, and regional behavioral flags.</p>", unsafe_allow_html=True)
    
    # Dynamic KPI stats from dataset summary
    total_cust = metrics["dataset_summary"]["total_samples"]
    total_churned = metrics["dataset_summary"]["total_churn"]
    churn_rate = metrics["dataset_summary"]["churn_rate"]
    active_rate = metrics["dataset_summary"].get("active_rate", 0.52)
    est_capital_risk = (total_churned * 72400) / 1_000_000 # Realistic avg deposit balance per churned customer in millions
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="saas-card">
            <div class="card-label">Managed Portfolio <span>👥</span></div>
            <div class="card-value">{total_cust:,}</div>
            <div class="card-meta meta-blue">Active Banking Accounts</div>
        </div>
        <div class="saas-card">
            <div class="card-label">Identified Churn Risk <span>⚠️</span></div>
            <div class="card-value">{(churn_rate*100):.1f}%</div>
            <div class="card-meta meta-red">{total_churned:,} Accounts Flagged</div>
        </div>
        <div class="saas-card">
            <div class="card-label">Active E-Banking Rate <span>⚡</span></div>
            <div class="card-value">{(active_rate*100):.1f}%</div>
            <div class="card-meta meta-green">Regular App/Web Users</div>
        </div>
        <div class="saas-card">
            <div class="card-label">Capital at Exposure <span>💼</span></div>
            <div class="card-value">${est_capital_risk:.1f}M</div>
            <div class="card-meta meta-yellow">Estimated Deposit Attrition</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Portfolio Visualizations Section
    col1, col2 = st.columns(2)
    sample_df = demo_df.copy()
    sample_df["Exited_Label"] = sample_df["Exited"].map({1: "High Risk (Exit)", 0: "Retained (Stable)"})
    
    with col1:
        # Chart 1: Country-wise Exit Distribution
        geo_churn = sample_df.groupby(["Geography", "Exited_Label"]).size().reset_index(name="Count")
        fig1 = px.bar(
            geo_churn, 
            x="Geography", 
            y="Count", 
            color="Exited_Label",
            barmode="group",
            title="Customer Retention Status by Territory",
            color_discrete_map={"High Risk (Exit)": "#EF4444", "Retained (Stable)": "#3B82F6"},
            template="plotly_dark"
        )
        apply_saas_theme(fig1, height=310)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Chart 3: Age Demographic Histogram
        fig3 = px.histogram(
            sample_df, 
            x="Age", 
            color="Exited_Label",
            nbins=16,
            title="Age Demographics vs. Churn Propensity",
            color_discrete_map={"High Risk (Exit)": "#EF4444", "Retained (Stable)": "#22D3EE"},
            marginal="box",
            template="plotly_dark"
        )
        apply_saas_theme(fig3, height=310)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Chart 2: Product Count vs Churn Risk
        prod_churn = sample_df.groupby(["NumOfProducts", "Exited_Label"]).size().reset_index(name="Count")
        fig2 = px.bar(
            prod_churn, 
            x="NumOfProducts", 
            y="Count", 
            color="Exited_Label",
            barmode="stack",
            title="Account Retention by Number of Products Held",
            color_discrete_map={"High Risk (Exit)": "#EF4444", "Retained (Stable)": "#10B981"},
            template="plotly_dark"
        )
        apply_saas_theme(fig2, height=310)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Chart 4: Balance Distribution Box Plot
        fig4 = px.box(
            sample_df,
            x="Exited_Label",
            y="Balance",
            color="Exited_Label",
            title="Deposit Balance Distribution by Churn Status",
            color_discrete_map={"High Risk (Exit)": "#EF4444", "Retained (Stable)": "#3B82F6"},
            template="plotly_dark"
        )
        apply_saas_theme(fig4, height=310)
        st.plotly_chart(fig4, use_container_width=True)

    # Customer Priority Monitor Table with interactive search
    st.markdown("---")
    st.markdown("<h3 style='font-size:1.2rem; font-weight:700; color:#F8FAFC; margin-bottom:4px;'>⚠️ High-Priority Customer Watchlist</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.85rem; color:#94A3B8; margin-bottom:14px;'>Customer profiles displaying multiple critical indicators requiring immediate proactive outreach:</p>", unsafe_allow_html=True)
    
    # Filter controls for the table
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        country_filter = st.selectbox("Filter Territory", ["All Territories", "France", "Germany", "Spain"])
    with f_col2:
        search_query = st.text_input("Search Customer Surname / ID", placeholder="Type surname or customer ID...")
        
    filtered_df = sample_df[sample_df["Exited"] == 1].copy()
    if country_filter != "All Territories":
        filtered_df = filtered_df[filtered_df["Geography"] == country_filter]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["Surname"].str.lower().str.contains(q) | 
            filtered_df["CustomerId"].astype(str).str.contains(q)
        ]
        
    display_table = filtered_df[["CustomerId", "Surname", "Geography", "Gender", "Age", "Balance", "NumOfProducts", "CreditScore", "IsActiveMember"]].head(8).copy()
    display_table["Balance"] = display_table["Balance"].map("${:,.2f}".format)
    display_table["IsActiveMember"] = display_table["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    
    st.dataframe(display_table, use_container_width=True, hide_index=True)

# =========================================================
# Feature Panel 2: Churn Simulator (Single Customer)
# =========================================================
elif app_mode == "🔮 Churn Simulator":
    st.markdown("<h2 class='view-title'>Single Customer Risk Simulator</h2>", unsafe_allow_html=True)
    st.markdown("<p class='view-subtitle'>Input demographic, financial, and product parameters to generate an instant real-time risk profile and personalized retention recommendations.</p>", unsafe_allow_html=True)
    
    sim_col1, sim_col2 = st.columns([1, 1.2], gap="large")
    
    with sim_col1:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#F8FAFC; font-weight:700; margin-bottom:16px;'>👤 Customer Parameters</h4>", unsafe_allow_html=True)
        
        c_name = st.text_input("Customer Name / Identifier", value="Alexander Wright")
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            c_age = st.slider("Customer Age", min_value=18, max_value=90, value=48, help="Older demographics (>42) statistically demonstrate higher bank switching rates.")
            c_score = st.slider("Credit Rating Score", min_value=350, max_value=850, value=580)
            c_geo = st.selectbox("Territory (Country)", ["France", "Germany", "Spain"])
            c_gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        with c_sub2:
            c_tenure = st.slider("Tenure with Bank (Years)", min_value=0, max_value=10, value=4)
            c_balance = st.number_input("Current Deposit Balance ($)", min_value=0.0, value=85000.0, step=5000.0)
            c_products = st.selectbox("Number of Active Products Held", [1, 2, 3, 4], index=2, help="Holders of 3+ products show elevated friction and complaints.")
            c_salary = st.number_input("Estimated Annual Salary ($)", min_value=5000.0, value=92000.0, step=5000.0)
            
        st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#CBD5E1; margin-top:12px; margin-bottom:6px;'>Account Engagement Options</p>", unsafe_allow_html=True)
        cb1, cb2 = st.columns(2)
        with cb1:
            c_card = st.checkbox("Holds Helix Credit Card", value=True)
        with cb2:
            c_active = st.checkbox("Active Digital Member (Monthly)", value=False)
            
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        run_sim = st.button("⚡ Run Predictive Assessment", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with sim_col2:
        if run_sim:
            input_data = pd.DataFrame({
                "CreditScore": [c_score],
                "Geography": [c_geo],
                "Gender": [c_gender],
                "Age": [c_age],
                "Tenure": [c_tenure],
                "Balance": [c_balance],
                "NumOfProducts": [c_products],
                "HasCrCard": [1 if c_card else 0],
                "IsActiveMember": [1 if c_active else 0],
                "EstimatedSalary": [c_salary]
            })
            
            # Predict churn probability
            churn_prob = pipeline.predict_proba(input_data)[0][1]
            prob_pct = churn_prob * 100
            
            st.markdown(f"<h3 style='font-size:1.3rem; font-weight:700; color:#F8FAFC;'>Assessment Results: {c_name}</h3>", unsafe_allow_html=True)
            
            # Sleek Gauge Chart matching Electric Blue / Midnight Navy system
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Calculated Churn Probability", 'font': {'size': 15, 'color': '#94A3B8', 'family': 'Inter'}},
                number={'suffix': "%", 'font': {'color': '#F8FAFC', 'size': 38, 'family': 'Inter'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#293548"},
                    'bar': {'color': '#3B82F6', 'thickness': 0.28},
                    'bgcolor': "#151F32",
                    'borderwidth': 1,
                    'bordercolor': "#293548",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                    ]
                }
            ))
            
            gauge_fig.update_layout(
                paper_bgcolor='#151F32', 
                plot_bgcolor='#151F32',
                margin=dict(l=25, r=25, t=40, b=10),
                height=220
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Risk Category Classification Chip
            if churn_prob >= 0.70:
                risk_badge = "<span class='badge badge-critical'>🚨 Critical Churn Threat (>70%)</span>"
            elif churn_prob >= 0.40:
                risk_badge = "<span class='badge badge-warning'>⚠️ Moderate Attrition Risk (40–70%)</span>"
            else:
                risk_badge = "<span class='badge badge-safe'>🟢 Stable / Retained Tier (<40%)</span>"
                
            st.markdown(f"<div style='text-align:center; margin-bottom:18px;'>Classification: {risk_badge}</div>", unsafe_allow_html=True)
            
            # Detailed Risk Factor Attribution
            st.markdown("<h4 style='font-size:1.05rem; font-weight:700; color:#F8FAFC; margin-bottom:10px;'>⚡ Behavioral Factor Attribution</h4>", unsafe_allow_html=True)
            
            risk_factors = []
            if c_active == False:
                risk_factors.append(("💤 Inactive Banking Status", "No digital or branch transactions detected in the last 30 days. High correlation with account dormancy and competitor migration.", "red"))
            if c_products >= 3:
                risk_factors.append(("⚠️ Product Saturation Complication", f"Customer maintains {c_products} separate banking products. Multi-product accounts experience elevated operational friction and fee compounding.", "red"))
            if c_age > 42:
                risk_factors.append(("👴 Mature Demographic Risk", f"Customer is aged {c_age}. Customers over 42 exhibit higher wealth re-allocation to alternative retirement/fixed funds.", "yellow"))
            if c_geo == "Germany":
                risk_factors.append(("🇩🇪 Regional Territory Elevation", "German retail banking market currently exhibits elevated customer switching velocity compared to France and Spain.", "yellow"))
            if c_score < 540:
                risk_factors.append(("📉 Sub-optimal Credit Health", f"Credit score ({c_score}) is below standard prime levels, creating sensitivity to borrowing costs.", "yellow"))
            if c_balance > 80000 and c_active == False:
                risk_factors.append(("💰 High Idle Capital", f"Significant balance (${c_balance:,.0f}) is left idle without active banking engagement, making customer a prime target for rival yields.", "cyan"))
                
            if len(risk_factors) == 0:
                st.markdown("<p style='color:#10B981;'>🟢 No critical risk flags detected. Account health indicators are positive.</p>", unsafe_allow_html=True)
            else:
                for factor_title, factor_desc, color_tone in risk_factors:
                    border_cls = "action-block-red" if color_tone == "red" else ("action-block-cyan" if color_tone == "cyan" else "action-block")
                    st.markdown(f"""
                    <div class="action-block {border_cls}">
                        <div class="action-title">{factor_title}</div>
                        <p class="action-desc">{factor_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Retention Action Plan
            st.markdown("<h4 style='font-size:1.05rem; font-weight:700; color:#F8FAFC; margin-top:20px; margin-bottom:10px;'>💡 Recommended Retention Plan</h4>", unsafe_allow_html=True)
            if churn_prob >= 0.40:
                st.markdown("""
                <div class="action-block action-block-cyan">
                    <div class="action-title">1. Proactive Relationship Manager Review</div>
                    <p class="action-desc">Schedule a priority consultation within 48 hours. Offer fee consolidation across multi-product holdings to resolve billing friction.</p>
                </div>
                <div class="action-block action-block-green">
                    <div class="action-title">2. Yield Optimization & Deposit Lock</div>
                    <p class="action-desc">Incentivize retention of idle capital with a customized 12-month High-Yield Fixed Term Certificate (+1.8% APY).</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="action-block action-block-green">
                    <div class="action-title">Standard Service Maintenance</div>
                    <p class="action-desc">Account is stable. Continue normal automated engagement and evaluate for cross-selling premier credit card perks.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="saas-card" style="display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:360px; text-align:center;">
                <div style="font-size: 2.5rem; margin-bottom: 12px; color:#3B82F6;">🔮</div>
                <h4 style="color:#F8FAFC; font-weight:600; margin-bottom:6px;">Ready for Simulation</h4>
                <p style="color:#94A3B8; font-size:0.88rem; max-width:380px;">Configure the customer's behavioral and financial profile on the left and click 'Run Predictive Assessment' to generate full risk telemetry.</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# Feature Panel 3: Batch Predictor (CSV Processing)
# =========================================================
elif app_mode == "🚀 Batch Predictor":
    st.markdown("<h2 class='view-title'>Batch Customer Churn Processor</h2>", unsafe_allow_html=True)
    st.markdown("<p class='view-subtitle'>Upload transactional customer data files to calculate churn probabilities at scale and download scored portfolios.</p>", unsafe_allow_html=True)
    
    b_col1, b_col2 = st.columns([1, 1.8], gap="large")
    
    with b_col1:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#F8FAFC; font-weight:700; margin-bottom:12px;'>📂 Ingest Customer Dataset</h4>", unsafe_allow_html=True)
        
        uploaded_csv = st.file_uploader("Upload CSV formatted spreadsheet", type=["csv"])
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#CBD5E1; margin-bottom:8px;'>Template & Demo Data</p>", unsafe_allow_html=True)
        
        # Download Template
        template_df = demo_df.drop(columns=["Exited"]).head(10).copy()
        csv_template_bytes = template_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Download Sample Input Template (CSV)",
            data=csv_template_bytes,
            file_name="helix_bank_churn_input_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
        run_demo_button = st.button("⚡ Test with Live 200 Customer Demo Records", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with b_col2:
        df_to_run = None
        if uploaded_csv is not None:
            try:
                df_to_run = pd.read_csv(uploaded_csv)
                st.success(f"File loaded successfully: {len(df_to_run):,} customer records found.")
            except Exception as ex:
                st.error(f"Error parsing uploaded file: {ex}")
        elif run_demo_button:
            df_to_run = demo_df.drop(columns=["Exited"]).copy()
            st.info("Loaded demo dataset containing 200 banking customer profiles.")
            
        if df_to_run is not None:
            required_features = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
            missing_cols = [col for col in required_features if col not in df_to_run.columns]
            
            if len(missing_cols) > 0:
                st.error(f"Missing required columns in dataset: {missing_cols}. Please use the template format.")
            else:
                with st.spinner("Processing records through Random Forest Pipeline..."):
                    features_subset = df_to_run[required_features].copy()
                    batch_probs = pipeline.predict_proba(features_subset)[:, 1]
                    
                    results_df = df_to_run.copy()
                    results_df["Churn_Probability"] = np.round(batch_probs * 100, 2)
                    results_df["Risk_Category"] = np.where(batch_probs >= 0.70, "CRITICAL", np.where(batch_probs >= 0.40, "MODERATE", "STABLE"))
                    
                # Summary KPIs
                crit_count = (results_df["Risk_Category"] == "CRITICAL").sum()
                mod_count = (results_df["Risk_Category"] == "MODERATE").sum()
                safe_count = (results_df["Risk_Category"] == "STABLE").sum()
                
                st.markdown(f"""
                <div class="metric-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom:18px;">
                    <div class="saas-card" style="padding:14px;">
                        <div class="card-label">Critical Threat</div>
                        <div class="card-value meta-red">{crit_count}</div>
                        <div class="card-meta">Immediate Action Required</div>
                    </div>
                    <div class="saas-card" style="padding:14px;">
                        <div class="card-label">Moderate Risk</div>
                        <div class="card-value meta-yellow">{mod_count}</div>
                        <div class="card-meta">Automated Nudges</div>
                    </div>
                    <div class="saas-card" style="padding:14px;">
                        <div class="card-label">Stable / Healthy</div>
                        <div class="card-value meta-green">{safe_count}</div>
                        <div class="card-meta">Retained Accounts</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk Breakdown Donut Chart
                chart_data = pd.DataFrame({
                    "Category": ["CRITICAL", "MODERATE", "STABLE"],
                    "Count": [crit_count, mod_count, safe_count]
                })
                fig_donut = px.pie(
                    chart_data, 
                    values="Count", 
                    names="Category", 
                    hole=0.55,
                    title="Portfolio Risk Classification Breakdown",
                    color="Category",
                    color_discrete_map={"CRITICAL": "#EF4444", "MODERATE": "#F59E0B", "STABLE": "#10B981"},
                    template="plotly_dark"
                )
                apply_saas_theme(fig_donut, height=260)
                st.plotly_chart(fig_donut, use_container_width=True)
                
                # Download Button for Results
                export_csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Scored Portfolio with Churn Risk Values (CSV)",
                    data=export_csv,
                    file_name="helix_bank_churn_scored_results.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
                
                # Data Preview Table with search
                st.markdown("<h4 style='font-size:1rem; font-weight:600; color:#F8FAFC; margin-top:20px; margin-bottom:8px;'>Scored Customer Records Preview</h4>", unsafe_allow_html=True)
                display_cols = ["CustomerId", "Surname", "Geography", "Age", "Balance", "NumOfProducts", "Churn_Probability", "Risk_Category"]
                valid_cols = [c for c in display_cols if c in results_df.columns]
                st.dataframe(results_df[valid_cols].head(25), use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div class="saas-card" style="display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:360px; text-align:center;">
                <div style="font-size: 2.5rem; margin-bottom: 12px; color:#3B82F6;">🚀</div>
                <h4 style="color:#F8FAFC; font-weight:600; margin-bottom:6px;">Batch Engine Ready</h4>
                <p style="color:#94A3B8; font-size:0.88rem; max-width:380px;">Upload a banking customer CSV or click 'Test with Live 200 Customer Demo Records' to execute batch predictions across entire portfolios.</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# Feature Panel 4: Model Performance & Explainability
# =========================================================
elif app_mode == "🧠 Model Performance":
    st.markdown("<h2 class='view-title'>Model Architecture & Performance Telemetry</h2>", unsafe_allow_html=True)
    st.markdown("<p class='view-subtitle'>Detailed statistical evaluation of the trained Random Forest ensemble on the stratified test split.</p>", unsafe_allow_html=True)
    
    # Technical scorecard
    m_acc = metrics["accuracy"] * 100
    m_prec = metrics["precision"] * 100
    m_rec = metrics["recall"] * 100
    m_f1 = metrics["f1_score"] * 100
    m_auc = metrics["roc_curve"]["auc"]
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="saas-card">
            <div class="card-label">Accuracy <span>🎯</span></div>
            <div class="card-value meta-green">{m_acc:.2f}%</div>
            <div class="card-meta">Overall Classification Rate</div>
        </div>
        <div class="saas-card">
            <div class="card-label">Precision Score <span>🔍</span></div>
            <div class="card-value meta-blue">{m_prec:.2f}%</div>
            <div class="card-meta">Positive Predictive Value</div>
        </div>
        <div class="saas-card">
            <div class="card-label">Recall / Sensitivity <span>⚡</span></div>
            <div class="card-value meta-cyan">{m_rec:.2f}%</div>
            <div class="card-meta">True Positive Capture Rate</div>
        </div>
        <div class="saas-card">
            <div class="card-label">F1-Score / Balanced Mean <span>📊</span></div>
            <div class="card-value meta-green">{m_f1:.2f}%</div>
            <div class="card-meta">Harmonic Precision-Recall</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        # Confusion Matrix Heatmap
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Retained (0)", "Actual Churned (1)"],
            columns=["Predicted Retained (0)", "Predicted Churned (1)"]
        )
        
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale=[[0, "#0E1626"], [0.5, "#1E3A8A"], [1, "#3B82F6"]],
            title="Confusion Matrix (Stratified Test Split)",
            labels=dict(x="Predicted Retention Class", y="Actual Verified Status", color="Count"),
            template="plotly_dark"
        )
        apply_saas_theme(fig_cm, height=330)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # ROC Curve Plot
        roc_data = metrics["roc_curve"]
        fig_roc = px.line(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            title=f"Receiver Operating Characteristic (ROC) • AUC = {m_auc:.4f}",
            labels={"x": "False Positive Rate (1 - Specificity)", "y": "True Positive Rate (Sensitivity)"},
            template="plotly_dark"
        )
        fig_roc.update_traces(line_color="#3B82F6", line_width=3)
        fig_roc.add_shape(
            type="line", 
            line=dict(dash='dash', color='#64748B', width=1.5),
            x0=0, x1=1, y0=0, y1=1
        )
        apply_saas_theme(fig_roc, height=330)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_eval2:
        # Feature Importance Horizontal Bar Chart
        feat_df = pd.DataFrame(metrics["feature_importances"])
        feat_df["feature"] = feat_df["feature"].replace({
            "NumOfProducts": "Product Count Held",
            "IsActiveMember": "Active Member Status",
            "Age": "Customer Age",
            "Balance": "Account Balance",
            "CreditScore": "Credit Score",
            "EstimatedSalary": "Salary Estimate",
            "Tenure": "Tenure (Years with Bank)",
            "HasCrCard": "Credit Card Ownership",
            "Geography_Germany": "Country: Germany",
            "Geography_France": "Country: France",
            "Geography_Spain": "Country: Spain",
            "Gender_Female": "Gender: Female",
            "Gender_Male": "Gender: Male",
        })
        
        fig_feat = px.bar(
            feat_df.head(10).sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance Ranking (Gini Impurity Metric)",
            color="importance",
            color_continuous_scale=[[0, "#1E3A8A"], [1, "#22D3EE"]],
            template="plotly_dark"
        )
        apply_saas_theme(fig_feat, height=690)
        st.plotly_chart(fig_feat, use_container_width=True)


