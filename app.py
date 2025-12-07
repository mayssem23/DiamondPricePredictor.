# app.py
# -----------------------------
# DIAMOND PRICE PREDICTION DASHBOARD
# -----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# LOAD MODELS AND COLUMNS
# -----------------------------
best_xgb = joblib.load("best_xgb_model.pkl")
best_lgb = joblib.load("best_lgb_model.pkl")
stack_model = joblib.load("stacked_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Diamond Price Predictor", layout="centered")
st.title("💎 Diamond Price Prediction App")
st.write("""
Predict diamond prices using advanced ML models (XGBoost, LightGBM, and Stacking).  
Enter diamond characteristics below and get an instant prediction.
""")

# -----------------------------
# INPUT WIDGETS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    carat = st.slider("Carat", min_value=0.2, max_value=5.0, value=1.0, step=0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])

with col2:
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
    depth = st.slider("Depth %", min_value=55.0, max_value=70.0, value=61.0, step=0.1)
    table = st.slider("Table %", min_value=50.0, max_value=70.0, value=57.0, step=0.1)

# -----------------------------
# FEATURE PROCESSING
# -----------------------------
input_dict = {
    "carat": carat,
    "cut": cut,
    "color": color,
    "clarity": clarity,
    "depth": depth,
    "table": table
}

df_input = pd.DataFrame([input_dict])

# One-hot encoding for categorical variables
df_input = pd.get_dummies(df_input)

# Align with model training columns
df_input = df_input.reindex(columns=model_cols, fill_value=0)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict Price 💰"):
    prediction = stack_model.predict(df_input)[0]
    st.success(f"Estimated Price: **${prediction:,.2f}**")

    # -----------------------------
    # SHAP EXPLAINABILITY
    # -----------------------------
    st.subheader("🔍 Why This Price?")
    explainer = shap.TreeExplainer(best_lgb)
    shap_values = explainer.shap_values(df_input)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
    st.pyplot(fig)

