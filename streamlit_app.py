import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Diamond Price Prediction App", layout="wide")

# -----------------------
# Load models + columns
# -----------------------
best_xgb = joblib.load("best_xgb_model.pkl")
best_lgb = joblib.load("best_lgb_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# -----------------------
# Sidebar navigation
# -----------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["💎 Prediction", "📊 Data & Model Info"])

# ============================================================
#                         PAGE 1 — PREDICTION
# ============================================================
if page == "💎 Prediction":

    st.title("💎 Diamond Price Prediction")

    # Input widgets
    carat = st.slider("Carat", 0.2, 5.0, 1.0)
    cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"])
    color = st.selectbox("Color", ["J","I","H","G","F","E","D"])
    clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])
    depth = st.slider("Depth %", 55.0, 70.0, 61.0)
    table = st.slider("Table %", 50.0, 70.0, 57.0)

    # Prepare input dataframe
    df_input = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table
    }])

    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model_cols, fill_value=0)

    # Prediction
    if st.button("Predict Price 💰"):
        # Predictions
        lgb_price = best_lgb.predict(df_input)[0]
        xgb_price = best_xgb.predict(df_input)[0]
        avg_price = (lgb_price + xgb_price) / 2

        st.subheader("📌 Predicted Prices")
        st.write(f"**LGB Prediction:** ${lgb_price:,.2f}")
        st.write(f"**XGB Prediction:** ${xgb_price:,.2f}")
        st.success(f"🎯 **Average Price:** ${avg_price:,.2f}")

        # SHAP explainability
        st.subheader("🔍 Feature Importance (SHAP)")
        explainer = shap.TreeExplainer(best_lgb)
        shap_values = explainer.shap_values(df_input)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
        st.pyplot(fig)

# ============================================================
#                    PAGE 2 — DATA & MODEL INFO
# ============================================================
else:
    st.title("📊 Dataset & Model Information")

    st.write("Upload the dataset you used for training to display statistics:")

    dataset_file = st.file_uploader("Upload CSV dataset", type=["csv"])

    if dataset_file:
        df = pd.read_csv(dataset_file)

        st.subheader("📄 Preview")
        st.write(df.head())

        st.subheader("📏 Dataset Shape")
        st.write(df.shape)

        st.subheader("📈 Statistics")
        st.write(df.describe())

        # -------------------------------
        # Model Comparison Section
        # -------------------------------
        st.header("⚔️ Model Comparison: LGB vs XGB")

        # Prepare dataset
        df_model = df.copy()
        y = df_model["price"]
        X = df_model.drop(columns=["price"])

        X = pd.get_dummies(X)
        X = X.reindex(columns=model_cols, fill_value=0)

        # Predictions
        pred_lgb = best_lgb.predict(X)
        pred_xgb = best_xgb.predict(X)

        # Metrics
        metrics = {
            "Model": ["LGB", "XGB"],
            "RMSE": [
                mean_squared_error(y, pred_lgb, squared=False),
                mean_squared_error(y, pred_xgb, squared=False)
            ],
            "MAE": [
                mean_absolute_error(y, pred_lgb),
                mean_absolute_error(y, pred_xgb)
            ],
            "R² Score": [
                r2_score(y, pred_lgb),
                r2_score(y, pred_xgb)
            ]
        }

        metrics_df = pd.DataFrame(metrics)
        st.subheader("📊 Accuracy Metrics")
        st.dataframe(metrics_df)

        # -------------------------------
        # Plot comparison
        # -------------------------------
        st.subheader("📉 RMSE Comparison Plot")

        fig, ax = plt.subplots()
        ax.bar(["LGB", "XGB"], metrics_df["RMSE"])
        ax.set_ylabel("RMSE (lower = better)")
        st.pyplot(fig)
