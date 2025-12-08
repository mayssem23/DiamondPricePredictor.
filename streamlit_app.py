import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# --------------------------------------------------------
# Load trained models
# --------------------------------------------------------
best_xgb = joblib.load("best_xgb_model.pkl")
best_lgb = joblib.load("best_lgb_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# --------------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction Tool", "Dataset & Visualization"])

# ========================================================
# PAGE 1 — Prediction Tool
# ========================================================
if page == "Prediction Tool":
    st.title("Diamond Price Prediction")

    st.markdown("""
    Adjust the parameters below and compare predicted prices between models.  
    Accuracy of each model appears under the prediction results.
    """)

    # -----------------------------
    # Input widgets
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        carat = st.slider("Carat", 0.2, 5.0, 1.0)
        cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
        clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

    with col2:
        color = st.selectbox("Color", ["J","I","H","G","F","E","D"])
        depth = st.slider("Depth %", 55.0, 70.0, 61.0)
        table = st.slider("Table %", 50.0, 70.0, 57.0)

    # Prepare dataframe
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

    # -----------------------------
    # Predict
    # -----------------------------
    if st.button("Generate Predictions"):
        st.subheader("Model Outputs")

        # LGB Prediction
        lgb_price = best_lgb.predict(df_input)[0]
        st.write(f"**LightGBM Price:** ${lgb_price:,.2f}")

        if hasattr(best_lgb, "score"):
            st.write(f"LightGBM Accuracy: {best_lgb.score(df_input, [lgb_price]):.3f}")

        st.markdown("---")

        # XGB Prediction
        xgb_price = best_xgb.predict(df_input)[0]
        st.write(f"**XGBoost Price:** ${xgb_price:,.2f}")

        if hasattr(best_xgb, "score"):
            st.write(f"XGBoost Accuracy: {best_xgb.score(df_input, [xgb_price]):.3f}")

        st.markdown("---")

        # SHAP — Explainability (using LGB only for clarity)
        st.subheader("Feature Contribution (SHAP)")

        explainer = shap.TreeExplainer(best_lgb)
        shap_values = explainer.shap_values(df_input)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
        st.pyplot(fig)


# ========================================================
# PAGE 2 — Dataset & Visualization
# ========================================================
elif page == "Dataset & Visualization":

    st.title("Dataset Exploration")

    st.markdown("""
    Upload your dataset and image. All visualizations update automatically.  
    The uploaded image always remains visible below.
    """)

    # ----------------------------------------------------
    # Upload image
    # ----------------------------------------------------
    st.subheader("Upload an Image (PNG/JPEG)")
    image_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

    st.markdown("---")

    # ----------------------------------------------------
    # Upload dataset
    # ----------------------------------------------------
    st.subheader("Upload Dataset (CSV)")
    data_file = st.file_uploader("Upload CSV file", type=["csv"], key="data_upload")

    if data_file:
        df = pd.read_csv(data_file)
        st.write("Dataset Preview:")
        st.dataframe(df)

        st.markdown("---")
        st.subheader("Interactive Visualizations")

        # Plot 1 — Carat Distribution
        if "carat" in df.columns:
            fig1 = px.histogram(df, x="carat", nbins=40, title="Carat Distribution")
            st.plotly_chart(fig1)

        # Plot 2 — Price vs Carat
        if "price" in df.columns and "carat" in df.columns:
            fig2 = px.scatter(df, x="carat", y="price", trendline="ols",
                              title="Carat vs Price")
            st.plotly_chart(fig2)

        # Plot 3 — Cut Distribution
        if "cut" in df.columns:
            fig3 = px.histogram(df, x="cut", color="cut", title="Cut Distribution")
            st.plotly_chart(fig3)

