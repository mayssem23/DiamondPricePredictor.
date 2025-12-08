import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------
# Load Models
# ---------------------------------------
best_xgb = joblib.load("best_xgb_model.pkl")
best_lgb = joblib.load("best_lgb_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# ---------------------------------------
# App Layout
# ---------------------------------------
st.set_page_config(
    page_title="Diamond Price App",
    layout="wide"
)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Price Prediction", "Dataset & Insights"]
)

# ---------------------------------------
# PAGE 1: Prediction
# ---------------------------------------
if page == "Price Prediction":
    st.title("Diamond Price Prediction")

    st.markdown("""
    A simple interface to predict diamond prices using two machine-learning models:
    **LightGBM** and **XGBoost**.
    """)

    # Input columns
    col1, col2, col3 = st.columns(3)

    with col1:
        carat = st.slider("Carat", 0.2, 5.0, 1.0)
        clarity = st.selectbox("Clarity", 
                               ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])
    with col2:
        cut = st.selectbox("Cut", 
                           ["Fair","Good","Very Good","Premium","Ideal"])
        depth = st.slider("Depth %", 55.0, 70.0, 61.0)
    with col3:
        color = st.selectbox("Color", ["J","I","H","G","F","E","D"])
        table = st.slider("Table %", 50.0, 70.0, 57.0)

    # Prepare dataframe
    df_input = pd.DataFrame([{
        "carat": carat, "cut": cut,
        "color": color, "clarity": clarity,
        "depth": depth, "table": table
    }])

    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model_cols, fill_value=0)

    st.write("Input Data:")
    st.dataframe(df_input)

    # Prediction button
    if st.button("Generate Price Prediction"):
        colA, colB = st.columns(2)

        # Predict
        price_lgb = best_lgb.predict(df_input)[0]
        price_xgb = best_xgb.predict(df_input)[0]

        with colA:
            st.subheader("LightGBM Prediction")
            st.info(f"Price: ${price_lgb:,.2f}")

        with colB:
            st.subheader("XGBoost Prediction")
            st.info(f"Price: ${price_xgb:,.2f}")

        # ---------------- SHAP EXPLAINABILITY ----------------
        st.subheader("Feature Importance (SHAP)")

        explainer = shap.TreeExplainer(best_lgb)
        shap_values = explainer.shap_values(df_input)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
        st.pyplot(fig)

# ---------------------------------------
# PAGE 2: Dataset & Insights
# ---------------------------------------
if page == "Dataset & Insights":
    st.title("Dataset & Insights")

    st.markdown("""
    Explore dataset statistics, correlations and upload images for documentation.
    """)

    uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Basic Statistics")
        st.write(df.describe())

        # -------- Interactive Plot: Carat vs Price --------
        if "price" in df.columns and "carat" in df.columns:
            st.subheader("Carat vs Price (Interactive)")

            fig = px.scatter(
                df, x="carat", y="price",
                trendline="lowess",
                height=500,
                title="Carat vs Price"
            )
            st.plotly_chart(fig, use_container_width=True)

        # -------- Correlation Heatmap --------
        st.subheader("Correlation Heatmap")
        corr = df.corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Blues",
            height=600,
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------- PNG Upload Section --------
    st.subheader("Upload Images (PNG)")
    png = st.file_uploader("Upload PNG Images", type=["png"])

    if png:
        st.image(png, caption="Uploaded Image", use_column_width=True)
