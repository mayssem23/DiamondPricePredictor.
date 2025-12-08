import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------
# PAGE SETTINGS
# -------------------------------------------
st.set_page_config(
    page_title="Diamond Price App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode via custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #d50816;
        color: #ffffff;
        font-weight: bold;
    }
    .stSlider>div>div>div>div>div {
        color: #d50816;
    }
    .stSelectbox>div>div>div>div {
        color: #d50816;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #d50816;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------
# LOAD MODELS
# -------------------------------------------
best_xgb = joblib.load("best_xgb_model.pkl")
best_lgb = joblib.load("best_lgb_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# -------------------------------------------
# LOAD DATASET
# -------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diamonds.csv")
    return df

try:
    df = load_data()
except:
    df = None

# -------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Data Dashboard", "Price Prediction"]
)

# -------------------------------------------
# PAGE 1: DATA DASHBOARD
# -------------------------------------------
if page == "Data Dashboard":

    st.title("Diamond Dataset Dashboard")

    if df is not None:
        # Interactive plots
        st.subheader("Interactive Scatter Plot: Carat vs Price")
        fig1 = px.scatter(df, x="carat", y="price", color="clarity", hover_data=["cut", "color"])
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Distribution of Diamond Prices")
        fig2 = px.histogram(df, x="price", nbins=50, color="cut", marginal="box")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Correlation Heatmap")
        fig3 = px.imshow(df.corr(), text_auto=True, color_continuous_scale='reds')
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Price vs Depth by Color")
        fig4 = px.scatter(df, x="depth", y="price", color="color", size="carat", hover_data=["cut", "clarity"])
        st.plotly_chart(fig4, use_container_width=True)

        # Always show dataset image if available
        st.subheader("Dataset Images")
        images = ["1.png","2.png","3.png","4.png","5.png","6.png","7.png"]
        for img in images:
            try:
                st.image(img, use_column_width=True)
            except:
                pass

# -------------------------------------------
# PAGE 2: PRICE PREDICTION
# -------------------------------------------
elif page == "Price Prediction":

    st.title("Diamond Price Prediction")

    # ----------------- INPUTS -----------------
    col1, col2, col3 = st.columns(3)
    with col1:
        carat = st.slider("Carat", 0.2, 5.0, 1.0)
        depth = st.slider("Depth (%)", 55.0, 70.0, 61.0)
    with col2:
        cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"])
        table = st.slider("Table (%)", 50.0, 70.0, 57.0)
    with col3:
        color = st.selectbox("Color", ["J","I","H","G","F","E","D"])
        clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

    # Prepare dataframe
    df_input = pd.DataFrame([{
        "carat": carat, "cut": cut, "color": color, "clarity": clarity,
        "depth": depth, "table": table
    }])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model_cols, fill_value=0)

    # ----------------- PREDICT -----------------
    if st.button("Predict Price"):

        lgb_price = best_lgb.predict(df_input)[0]
        xgb_price = best_xgb.predict(df_input)[0]

        st.subheader("Prediction Results")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**LightGBM Model**")
            st.markdown(f"<span style='color:#d50816;font-size:18px;'>Predicted Price: ${lgb_price:,.2f}</span>", unsafe_allow_html=True)
        with colB:
            st.markdown("**XGBoost Model**")
            st.markdown(f"<span style='color:#d50816;font-size:18px;'>Predicted Price: ${xgb_price:,.2f}</span>", unsafe_allow_html=True)

        # ----------------- MODEL ACCURACY -----------------
        if df is not None:
            X = pd.get_dummies(df[["carat","cut","color","clarity","depth","table"]])
            X = X.reindex(columns=model_cols, fill_value=0)
            y = df["price"]

            def get_metrics(model):
                preds = model.predict(X)
                return (
                    mean_squared_error(y, preds, squared=False),
                    mean_absolute_error(y, preds),
                    r2_score(y, preds)
                )

            rmse_lgb, mae_lgb, r2_lgb = get_metrics(best_lgb)
            rmse_xgb, mae_xgb, r2_xgb = get_metrics(best_xgb)

            st.subheader("Model Accuracy")
            st.markdown(f"<span style='color:#d50816;'>LightGBM:</span> RMSE: {rmse_lgb:.2f}, MAE: {mae_lgb:.2f}, R²: {r2_lgb:.3f}", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#d50816;'>XGBoost:</span> RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}, R²: {r2_xgb:.3f}", unsafe_allow_html=True)

        # ----------------- SHAP -----------------
        st.subheader("Feature Importance (LightGBM)")
        explainer = shap.TreeExplainer(best_lgb)
        shap_values = explainer.shap_values(df_input)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
        st.pyplot(fig)
