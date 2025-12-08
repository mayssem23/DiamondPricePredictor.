import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------
# PAGE SETTINGS
# -------------------------------------------
st.set_page_config(
    page_title="Diamond Price App",
    layout="wide",
)

# -------------------------------------------
# LOAD MODELS
# -------------------------------------------
best_xgb = joblib.load("best_xgb_model.pkl")
best_lgb = joblib.load("best_lgb_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# -------------------------------------------
# LOAD DATASET + ALWAYS SHOW IMAGE
# -------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diamonds.csv")   # If you already have your dataset
    return df

try:
    df = load_data()
except:
    df = None

# -------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------
page = st.sidebar.selectbox(
    "Navigation",
    ["Data Dashboard", "Price Prediction"]
)

# -------------------------------------------
# PAGE 1: DATA DASHBOARD
# -------------------------------------------
if page == "Data Dashboard":

    st.title("Diamond Dataset Overview")

    # Dataset Image Always Visible
    st.subheader("Dataset Image")
    st.image("1.png", use_column_width=True)  
    st.image("2.png", use_column_width=True)
    st.image("3.png", use_column_width=True)  
    st.image("4.png", use_column_width=True)
    st.image("5.png", use_column_width=True)  
    st.image("6.png", use_column_width=True)
    st.image("7.png", use_column_width=True)  

    

 

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
            st.write("LightGBM Model")
            st.write(f"Predicted Price: **${lgb_price:,.2f}**")
        with colB:
            st.write("XGBoost Model")
            st.write(f"Predicted Price: **${xgb_price:,.2f}**")

        # ----------------- MODEL ACCURACY -----------------
        if df is not None:
            X = pd.get_dummies(df[["carat","cut","color","clarity","depth","table"]])
            X = X.reindex(columns=model_cols, fill_value=0)
            y = df["price"]

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
            st.write("LightGBM:")
            st.write(f"RMSE: {rmse_lgb:.2f}")
            st.write(f"MAE: {mae_lgb:.2f}")
            st.write(f"R²: {r2_lgb:.3f}")

            st.write("XGBoost:")
            st.write(f"RMSE: {rmse_xgb:.2f}")
            st.write(f"MAE: {mae_xgb:.2f}")
            st.write(f"R²: {r2_xgb:.3f}")

        # ----------------- SHAP -----------------
        st.subheader("Model Explainability (LightGBM)")
        explainer = shap.TreeExplainer(best_lgb)
        shap_values = explainer.shap_values(df_input)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
        st.pyplot(fig)
