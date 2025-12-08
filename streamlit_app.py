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
    df = pd.read_csv("diamonds.csv")
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
    st.subheader("Dataset Images")
    
    # Always show dataset images
    images = ["1.png","2.png","3.png","4.png","5.png","6.png","7.png"]
    for img in images:
        st.image(img, use_column_width=True)
    
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        st.subheader("Interactive Plots")
        fig1 = px.histogram(df, x="carat", nbins=50, title="Carat Distribution",
                            color_discrete_sequence=["#d50816"])
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter(df, x="carat", y="price", color="cut",
                          title="Price vs Carat by Cut",
                          color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(df, x="color", y="price", color="clarity",
                      title="Price Distribution by Color & Clarity",
                      color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig3, use_container_width=True)

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
            try:
                X = pd.get_dummies(df[["carat","cut","color","clarity","depth","table"]])
                X = X.reindex(columns=model_cols, fill_value=0)
                y = df["price"]

                def get_metrics(model):
                    preds = model.predict(X)
                    return (
                        mean_squared_error(y, preds, squared=False),  # RMSE
                        mean_absolute_error(y, preds),               # MAE
                        r2_score(y, preds)                           # R²
                    )

                rmse_lgb, mae_lgb, r2_lgb = get_metrics(best_lgb)
                rmse_xgb, mae_xgb, r2_xgb = get_metrics(best_xgb)

                # Normalize RMSE and MAE for plotting
                max_val = max(rmse_lgb, rmse_xgb, mae_lgb, mae_xgb)
                rmse_lgb_norm = rmse_lgb / max_val
                mae_lgb_norm = mae_lgb / max_val
                rmse_xgb_norm = rmse_xgb / max_val
                mae_xgb_norm = mae_xgb / max_val

                # ----------------- Error Comparison Plot -----------------
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["RMSE","MAE"],
                    y=[rmse_lgb_norm, mae_lgb_norm],
                    name="LightGBM",
                    marker_color="#d50816"
                ))
                fig.add_trace(go.Bar(
                    x=["RMSE","MAE"],
                    y=[rmse_xgb_norm, mae_xgb_norm],
                    name="XGBoost",
                    marker_color="darkred"
                ))
                fig.update_layout(
                    title="Model Error Comparison (Normalized)",
                    template="plotly_dark",
                    yaxis_title="Normalized Value",
                    barmode="group"
                )
                st.plotly_chart(fig, use_container_width=True)

                # ----------------- R² Comparison Plot -----------------
                fig_r2 = go.Figure()
                fig_r2.add_trace(go.Bar(
                    x=["LightGBM","XGBoost"],
                    y=[r2_lgb, r2_xgb],
                    marker_color=["#d50816","darkred"]
                ))
                fig_r2.update_layout(
                    title="Model R² Comparison",
                    template="plotly_dark",
                    yaxis_title="R² Score"
                )
                st.plotly_chart(fig_r2, use_container_width=True)

            except Exception as e:
                st.error(f"Error calculating model metrics: {e}")

        # ----------------- SHAP -----------------
        st.subheader("Model Explainability (LightGBM)")
        explainer = shap.TreeExplainer(best_lgb)
        shap_values = explainer.shap_values(df_input)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
        st.pyplot(fig)
