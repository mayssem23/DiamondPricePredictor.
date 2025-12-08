import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------
# PAGE SETTINGS
# -------------------------------------------
st.set_page_config(
    page_title="Diamond Price App",
    layout="wide"
)

# Dark theme + red accent
st.markdown("""
<style>
body {background-color: #121212; color: #ffffff;}
h1,h2,h3,h4 {color: #d50816;}
.stButton>button {background-color: #d50816; color: #ffffff; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# LOAD MODELS
# -------------------------------------------
lightgbm_tuned = joblib.load("lightgbm_tuned_model.pkl")
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
st.sidebar.markdown("## Navigation")
page = st.sidebar.selectbox(
    "Go to page:",
    ["Data Dashboard", "Price Prediction"]
)

# -------------------------------------------
# PAGE 1: DATA DASHBOARD
# -------------------------------------------
if page == "Data Dashboard":
    st.title("Diamond Dataset Dashboard")

    if df is not None:
        # Filter options
        st.sidebar.subheader("Filters")
        cut_filter = st.sidebar.multiselect("Select Cut", df["cut"].unique(), default=df["cut"].unique())
        color_filter = st.sidebar.multiselect("Select Color", df["color"].unique(), default=df["color"].unique())
        clarity_filter = st.sidebar.multiselect("Select Clarity", df["clarity"].unique(), default=df["clarity"].unique())
        
        filtered_df = df[df["cut"].isin(cut_filter) & df["color"].isin(color_filter) & df["clarity"].isin(clarity_filter)]

        # Scatter plot: Carat vs Price
        st.subheader("Carat vs Price")
        fig1 = px.scatter(
            filtered_df, x="carat", y="price", color="clarity", hover_data=["cut", "color"],
            color_discrete_sequence=px.colors.sequential.Reds
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Price distribution
        st.subheader("Price Distribution by Cut")
        fig2 = px.histogram(
            filtered_df, x="price", nbins=50, color="cut", marginal="box",
            color_discrete_sequence=px.colors.sequential.Reds
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig3 = px.imshow(filtered_df.corr(), text_auto=True, color_continuous_scale='reds')
        st.plotly_chart(fig3, use_container_width=True)

        # Price vs Depth
        st.subheader("Price vs Depth by Color")
        fig4 = px.scatter(
            filtered_df, x="depth", y="price", color="color", size="carat",
            hover_data=["cut", "clarity"], color_discrete_sequence=px.colors.sequential.Reds
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Always show dataset images with titles
    st.subheader("Dataset Images")
    images = [
        ("Feature Distributions", "1.png"),
        ("Pearson Correlation Matrix", "2.png"),
        ("Model Comparison - RMSE (Lower is Better)", "3.png"),
        ("Model Comparison - R2 Score (Higher is Better)", "4.png"),
        ("Model Comparison - R2 Score Tuned", "5.png"),
        ("Model Comparison - RMSE Tuned", "6.png")
    ]
    for title, img in images:
        st.markdown(f"### <span style='color:#d50816'>{title}</span>", unsafe_allow_html=True)
        try:
            st.image(img, use_column_width=True)
        except:
            st.warning(f"Image {img} not found")

# -------------------------------------------
# PAGE 2: PRICE PREDICTION
# -------------------------------------------
elif page == "Price Prediction":
    st.title("Diamond Price Prediction")

    # INPUTS
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

    # PREDICT
    if st.button("Predict Price"):
        lightgbm_tuned_price = lightgbm_tuned.predict(df_input)[0]

        st.subheader("Prediction Results")
        fig_bar = go.Figure(data=[
            go.Bar(name='lightgbm_tuned', x=['Price'], y=[lightgbm_tuned_price], marker_color='darkred')
        ])
        fig_bar.update_layout(title="Price Comparison", template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

        # MODEL ACCURACY
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
            metrics_df = pd.DataFrame({
                "Model": ["LightGBM", "XGBoost"],
                "RMSE": [rmse_lgb, rmse_xgb],
                "MAE": [mae_lgb, mae_xgb],
                "R²": [r2_lgb, r2_xgb]
            })
            st.dataframe(metrics_df.style.format("{:.2f}").set_properties(**{'color':'#d50816','font-weight':'bold'}))

 # SHAP Interactive Plot
st.subheader("Feature Importance (LightGBM)")

# Use the underlying booster
explainer = shap.TreeExplainer(lightgbm_tuned.booster_)
shap_values = explainer.shap_values(df_input)

# Prepare SHAP DataFrame
shap_df = pd.DataFrame({
    "Feature": df_input.columns,
    "SHAP Value": shap_values.mean(axis=0)
})

# Plot
shap_fig = px.bar(
    shap_df, x="Feature", y="SHAP Value", color="SHAP Value",
    color_continuous_scale="Reds", template="plotly_dark"
)
st.plotly_chart(shap_fig, use_container_width=True)

