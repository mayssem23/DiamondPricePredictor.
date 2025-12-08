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

# Always show dataset images
st.subheader("Dataset Images")

# List of images with titles
images = [
    ("Diamond Cut Overview", "1.png"),
    ("Diamond Color Distribution", "2.png"),
    ("Diamond Clarity Distribution", "3.png"),
    ("Carat vs Price Scatter", "4.png"),
    ("Depth vs Price Scatter", "5.png"),
    ("Table vs Price Scatter", "6.png")
]

for title, img in images:
    st.markdown(f"### <span style='color:#d50816'>{title}</span>", unsafe_allow_html=True)  # Red title
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
        lgb_price = best_lgb.predict(df_input)[0]
        xgb_price = best_xgb.predict(df_input)[0]

        st.subheader("Prediction Results")
        fig_bar = go.Figure(data=[
            go.Bar(name='LightGBM', x=['Price'], y=[lgb_price], marker_color='#d50816'),
            go.Bar(name='XGBoost', x=['Price'], y=[xgb_price], marker_color='darkred')
        ])
        fig_bar.update_layout(title="Price Comparison", template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    

        # SHAP Interactive Plot
        st.subheader("Feature Importance (LightGBM)")
        explainer = shap.TreeExplainer(best_lgb)
        shap_values = explainer.shap_values(df_input)
        shap_df = pd.DataFrame(list(zip(df_input.columns, shap_values[0])), columns=["Feature", "SHAP Value"])
        shap_fig = px.bar(shap_df, x="Feature", y="SHAP Value", color="SHAP Value", color_continuous_scale="Reds", template="plotly_dark")
        st.plotly_chart(shap_fig, use_container_width=True)
