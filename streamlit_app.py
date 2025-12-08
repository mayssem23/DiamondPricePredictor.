import streamlit as st
import pandas as pd
import joblib
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
        price_pred = lightgbm_tuned.predict(df_input)[0]

        st.subheader("Prediction Result")
        fig = go.Figure(go.Bar(
            x=["Predicted Price"], y=[price_pred],
            marker_color="darkred"
        ))
        fig.update_layout(template="plotly_dark", title="Predicted Diamond Price")
        st.plotly_chart(fig, use_container_width=True)

        # MODEL ACCURACY
        if df is not None:
            X = pd.get_dummies(df[["carat","cut","color","clarity","depth","table"]])
            X = X.reindex(columns=model_cols, fill_value=0)
            y = df["price"]

            rmse = mean_squared_error(y, lightgbm_tuned.predict(X), squared=False)
            mae = mean_absolute_error(y, lightgbm_tuned.predict(X))
            r2 = r2_score(y, lightgbm_tuned.predict(X))

            st.subheader("Model Accuracy (LightGBM Tuned)")
            st.markdown(f"- **RMSE:** {rmse:.2f}")
            st.markdown(f"- **MAE:** {mae:.2f}")
            st.markdown(f"- **R² Score:** {r2:.2f}")

        # FEATURE IMPORTANCE (Safe alternative to SHAP)
        st.subheader("Feature Importance (LightGBM)")
        importance_df = pd.DataFrame({
            "Feature": df_input.columns,
            "Importance": lightgbm_tuned.feature_importances_
        })
        fig_imp = px.bar(
            importance_df, x="Feature", y="Importance",
            color="Importance", color_continuous_scale="Reds",
            template="plotly_dark", title="Feature Importance"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
