import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import io

st.set_page_config(page_title="Diamond Price Predictor", layout="centered")
st.title("💎 Diamond Price Prediction App")

# -----------------------------
# Define the custom stacking class (as used in your model)
# -----------------------------
class MyStackingModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        # Replace this with your original stacking logic
        # For example, average predictions from all models
        preds = [model.predict(X) for model in self.models]
        return sum(preds) / len(preds)

# -----------------------------
# Load models
# -----------------------------
best_xgb = joblib.load("best_xgb_model.pkl")
best_lgb = joblib.load("best_lgb_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Load stacked model from Dropbox
STACKED_MODEL_URL = "https://www.dropbox.com/s/.../stacked_model_compressed.pkl?dl=1"

@st.cache_resource(show_spinner=True)
def load_stacked_model(url):
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

stack_model = load_stacked_model(STACKED_MODEL_URL)

# -----------------------------
# Input form
# -----------------------------
carat = st.slider("Carat", 0.2, 5.0, 1.0)
cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"])
color = st.selectbox("Color", ["J","I","H","G","F","E","D"])
clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])
depth = st.slider("Depth %", 55.0, 70.0, 61.0)
table = st.slider("Table %", 50.0, 70.0, 57.0)

df_input = pd.DataFrame([{
    "carat": carat, "cut": cut, "color": color, "clarity": clarity,
    "depth": depth, "table": table
}])
df_input = pd.get_dummies(df_input)
df_input = df_input.reindex(columns=model_cols, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price 💰"):
    try:
        price = stack_model.predict(df_input)[0]
        st.success(f"Estimated Price: ${price:,.2f}")

        # SHAP explainability
        explainer = shap.TreeExplainer(best_lgb)
        shap_values = explainer.shap_values(df_input)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
