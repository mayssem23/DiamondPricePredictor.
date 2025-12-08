import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt



# Load models
best_xgb = joblib.load("best_xgb_model.pkl")
best_lgb = joblib.load("best_lgb_model.pkl")
model_cols = joblib.load("model_columns.pkl")

st.title("💎 Diamond Price Prediction App")

# Input widgets
carat = st.slider("Carat", 0.2, 5.0, 1.0)
cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"])
color = st.selectbox("Color", ["J","I","H","G","F","E","D"])
clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])
depth = st.slider("Depth %", 55.0, 70.0, 61.0)
table = st.slider("Table %", 50.0, 70.0, 57.0)

# Prepare input dataframe
df_input = pd.DataFrame([{
    "carat": carat, "cut": cut, "color": color, "clarity": clarity,
    "depth": depth, "table": table
}])
df_input = pd.get_dummies(df_input)
df_input = df_input.reindex(columns=model_cols, fill_value=0)

# Predict
if st.button("Predict Price 💰"):
    price = best_lgb.predict(df_input)[0]
    st.success(f"Estimated Price: ${price:,.2f}")

    # SHAP explainability
    explainer = shap.TreeExplainer(best_lgb)
    shap_values = explainer.shap_values(df_input)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
    st.pyplot(fig)
