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
