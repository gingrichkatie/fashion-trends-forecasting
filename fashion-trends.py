import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and model
@st.cache_data
def load_data():
    return pd.read_csv("Fashion_Retail_Sales.csv")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

df = load_data()
model, model_columns = load_model()

st.set_page_config(page_title="Fashion Sales Predictor", layout="wide")

# Tabs
tabs = st.tabs(["Prediction Tool", "Model Performance", "Data Explorer"])

# ================================
# Tab 1: Prediction Tool
# ================================
with tabs[0]:
    st.title("Fashion Sales Prediction")
    st.markdown("Use the form below to forecast purchase amount (USD) based on transaction details.")

    with st.form("prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            item = st.selectbox("Item Purchased", df["Item Purchased"].unique())
            payment = st.selectbox("Payment Method", df["Payment Method"].unique())

        with col2:
            rating = st.slider("Review Rating", 1.0, 5.0, 4.0, step=0.1)

        # THIS LINE IS CRUCIAL — must be inside form
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build input vector
        input_dict = {
            "Review Rating": rating,
            f"Item Purchased_{item}": 1,
            f"Payment Method_{payment}": 1
        }

        input_df = pd.DataFrame([{col: input_dict.get(col, 0) for col in model_columns}])
        prediction = model.predict(input_df)[0]

        # Confidence interval using variation in tree predictions
        preds = [tree.predict(input_df)[0] for tree in model.estimators_]
        std_dev = np.std(preds)
        lower = prediction - 1.96 * std_dev
        upper = prediction + 1.96 * std_dev

        st.subheader("Prediction Results")
        st.write(f"Estimated Purchase Amount: ${prediction:,.2f}")
        st.write(f"95% Confidence Interval: ${lower:,.2f} to ${upper:,.2f}")

# ================================
# Tab 2: Model Performance
# ================================
with tabs[1]:
    st.title("Model Performance")
    st.write("Cross-validated metrics for each model:")

    metrics = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Linear Regression": [13.7, 18.2, 0.51],
    }

