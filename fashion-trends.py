import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and columns
pd.read_csv("Fashion_Retail_Sales.csv")
joblib.load("model.pkl")
joblib.load("model_columns.pkl")

st.set_page_config(page_title="Fashion Sales Forecast", layout="wide")

# --- Custom CSS styling ---
st.markdown("""
    <style>
        .main { background-color: #f5f5f5; }
        .stApp { font-family: 'Segoe UI', sans-serif; }
        h1, h2, h3 { color: #202124; }
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Tab layout ---
tabs = st.tabs(["Prediction Tool", "Model Performance", "Data Exploration"])

# ========== TAB 1: Prediction ==========
with tabs[0]:
    st.title("Fashion Sales Prediction")
    st.markdown("Use the form below to forecast purchase amount (USD) based on transaction details.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            item = st.selectbox("Item Purchased", df["Item Purchased"].unique(), help="Select the type of fashion item")
            payment = st.selectbox("Payment Method", df["Payment Method"].unique(), help="Select the payment method used")

        with col2:
            rating = st.slider("Review Rating", min_value=1.0, max_value=5.0, step=0.1, value=4.0, help="Rate the item (1 to 5)")

        submitted = st.form_submit_button("Predict")

    if submitted:
        # One-hot encode user input
        input_dict = {
            "Review Rating": rating,
            f"Item Purchased_{item}": 1,
            f"Payment Method_{payment}": 1
        }
        input_df = pd.DataFrame([{col: input_dict.get(col, 0) for col in model_columns}])

        # Prediction
        prediction = model.predict(input_df)[0]

        # Confidence Interval (± std deviation from 10 trees)
        preds = [tree.predict(input_df)[0] for tree in model.estimators_]
        std_dev = np.std(preds)
        lower = prediction - 1.96 * std_dev
        upper = prediction + 1.96 * std_dev

        st.subheader("Prediction Results")
        st.write(f"Estimated Purchase Amount: **${prediction:,.2f}**")
        st.write(f"95% Confidence Interval: **${lower:,.2f} to ${upper:,.2f}**")

# ========== TAB 2: Model Performance ==========
with tabs[1]:
    st.title("Model Performance Dashboard")

    st.markdown("""
    This section summarizes the model's cross-validated performance using key metrics.
    """)

    metrics = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Linear Regression": [13.7, 18.2, 0.51],
        "Random Forest": [9.2, 12.4, 0.71]
    })

    st.dataframe(metrics.set_index("Metric"))

    st.subheader("Visual Comparison")
    melted = metrics.melt(id_vars="Metric", var_name="Model", value_name="Score")
    fig, ax = plt.subplots()
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", ax=ax)
    st.pyplot(fig)

# ========== TAB 3: Data Exploration ==========
with tabs[2]:
    st.title("Data Exploration")

    st.markdown("""
    This section provides insights into purchase patterns from the dataset.
    Use filters below to explore trends.
    """)

    # Filters
    with st.expander("Filters"):
        selected_items = st.multiselect("Filter by Item Purchased", df["Item Purchased"].unique(), default=list(df["Item Purchased"].unique()))
        selected_payments = st.multiselect("Filter by Payment Method", df["Payment Method"].unique(), default=list(df["Payment Method"].unique()))

    # Filtered data
    filtered_df = df[df["Item Purchased"].isin(selected_items) & df["Payment Method"].isin(selected_payments)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Purchase Amount by Item")
        item_avg = filtered_df.groupby("Item Purchased")["Purchase Amount (USD)"].mean().sort_values()
        st.bar_chart(item_avg)

    with col2:
        st.subheader("Review Rating Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df["Review Rating"], bins=10, kde=True, ax=ax)
        st.pyplot(fig)
