# Fashion Retail Sales Forecasting

This project is a regression-based AI tool that predicts purchase amounts in USD based on fashion retail transaction data. It was built as part of a graduate course on applied machine learning and deployed as a live web app using Streamlit.

## Project Overview

In the fast fashion industry, timing is critical. Brands like Zara and H&M rely on rapid product decisions to stay competitive. The goal of this project was to build a machine learning agent that could forecast sales based on transaction-level data — helping improve inventory planning and reduce overproduction.

The app accepts user-defined inputs (like item type, rating, and payment method) and generates real-time purchase predictions with confidence intervals.

## Features

- Real-time regression predictions (Random Forest)
- Confidence intervals based on ensemble variation
- Clean and professional Streamlit interface
- Model performance dashboard (MAE, RMSE, R²)
- Data exploration tools for quick trend insights
- Fully deployed with 24/7 public access

## Models Used

- Linear Regression (baseline)
- Random Forest Regressor (final model)
- Evaluation metrics: MAE, RMSE, R²
- Cross-validation: 5-fold

## Make sure the following files are present in the same directory:
- model.pkl (trained model)
- model_columns.pkl (feature list used by the model)
- Fashion_Retail_Sales.csv (dataset used for visuals and exploration)

## Usage Guide
- Once the app is running:
- Go to the "Prediction Tool" tab
- Select an item, a review rating, and a payment method
- Click "Predict" to see the estimated purchase amount and 95% confidence interval
- Use the "Model Performance" tab to compare models
- Use the "Data Explorer" tab to explore patterns and trends in the dataset

## Files Included
- fashion-trends.py – Streamlit app interface
- model.pkl – Trained Random Forest model
- model_columns.pkl – List of model input features
- Fashion_Retail_Sales.csv – Raw dataset
- requirements.txt – Python dependencies
- README.md – Project overview and usage
