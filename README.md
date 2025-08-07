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
