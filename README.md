Stock Price Prediction with TPOT & Random Forest
This project explores machine learning-based stock price prediction using Yahoo Finance data, a Random Forest model, and TPOT AutoML for automated model optimization.

Project Overview
Fetches historical stock data for Apple (AAPL) using yfinance.

Creates technical indicators (daily returns, moving averages).

Builds a Random Forest model to predict if the stock price will go up or down the next day.

Uses TPOT (AutoML) to automate model selection and hyperparameter tuning.


Source: Yahoo Finance (yfinance)

Features:

Returns - Daily percentage change in stock price.

MA20 - 20-day moving average.

MA50 - 50-day moving average.

Open, High, Low, Close, Volume (for TPOT model).

Target Variable:

1 if the stock price goes up the next day.

0 if it goes down.

Ensure you have Python 3.10+ installed.


Modeling Approach
Random Forest Model
Uses Returns, MA20, and MA50 as features.

Splits data into 80% training / 20% testing.

Achieves a classification accuracy score.

TPOT AutoML
Automatically finds the best machine learning pipeline.

Tests multiple models (Random Forest, XGBoost, Logistic Regression, etc.).

Optimizes feature selection and hyperparameters.

Results
The Random Forest model gives an accuracy score.
TPOT automatically optimizes a pipeline to improve performance.
The best TPOT pipeline is exported to best_model_pipeline.py.

Strengths:

Automates model selection and hyperparameter tuning. Quick and easy implementation for beginners.

Limitations:
-No advanced financial indicators (e.g., RSI, MACD).
-TPOT can be slow on large datasets.
-Accuracy alone isn't enough for trading strategies—we need risk-adjusted returns.

Future Work:
-Add other parameters and technical indicators for better predictions.
-Test on multiple stocks to generalize performance.
-Implement risk management to make it viable for real trading.


Conclusion
This project is a starting point for using machine learning in stock market prediction. TPOT is used to automate model selection, but domain-specific feature engineering is essential for real-world trading.
