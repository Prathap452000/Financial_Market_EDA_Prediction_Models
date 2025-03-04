# Tesla Stock Price Prediction with LSTM

This project uses an LSTM (Long Short-Term Memory) neural network to predict Tesla (TSLA) stock prices based on historical closing prices from March 04, 2012, to March 04, 2024. The model is trained, evaluated, and used to forecast prices for the next 10 days, with a comparison to actual historical prices.

## Features
- Fetches Tesla stock data using `yfinance`.
- Trains an LSTM model with 12 years of data.
- Evaluates model performance with RMSE, MAE, and R² metrics.
- Predicts stock prices for March 02-11, 2024.
- Compares predictions with actual prices.

## Prerequisites
- Python 3.7+
- Google Colab or a local Python environment

