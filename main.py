import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import matplotlib.pyplot as plt

# Function to fetch stock data using yfinance
def fetch_stock_data(stock_code, period='1d'):
    stock_data = yf.Ticker(stock_code)
    return stock_data.history(period=period)

# Define the Moving Average function
def moving_average(data, window):
    return data['Close'].rolling(window).mean()

# Define the Exponential Smoothing function
def exponential_smoothing(data, alpha):
    model = SimpleExpSmoothing(data['Close'])
    model_fit = model.fit(smoothing_level=alpha)
    return model_fit.fittedvalues

# Define the Holt-Winters function
def holt_winters(data, period):
    model = ExponentialSmoothing(data['Close'], trend='add', seasonal='add', seasonal_periods=period)
    model_fit = model.fit()
    return model_fit.fittedvalues

# Streamlit UI
st.title('Stock Forecasting Application')

# Input for stock ticker symbol
stock_code = st.text_input('Enter Stock Ticker Symbol', 'AAPL')

# Dropdown for choosing the forecasting model
model_choice = st.selectbox('Choose the Forecasting Model', ['Moving Average', 'Exponential Smoothing', 'Holt-Winters'])

# Dropdown for choosing the time frame for stock data
time_frame = st.selectbox('Choose Time Frame for Stock Data', ['1d', '1wk', '1mo'])

# Number input for model parameters
window = st.slider('Moving Average Window', 3, 30, 3) if model_choice == 'Moving Average' else None
alpha = st.slider('Alpha', 0.01, 1.0, 0.1) if model_choice == 'Exponential Smoothing' else None
period = st.slider('Seasonal Period', 2, 12, 4) if model_choice == 'Holt-Winters' else None

# Button to generate forecast
if st.button('Generate Forecast'):
    # Fetch stock data
    data = fetch_stock_data(stock_code, period=time_frame)

    # Generate forecast
    if model_choice == 'Moving Average':
        forecast = moving_average(data, window)
    elif model_choice == 'Exponential Smoothing':
        forecast = exponential_smoothing(data, alpha)
    else:  # Holt-Winters
        forecast = holt_winters(data, period)
    
    # Plot the forecast and actual prices
    plt.figure(figsize=(10, 4))
    plt.plot(data['Close'], label='Actual Price')
    plt.plot(forecast, label='Forecast')
    plt.legend()
    st.pyplot(plt)
