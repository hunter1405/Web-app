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
def exponential_smoothing(series, alpha):
    model = SimpleExpSmoothing(series)
    fitted_model = model.fit(smoothing_level=alpha)
    return fitted_model.fittedvalues

# Define the Holt-Winters function
def holt_winters(series, period):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=period)
    fitted_model = model.fit()
    return fitted_model.fittedvalues

# Streamlit UI
st.title('Stock Forecasting Application')

# Input for stock ticker symbol
stock_code = st.text_input('Enter Stock Ticker Symbol', 'AAPL')

# Dropdown for choosing the forecasting model
model_choice = st.selectbox('Choose the Forecasting Model', ['Moving Average', 'Exponential Smoothing', 'Holt-Winters'])

# Dropdown for choosing the time frame for stock data
time_frame = st.selectbox('Choose Time Frame for Stock Data', ['1d', '1wk', '1mo'])

# Generate forecast
if st.button('Generate Forecast'):
    # Fetch stock data
    data = fetch_stock_data(stock_code, period=time_frame)

    # Ensure we are passing a Series to the forecasting functions
    series = data['Close']

    # Generate forecast
    if model_choice == 'Moving Average':
        window = st.slider('Moving Average Window', 3, 30, 3)
        forecast = moving_average(data, window)
    elif model_choice == 'Exponential Smoothing':
        alpha = st.slider('Alpha', 0.01, 1.0, 0.1)
        forecast = exponential_smoothing(series, alpha)
    else:  # Holt-Winters
        period = st.slider('Seasonal Period', 2, 12, 4)
        forecast = holt_winters(series, period)
    
        # Plot the forecast and actual prices
    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series, label='Actual Price')
    plt.plot(forecast.index, forecast, label='Forecast')
    
    # Format the dates on the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Forecast')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate dates for better readability
    st.pyplot(plt)
