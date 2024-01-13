import streamlit as st
from datetime import date

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from statsmodels.tsa.api import Holt

# Function to load data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to plot raw data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Function for Holt Linear Trend Forecast
def holt_linear_trend_forecast(series, alpha, slope):
    model = Holt(series).fit(smoothing_level=alpha, smoothing_slope=slope)
    return model.fittedvalues

# Streamlit App
START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

# Select stock
stocks = ('GOOG', 'AAPL', 'MSFT', 'AMZN', 'META')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Select years of prediction
n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365

# Load data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

plot_raw_data(data)

# Select Forecasting Model
models = ['Moving Average', 'Holt Linear Trend']  # Add other models here
model = st.selectbox('Select Forecasting Model', models)

# Forecasting with the selected model
if model == 'Moving Average':
    # Moving Average model implementation
    window = st.slider('Select Moving Average Window', 15, 90, 30)
    df_train = data[['Date', 'Close']].set_index('Date')
    df_train['Moving Average'] = df_train['Close'].rolling(window=window).mean()

    st.subheader('Moving Average Forecast')
    st.write(df_train.tail())

    st.write(f'Moving Average plot for {window} day window')
    plt.figure(figsize=(10, 4))
    plt.plot(df_train['Close'], label='Actual Price')
    plt.plot(df_train['Moving Average'], label='Moving Average')
    plt.legend()
    st.pyplot(plt)

elif model == 'Holt Linear Trend':
    # Holt Linear Trend model implementation
    alpha = st.slider('Select Smoothing Level (alpha)', 0.01, 1.0, 0.1)
    slope = st.slider('Select Smoothing Slope (beta)', 0.01, 1.0, 0.1)
    
    # Perform Holt Linear Trend forecasting
    df_train = data[['Date', 'Close']].set_index('Date')
    forecast_values = holt_linear_trend_forecast(df_train['Close'], alpha, slope)
    df_train['Holt Linear Trend Forecast'] = forecast_values

    st.subheader('Holt Linear Trend Forecast')
    st.write(df_train.tail())

    st.write(f'Holt Linear Trend plot for alpha={alpha} and slope={slope}')
    plt.figure(figsize=(10, 4))
    plt.plot(df_train['Close'], label='Actual Price')
    plt.plot(df_train['Holt Linear Trend Forecast'], label='Holt Linear Trend Forecast')
    plt.legend()
    st.pyplot(plt)
