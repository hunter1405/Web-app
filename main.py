# pip install streamlit yfinance plotly pandas matplotlib
import streamlit as st
from datetime import date

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go

# If using Prophet, uncomment the following lines:
# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

# Function to load data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

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

plot_raw_data()

# Select Forecasting Model
model = st.selectbox('Select Forecasting Model', ['Prophet', 'Moving Average'])

# Forecasting with the selected model
if model == 'Prophet':
    # Prophet model implementation
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

elif model == 'Moving Average':
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

# Add other models as elif statements similar to above with their respective implementations.
