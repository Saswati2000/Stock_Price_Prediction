import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD

# Streamlit setup
st.set_page_config(page_title="Stock Price Forecast", layout="wide")
st.title("📈 Stock Price Forecast (Next 30 Days) using LSTM")

# Ticker input
ticker = st.text_input("Enter Stock Ticker (e.g. IEX, INFY, TCS)", value="IEX")

if st.button("Predict"):

    # Download stock data
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=365 * 10)

    df = yf.download(ticker, start=start_date, end=today)

    if df.empty:
        st.error("Failed to fetch stock data. Please check the ticker symbol.")
    else:
        df = df[['Close']].copy()
        df.reset_index(inplace=True)
        df.columns = ['Date', 'Close']

        # ===================== Describing Data =====================
        st.subheader('📊 Data Summary (From 2015 to 2025)')
        st.write(df.describe())

        # ===================== Visualization =====================
        st.subheader('📈 Closing Price vs Time Chart')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], label='Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        st.pyplot(fig)

        st.subheader('📈 Closing Price vs Time Chart with 100-Day Moving Average')
        ma100 = df['Close'].rolling(100).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], label='Closing Price')
        plt.plot(df['Date'], ma100, 'r', label='100MA')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

        st.subheader('📈 Closing Price vs Time Chart with 100MA & 200MA')
        ma200 = df['Close'].rolling(200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], 'g', label='Closing Price')
        plt.plot(df['Date'], ma100, 'r', label='100MA')
        plt.plot(df['Date'], ma200, 'b', label='200MA')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

        # ===================== Technical Indicators =====================
        df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df['MACD'] = MACD(df['Close']).macd()
        df.dropna(inplace=True)

        # ===================== Scaling =====================
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Close', 'EMA_20', 'RSI', 'MACD']])

        x_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            x_train.append(scaled_data[i - 60:i])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # ===================== LSTM Model =====================
        # model = Sequential()
        # model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 4)))
        # model.add(LSTM(50))
        # model.add(Dense(1))
        # model.compile(optimizer='adam', loss='mean_squared_error')
        # model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

        # ===================== Load Pre-trained Model =====================
        from keras.models import load_model
        model = load_model('keras_model.h5')


        # ===================== Forecasting =====================
        past_60_days = scaled_data[-60:]
        input_seq = past_60_days.copy()
        future_prices = []

        for _ in range(30):
            pred = model.predict(input_seq.reshape(1, 60, 4), verbose=0)
            future_prices.append(pred[0][0])
            next_input = [pred[0][0], input_seq[-1][1], input_seq[-1][2], input_seq[-1][3]]
            input_seq = np.append(input_seq[1:], [next_input], axis=0)

        dummy = np.zeros((30, 4))
        dummy[:, 0] = future_prices
        predicted_prices_30_days = scaler.inverse_transform(dummy)[:, 0]

        # ===================== Dates =====================
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]

        # ===================== Train Prediction Plot =====================
        train_predictions = model.predict(x_train, verbose=0)
        train_plot = np.zeros((len(scaled_data), 1))
        train_plot[60:] = train_predictions
        train_plot = scaler.inverse_transform(np.hstack([train_plot, scaled_data[:, 1:]]))[:, 0]
        original_prices = df['Close'].values

        st.subheader('🧠 Model Fit (Original vs Predicted on Training Data)')
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(df['Date'], original_prices, label='Original Price')
        ax1.plot(df['Date'], train_plot, label='Predicted (Train)', linestyle='--')
        ax1.set_title('Original Price vs Predicted Price (Train)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (INR)')
        ax1.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        # ===================== Future Prediction Plot =====================
        st.subheader('🔮 Stock Price Forecast (Next 30 Days)')
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(future_dates, predicted_prices_30_days, marker='o', label='Predicted Price (Next 30 days)')
        ax2.set_title('Stock Price Forecast')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price (INR)')
        ax2.grid(True)
        ax2.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # ===================== Show Table =====================
        st.subheader("📅 Forecasted Prices")
        result_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': np.round(predicted_prices_30_days, 2)})
        st.dataframe(result_df.set_index('Date'))
