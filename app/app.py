import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import date, timedelta
from src.feature_engineering import create_features

model = joblib.load("model.plk")

st.title("Stock Movement PRediction")
st.write("Predict if the stock's closing price will go up or down tomorrow!")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL")

default_end_date = date.today()
default_start_date = default_end_date - timedelta(days=365)
start_date = st.date_input("Start date", value=default_start_date)
end_date = st.date_input("End date", value=default_end_date)

model_path = "model.plk"

if st.button("Predict Next Day Movement"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found. Please check the ticker or date range.")
    else:
        df = create_features(df)

        X = df.drop(["Tomorrow_Close"], axis=1, errors="ignore")
        latest_date = X.iloc[[-1]]
        
        prediction = model.predict(latest_date)

        if prediction[0] == 1:
            st.success("The model predicts that the stock price will increase tomorrow.")
        else:
            st.warning("The model predicts that the stock price will decrease tomorrow.")