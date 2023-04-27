import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import os
most_bought_stocks = ["TSLA", "AMZN", "MSFT", "META", "NVDA", "GOOGL", "NIO", "AMD", "DIS"]
most_bought_stocks_string = "tsla amzn msft meta nvda goog nio amd dis"


stock_data = yf.download("aapl tsla amzn msft meta nvda goog nio amd dis", start="2022-01-01", end="2022-12-31")
volume = stock_data["Volume"]
adj_close = stock_data["Adj Close"]



data_folder = "data"


economic_calendar_path = os.path.join(data_folder, "economic_calendar.csv")
dji_path = os.path.join(data_folder, "^DJI.csv")
sp500_path = os.path.join(data_folder, "^GSPC.csv")
nasdaq_path = os.path.join(data_folder, "^IXIC.csv")

economic_calendar = pd.read_csv(economic_calendar_path)
dji = pd.read_csv(dji_path)
sp500 = pd.read_csv(sp500_path)
nasdaq = pd.read_csv(nasdaq_path)

dji["Date"] = pd.to_datetime(dji["date"])
sp500["Date"] = pd.to_datetime(sp500["date"])
nasdaq["Date"] = pd.to_datetime(nasdaq["date"])

print(dji.head())
print(adj_close.head())
adj_close_reset = adj_close.reset_index()
print(adj_close_reset.head())




# CPI data
data = {
    "Date": [
        "2022-01-01",
        "2022-02-01",
        "2022-03-01",
        "2022-04-01",
        "2022-05-01",
        "2022-06-01",
        "2022-07-01",
        "2022-08-01",
        "2022-09-01",
        "2022-10-01",
        "2022-11-01",
        "2022-12-01",
    ],
    "CPI": [
        0.07479872468289131,
        0.07871063897739279,
        0.0854245555484244,
        0.08258629340882359,
        0.08581511543676511,
        0.0905975796478416,
        0.08524814745625531,
        0.08262692503116231,
        0.0820166964383362,
        0.07745427330804901,
        0.0711032279419173,
        0.06454401331410821,
    ],
}

cpi_monthly = pd.DataFrame(data)
cpi_monthly["Date"] = pd.to_datetime(cpi_monthly["Date"])
cpi_monthly.set_index("Date", inplace=True)

cpi_daily_index = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
cpi_daily = cpi_monthly.reindex(cpi_daily_index)
cpi_daily = cpi_daily.interpolate(method='linear', axis=0)
cpi_daily.columns = ['CPI']  # Rename the column


cpi_daily_scaled = cpi_daily.copy()  # Create a copy of the cpi_daily DataFrame
cpi_daily_scaled['CPI'] = cpi_daily_scaled['CPI'] * 100  # Multiply all values in the CPI column by 100

cpi_monthly_scaled = cpi_monthly.copy()  # Create a copy of the cpi_daily DataFrame
cpi_monthly_scaled['CPI'] = cpi_monthly_scaled['CPI'] * 100  # Multiply all values in the CPI column by 100

