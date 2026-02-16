import yfinance as yf
import pandas as pd

def fetch_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="2y")
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
