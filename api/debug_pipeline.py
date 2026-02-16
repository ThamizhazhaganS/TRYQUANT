import sys
import os
import traceback
import pandas as pd
import numpy as np

# Add current directory to path so imports work
sys.path.append(os.getcwd())

from data_loader import fetch_data
from model_engine import run_prediction_pipeline, monte_carlo_simulation

def test_pipeline(ticker="AAPL"):
    print(f"--- Testing Pipeline for {ticker} ---")
    try:
        # 1. Fetch Data
        print("1. Fetching data...")
        df = fetch_data(ticker)
        if df is None:
            print("Error: DataFrame is None (Ticker not found)")
            return

        print(f"Data fetched. Shape: {df.shape}")
        print(f"Columns: {df.columns}")
        if isinstance(df.columns, pd.MultiIndex):
            print("Detected MultiIndex columns")
        else:
            print("Detected SingleIndex columns")
            
        print("Head:\n", df.head(3))

        # 2. Run Prediction Pipeline
        print("\n2. Running Prediction Pipeline...")
        metrics = run_prediction_pipeline(df)
        print("Metrics calculated:", metrics)

        # 3. Sentiment (Mocking it for simplicity or importing if needed)
        print("\n3. Calculating Sentiment (Mocked)...")
        sentiment = 50

        # 4. Monte Carlo
        print("\n4. Running Monte Carlo Simulation...")
        sim_days = 30
        vol_multiplier = 1.0
        
        sim_df = monte_carlo_simulation(
            metrics["last_close"], 
            metrics["volatility"], 
            days=sim_days, 
            simulations=100,
            vol_multiplier=vol_multiplier
        )
        print(f"Simulation DF Shape: {sim_df.shape}")
        print("Simulation Head:\n", sim_df.head(3))

        # 5. Fusion Signal
        print("\n5. Generating Fusion Signal...")
        def get_fusion_signal(m, s):
            score = 0
            if m['prediction'] > m['last_close']: score += 1
            if m['prediction'] > m['last_close'] * 1.02: score += 1
            if m['rsi'] < 35: score += 2
            elif m['rsi'] > 65: score -= 2
            if s > 65: score += 1
            if s < 35: score -= 1
            
            if score >= 3: return "Strong Buy"
            if score >= 1: return "Buy"
            if score <= -3: return "Strong Sell"
            if score <= -1: return "Sell"
            return "Neutral"

        signal = get_fusion_signal(metrics, sentiment)
        print(f"Signal: {signal}")

        # 6. Post-processing for JSON (as done in main.py)
        print("\n6. Post-processing data...")
        
        # Sim Data
        sim_data = []
        for i in range(len(sim_df)):
            row = sim_df.iloc[i]
            sim_data.append({
                "day": int(i),
                "mean": float(row.mean()),
                "p90": float(row.quantile(0.9)),
                "p10": float(row.quantile(0.1))
            })
        print(f"Processed {len(sim_data)} simulation days.")

        # Historical Data
        temp_df = df.copy()
        if isinstance(temp_df.columns, pd.MultiIndex):
            temp_df.columns = temp_df.columns.get_level_values(0)
        
        hist_subset = temp_df[['Close']].copy()
        hist_subset.reset_index(inplace=True)
        hist_subset.columns = ['Date', 'Close']
        hist_subset['Date'] = hist_subset['Date'].astype(str)
        hist_subset['Close'] = hist_subset['Close'].astype(float)
        formatted_hist = hist_subset.to_dict(orient='records')[-5:] # Just last 5
        print("History sample:", formatted_hist)

        print("\n--- TEST SUCCESSFUL ---")

    except Exception:
        print("\n!!! EXCEPTION OCCURRED !!!")
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
