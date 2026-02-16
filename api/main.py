from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from data_loader import fetch_data
from model_engine import run_prediction_pipeline, monte_carlo_simulation
import pandas as pd
import numpy as np
import yfinance as yf
import os
import requests

# Define project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search")
async def search_ticker(query: str = Query(..., min_length=1)):
    """Real-time Search Engine for Tickers and Company Names"""
    try:
        # Use Yahoo Finance's internal search API for fast suggestions
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code != 200:
            return []
            
        data = response.json()
        results = []
        for quote in data.get('quotes', [])[:8]: # Limit to top 8
            results.append({
                "symbol": quote.get('symbol'),
                "name": quote.get('shortname') or quote.get('longname'),
                "exchange": quote.get('exchDisp'),
                "type": quote.get('quoteType')
            })
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

@app.get("/api/health")
def read_root():
    return {"status": "MarketProb API is running", "version": "1.0.0"}

def get_weighted_sentiment(ticker: str, df: pd.DataFrame):
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news: return 50
        
        bull_words = ['up', 'rise', 'growth', 'gain', 'high', 'bull', 'buy', 'positive', 'surge', 'jump', 'profit', 'outperform', 'upgrade', 'beat']
        bear_words = ['down', 'drop', 'fall', 'loss', 'low', 'bear', 'sell', 'negative', 'crash', 'plunge', 'risk', 'underperform', 'downgrade', 'miss']
        
        # Calculate volume weight (Current volume vs Average)
        curr_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        vol_multiplier = max(1, min(3, curr_vol / (avg_vol + 1e-9)))

        score = 0
        for n in news:
            try:
                title = n.get('title', '').lower()
                if not title: continue
                for w in bull_words:
                    if w in title: score += 1
                for w in bear_words:
                    if w in title: score -= 1
            except: continue
        
        sentiment = 50 + (score * 5 * vol_multiplier)
        return max(0, min(100, sentiment))
    except: return 50

@app.get("/predict/{ticker}")
async def predict(
    ticker: str, 
    vol_multiplier: float = Query(1.0, gt=0, lt=5),
    sentiment_bias: float = Query(0.0),
    iterations: int = Query(1000, gt=99, lt=5001),
    days: int = Query(60, gt=6, lt=61)
):
    try:
        df = fetch_data(ticker)
        if df is None:
            raise HTTPException(status_code=404, detail="Ticker not found")
        
        metrics = run_prediction_pipeline(df)
        sentiment = get_weighted_sentiment(ticker, df)
        
        sim_df = monte_carlo_simulation(
            metrics["last_close"], 
            metrics["volatility"], 
            days=days, 
            simulations=iterations,
            vol_multiplier=vol_multiplier,
            sentiment_bias=sentiment_bias
        )

        def get_market_analysis(m, s):
            score = 0
            reasons = []
            
            # Ensemble Agreement check
            if m.get('ensemble_agreement', 0) > 70:
                score += 1
                reasons.append("Neural Engines in High Agreement")
            
            # Price Momentum
            if m['prediction'] > m['last_close']:
                score += 1
                reasons.append("Bullish price momentum detected")
            
            # Technical Indicators
            if m['rsi'] < 35:
                score += 2
                reasons.append("Asset is currently oversold (RSI)")
            elif m['rsi'] > 65:
                score -= 2
                reasons.append("Asset is currently overbought (RSI)")
                
            # Social Sentiment
            if s > 65:
                score += 1
                reasons.append("High-Volume positive sentiment surge")
            elif s < 35:
                score -= 1
                reasons.append("High-Volume negative sentiment detected")
            
            # Breakout Probability
            if m['institutional'].get('breakout_prob', 0) > 70:
                score += 1
                reasons.append("High explosive breakout potential")

            signal = "Neutral"
            if score >= 3: signal = "Strong Buy"
            elif score >= 1: signal = "Buy"
            elif score <= -3: signal = "Strong Sell"
            elif score <= -1: signal = "Sell"
            
            return signal, reasons
            
        # Market Overview
        market_overview = []
        try:
            indices = [
                {"name": "S&P 500", "symbol": "^GSPC"},
                {"name": "Nasdaq", "symbol": "^IXIC"},
                {"name": "Bitcoin", "symbol": "BTC-USD"},
                {"name": "Gold", "symbol": "GC=F"},
                {"name": "VIX", "symbol": "^VIX"}
            ]
            
            # Fetch data using Ticker for latest info
            avg_change = 0
            count = 0
            vix_val = 0
            
            for idx in indices:
                try:
                    t_idx = yf.Ticker(idx["symbol"])
                    hist = t_idx.history(period="5d")
                    if len(hist) >= 2:
                        current = float(hist["Close"].iloc[-1])
                        prev = float(hist["Close"].iloc[-2])
                        change = ((current - prev) / prev) * 100
                        
                        if idx["symbol"] == "^VIX":
                            vix_val = current
                        elif idx["symbol"] in ["^GSPC", "^IXIC"]:
                            avg_change += change
                            count += 1
                            
                        market_overview.append({
                            "name": idx["name"],
                            "price": current,
                            "change": change
                        })
                except:
                    pass
            
            # Determine Market Mood
            market_mood = "Neutral"
            if count > 0:
                avg_change /= count
                if avg_change > 0.5: market_mood = "Bullish"
                elif avg_change < -0.5: market_mood = "Bearish"
            
            if vix_val > 20: market_mood = "Fear / High Volatility"
            if vix_val > 30: market_mood = "Extreme Fear"
        except Exception:
            market_mood = "Neutral"
            
        market_signal, signal_reasons = get_market_analysis(metrics, sentiment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    sim_data = []
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        # Ensure all values are finite and non-NaN for JSON stability
        m_val = float(np.nan_to_num(row.mean(), nan=metrics["last_close"]))
        p90_val = float(np.nan_to_num(row.quantile(0.9), nan=metrics["last_close"]))
        p10_val = float(np.nan_to_num(row.quantile(0.1), nan=metrics["last_close"]))
        
        sim_data.append({
            "day": int(i),
            "mean": m_val,
            "p90": p90_val,
            "p10": p10_val
        })
    
    # Format historical data
    temp_df = df.copy()
    if isinstance(temp_df.columns, pd.MultiIndex):
        temp_df.columns = temp_df.columns.get_level_values(0)
    
    hist_subset = temp_df[['Close']].copy()
    hist_subset.reset_index(inplace=True)
    hist_subset.columns = ['Date', 'Close']
    hist_subset['Date'] = hist_subset['Date'].astype(str)
    hist_subset['Close'] = hist_subset['Close'].astype(float)
    formatted_hist = hist_subset.to_dict(orient='records')[-365:]
    
    final_prices = sim_df.iloc[-1]
    target_price = float(metrics["last_close"]) * 1.05
    # Safe mean calculation
    prob_increase = float(np.nan_to_num((final_prices > target_price).mean() * 100, nan=0.0))
        
    return {
        "ticker": str(ticker),
        "last_close": float(metrics["last_close"]),
        "predicted_next_day_lstm": float(metrics["prediction"]), 
        "volatility": float(metrics["volatility"]),
        "rsi": float(metrics["rsi"]),
        "macd": float(metrics["macd"]),
        "sentiment": int(sentiment),
        "market_signal": market_signal,
        "signal_reasons": signal_reasons,
        "market_overview": market_overview,
        "simulation_data": sim_data,
        "historical_data": formatted_hist,
        "probability_increase_5_percent": prob_increase,
        "radar": metrics["radar"],
        "confidence": float(metrics["confidence"]),
        "institutional": metrics["institutional"],
        "ensemble_agreement": metrics.get('ensemble_agreement', 0),
        "fundamentals": {},
        "news": [],
        "market_mood": market_mood
    }

# --- Unified Frontend Serving Logic ---

# Mount static assets (JS, CSS, Images)
# assets_path = os.path.join(FRONTEND_DIST, "assets")
# if os.path.exists(assets_path):
#     app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

# # Catch-all route to serve the React application
# @app.get("/{full_path:path}")
# async def serve_frontend(request: Request, full_path: str):
#     # Check if the requested path is a file in the dist folder (e.g., favicon)
#     file_path = os.path.join(FRONTEND_DIST, full_path)
#     if os.path.isfile(file_path):
#         return FileResponse(file_path)
    
#     # Fallback to index.html for all other routes (handles React Router)
#     index_path = os.path.join(FRONTEND_DIST, "index.html")
#     if os.path.exists(index_path):
#         return FileResponse(index_path)
    
#     return {"error": "Frontend build not found. Run 'npm run build' in the frontend folder."}
