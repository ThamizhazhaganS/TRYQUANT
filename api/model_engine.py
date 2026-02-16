import numpy as np
import pandas as pd
# Removed heavy scikit-learn dependency to fit within Vercel Serverless Function limits
# Implemented lightweight polynomial ridge regression using NumPy

class RidgePoly:
    def __init__(self, degree=2, alpha=1.0):
        self.degree = degree
        self.alpha = alpha
        self.w = None
        self.mean_y = 0.0

    def fit(self, X, y):
        self.mean_y = np.mean(y)
        # Add bias term and polynomial features (Linear + Squared)
        X_poly = self._transform(X)
        n_features = X_poly.shape[1]
        
        # Identity matrix for regularization
        I = np.eye(n_features)
        I[0, 0] = 0  # Do not regularize the bias term (index 0)
        
        try:
            # Closed-form Ridge Regression: w = (X^T X + alpha I)^-1 X^T y
            # Using linalg.solve for numerical stability
            A = X_poly.T @ X_poly + self.alpha * I
            b = X_poly.T @ y
            self.w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            self.w = np.zeros(n_features)
            self.w[0] = self.mean_y

    def predict(self, X):
        if self.w is None:
            return np.full((X.shape[0],), self.mean_y)
        X_poly = self._transform(X)
        return X_poly @ self.w

    def score(self, X, y):
        # R^2 Score
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - np.mean(y)) ** 2).sum()
        return 1 - u/v if v != 0 else 0

    def _transform(self, X):
        # Generate polynomial features: Bias + Linear + Squared
        # Skipping interaction terms to keep it lightweight while capturing non-linearity
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        X = np.atleast_2d(X)
        
        # Bias term (column of 1s)
        X_poly = np.hstack([np.ones((n_samples, 1)), X])
        
        if self.degree > 1:
            # Add squared terms
            X_poly = np.hstack([X_poly, X**2])
            
        return X_poly

def polynomial_regression(X, y, degree=2):
    try:
        model = RidgePoly(degree=degree, alpha=1.0)
        model.fit(X, y)
        return model
    except:
        return None

def calculate_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Std_Dev'] * 2)
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # ADX
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['PlusDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['MinusDM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    df['PlusDI'] = 100 * (df['PlusDM'].rolling(14).mean() / (df['ATR'] + 1e-9))
    df['MinusDI'] = 100 * (df['MinusDM'].rolling(14).mean() / (df['ATR'] + 1e-9))
    dx = 100 * np.abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'] + 1e-9)
    df['ADX'] = dx.rolling(14).mean()
    
    return df

def calculate_technical_radar(df):
    latest = df.iloc[-1]
    momentum = float(latest['RSI'])
    trend_strength = float(max(0, min(100, latest['ADX'] * 2)))
    bb_width = (latest['BB_Upper'] - latest['BB_Lower']) / (latest['SMA_20'] + 1e-9)
    vol_score = float(max(0, min(100, (1 - bb_width) * 100)))
    avg_obv = df['OBV'].rolling(window=20).mean().iloc[-1]
    strength = float(max(0, min(100, (latest['OBV'] / (avg_obv + 1e-9)) * 50)))
    daily_range = (latest['High'] - latest['Low']) / (latest['Close'] + 1e-9)
    range_score = float(max(0, min(100, daily_range * 1000)))
    
    return {
        "momentum": momentum,
        "trend": trend_strength,
        "volatility": vol_score,
        "strength": strength,
        "volume": range_score
    }

def calculate_institutional_metrics(df):
    latest = df.iloc[-1]
    support = float(df['Low'].tail(50).min())
    resistance = float(df['High'].tail(50).max())
    obv_5d = float(df['OBV'].pct_change(5).iloc[-1])
    flow_status = "ACCUMULATION" if obv_5d > 0 else "DISTRIBUTION"
    bb_width = (latest['BB_Upper'] - latest['BB_Lower']) / (latest['SMA_20'] + 1e-9)
    bb_squeeze = bb_width < 0.1
    vol_surge = latest['Volume'] > df['Volume'].rolling(window=20).mean().iloc[-1] * 1.5
    breakout_prob = 85 if (bb_squeeze and vol_surge) else (60 if bb_squeeze else 30)
    risk_level = float(latest['ATR'] * 2)
    stop_loss = float(latest['Close'] - risk_level)
    
    return {
        "support": support,
        "resistance": resistance,
        "flow_status": flow_status,
        "breakout_prob": breakout_prob,
        "stop_loss": stop_loss,
        "risk_level": risk_level,
        "adx": float(latest['ADX']),
        "bb_width": float(bb_width * 100), # Return as percentage
        "obv_change": float(obv_5d * 100)  # Return as percentage
    }

def run_prediction_pipeline(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = calculate_indicators(df.copy())
    last_close = float(df['Close'].iloc[-1])
    
    # Volatility
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    volatility = float(log_returns.std())
    
    # Feature Engineering
    num_lags = 5
    for i in range(1, num_lags + 1):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    
    df_train = df.dropna()
    if len(df_train) < 50:
        return {"prediction": last_close, "volatility": volatility, "last_close": last_close, "rsi": 50, "macd": 0, "radar": {}, "confidence": 0, "institutional": {}, "ensemble_agreement": 0}
        
    feature_cols = [f'Lag_{i}' for i in range(1, num_lags + 1)] + ['RSI', 'MACD', 'ADX']
    X = df_train[feature_cols].values
    y = df_train['Close'].values
    
    # --- MODEL 1: Trend Engine (Polynomial Ridge) ---
    model_trend = polynomial_regression(X, y, degree=2)
    pred_trend = last_close
    conf_trend = 0
    if model_trend:
        # Use our custom model's predict which handles input shape/expansion
        pred_trend = float(model_trend.predict(df[feature_cols].iloc[-1].values.reshape(1, -1))[0])
        conf_trend = float(model_trend.score(X, y) * 100)
    
    # --- MODEL 2: Mean-Reversion Engine (Statistical SMA/BB) ---
    # Prediction based on returning to the 20-day SMA if over-extended
    sma_20 = float(df['SMA_20'].iloc[-1])
    rsi = float(df['RSI'].iloc[-1])
    
    # Heuristic Mean Reversion
    if rsi > 70: # Overbought -> Revert Down
        pred_revert = sma_20 * 1.01 
    elif rsi < 30: # Oversold -> Revert Up
        pred_revert = sma_20 * 0.99
    else: 
        pred_revert = last_close * (1 + (pred_trend - last_close) / last_close * 0.5)

    # --- ENSEMBLE LOGIC ---
    # Agreement score (0-100)
    # Are both models pointing in the same direction?
    dir_trend = np.sign(pred_trend - last_close)
    dir_revert = np.sign(pred_revert - last_close)
    ensemble_agreement = 100 if dir_trend == dir_revert else 30
    
    # Weighted Ensemble Prediction
    # In strong trends (ADX > 25), lean towards Trend Engine. Otherwise, favor Mean Reversion.
    adx = float(df['ADX'].iloc[-1])
    if adx > 25:
        final_prediction = (pred_trend * 0.7) + (pred_revert * 0.3)
    else:
        final_prediction = (pred_trend * 0.4) + (pred_revert * 0.6)

    # Volatility Adjustment for Confidence
    vol_penalty = (volatility * 100) * 2
    final_confidence = max(0, min(100, conf_trend - vol_penalty))

    radar = calculate_technical_radar(df)
    institutional = calculate_institutional_metrics(df)
    
    return {
        "prediction": final_prediction,
        "pred_trend": pred_trend,
        "pred_revert": pred_revert,
        "volatility": volatility,
        "last_close": last_close,
        "rsi": float(df['RSI'].iloc[-1]),
        "macd": float(df['MACD'].iloc[-1]),
        "radar": radar,
        "confidence": final_confidence,
        "institutional": institutional,
        "ensemble_agreement": ensemble_agreement
    }

def monte_carlo_simulation(last_close, volatility, days=30, simulations=100, vol_multiplier=1.0, sentiment_bias=0.0):
    # Hard Safety Limit: Cap volatility shock at 3.0x to prevent numerical breakdown
    # Also cap sentiment bias at ±0.5 for stability
    safe_multiplier = min(3.0, vol_multiplier)
    safe_bias = max(-0.5, min(0.5, sentiment_bias))
    
    # Numerical stability: Clip returns to prevent extreme values that cause Infinity/NaN
    # The bias shifts the mean of the distribution to simulate directional drift
    drift = safe_bias * 0.005 # Constant daily drift factor
    returns = np.random.normal(drift, volatility * safe_multiplier, (simulations, days))
    returns = np.clip(returns, -0.5, 0.5) # Max 50% daily move as extreme safeguard
    
    price_paths = last_close * np.exp(np.cumsum(returns, axis=1))
    
    # Final check: Clip price paths to prevent overflow
    price_paths = np.clip(price_paths, 0.01, last_close * 100) # Floor at 0.01, Cap at 100x
    
    return pd.DataFrame(price_paths.T)
