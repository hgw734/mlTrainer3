"""
Technical Indicators Calculator
Using native calculations without external dependencies
"""

import pandas as pd
import numpy as np

def compute_indicators(prices: pd.Series, params: dict) -> pd.DataFrame:
    """Compute technical indicators from price data"""
    df = pd.DataFrame(prices)
    df.columns = ["close"]
    
    # RSI calculation
    rsi_period = params.get("rsi_period", 14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # MACD calculation
    macd_fast = params.get("macd_fast", 12)
    macd_slow = params.get("macd_slow", 26)
    ema_fast = df["close"].ewm(span=macd_fast).mean()
    ema_slow = df["close"].ewm(span=macd_slow).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Momentum calculation
    df["momentum"] = df["close"].pct_change(periods=5)
    
    # Bollinger Bands
    bollinger_window = params.get("bollinger_window", 20)
    bollinger_std = params.get("bollinger_std", 2.0)
    rolling_mean = df["close"].rolling(window=bollinger_window).mean()
    rolling_std = df["close"].rolling(window=bollinger_window).std()
    df["bb_upper"] = rolling_mean + (rolling_std * bollinger_std)
    df["bb_lower"] = rolling_mean - (rolling_std * bollinger_std)
    
    # Volume indicators (if available)
    df["volume_signal"] = 1.0  # Default when volume not available
    
    df.dropna(inplace=True)
    return df