import numpy as np
import pandas as pd
import requests
from sklearn.mixture import GaussianMixture
from utils.config_loader import CONFIG
from utils.symbol_mapper import normalize_symbol

def get_historical_prices(symbol: str, days: int = 60) -> pd.Series:
    """
    Fetch historical close prices from Polygon.io for a given symbol.
    Returns a pandas Series of closing prices.
    """
    key = CONFIG['polygon_api_key']
    norm = normalize_symbol(symbol)
    url = f"https://api.polygon.io/v2/aggs/ticker/{norm}/range/1/day/2023-01-01/2024-12-31?adjusted=true&sort=desc&limit={days}&apiKey={key}"
    
    try:
        response = requests.get(url)
        data = response.json().get("results", [])
        prices = [bar['c'] for bar in data]
        return pd.Series(prices[::-1])  # chronological order
    except Exception as e:
        print(f"[ERROR] Price fetch failed for {symbol}: {e}")
        return pd.Series(dtype=float)

def classify_regime(prices: pd.Series, n_states=3) -> int:
    """
    ML-based regime classifier using Gaussian Mixture Model (GMM).
    Input:  Series of historical close prices.
    Output: Latest regime label (0 to n_states-1).
    """
    if prices.empty or len(prices) < 10:
        return -1  # fallback value for invalid data

    log_returns = np.log(prices / prices.shift(1)).dropna().values.reshape(-1, 1)

    model = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
    model.fit(log_returns)

    states = model.predict(log_returns)
    return states[-1]  # latest regime