import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_sources.polygon_api import fetch_polygon_ohlcv
from data_sources.fred_api import fetch_regime_factors

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_polygon_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Load verified OHLCV data for a symbol from Polygon API.
    """
    logger.info(f"📥 Fetching Polygon data for {symbol} ({start} to {end})")
    df = fetch_polygon_ohlcv(symbol, start, end)

    if df is None or df.empty:
        logger.warning(f"⚠️ No data returned from Polygon for {symbol}")
        raise ValueError("Polygon data fetch failed or returned empty DataFrame")

    df = df.sort_index()
    return df

def integrate_fred_data(start: str, end: str) -> pd.DataFrame:
    """
    Fetch macro regime indicators from FRED (e.g. VIX, interest rates).
    """
    logger.info(f"🌐 Integrating FRED regime indicators ({start} to {end})")
    fred_df = fetch_regime_factors(start, end)

    if fred_df is None or fred_df.empty:
        logger.warning("⚠️ No regime data returned from FRED")
        raise ValueError("FRED regime data fetch failed or returned empty DataFrame")

    return fred_df

def preprocess_data(price_df: pd.DataFrame, regime_df: pd.DataFrame, target_column: str = "close"):
    """
    Merge price and macro data, build model-ready X/y datasets.
    """
    df = price_df.copy()

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    df = df.join(regime_df, how="left").fillna(method="ffill")

    df["target"] = df[target_column].shift(-1)
    df.dropna(inplace=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    return X.values, y.values
