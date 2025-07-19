import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime

API_CONFIG_PATH = "ai_config.json"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_polygon_api_key() -> str:
    """
    Load the Polygon API key using centralized configuration.
    """
    from core.configuration import get_api_key
    try:
        return get_api_key("polygon")
    except Exception as e:
        logger.error(f"❌ Failed to load Polygon API key: {e}")
        raise e

def fetch_polygon_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given symbol using Polygon.io's REST API (15-min delayed or historical).
    """
    api_key = load_polygon_api_key()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=10000&apiKey={api_key}"

    try:
        logger.info(f"🌐 Fetching data from Polygon for {symbol}: {start} to {end}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "results" not in data:
            logger.warning(f"⚠️ No results in Polygon response: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])
        df["timestamp"] = pd.to_datetime(df["t"], unit='ms')
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.set_index("timestamp", inplace=True)
        return df

    except Exception as e:
        logger.error(f"❌ Polygon fetch failed: {e}")
        return pd.DataFrame()
