import os
import json
import logging
import requests
import pandas as pd

API_CONFIG_PATH = "ai_config.json"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_fred_api_key() -> str:
    """
    Load the FRED API key using centralized configuration.
    """
    from core.configuration import get_api_key
    try:
        return get_api_key("fred")
    except Exception as e:
        logger.error(f"❌ Failed to load FRED API key: {e}")
        raise e

def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """
    Generic fetch from FRED for any time series (e.g. VIX, interest rates).
    """
    api_key = load_fred_api_key()
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end
    }

    try:
        logger.info(f"📈 Fetching FRED series: {series_id}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()["observations"]

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df["value"].replace(".", None).astype(float)
        df.name = series_id
        return df

    except Exception as e:
        logger.error(f"❌ FRED fetch failed for {series_id}: {e}")
        return pd.Series(name=series_id)

# Specific economic indicators used in regime classification:

def fetch_vix(start: str, end: str) -> pd.Series:
    return fetch_fred_series("VIXCLS", start, end)

def fetch_interest_rates(start: str, end: str) -> pd.Series:
    return fetch_fred_series("DFF", start, end)  # Daily Federal Funds Rate

def fetch_yield_spread(start: str, end: str) -> pd.Series:
    return fetch_fred_series("T10Y2Y", start, end)  # 10Y minus 2Y Treasury spread
