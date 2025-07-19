import os
import json
import logging
import requests
import pandas as pd

API_CONFIG_PATH = "ai_config.json"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_quiver_api_key() -> str:
    """
    Load the QuiverQuant API key from ai_config.json.
    """
    if not os.path.exists(API_CONFIG_PATH):
        raise FileNotFoundError("❌ ai_config.json not found")

    try:
        with open(API_CONFIG_PATH, "r") as f:
            config = json.load(f)
        quiver_key = config.get("quiverquant", {}).get("api_key")
        if not quiver_key:
            raise KeyError("❌ 'quiverquant' API key not found in ai_config.json")
        return quiver_key
    except Exception as e:
        logger.error(f"❌ Failed to load QuiverQuant API key: {e}")
        raise e

def fetch_congress_trades(symbol: str) -> pd.DataFrame:
    """
    Fetch recent congressional trades for a given stock from QuiverQuant.
    Returns a Pandas DataFrame. Returns empty DataFrame if fetch fails.
    """
    if not symbol or not symbol.strip():
        logger.warning("⚠️ No symbol provided to fetch_congress_trades()")
        return pd.DataFrame()

    api_key = load_quiver_api_key()
    url = f"https://api.quiverquant.com/beta/historical/congresstrading/{symbol.upper()}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        logger.info(f"🏛️ Fetching QuiverQuant congressional trades for {symbol}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            logger.warning(f"⚠️ Unexpected response format from QuiverQuant for {symbol}")
            return pd.DataFrame()

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ HTTP error from QuiverQuant for {symbol}: {http_err}")
    except Exception as e:
        logger.error(f"❌ QuiverQuant fetch failed for {symbol}: {e}")

    return pd.DataFrame()
