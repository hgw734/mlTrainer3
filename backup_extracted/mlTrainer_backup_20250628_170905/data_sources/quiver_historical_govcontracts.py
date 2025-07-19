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
        api_key = config.get("quiverquant", {}).get("api_key")
        if not api_key:
            raise KeyError("❌ 'quiverquant' API key not found in ai_config.json")
        return api_key
    except Exception as e:
        logger.error(f"❌ Failed to load QuiverQuant API key: {e}")
        raise e

def fetch_historical_gov_contracts(ticker: str) -> pd.DataFrame:
    """
    Fetch historical government contracts for a given ticker.
    Example: LMT, RTX, BA
    """
    api_key = load_quiver_api_key()
    url = f"https://api.quiverquant.com/beta/historical/govcontracts/{ticker.upper()}"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        logger.info(f"🏛️ Fetching historical gov contracts for {ticker}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"❌ QuiverQuant fetch failed for {ticker}: {e}")
        return pd.DataFrame()
