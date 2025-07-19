import os
import json
import logging
import requests
import pandas as pd
from data_sources.quiver_load_quiver_api_key import load_quiver_api_key

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fetch_quiver_house_trading(ticker: str) -> pd.DataFrame:
    """
    Fetch historical House of Representatives trading activity for a given ticker.
    Endpoint: /v1/beta/historical/housetrading/{ticker}
    """
    try:
        api_key = load_quiver_api_key()
        url = f"https://api.quiverquant.com/v1/beta/historical/housetrading/{ticker}"
        headers = {"Authorization": f"Bearer {api_key}"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, list):
            logger.warning(f"⚠️ Unexpected response format for {ticker}: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['ticker'] = ticker
        logger.info(f"✅ Retrieved {len(df)} House trading entries for {ticker}")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch House trading data for {ticker}: {e}")
        return pd.DataFrame()
