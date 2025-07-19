import logging
import requests
import pandas as pd
from data_sources.quiver_load_quiver_api_key import load_quiver_api_key

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fetch_quiver_housetrading(ticker: str) -> pd.DataFrame:
    """
    Fetch historical House of Representatives trading activity for a given ticker.
    Endpoint: /v1/beta/historical/housetrading/{ticker}
    """
    try:
        api_key = load_quiver_api_key()
        url = f"https://api.quiverquant.com/v1/beta/historical/housetrading/{ticker}"
        headers = {"Authorization": f"Bearer {api_key}"}

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        if not isinstance(data, list):
            logger.warning(f"⚠️ Unexpected format for housetrading {ticker}: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['ticker'] = ticker
        logger.info(f"✅ Retrieved {len(df)} House trading records for {ticker}")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch housetrading for {ticker}: {e}")
        return pd.DataFrame()
