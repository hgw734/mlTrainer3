import logging
import requests
import pandas as pd
from data_sources.quiver_load_quiver_api_key import load_quiver_api_key

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fetch_quiver_lobbying(ticker: str) -> pd.DataFrame:
    """
    Fetch historical lobbying data for a given ticker.
    Endpoint: /v1/beta/historical/lobbying/{ticker}
    """
    try:
        api_key = load_quiver_api_key()
        url = f"https://api.quiverquant.com/v1/beta/historical/lobbying/{ticker}"
        headers = {"Authorization": f"Bearer {api_key}"}

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        if not isinstance(data, list):
            logger.warning(f"⚠️ Unexpected format for lobbying data {ticker}: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["ticker"] = ticker
        logger.info(f"✅ Retrieved {len(df)} lobbying records for {ticker}")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch lobbying data for {ticker}: {e}")
        return pd.DataFrame()
