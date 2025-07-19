import logging
import requests
import pandas as pd
from data_sources.quiver_load_quiver_api_key import load_quiver_api_key

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fetch_quiver_govcontractsall() -> pd.DataFrame:
    """
    Fetch the historical universe of government contract awards.
    Endpoint: /v1/beta/historical/govcontractsall
    """
    try:
        api_key = load_quiver_api_key()
        url = "https://api.quiverquant.com/v1/beta/historical/govcontractsall"
        headers = {"Authorization": f"Bearer {api_key}"}

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        if not isinstance(data, list):
            logger.warning(f"⚠️ Unexpected format for govcontractsall: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        logger.info(f"✅ Retrieved {len(df)} government contract records (all tickers)")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch govcontractsall: {e}")
        return pd.DataFrame()
