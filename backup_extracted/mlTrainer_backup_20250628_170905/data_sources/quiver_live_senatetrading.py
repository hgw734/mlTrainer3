import logging
import requests
import pandas as pd
from data_sources.quiver_load_quiver_api_key import load_quiver_api_key

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fetch_quiver_live_senatetrading() -> pd.DataFrame:
    """
    Fetch current/live Senate trading activity.
    Endpoint: /v1/beta/live/senatetrading
    """
    try:
        api_key = load_quiver_api_key()
        url = "https://api.quiverquant.com/v1/beta/live/senatetrading"
        headers = {"Authorization": f"Bearer {api_key}"}

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        if not isinstance(data, list):
            logger.warning(f"⚠️ Unexpected format from live senatetrading endpoint: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        logger.info(f"✅ Retrieved {len(df)} live Senate trading records")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch live senatetrading data: {e}")
        return pd.DataFrame()
