import logging
import requests
import pandas as pd
from data_sources.quiver_load_quiver_api_key import load_quiver_api_key

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fetch_quiver_live_govcontracts() -> pd.DataFrame:
    """
    Fetch currently active/most recent government contract awards.
    Endpoint: /v1/beta/live/govcontracts
    """
    try:
        api_key = load_quiver_api_key()
        url = "https://api.quiverquant.com/v1/beta/live/govcontracts"
        headers = {"Authorization": f"Bearer {api_key}"}

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        if not isinstance(data, list):
            logger.warning(f"⚠️ Unexpected format from live govcontracts endpoint: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        logger.info(f"✅ Retrieved {len(df)} live government contract records")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch live govcontracts data: {e}")
        return pd.DataFrame()
