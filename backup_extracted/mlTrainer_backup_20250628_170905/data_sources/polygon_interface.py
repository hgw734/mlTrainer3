# data_sources/polygon_interface.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from core.immutable_gateway import enforce_verified_source
from core.compliance_mode import enforce_compliance
from monitoring.error_monitor import log_error
import logging

# Get logger
logger = logging.getLogger(__name__)

# Load API key securely from environment (Replit Secret)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

BASE_URL = "https://api.polygon.io"
DELAYED_MINUTES = 15  # Only use 15-minute delayed data for live

def verify_real_api_connection() -> bool:
    """
    Verify that we can actually connect to real Polygon API.
    NO FALLBACKS TO SYNTHETIC DATA ALLOWED.
    """
    if not POLYGON_API_KEY:
        logger.error("❌ CRITICAL: No Polygon API key - cannot guarantee real data")
        raise PermissionError("BLOCKED: Polygon API key missing - no synthetic data allowed")

    try:
        # Test connection with a simple API call
        test_url = f"{BASE_URL}/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-02"
        headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
        response = requests.get(test_url, headers=headers, timeout=10)

        if response.status_code != 200:
            logger.error(f"❌ CRITICAL: Polygon API connection failed - status {response.status_code}")
            raise PermissionError("BLOCKED: Cannot connect to real Polygon API - no synthetic data allowed")

        logger.info("✅ REAL API VERIFIED: Polygon connection successful")
        return True

    except Exception as e:
        logger.error(f"❌ CRITICAL: Polygon API verification failed: {e}")
        raise PermissionError("BLOCKED: Cannot verify real API connection - no synthetic data allowed")

@enforce_compliance  
@enforce_verified_source
def get_historical_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch historical daily price data from Polygon.io"""
    try:
        url = (
            f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start_date.date()}/{end_date.date()}?adjusted=true&sort=asc&limit=50000"
            f"&apiKey={POLYGON_API_KEY}"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("results", [])

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.rename(columns={
            'o': 'open', 'h': 'high', 'l': 'low',
            'c': 'close', 'v': 'volume'
        }, inplace=True)

        return df[['open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        log_error(f"❌ Failed to load historical data for {symbol}", details=str(e))
        return pd.DataFrame()

@enforce_compliance
@enforce_verified_source
def get_live_price(symbol: str) -> float:
    """Fetch 15-minute delayed last quote for a symbol"""
    try:
        now = datetime.utcnow() - timedelta(minutes=DELAYED_MINUTES)
        date_str = now.strftime('%Y-%m-%d')
        url = (
            f"{BASE_URL}/v1/open-close/{symbol}/{date_str}?adjusted=true&apiKey={POLYGON_API_KEY}"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "close" in data:
            return float(data["close"])

        return None

    except Exception as e:
        log_error(f"❌ Failed to fetch live (delayed) price for {symbol}", details=str(e))
        return None