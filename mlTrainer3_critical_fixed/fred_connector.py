"""
FRED API Connector - Economic Data Interface
===========================================

Connector for Federal Reserve Economic Data (FRED) API.
Provides access to economic indicators and time series data.
"""

import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import from existing configuration
from config.api_config import FRED_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EconomicData:
    """Economic data structure"""

    series_id: str
    name: str
    data: pd.DataFrame
    units: str
    frequency: str
    last_updated: datetime
    source: str = "fred"

    class FREDConnector:
        """FRED API connector"""

        def __init__(self):
            self.api_key = FRED_API_KEY
            self.base_url = "https://api.stlouisfed.org/fred"

            if not self.api_key:
                raise ValueError("FRED API key not configured")

                logger.info("FRED connector initialized")

                def _make_request(
                        self,
                        endpoint: str,
                        params: Optional[Dict] = None) -> Optional[Dict]:
                    """Make API request to FRED"""
                    url = f"{self.base_url}/{endpoint}"

                    # Add API key to params
                    if params is None:
                        params = {}
                        params["api_key"] = self.api_key
                        params["file_type"] = "json"

                        try:
                            response = requests.get(
                                url, params=params, timeout=30)
                            response.raise_for_status()
                            return response.json()
                            except Exception as e:
                                logger.error(f"FRED API request failed: {e}")
                                return None

                                def get_series_info(
                                        self, series_id: str) -> Optional[Dict]:
                                    """Get information about a series"""
                                    data = self._make_request(
                                        "series", params={"series_id": series_id})
                                    if data and "seriess" in data and data["seriess"]:
                                        return data["seriess"][0]
                                        return None

                                        def get_series_data(
                                                self,
                                                series_id: str,
                                                start_date: Optional[str] = None,
                                                end_date: Optional[str] = None,
                                                limit: int = 10000) -> Optional[EconomicData]:
                                            """Get time series data"""
                                            try:
                                                # Get series info first
                                                info = self.get_series_info(
                                                    series_id)
                                                if not info:
                                                    logger.error(
                                                        f"Series {series_id} not found")
                                                    return None

                                                    # Default date range (5
                                                    # years)
                                                    if not end_date:
                                                        end_date = datetime.now().strftime("%Y-%m-%d")
                                                        if not start_date:
                                                            start_date = (
                                                                datetime.now() -
                                                                timedelta(
                                                                    days=1825)).strftime("%Y-%m-%d")

                                                            # Get observations
                                                            params = {
                                                                "series_id": series_id,
                                                                "observation_start": start_date,
                                                                "observation_end": end_date,
                                                                "limit": limit,
                                                            }

                                                            data = self._make_request(
                                                                "series/observations", params=params)

                                                            if not data or "observations" not in data:
                                                                return None

                                                                # Convert to
                                                                # DataFrame
                                                                observations = data["observations"]
                                                                if not observations:
                                                                    return None

                                                                    df = pd.DataFrame(
                                                                        observations)
                                                                    df["date"] = pd.to_datetime(
                                                                        df["date"])
                                                                    df["value"] = pd.to_numeric(
                                                                        df["value"], errors="coerce")
                                                                    df = df[[
                                                                        "date", "value"]].dropna()
                                                                    df = df.set_index(
                                                                        "date")

                                                                    return EconomicData(
                                                                        series_id=series_id,
                                                                        name=info["title"],
                                                                        data=df,
                                                                        units=info["units"],
                                                                        frequency=info["frequency"],
                                                                        last_updated=datetime.strptime(
                                                                            info["last_updated"],
                                                                            "%Y-%m-%d %H:%M:%S%z"),
                                                                    )

                                                                    except Exception as e:
                                                                        logger.error(
                                                                            f"Error getting data for {series_id}: {e}")
                                                                        return None

                                                                        def get_popular_series(
                                                                                self) -> Dict[str, str]:
                                                                            """Get popular economic series"""
                                                                            return {
                                                                                "GDP": "GDP",
                                                                                "UNRATE": "Unemployment Rate",
                                                                                "CPIAUCSL": "Consumer Price Index",
                                                                                "DFF": "Federal Funds Rate",
                                                                                "DGS10": "10-Year Treasury Rate",
                                                                                "DEXUSEU": "USD/EUR Exchange Rate",
                                                                                "HOUST": "Housing Starts",
                                                                                "INDPRO": "Industrial Production Index",
                                                                                "PAYEMS": "Nonfarm Payrolls",
                                                                                "UMCSENT": "Consumer Sentiment",
                                                                            }

                                                                            def search_series(
                                                                                    self, query: str, limit: int = 10) -> List[Dict]:
                                                                                """Search for series by text"""
                                                                                try:
                                                                                    data = self._make_request(
                                                                                        "series/search", params={"search_text": query, "limit": limit})

                                                                                    if not data or "seriess" not in data:
                                                                                        return []

                                                                                        results = []
                                                                                        for series in data["seriess"]:
                                                                                            results.append(
                                                                                                {
                                                                                                    "id": series["id"],
                                                                                                    "title": series["title"],
                                                                                                    "units": series["units"],
                                                                                                    "frequency": series["frequency"],
                                                                                                    "observation_start": series["observation_start"],
                                                                                                    "observation_end": series["observation_end"],
                                                                                                }
                                                                                            )

                                                                                            return results

                                                                                            except Exception as e:
                                                                                                logger.error(
                                                                                                    f"Error searching series: {e}")
                                                                                                return []

                                                                                                # Global
                                                                                                # connector
                                                                                                # instance
                                                                                                _fred_connector = None

                                                                                                def get_fred_connector() -> FREDConnector:
                                                                                                    """Get global FRED connector instance"""
                                                                                                    global _fred_connector
                                                                                                    if _fred_connector is None:
                                                                                                        _fred_connector = FREDConnector()
                                                                                                        return _fred_connector
