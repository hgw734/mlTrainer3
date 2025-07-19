"""
Polygon API Connector - Market Data Interface
============================================

Simplified connector for Polygon.io API using rate limiter and existing configuration.
Provides real-time and historical market data with compliance verification.
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import rate limiter
from polygon_rate_limiter import get_polygon_rate_limiter

# Import from existing configuration
from config.api_config import POLYGON_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data structure"""

    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    previous_close: float
    timestamp: datetime
    source: str = "polygon"

    @dataclass
    class HistoricalData:
        """Historical data structure"""

        symbol: str
        data: pd.DataFrame
        timeframe: str
        start_date: datetime
        end_date: datetime
        source: str = "polygon"

        class PolygonConnector:
            """Polygon API connector with rate limiting"""

            def __init__(self):
                self.api_key = POLYGON_API_KEY
                self.base_url = "https://api.polygon.io"
                self.rate_limiter = get_polygon_rate_limiter()

                if not self.api_key:
                    raise ValueError("Polygon API key not configured")

                    logger.info(
                        "Polygon connector initialized with rate limiting")

                    def _make_request(
                            self,
                            endpoint: str,
                            params: Optional[Dict] = None) -> Optional[Dict]:
                        """Make rate-limited API request"""
                        url = f"{self.base_url}{endpoint}"

                        # Add API key to params
                        if params is None:
                            params = {}
                            params["apikey"] = self.api_key

                            # Use rate limiter
                            result = self.rate_limiter.make_request(
                                url, params=params)

                            if result["success"]:
                                return result["data"]
                                else:
                                    logger.error(
                                        f"API request failed: {result['error']}")
                                    return None

                                    def get_quote(
                                            self, symbol: str) -> Optional[MarketData]:
                                        """Get real-time quote for a symbol"""
                                        try:
                                            # Get previous close
                                            prev_data = self._make_request(
                                                f"/v2/aggs/ticker/{symbol}/prev")
                                            if not prev_data or "results" not in prev_data:
                                                return None

                                                prev_close = prev_data["results"][0]["c"]

                                                # Get current data
                                                today = datetime.now().strftime("%Y-%m-%d")
                                                agg_data = self._make_request(
                                                    f"/v2/aggs/ticker/{symbol}/range/1/minute/{today}/{today}",
                                                    params={
                                                        "limit": 1,
                                                        "sort": "desc"})

                                                if not agg_data or "results" not in agg_data or not agg_data[
                                                        "results"]:
                                                    # Fallback to previous
                                                    # day's data
                                                    current_price = prev_close
                                                    volume = 0
                                                    high = low = open_price = prev_close
                                                    else:
                                                        latest = agg_data["results"][0]
                                                        current_price = latest["c"]
                                                        volume = latest["v"]
                                                        high = latest["h"]
                                                        low = latest["l"]
                                                        open_price = latest["o"]

                                                        # Calculate change
                                                        change = current_price - prev_close
                                                        change_percent = (
                                                            change / prev_close) * 100 if prev_close > 0 else 0

                                                        return MarketData(
                                                            symbol=symbol.upper(),
                                                            price=current_price,
                                                            change=change,
                                                            change_percent=change_percent,
                                                            volume=volume,
                                                            high=high,
                                                            low=low,
                                                            open=open_price,
                                                            previous_close=prev_close,
                                                            timestamp=datetime.now(),
                                                        )

                                                        except Exception as e:
                                                            logger.error(
                                                                f"Error getting quote for {symbol}: {e}")
                                                            return None

                                                            def get_historical_data(
                                                                    self, symbol: str, days: int = 365, timespan: str = "day") -> Optional[HistoricalData]:
                                                                """Get historical data for a symbol"""
                                                                try:
                                                                    end_date = datetime.now().strftime("%Y-%m-%d")
                                                                    start_date = (
                                                                        datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

                                                                    data = self._make_request(
                                                                        f"/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start_date}/{end_date}",
                                                                        params={"adjusted": "true", "sort": "asc", "limit": 50000},
                                                                    )

                                                                    if not data or "results" not in data:
                                                                        return None

                                                                        # Convert
                                                                        # to
                                                                        # DataFrame
                                                                        df = pd.DataFrame(
                                                                            data["results"])
                                                                        if df.empty:
                                                                            return None

                                                                            # Rename
                                                                            # columns
                                                                            df = df.rename(
                                                                                columns={
                                                                                    "t": "timestamp",
                                                                                    "o": "open",
                                                                                    "h": "high",
                                                                                    "l": "low",
                                                                                    "c": "close",
                                                                                    "v": "volume"})

                                                                            # Convert
                                                                            # timestamp
                                                                            df["datetime"] = pd.to_datetime(
                                                                                df["timestamp"], unit="ms")
                                                                            df = df.set_index(
                                                                                "datetime")

                                                                            return HistoricalData(
                                                                                symbol=symbol.upper(),
                                                                                data=df,
                                                                                timeframe=timespan,
                                                                                start_date=datetime.strptime(start_date, "%Y-%m-%d"),
                                                                                end_date=datetime.strptime(end_date, "%Y-%m-%d"),
                                                                            )

                                                                            except Exception as e:
                                                                                logger.error(
                                                                                    f"Error getting historical data for {symbol}: {e}")
                                                                                return None

                                                                                def get_quality_metrics(
                                                                                        self) -> Dict[str, Any]:
                                                                                    """Get current API quality metrics"""
                                                                                    return self.rate_limiter.get_quality_summary()

                                                                                    # Global
                                                                                    # connector
                                                                                    # instance
                                                                                    _polygon_connector = None

                                                                                    def get_polygon_connector() -> PolygonConnector:
                                                                                        """Get global Polygon connector instance"""
                                                                                        global _polygon_connector
                                                                                        if _polygon_connector is None:
                                                                                            _polygon_connector = PolygonConnector()
                                                                                            return _polygon_connector
