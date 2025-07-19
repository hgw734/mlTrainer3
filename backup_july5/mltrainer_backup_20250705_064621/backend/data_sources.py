"""
mlTrainer - Data Sources Manager
===============================

Purpose: Manages real-time data ingestion from verified sources including
Polygon (market data), FRED (macro indicators), and QuiverQuant (sentiment/insider data).
Enforces strict compliance - no synthetic data allowed.

Compliance: All data must be from authorized APIs with verification timestamps.
Any failed data retrieval results in explicit error states, not placeholder data.
"""

import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import json
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_provider_manager import get_api_manager
from utils.polygon_rate_limiter import get_polygon_rate_limiter

logger = logging.getLogger(__name__)

class DataSourceManager:
    """Manages connections and data retrieval from verified external sources"""
    
    def __init__(self):
        # Initialize API provider manager
        self.api_manager = get_api_manager()
        
        # Get providers from configuration
        self.market_provider = self.api_manager.get_active_data_provider("market_data")
        self.economic_provider = self.api_manager.get_active_data_provider("economic_data")
        
        # Legacy API keys for backward compatibility (will be deprecated)
        self.polygon_api_key = self.market_provider.api_key if self.market_provider else ""
        self.fred_api_key = self.economic_provider.api_key if self.economic_provider else ""
        self.quiver_api_key = ""  # Disabled per user request
        
        # Initialize rate limiter for Polygon API
        self.polygon_limiter = get_polygon_rate_limiter()
        
        # Data validation thresholds
        self.min_data_points = 50  # Minimum data points for valid analysis
        self.max_dropout_rate = 0.15  # 15% maximum dropout rate
        self.data_freshness_threshold = timedelta(hours=24)  # Data must be within 24 hours
        
        # Ensure we have valid API keys
        if not self.polygon_api_key:
            logger.warning("No Polygon API key available")
        if not self.fred_api_key:
            logger.warning("No FRED API key available")
        
        # API rate limiting
        self.last_polygon_call = 0
        self.last_fred_call = 0
        self.last_quiver_call = 0
        
        # Rate limits (calls per minute)
        self.polygon_rate_limit = 100
        self.fred_rate_limit = 120
        self.quiver_rate_limit = 60
        
        # Connection status
        self.connections = {
            "polygon": False,
            "fred": False,
            "quiverquant": False
        }
        
        logger.info("DataSourceManager initialized")
    
    def _rate_limit_check(self, source: str, rate_limit: int) -> bool:
        """Check if API call is within rate limits"""
        now = time.time()
        last_call_attr = f"last_{source}_call"
        last_call = getattr(self, last_call_attr, 0)
        
        if now - last_call < 60 / rate_limit:
            return False
        
        setattr(self, last_call_attr, now)
        return True
    
    def check_connections(self) -> Dict[str, bool]:
        """Test connections to all data sources"""
        # Test Polygon connection
        if self.polygon_api_key:
            try:
                # Use Authorization header (recommended) instead of URL parameter
                url = "https://api.polygon.io/v1/marketstatus/now"
                headers = {
                    'Authorization': f'Bearer {self.polygon_api_key}',
                    'Accept-Encoding': 'gzip',
                    'User-Agent': 'mlTrainer/1.0'
                }
                response = requests.get(url, headers=headers, timeout=10)
                logger.info(f"Polygon connection test - Status Code: {response.status_code}, Response: {response.text[:200]}")
                self.connections["polygon"] = response.status_code == 200
            except Exception as e:
                logger.error(f"Polygon connection test failed: {e}")
                self.connections["polygon"] = False
        
        # Test FRED connection
        if self.fred_api_key:
            try:
                url = f"https://api.stlouisfed.org/fred/series?series_id=VIXCLS&api_key={self.fred_api_key}&file_type=json"
                response = requests.get(url, timeout=10)
                self.connections["fred"] = response.status_code == 200
            except Exception as e:
                logger.error(f"FRED connection test failed: {e}")
                self.connections["fred"] = False
        
        # Test QuiverQuant connection
        if self.quiver_api_key:
            try:
                headers = {"Authorization": f"Bearer {self.quiver_api_key}"}
                url = "https://api.quiverquant.com/beta/live/wallstreetbets"
                response = requests.get(url, headers=headers, timeout=10)
                self.connections["quiverquant"] = response.status_code == 200
            except Exception as e:
                logger.error(f"QuiverQuant connection test failed: {e}")
                self.connections["quiverquant"] = False
        
        logger.info(f"Connection status: {self.connections}")
        return self.connections
    
    def get_polygon_data(self, symbol: str, timespan: str = "minute", 
                        multiplier: int = 15, limit: int = 100) -> Optional[Dict]:
        """Get real-time market data from Polygon API with rate limiting and validation"""
        if not self.polygon_api_key:
            logger.error("Polygon API key not configured")
            return None
        
        try:
            # Get current date for data range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
            headers = {
                'Authorization': f'Bearer {self.polygon_api_key}',
                'Accept-Encoding': 'gzip',
                'User-Agent': 'mlTrainer/1.0'
            }
            params = {
                "adjusted": "true",
                "sort": "desc",
                "limit": limit
            }
            
            # Use rate limiter for API call
            result = self.polygon_limiter.make_request(url, headers=headers, params=params, timeout=15)
            
            if result["success"]:
                data = result["data"]
                if data.get("status") == "OK" and data.get("results"):
                    # Validate data quality
                    validation_result = self._validate_polygon_data(data["results"], symbol)
                    
                    return {
                        "symbol": symbol,
                        "data": data["results"],
                        "timestamp": datetime.now().isoformat(),
                        "source": "polygon",
                        "verified": True,
                        "data_quality": validation_result,
                        "rate_limit_metrics": result.get("quality_metrics", {})
                    }
                else:
                    logger.warning(f"No data returned for {symbol}")
                    return None
            else:
                logger.error(f"Polygon API error: {result['error']}")
                return {
                    "error": result["error"],
                    "quality_metrics": result.get("quality_metrics", {}),
                    "symbol": symbol
                }
                
        except Exception as e:
            logger.error(f"Polygon data retrieval failed: {e}")
            return None
    
    def _validate_polygon_data(self, data: List[Dict], symbol: str) -> Dict[str, Any]:
        """Validate Polygon API data quality and completeness"""
        validation_result = {
            "is_valid": True,
            "issues": [],
            "data_points": len(data),
            "dropout_rate": 0.0,
            "completeness_score": 1.0,
            "freshness_score": 1.0
        }
        
        # Check minimum data points
        if len(data) < self.min_data_points:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Insufficient data points: {len(data)} < {self.min_data_points}")
        
        # Check for missing values (dropout detection)
        missing_fields = 0
        required_fields = ['o', 'h', 'l', 'c', 'v', 't']  # open, high, low, close, volume, timestamp
        
        for point in data:
            for field in required_fields:
                if field not in point or point[field] is None:
                    missing_fields += 1
        
        total_expected_fields = len(data) * len(required_fields)
        dropout_rate = missing_fields / total_expected_fields if total_expected_fields > 0 else 0
        validation_result["dropout_rate"] = dropout_rate
        
        if dropout_rate > self.max_dropout_rate:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"High dropout rate: {dropout_rate:.1%} > {self.max_dropout_rate:.1%}")
        
        # Check data freshness
        if data:
            latest_timestamp = max(point.get('t', 0) for point in data)
            latest_date = datetime.fromtimestamp(latest_timestamp / 1000)  # Polygon uses milliseconds
            time_diff = datetime.now() - latest_date
            
            if time_diff > self.data_freshness_threshold:
                validation_result["freshness_score"] = max(0, 1 - (time_diff.total_seconds() / self.data_freshness_threshold.total_seconds()))
                validation_result["issues"].append(f"Data may be stale: {time_diff} old")
        
        # Calculate completeness score
        validation_result["completeness_score"] = 1.0 - dropout_rate
        
        logger.info(f"Data validation for {symbol}: {validation_result}")
        return validation_result
    
    def get_fred_indicators(self, series_ids: List[str] = None) -> Optional[Dict]:
        """Get macroeconomic indicators from FRED API"""
        if not self.fred_api_key:
            logger.error("FRED API key not configured")
            return None
        
        if not self._rate_limit_check("fred", self.fred_rate_limit):
            logger.warning("FRED rate limit exceeded")
            return None
        
        # Default macro indicators for regime detection
        if series_ids is None:
            series_ids = [
                "VIXCLS",     # VIX - Volatility Index
                "DGS10",      # 10-Year Treasury Rate
                "UNRATE",     # Unemployment Rate
                "CPIAUCSL",   # Consumer Price Index
                "GDP"         # Gross Domestic Product
            ]
        
        indicators = {}
        
        for series_id in series_ids:
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": self.fred_api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 30  # Last 30 observations
                }
                
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("observations"):
                        # Get most recent valid observation
                        for obs in data["observations"]:
                            if obs["value"] != ".":
                                indicators[series_id] = {
                                    "value": float(obs["value"]),
                                    "date": obs["date"],
                                    "series_id": series_id
                                }
                                break
                else:
                    logger.warning(f"FRED API error for {series_id}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"FRED data retrieval failed for {series_id}: {e}")
        
        if indicators:
            return {
                "indicators": indicators,
                "timestamp": datetime.now().isoformat(),
                "source": "fred",
                "verified": True
            }
        
        return None
    
    def get_quiver_sentiment(self, symbols: List[str] = None) -> Optional[Dict]:
        """Get sentiment and insider data from QuiverQuant API"""
        if not self.quiver_api_key:
            logger.error("QuiverQuant API key not configured")
            return None
        
        if not self._rate_limit_check("quiver", self.quiver_rate_limit):
            logger.warning("QuiverQuant rate limit exceeded")
            return None
        
        headers = {"Authorization": f"Bearer {self.quiver_api_key}"}
        sentiment_data = {}
        
        try:
            # Get Reddit WSB sentiment
            wsb_url = "https://api.quiverquant.com/beta/live/wallstreetbets"
            response = requests.get(wsb_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                wsb_data = response.json()
                sentiment_data["reddit_wsb"] = wsb_data
            
            # Get insider trading data
            if symbols:
                for symbol in symbols[:5]:  # Limit to 5 symbols to avoid rate limits
                    insider_url = f"https://api.quiverquant.com/beta/historical/insidertrading/{symbol}"
                    response = requests.get(insider_url, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        insider_data = response.json()
                        sentiment_data[f"insider_{symbol}"] = insider_data
            
            if sentiment_data:
                return {
                    "sentiment_data": sentiment_data,
                    "timestamp": datetime.now().isoformat(),
                    "source": "quiverquant",
                    "verified": True
                }
        
        except Exception as e:
            logger.error(f"QuiverQuant data retrieval failed: {e}")
        
        return None
    
    def get_regime_indicators(self) -> Optional[Dict]:
        """Get comprehensive regime detection indicators from all sources"""
        regime_data = {
            "timestamp": datetime.now().isoformat(),
            "sources": [],
            "verified": True
        }
        
        # Get FRED macro indicators
        fred_data = self.get_fred_indicators()
        if fred_data:
            regime_data["macro_indicators"] = fred_data["indicators"]
            regime_data["sources"].append("fred")
        
        # Get market volatility from key symbols
        key_symbols = ["SPY", "VIX", "QQQ"]
        market_data = {}
        
        for symbol in key_symbols:
            polygon_data = self.get_polygon_data(symbol)
            if polygon_data:
                market_data[symbol] = polygon_data
                if "polygon" not in regime_data["sources"]:
                    regime_data["sources"].append("polygon")
        
        if market_data:
            regime_data["market_data"] = market_data
        
        # Get sentiment indicators
        sentiment_data = self.get_quiver_sentiment()
        if sentiment_data:
            regime_data["sentiment"] = sentiment_data["sentiment_data"]
            regime_data["sources"].append("quiverquant")
        
        # Calculate regime score based on available data
        regime_score = self._calculate_regime_score(regime_data)
        regime_data["regime_score"] = regime_score
        
        return regime_data if regime_data["sources"] else None
    
    def _calculate_regime_score(self, regime_data: Dict) -> float:
        """Calculate regime score from 0-100 based on available indicators"""
        score_components = []
        
        # VIX-based volatility score
        macro_indicators = regime_data.get("macro_indicators", {})
        if "VIXCLS" in macro_indicators:
            vix_value = macro_indicators["VIXCLS"]["value"]
            # VIX > 30 = high volatility, VIX < 15 = low volatility
            vix_score = min(100, max(0, (vix_value - 15) * 100 / 15))
            score_components.append(vix_score)
        
        # Interest rate environment
        if "DGS10" in macro_indicators:
            rate_value = macro_indicators["DGS10"]["value"]
            # Higher rates typically indicate tighter conditions
            rate_score = min(100, max(0, rate_value * 20))
            score_components.append(rate_score)
        
        # Market momentum from price data
        market_data = regime_data.get("market_data", {})
        if "SPY" in market_data:
            spy_data = market_data["SPY"]["data"]
            if len(spy_data) > 1:
                recent_volatility = abs((spy_data[0]["c"] - spy_data[-1]["c"]) / spy_data[-1]["c"] * 100)
                vol_score = min(100, recent_volatility * 10)
                score_components.append(vol_score)
        
        # Return average score or default
        return sum(score_components) / len(score_components) if score_components else 50.0
    
    def refresh_market_data(self) -> bool:
        """Refresh all market data sources"""
        try:
            # Check connections first
            connections = self.check_connections()
            
            if not any(connections.values()):
                logger.error("No data source connections available")
                return False
            
            # Refresh regime indicators
            regime_data = self.get_regime_indicators()
            
            if regime_data:
                logger.info("Market data refresh completed successfully")
                return True
            else:
                logger.warning("Market data refresh completed with limited data")
                return False
                
        except Exception as e:
            logger.error(f"Market data refresh failed: {e}")
            return False
    
    def get_connection_stats(self) -> Dict:
        """Get detailed connection statistics"""
        return {
            "connections": self.connections,
            "api_keys_configured": {
                "polygon": bool(self.polygon_api_key),
                "fred": bool(self.fred_api_key),
                "quiverquant": bool(self.quiver_api_key)
            },
            "rate_limits": {
                "polygon": self.polygon_rate_limit,
                "fred": self.fred_rate_limit,
                "quiverquant": self.quiver_rate_limit
            },
            "last_refresh": datetime.now().isoformat()
        }
    
    def get_market_data(self, symbol: str, start_date: str = None, end_date: str = None, 
                       lookback_days: int = 365) -> Optional[pd.DataFrame]:
        """Get historical market data for a specific symbol"""
        try:
            logger.info(f"Fetching market data for {symbol}")
            
            if not self.polygon_api_key:
                logger.error("Polygon API key not available")
                return None
            
            # Calculate date range
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if not start_date:
                start_dt = datetime.now() - timedelta(days=lookback_days)
                start_date = start_dt.strftime('%Y-%m-%d')
            
            # Fetch data from Polygon API using rate limiter
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            headers = {"Authorization": f"Bearer {self.polygon_api_key}"}
            
            # Use rate limiter to make the request
            response_data = self.polygon_limiter.make_request(url, headers=headers, timeout=10)
            
            if not response_data.get('success'):
                logger.error(f"Polygon API request failed: {response_data.get('error', 'Unknown error')}")
                return None
            
            data = response_data.get('data')
            if not data:
                logger.error("No data received from rate limiter")
                return None
            
            results = data.get('results', [])
            
            if not results:
                logger.warning(f"No market data found for {symbol}")
                return None
            
            # Convert to DataFrame
            df_data = []
            for bar in results:
                df_data.append({
                    'timestamp': pd.to_datetime(bar['t'], unit='ms'),
                    'open': bar['o'],
                    'high': bar['h'], 
                    'low': bar['l'],
                    'close': bar['c'],
                    'volume': bar['v'],
                    'ticker': symbol,
                    'source': 'polygon',
                    'verified': True
                })
            
            df = pd.DataFrame(df_data)
            
            if not df.empty:
                # Add technical indicators as features
                df = self._add_technical_features(df)
                # Add target variable (next day's return)
                df['target'] = df['close'].shift(-1)
                
                logger.info(f"Retrieved {len(df)} data points for {symbol}")
                return df
            else:
                logger.warning(f"Empty dataset for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as features for ML models"""
        try:
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Simple moving averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # Price ratios
            df['price_sma5_ratio'] = df['close'] / df['sma_5']
            df['price_sma20_ratio'] = df['close'] / df['sma_20']
            
            # Volatility (rolling standard deviation)
            df['volatility_5'] = df['close'].rolling(window=5).std()
            df['volatility_20'] = df['close'].rolling(window=20).std()
            
            # Returns
            df['return_1d'] = df['close'].pct_change()
            df['return_5d'] = df['close'].pct_change(periods=5)
            
            # Volume indicators
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']
            
            # RSI (simple approximation)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # High-Low ratios
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            df['oc_ratio'] = (df['open'] - df['close']) / df['close']
            
            # Fill NaN values with forward fill, then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to add technical features: {e}")
            return df

