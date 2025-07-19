import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import requests
import time
from datetime import datetime, timedelta
import os
from polygon import RESTClient
import json
import pytz

logger = logging.getLogger(__name__)

class DataProvider:
    """
    Compliant data provider for institutional-grade stock analysis
    using ONLY Polygon.io for stock data and FRED for VIX data.
    """
    
    def __init__(self):
        """Initialize data provider with ONLY authorized API configurations"""
        # Import centralized API key management
        from config.api_keys import get_polygon_key
        
        # API Keys - ONLY authorized sources
        self.polygon_api_key = get_polygon_key()
        
        # Initialize Polygon client
        try:
            if self.polygon_api_key:
                self.polygon_client = RESTClient(api_key=self.polygon_api_key)
                logger.info(f"Polygon.io client initialized successfully with key: {self.polygon_api_key[:10]}...")
                
                # Test the connection
                test_response = self.polygon_client.get_ticker_details('AAPL')
                logger.info(f"API test successful: {test_response.name if hasattr(test_response, 'name') else 'Connection verified'}")
            else:
                self.polygon_client = None
                logger.error("Polygon.io API key not configured")
        except Exception as e:
            self.polygon_client = None
            logger.error(f"Failed to initialize Polygon.io client: {e}")
        
        # API endpoints - ONLY authorized sources
        self.polygon_base_url = "https://api.polygon.io"
        
        # Rate limiting
        self.last_request_times = {}
        self.rate_limits = {
            'polygon': 0.1  # 10 requests per second for Polygon
        }
        
        # Usage tracking
        self.api_usage_today = {
            'polygon_calls_today': 0,
            'total_calls_today': 0
        }
        
        logger.info("Compliant data provider initialized - ONLY Polygon API for stock data")
    
    def get_price_data(self, symbol: str, period: str = '1mo', interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Get stock price data from Polygon API ONLY
        
        Args:
            symbol: Stock symbol
            period: Time period ('1mo', '3mo', '6mo', '1y', '2y')
            interval: Data interval ('1d', '1h', '5m', '15m')
        
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            if not self.polygon_client:
                logger.error("Polygon client not available")
                return None
            
            # Calculate date range based on period
            end_date = datetime.now()
            period_days = {
                '1mo': 30, '3mo': 90, '6mo': 180, 
                '1y': 365, '2y': 730, '5y': 1825
            }
            start_date = end_date - timedelta(days=period_days.get(period, 90))
            
            # Convert interval for Polygon API
            if interval == '1d':
                multiplier, timespan = 1, 'day'
            elif interval == '1h':
                multiplier, timespan = 1, 'hour'
            elif interval == '5m':
                multiplier, timespan = 5, 'minute'
            elif interval == '15m':
                multiplier, timespan = 15, 'minute'
            else:
                multiplier, timespan = 1, 'day'
            
            self._enforce_rate_limit('polygon')
            
            # Get aggregates from Polygon
            aggs = self.polygon_client.get_aggs(
                ticker=symbol.upper(),
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                adjusted=True,
                sort='asc',
                limit=50000
            )
            
            if not aggs or len(aggs) == 0:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            self._update_api_usage('polygon')
            logger.info(f"Retrieved {len(df)} data points for {symbol} from Polygon API")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price from Polygon API ONLY
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None if error
        """
        try:
            if not self.polygon_client:
                logger.error("Polygon client not available")
                return None
            
            self._enforce_rate_limit('polygon')
            
            # Get last trade from Polygon
            last_trade = self.polygon_client.get_last_trade(ticker=symbol.upper())
            
            if last_trade and hasattr(last_trade, 'price'):
                self._update_api_usage('polygon')
                return float(last_trade.price)
            else:
                logger.warning(f"No current price available for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def get_financial_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get financial data from Polygon API ONLY
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with financial metrics or None if error
        """
        try:
            if not self.polygon_client:
                logger.error("Polygon client not available")
                return None
            
            self._enforce_rate_limit('polygon')
            
            # Get ticker details from Polygon
            details = self.polygon_client.get_ticker_details(symbol.upper())
            
            if not details:
                logger.warning(f"No financial data available for {symbol}")
                return None
            
            # Get financials if available
            financials = None
            try:
                financials = self.polygon_client.vx.list_stock_financials(
                    ticker=symbol.upper(),
                    limit=1
                )
                if financials and len(financials) > 0:
                    financials = financials[0]
            except:
                pass
            
            financial_data = {
                'symbol': symbol.upper(),
                'name': getattr(details, 'name', ''),
                'market_cap': getattr(details, 'market_cap', 0),
                'shares_outstanding': getattr(details, 'share_class_shares_outstanding', 0),
                'weighted_shares_outstanding': getattr(details, 'weighted_shares_outstanding', 0),
            }
            
            # Add financials if available
            if financials:
                financial_data.update({
                    'revenue': getattr(financials, 'revenues', 0),
                    'net_income': getattr(financials, 'net_income_loss', 0),
                    'total_assets': getattr(financials, 'assets', 0),
                    'total_debt': getattr(financials, 'liabilities', 0),
                })
            
            self._update_api_usage('polygon')
            return financial_data
            
        except Exception as e:
            logger.error(f"Failed to get financial data for {symbol}: {e}")
            return None
    
    def _enforce_rate_limit(self, api_name: str):
        """Enforce rate limiting for API calls"""
        current_time = time.time()
        last_request = self.last_request_times.get(api_name, 0)
        time_diff = current_time - last_request
        min_interval = self.rate_limits.get(api_name, 0.1)
        
        if time_diff < min_interval:
            sleep_time = min_interval - time_diff
            time.sleep(sleep_time)
        
        self.last_request_times[api_name] = time.time()
    
    def _update_api_usage(self, api_name: str):
        """Update API usage tracking"""
        if api_name == 'polygon':
            self.api_usage_today['polygon_calls_today'] += 1
        self.api_usage_today['total_calls_today'] += 1
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Get API connection status - ONLY authorized APIs
        
        Returns:
            Dictionary with API status information
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'polygon': False,
            'total_calls_today': self.api_usage_today['total_calls_today'],
            'polygon_calls_today': self.api_usage_today['polygon_calls_today']
        }
        
        # Test Polygon connection
        try:
            if self.polygon_api_key:
                response = requests.get(
                    f"{self.polygon_base_url}/v3/reference/tickers/AAPL",
                    params={'apikey': self.polygon_api_key},
                    timeout=10
                )
                results['polygon'] = response.status_code == 200
            else:
                results['polygon'] = False
        except:
            results['polygon'] = False
        
        return results
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get API usage summary - ONLY authorized APIs
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'polygon_calls': self.api_usage_today['polygon_calls_today'],
            'total_calls': self.api_usage_today['total_calls_today'],
            'rate_limits': self.rate_limits,
            'compliant': True,
            'authorized_sources': ['Polygon API'],
            'unauthorized_sources_removed': ['Alpha Vantage', 'News API', 'Yahoo Finance']
        }