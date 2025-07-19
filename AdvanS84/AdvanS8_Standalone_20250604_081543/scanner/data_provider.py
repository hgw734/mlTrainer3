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
    Professional data provider for institutional-grade stock analysis
    using Polygon.io and other financial APIs.
    """
    
    def __init__(self):
        """Initialize data provider with API configurations"""
        # Import centralized API key management
        from config.api_keys import get_polygon_key
        
        # API Keys from centralized config - ONLY Polygon for stock data
        self.polygon_api_key = get_polygon_key()
        # REMOVED: Alpha Vantage - violates data integrity policy
        # REMOVED: News API - not required for stock market data
        
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
        
        # API endpoints - ONLY Polygon for stock data
        self.polygon_base_url = "https://api.polygon.io"
        
        # Rate limiting
        self.last_request_times = {}
        self.rate_limits = {
            'polygon': 80,  # 100 requests per second, using 80 for optimal performance
            'news': 1000,  # 1000 requests per day
            'alpha_vantage': 5  # 5 requests per minute
        }
        
        # API usage tracking
        self.api_usage = {
            'polygon_calls_today': 0,
            'news_calls_today': 0,
            'alpha_vantage_calls_today': 0,
            'last_reset': datetime.now().date()
        }
        
        # Market hours configuration
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.market_open_time = "09:30"
        self.market_close_time = "16:00"
    
    def is_market_holiday(self, date: datetime) -> bool:
        """Check if a given date is a US stock market holiday"""
        try:
            year = date.year
            
            # Fixed date holidays
            holidays = [
                datetime(year, 1, 1),   # New Year's Day
                datetime(year, 7, 4),   # Independence Day
                datetime(year, 12, 25), # Christmas Day
            ]
            
            # Third Monday in January (MLK Day)
            jan_1 = datetime(year, 1, 1)
            days_to_third_monday = (21 - jan_1.weekday()) % 7
            if days_to_third_monday == 0:
                days_to_third_monday = 14
            holidays.append(jan_1 + timedelta(days=days_to_third_monday))
            
            # Third Monday in February (Presidents Day)
            feb_1 = datetime(year, 2, 1)
            days_to_third_monday = (21 - feb_1.weekday()) % 7
            if days_to_third_monday == 0:
                days_to_third_monday = 14
            holidays.append(feb_1 + timedelta(days=days_to_third_monday))
            
            # Last Monday in May (Memorial Day)
            may_31 = datetime(year, 5, 31)
            days_back_to_monday = (may_31.weekday() + 1) % 7
            holidays.append(may_31 - timedelta(days=days_back_to_monday))
            
            # Juneteenth (June 19)
            holidays.append(datetime(year, 6, 19))
            
            # First Monday in September (Labor Day)
            sep_1 = datetime(year, 9, 1)
            days_to_monday = (7 - sep_1.weekday()) % 7
            holidays.append(sep_1 + timedelta(days=days_to_monday))
            
            # Fourth Thursday in November (Thanksgiving)
            nov_1 = datetime(year, 11, 1)
            days_to_thursday = (3 - nov_1.weekday()) % 7
            thanksgiving = nov_1 + timedelta(days=days_to_thursday + 21)
            holidays.append(thanksgiving)
            
            # Good Friday (Friday before Easter) - Complex calculation
            # Using simplified approximation for most years
            easter_dates = {
                2024: datetime(2024, 3, 31),
                2025: datetime(2025, 4, 20),
                2026: datetime(2026, 4, 5),
                2027: datetime(2027, 3, 28),
                2028: datetime(2028, 4, 16),
            }
            if year in easter_dates:
                good_friday = easter_dates[year] - timedelta(days=2)
                holidays.append(good_friday)
            
            # Check if date matches any holiday (ignoring time)
            check_date = date.date() if hasattr(date, 'date') else date
            for holiday in holidays:
                if holiday.date() == check_date:
                    return True
                    
            return False
            
        except Exception as e:
            logger.warning(f"Error checking market holidays: {e}")
            return False

    def is_market_open(self) -> bool:
        """Check if the US stock market is currently open"""
        try:
            # Get current Eastern time
            et_now = datetime.now(self.eastern_tz)
            current_time = et_now.time()
            current_weekday = et_now.weekday()
            
            # Market is closed on weekends (Saturday=5, Sunday=6)
            if current_weekday >= 5:
                return False
            
            # Market is closed on holidays
            if self.is_market_holiday(et_now):
                return False
            
            # Check if current time is within market hours
            open_time = datetime.strptime(self.market_open_time, "%H:%M").time()
            close_time = datetime.strptime(self.market_close_time, "%H:%M").time()
            
            return open_time <= current_time <= close_time
            
        except Exception as e:
            logger.warning(f"Error checking market hours: {e}")
            return False
    
    def get_last_market_close_date(self) -> datetime:
        """Get the date of the last market close, accounting for holidays and weekends"""
        try:
            et_now = datetime.now(self.eastern_tz)
            
            # Start from today and work backwards to find the last trading day
            check_date = et_now
            
            # If market is currently open, use yesterday as starting point
            if self.is_market_open():
                check_date = et_now - timedelta(days=1)
            # If it's after market close today and today was a trading day, use today
            elif (et_now.time() > datetime.strptime(self.market_close_time, "%H:%M").time() and
                  et_now.weekday() < 5 and not self.is_market_holiday(et_now)):
                return et_now
            else:
                # Market hasn't opened today or today is not a trading day
                check_date = et_now - timedelta(days=1)
            
            # Find the most recent trading day
            max_days_back = 10  # Safety limit
            for i in range(max_days_back):
                # Skip weekends
                if check_date.weekday() >= 5:
                    check_date = check_date - timedelta(days=1)
                    continue
                
                # Skip holidays
                if self.is_market_holiday(check_date):
                    check_date = check_date - timedelta(days=1)
                    continue
                
                # Found a valid trading day
                return check_date
            
            # Fallback if we can't find a trading day within 10 days
            logger.warning("Could not find a recent trading day within 10 days")
            return et_now - timedelta(days=1)
                    
        except Exception as e:
            logger.warning(f"Error getting last market close date: {e}")
            return datetime.now() - timedelta(days=1)
    
    def get_available_date_range(self) -> Dict[str, Any]:
        """
        Dynamically determine the maximum available historical data range
        by checking actual data availability from Polygon API
        """
        try:
            if not self.polygon_client:
                # Return reasonable defaults if no API access
                return {
                    'earliest_date': pd.to_datetime("2021-01-01"),
                    'latest_date': pd.to_datetime.now().date() - timedelta(days=1),
                    'data_source': 'estimated',
                    'verified': False
                }
            
            # Test with a liquid stock to check actual data availability
            test_symbol = 'SPY'  # Highly liquid ETF, should have maximum data range
            
            # Check earliest available data (go back progressively)
            earliest_date = None
            for years_back in [4, 3, 2, 1]:
                test_date = datetime.now() - timedelta(days=years_back * 365)
                try:
                    # Try to get data for this date
                    aggs = self.polygon_client.get_aggs(
                        ticker=test_symbol,
                        multiplier=1,
                        timespan="day",
                        from_=test_date.strftime('%Y-%m-%d'),
                        to_=(test_date + timedelta(days=5)).strftime('%Y-%m-%d'),
                        limit=5
                    )
                    
                    if aggs and len(aggs) > 0:
                        earliest_date = test_date.date()
                        break
                        
                    time.sleep(0.1)  # Small delay between checks
                    
                except Exception as e:
                    logger.debug(f"Date check failed for {test_date}: {e}")
                    continue
            
            # Check latest available data (should be recent trading day)
            latest_date = datetime.now().date()
            for days_back in range(5):  # Check last 5 days for latest trading day
                check_date = latest_date - timedelta(days=days_back)
                try:
                    aggs = self.polygon_client.get_aggs(
                        ticker=test_symbol,
                        multiplier=1,
                        timespan="day", 
                        from_=check_date.strftime('%Y-%m-%d'),
                        to_=check_date.strftime('%Y-%m-%d'),
                        limit=1
                    )
                    
                    if aggs and len(aggs) > 0:
                        latest_date = check_date
                        break
                        
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"Latest date check failed for {check_date}: {e}")
                    continue
            
            return {
                'earliest_date': pd.to_datetime(earliest_date) if earliest_date else pd.to_datetime("2021-01-01"),
                'latest_date': latest_date,
                'data_source': 'polygon_verified',
                'verified': True,
                'test_symbol': test_symbol
            }
            
        except Exception as e:
            logger.warning(f"Could not verify date range from API: {e}")
            # Return conservative estimates if verification fails
            return {
                'earliest_date': pd.to_datetime("2021-01-01"),
                'latest_date': pd.to_datetime.now().date() - timedelta(days=1),
                'data_source': 'fallback_estimate',
                'verified': False,
                'error': str(e)
            }
    
    def get_market_data(self, symbol: str, timespan: str = "day", 
                       multiplier: int = 1, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get market data (OHLCV) for a symbol
        
        Args:
            symbol: Stock symbol
            timespan: Timespan ('minute', 'hour', 'day', 'week', 'month')
            multiplier: Multiplier for timespan
            limit: Number of bars to return
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            if not self.polygon_client:
                logger.error("Polygon client not initialized")
                return None
            
            # Rate limiting
            self._enforce_rate_limit('polygon')
            
            # Calculate date range based on market status
            if self.is_market_open():
                # Market is open, use current date
                end_date = datetime.now().date()
            else:
                # Market is closed, use last market close date
                last_close = self.get_last_market_close_date()
                end_date = last_close.date()
                logger.info(f"Market closed, using last trading day data: {end_date}")
            
            start_date = end_date - timedelta(days=limit * 2)  # Extra buffer
            
            # Get aggregates from Polygon
            aggs = self.polygon_client.get_aggs(
                ticker=symbol.upper(),
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                limit=limit
            )
            
            if not aggs or len(aggs) == 0:
                logger.warning(f"No market data returned for {symbol}")
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
                    'volume': agg.volume,
                    'vwap': getattr(agg, 'vwap', None)
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Update API usage
            self._update_api_usage('polygon')
            
            logger.info(f"Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def get_price_data(self, symbol: str, period: str = "1y", interval: str = "1d", **kwargs) -> Optional[pd.DataFrame]:
        """
        Alias for get_market_data to maintain compatibility with scanner expectations
        
        Args:
            symbol: Stock symbol
            period: Time period (e.g., '1y', '6mo', '3mo') - converted to limit
            interval: Data interval (e.g., '1d', '1h') - converted to timespan
            **kwargs: Additional parameters for backward compatibility
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Convert period to limit (approximate)
        period_to_limit = {
            '1y': 252,   # ~252 trading days in a year
            '6mo': 126,  # ~126 trading days in 6 months
            '3mo': 63,   # ~63 trading days in 3 months
            '1mo': 21,   # ~21 trading days in a month
            '1w': 5,     # ~5 trading days in a week
            '5d': 5,     # 5 days
            '1d': 1,     # 1 day
        }
        
        # Convert interval to timespan
        interval_to_timespan = {
            '1d': 'day',
            '1h': 'hour',
            '30m': 'minute',
            '15m': 'minute',
            '5m': 'minute',
            '1m': 'minute'
        }
        
        # Extract parameters
        limit = period_to_limit.get(period, 100)
        timespan = interval_to_timespan.get(interval, 'day')
        
        # Handle minute intervals with multiplier
        multiplier = 1
        if interval in ['30m', '15m', '5m']:
            multiplier = int(interval[:-1])  # Extract number from '30m', '15m', etc.
        
        return self.get_market_data(symbol, timespan, multiplier, limit)
    
    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company information and fundamentals"""
        try:
            if not self.polygon_client:
                return None
            
            self._enforce_rate_limit('polygon')
            
            # Get ticker details
            ticker_details = self.polygon_client.get_ticker_details(symbol.upper())
            
            if not ticker_details:
                return None
            
            company_info = {
                'symbol': symbol.upper(),
                'name': getattr(ticker_details, 'name', ''),
                'market_cap': getattr(ticker_details, 'market_cap', 0),
                'shares_outstanding': getattr(ticker_details, 'share_class_shares_outstanding', 0),
                'sector': getattr(ticker_details, 'sic_description', ''),
                'industry': getattr(ticker_details, 'sic_description', ''),
                'website': getattr(ticker_details, 'homepage_url', ''),
                'description': getattr(ticker_details, 'description', ''),
                'employees': getattr(ticker_details, 'total_employees', 0),
                'exchange': getattr(ticker_details, 'primary_exchange', ''),
                'currency': getattr(ticker_details, 'currency_name', 'USD')
            }
            
            self._update_api_usage('polygon')
            return company_info
            
        except Exception as e:
            logger.error(f"Failed to get company info for {symbol}: {e}")
            return None
    
    def get_financial_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get financial data including earnings, revenue, etc."""
        try:
            if not self.polygon_client:
                return None
            
            self._enforce_rate_limit('polygon')
            
            # Get financials from Polygon
            financials = self.polygon_client.get_ticker_details(symbol.upper())
            
            if not financials:
                return None
            
            financial_data = {
                'symbol': symbol.upper(),
                'market_cap': getattr(financials, 'market_cap', 0),
                'shares_outstanding': getattr(financials, 'share_class_shares_outstanding', 0),
                'weighted_shares_outstanding': getattr(financials, 'weighted_shares_outstanding', 0),
            }
            
            # REMOVED: Alpha Vantage backup - violates data integrity policy
            # Only Polygon API is authorized for stock market data
            
            self._update_api_usage('polygon')
            return financial_data
            
        except Exception as e:
            logger.error(f"Failed to get financial data for {symbol}: {e}")
            return None
    
    def get_news_data(self, symbol: str, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        """Get news articles for a symbol"""
        try:
            news_articles = []
            
            # Try Polygon news first
            if self.polygon_client:
                try:
                    self._enforce_rate_limit('polygon')
                    
                    # Get news from Polygon
                    news = self.polygon_client.list_ticker_news(
                        ticker=symbol.upper(),
                        limit=limit
                    )
                    
                    if news:
                        for article in news:
                            # Handle publisher object correctly
                            publisher = getattr(article, 'publisher', None)
                            if publisher:
                                if hasattr(publisher, 'name'):
                                    source_name = publisher.name
                                elif isinstance(publisher, dict):
                                    source_name = publisher.get('name', 'Unknown')
                                else:
                                    source_name = str(publisher)
                            else:
                                source_name = 'Unknown'
                            
                            news_articles.append({
                                'title': getattr(article, 'title', ''),
                                'description': getattr(article, 'description', ''),
                                'url': getattr(article, 'article_url', ''),
                                'published_at': getattr(article, 'published_utc', ''),
                                'source': source_name,
                                'sentiment': None  # Will be analyzed separately
                            })
                    
                    self._update_api_usage('polygon')
                    
                except Exception as e:
                    logger.warning(f"Polygon news failed for {symbol}: {e}")
            
            # REMOVED: NewsAPI - violates data integrity policy
            # Only Polygon API is authorized for stock market data
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for article in data.get('articles', []):
                            news_articles.append({
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'url': article.get('url', ''),
                                'published_at': article.get('publishedAt', ''),
                                'source': article.get('source', {}).get('name', 'Unknown'),
                                'sentiment': None
                            })
                    
                    self._update_api_usage('news')
                    
                except Exception as e:
                    logger.warning(f"NewsAPI failed for {symbol}: {e}")
            
            return news_articles if news_articles else None
            
        except Exception as e:
            logger.error(f"Failed to get news for {symbol}: {e}")
            return None
    
    def get_analyst_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get analyst ratings and recommendations"""
        try:
            # For now, we'll use Alpha Vantage for analyst data
            if not self.alpha_vantage_key:
                logger.warning("Alpha Vantage API key required for analyst data")
                return None
            
            self._enforce_rate_limit('alpha_vantage')
            
            # Unfortunately, free APIs don't typically provide analyst data
            # In production, you would use:
            # - Thomson Reuters Eikon
            # - Bloomberg Terminal API
            # - FactSet
            # - Refinitiv
            # - Financial Modeling Prep (paid tier)
            
            # For now, return empty structure indicating data unavailable
            logger.info(f"Analyst data not available for {symbol} with current API tier")
            return {
                'symbol': symbol.upper(),
                'analyst_count': 0,
                'consensus_rating': None,
                'target_price_avg': None,
                'target_price_high': None,
                'target_price_low': None,
                'strong_buy_count': 0,
                'buy_count': 0,
                'hold_count': 0,
                'sell_count': 0,
                'data_available': False,
                'message': 'Analyst data requires premium API subscription'
            }
            
        except Exception as e:
            logger.error(f"Failed to get analyst data for {symbol}: {e}")
            return None
    
    def get_earnings_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get earnings data and surprises"""
        try:
            if not self.alpha_vantage_key:
                logger.warning("Alpha Vantage API key required for earnings data")
                return None
            
            self._enforce_rate_limit('alpha_vantage')
            
            params = {
                'function': 'EARNINGS',
                'symbol': symbol.upper(),
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(self.alpha_vantage_base, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Error Message' in data:
                    logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                    return None
                
                if 'Note' in data:
                    logger.warning(f"Alpha Vantage rate limit hit for {symbol}")
                    return None
                
                earnings_data = {
                    'symbol': symbol.upper(),
                    'quarterly_earnings': data.get('quarterlyEarnings', []),
                    'annual_earnings': data.get('annualEarnings', []),
                    'data_available': True
                }
                
                self._update_api_usage('alpha_vantage')
                return earnings_data
            else:
                logger.error(f"Alpha Vantage request failed for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get earnings data for {symbol}: {e}")
            return None
    
    def get_institutional_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get institutional ownership data"""
        try:
            # Institutional data typically requires premium APIs
            # Free APIs don't usually provide 13F filing data
            
            logger.info(f"Institutional data not available for {symbol} with current API tier")
            return {
                'symbol': symbol.upper(),
                'institutional_ownership': None,
                'top_holders': [],
                'recent_changes': [],
                'data_available': False,
                'message': 'Institutional data requires premium API subscription (13F filings)'
            }
            
        except Exception as e:
            logger.error(f"Failed to get institutional data for {symbol}: {e}")
            return None
    
    def _get_alpha_vantage_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data from Alpha Vantage"""
        try:
            if not self.alpha_vantage_key:
                return None
            
            self._enforce_rate_limit('alpha_vantage')
            
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol.upper(),
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(self.alpha_vantage_base, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Error Message' in data or 'Note' in data:
                    return None
                
                # Extract relevant financial metrics
                fundamentals = {
                    'pe_ratio': self._safe_float(data.get('PERatio')),
                    'peg_ratio': self._safe_float(data.get('PEGRatio')),
                    'price_to_book': self._safe_float(data.get('PriceToBookRatio')),
                    'price_to_sales': self._safe_float(data.get('PriceToSalesRatioTTM')),
                    'ev_revenue': self._safe_float(data.get('EVToRevenue')),
                    'ev_ebitda': self._safe_float(data.get('EVToEBITDA')),
                    'profit_margin': self._safe_float(data.get('ProfitMargin')),
                    'operating_margin': self._safe_float(data.get('OperatingMarginTTM')),
                    'return_on_assets': self._safe_float(data.get('ReturnOnAssetsTTM')),
                    'return_on_equity': self._safe_float(data.get('ReturnOnEquityTTM')),
                    'revenue_ttm': self._safe_float(data.get('RevenueTTM')),
                    'gross_profit_ttm': self._safe_float(data.get('GrossProfitTTM')),
                    'diluted_eps_ttm': self._safe_float(data.get('DilutedEPSTTM')),
                    'quarterly_earnings_growth': self._safe_float(data.get('QuarterlyEarningsGrowthYOY')),
                    'quarterly_revenue_growth': self._safe_float(data.get('QuarterlyRevenueGrowthYOY')),
                    'analyst_target_price': self._safe_float(data.get('AnalystTargetPrice')),
                    'trailing_pe': self._safe_float(data.get('TrailingPE')),
                    'forward_pe': self._safe_float(data.get('ForwardPE')),
                    'dividend_yield': self._safe_float(data.get('DividendYield')),
                    'beta': self._safe_float(data.get('Beta'))
                }
                
                self._update_api_usage('alpha_vantage')
                return fundamentals
            
            return None
            
        except Exception as e:
            logger.error(f"Alpha Vantage fundamentals failed for {symbol}: {e}")
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        try:
            if value is None or value == 'None' or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _enforce_rate_limit(self, api_name: str):
        """Enforce rate limiting for API calls"""
        current_time = time.time()
        
        if api_name not in self.last_request_times:
            self.last_request_times[api_name] = []
        
        # Clean old timestamps based on API type
        if api_name == 'polygon':
            cutoff_time = current_time - 1  # 1 second window for Polygon (100 req/sec)
        else:
            cutoff_time = current_time - 60  # 1 minute window for others
            
        self.last_request_times[api_name] = [
            t for t in self.last_request_times[api_name] if t > cutoff_time
        ]
        
        # Check if we're within rate limits
        rate_limit = self.rate_limits.get(api_name, 5)
        if len(self.last_request_times[api_name]) >= rate_limit:
            if api_name == 'polygon':
                sleep_time = 1.01 - (current_time - self.last_request_times[api_name][0])  # 1 second window
            else:
                sleep_time = 61 - (current_time - self.last_request_times[api_name][0])  # 1 minute window
                
            if sleep_time > 0:
                logger.info(f"Rate limiting {api_name} API - sleeping {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        # Record this request
        self.last_request_times[api_name].append(current_time)
    
    def _update_api_usage(self, api_name: str):
        """Update API usage tracking"""
        today = datetime.now().date()
        
        # Reset counters if it's a new day
        if self.api_usage['last_reset'] != today:
            self.api_usage = {
                'polygon_calls_today': 0,
                'news_calls_today': 0,
                'alpha_vantage_calls_today': 0,
                'last_reset': today
            }
        
        # Increment counter
        counter_key = f"{api_name}_calls_today"
        if counter_key in self.api_usage:
            self.api_usage[counter_key] += 1
    
    def get_api_usage(self) -> Dict[str, int]:
        """Get current API usage statistics"""
        return self.api_usage.copy()
    
    def test_api_connections(self) -> Dict[str, bool]:
        """Test all API connections"""
        results = {}
        
        # Test Polygon
        try:
            if self.polygon_client:
                # Test with a simple request
                test_data = self.polygon_client.get_ticker_details("AAPL")
                results['polygon'] = test_data is not None
            else:
                results['polygon'] = False
        except Exception as e:
            logger.error(f"Polygon API test failed: {e}")
            results['polygon'] = False
        
        # Test Alpha Vantage
        try:
            if self.alpha_vantage_key:
                params = {
                    'function': 'OVERVIEW',
                    'symbol': 'AAPL',
                    'apikey': self.alpha_vantage_key
                }
                response = requests.get(self.alpha_vantage_base, params=params, timeout=10)
                results['alpha_vantage'] = response.status_code == 200
            else:
                results['alpha_vantage'] = False
        except Exception as e:
            logger.error(f"Alpha Vantage API test failed: {e}")
            results['alpha_vantage'] = False
        
        # Test News API
        try:
            if self.news_api_key:
                params = {
                    'q': 'test',
                    'apiKey': self.news_api_key,
                    'pageSize': 1
                }
                response = requests.get(f"{self.news_api_base}/everything", params=params, timeout=10)
                results['news_api'] = response.status_code == 200
            else:
                results['news_api'] = False
        except Exception as e:
            logger.error(f"News API test failed: {e}")
            results['news_api'] = False
        
        return results
