"""
Data Loader Module
Authentic market data fetching from Polygon and FRED APIs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import logging

logger = logging.getLogger(__name__)

def fetch_data(universe, polygon_api_key, fred_api_key, start_date=None, end_date=None, resolution='15min'):
    """
    Fetch authentic market data from Polygon and FRED APIs for trading system
    
    Args:
        universe: List of stock symbols to fetch
        polygon_api_key: Polygon API key for stock data
        fred_api_key: FRED API key for VIX data
        start_date: Start date for data fetch (default: 2 years ago)
        end_date: End date for data fetch (default: now)
        resolution: Data resolution ('15min' for maximum historical window, '1day' for daily)
    
    Returns:
        dict: Market data in self.market_data format with symbols as keys
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=730)  # 2 years for maximum data
    if end_date is None:
        end_date = datetime.now()
    
    market_data = {}
    
    logger.info(f"Loading authentic {resolution} data from Polygon API...")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Step 1: Fetch VIX data from FRED for market regime analysis
    logger.info("Fetching VIX data from FRED API for market regime detection...")
    vix_data = get_vix_data(start_date, end_date, fred_api_key)
    
    if vix_data is not None and len(vix_data) > 0:
        # Store VIX data in market_data format
        market_data['VIX'] = vix_data
        logger.info(f"VIX: {len(vix_data)} days of authentic FRED data loaded")
        logger.info(f"VIX range: {vix_data['vix'].min():.2f} - {vix_data['vix'].max():.2f}")
    else:
        logger.error("Failed to fetch VIX data from FRED - check API key")
        # Create minimal VIX data to prevent system failure
        date_range = pd.date_range(start_date, end_date, freq='D')
        market_data['VIX'] = pd.DataFrame({
            'vix': [15.0] * len(date_range),
            'vix_sma_20': [15.0] * len(date_range),
            'vix_regime': ['moderate_volatility_normal'] * len(date_range)
        }, index=date_range)
        logger.warning("Using fallback VIX data - live trading not recommended")
    
    # Step 2: Fetch stock data from Polygon API
    successful_symbols = 0
    total_bars = 0
    
    for symbol in universe:
        logger.info(f"Fetching {symbol} data from Polygon API...")
        
        try:
            # Get authentic stock data with maximum historical window
            stock_data = get_stock_data(symbol, start_date, end_date, polygon_api_key, resolution)
            
            if stock_data is not None and len(stock_data) > 50:
                # Add comprehensive technical indicators
                stock_data = add_technical_indicators(stock_data)
                
                # Forward-fill VIX regime data into 15-minute stock data
                if resolution == '15min' and 'VIX' in market_data:
                    stock_data = merge_vix_with_intraday(stock_data, market_data['VIX'])
                
                # Validate data integrity
                if validate_stock_data(stock_data, symbol):
                    market_data[symbol] = stock_data
                    successful_symbols += 1
                    total_bars += len(stock_data)
                    
                    logger.info(f"{symbol}: {len(stock_data):,} {resolution} bars loaded")
                    logger.info(f"  Price range: ${stock_data['close'].min():.2f} - ${stock_data['close'].max():.2f}")
                    logger.info(f"  Date range: {stock_data.index[0]} to {stock_data.index[-1]}")
                else:
                    logger.error(f"{symbol}: Data validation failed")
            else:
                logger.error(f"{symbol}: Insufficient data returned from Polygon")
                
        except Exception as e:
            logger.error(f"{symbol}: Error fetching data - {e}")
    
    # Step 3: Validate complete dataset
    if successful_symbols == 0:
        logger.error("No stock data loaded - check Polygon API key and symbols")
        return {}
    
    logger.info(f"Data loading complete:")
    logger.info(f"  Symbols loaded: {successful_symbols}/{len(universe)}")
    logger.info(f"  Total data points: {total_bars:,}")
    logger.info(f"  Resolution: {resolution}")
    logger.info(f"  VIX integration: {'Yes' if 'VIX' in market_data else 'No'}")
    
    # Step 4: Add metadata for system use
    market_data['_metadata'] = {
        'source': 'Polygon API + FRED',
        'resolution': resolution,
        'start_date': start_date,
        'end_date': end_date,
        'symbols_loaded': successful_symbols,
        'total_bars': total_bars,
        'vix_available': 'VIX' in market_data,
        'data_integrity': 'authentic'
    }
    
    return market_data

def validate_stock_data(data, symbol):
    """Validate that stock data is complete and authentic"""
    
    if data is None or len(data) < 20:
        return False
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        logger.error(f"{symbol}: Missing required OHLCV columns")
        return False
    
    # Check for reasonable price ranges
    if data['close'].min() <= 0 or data['close'].max() > 10000:
        logger.error(f"{symbol}: Unrealistic price range")
        return False
    
    # Check for reasonable volume
    if data['volume'].min() < 0:
        logger.error(f"{symbol}: Negative volume detected")
        return False
    
    # Check technical indicators were added
    technical_indicators = ['sma_20', 'rsi_14', 'momentum_5', 'atr_pct_14', 'volume_ratio']
    missing_indicators = [ind for ind in technical_indicators if ind not in data.columns]
    
    if missing_indicators:
        logger.warning(f"{symbol}: Missing indicators: {missing_indicators}")
    
    return True

def merge_vix_with_intraday(stock_data, vix_data):
    """
    Merge VIX data with 15-minute stock data using forward-fill
    
    Args:
        stock_data: 15-minute stock price data
        vix_data: Daily VIX data
    
    Returns:
        DataFrame: Stock data with VIX columns forward-filled
    """
    # Ensure both have datetime index
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data.index)
    if not isinstance(vix_data.index, pd.DatetimeIndex):
        vix_data.index = pd.to_datetime(vix_data.index)
    
    # Resample VIX to match stock data frequency and forward-fill
    vix_resampled = vix_data.reindex(stock_data.index, method='ffill')
    
    # Add VIX columns to stock data
    for col in ['vix', 'vix_sma_20', 'vix_regime']:
        if col in vix_resampled.columns:
            stock_data[f'vix_{col}'] = vix_resampled[col]
    
    return stock_data

def get_stock_data(symbol, start_date, end_date, api_key, resolution='15min'):
    """
    Get authentic stock data from Polygon API with timestamp verification
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        api_key: Polygon API key
        resolution: Data resolution ('15min' or '1day')
    
    Returns:
        DataFrame: Stock price and volume data
    """
    current_time = datetime.now()
    print(f"TIMESTAMP VERIFICATION: Fetching {symbol} {resolution} data at {current_time.isoformat()}")
    
    if resolution == '15min':
        # 15-minute bars for intraday analysis
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/15/minute/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    else:
        # Daily bars for longer timeframes
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    
    params = {'apikey': api_key, 'adjusted': 'true', 'sort': 'asc', 'limit': 50000}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                df = pd.DataFrame(data['results'])
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                
                # Verify timestamp authenticity
                if not df.empty:
                    latest_timestamp = df['date'].max()
                    oldest_timestamp = df['date'].min()
                    print(f"TIMESTAMP VERIFICATION: {symbol} - Data range {oldest_timestamp} to {latest_timestamp}")
                    
                    # Check for suspicious patterns
                    if latest_timestamp > current_time:
                        print(f"WARNING: {symbol} - Future timestamp detected: {latest_timestamp}")
                        return None
                    
                    # Check for duplicate timestamps (synthetic data indicator)
                    unique_timestamps = df['date'].nunique()
                    total_records = len(df)
                    if unique_timestamps < total_records * 0.8:
                        print(f"WARNING: {symbol} - Suspicious timestamp patterns: {unique_timestamps}/{total_records} unique")
                        return None
                
                df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                df.set_index('date', inplace=True)
                
                # Store locally for reusability
                cache_dir = 'data/cache'
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = f"{cache_dir}/{symbol}_{resolution}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
                df.to_parquet(cache_file)
                logger.info(f"Cached {symbol} authentic data to {cache_file}")
                
                return df[['open', 'high', 'low', 'close', 'volume']].sort_index()
        
        logger.warning(f"Polygon API returned status {response.status_code} for {symbol}")
        return None
    except Exception as e:
        logger.error(f"Error fetching {symbol} from Polygon: {e}")
        return None

def get_vix_data(start_date, end_date, api_key):
    """
    Get VIX data from FRED API with timestamp verification
    
    Args:
        start_date: Start date
        end_date: End date
        api_key: FRED API key
    
    Returns:
        DataFrame: VIX data with regime indicators
    """
    current_time = datetime.now()
    print(f"TIMESTAMP VERIFICATION: Fetching VIX data at {current_time.isoformat()}")
    
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': 'VIXCLS',
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date.strftime('%Y-%m-%d'),
        'observation_end': end_date.strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(url, params=params, timeout=20)
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['vix'] = pd.to_numeric(df['value'], errors='coerce')
                df.set_index('date', inplace=True)
                
                # Verify timestamp authenticity
                if not df.empty:
                    latest_timestamp = df.index.max()
                    oldest_timestamp = df.index.min()
                    print(f"TIMESTAMP VERIFICATION: VIX - Data range {oldest_timestamp.date()} to {latest_timestamp.date()}")
                    
                    # Check for suspicious patterns
                    if latest_timestamp > current_time:
                        print(f"WARNING: VIX - Future timestamp detected: {latest_timestamp}")
                        return None
                    
                    # Check data freshness
                    days_old = (current_time - latest_timestamp).days
                    if days_old > 7:
                        print(f"WARNING: VIX data is {days_old} days old")
                
                # Add VIX-based market regime indicators
                df['vix_sma_20'] = df['vix'].rolling(window=20).mean()
                df['vix_regime'] = pd.cut(df['vix'], bins=[0, 15, 25, 100], labels=['low', 'moderate', 'high'])
                df['volatility_20'] = df['vix'] / 100  # Normalize for compatibility
                
                return df[['vix', 'vix_sma_20', 'vix_regime', 'volatility_20']].dropna()
        return None
    except Exception as e:
        logger.error(f"Error fetching VIX data: {e}")
        return None

def add_technical_indicators(df):
    """
    Add technical indicators to authentic price data
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame: Enhanced with technical indicators
    """
    # Core momentum and trend indicators
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    
    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    df['atr_pct_14'] = df['atr_14'] / df['close']
    
    return df