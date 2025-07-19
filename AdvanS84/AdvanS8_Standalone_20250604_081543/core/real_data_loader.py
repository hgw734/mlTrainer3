"""
Real Data Loader Module
Pulls authentic data from Polygon + FRED APIs for ML model training
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def fetch_data(universe, polygon_api_key, fred_api_key, start_date=None, end_date=None, resolution='15min'):
    """
    Fetch real Polygon + FRED data for ML model training
    
    Args:
        universe: List of stock symbols
        polygon_api_key: Polygon API key
        fred_api_key: FRED API key
        start_date: Start date (default: 2 years ago)
        end_date: End date (default: now)
        resolution: '15min' for maximum historical window or '1day'
    
    Returns:
        dict: self.market_data format with VIX merged into each stock DataFrame
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=730)  # 2 years for maximum data
    if end_date is None:
        end_date = datetime.now()
    
    market_data = {}
    
    logger.info(f"Loading real {resolution} data from Polygon API...")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Step 1: Call FRED's VIX series (VIXCLS) as specified
    logger.info("Calling FRED API for VIX data (VIXCLS series)...")
    vix_data = get_vix_from_fred(start_date, end_date, fred_api_key)
    
    if vix_data is None:
        logger.error("Failed to fetch VIX data from FRED - check API key")
        return {}
    
    market_data['VIX'] = vix_data
    logger.info(f"VIX: {len(vix_data)} days loaded from FRED")
    
    # Step 2: Call Polygon's /v2/aggs/ticker/{symbol}/range/15/minute/... for each stock
    successful_symbols = 0
    
    for symbol in universe:
        logger.info(f"Calling Polygon API for {symbol}...")
        
        stock_data = get_stock_from_polygon(symbol, start_date, end_date, polygon_api_key, resolution)
        
        if stock_data is not None:
            # Add technical columns as specified
            stock_data = add_technical_columns(stock_data)
            
            # Merge VIX into stock DataFrame using pd.merge_asof() as specified
            stock_data = merge_vix_with_stock(stock_data, vix_data)
            
            market_data[symbol] = stock_data
            successful_symbols += 1
            
            logger.info(f"{symbol}: {len(stock_data)} bars with VIX merged")
        else:
            logger.error(f"{symbol}: Failed to fetch from Polygon")
    
    if successful_symbols == 0:
        logger.error("No stock data loaded - check Polygon API key")
        return {}
    
    logger.info(f"Real data loading complete: {successful_symbols}/{len(universe)} symbols")
    return market_data

def get_stock_from_polygon(symbol, start_date, end_date, api_key, resolution):
    """
    Call Polygon's /v2/aggs/ticker/{symbol}/range/15/minute/... endpoint
    """
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check cache first for fast reload
    cache_file = f"{cache_dir}/{symbol}_{resolution}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
    if os.path.exists(cache_file):
        logger.debug(f"Loading cached {symbol} data")
        return pd.read_parquet(cache_file)
    
    # Use exact Polygon API endpoint as specified
    if resolution == '15min':
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/15/minute/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    else:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    
    params = {
        'apikey': api_key,
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'results' in data and data['results']:
                results = data['results']
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                
                # Convert timestamps (Polygon returns milliseconds)
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
                df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
                
                # Rename to standard OHLCV format
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high', 
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                })
                
                df = df.set_index('timestamp')
                df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
                
                # Clean data for ML training
                df = df.dropna()
                df = df[df['volume'] > 0]
                df = df[df['close'] > 0]
                
                # Save to disk (.parquet) for fast reload as specified
                df.to_parquet(cache_file)
                
                return df
            else:
                logger.error(f"{symbol}: No results from Polygon")
                return None
                
        elif response.status_code == 429:
            logger.error(f"{symbol}: Rate limited by Polygon")
            return None
        else:
            logger.error(f"{symbol}: Polygon API error {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"{symbol}: Error calling Polygon - {e}")
        return None

def get_vix_from_fred(start_date, end_date, api_key):
    """
    Call FRED's VIX series (VIXCLS) as specified
    """
    cache_file = f"data/cache/VIX_FRED_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
    
    if os.path.exists(cache_file):
        logger.debug("Loading cached VIX data")
        return pd.read_parquet(cache_file)
    
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': 'VIXCLS',
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date.strftime('%Y-%m-%d'),
        'observation_end': end_date.strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'observations' in data:
                observations = data['observations']
                
                # Convert to DataFrame
                df = pd.DataFrame(observations)
                df['date'] = pd.to_datetime(df['date'])
                df['vix'] = pd.to_numeric(df['value'], errors='coerce')
                
                df = df.set_index('date')
                df = df[['vix']].dropna()
                
                # Add VIX technical indicators
                df['vix_sma_20'] = df['vix'].rolling(20).mean()
                df['vix_volatility'] = df['vix'].rolling(20).std()
                
                # VIX regime classification
                df['vix_regime'] = 'moderate_volatility_normal'
                df.loc[df['vix'] < 12, 'vix_regime'] = 'low_volatility_normal'
                df.loc[df['vix'] > 20, 'vix_regime'] = 'high_volatility_stress'
                df.loc[df['vix'] > 30, 'vix_regime'] = 'crisis_panic'
                
                # Save to disk
                os.makedirs('data/cache', exist_ok=True)
                df.to_parquet(cache_file)
                
                return df
            else:
                logger.error("No VIX observations from FRED")
                return None
                
        else:
            logger.error(f"FRED API error {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error calling FRED API - {e}")
        return None

def merge_vix_with_stock(stock_data, vix_data):
    """
    Merge VIX into each stock DataFrame using pd.merge_asof() as specified
    """
    try:
        # Prepare stock data for merge
        stock_reset = stock_data.reset_index()
        stock_reset = stock_reset.rename(columns={'index': 'timestamp'})
        
        # Prepare VIX data for merge
        vix_reset = vix_data.reset_index()
        vix_reset = vix_reset.rename(columns={'index': 'date'})
        
        # Ensure datetime columns
        stock_reset['timestamp'] = pd.to_datetime(stock_reset['timestamp'])
        vix_reset['date'] = pd.to_datetime(vix_reset['date'])
        
        # Use pd.merge_asof() for time-series merge as specified
        merged = pd.merge_asof(
            stock_reset.sort_values('timestamp'),
            vix_reset.sort_values('date'),
            left_on='timestamp',
            right_on='date',
            direction='backward'  # Use most recent VIX value
        )
        
        # Clean up merge artifacts
        merged = merged.drop(columns=['date'], errors='ignore')
        
        # Set timestamp back as index
        merged = merged.set_index('timestamp')
        
        # Ensure proper column order for self.market_data format
        stock_cols = ['open', 'high', 'low', 'close', 'volume']
        tech_cols = ['rsi_14', 'momentum_5', 'atr_pct_14', 'trend_strength', 'sma_20', 'volume_ratio']
        vix_cols = ['vix', 'vix_sma_20', 'vix_volatility', 'vix_regime']
        
        # Keep all available columns in proper order
        keep_cols = []
        for col_group in [stock_cols, tech_cols, vix_cols]:
            for col in col_group:
                if col in merged.columns:
                    keep_cols.append(col)
        
        # Add any remaining columns not in the standard lists
        remaining_cols = [col for col in merged.columns if col not in keep_cols]
        keep_cols.extend(remaining_cols)
        
        merged = merged[keep_cols]
        
        logger.info(f"VIX merge successful: {len(vix_cols)} VIX columns added")
        return merged
        
    except Exception as e:
        logger.error(f"Error merging VIX data: {e}")
        logger.info("Returning stock data without VIX merge")
        return stock_data

def add_technical_columns(df):
    """
    Add technical columns: rsi_14, momentum_5, atr_pct_14, trend_strength as specified
    """
    # RSI 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Momentum 5
    df['momentum_5'] = df['close'].pct_change(5)
    
    # ATR percentage 14
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['atr_pct_14'] = atr / df['close']
    
    # Trend strength
    df['sma_20'] = df['close'].rolling(20).mean()
    df['trend_strength'] = (df['close'] - df['sma_20']) / df['sma_20']
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    return df

# Factory function for compatibility
def create_data_loader():
    """Create data loader instance"""
    return {
        'fetch_data': fetch_data
    }