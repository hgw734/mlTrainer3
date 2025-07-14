#!/usr/bin/env python3
"""
Data Pipeline for mlTrainer
Connects to real data sources and prepares data for ML training
NO MOCK DATA - REAL HISTORICAL DATA ONLY
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta
import logging

# Import real data connectors
from polygon_connector import PolygonConnector
from fred_connector import FREDConnector

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Real data pipeline that fetches and prepares historical market data
    Uses only approved data sources (Polygon, FRED)
    """
    
    def __init__(self):
        """Initialize with real data connectors"""
        self.polygon = PolygonConnector()
        self.fred = FREDConnector()
        logger.info("DataPipeline initialized with real data connectors")
    
    def fetch_historical_data(self, symbol: str, start_date: str = None, 
                            end_date: str = None, days: int = None) -> pd.DataFrame:
        """
        Fetch real historical data from Polygon
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            days: Alternative to date range - fetch last N days
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if days:
                # Use days parameter
                historical = self.polygon.get_historical_data(symbol, days=days)
            else:
                # Use date range
                historical = self.polygon.get_historical_data(
                    symbol, 
                    start_date=start_date, 
                    end_date=end_date
                )
            
            if historical and hasattr(historical, 'data'):
                logger.info(f"Fetched {len(historical.data)} days of data for {symbol}")
                return historical.data
            else:
                raise ValueError(f"No data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Engineer features from raw OHLCV data
        
        Features include:
        - Price returns
        - Technical indicators (moving averages, RSI, Bollinger Bands)
        - Volume indicators
        - Market microstructure features
        """
        features = []
        
        # 1. Price-based features
        features.append(df['close'].pct_change().fillna(0))  # Daily returns
        features.append((df['high'] - df['low']) / df['close'])  # Daily range
        features.append((df['close'] - df['open']) / df['open'])  # Intraday return
        
        # 2. Volume features
        volume_ma = df['volume'].rolling(20).mean()
        features.append(df['volume'] / volume_ma)  # Relative volume
        features.append(df['volume'].pct_change().fillna(0))  # Volume change
        
        # 3. Moving averages
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                ma = df['close'].rolling(period).mean()
                features.append((df['close'] - ma) / ma)  # Distance from MA
                
        # 4. Volatility features
        for period in [5, 10, 20]:
            if len(df) >= period:
                volatility = df['close'].pct_change().rolling(period).std()
                features.append(volatility)
        
        # 5. RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.fillna(50))
        
        # 6. Bollinger Bands
        for period in [20]:
            if len(df) >= period:
                ma = df['close'].rolling(period).mean()
                std = df['close'].rolling(period).std()
                upper_band = ma + 2 * std
                lower_band = ma - 2 * std
                features.append((df['close'] - upper_band) / df['close'])  # Distance from upper
                features.append((df['close'] - lower_band) / df['close'])  # Distance from lower
                features.append((upper_band - lower_band) / ma)  # Band width
        
        # 7. Market microstructure
        features.append((df['high'] - df['close']) / df['close'])  # Upper shadow
        features.append((df['close'] - df['low']) / df['close'])  # Lower shadow
        
        # Convert to numpy array
        feature_matrix = np.column_stack([f.fillna(0).values for f in features])
        
        return feature_matrix
    
    def create_training_dataset(self, symbols: List[str], lookback_days: int = 365,
                              target_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create a complete training dataset from multiple symbols
        
        Args:
            symbols: List of stock symbols
            lookback_days: Days of historical data to fetch
            target_horizon: Days ahead to predict (1 = next day)
            
        Returns:
            X: Feature matrix
            y: Target values
            metadata: DataFrame with symbol and date information
        """
        all_features = []
        all_targets = []
        all_metadata = []
        
        for symbol in symbols:
            try:
                # Fetch historical data
                df = self.fetch_historical_data(symbol, days=lookback_days)
                
                if len(df) < 50:  # Minimum data requirement
                    logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue
                
                # Prepare features
                features = self.prepare_features(df)
                
                # Create target (future returns)
                target = df['close'].pct_change(target_horizon).shift(-target_horizon).fillna(0).values
                
                # Align features and target (remove last target_horizon rows)
                features = features[:-target_horizon]
                target = target[:-target_horizon]
                
                # Create metadata
                metadata = pd.DataFrame({
                    'symbol': symbol,
                    'date': df.index[:-target_horizon]
                })
                
                all_features.append(features)
                all_targets.append(target)
                all_metadata.append(metadata)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid data collected from any symbols")
        
        # Concatenate all data
        X = np.vstack(all_features)
        y = np.hstack(all_targets)
        metadata = pd.concat(all_metadata, ignore_index=True)
        
        logger.info(f"Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
        
        return X, y, metadata
    
    def calculate_historical_volatility(self, symbol: str, window: int = 20, 
                                      lookback_days: int = 100) -> pd.Series:
        """
        Calculate actual historical volatility from real market data
        
        Args:
            symbol: Stock symbol
            window: Rolling window for volatility calculation
            lookback_days: Days of historical data to use
            
        Returns:
            Series of annualized volatility values
        """
        # Fetch real historical data
        df = self.fetch_historical_data(symbol, days=lookback_days)
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        # Calculate rolling volatility (annualized)
        volatility = returns.rolling(window).std() * np.sqrt(252)
        
        return volatility
    
    def get_market_statistics(self, symbol: str, lookback_days: int = 252) -> Dict[str, float]:
        """
        Calculate real market statistics from historical data
        
        Returns:
            Dictionary with mean return, volatility, sharpe ratio, etc.
        """
        df = self.fetch_historical_data(symbol, days=lookback_days)
        returns = df['close'].pct_change().dropna()
        
        stats = {
            'mean_return': returns.mean() * 252,  # Annualized
            'volatility': returns.std() * np.sqrt(252),  # Annualized
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'win_rate': (returns > 0).sum() / len(returns),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0
        }
        
        return stats
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def fetch_economic_indicators(self, indicators: List[str] = None) -> pd.DataFrame:
        """
        Fetch economic indicators from FRED
        
        Args:
            indicators: List of FRED series IDs (e.g., ['DGS10', 'UNRATE'])
            
        Returns:
            DataFrame with economic indicators
        """
        if indicators is None:
            # Default important indicators
            indicators = [
                'DGS10',    # 10-Year Treasury Rate
                'DGS2',     # 2-Year Treasury Rate  
                'UNRATE',   # Unemployment Rate
                'CPIAUCSL', # Consumer Price Index
                'DFF',      # Federal Funds Rate
                'DEXUSEU',  # USD/EUR Exchange Rate
            ]
        
        economic_data = {}
        
        for indicator in indicators:
            try:
                data = self.fred.get_series(indicator)
                if data is not None:
                    economic_data[indicator] = data
                    logger.info(f"Fetched {indicator} from FRED")
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {e}")
        
        if economic_data:
            return pd.DataFrame(economic_data)
        else:
            raise ValueError("No economic data could be fetched")
    
    def create_market_regime_features(self, symbol: str, lookback_days: int = 252) -> pd.DataFrame:
        """
        Create features that capture market regime (bull/bear/sideways)
        """
        df = self.fetch_historical_data(symbol, days=lookback_days)
        
        features = pd.DataFrame(index=df.index)
        
        # Trend indicators
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        features['sma_200'] = df['close'].rolling(200).mean()
        
        # Trend strength
        features['trend_20'] = (df['close'] - features['sma_20']) / features['sma_20']
        features['trend_50'] = (df['close'] - features['sma_50']) / features['sma_50']
        
        # Volatility regime
        returns = df['close'].pct_change()
        features['vol_10'] = returns.rolling(10).std() * np.sqrt(252)
        features['vol_30'] = returns.rolling(30).std() * np.sqrt(252)
        features['vol_ratio'] = features['vol_10'] / features['vol_30']
        
        # Volume regime
        features['volume_sma'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma']
        
        return features