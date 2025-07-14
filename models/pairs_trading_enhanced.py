#!/usr/bin/env python3
"""
Enhanced Pairs Trading Model
Statistical arbitrage using cointegration and mean reversion
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

@dataclass
class PairsTradingEnhanced:
    """
    Pairs trading strategy using cointegration
    Identifies and trades mean-reverting spreads between related assets
    """
    
    lookback_period: int = 60  # Days for cointegration test
    entry_zscore: float = 2.0  # Z-score threshold for entry
    exit_zscore: float = 0.5  # Z-score threshold for exit
    stop_loss_zscore: float = 3.5  # Stop loss threshold
    min_half_life: int = 5  # Minimum half-life in days
    max_half_life: int = 60  # Maximum half-life in days
    hedge_ratio_method: str = 'ols'  # 'ols' or 'tls' (total least squares)
    use_dynamic_hedge: bool = True  # Update hedge ratio over time
    
    def __post_init__(self):
        """Initialize internal state"""
        self.is_fitted = False
        self.pair = None
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None
        self.half_life = None
        self.cointegration_pvalue = None
        
    def fit(self, data1: pd.DataFrame, data2: pd.DataFrame) -> 'PairsTradingEnhanced':
        """
        Fit the pairs trading model
        
        Args:
            data1: OHLCV data for first asset
            data2: OHLCV data for second asset
            
        Returns:
            Self for chaining
        """
        # Align data
        common_index = data1.index.intersection(data2.index)
        price1 = data1.loc[common_index, 'close']
        price2 = data2.loc[common_index, 'close']
        
        # Test for cointegration
        self.cointegration_pvalue = self._test_cointegration(price1, price2)
        
        if self.cointegration_pvalue > 0.05:
            logger.warning(f"No cointegration found (p-value: {self.cointegration_pvalue:.4f})")
        
        # Calculate hedge ratio
        self.hedge_ratio = self._calculate_hedge_ratio(price1, price2)
        
        # Calculate spread
        spread = self._calculate_spread(price1, price2, self.hedge_ratio)
        
        # Calculate spread statistics
        self.spread_mean = spread.mean()
        self.spread_std = spread.std()
        
        # Calculate half-life of mean reversion
        self.half_life = self._calculate_half_life(spread)
        
        self.is_fitted = True
        self.pair = ('asset1', 'asset2')  # Placeholder names
        
        logger.info(f"Pairs trading model fitted. Hedge ratio: {self.hedge_ratio:.4f}, "
                   f"Half-life: {self.half_life:.1f} days, "
                   f"Cointegration p-value: {self.cointegration_pvalue:.4f}")
        
        return self
    
    def predict(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for the pair
        
        Args:
            data1: OHLCV data for first asset
            data2: OHLCV data for second asset
            
        Returns:
            Series of signals: 1 (long spread), 0 (neutral), -1 (short spread)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Align data
        common_index = data1.index.intersection(data2.index)
        price1 = data1.loc[common_index, 'close']
        price2 = data2.loc[common_index, 'close']
        
        signals = pd.Series(0, index=common_index)
        positions = pd.Series(0, index=common_index)
        
        # Rolling window for dynamic updates
        for i in range(self.lookback_period, len(common_index)):
            current_idx = common_index[i]
            
            # Get historical window
            hist_price1 = price1.iloc[i-self.lookback_period:i]
            hist_price2 = price2.iloc[i-self.lookback_period:i]
            
            # Update hedge ratio if using dynamic hedging
            if self.use_dynamic_hedge:
                hedge_ratio = self._calculate_hedge_ratio(hist_price1, hist_price2)
            else:
                hedge_ratio = self.hedge_ratio
            
            # Calculate spread and z-score
            spread = self._calculate_spread(hist_price1, hist_price2, hedge_ratio)
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            current_spread = price1.iloc[i] - hedge_ratio * price2.iloc[i]
            zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Generate signals based on z-score
            current_position = positions.iloc[i-1] if i > 0 else 0
            
            if current_position == 0:  # No position
                if zscore > self.entry_zscore:
                    signals.iloc[i] = -1  # Short spread (long asset2, short asset1)
                    positions.iloc[i] = -1
                elif zscore < -self.entry_zscore:
                    signals.iloc[i] = 1  # Long spread (long asset1, short asset2)
                    positions.iloc[i] = 1
                else:
                    positions.iloc[i] = 0
            else:  # Have position
                # Check exit conditions
                if current_position == 1:  # Long spread
                    if zscore > -self.exit_zscore or zscore < -self.stop_loss_zscore:
                        signals.iloc[i] = -1  # Close long
                        positions.iloc[i] = 0
                    else:
                        positions.iloc[i] = 1
                else:  # Short spread
                    if zscore < self.exit_zscore or zscore > self.stop_loss_zscore:
                        signals.iloc[i] = 1  # Close short
                        positions.iloc[i] = 0
                    else:
                        positions.iloc[i] = -1
        
        return signals
    
    def get_spread_data(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get spread data for analysis
        
        Returns dictionary with spread, z-score, and bounds
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting spread data")
        
        # Align data
        common_index = data1.index.intersection(data2.index)
        price1 = data1.loc[common_index, 'close']
        price2 = data2.loc[common_index, 'close']
        
        # Calculate spread
        spread = price1 - self.hedge_ratio * price2
        
        # Calculate rolling z-score
        rolling_mean = spread.rolling(self.lookback_period).mean()
        rolling_std = spread.rolling(self.lookback_period).std()
        zscore = (spread - rolling_mean) / rolling_std
        
        return {
            'spread': spread,
            'zscore': zscore,
            'upper_band': rolling_mean + self.entry_zscore * rolling_std,
            'lower_band': rolling_mean - self.entry_zscore * rolling_std,
            'mean': rolling_mean,
            'hedge_ratio': pd.Series(self.hedge_ratio, index=common_index)
        }
    
    def _test_cointegration(self, price1: pd.Series, price2: pd.Series) -> float:
        """Test for cointegration between two price series"""
        # Engle-Granger two-step cointegration test
        _, pvalue, _ = coint(price1, price2)
        return pvalue
    
    def _calculate_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        """Calculate optimal hedge ratio"""
        if self.hedge_ratio_method == 'ols':
            # Ordinary Least Squares
            model = LinearRegression()
            model.fit(price2.values.reshape(-1, 1), price1.values)
            return model.coef_[0]
        elif self.hedge_ratio_method == 'tls':
            # Total Least Squares (orthogonal regression)
            return self._total_least_squares(price1, price2)
        else:
            raise ValueError(f"Unknown hedge ratio method: {self.hedge_ratio_method}")
    
    def _total_least_squares(self, y: pd.Series, x: pd.Series) -> float:
        """Calculate hedge ratio using total least squares"""
        # Center the data
        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        # Create data matrix
        data_matrix = np.column_stack([x_centered, y_centered])
        
        # SVD
        _, _, Vt = np.linalg.svd(data_matrix)
        
        # The hedge ratio is the ratio of the components of the first right singular vector
        return -Vt[1, 0] / Vt[1, 1]
    
    def _calculate_spread(self, price1: pd.Series, price2: pd.Series, hedge_ratio: float) -> pd.Series:
        """Calculate the spread between two assets"""
        return price1 - hedge_ratio * price2
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion using Ornstein-Uhlenbeck process"""
        # Fit AR(1) model: spread_t = a + b * spread_t-1 + error
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        # Remove NaN values
        mask = ~(spread_lag.isna() | spread_diff.isna())
        spread_lag_clean = spread_lag[mask]
        spread_diff_clean = spread_diff[mask]
        
        if len(spread_lag_clean) < 2:
            return np.nan
        
        # OLS regression
        model = sm.OLS(spread_diff_clean, sm.add_constant(spread_lag_clean))
        results = model.fit()
        
        # Half-life = -log(2) / log(1 + b)
        b = results.params[1]
        if b < 0 and b > -1:
            half_life = -np.log(2) / np.log(1 + b)
            # Bound half-life to reasonable values
            return np.clip(half_life, self.min_half_life, self.max_half_life)
        else:
            return self.max_half_life
    
    def calculate_performance_metrics(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for the pairs strategy"""
        spread_data = self.get_spread_data(data1, data2)
        zscore = spread_data['zscore'].dropna()
        
        metrics = {
            'mean_reversion_pct': (np.abs(zscore) < 1).sum() / len(zscore) * 100,
            'extreme_zscore_pct': (np.abs(zscore) > 3).sum() / len(zscore) * 100,
            'spread_volatility': spread_data['spread'].std(),
            'spread_mean': spread_data['spread'].mean(),
            'max_zscore': zscore.max(),
            'min_zscore': zscore.min(),
            'current_zscore': zscore.iloc[-1] if len(zscore) > 0 else 0,
            'hedge_ratio': self.hedge_ratio,
            'half_life_days': self.half_life,
            'cointegration_pvalue': self.cointegration_pvalue
        }
        
        return metrics
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        params = {
            'lookback_period': self.lookback_period,
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'stop_loss_zscore': self.stop_loss_zscore,
            'hedge_ratio_method': self.hedge_ratio_method,
            'use_dynamic_hedge': self.use_dynamic_hedge,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            params.update({
                'hedge_ratio': self.hedge_ratio,
                'spread_mean': self.spread_mean,
                'spread_std': self.spread_std,
                'half_life': self.half_life,
                'cointegration_pvalue': self.cointegration_pvalue
            })
        
        return params
    
    def identify_pairs(self, assets_data: Dict[str, pd.DataFrame], 
                      min_correlation: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Identify potential pairs from a universe of assets
        
        Args:
            assets_data: Dictionary of asset names to OHLCV DataFrames
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of tuples (asset1, asset2, cointegration_pvalue)
        """
        pairs = []
        asset_names = list(assets_data.keys())
        
        for i in range(len(asset_names)):
            for j in range(i + 1, len(asset_names)):
                asset1, asset2 = asset_names[i], asset_names[j]
                data1, data2 = assets_data[asset1], assets_data[asset2]
                
                # Align data
                common_index = data1.index.intersection(data2.index)
                if len(common_index) < self.lookback_period:
                    continue
                
                price1 = data1.loc[common_index, 'close']
                price2 = data2.loc[common_index, 'close']
                
                # Check correlation
                correlation = price1.corr(price2)
                if correlation < min_correlation:
                    continue
                
                # Test cointegration
                pvalue = self._test_cointegration(price1, price2)
                
                if pvalue < 0.05:  # Significant cointegration
                    pairs.append((asset1, asset2, pvalue))
        
        # Sort by p-value (lower is better)
        pairs.sort(key=lambda x: x[2])
        
        return pairs