#!/usr/bin/env python3
"""
Enhanced Momentum Breakout Model
Uses real historical data and proper feature engineering
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MomentumBreakoutEnhanced:
    """
    Enhanced momentum breakout strategy using real market data
    Identifies breakouts based on price momentum, volume, and volatility
    """
    
    lookback_period: int = 20  # Days to calculate momentum
    breakout_threshold: float = 2.0  # Standard deviations for breakout
    volume_confirmation: bool = True  # Require volume confirmation
    volatility_filter: bool = True  # Filter based on volatility regime
    
    def __post_init__(self):
        """Initialize internal state"""
        self.is_fitted = False
        self.momentum_mean = None
        self.momentum_std = None
        self.volume_ratio_threshold = None
        self.volatility_regime = None
        
    def fit(self, data: pd.DataFrame) -> 'MomentumBreakoutEnhanced':
        """
        Fit the model using historical data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Self for chaining
        """
        if len(data) < self.lookback_period * 2:
            raise ValueError(f"Insufficient data for fitting. Need at least {self.lookback_period * 2} days")
        
        # Calculate momentum metrics from training data
        returns = data['close'].pct_change()
        momentum = returns.rolling(self.lookback_period).sum()
        
        # Store statistics for normalization
        self.momentum_mean = momentum.mean()
        self.momentum_std = momentum.std()
        
        # Calculate volume patterns
        if self.volume_confirmation and 'volume' in data.columns:
            volume_ma = data['volume'].rolling(self.lookback_period).mean()
            volume_ratio = data['volume'] / volume_ma
            self.volume_ratio_threshold = volume_ratio.quantile(0.75)  # Top 25% volume
        
        # Analyze volatility regimes
        if self.volatility_filter:
            volatility = returns.rolling(self.lookback_period).std()
            self.volatility_regime = {
                'low': volatility.quantile(0.33),
                'high': volatility.quantile(0.67)
            }
        
        self.is_fitted = True
        logger.info(f"Model fitted on {len(data)} days of data")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on momentum breakouts
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of signals: 1 (buy), 0 (hold), -1 (sell)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Initialize signals
        signals = pd.Series(0, index=data.index, dtype=int)
        
        # Calculate momentum
        returns = data['close'].pct_change()
        momentum = returns.rolling(self.lookback_period).sum()
        
        # Standardize momentum
        momentum_z = (momentum - self.momentum_mean) / self.momentum_std
        
        # Basic momentum signals
        signals[momentum_z > self.breakout_threshold] = 1  # Strong upward momentum
        signals[momentum_z < -self.breakout_threshold] = -1  # Strong downward momentum
        
        # Apply volume confirmation filter
        if self.volume_confirmation and 'volume' in data.columns:
            volume_ma = data['volume'].rolling(self.lookback_period).mean()
            volume_ratio = data['volume'] / volume_ma
            
            # Only confirm signals with high volume
            volume_mask = volume_ratio > self.volume_ratio_threshold
            signals = signals * volume_mask.astype(int)
        
        # Apply volatility filter
        if self.volatility_filter:
            current_volatility = returns.rolling(self.lookback_period).std()
            
            # Different thresholds for different volatility regimes
            for i in range(len(data)):
                if pd.notna(current_volatility.iloc[i]):
                    if current_volatility.iloc[i] < self.volatility_regime['low']:
                        # Low volatility: require stronger breakout
                        if abs(momentum_z.iloc[i]) < self.breakout_threshold * 1.5:
                            signals.iloc[i] = 0
                    elif current_volatility.iloc[i] > self.volatility_regime['high']:
                        # High volatility: be more conservative
                        if abs(momentum_z.iloc[i]) < self.breakout_threshold * 0.8:
                            signals.iloc[i] = 0
        
        # Additional filters
        signals = self._apply_market_structure_filters(data, signals)
        
        return signals
    
    def _apply_market_structure_filters(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Apply additional market structure filters"""
        
        # Don't trade in choppy markets (high/low ratio close to 1)
        if 'high' in data.columns and 'low' in data.columns:
            daily_range = (data['high'] - data['low']) / data['close']
            choppy_market = daily_range.rolling(5).mean() < 0.005  # Less than 0.5% daily range
            signals[choppy_market] = 0
        
        # Confirm with price above/below moving average
        ma_50 = data['close'].rolling(50).mean()
        if len(data) >= 50:
            # Long signals only above MA50
            signals[(signals == 1) & (data['close'] < ma_50)] = 0
            # Short signals only below MA50
            signals[(signals == -1) & (data['close'] > ma_50)] = 0
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'lookback_period': self.lookback_period,
            'breakout_threshold': self.breakout_threshold,
            'volume_confirmation': self.volume_confirmation,
            'volatility_filter': self.volatility_filter,
            'is_fitted': self.is_fitted,
            'momentum_mean': float(self.momentum_mean) if self.momentum_mean is not None else None,
            'momentum_std': float(self.momentum_std) if self.momentum_std is not None else None
        }
    
    def calculate_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength (0-1) for position sizing
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of signal strengths
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating signal strength")
        
        # Calculate momentum z-score
        returns = data['close'].pct_change()
        momentum = returns.rolling(self.lookback_period).sum()
        momentum_z = abs((momentum - self.momentum_mean) / self.momentum_std)
        
        # Convert to 0-1 scale
        # Threshold = 0.5, 2*threshold = 1.0
        signal_strength = (momentum_z - self.breakout_threshold) / self.breakout_threshold
        signal_strength = signal_strength.clip(0, 1)
        
        # Adjust for volume if available
        if self.volume_confirmation and 'volume' in data.columns:
            volume_ma = data['volume'].rolling(self.lookback_period).mean()
            volume_ratio = data['volume'] / volume_ma
            volume_strength = (volume_ratio - 1).clip(0, 2) / 2  # 0-1 scale
            
            # Combine momentum and volume strength
            signal_strength = 0.7 * signal_strength + 0.3 * volume_strength
        
        return signal_strength
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for interpretability"""
        importance = {
            'momentum': 0.5,
            'volume': 0.3 if self.volume_confirmation else 0.0,
            'volatility_regime': 0.1 if self.volatility_filter else 0.0,
            'market_structure': 0.1
        }
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance