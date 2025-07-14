#!/usr/bin/env python3
"""
Enhanced Mean Reversion Model
Uses real historical data to identify mean reversion opportunities
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MeanReversionEnhanced:
    """
    Enhanced mean reversion strategy using real market data
    Identifies oversold/overbought conditions for reversal trades
    """
    
    lookback_period: int = 20  # Days for calculating statistics
    entry_threshold: float = -2.0  # Z-score for entry (negative = oversold)
    exit_threshold: float = 0.0  # Z-score for exit
    use_bollinger: bool = True  # Use Bollinger Bands
    volume_filter: bool = True  # Require volume spike on reversals
    rsi_confirmation: bool = True  # Use RSI for confirmation
    
    def __post_init__(self):
        """Initialize internal state"""
        self.is_fitted = False
        self.price_mean = None
        self.price_std = None
        self.volume_threshold = None
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
    def fit(self, data: pd.DataFrame) -> 'MeanReversionEnhanced':
        """
        Fit the model using historical data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Self for chaining
        """
        if len(data) < self.lookback_period * 3:
            raise ValueError(f"Insufficient data for fitting. Need at least {self.lookback_period * 3} days")
        
        # Calculate price statistics for normalization
        self.price_mean = data['close'].rolling(self.lookback_period).mean().mean()
        self.price_std = data['close'].rolling(self.lookback_period).std().mean()
        
        # Calculate volume threshold for spike detection
        if self.volume_filter and 'volume' in data.columns:
            volume_ma = data['volume'].rolling(self.lookback_period).mean()
            volume_spikes = data['volume'] / volume_ma
            self.volume_threshold = volume_spikes.quantile(0.8)  # Top 20% volume days
        
        self.is_fitted = True
        logger.info(f"Mean reversion model fitted on {len(data)} days of data")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on mean reversion
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of signals: 1 (buy), 0 (hold), -1 (sell)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Initialize signals
        signals = pd.Series(0, index=data.index, dtype=int)
        
        # Calculate z-scores
        z_scores = self._calculate_z_scores(data)
        
        # Basic mean reversion signals
        # Buy when oversold (z < entry_threshold)
        signals[z_scores < self.entry_threshold] = 1
        # Sell when overbought (z > -self.entry_threshold)
        signals[z_scores > -self.entry_threshold] = -1
        
        # Apply Bollinger Bands filter
        if self.use_bollinger:
            signals = self._apply_bollinger_filter(data, signals)
        
        # Apply volume filter
        if self.volume_filter and 'volume' in data.columns:
            signals = self._apply_volume_filter(data, signals)
        
        # Apply RSI confirmation
        if self.rsi_confirmation:
            signals = self._apply_rsi_confirmation(data, signals)
        
        # Apply exit rules
        signals = self._apply_exit_rules(data, signals, z_scores)
        
        return signals
    
    def _calculate_z_scores(self, data: pd.DataFrame) -> pd.Series:
        """Calculate z-scores for mean reversion"""
        ma = data['close'].rolling(self.lookback_period).mean()
        std = data['close'].rolling(self.lookback_period).std()
        
        # Avoid division by zero
        std = std.replace(0, 1)
        
        z_scores = (data['close'] - ma) / std
        return z_scores
    
    def _apply_bollinger_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Filter signals using Bollinger Bands"""
        ma = data['close'].rolling(self.lookback_period).mean()
        std = data['close'].rolling(self.lookback_period).std()
        
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std
        
        # Only buy near lower band
        buy_mask = data['close'] <= lower_band * 1.02  # Within 2% of lower band
        signals[(signals == 1) & ~buy_mask] = 0
        
        # Only sell near upper band
        sell_mask = data['close'] >= upper_band * 0.98  # Within 2% of upper band
        signals[(signals == -1) & ~sell_mask] = 0
        
        return signals
    
    def _apply_volume_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Filter signals based on volume"""
        volume_ma = data['volume'].rolling(self.lookback_period).mean()
        volume_ratio = data['volume'] / volume_ma
        
        # Require volume spike for reversal signals
        volume_mask = volume_ratio > self.volume_threshold
        
        # Keep signals only with volume confirmation
        signals = signals * volume_mask.astype(int)
        
        return signals
    
    def _apply_rsi_confirmation(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Confirm signals with RSI"""
        # Calculate RSI
        rsi = self._calculate_rsi(data['close'])
        
        # Buy signals need oversold RSI
        buy_mask = rsi < self.rsi_oversold
        signals[(signals == 1) & ~buy_mask] = 0
        
        # Sell signals need overbought RSI
        sell_mask = rsi > self.rsi_overbought
        signals[(signals == -1) & ~sell_mask] = 0
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _apply_exit_rules(self, data: pd.DataFrame, signals: pd.Series, 
                         z_scores: pd.Series) -> pd.Series:
        """Apply exit rules for mean reversion"""
        # Create a copy to track positions
        positions = signals.copy()
        in_position = 0
        
        for i in range(1, len(positions)):
            # Carry forward position if no signal
            if positions.iloc[i] == 0 and in_position != 0:
                # Check exit conditions
                if abs(z_scores.iloc[i]) < abs(self.exit_threshold):
                    # Price returned to mean - exit
                    positions.iloc[i] = 0
                    in_position = 0
                else:
                    # Stay in position
                    positions.iloc[i] = in_position
            elif positions.iloc[i] != 0:
                # New position
                in_position = positions.iloc[i]
        
        return positions
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'lookback_period': self.lookback_period,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'use_bollinger': self.use_bollinger,
            'volume_filter': self.volume_filter,
            'rsi_confirmation': self.rsi_confirmation,
            'is_fitted': self.is_fitted
        }
    
    def calculate_reversion_probability(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate probability of mean reversion
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of reversion probabilities (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating probabilities")
        
        z_scores = self._calculate_z_scores(data)
        
        # Convert z-scores to probabilities using sigmoid
        # Higher absolute z-score = higher reversion probability
        probabilities = 1 / (1 + np.exp(-abs(z_scores) + 2))
        
        # Adjust for volume if available
        if self.volume_filter and 'volume' in data.columns:
            volume_ma = data['volume'].rolling(self.lookback_period).mean()
            volume_ratio = data['volume'] / volume_ma
            volume_factor = (volume_ratio - 1).clip(0, 1)
            
            probabilities = 0.7 * probabilities + 0.3 * volume_factor
        
        return probabilities
    
    def get_entry_points(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed entry point analysis
        
        Returns DataFrame with:
        - Entry signals
        - Z-scores
        - RSI values
        - Volume ratios
        - Reversion probability
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analysis")
        
        signals = self.predict(data)
        z_scores = self._calculate_z_scores(data)
        rsi = self._calculate_rsi(data['close'])
        probabilities = self.calculate_reversion_probability(data)
        
        entry_analysis = pd.DataFrame({
            'signal': signals,
            'z_score': z_scores,
            'rsi': rsi,
            'reversion_probability': probabilities,
            'close': data['close']
        }, index=data.index)
        
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(self.lookback_period).mean()
            entry_analysis['volume_ratio'] = data['volume'] / volume_ma
        
        # Filter to only entry points
        entry_points = entry_analysis[entry_analysis['signal'] != 0].copy()
        
        return entry_points