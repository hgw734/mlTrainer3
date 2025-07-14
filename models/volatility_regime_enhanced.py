#!/usr/bin/env python3
"""
Enhanced Volatility Regime Model
Adapts trading strategy based on market volatility conditions
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class VolatilityRegimeEnhanced:
    """
    Enhanced volatility regime strategy using real market data
    Adapts strategy based on current volatility environment
    """
    
    vol_lookback: int = 20  # Days for volatility calculation
    regime_lookback: int = 60  # Days for regime identification
    low_vol_threshold: float = 0.33  # Percentile for low volatility
    high_vol_threshold: float = 0.67  # Percentile for high volatility
    adapt_strategy: bool = True  # Adapt strategy to regime
    use_garch: bool = False  # Use GARCH for volatility forecast
    
    def __post_init__(self):
        """Initialize internal state"""
        self.is_fitted = False
        self.vol_thresholds = {}
        self.regime_strategies = {
            'low': {'trend_following': 0.7, 'mean_reversion': 0.3},
            'medium': {'trend_following': 0.5, 'mean_reversion': 0.5},
            'high': {'trend_following': 0.3, 'mean_reversion': 0.7}
        }
        self.current_regime = None
        
    def fit(self, data: pd.DataFrame) -> 'VolatilityRegimeEnhanced':
        """
        Fit the model using historical data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Self for chaining
        """
        if len(data) < self.regime_lookback * 2:
            raise ValueError(f"Insufficient data for fitting. Need at least {self.regime_lookback * 2} days")
        
        # Calculate historical volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
        
        # Determine volatility thresholds
        self.vol_thresholds = {
            'low': volatility.quantile(self.low_vol_threshold),
            'high': volatility.quantile(self.high_vol_threshold)
        }
        
        # Analyze regime characteristics
        self._analyze_regime_behavior(data, volatility)
        
        self.is_fitted = True
        logger.info(f"Volatility regime model fitted on {len(data)} days of data")
        logger.info(f"Vol thresholds - Low: {self.vol_thresholds['low']:.3f}, High: {self.vol_thresholds['high']:.3f}")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on volatility regime
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of signals: 1 (buy), 0 (hold), -1 (sell)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Calculate current volatility
        returns = data['close'].pct_change()
        current_vol = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
        
        # Identify regimes
        regimes = self._identify_regimes(current_vol)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index, dtype=int)
        
        # Apply regime-specific strategies
        for regime in ['low', 'medium', 'high']:
            regime_mask = regimes == regime
            if regime_mask.any():
                regime_signals = self._apply_regime_strategy(
                    data[regime_mask], 
                    regime,
                    current_vol[regime_mask]
                )
                signals[regime_mask] = regime_signals
        
        return signals
    
    def _identify_regimes(self, volatility: pd.Series) -> pd.Series:
        """Identify volatility regimes"""
        regimes = pd.Series('medium', index=volatility.index)
        
        regimes[volatility <= self.vol_thresholds['low']] = 'low'
        regimes[volatility >= self.vol_thresholds['high']] = 'high'
        
        # Smooth regime transitions
        regimes = self._smooth_regime_transitions(regimes)
        
        return regimes
    
    def _smooth_regime_transitions(self, regimes: pd.Series, min_duration: int = 5) -> pd.Series:
        """Avoid regime whipsaws by requiring minimum duration"""
        smooth_regimes = regimes.copy()
        current_regime = regimes.iloc[0]
        regime_start = 0
        
        for i in range(1, len(regimes)):
            if regimes.iloc[i] != current_regime:
                # Check if regime lasted minimum duration
                if i - regime_start < min_duration:
                    # Revert to previous regime
                    smooth_regimes.iloc[regime_start:i] = smooth_regimes.iloc[max(0, regime_start-1)]
                else:
                    current_regime = regimes.iloc[i]
                    regime_start = i
        
        return smooth_regimes
    
    def _apply_regime_strategy(self, data: pd.DataFrame, regime: str, 
                              volatility: pd.Series) -> pd.Series:
        """Apply regime-specific trading strategy"""
        strategies = self.regime_strategies[regime]
        
        # Combine trend following and mean reversion based on regime
        trend_signals = self._trend_following_signals(data)
        reversion_signals = self._mean_reversion_signals(data)
        
        # Weight signals based on regime
        combined_signals = (
            strategies['trend_following'] * trend_signals +
            strategies['mean_reversion'] * reversion_signals
        )
        
        # Convert to discrete signals
        signals = pd.Series(0, index=data.index)
        signals[combined_signals > 0.5] = 1
        signals[combined_signals < -0.5] = -1
        
        # Apply regime-specific filters
        if regime == 'high':
            # More conservative in high volatility
            signals = self._apply_high_vol_filters(data, signals, volatility)
        elif regime == 'low':
            # More aggressive in low volatility
            signals = self._apply_low_vol_filters(data, signals)
        
        return signals
    
    def _trend_following_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trend following signals"""
        # Simple moving average crossover
        ma_short = data['close'].rolling(10).mean()
        ma_long = data['close'].rolling(30).mean()
        
        signals = pd.Series(0.0, index=data.index)
        signals[ma_short > ma_long] = 1.0
        signals[ma_short < ma_long] = -1.0
        
        return signals
    
    def _mean_reversion_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals"""
        # Z-score based
        ma = data['close'].rolling(20).mean()
        std = data['close'].rolling(20).std()
        z_score = (data['close'] - ma) / std
        
        signals = pd.Series(0.0, index=data.index)
        signals[z_score < -2] = 1.0  # Oversold
        signals[z_score > 2] = -1.0  # Overbought
        
        return signals
    
    def _apply_high_vol_filters(self, data: pd.DataFrame, signals: pd.Series, 
                               volatility: pd.Series) -> pd.Series:
        """Apply filters for high volatility regime"""
        # Reduce position size implicitly through fewer signals
        # Only take signals with strong conviction
        
        # Require larger price moves
        returns = data['close'].pct_change()
        momentum = returns.rolling(5).sum()
        
        # Filter out weak signals
        weak_buy = (signals == 1) & (momentum < 0.02)
        weak_sell = (signals == -1) & (momentum > -0.02)
        
        signals[weak_buy | weak_sell] = 0
        
        # Avoid trading on volatility spikes
        vol_spike = volatility > volatility.rolling(10).mean() * 1.5
        signals[vol_spike] = 0
        
        return signals
    
    def _apply_low_vol_filters(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Apply filters for low volatility regime"""
        # Be more aggressive with smaller moves
        
        # Enhance weak signals
        returns = data['close'].pct_change()
        
        # Look for any directional bias
        bias = returns.rolling(5).mean()
        
        # Add signals on consistent small moves
        signals[(bias > 0.001) & (signals == 0)] = 1
        signals[(bias < -0.001) & (signals == 0)] = -1
        
        return signals
    
    def _analyze_regime_behavior(self, data: pd.DataFrame, volatility: pd.Series):
        """Analyze how different strategies perform in each regime"""
        regimes = self._identify_regimes(volatility)
        
        # Calculate strategy performance in each regime
        for regime in ['low', 'medium', 'high']:
            regime_data = data[regimes == regime]
            if len(regime_data) > 50:
                # Test trend following
                trend_perf = self._test_strategy_performance(
                    regime_data, 
                    self._trend_following_signals
                )
                
                # Test mean reversion
                reversion_perf = self._test_strategy_performance(
                    regime_data, 
                    self._mean_reversion_signals
                )
                
                # Update optimal weights if adaptive
                if self.adapt_strategy:
                    total = abs(trend_perf) + abs(reversion_perf)
                    if total > 0:
                        self.regime_strategies[regime] = {
                            'trend_following': abs(trend_perf) / total,
                            'mean_reversion': abs(reversion_perf) / total
                        }
    
    def _test_strategy_performance(self, data: pd.DataFrame, 
                                  strategy_func: callable) -> float:
        """Test strategy performance on historical data"""
        signals = strategy_func(data)
        returns = data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # Return Sharpe ratio
        if strategy_returns.std() > 0:
            return (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252))
        return 0.0
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'vol_lookback': self.vol_lookback,
            'regime_lookback': self.regime_lookback,
            'low_vol_threshold': self.low_vol_threshold,
            'high_vol_threshold': self.high_vol_threshold,
            'adapt_strategy': self.adapt_strategy,
            'vol_thresholds': self.vol_thresholds,
            'regime_strategies': self.regime_strategies,
            'is_fitted': self.is_fitted
        }
    
    def get_regime_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed regime analysis
        
        Returns DataFrame with:
        - Current regime
        - Volatility levels
        - Strategy weights
        - Signal strength
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analysis")
        
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
        regimes = self._identify_regimes(volatility)
        
        analysis = pd.DataFrame({
            'regime': regimes,
            'volatility': volatility,
            'vol_percentile': volatility.rank(pct=True),
        }, index=data.index)
        
        # Add strategy weights for each regime
        for i, regime in enumerate(regimes):
            if pd.notna(regime):
                weights = self.regime_strategies.get(regime, {})
                analysis.loc[analysis.index[i], 'trend_weight'] = weights.get('trend_following', 0.5)
                analysis.loc[analysis.index[i], 'reversion_weight'] = weights.get('mean_reversion', 0.5)
        
        return analysis
    
    def forecast_volatility(self, data: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """
        Forecast future volatility
        
        Args:
            data: Historical OHLCV data
            horizon: Forecast horizon in days
            
        Returns:
            Series of volatility forecasts
        """
        returns = data['close'].pct_change().dropna()
        current_vol = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
        
        # Simple EWMA forecast
        ewma_vol = current_vol.ewm(span=self.vol_lookback).mean()
        
        # Extend forecast
        last_vol = ewma_vol.iloc[-1]
        mean_vol = current_vol.mean()
        
        # Mean reversion in volatility
        forecast = []
        vol = last_vol
        for _ in range(horizon):
            vol = vol * 0.9 + mean_vol * 0.1  # 10% mean reversion per day
            forecast.append(vol)
        
        forecast_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        return pd.Series(forecast, index=forecast_dates)