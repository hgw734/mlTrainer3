#!/usr/bin/env python3
"""
Market Regime Detection Model
Uses Hidden Markov Models to identify market states
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logging.warning("hmmlearn not available. Install with: pip install hmmlearn")

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class MarketRegimeDetector:
    """
    Detects market regimes (bull, bear, sideways) using Hidden Markov Models
    Adapts trading strategy based on current market state
    """
    
    n_states: int = 3  # Number of hidden states
    n_features: int = 5  # Number of observable features
    covariance_type: str = 'diag'  # Type of covariance matrix
    n_iter: int = 100  # Number of EM iterations
    random_state: int = 42
    min_samples: int = 100  # Minimum samples for training
    
    # Regime labels
    REGIME_NAMES = {
        0: 'bear_volatile',
        1: 'sideways_calm', 
        2: 'bull_trending'
    }
    
    def __post_init__(self):
        """Initialize internal state"""
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required but not installed")
            
        self.is_fitted = False
        self.model = None
        self.scaler = StandardScaler()
        self.regime_stats = None
        self.current_regime = None
        
    def fit(self, data: pd.DataFrame) -> 'MarketRegimeDetector':
        """
        Fit the HMM model to identify market regimes
        
        Args:
            data: OHLCV data
            
        Returns:
            Self for chaining
        """
        # Calculate features for regime detection
        features = self._calculate_regime_features(data)
        
        if len(features) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples, got {len(features)}")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(features_scaled)
        
        # Predict states
        states = self.model.predict(features_scaled)
        
        # Calculate regime statistics
        self._calculate_regime_statistics(data, states, features)
        
        self.is_fitted = True
        logger.info(f"Market regime model fitted with {self.n_states} states")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate regime-aware trading signals
        
        Args:
            data: OHLCV data
            
        Returns:
            Series of signals: 1 (buy), 0 (hold), -1 (sell)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get current regimes
        regimes = self.get_regimes(data)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index[len(data) - len(regimes):])
        
        # Apply regime-specific strategies
        for i, (date, regime) in enumerate(regimes.items()):
            if i < 20:  # Need history for indicators
                continue
                
            # Get recent data window
            window_end = data.index.get_loc(date) + 1
            window_start = max(0, window_end - 50)
            window_data = data.iloc[window_start:window_end]
            
            # Apply strategy based on regime
            signal = self._get_regime_signal(window_data, regime)
            signals.loc[date] = signal
        
        return signals.astype(int)
    
    def get_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Get market regime for each time period
        
        Returns Series with regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting regimes")
        
        # Calculate features
        features = self._calculate_regime_features(data)
        features_scaled = self.scaler.transform(features)
        
        # Predict states
        states = self.model.predict(features_scaled)
        
        # Convert to regime names
        regimes = pd.Series(
            [self.REGIME_NAMES[state] for state in states],
            index=data.index[len(data) - len(states):]
        )
        
        return regimes
    
    def get_regime_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get probability of each regime at each time point
        
        Returns DataFrame with probabilities for each regime
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting probabilities")
        
        # Calculate features
        features = self._calculate_regime_features(data)
        features_scaled = self.scaler.transform(features)
        
        # Get posterior probabilities
        _, posteriors = self.model.score_samples(features_scaled)
        
        # Create DataFrame
        prob_df = pd.DataFrame(
            posteriors,
            columns=[f'prob_{self.REGIME_NAMES[i]}' for i in range(self.n_states)],
            index=data.index[len(data) - len(features):]
        )
        
        return prob_df
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate features for regime detection"""
        features = []
        
        # 1. Returns
        returns = data['close'].pct_change()
        features.append(returns.fillna(0))
        
        # 2. Volatility (20-day realized)
        volatility = returns.rolling(20).std() * np.sqrt(252)
        features.append(volatility.fillna(method='bfill'))
        
        # 3. Trend strength (price vs 50-day MA)
        ma_50 = data['close'].rolling(50).mean()
        trend = (data['close'] - ma_50) / ma_50
        features.append(trend.fillna(0))
        
        # 4. Volume ratio
        if 'volume' in data.columns:
            volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
            features.append(volume_ratio.fillna(1))
        else:
            features.append(pd.Series(1, index=data.index))
        
        # 5. Market breadth (high-low spread)
        breadth = (data['high'] - data['low']) / data['close']
        features.append(breadth.fillna(method='bfill'))
        
        # Stack features
        feature_matrix = np.column_stack([f.values for f in features])
        
        # Remove NaN rows
        mask = ~np.isnan(feature_matrix).any(axis=1)
        
        return feature_matrix[mask]
    
    def _calculate_regime_statistics(self, data: pd.DataFrame, states: np.ndarray, 
                                   features: np.ndarray):
        """Calculate statistics for each regime"""
        returns = data['close'].pct_change().iloc[-len(states):].values
        
        self.regime_stats = {}
        
        for state in range(self.n_states):
            mask = states == state
            
            if mask.sum() > 0:
                regime_returns = returns[mask]
                regime_features = features[mask]
                
                self.regime_stats[self.REGIME_NAMES[state]] = {
                    'mean_return': np.nanmean(regime_returns),
                    'volatility': np.nanstd(regime_returns) * np.sqrt(252),
                    'frequency': mask.sum() / len(states),
                    'avg_duration': self._calculate_avg_duration(states, state),
                    'feature_means': np.mean(regime_features, axis=0)
                }
        
        # Sort states by mean return to assign proper labels
        sorted_states = sorted(
            [(state, stats['mean_return']) for state, stats in self.regime_stats.items()],
            key=lambda x: x[1]
        )
        
        # Reassign labels based on returns
        if len(sorted_states) == 3:
            # Lowest return = bear, middle = sideways, highest = bull
            relabeling = {
                sorted_states[0][0]: 'bear_volatile',
                sorted_states[1][0]: 'sideways_calm',
                sorted_states[2][0]: 'bull_trending'
            }
            
            # Update regime stats with correct labels
            new_stats = {}
            for old_label, new_label in relabeling.items():
                new_stats[new_label] = self.regime_stats[old_label]
            self.regime_stats = new_stats
    
    def _calculate_avg_duration(self, states: np.ndarray, target_state: int) -> float:
        """Calculate average duration of a regime"""
        durations = []
        current_duration = 0
        
        for state in states:
            if state == target_state:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def _get_regime_signal(self, window_data: pd.DataFrame, regime: str) -> int:
        """Get trading signal based on current regime"""
        
        # Calculate indicators
        close = window_data['close']
        returns = close.pct_change()
        
        # Short-term momentum
        momentum_5 = close.iloc[-1] / close.iloc[-5] - 1
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands position
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_position = (close.iloc[-1] - ma_20.iloc[-1]) / (2 * std_20.iloc[-1])
        
        # Apply regime-specific strategy
        if regime == 'bull_trending':
            # Trend following in bull market
            if momentum_5 > 0.01 and rsi < 70:
                return 1  # Buy on momentum
            elif momentum_5 < -0.02 or rsi > 80:
                return -1  # Sell on weakness or overbought
                
        elif regime == 'bear_volatile':
            # Conservative in bear market
            if bb_position < -1.5 and rsi < 30:
                return 1  # Buy extreme oversold
            elif bb_position > 0 or rsi > 60:
                return -1  # Sell rallies
                
        elif regime == 'sideways_calm':
            # Mean reversion in sideways market
            if bb_position < -1 and rsi < 35:
                return 1  # Buy at lower band
            elif bb_position > 1 and rsi > 65:
                return -1  # Sell at upper band
        
        return 0  # Hold
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        params = {
            'n_states': self.n_states,
            'n_features': self.n_features,
            'covariance_type': self.covariance_type,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            params['regime_stats'] = self.regime_stats
            params['transition_matrix'] = self.model.transmat_.tolist()
            
        return params
    
    def plot_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data for regime visualization"""
        regimes = self.get_regimes(data)
        probabilities = self.get_regime_probabilities(data)
        
        # Prepare plot data
        plot_data = {
            'dates': regimes.index.tolist(),
            'regimes': regimes.tolist(),
            'prices': data.loc[regimes.index, 'close'].tolist(),
            'probabilities': probabilities.to_dict('records'),
            'regime_colors': {
                'bear_volatile': '#dc3545',     # Red
                'sideways_calm': '#ffc107',     # Yellow
                'bull_trending': '#28a745'      # Green
            }
        }
        
        return plot_data