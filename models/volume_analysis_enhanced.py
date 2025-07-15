#!/usr/bin/env python3
"""
Enhanced Volume Analysis Trading Models
Implements OBV, Volume Spike, VPA, and more volume-based strategies
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class OBVModel:
    """On-Balance Volume trading model"""
    
    ma_period: int = 20
    divergence_lookback: int = 30
    breakout_threshold: float = 0.05  # 5% above MA for breakout
    use_signal_line: bool = True
    signal_period: int = 9
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on OBV"""
        if 'volume' not in data.columns:
            logger.warning("No volume data available for OBV")
            return pd.Series(0, index=data.index)
        
        # Calculate OBV
        obv = self._calculate_obv(data)
        
        # Calculate OBV moving average
        obv_ma = obv.rolling(self.ma_period).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # OBV breakout signals
        obv_breakout_up = obv > obv_ma * (1 + self.breakout_threshold)
        obv_breakout_down = obv < obv_ma * (1 - self.breakout_threshold)
        
        # Basic signals
        signals[obv_breakout_up] = 1
        signals[obv_breakout_down] = -1
        
        # Signal line crosses
        if self.use_signal_line:
            obv_signal = obv.ewm(span=self.signal_period, adjust=False).mean()
            
            obv_cross_up = (obv > obv_signal) & (obv.shift(1) <= obv_signal.shift(1))
            obv_cross_down = (obv < obv_signal) & (obv.shift(1) >= obv_signal.shift(1))
            
            # Combine with existing signals
            signals[(signals == 0) & obv_cross_up] = 1
            signals[(signals == 0) & obv_cross_down] = -1
        
        # Divergence detection
        divergence_signals = self._detect_obv_divergence(data['close'], obv)
        signals = signals + divergence_signals
        signals = signals.clip(-1, 1)
        
        return signals.astype(int)
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(0, index=data.index, dtype=float)
        obv.iloc[0] = data['volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _detect_obv_divergence(self, prices: pd.Series, obv: pd.Series) -> pd.Series:
        """Detect divergences between price and OBV"""
        signals = pd.Series(0, index=prices.index)
        
        for i in range(self.divergence_lookback, len(prices)):
            window_prices = prices.iloc[i-self.divergence_lookback:i]
            window_obv = obv.iloc[i-self.divergence_lookback:i]
            
            # Find peaks and troughs
            price_peaks = (window_prices.shift(-1) < window_prices) & (window_prices.shift(1) < window_prices)
            price_troughs = (window_prices.shift(-1) > window_prices) & (window_prices.shift(1) > window_prices)
            
            # Bearish divergence: price higher high, OBV lower high
            if price_peaks.any():
                last_peak_idx = price_peaks[price_peaks].index[-1]
                if prices.iloc[i] > prices.loc[last_peak_idx] and obv.iloc[i] < obv.loc[last_peak_idx]:
                    signals.iloc[i] = -1
            
            # Bullish divergence: price lower low, OBV higher low
            if price_troughs.any():
                last_trough_idx = price_troughs[price_troughs].index[-1]
                if prices.iloc[i] < prices.loc[last_trough_idx] and obv.iloc[i] > obv.loc[last_trough_idx]:
                    signals.iloc[i] = 1
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'ma_period': self.ma_period,
            'divergence_lookback': self.divergence_lookback,
            'breakout_threshold': self.breakout_threshold,
            'use_signal_line': self.use_signal_line,
            'signal_period': self.signal_period
        }


@dataclass
class VolumeSpikeModel:
    """Volume spike detection trading model"""
    
    spike_threshold: float = 2.0  # Volume must be 2x average
    lookback_period: int = 20
    price_change_threshold: float = 0.01  # 1% minimum price change
    consolidation_periods: int = 5  # Periods of low volume before spike
    use_climax_detection: bool = True
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on volume spikes"""
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)
        
        signals = pd.Series(0, index=data.index)
        
        # Calculate volume metrics
        avg_volume = data['volume'].rolling(self.lookback_period).mean()
        volume_ratio = data['volume'] / avg_volume
        
        # Price change
        price_change = data['close'].pct_change()
        
        # Detect volume spikes
        volume_spike = volume_ratio > self.spike_threshold
        
        # Detect consolidation before spike
        low_volume_periods = pd.Series(0, index=data.index)
        for i in range(self.consolidation_periods, len(data)):
            recent_volumes = volume_ratio.iloc[i-self.consolidation_periods:i]
            if (recent_volumes < 0.8).all():  # All below 80% of average
                low_volume_periods.iloc[i] = 1
        
        # Generate signals
        for i in range(1, len(signals)):
            if volume_spike.iloc[i]:
                # Bullish spike: high volume + positive price change
                if price_change.iloc[i] > self.price_change_threshold:
                    # Extra bullish if coming from consolidation
                    if low_volume_periods.iloc[i] == 1:
                        signals.iloc[i] = 1
                    # Or if it's a breakout
                    elif data['close'].iloc[i] > data['high'].iloc[i-self.lookback_period:i].max():
                        signals.iloc[i] = 1
                
                # Bearish spike: high volume + negative price change
                elif price_change.iloc[i] < -self.price_change_threshold:
                    signals.iloc[i] = -1
        
        # Climax volume detection
        if self.use_climax_detection:
            climax_signals = self._detect_volume_climax(data, volume_ratio)
            signals = signals + climax_signals
            signals = signals.clip(-1, 1)
        
        return signals.astype(int)
    
    def _detect_volume_climax(self, data: pd.DataFrame, volume_ratio: pd.Series) -> pd.Series:
        """Detect buying/selling climax"""
        signals = pd.Series(0, index=data.index)
        
        # Climax = extreme volume + reversal pattern
        extreme_volume = volume_ratio > 3.0  # 3x average volume
        
        for i in range(2, len(data)):
            if extreme_volume.iloc[i-1]:
                # Buying climax: huge volume on up day followed by reversal
                if (data['close'].iloc[i-1] > data['open'].iloc[i-1] and 
                    data['close'].iloc[i] < data['open'].iloc[i]):
                    signals.iloc[i] = -1
                
                # Selling climax: huge volume on down day followed by reversal
                elif (data['close'].iloc[i-1] < data['open'].iloc[i-1] and 
                      data['close'].iloc[i] > data['open'].iloc[i]):
                    signals.iloc[i] = 1
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'spike_threshold': self.spike_threshold,
            'lookback_period': self.lookback_period,
            'price_change_threshold': self.price_change_threshold,
            'consolidation_periods': self.consolidation_periods,
            'use_climax_detection': self.use_climax_detection
        }


@dataclass
class VolumePriceAnalysisModel:
    """Volume Price Analysis (VPA) trading model based on Wyckoff method"""
    
    lookback_period: int = 20
    spread_threshold: float = 0.015  # 1.5% for wide spread
    volume_threshold: float = 1.5  # 1.5x average for high volume
    use_effort_result: bool = True
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on VPA"""
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)
        
        signals = pd.Series(0, index=data.index)
        
        # Calculate spread and volume metrics
        spread = (data['high'] - data['low']) / data['close']
        avg_spread = spread.rolling(self.lookback_period).mean()
        
        avg_volume = data['volume'].rolling(self.lookback_period).mean()
        volume_ratio = data['volume'] / avg_volume
        
        # Classify bars
        bar_types = self._classify_vpa_bars(data, spread, avg_spread, volume_ratio)
        
        # Generate signals based on bar types
        for i in range(1, len(signals)):
            bar_type = bar_types.iloc[i]
            
            if bar_type == 'stopping_volume':
                # Potential reversal
                if data['close'].iloc[i] < data['close'].iloc[i-1]:
                    signals.iloc[i] = 1  # Selling exhaustion
                else:
                    signals.iloc[i] = -1  # Buying exhaustion
                    
            elif bar_type == 'test':
                # Successful test = continuation
                if data['close'].iloc[i] > data['open'].iloc[i]:
                    signals.iloc[i] = 1
                else:
                    signals.iloc[i] = -1
                    
            elif bar_type == 'no_demand':
                signals.iloc[i] = -1  # Bearish
                
            elif bar_type == 'no_supply':
                signals.iloc[i] = 1  # Bullish
        
        # Effort vs Result analysis
        if self.use_effort_result:
            effort_signals = self._analyze_effort_result(data, volume_ratio)
            signals = signals + effort_signals
            signals = signals.clip(-1, 1)
        
        return signals.astype(int)
    
    def _classify_vpa_bars(self, data: pd.DataFrame, spread: pd.Series, 
                          avg_spread: pd.Series, volume_ratio: pd.Series) -> pd.Series:
        """Classify bars according to VPA principles"""
        bar_types = pd.Series('normal', index=data.index)
        
        for i in range(1, len(data)):
            high_volume = volume_ratio.iloc[i] > self.volume_threshold
            low_volume = volume_ratio.iloc[i] < 0.5
            wide_spread = spread.iloc[i] > avg_spread.iloc[i] * 1.5
            narrow_spread = spread.iloc[i] < avg_spread.iloc[i] * 0.5
            
            up_bar = data['close'].iloc[i] > data['close'].iloc[i-1]
            down_bar = data['close'].iloc[i] < data['close'].iloc[i-1]
            
            # Stopping Volume
            if high_volume and wide_spread:
                bar_types.iloc[i] = 'stopping_volume'
            
            # No Demand
            elif up_bar and narrow_spread and low_volume:
                bar_types.iloc[i] = 'no_demand'
            
            # No Supply  
            elif down_bar and narrow_spread and low_volume:
                bar_types.iloc[i] = 'no_supply'
            
            # Test
            elif low_volume and narrow_spread:
                bar_types.iloc[i] = 'test'
            
            # Effort to rise
            elif up_bar and high_volume and not wide_spread:
                bar_types.iloc[i] = 'effort_to_rise'
            
            # Effort to fall
            elif down_bar and high_volume and not wide_spread:
                bar_types.iloc[i] = 'effort_to_fall'
        
        return bar_types
    
    def _analyze_effort_result(self, data: pd.DataFrame, volume_ratio: pd.Series) -> pd.Series:
        """Analyze effort (volume) vs result (price movement)"""
        signals = pd.Series(0, index=data.index)
        
        price_change = data['close'].pct_change()
        
        for i in range(5, len(data)):
            # High effort, low result = potential reversal
            if volume_ratio.iloc[i] > 2.0 and abs(price_change.iloc[i]) < 0.002:
                # Check direction of previous move
                if price_change.iloc[i-5:i].sum() > 0:
                    signals.iloc[i] = -1  # Bullish exhaustion
                else:
                    signals.iloc[i] = 1  # Bearish exhaustion
            
            # Low effort, high result = trend strength
            elif volume_ratio.iloc[i] < 0.7 and abs(price_change.iloc[i]) > 0.01:
                if price_change.iloc[i] > 0:
                    signals.iloc[i] = 1  # Bullish strength
                else:
                    signals.iloc[i] = -1  # Bearish strength
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'lookback_period': self.lookback_period,
            'spread_threshold': self.spread_threshold,
            'volume_threshold': self.volume_threshold,
            'use_effort_result': self.use_effort_result
        }


@dataclass
class VolumeConfirmedBreakoutModel:
    """Breakout model with volume confirmation"""
    
    breakout_period: int = 20
    volume_multiplier: float = 1.5
    price_threshold: float = 0.02  # 2% above resistance
    consolidation_min_periods: int = 10
    retest_window: int = 5
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals for volume-confirmed breakouts"""
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)
        
        signals = pd.Series(0, index=data.index)
        
        # Calculate resistance and support levels
        resistance = data['high'].rolling(self.breakout_period).max()
        support = data['low'].rolling(self.breakout_period).min()
        
        # Volume metrics
        avg_volume = data['volume'].rolling(20).mean()
        
        # Detect consolidation
        consolidation = self._detect_consolidation(data, self.consolidation_min_periods)
        
        for i in range(self.breakout_period, len(data)):
            # Breakout conditions
            breakout_up = (data['close'].iloc[i] > resistance.iloc[i-1] * (1 + self.price_threshold))
            breakout_down = (data['close'].iloc[i] < support.iloc[i-1] * (1 - self.price_threshold))
            
            volume_surge = data['volume'].iloc[i] > avg_volume.iloc[i] * self.volume_multiplier
            
            # Bullish breakout
            if breakout_up and volume_surge and consolidation.iloc[i-1]:
                signals.iloc[i] = 1
                
                # Look for successful retest
                if i + self.retest_window < len(data):
                    retest_low = data['low'].iloc[i+1:i+self.retest_window+1].min()
                    if retest_low > resistance.iloc[i-1] * 0.99:  # Successful retest
                        signals.iloc[i+self.retest_window] = 1
            
            # Bearish breakout
            elif breakout_down and volume_surge:
                signals.iloc[i] = -1
        
        return signals.astype(int)
    
    def _detect_consolidation(self, data: pd.DataFrame, min_periods: int) -> pd.Series:
        """Detect price consolidation periods"""
        consolidation = pd.Series(False, index=data.index)
        
        # Calculate ATR for volatility measure
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()
        
        # Low ATR = consolidation
        low_volatility = atr < atr.rolling(50).quantile(0.3)
        
        # Mark consolidation periods
        for i in range(min_periods, len(data)):
            if low_volatility.iloc[i-min_periods:i].sum() >= min_periods * 0.7:
                consolidation.iloc[i] = True
        
        return consolidation
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'breakout_period': self.breakout_period,
            'volume_multiplier': self.volume_multiplier,
            'price_threshold': self.price_threshold,
            'consolidation_min_periods': self.consolidation_min_periods,
            'retest_window': self.retest_window
        }


@dataclass
class VolumeWeightedPriceModel:
    """VWAP-based trading model"""
    
    lookback_period: int = 20
    num_std: float = 2.0
    use_anchored_vwap: bool = True
    anchor_days: int = 65  # Quarterly anchor
    trend_filter: bool = True
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on VWAP"""
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)
        
        signals = pd.Series(0, index=data.index)
        
        # Calculate VWAP
        vwap = self._calculate_vwap(data)
        
        # Calculate VWAP bands
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap_std = self._calculate_vwap_std(data, vwap, typical_price)
        
        upper_band = vwap + vwap_std * self.num_std
        lower_band = vwap - vwap_std * self.num_std
        
        # Trend filter
        if self.trend_filter:
            trend = data['close'].rolling(50).mean()
            uptrend = data['close'] > trend
            downtrend = data['close'] < trend
        else:
            uptrend = pd.Series(True, index=data.index)
            downtrend = pd.Series(True, index=data.index)
        
        # Generate signals
        for i in range(1, len(signals)):
            # Mean reversion at bands
            if data['low'].iloc[i] <= lower_band.iloc[i] and uptrend.iloc[i]:
                signals.iloc[i] = 1
            elif data['high'].iloc[i] >= upper_band.iloc[i] and downtrend.iloc[i]:
                signals.iloc[i] = -1
            
            # VWAP crosses
            if (data['close'].iloc[i] > vwap.iloc[i] and 
                data['close'].iloc[i-1] <= vwap.iloc[i-1] and
                uptrend.iloc[i]):
                signals.iloc[i] = 1
            elif (data['close'].iloc[i] < vwap.iloc[i] and 
                  data['close'].iloc[i-1] >= vwap.iloc[i-1] and
                  downtrend.iloc[i]):
                signals.iloc[i] = -1
        
        return signals.astype(int)
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        if self.use_anchored_vwap:
            vwap = pd.Series(index=data.index, dtype=float)
            
            for i in range(0, len(data), self.anchor_days):
                end_idx = min(i + self.anchor_days, len(data))
                
                cumulative_tpv = (typical_price.iloc[i:end_idx] * data['volume'].iloc[i:end_idx]).cumsum()
                cumulative_volume = data['volume'].iloc[i:end_idx].cumsum()
                
                vwap.iloc[i:end_idx] = cumulative_tpv / cumulative_volume
        else:
            cumulative_tpv = (typical_price * data['volume']).rolling(self.lookback_period).sum()
            cumulative_volume = data['volume'].rolling(self.lookback_period).sum()
            vwap = cumulative_tpv / cumulative_volume
        
        return vwap
    
    def _calculate_vwap_std(self, data: pd.DataFrame, vwap: pd.Series, 
                           typical_price: pd.Series) -> pd.Series:
        """Calculate VWAP standard deviation"""
        squared_diff = (typical_price - vwap) ** 2
        
        if self.use_anchored_vwap:
            vwap_variance = pd.Series(index=data.index, dtype=float)
            
            for i in range(0, len(data), self.anchor_days):
                end_idx = min(i + self.anchor_days, len(data))
                
                weighted_squared_diff = squared_diff.iloc[i:end_idx] * data['volume'].iloc[i:end_idx]
                cumulative_volume = data['volume'].iloc[i:end_idx].cumsum()
                
                variance = weighted_squared_diff.cumsum() / cumulative_volume
                vwap_variance.iloc[i:end_idx] = variance
        else:
            weighted_squared_diff = squared_diff * data['volume']
            cumulative_volume = data['volume'].rolling(self.lookback_period).sum()
            
            vwap_variance = weighted_squared_diff.rolling(self.lookback_period).sum() / cumulative_volume
        
        return np.sqrt(vwap_variance)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'lookback_period': self.lookback_period,
            'num_std': self.num_std,
            'use_anchored_vwap': self.use_anchored_vwap,
            'anchor_days': self.anchor_days,
            'trend_filter': self.trend_filter
        }


@dataclass
class VolumeAnalysisEnsemble:
    """Ensemble of all volume analysis models"""
    
    def __init__(self):
        self.models = {
            'obv': OBVModel(),
            'volume_spike': VolumeSpikeModel(),
            'vpa': VolumePriceAnalysisModel(),
            'breakout': VolumeConfirmedBreakoutModel(),
            'vwap': VolumeWeightedPriceModel()
        }
        
        self.weights = {
            'obv': 0.20,
            'volume_spike': 0.15,
            'vpa': 0.25,
            'breakout': 0.20,
            'vwap': 0.20
        }
    
    def predict(self, data: pd.DataFrame, voting_threshold: float = 0.3) -> pd.Series:
        """Generate ensemble signals from all volume models"""
        all_signals = {}
        
        # Get signals from each model
        for name, model in self.models.items():
            try:
                signals = model.predict(data)
                all_signals[name] = signals
            except Exception as e:
                logger.warning(f"Volume model {name} failed: {e}")
                all_signals[name] = pd.Series(0, index=data.index)
        
        # Weighted voting
        weighted_sum = pd.Series(0.0, index=data.index)
        
        for name, signals in all_signals.items():
            weighted_sum += signals * self.weights.get(name, 0.2)
        
        # Generate final signals
        final_signals = pd.Series(0, index=data.index)
        final_signals[weighted_sum > voting_threshold] = 1
        final_signals[weighted_sum < -voting_threshold] = -1
        
        return final_signals.astype(int)
    
    def get_individual_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get signals from each model separately"""
        results = {}
        for name, model in self.models.items():
            try:
                results[name] = model.predict(data)
            except Exception as e:
                logger.error(f"Volume model {name} failed: {e}")
                results[name] = pd.Series(0, index=data.index)
        return results
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'models': list(self.models.keys()),
            'weights': self.weights,
            'individual_params': {
                name: model.get_parameters() 
                for name, model in self.models.items()
            }
        }