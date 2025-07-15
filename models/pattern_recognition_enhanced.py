#!/usr/bin/env python3
"""
Enhanced Pattern Recognition Trading Models
Implements candlestick patterns, chart patterns, and support/resistance detection
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
import warnings
from scipy.signal import argrelextrema
from scipy.stats import linregress

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class CandlestickPatternsModel:
    """Candlestick pattern recognition trading model"""
    
    min_body_size: float = 0.001  # Minimum body size as % of price
    doji_threshold: float = 0.0003  # Max body size for doji
    trend_lookback: int = 10  # Periods to determine trend
    confirmation_required: bool = True
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on candlestick patterns"""
        signals = pd.Series(0, index=data.index)
        
        # Calculate candlestick components
        body = data['close'] - data['open']
        body_size = abs(body)
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        full_range = data['high'] - data['low']
        
        # Determine trend
        trend = self._determine_trend(data['close'], self.trend_lookback)
        
        # Detect patterns
        for i in range(2, len(data)):
            # Hammer/Hanging Man
            if self._is_hammer(body_size.iloc[i], lower_shadow.iloc[i], 
                              upper_shadow.iloc[i], full_range.iloc[i]):
                if trend.iloc[i] == -1:  # Hammer in downtrend
                    signals.iloc[i] = 1
                elif trend.iloc[i] == 1:  # Hanging man in uptrend
                    signals.iloc[i] = -1
            
            # Inverted Hammer/Shooting Star
            elif self._is_inverted_hammer(body_size.iloc[i], upper_shadow.iloc[i], 
                                         lower_shadow.iloc[i], full_range.iloc[i]):
                if trend.iloc[i] == -1:  # Inverted hammer in downtrend
                    signals.iloc[i] = 1
                elif trend.iloc[i] == 1:  # Shooting star in uptrend
                    signals.iloc[i] = -1
            
            # Bullish/Bearish Engulfing
            elif i >= 1:
                if self._is_bullish_engulfing(body.iloc[i-1], body.iloc[i],
                                             data.iloc[i-1], data.iloc[i]):
                    signals.iloc[i] = 1
                elif self._is_bearish_engulfing(body.iloc[i-1], body.iloc[i],
                                               data.iloc[i-1], data.iloc[i]):
                    signals.iloc[i] = -1
            
            # Doji patterns
            if self._is_doji(body_size.iloc[i], data['close'].iloc[i]):
                doji_signal = self._analyze_doji(data.iloc[i], trend.iloc[i], i, data)
                if doji_signal != 0:
                    signals.iloc[i] = doji_signal
            
            # Morning/Evening Star (3-candle patterns)
            if i >= 2:
                if self._is_morning_star(data.iloc[i-2:i+1], body.iloc[i-2:i+1]):
                    signals.iloc[i] = 1
                elif self._is_evening_star(data.iloc[i-2:i+1], body.iloc[i-2:i+1]):
                    signals.iloc[i] = -1
        
        # Apply confirmation if required
        if self.confirmation_required:
            signals = self._apply_confirmation(signals, data)
        
        return signals.astype(int)
    
    def _determine_trend(self, prices: pd.Series, lookback: int) -> pd.Series:
        """Determine trend direction"""
        trend = pd.Series(0, index=prices.index)
        ma = prices.rolling(lookback).mean()
        
        trend[prices > ma * 1.01] = 1  # Uptrend
        trend[prices < ma * 0.99] = -1  # Downtrend
        
        return trend
    
    def _is_hammer(self, body_size: float, lower_shadow: float, 
                   upper_shadow: float, full_range: float) -> bool:
        """Check if candle is a hammer pattern"""
        if full_range == 0:
            return False
        
        return (lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5 and
                body_size > self.min_body_size)
    
    def _is_inverted_hammer(self, body_size: float, upper_shadow: float,
                           lower_shadow: float, full_range: float) -> bool:
        """Check if candle is an inverted hammer"""
        if full_range == 0:
            return False
        
        return (upper_shadow > body_size * 2 and 
                lower_shadow < body_size * 0.5 and
                body_size > self.min_body_size)
    
    def _is_bullish_engulfing(self, prev_body: float, curr_body: float,
                             prev_candle: pd.Series, curr_candle: pd.Series) -> bool:
        """Check for bullish engulfing pattern"""
        return (prev_body < 0 and  # Previous was bearish
                curr_body > 0 and  # Current is bullish
                curr_candle['open'] < prev_candle['close'] and
                curr_candle['close'] > prev_candle['open'])
    
    def _is_bearish_engulfing(self, prev_body: float, curr_body: float,
                             prev_candle: pd.Series, curr_candle: pd.Series) -> bool:
        """Check for bearish engulfing pattern"""
        return (prev_body > 0 and  # Previous was bullish
                curr_body < 0 and  # Current is bearish
                curr_candle['open'] > prev_candle['close'] and
                curr_candle['close'] < prev_candle['open'])
    
    def _is_doji(self, body_size: float, close: float) -> bool:
        """Check if candle is a doji"""
        return body_size / close < self.doji_threshold
    
    def _analyze_doji(self, candle: pd.Series, trend: int, idx: int, 
                     data: pd.DataFrame) -> int:
        """Analyze doji pattern in context"""
        # Doji at trend extremes suggests reversal
        if trend == 1 and idx > 20:
            # Check if at recent high
            if candle['high'] >= data['high'].iloc[idx-20:idx].max():
                return -1
        elif trend == -1 and idx > 20:
            # Check if at recent low
            if candle['low'] <= data['low'].iloc[idx-20:idx].min():
                return 1
        
        return 0
    
    def _is_morning_star(self, candles: pd.DataFrame, bodies: pd.Series) -> bool:
        """Check for morning star pattern"""
        if len(candles) < 3:
            return False
        
        # First: large bearish candle
        # Second: small body (star)
        # Third: large bullish candle
        return (bodies.iloc[0] < -self.min_body_size * 2 and
                abs(bodies.iloc[1]) < self.min_body_size and
                bodies.iloc[2] > self.min_body_size * 2 and
                candles['close'].iloc[2] > candles['open'].iloc[0])
    
    def _is_evening_star(self, candles: pd.DataFrame, bodies: pd.Series) -> bool:
        """Check for evening star pattern"""
        if len(candles) < 3:
            return False
        
        # First: large bullish candle
        # Second: small body (star)
        # Third: large bearish candle
        return (bodies.iloc[0] > self.min_body_size * 2 and
                abs(bodies.iloc[1]) < self.min_body_size and
                bodies.iloc[2] < -self.min_body_size * 2 and
                candles['close'].iloc[2] < candles['open'].iloc[0])
    
    def _apply_confirmation(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply confirmation to signals"""
        confirmed_signals = pd.Series(0, index=signals.index)
        
        for i in range(1, len(signals)-1):
            if signals.iloc[i] != 0:
                # Confirm with next candle
                if signals.iloc[i] == 1:  # Bullish signal
                    if data['close'].iloc[i+1] > data['close'].iloc[i]:
                        confirmed_signals.iloc[i+1] = 1
                else:  # Bearish signal
                    if data['close'].iloc[i+1] < data['close'].iloc[i]:
                        confirmed_signals.iloc[i+1] = -1
        
        return confirmed_signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'min_body_size': self.min_body_size,
            'doji_threshold': self.doji_threshold,
            'trend_lookback': self.trend_lookback,
            'confirmation_required': self.confirmation_required
        }


@dataclass
class SupportResistanceModel:
    """Support and resistance level detection and trading"""
    
    lookback_period: int = 100
    min_touches: int = 3
    tolerance: float = 0.002  # 0.2% tolerance for level
    breakout_threshold: float = 0.003  # 0.3% beyond level
    volume_confirmation: bool = True
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on support/resistance"""
        signals = pd.Series(0, index=data.index)
        
        # Detect support and resistance levels
        support_levels, resistance_levels = self._detect_sr_levels(data)
        
        # Generate signals based on price action at levels
        for i in range(1, len(data)):
            price = data['close'].iloc[i]
            prev_price = data['close'].iloc[i-1]
            
            # Check for bounces and breakouts at each level
            for resistance in resistance_levels:
                # Resistance bounce
                if (prev_price < resistance and 
                    data['high'].iloc[i] >= resistance * (1 - self.tolerance) and
                    price < resistance):
                    signals.iloc[i] = -1
                    break
                
                # Resistance breakout
                if (prev_price < resistance and 
                    price > resistance * (1 + self.breakout_threshold)):
                    if self._confirm_breakout(data, i, 'bullish'):
                        signals.iloc[i] = 1
                        break
            
            for support in support_levels:
                # Support bounce
                if (prev_price > support and 
                    data['low'].iloc[i] <= support * (1 + self.tolerance) and
                    price > support):
                    signals.iloc[i] = 1
                    break
                
                # Support breakdown
                if (prev_price > support and 
                    price < support * (1 - self.breakout_threshold)):
                    if self._confirm_breakout(data, i, 'bearish'):
                        signals.iloc[i] = -1
                        break
        
        return signals.astype(int)
    
    def _detect_sr_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Detect support and resistance levels"""
        # Find local extrema
        highs = data['high'].iloc[-self.lookback_period:]
        lows = data['low'].iloc[-self.lookback_period:]
        
        # Find peaks and troughs
        peak_indices = argrelextrema(highs.values, np.greater, order=5)[0]
        trough_indices = argrelextrema(lows.values, np.less, order=5)[0]
        
        # Cluster similar levels
        resistance_levels = self._cluster_levels(
            highs.iloc[peak_indices].values if len(peak_indices) > 0 else []
        )
        support_levels = self._cluster_levels(
            lows.iloc[trough_indices].values if len(trough_indices) > 0 else []
        )
        
        # Filter by minimum touches
        resistance_levels = self._filter_by_touches(
            resistance_levels, data, 'resistance'
        )
        support_levels = self._filter_by_touches(
            support_levels, data, 'support'
        )
        
        return support_levels, resistance_levels
    
    def _cluster_levels(self, levels: np.ndarray) -> List[float]:
        """Cluster nearby levels"""
        if len(levels) == 0:
            return []
        
        clustered = []
        sorted_levels = sorted(levels)
        
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if level <= current_cluster[-1] * (1 + self.tolerance):
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def _filter_by_touches(self, levels: List[float], data: pd.DataFrame, 
                          level_type: str) -> List[float]:
        """Filter levels by minimum number of touches"""
        filtered = []
        
        for level in levels:
            touches = 0
            
            for i in range(len(data)):
                if level_type == 'resistance':
                    if (data['high'].iloc[i] >= level * (1 - self.tolerance) and
                        data['high'].iloc[i] <= level * (1 + self.tolerance)):
                        touches += 1
                else:  # support
                    if (data['low'].iloc[i] <= level * (1 + self.tolerance) and
                        data['low'].iloc[i] >= level * (1 - self.tolerance)):
                        touches += 1
            
            if touches >= self.min_touches:
                filtered.append(level)
        
        return filtered
    
    def _confirm_breakout(self, data: pd.DataFrame, idx: int, direction: str) -> bool:
        """Confirm breakout with volume"""
        if not self.volume_confirmation or 'volume' not in data.columns:
            return True
        
        # Check for volume surge
        if idx >= 20:
            avg_volume = data['volume'].iloc[idx-20:idx].mean()
            current_volume = data['volume'].iloc[idx]
            
            return current_volume > avg_volume * 1.5
        
        return True
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'lookback_period': self.lookback_period,
            'min_touches': self.min_touches,
            'tolerance': self.tolerance,
            'breakout_threshold': self.breakout_threshold,
            'volume_confirmation': self.volume_confirmation
        }


@dataclass
class ChartPatternRecognitionModel:
    """Chart pattern recognition (triangles, channels, etc.)"""
    
    min_pattern_length: int = 20
    max_pattern_length: int = 60
    min_touches: int = 3
    tolerance: float = 0.02  # 2% tolerance
    volume_decrease_threshold: float = 0.7  # Volume should decrease to 70%
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on chart patterns"""
        signals = pd.Series(0, index=data.index)
        
        # Scan for patterns at each point
        for i in range(self.max_pattern_length, len(data)):
            # Try different pattern lengths
            for length in range(self.min_pattern_length, 
                              min(self.max_pattern_length, i)):
                
                window = data.iloc[i-length:i+1]
                
                # Check for triangle patterns
                triangle_type = self._detect_triangle(window)
                if triangle_type:
                    signal = self._triangle_breakout_signal(window, triangle_type)
                    if signal != 0:
                        signals.iloc[i] = signal
                        break
                
                # Check for channel patterns
                channel_type = self._detect_channel(window)
                if channel_type:
                    signal = self._channel_signal(window, channel_type)
                    if signal != 0:
                        signals.iloc[i] = signal
                        break
                
                # Check for flag/pennant
                if self._detect_flag(window, data.iloc[max(0, i-length-20):i-length]):
                    signals.iloc[i] = 1 if window['close'].iloc[-1] > window['close'].iloc[0] else -1
                    break
        
        return signals.astype(int)
    
    def _detect_triangle(self, window: pd.DataFrame) -> Optional[str]:
        """Detect triangle patterns"""
        highs = window['high'].values
        lows = window['low'].values
        
        # Find peaks and troughs
        peak_indices = argrelextrema(highs, np.greater, order=2)[0]
        trough_indices = argrelextrema(lows, np.less, order=2)[0]
        
        if len(peak_indices) < 2 or len(trough_indices) < 2:
            return None
        
        # Fit trend lines
        x_peaks = peak_indices
        y_peaks = highs[peak_indices]
        x_troughs = trough_indices
        y_troughs = lows[trough_indices]
        
        # Calculate slopes
        if len(x_peaks) >= 2:
            slope_resistance, _, _, _, _ = linregress(x_peaks, y_peaks)
        else:
            return None
            
        if len(x_troughs) >= 2:
            slope_support, _, _, _, _ = linregress(x_troughs, y_troughs)
        else:
            return None
        
        # Classify triangle type
        if abs(slope_resistance) < 0.0001 and slope_support > 0.0001:
            return 'ascending'
        elif slope_resistance < -0.0001 and abs(slope_support) < 0.0001:
            return 'descending'
        elif slope_resistance < -0.0001 and slope_support > 0.0001:
            return 'symmetrical'
        
        return None
    
    def _triangle_breakout_signal(self, window: pd.DataFrame, triangle_type: str) -> int:
        """Generate signal based on triangle breakout"""
        last_price = window['close'].iloc[-1]
        
        # Check for volume pattern (should decrease during formation)
        if 'volume' in window.columns:
            vol_start = window['volume'].iloc[:10].mean()
            vol_end = window['volume'].iloc[-10:-1].mean()
            
            if vol_end > vol_start * self.volume_decrease_threshold:
                return 0  # Volume pattern doesn't confirm
        
        if triangle_type == 'ascending':
            # Bullish bias
            resistance = window['high'].iloc[-20:].max()
            if last_price > resistance * (1 + self.tolerance):
                return 1
                
        elif triangle_type == 'descending':
            # Bearish bias
            support = window['low'].iloc[-20:].min()
            if last_price < support * (1 - self.tolerance):
                return -1
                
        elif triangle_type == 'symmetrical':
            # Direction of breakout determines signal
            mid_point = (window['high'].mean() + window['low'].mean()) / 2
            if last_price > window['high'].iloc[-5:].max():
                return 1
            elif last_price < window['low'].iloc[-5:].min():
                return -1
        
        return 0
    
    def _detect_channel(self, window: pd.DataFrame) -> Optional[str]:
        """Detect channel patterns"""
        highs = window['high'].values
        lows = window['low'].values
        
        # Fit parallel lines
        x = np.arange(len(window))
        
        # Fit resistance line
        slope_high, intercept_high, r_high, _, _ = linregress(x, highs)
        
        # Fit support line
        slope_low, intercept_low, r_low, _, _ = linregress(x, lows)
        
        # Check if lines are roughly parallel
        slope_diff = abs(slope_high - slope_low)
        avg_slope = (abs(slope_high) + abs(slope_low)) / 2
        
        if avg_slope > 0:
            relative_diff = slope_diff / avg_slope
        else:
            relative_diff = 0
        
        if relative_diff < 0.3 and r_high**2 > 0.7 and r_low**2 > 0.7:
            if slope_high > 0.0001:
                return 'ascending_channel'
            elif slope_high < -0.0001:
                return 'descending_channel'
            else:
                return 'horizontal_channel'
        
        return None
    
    def _channel_signal(self, window: pd.DataFrame, channel_type: str) -> int:
        """Generate signal based on channel pattern"""
        last_price = window['close'].iloc[-1]
        channel_high = window['high'].iloc[-10:].mean()
        channel_low = window['low'].iloc[-10:].mean()
        
        if channel_type == 'ascending_channel':
            # Trade with the trend, buy at support
            if last_price <= channel_low * (1 + self.tolerance):
                return 1
        elif channel_type == 'descending_channel':
            # Trade with the trend, sell at resistance
            if last_price >= channel_high * (1 - self.tolerance):
                return -1
        else:  # horizontal_channel
            # Trade the range
            if last_price <= channel_low * (1 + self.tolerance):
                return 1
            elif last_price >= channel_high * (1 - self.tolerance):
                return -1
        
        return 0
    
    def _detect_flag(self, flag_window: pd.DataFrame, pole_window: pd.DataFrame) -> bool:
        """Detect flag/pennant pattern"""
        if len(pole_window) < 10:
            return False
        
        # Check for strong move (pole)
        pole_return = (pole_window['close'].iloc[-1] - pole_window['close'].iloc[0]) / pole_window['close'].iloc[0]
        
        if abs(pole_return) < 0.05:  # Need at least 5% move
            return False
        
        # Check for consolidation (flag)
        flag_range = flag_window['high'].max() - flag_window['low'].min()
        pole_range = pole_window['high'].max() - pole_window['low'].min()
        
        # Flag should be tight consolidation
        if flag_range > pole_range * 0.5:
            return False
        
        # Volume should decrease during flag
        if 'volume' in flag_window.columns:
            flag_volume = flag_window['volume'].mean()
            pole_volume = pole_window['volume'].mean()
            
            if flag_volume > pole_volume * 0.7:
                return False
        
        return True
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'min_pattern_length': self.min_pattern_length,
            'max_pattern_length': self.max_pattern_length,
            'min_touches': self.min_touches,
            'tolerance': self.tolerance,
            'volume_decrease_threshold': self.volume_decrease_threshold
        }


@dataclass
class BreakoutDetectionModel:
    """Consolidation breakout detection model"""
    
    consolidation_period: int = 20
    min_consolidation_ratio: float = 0.7  # Range must shrink to 70% of initial
    breakout_multiplier: float = 1.5  # Volume must be 1.5x average
    atr_multiplier: float = 1.2  # Breakout must be 1.2x ATR
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on consolidation breakouts"""
        signals = pd.Series(0, index=data.index)
        
        # Calculate ATR
        atr = self._calculate_atr(data, 14)
        
        # Detect consolidation and breakouts
        for i in range(self.consolidation_period, len(data)):
            window = data.iloc[i-self.consolidation_period:i]
            
            if self._is_consolidating(window):
                # Check for breakout
                current = data.iloc[i]
                prev_high = window['high'].max()
                prev_low = window['low'].min()
                
                # Upward breakout
                if (current['close'] > prev_high and 
                    current['close'] - prev_high > atr.iloc[i] * self.atr_multiplier):
                    if self._confirm_with_volume(data, i):
                        signals.iloc[i] = 1
                
                # Downward breakout
                elif (current['close'] < prev_low and 
                      prev_low - current['close'] > atr.iloc[i] * self.atr_multiplier):
                    if self._confirm_with_volume(data, i):
                        signals.iloc[i] = -1
        
        return signals.astype(int)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _is_consolidating(self, window: pd.DataFrame) -> bool:
        """Check if price is consolidating"""
        # Calculate range contraction
        first_half = window.iloc[:len(window)//2]
        second_half = window.iloc[len(window)//2:]
        
        first_range = first_half['high'].max() - first_half['low'].min()
        second_range = second_half['high'].max() - second_half['low'].min()
        
        if first_range == 0:
            return False
        
        # Range should be contracting
        range_ratio = second_range / first_range
        
        return range_ratio < self.min_consolidation_ratio
    
    def _confirm_with_volume(self, data: pd.DataFrame, idx: int) -> bool:
        """Confirm breakout with volume"""
        if 'volume' not in data.columns:
            return True
        
        current_volume = data['volume'].iloc[idx]
        avg_volume = data['volume'].iloc[idx-20:idx].mean()
        
        return current_volume > avg_volume * self.breakout_multiplier
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'consolidation_period': self.consolidation_period,
            'min_consolidation_ratio': self.min_consolidation_ratio,
            'breakout_multiplier': self.breakout_multiplier,
            'atr_multiplier': self.atr_multiplier
        }


@dataclass
class HighTightFlagModel:
    """High tight flag pattern detection"""
    
    min_advance: float = 0.90  # 90% minimum advance
    max_advance_days: int = 8  # Maximum 2 months for advance
    max_flag_depth: float = 0.25  # Maximum 25% pullback
    min_flag_days: int = 3
    max_flag_days: int = 25
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on high tight flag pattern"""
        signals = pd.Series(0, index=data.index)
        
        for i in range(self.max_advance_days + self.max_flag_days, len(data)):
            # Look for the pattern
            pattern_start = max(0, i - self.max_advance_days - self.max_flag_days)
            
            # Find the strongest advance in the window
            best_advance_start = None
            best_advance_end = None
            best_advance_return = 0
            
            for start in range(pattern_start, i - self.min_flag_days):
                for end in range(start + 5, min(start + self.max_advance_days, i - self.min_flag_days)):
                    advance_return = (data['close'].iloc[end] - data['close'].iloc[start]) / data['close'].iloc[start]
                    
                    if advance_return > best_advance_return and advance_return >= self.min_advance:
                        best_advance_start = start
                        best_advance_end = end
                        best_advance_return = advance_return
            
            # Check if we found a qualifying advance
            if best_advance_start is not None:
                # Check for flag formation
                flag_start = best_advance_end
                flag_end = i
                
                if flag_end - flag_start >= self.min_flag_days:
                    flag_high = data['high'].iloc[flag_start:flag_end].max()
                    flag_low = data['low'].iloc[flag_start:flag_end].min()
                    advance_high = data['high'].iloc[best_advance_start:best_advance_end+1].max()
                    
                    # Calculate pullback
                    pullback = (advance_high - flag_low) / (advance_high - data['low'].iloc[best_advance_start])
                    
                    # Check if it qualifies as a tight flag
                    if pullback <= self.max_flag_depth:
                        # Check for breakout
                        if data['close'].iloc[i] > flag_high:
                            signals.iloc[i] = 1
        
        return signals.astype(int)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'min_advance': self.min_advance,
            'max_advance_days': self.max_advance_days,
            'max_flag_depth': self.max_flag_depth,
            'min_flag_days': self.min_flag_days,
            'max_flag_days': self.max_flag_days
        }


@dataclass
class PatternRecognitionEnsemble:
    """Ensemble of all pattern recognition models"""
    
    def __init__(self):
        self.models = {
            'candlestick': CandlestickPatternsModel(),
            'support_resistance': SupportResistanceModel(),
            'chart_patterns': ChartPatternRecognitionModel(),
            'breakout': BreakoutDetectionModel(),
            'high_tight_flag': HighTightFlagModel()
        }
        
        self.weights = {
            'candlestick': 0.20,
            'support_resistance': 0.25,
            'chart_patterns': 0.20,
            'breakout': 0.20,
            'high_tight_flag': 0.15
        }
    
    def predict(self, data: pd.DataFrame, voting_threshold: float = 0.3) -> pd.Series:
        """Generate ensemble signals from all pattern models"""
        all_signals = {}
        
        # Get signals from each model
        for name, model in self.models.items():
            try:
                signals = model.predict(data)
                all_signals[name] = signals
            except Exception as e:
                logger.warning(f"Pattern model {name} failed: {e}")
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
                logger.error(f"Pattern model {name} failed: {e}")
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