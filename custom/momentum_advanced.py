"""
Advanced Momentum Models
========================
Implements sophisticated momentum indicators and strategies for
identifying trend strength, momentum shifts, and divergences.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MomentumSignal:
    """Momentum indicator signal"""
    indicator_name: str
    signal_type: str  # buy/sell/neutral
    strength: float  # 0-1 scale
    value: float  # Current indicator value
    threshold: float  # Signal threshold
    divergence: Optional[str]  # bullish/bearish/none
    timestamp: datetime
    additional_data: Dict[str, Any] = None


class BaseMomentumModel(ABC):
    """Base class for momentum models"""
    
    def __init__(self, lookback_period: int = 14):
        self.lookback_period = lookback_period
        self.signals_history = []
        
    @abstractmethod
    def calculate_momentum(self, data: pd.DataFrame, **kwargs) -> MomentumSignal:
        """Calculate momentum indicator and generate signal"""
        pass
    
    def validate_data(self, data: pd.DataFrame, required_periods: int = None) -> bool:
        """Validate input data"""
        required = required_periods or self.lookback_period
        if data is None or data.empty:
            return False
        if len(data) < required:
            return False
        return True
    
    def detect_divergence(self, price: pd.Series, indicator: pd.Series, 
                         window: int = 10) -> Optional[str]:
        """Detect divergence between price and indicator"""
        if len(price) < window * 2 or len(indicator) < window * 2:
            return None
        
        # Find recent peaks and troughs
        price_highs = price.rolling(window=window).max()
        price_lows = price.rolling(window=window).min()
        ind_highs = indicator.rolling(window=window).max()
        ind_lows = indicator.rolling(window=window).min()
        
        # Check for divergence at recent points
        recent_price = price.tail(window)
        recent_ind = indicator.tail(window)
        
        # Bearish divergence: price makes higher high, indicator makes lower high
        if (recent_price.iloc[-1] > recent_price.iloc[0] and 
            recent_ind.iloc[-1] < recent_ind.iloc[0] and
            recent_price.iloc[-1] == price_highs.iloc[-1]):
            return 'bearish'
        
        # Bullish divergence: price makes lower low, indicator makes higher low
        if (recent_price.iloc[-1] < recent_price.iloc[0] and 
            recent_ind.iloc[-1] > recent_ind.iloc[0] and
            recent_price.iloc[-1] == price_lows.iloc[-1]):
            return 'bullish'
        
        return None


class RateOfChangeModel(BaseMomentumModel):
    """
    Momentum measurement:
    - Multiple timeframes
    - Divergence detection
    - Smoothing options
    - Signal generation
    """
    
    def __init__(self, period: int = 12, smooth_period: int = 3,
                 use_percentage: bool = True):
        super().__init__(lookback_period=period)
        self.period = period
        self.smooth_period = smooth_period
        self.use_percentage = use_percentage
        self.overbought = 10  # 10% for percentage ROC
        self.oversold = -10
        
    def calculate_momentum(self, data: pd.DataFrame, **kwargs) -> MomentumSignal:
        """Calculate Rate of Change indicator"""
        if not self.validate_data(data, self.period + self.smooth_period):
            return self._default_signal()
        
        close = data['close']
        
        # Calculate ROC
        if self.use_percentage:
            roc = ((close - close.shift(self.period)) / close.shift(self.period)) * 100
        else:
            roc = close - close.shift(self.period)
        
        # Apply smoothing if specified
        if self.smooth_period > 1:
            roc_smooth = roc.rolling(window=self.smooth_period).mean()
        else:
            roc_smooth = roc
        
        # Get current values
        current_roc = roc_smooth.iloc[-1]
        
        # Multi-timeframe analysis
        roc_short = self._calculate_roc(close, self.period // 2)
        roc_long = self._calculate_roc(close, self.period * 2)
        
        # Detect divergence
        divergence = self.detect_divergence(close.tail(20), roc_smooth.tail(20))
        
        # Generate signal
        signal_type = self._generate_signal(current_roc, roc_short, roc_long)
        strength = self._calculate_strength(current_roc, roc_smooth)
        
        return MomentumSignal(
            indicator_name='Rate of Change',
            signal_type=signal_type,
            strength=strength,
            value=current_roc,
            threshold=self.overbought if current_roc > 0 else self.oversold,
            divergence=divergence,
            timestamp=datetime.now(),
            additional_data={
                'roc_short': roc_short,
                'roc_long': roc_long,
                'smoothed': self.smooth_period > 1,
                'period': self.period,
                'histogram': roc_smooth.tail(50).tolist()
            }
        )
    
    def _calculate_roc(self, close: pd.Series, period: int) -> float:
        """Calculate ROC for a specific period"""
        if len(close) < period:
            return 0
        
        if self.use_percentage:
            return ((close.iloc[-1] - close.iloc[-period-1]) / close.iloc[-period-1]) * 100
        else:
            return close.iloc[-1] - close.iloc[-period-1]
    
    def _generate_signal(self, current: float, short: float, long: float) -> str:
        """Generate trading signal based on ROC values"""
        # Strong buy: oversold and turning up across timeframes
        if current < self.oversold and short > current and long < 0:
            return 'buy'
        
        # Strong sell: overbought and turning down
        elif current > self.overbought and short < current and long > 0:
            return 'sell'
        
        # Moderate buy: positive momentum building
        elif current > 0 and short > long and current > short:
            return 'buy'
        
        # Moderate sell: negative momentum building
        elif current < 0 and short < long and current < short:
            return 'sell'
        
        else:
            return 'neutral'
    
    def _calculate_strength(self, current: float, roc_series: pd.Series) -> float:
        """Calculate signal strength"""
        # Normalize current value
        recent_max = roc_series.tail(50).max()
        recent_min = roc_series.tail(50).min()
        
        if recent_max == recent_min:
            return 0.5
        
        normalized = (current - recent_min) / (recent_max - recent_min)
        
        # Adjust for extremes
        if abs(current) > abs(self.overbought * 1.5):
            strength = 0.9
        elif abs(current) > abs(self.overbought):
            strength = 0.7
        else:
            strength = 0.5 + (normalized - 0.5) * 0.4
        
        return max(0, min(1, strength))
    
    def _default_signal(self) -> MomentumSignal:
        """Return default signal when calculation fails"""
        return MomentumSignal(
            indicator_name='Rate of Change',
            signal_type='neutral',
            strength=0,
            value=0,
            threshold=0,
            divergence=None,
            timestamp=datetime.now()
        )


class ChandeMomentumModel(BaseMomentumModel):
    """
    Advanced momentum:
    - True momentum
    - Overbought/oversold
    - Trend strength
    - Divergences
    """
    
    def __init__(self, period: int = 20):
        super().__init__(lookback_period=period)
        self.period = period
        self.overbought = 50
        self.oversold = -50
        
    def calculate_momentum(self, data: pd.DataFrame, **kwargs) -> MomentumSignal:
        """Calculate Chande Momentum Oscillator"""
        if not self.validate_data(data, self.period + 1):
            return self._default_signal()
        
        close = data['close']
        
        # Calculate price changes
        changes = close.diff()
        
        # Separate gains and losses
        gains = changes.where(changes > 0, 0)
        losses = -changes.where(changes < 0, 0)
        
        # Calculate sums over period
        sum_gains = gains.rolling(window=self.period).sum()
        sum_losses = losses.rolling(window=self.period).sum()
        
        # Calculate CMO
        cmo = ((sum_gains - sum_losses) / (sum_gains + sum_losses)) * 100
        
        # Get current value
        current_cmo = cmo.iloc[-1]
        
        # Calculate trend strength
        cmo_ma = cmo.rolling(window=10).mean()
        trend_strength = self._calculate_trend_strength(cmo, cmo_ma)
        
        # Detect divergence
        divergence = self.detect_divergence(close.tail(30), cmo.tail(30))
        
        # Generate signal
        signal_type = self._generate_signal(current_cmo, cmo, cmo_ma)
        
        # Calculate signal strength
        strength = abs(current_cmo) / 100
        
        return MomentumSignal(
            indicator_name='Chande Momentum Oscillator',
            signal_type=signal_type,
            strength=strength,
            value=current_cmo,
            threshold=self.overbought if current_cmo > 0 else self.oversold,
            divergence=divergence,
            timestamp=datetime.now(),
            additional_data={
                'trend_strength': trend_strength,
                'cmo_ma': cmo_ma.iloc[-1] if not cmo_ma.empty else 0,
                'sum_gains': sum_gains.iloc[-1],
                'sum_losses': sum_losses.iloc[-1],
                'histogram': cmo.tail(50).tolist()
            }
        )
    
    def _calculate_trend_strength(self, cmo: pd.Series, cmo_ma: pd.Series) -> float:
        """Calculate trend strength based on CMO behavior"""
        if len(cmo) < 20:
            return 0.5
        
        # Check if CMO is consistently above/below MA
        recent_cmo = cmo.tail(10)
        recent_ma = cmo_ma.tail(10)
        
        above_ma = (recent_cmo > recent_ma).sum()
        
        # Strong trend if consistently on one side
        if above_ma >= 8:
            return 0.8
        elif above_ma <= 2:
            return 0.8
        else:
            return 0.4
    
    def _generate_signal(self, current: float, cmo: pd.Series, cmo_ma: pd.Series) -> str:
        """Generate trading signal"""
        if len(cmo) < 5:
            return 'neutral'
        
        # Recent values
        prev_cmo = cmo.iloc[-2]
        ma_current = cmo_ma.iloc[-1]
        
        # Oversold bounce
        if current < self.oversold and current > prev_cmo:
            return 'buy'
        
        # Overbought reversal
        elif current > self.overbought and current < prev_cmo:
            return 'sell'
        
        # Trend following
        elif current > 0 and current > ma_current and prev_cmo < ma_current:
            return 'buy'
        
        elif current < 0 and current < ma_current and prev_cmo > ma_current:
            return 'sell'
        
        else:
            return 'neutral'
    
    def _default_signal(self) -> MomentumSignal:
        """Return default signal when calculation fails"""
        return MomentumSignal(
            indicator_name='Chande Momentum Oscillator',
            signal_type='neutral',
            strength=0,
            value=0,
            threshold=0,
            divergence=None,
            timestamp=datetime.now()
        )


class TRIXModel(BaseMomentumModel):
    """
    Triple smoothed:
    - Trend filtering
    - Signal crosses
    - Zero-line breaks
    - Momentum shifts
    """
    
    def __init__(self, period: int = 14, signal_period: int = 9):
        super().__init__(lookback_period=period * 3)  # Need more data for triple smoothing
        self.period = period
        self.signal_period = signal_period
        
    def calculate_momentum(self, data: pd.DataFrame, **kwargs) -> MomentumSignal:
        """Calculate TRIX indicator"""
        if not self.validate_data(data, self.period * 4):
            return self._default_signal()
        
        close = data['close']
        
        # First EMA
        ema1 = close.ewm(span=self.period, adjust=False).mean()
        
        # Second EMA (EMA of EMA)
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        
        # Third EMA (EMA of EMA of EMA)
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()
        
        # TRIX = 1-period percent change of ema3
        trix = (ema3.pct_change() * 10000)  # Multiply by 10000 for readability
        
        # Signal line (EMA of TRIX)
        signal = trix.ewm(span=self.signal_period, adjust=False).mean()
        
        # Current values
        current_trix = trix.iloc[-1]
        current_signal = signal.iloc[-1]
        
        # Detect crossovers
        crossover = self._detect_crossover(trix, signal)
        
        # Detect zero-line breaks
        zero_break = self._detect_zero_break(trix)
        
        # Momentum shift
        momentum_shift = self._detect_momentum_shift(trix)
        
        # Generate signal
        signal_type = self._generate_signal(current_trix, current_signal, 
                                          crossover, zero_break)
        
        # Calculate strength
        strength = self._calculate_strength(trix, signal)
        
        return MomentumSignal(
            indicator_name='TRIX',
            signal_type=signal_type,
            strength=strength,
            value=current_trix,
            threshold=current_signal,
            divergence=self.detect_divergence(close.tail(30), trix.tail(30)),
            timestamp=datetime.now(),
            additional_data={
                'signal_line': current_signal,
                'crossover': crossover,
                'zero_break': zero_break,
                'momentum_shift': momentum_shift,
                'histogram': (trix - signal).tail(50).tolist()
            }
        )
    
    def _detect_crossover(self, trix: pd.Series, signal: pd.Series) -> Optional[str]:
        """Detect TRIX/Signal crossovers"""
        if len(trix) < 2:
            return None
        
        current_diff = trix.iloc[-1] - signal.iloc[-1]
        prev_diff = trix.iloc[-2] - signal.iloc[-2]
        
        if prev_diff <= 0 and current_diff > 0:
            return 'bullish_cross'
        elif prev_diff >= 0 and current_diff < 0:
            return 'bearish_cross'
        
        return None
    
    def _detect_zero_break(self, trix: pd.Series) -> Optional[str]:
        """Detect zero-line breaks"""
        if len(trix) < 2:
            return None
        
        current = trix.iloc[-1]
        previous = trix.iloc[-2]
        
        if previous <= 0 and current > 0:
            return 'bullish_break'
        elif previous >= 0 and current < 0:
            return 'bearish_break'
        
        return None
    
    def _detect_momentum_shift(self, trix: pd.Series) -> str:
        """Detect momentum shifts"""
        if len(trix) < 10:
            return 'neutral'
        
        # Calculate rate of change of TRIX
        trix_roc = trix.diff()
        recent_roc = trix_roc.tail(5).mean()
        
        if recent_roc > 0 and trix.iloc[-1] > trix.iloc[-5]:
            return 'accelerating'
        elif recent_roc < 0 and trix.iloc[-1] < trix.iloc[-5]:
            return 'decelerating'
        else:
            return 'stable'
    
    def _generate_signal(self, trix: float, signal: float, 
                        crossover: Optional[str], zero_break: Optional[str]) -> str:
        """Generate trading signal"""
        # Strong signals from crossovers
        if crossover == 'bullish_cross' and trix < 0:
            return 'buy'
        elif crossover == 'bearish_cross' and trix > 0:
            return 'sell'
        
        # Zero-line breaks
        elif zero_break == 'bullish_break':
            return 'buy'
        elif zero_break == 'bearish_break':
            return 'sell'
        
        # Trend following
        elif trix > signal and trix > 0:
            return 'buy'
        elif trix < signal and trix < 0:
            return 'sell'
        
        else:
            return 'neutral'
    
    def _calculate_strength(self, trix: pd.Series, signal: pd.Series) -> float:
        """Calculate signal strength"""
        if len(trix) < 20:
            return 0.5
        
        current_trix = trix.iloc[-1]
        current_signal = signal.iloc[-1]
        
        # Distance from signal line
        distance = abs(current_trix - current_signal)
        avg_distance = abs(trix - signal).tail(20).mean()
        
        if avg_distance == 0:
            return 0.5
        
        relative_distance = distance / avg_distance
        
        # Combine with absolute value
        abs_strength = min(abs(current_trix) / 50, 1)  # Normalize to typical range
        
        strength = (relative_distance * 0.5 + abs_strength * 0.5)
        
        return max(0, min(1, strength))
    
    def _default_signal(self) -> MomentumSignal:
        """Return default signal when calculation fails"""
        return MomentumSignal(
            indicator_name='TRIX',
            signal_type='neutral',
            strength=0,
            value=0,
            threshold=0,
            divergence=None,
            timestamp=datetime.now()
        )


class KnowSureThingModel(BaseMomentumModel):
    """
    Multi-ROC blend:
    - 4 timeframes
    - Weighted average
    - Signal line
    - Major trends
    """
    
    def __init__(self):
        # KST uses specific ROC periods and weights
        self.roc_periods = [10, 15, 20, 30]
        self.sma_periods = [10, 10, 10, 15]
        self.weights = [1, 2, 3, 4]
        self.signal_period = 9
        super().__init__(lookback_period=max(self.roc_periods) + max(self.sma_periods))
        
    def calculate_momentum(self, data: pd.DataFrame, **kwargs) -> MomentumSignal:
        """Calculate Know Sure Thing (KST) indicator"""
        if not self.validate_data(data, self.lookback_period + self.signal_period):
            return self._default_signal()
        
        close = data['close']
        
        # Calculate ROCs and their SMAs
        roc_smas = []
        for roc_period, sma_period in zip(self.roc_periods, self.sma_periods):
            # Calculate ROC
            roc = ((close - close.shift(roc_period)) / close.shift(roc_period)) * 100
            # Smooth with SMA
            roc_sma = roc.rolling(window=sma_period).mean()
            roc_smas.append(roc_sma)
        
        # Calculate weighted KST
        kst = sum(roc_sma * weight for roc_sma, weight in zip(roc_smas, self.weights))
        
        # Signal line
        signal = kst.rolling(window=self.signal_period).mean()
        
        # Current values
        current_kst = kst.iloc[-1]
        current_signal = signal.iloc[-1]
        
        # Detect major trend
        major_trend = self._detect_major_trend(kst, signal)
        
        # Detect crossovers
        crossover = self._detect_crossover(kst, signal)
        
        # Calculate momentum strength
        momentum_strength = self._calculate_momentum_strength(kst)
        
        # Generate signal
        signal_type = self._generate_signal(current_kst, current_signal, 
                                          crossover, major_trend)
        
        # Calculate overall strength
        strength = self._calculate_strength(kst, signal, momentum_strength)
        
        return MomentumSignal(
            indicator_name='Know Sure Thing',
            signal_type=signal_type,
            strength=strength,
            value=current_kst,
            threshold=current_signal,
            divergence=self.detect_divergence(close.tail(40), kst.tail(40)),
            timestamp=datetime.now(),
            additional_data={
                'signal_line': current_signal,
                'major_trend': major_trend,
                'crossover': crossover,
                'momentum_strength': momentum_strength,
                'roc_values': [float(roc_sma.iloc[-1]) for roc_sma in roc_smas],
                'histogram': (kst - signal).tail(50).tolist()
            }
        )
    
    def _detect_major_trend(self, kst: pd.Series, signal: pd.Series) -> str:
        """Detect major trend direction"""
        if len(kst) < 50:
            return 'neutral'
        
        # Long-term KST average
        kst_lt = kst.rolling(window=50).mean()
        
        current_kst = kst.iloc[-1]
        current_signal = signal.iloc[-1]
        lt_kst = kst_lt.iloc[-1]
        
        if current_kst > current_signal and current_kst > lt_kst and lt_kst > 0:
            return 'strong_bullish'
        elif current_kst > current_signal and current_kst > 0:
            return 'bullish'
        elif current_kst < current_signal and current_kst < lt_kst and lt_kst < 0:
            return 'strong_bearish'
        elif current_kst < current_signal and current_kst < 0:
            return 'bearish'
        else:
            return 'neutral'
    
    def _detect_crossover(self, kst: pd.Series, signal: pd.Series) -> Optional[str]:
        """Detect KST/Signal crossovers"""
        if len(kst) < 2:
            return None
        
        current_diff = kst.iloc[-1] - signal.iloc[-1]
        prev_diff = kst.iloc[-2] - signal.iloc[-2]
        
        if prev_diff <= 0 and current_diff > 0:
            return 'bullish_cross'
        elif prev_diff >= 0 and current_diff < 0:
            return 'bearish_cross'
        
        return None
    
    def _calculate_momentum_strength(self, kst: pd.Series) -> float:
        """Calculate momentum strength"""
        if len(kst) < 20:
            return 0.5
        
        # Rate of change of KST
        kst_change = kst.diff()
        recent_change = kst_change.tail(10).mean()
        
        # Normalize to 0-1 scale
        max_change = kst_change.tail(50).std() * 2
        if max_change == 0:
            return 0.5
        
        strength = abs(recent_change) / max_change
        return max(0, min(1, strength))
    
    def _generate_signal(self, kst: float, signal: float, 
                        crossover: Optional[str], trend: str) -> str:
        """Generate trading signal"""
        # Strong signals from crossovers in trend direction
        if crossover == 'bullish_cross' and trend in ['bullish', 'strong_bullish']:
            return 'buy'
        elif crossover == 'bearish_cross' and trend in ['bearish', 'strong_bearish']:
            return 'sell'
        
        # Trend following
        elif trend == 'strong_bullish':
            return 'buy'
        elif trend == 'strong_bearish':
            return 'sell'
        
        # Moderate signals
        elif kst > signal and kst > 0:
            return 'buy'
        elif kst < signal and kst < 0:
            return 'sell'
        
        else:
            return 'neutral'
    
    def _calculate_strength(self, kst: pd.Series, signal: pd.Series, 
                          momentum_strength: float) -> float:
        """Calculate overall signal strength"""
        if len(kst) < 20:
            return momentum_strength
        
        current_kst = kst.iloc[-1]
        current_signal = signal.iloc[-1]
        
        # Distance from signal
        distance = abs(current_kst - current_signal)
        avg_distance = abs(kst - signal).tail(20).mean()
        
        if avg_distance == 0:
            distance_strength = 0.5
        else:
            distance_strength = min(distance / avg_distance, 1)
        
        # Combine factors
        strength = (distance_strength * 0.4 + momentum_strength * 0.6)
        
        return max(0, min(1, strength))
    
    def _default_signal(self) -> MomentumSignal:
        """Return default signal when calculation fails"""
        return MomentumSignal(
            indicator_name='Know Sure Thing',
            signal_type='neutral',
            strength=0,
            value=0,
            threshold=0,
            divergence=None,
            timestamp=datetime.now()
        )


class UltimateOscillatorModel(BaseMomentumModel):
    """
    Multi-period momentum:
    - 3 timeframes
    - Weighted formula
    - Divergence signals
    - Extreme readings
    """
    
    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28):
        self.periods = [period1, period2, period3]
        self.weights = [4, 2, 1]  # Standard UO weights
        super().__init__(lookback_period=max(self.periods))
        self.overbought = 70
        self.oversold = 30
        
    def calculate_momentum(self, data: pd.DataFrame, **kwargs) -> MomentumSignal:
        """Calculate Ultimate Oscillator"""
        if not self.validate_data(data, self.lookback_period + 1):
            return self._default_signal()
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Buying Pressure (BP)
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        
        # Calculate True Range (TR)
        high_close = pd.concat([high, close.shift(1)], axis=1).max(axis=1)
        low_close = pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = high_close - low_close
        
        # Calculate averages for each period
        averages = []
        for period in self.periods:
            bp_sum = bp.rolling(window=period).sum()
            tr_sum = tr.rolling(window=period).sum()
            
            # Avoid division by zero
            avg = bp_sum / tr_sum.where(tr_sum != 0, 1)
            averages.append(avg)
        
        # Calculate weighted Ultimate Oscillator
        weighted_sum = sum(avg * weight for avg, weight in zip(averages, self.weights))
        weight_total = sum(self.weights)
        uo = (weighted_sum / weight_total) * 100
        
        # Current value
        current_uo = uo.iloc[-1]
        
        # Detect extremes
        extreme_reading = self._detect_extreme_reading(uo)
        
        # Detect divergence
        divergence = self.detect_divergence(close.tail(30), uo.tail(30))
        
        # Multi-timeframe analysis
        timeframe_signals = self._analyze_timeframes(bp, tr, close)
        
        # Generate signal
        signal_type = self._generate_signal(current_uo, uo, divergence, extreme_reading)
        
        # Calculate strength
        strength = self._calculate_strength(current_uo, uo, divergence)
        
        return MomentumSignal(
            indicator_name='Ultimate Oscillator',
            signal_type=signal_type,
            strength=strength,
            value=current_uo,
            threshold=self.overbought if current_uo > 50 else self.oversold,
            divergence=divergence,
            timestamp=datetime.now(),
            additional_data={
                'extreme_reading': extreme_reading,
                'period_values': [float(avg.iloc[-1]) * 100 for avg in averages],
                'timeframe_signals': timeframe_signals,
                'buying_pressure': float(bp.iloc[-1]),
                'true_range': float(tr.iloc[-1]),
                'histogram': uo.tail(50).tolist()
            }
        )
    
    def _detect_extreme_reading(self, uo: pd.Series) -> Optional[str]:
        """Detect extreme overbought/oversold readings"""
        if len(uo) < 5:
            return None
        
        current = uo.iloc[-1]
        recent_min = uo.tail(20).min()
        recent_max = uo.tail(20).max()
        
        if current > self.overbought and current >= recent_max * 0.95:
            return 'extreme_overbought'
        elif current < self.oversold and current <= recent_min * 1.05:
            return 'extreme_oversold'
        elif current > self.overbought:
            return 'overbought'
        elif current < self.oversold:
            return 'oversold'
        
        return None
    
    def _analyze_timeframes(self, bp: pd.Series, tr: pd.Series, close: pd.Series) -> Dict:
        """Analyze momentum across different timeframes"""
        signals = {}
        
        for i, period in enumerate(self.periods):
            if len(bp) < period:
                signals[f'period_{period}'] = 'neutral'
                continue
            
            # Calculate momentum for this timeframe
            bp_avg = bp.rolling(window=period).mean()
            tr_avg = tr.rolling(window=period).mean()
            
            if tr_avg.iloc[-1] == 0:
                momentum = 0
            else:
                momentum = bp_avg.iloc[-1] / tr_avg.iloc[-1]
            
            # Trend direction for this timeframe
            close_ma = close.rolling(window=period).mean()
            if close.iloc[-1] > close_ma.iloc[-1] and momentum > 0.5:
                signals[f'period_{period}'] = 'bullish'
            elif close.iloc[-1] < close_ma.iloc[-1] and momentum < 0.5:
                signals[f'period_{period}'] = 'bearish'
            else:
                signals[f'period_{period}'] = 'neutral'
        
        return signals
    
    def _generate_signal(self, current: float, uo: pd.Series, 
                        divergence: Optional[str], extreme: Optional[str]) -> str:
        """Generate trading signal"""
        if len(uo) < 5:
            return 'neutral'
        
        prev_uo = uo.iloc[-2]
        
        # Divergence-based signals (strongest)
        if divergence == 'bullish' and current < self.oversold:
            return 'buy'
        elif divergence == 'bearish' and current > self.overbought:
            return 'sell'
        
        # Extreme readings with reversal
        elif extreme == 'extreme_oversold' and current > prev_uo:
            return 'buy'
        elif extreme == 'extreme_overbought' and current < prev_uo:
            return 'sell'
        
        # Standard overbought/oversold
        elif current < self.oversold and current > prev_uo:
            return 'buy'
        elif current > self.overbought and current < prev_uo:
            return 'sell'
        
        else:
            return 'neutral'
    
    def _calculate_strength(self, current: float, uo: pd.Series, 
                          divergence: Optional[str]) -> float:
        """Calculate signal strength"""
        # Base strength from current position
        if current > self.overbought:
            base_strength = (current - self.overbought) / (100 - self.overbought)
        elif current < self.oversold:
            base_strength = (self.oversold - current) / self.oversold
        else:
            base_strength = 0.3
        
        # Boost for divergence
        if divergence:
            base_strength = min(base_strength + 0.3, 1.0)
        
        # Adjust for momentum
        if len(uo) >= 5:
            momentum = uo.diff().tail(5).mean()
            if abs(momentum) > 2:  # Strong momentum
                base_strength = min(base_strength + 0.2, 1.0)
        
        return max(0, min(1, base_strength))
    
    def _default_signal(self) -> MomentumSignal:
        """Return default signal when calculation fails"""
        return MomentumSignal(
            indicator_name='Ultimate Oscillator',
            signal_type='neutral',
            strength=0,
            value=50,
            threshold=50,
            divergence=None,
            timestamp=datetime.now()
        )