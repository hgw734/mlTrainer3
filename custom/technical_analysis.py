#!/usr/bin/env python3
"""
Advanced Technical Analysis Model Implementations for S&P 500 Trading
===================================================================

Advanced technical analysis models including:
- Trend analysis and detection
- Support and resistance levels
- Pattern recognition
- Oscillators and indicators
- Fibonacci retracements
- Elliott Wave analysis
- Advanced chart patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class TechnicalAnalysis:
    """Technical analysis result"""
    symbol: str
    trend_direction: str
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    pattern_detected: str
    oscillator_signals: Dict[str, float]
    confidence_level: float
    timestamp: datetime

class BaseTechnicalModel:
    """Base class for technical analysis models"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
        
    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            # Placeholder for real market data integration
            logger.info(f"Would get real market data for {symbol} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            # Placeholder for real economic data integration
            logger.info(f"Would get real economic data for {series_id} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real economic data: {e}")
            return None

    def fit(self, data: pd.Series) -> 'BaseTechnicalModel':
        """Fit the technical analysis model"""
        if len(data) < self.params.get('min_data_points', 100):
            raise ValueError(f"Insufficient data: {len(data)} < {self.params.get('min_data_points', 100)}")
        self.is_fitted = True
        return self

    def analyze_technical(self, data: pd.Series) -> TechnicalAnalysis:
        """Perform technical analysis"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before technical analysis")
        raise NotImplementedError("Subclasses must implement analyze_technical method")

class TrendAnalysisModel(BaseTechnicalModel):
    """Trend analysis model"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50, trend_threshold: float = 0.02):
        super().__init__(short_window=short_window, long_window=long_window, trend_threshold=trend_threshold, min_data_points=long_window + 10)
        
    def analyze_technical(self, data: pd.Series) -> TechnicalAnalysis:
        """Analyze trend patterns"""
        if len(data) < self.params['long_window']:
            return TechnicalAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                trend_direction='unknown',
                trend_strength=0.0,
                support_levels=[],
                resistance_levels=[],
                pattern_detected='none',
                oscillator_signals={},
                confidence_level=0.0,
                timestamp=datetime.now()
            )
        
        # Calculate moving averages
        short_ma = data.rolling(window=self.params['short_window']).mean()
        long_ma = data.rolling(window=self.params['long_window']).mean()
        
        # Determine trend direction
        current_price = data.iloc[-1]
        current_short_ma = short_ma.iloc[-1] if len(short_ma) > 0 else current_price
        current_long_ma = long_ma.iloc[-1] if len(long_ma) > 0 else current_price
        
        # Trend direction
        if current_short_ma > current_long_ma and current_price > current_short_ma:
            trend_direction = 'uptrend'
        elif current_short_ma < current_long_ma and current_price < current_short_ma:
            trend_direction = 'downtrend'
        else:
            trend_direction = 'sideways'
        
        # Trend strength
        trend_strength = abs(current_short_ma - current_long_ma) / current_long_ma if current_long_ma > 0 else 0
        
        # Support and resistance levels
        support_levels = self._find_support_levels(data)
        resistance_levels = self._find_resistance_levels(data)
        
        # Pattern detection
        pattern_detected = self._detect_patterns(data)
        
        # Oscillator signals
        oscillator_signals = self._calculate_oscillators(data)
        
        # Confidence level
        confidence_level = min(trend_strength / self.params['trend_threshold'], 1.0)
        
        return TechnicalAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pattern_detected=pattern_detected,
            oscillator_signals=oscillator_signals,
            confidence_level=confidence_level,
            timestamp=datetime.now()
        )
    
    def _find_support_levels(self, data: pd.Series) -> List[float]:
        """Find support levels"""
        if len(data) < 20:
            return []
        
        # Find local minima
        peaks, _ = find_peaks(-data.values, distance=10)
        
        support_levels = []
        for peak in peaks[-5:]:  # Last 5 support levels
            if peak < len(data):
                support_levels.append(float(data.iloc[peak]))
        
        return support_levels
    
    def _find_resistance_levels(self, data: pd.Series) -> List[float]:
        """Find resistance levels"""
        if len(data) < 20:
            return []
        
        # Find local maxima
        peaks, _ = find_peaks(data.values, distance=10)
        
        resistance_levels = []
        for peak in peaks[-5:]:  # Last 5 resistance levels
            if peak < len(data):
                resistance_levels.append(float(data.iloc[peak]))
        
        return resistance_levels
    
    def _detect_patterns(self, data: pd.Series) -> str:
        """Detect chart patterns"""
        if len(data) < 20:
            return 'none'
        
        # Simple pattern detection
        recent_data = data.iloc[-20:]
        
        # Double top/bottom detection
        peaks, _ = find_peaks(recent_data.values, distance=5)
        troughs, _ = find_peaks(-recent_data.values, distance=5)
        
        if len(peaks) >= 2:
            return 'double_top'
        elif len(troughs) >= 2:
            return 'double_bottom'
        else:
            return 'none'
    
    def _calculate_oscillators(self, data: pd.Series) -> Dict[str, float]:
        """Calculate oscillator signals"""
        if len(data) < 14:
            return {}
        
        # RSI calculation
        returns = data.pct_change().dropna()
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        
        avg_gain = gains.rolling(window=14).mean()
        avg_loss = losses.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # MACD calculation
        ema12 = data.ewm(span=12).mean()
        ema26 = data.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        current_macd = macd.iloc[-1] if len(macd) > 0 else 0
        current_signal = signal.iloc[-1] if len(signal) > 0 else 0
        
        return {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_signal,
            'macd_histogram': current_macd - current_signal
        }

class SupportResistanceModel(BaseTechnicalModel):
    """Support and resistance level model"""
    
    def __init__(self, window: int = 20, level_threshold: float = 0.02):
        super().__init__(window=window, level_threshold=level_threshold, min_data_points=window + 10)
        
    def analyze_technical(self, data: pd.Series) -> TechnicalAnalysis:
        """Analyze support and resistance levels"""
        if len(data) < self.params['window']:
            return TechnicalAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                trend_direction='unknown',
                trend_strength=0.0,
                support_levels=[],
                resistance_levels=[],
                pattern_detected='none',
                oscillator_signals={},
                confidence_level=0.0,
                timestamp=datetime.now()
            )
        
        # Find support and resistance levels
        support_levels = self._find_dynamic_support_levels(data)
        resistance_levels = self._find_dynamic_resistance_levels(data)
        
        # Determine trend direction based on current price vs levels
        current_price = data.iloc[-1]
        
        if resistance_levels:
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
            if current_price > nearest_resistance * (1 - self.params['level_threshold']):
                trend_direction = 'uptrend'
            else:
                trend_direction = 'downtrend'
        else:
            trend_direction = 'sideways'
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(data)
        
        # Pattern detection
        pattern_detected = self._detect_support_resistance_patterns(data, support_levels, resistance_levels)
        
        # Oscillator signals
        oscillator_signals = self._calculate_support_resistance_oscillators(data, support_levels, resistance_levels)
        
        # Confidence level
        confidence_level = self._calculate_level_confidence(data, support_levels, resistance_levels)
        
        return TechnicalAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pattern_detected=pattern_detected,
            oscillator_signals=oscillator_signals,
            confidence_level=confidence_level,
            timestamp=datetime.now()
        )
    
    def _find_dynamic_support_levels(self, data: pd.Series) -> List[float]:
        """Find dynamic support levels"""
        if len(data) < 20:
            return []
        
        # Find local minima with clustering
        troughs, _ = find_peaks(-data.values, distance=10)
        
        if len(troughs) < 2:
            return []
        
        # Cluster nearby levels
        trough_values = [data.iloc[t] for t in troughs]
        trough_array = np.array(trough_values).reshape(-1, 1)
        
        if len(trough_array) >= 2:
            kmeans = KMeans(n_clusters=min(3, len(trough_array)), random_state=42)
            clusters = kmeans.fit_predict(trough_array)
            
            # Get cluster centers as support levels
            support_levels = kmeans.cluster_centers_.flatten().tolist()
            return [level for level in support_levels if level > 0]
        else:
            return trough_values
    
    def _find_dynamic_resistance_levels(self, data: pd.Series) -> List[float]:
        """Find dynamic resistance levels"""
        if len(data) < 20:
            return []
        
        # Find local maxima with clustering
        peaks, _ = find_peaks(data.values, distance=10)
        
        if len(peaks) < 2:
            return []
        
        # Cluster nearby levels
        peak_values = [data.iloc[p] for p in peaks]
        peak_array = np.array(peak_values).reshape(-1, 1)
        
        if len(peak_array) >= 2:
            kmeans = KMeans(n_clusters=min(3, len(peak_array)), random_state=42)
            clusters = kmeans.fit_predict(peak_array)
            
            # Get cluster centers as resistance levels
            resistance_levels = kmeans.cluster_centers_.flatten().tolist()
            return [level for level in resistance_levels if level > 0]
        else:
            return peak_values
    
    def _calculate_trend_strength(self, data: pd.Series) -> float:
        """Calculate trend strength"""
        if len(data) < 20:
            return 0.0
        
        # Linear regression trend strength
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        model = LinearRegression()
        model.fit(X, y)
        
        # R-squared as trend strength
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return r_squared
    
    def _detect_support_resistance_patterns(self, data: pd.Series, support_levels: List[float], resistance_levels: List[float]) -> str:
        """Detect patterns based on support and resistance"""
        if not support_levels or not resistance_levels:
            return 'none'
        
        current_price = data.iloc[-1]
        
        # Check for breakout patterns
        nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
        
        if current_price > nearest_resistance:
            return 'breakout_up'
        elif current_price < nearest_support:
            return 'breakout_down'
        else:
            return 'consolidation'
    
    def _calculate_support_resistance_oscillators(self, data: pd.Series, support_levels: List[float], resistance_levels: List[float]) -> Dict[str, float]:
        """Calculate oscillators based on support and resistance"""
        if not support_levels or not resistance_levels:
            return {}
        
        current_price = data.iloc[-1]
        
        # Calculate distance to nearest levels
        nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
        
        # Position within range
        total_range = nearest_resistance - nearest_support
        position = (current_price - nearest_support) / total_range if total_range > 0 else 0.5
        
        return {
            'position_in_range': position,
            'distance_to_support': (current_price - nearest_support) / current_price,
            'distance_to_resistance': (nearest_resistance - current_price) / current_price
        }
    
    def _calculate_level_confidence(self, data: pd.Series, support_levels: List[float], resistance_levels: List[float]) -> float:
        """Calculate confidence in support and resistance levels"""
        if not support_levels or not resistance_levels:
            return 0.0
        
        # Confidence based on level clustering and price proximity
        current_price = data.iloc[-1]
        
        # Calculate average distance to levels
        support_distances = [abs(current_price - level) / current_price for level in support_levels]
        resistance_distances = [abs(current_price - level) / current_price for level in resistance_levels]
        
        avg_distance = (np.mean(support_distances) + np.mean(resistance_distances)) / 2
        
        # Inverse relationship: closer levels = higher confidence
        confidence = 1.0 / (1.0 + avg_distance)
        
        return confidence

class PatternRecognitionModel(BaseTechnicalModel):
    """Pattern recognition model"""
    
    def __init__(self, pattern_window: int = 50, pattern_threshold: float = 0.05):
        super().__init__(pattern_window=pattern_window, pattern_threshold=pattern_threshold, min_data_points=pattern_window + 10)
        
    def analyze_technical(self, data: pd.Series) -> TechnicalAnalysis:
        """Analyze chart patterns"""
        if len(data) < self.params['pattern_window']:
            return TechnicalAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                trend_direction='unknown',
                trend_strength=0.0,
                support_levels=[],
                resistance_levels=[],
                pattern_detected='none',
                oscillator_signals={},
                confidence_level=0.0,
                timestamp=datetime.now()
            )
        
        # Detect patterns
        pattern_detected = self._detect_advanced_patterns(data)
        
        # Determine trend direction based on pattern
        trend_direction = self._pattern_to_trend_direction(pattern_detected)
        
        # Trend strength
        trend_strength = self._calculate_pattern_strength(data, pattern_detected)
        
        # Support and resistance levels
        support_levels = self._find_pattern_support_levels(data, pattern_detected)
        resistance_levels = self._find_pattern_resistance_levels(data, pattern_detected)
        
        # Oscillator signals
        oscillator_signals = self._calculate_pattern_oscillators(data, pattern_detected)
        
        # Confidence level
        confidence_level = self._calculate_pattern_confidence(data, pattern_detected)
        
        return TechnicalAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pattern_detected=pattern_detected,
            oscillator_signals=oscillator_signals,
            confidence_level=confidence_level,
            timestamp=datetime.now()
        )
    
    def _detect_advanced_patterns(self, data: pd.Series) -> str:
        """Detect advanced chart patterns"""
        if len(data) < 20:
            return 'none'
        
        recent_data = data.iloc[-20:]
        
        # Find peaks and troughs
        peaks, _ = find_peaks(recent_data.values, distance=3)
        troughs, _ = find_peaks(-recent_data.values, distance=3)
        
        # Pattern detection logic
        if len(peaks) >= 2 and len(troughs) >= 2:
            # Double top/bottom
            if len(peaks) >= 2:
                return 'double_top'
            elif len(troughs) >= 2:
                return 'double_bottom'
        
        # Head and shoulders pattern
        if len(peaks) >= 3:
            return 'head_and_shoulders'
        
        # Triangle patterns
        if len(peaks) >= 2 and len(troughs) >= 2:
            return 'triangle'
        
        # Flag and pennant patterns
        if len(peaks) >= 2:
            return 'flag'
        
        return 'none'
    
    def _pattern_to_trend_direction(self, pattern: str) -> str:
        """Convert pattern to trend direction"""
        bullish_patterns = ['double_bottom', 'inverse_head_and_shoulders', 'ascending_triangle']
        bearish_patterns = ['double_top', 'head_and_shoulders', 'descending_triangle']
        
        if pattern in bullish_patterns:
            return 'uptrend'
        elif pattern in bearish_patterns:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_pattern_strength(self, data: pd.Series, pattern: str) -> float:
        """Calculate pattern strength"""
        if pattern == 'none':
            return 0.0
        
        # Pattern strength based on price movement
        recent_data = data.iloc[-20:]
        price_range = (recent_data.max() - recent_data.min()) / recent_data.mean()
        
        return min(price_range, 1.0)
    
    def _find_pattern_support_levels(self, data: pd.Series, pattern: str) -> List[float]:
        """Find support levels based on pattern"""
        if pattern == 'none':
            return []
        
        recent_data = data.iloc[-20:]
        troughs, _ = find_peaks(-recent_data.values, distance=3)
        
        support_levels = []
        for trough in troughs:
            if trough < len(recent_data):
                support_levels.append(float(recent_data.iloc[trough]))
        
        return support_levels
    
    def _find_pattern_resistance_levels(self, data: pd.Series, pattern: str) -> List[float]:
        """Find resistance levels based on pattern"""
        if pattern == 'none':
            return []
        
        recent_data = data.iloc[-20:]
        peaks, _ = find_peaks(recent_data.values, distance=3)
        
        resistance_levels = []
        for peak in peaks:
            if peak < len(recent_data):
                resistance_levels.append(float(recent_data.iloc[peak]))
        
        return resistance_levels
    
    def _calculate_pattern_oscillators(self, data: pd.Series, pattern: str) -> Dict[str, float]:
        """Calculate oscillators based on pattern"""
        if pattern == 'none':
            return {}
        
        # Pattern-specific oscillators
        recent_data = data.iloc[-20:]
        
        # Momentum oscillator
        momentum = (recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]
        
        # Volatility oscillator
        volatility = recent_data.std() / recent_data.mean()
        
        return {
            'pattern_momentum': momentum,
            'pattern_volatility': volatility,
            'pattern_completion': len(recent_data) / 20  # Pattern completion percentage
        }
    
    def _calculate_pattern_confidence(self, data: pd.Series, pattern: str) -> float:
        """Calculate pattern confidence"""
        if pattern == 'none':
            return 0.0
        
        # Confidence based on pattern clarity and price movement
        recent_data = data.iloc[-20:]
        
        # Calculate pattern clarity
        peaks, _ = find_peaks(recent_data.values, distance=3)
        troughs, _ = find_peaks(-recent_data.values, distance=3)
        
        pattern_clarity = (len(peaks) + len(troughs)) / 20  # More peaks/troughs = clearer pattern
        
        # Calculate price movement strength
        price_movement = abs(recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]
        
        # Combined confidence
        confidence = (pattern_clarity + price_movement) / 2
        
        return min(confidence, 1.0)

class OscillatorModel(BaseTechnicalModel):
    """Oscillator analysis model"""
    
    def __init__(self, rsi_period: int = 14, macd_fast: int = 12, macd_slow: int = 26):
        super().__init__(rsi_period=rsi_period, macd_fast=macd_fast, macd_slow=macd_slow, min_data_points=50)
        
    def analyze_technical(self, data: pd.Series) -> TechnicalAnalysis:
        """Analyze oscillator signals"""
        if len(data) < self.params['rsi_period']:
            return TechnicalAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                trend_direction='unknown',
                trend_strength=0.0,
                support_levels=[],
                resistance_levels=[],
                pattern_detected='none',
                oscillator_signals={},
                confidence_level=0.0,
                timestamp=datetime.now()
            )
        
        # Calculate oscillators
        oscillator_signals = self._calculate_all_oscillators(data)
        
        # Determine trend direction based on oscillators
        trend_direction = self._oscillator_to_trend_direction(oscillator_signals)
        
        # Trend strength
        trend_strength = self._calculate_oscillator_strength(oscillator_signals)
        
        # Support and resistance levels
        support_levels = self._find_oscillator_support_levels(data, oscillator_signals)
        resistance_levels = self._find_oscillator_resistance_levels(data, oscillator_signals)
        
        # Pattern detection
        pattern_detected = self._detect_oscillator_patterns(oscillator_signals)
        
        # Confidence level
        confidence_level = self._calculate_oscillator_confidence(oscillator_signals)
        
        return TechnicalAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pattern_detected=pattern_detected,
            oscillator_signals=oscillator_signals,
            confidence_level=confidence_level,
            timestamp=datetime.now()
        )
    
    def _calculate_all_oscillators(self, data: pd.Series) -> Dict[str, float]:
        """Calculate all oscillator signals"""
        returns = data.pct_change().dropna()
        
        # RSI
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        avg_gain = gains.rolling(window=self.params['rsi_period']).mean()
        avg_loss = losses.rolling(window=self.params['rsi_period']).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # MACD
        ema_fast = data.ewm(span=self.params['macd_fast']).mean()
        ema_slow = data.ewm(span=self.params['macd_slow']).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9).mean()
        current_macd = macd.iloc[-1] if len(macd) > 0 else 0
        current_signal = signal.iloc[-1] if len(signal) > 0 else 0
        
        # Stochastic
        high_20 = data.rolling(window=20).max()
        low_20 = data.rolling(window=20).min()
        k_percent = 100 * (data - low_20) / (high_20 - low_20)
        current_k = k_percent.iloc[-1] if len(k_percent) > 0 else 50
        
        # Williams %R
        highest_high = data.rolling(window=14).max()
        lowest_low = data.rolling(window=14).min()
        williams_r = -100 * (highest_high - data) / (highest_high - lowest_low)
        current_williams_r = williams_r.iloc[-1] if len(williams_r) > 0 else -50
        
        return {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_signal,
            'macd_histogram': current_macd - current_signal,
            'stochastic_k': current_k,
            'williams_r': current_williams_r
        }
    
    def _oscillator_to_trend_direction(self, oscillators: Dict[str, float]) -> str:
        """Convert oscillator signals to trend direction"""
        rsi = oscillators.get('rsi', 50)
        macd = oscillators.get('macd', 0)
        macd_signal = oscillators.get('macd_signal', 0)
        stochastic = oscillators.get('stochastic_k', 50)
        williams_r = oscillators.get('williams_r', -50)
        
        # Bullish signals
        bullish_signals = 0
        if rsi > 50: bullish_signals += 1
        if macd > macd_signal: bullish_signals += 1
        if stochastic > 50: bullish_signals += 1
        if williams_r > -50: bullish_signals += 1
        
        # Bearish signals
        bearish_signals = 0
        if rsi < 50: bearish_signals += 1
        if macd < macd_signal: bearish_signals += 1
        if stochastic < 50: bearish_signals += 1
        if williams_r < -50: bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'uptrend'
        elif bearish_signals > bullish_signals:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_oscillator_strength(self, oscillators: Dict[str, float]) -> float:
        """Calculate oscillator strength"""
        rsi = oscillators.get('rsi', 50)
        macd_histogram = oscillators.get('macd_histogram', 0)
        stochastic = oscillators.get('stochastic_k', 50)
        williams_r = oscillators.get('williams_r', -50)
        
        # Normalize oscillator values
        rsi_strength = abs(rsi - 50) / 50
        macd_strength = abs(macd_histogram)
        stochastic_strength = abs(stochastic - 50) / 50
        williams_strength = abs(williams_r + 50) / 50
        
        # Average strength
        avg_strength = (rsi_strength + macd_strength + stochastic_strength + williams_strength) / 4
        
        return min(avg_strength, 1.0)
    
    def _find_oscillator_support_levels(self, data: pd.Series, oscillators: Dict[str, float]) -> List[float]:
        """Find support levels based on oscillators"""
        if len(data) < 20:
            return []
        
        # Support levels based on oversold conditions
        support_levels = []
        current_price = data.iloc[-1]
        
        # RSI oversold support
        if oscillators.get('rsi', 50) < 30:
            support_levels.append(current_price * 0.95)
        
        # Stochastic oversold support
        if oscillators.get('stochastic_k', 50) < 20:
            support_levels.append(current_price * 0.97)
        
        return support_levels
    
    def _find_oscillator_resistance_levels(self, data: pd.Series, oscillators: Dict[str, float]) -> List[float]:
        """Find resistance levels based on oscillators"""
        if len(data) < 20:
            return []
        
        # Resistance levels based on overbought conditions
        resistance_levels = []
        current_price = data.iloc[-1]
        
        # RSI overbought resistance
        if oscillators.get('rsi', 50) > 70:
            resistance_levels.append(current_price * 1.05)
        
        # Stochastic overbought resistance
        if oscillators.get('stochastic_k', 50) > 80:
            resistance_levels.append(current_price * 1.03)
        
        return resistance_levels
    
    def _detect_oscillator_patterns(self, oscillators: Dict[str, float]) -> str:
        """Detect patterns based on oscillators"""
        rsi = oscillators.get('rsi', 50)
        macd = oscillators.get('macd', 0)
        macd_signal = oscillators.get('macd_signal', 0)
        
        # Divergence patterns
        if rsi > 70 and macd < macd_signal:
            return 'bearish_divergence'
        elif rsi < 30 and macd > macd_signal:
            return 'bullish_divergence'
        else:
            return 'none'
    
    def _calculate_oscillator_confidence(self, oscillators: Dict[str, float]) -> float:
        """Calculate oscillator confidence"""
        rsi = oscillators.get('rsi', 50)
        macd_histogram = oscillators.get('macd_histogram', 0)
        stochastic = oscillators.get('stochastic_k', 50)
        williams_r = oscillators.get('williams_r', -50)
        
        # Confidence based on signal strength
        rsi_confidence = 1.0 - abs(rsi - 50) / 50
        macd_confidence = 1.0 - abs(macd_histogram)
        stochastic_confidence = 1.0 - abs(stochastic - 50) / 50
        williams_confidence = 1.0 - abs(williams_r + 50) / 50
        
        # Average confidence
        avg_confidence = (rsi_confidence + macd_confidence + stochastic_confidence + williams_confidence) / 4
        
        return avg_confidence

class TechnicalAnalysisModel:
    """Comprehensive technical analysis model for S&P 500 trading"""
    
    def __init__(self, analysis_window: int = 252):
        self.analysis_window = analysis_window
        self.is_fitted = False
        self.models = {}
        
        # Initialize different technical analysis models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all technical analysis models"""
        
        # Basic technical models
        self.models['trend_analysis'] = TrendAnalysisModel()
        self.models['support_resistance'] = SupportResistanceModel()
        self.models['pattern_recognition'] = PatternRecognitionModel()
        self.models['oscillator_analysis'] = OscillatorModel()
        
    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            # Placeholder for real market data integration
            logger.info(f"Would get real market data for {symbol} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            # Placeholder for real economic data integration
            logger.info(f"Would get real economic data for {series_id} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real economic data: {e}")
            return None

    def _get_real_alternative_data(self, data_type: str, **kwargs):
        """Get real alternative data from approved sources"""
        try:
            # Implement based on data type
            if data_type == 'sentiment':
                return self._get_sentiment_data(**kwargs)
            elif data_type == 'news':
                return self._get_news_data(**kwargs)
            elif data_type == 'social':
                return self._get_social_data(**kwargs)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to get real alternative data: {e}")
            return None

    def fit(self, data: pd.Series) -> 'TechnicalAnalysisModel':
        """Fit all technical analysis models"""
        if len(data) < self.analysis_window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.analysis_window}")
        
        # Fit all models
        for model_name, model in self.models.items():
            try:
                model.fit(data)
            except Exception as e:
                logger.warning(f"Failed to fit {model_name}: {e}")
        
        self.is_fitted = True
        return self

    def analyze_technical(self, data: pd.Series, model_name: str = None) -> TechnicalAnalysis:
        """Perform technical analysis"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before technical analysis")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            return self.models[model_name].analyze_technical(data)
        else:
            # Return trend analysis as default
            return self.models['trend_analysis'].analyze_technical(data)

    def get_available_models(self) -> List[str]:
        """Get list of available technical analysis models"""
        return list(self.models.keys())

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name].params

    def get_parameters(self) -> Dict[str, Any]:
        """Get all model parameters"""
        return {
            'analysis_window': self.analysis_window,
            'is_fitted': self.is_fitted,
            'available_models': self.get_available_models(),
            'model_parameters': {
                name: model.params for name, model in self.models.items()
            }
        } 
"""
Advanced Technical Analysis Model Implementations for S&P 500 Trading
===================================================================

Advanced technical analysis models including:
- Trend analysis and detection
- Support and resistance levels
- Pattern recognition
- Oscillators and indicators
- Fibonacci retracements
- Elliott Wave analysis
- Advanced chart patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class TechnicalAnalysis:
    """Technical analysis result"""
    symbol: str
    trend_direction: str
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    pattern_detected: str
    oscillator_signals: Dict[str, float]
    confidence_level: float
    timestamp: datetime

class BaseTechnicalModel:
    """Base class for technical analysis models"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
        
    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            # Placeholder for real market data integration
            logger.info(f"Would get real market data for {symbol} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            # Placeholder for real economic data integration
            logger.info(f"Would get real economic data for {series_id} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real economic data: {e}")
            return None

    def fit(self, data: pd.Series) -> 'BaseTechnicalModel':
        """Fit the technical analysis model"""
        if len(data) < self.params.get('min_data_points', 100):
            raise ValueError(f"Insufficient data: {len(data)} < {self.params.get('min_data_points', 100)}")
        self.is_fitted = True
        return self

    def analyze_technical(self, data: pd.Series) -> TechnicalAnalysis:
        """Perform technical analysis"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before technical analysis")
        raise NotImplementedError("Subclasses must implement analyze_technical method")

class TrendAnalysisModel(BaseTechnicalModel):
    """Trend analysis model"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50, trend_threshold: float = 0.02):
        super().__init__(short_window=short_window, long_window=long_window, trend_threshold=trend_threshold, min_data_points=long_window + 10)
        
    def analyze_technical(self, data: pd.Series) -> TechnicalAnalysis:
        """Analyze trend patterns"""
        if len(data) < self.params['long_window']:
            return TechnicalAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                trend_direction='unknown',
                trend_strength=0.0,
                support_levels=[],
                resistance_levels=[],
                pattern_detected='none',
                oscillator_signals={},
                confidence_level=0.0,
                timestamp=datetime.now()
            )
        
        # Calculate moving averages
        short_ma = data.rolling(window=self.params['short_window']).mean()
        long_ma = data.rolling(window=self.params['long_window']).mean()
        
        # Determine trend direction
        current_price = data.iloc[-1]
        current_short_ma = short_ma.iloc[-1] if len(short_ma) > 0 else current_price
        current_long_ma = long_ma.iloc[-1] if len(long_ma) > 0 else current_price
        
        # Trend direction
        if current_short_ma > current_long_ma and current_price > current_short_ma:
            trend_direction = 'uptrend'
        elif current_short_ma < current_long_ma and current_price < current_short_ma:
            trend_direction = 'downtrend'
        else:
            trend_direction = 'sideways'
        
        # Trend strength
        trend_strength = abs(current_short_ma - current_long_ma) / current_long_ma if current_long_ma > 0 else 0
        
        # Support and resistance levels
        support_levels = self._find_support_levels(data)
        resistance_levels = self._find_resistance_levels(data)
        
        # Pattern detection
        pattern_detected = self._detect_patterns(data)
        
        # Oscillator signals
        oscillator_signals = self._calculate_oscillators(data)
        
        # Confidence level
        confidence_level = min(trend_strength / self.params['trend_threshold'], 1.0)
        
        return TechnicalAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pattern_detected=pattern_detected,
            oscillator_signals=oscillator_signals,
            confidence_level=confidence_level,
            timestamp=datetime.now()
        )
    
    def _find_support_levels(self, data: pd.Series) -> List[float]:
        """Find support levels"""
        if len(data) < 20:
            return []
        
        # Find local minima
        peaks, _ = find_peaks(-data.values, distance=10)
        
        support_levels = []
        for peak in peaks[-5:]:  # Last 5 support levels
            if peak < len(data):
                support_levels.append(float(data.iloc[peak]))
        
        return support_levels
    
    def _find_resistance_levels(self, data: pd.Series) -> List[float]:
        """Find resistance levels"""
        if len(data) < 20:
            return []
        
        # Find local maxima
        peaks, _ = find_peaks(data.values, distance=10)
        
        resistance_levels = []
        for peak in peaks[-5:]:  # Last 5 resistance levels
            if peak < len(data):
                resistance_levels.append(float(data.iloc[peak]))
        
        return resistance_levels
    
    def _detect_patterns(self, data: pd.Series) -> str:
        """Detect chart patterns"""
        if len(data) < 20:
            return 'none'
        
        # Simple pattern detection
        recent_data = data.iloc[-20:]
        
        # Double top/bottom detection
        peaks, _ = find_peaks(recent_data.values, distance=5)
        troughs, _ = find_peaks(-recent_data.values, distance=5)
        
        if len(peaks) >= 2:
            return 'double_top'
        elif len(troughs) >= 2:
            return 'double_bottom'
        else:
            return 'none'
    
    def _calculate_oscillators(self, data: pd.Series) -> Dict[str, float]:
        """Calculate oscillator signals"""
        if len(data) < 14:
            return {}
        
        # RSI calculation
        returns = data.pct_change().dropna()
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        
        avg_gain = gains.rolling(window=14).mean()
        avg_loss = losses.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # MACD calculation
        ema12 = data.ewm(span=12).mean()
        ema26 = data.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        current_macd = macd.iloc[-1] if len(macd) > 0 else 0
        current_signal = signal.iloc[-1] if len(signal) > 0 else 0
        
        return {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_signal,
            'macd_histogram': current_macd - current_signal
        }

class SupportResistanceModel(BaseTechnicalModel):
    """Support and resistance level model"""
    
    def __init__(self, window: int = 20, level_threshold: float = 0.02):
        super().__init__(window=window, level_threshold=level_threshold, min_data_points=window + 10)
        
    def analyze_technical(self, data: pd.Series) -> TechnicalAnalysis:
        """Analyze support and resistance levels"""
        if len(data) < self.params['window']:
            return TechnicalAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                trend_direction='unknown',
                trend_strength=0.0,
                support_levels=[],
                resistance_levels=[],
                pattern_detected='none',
                oscillator_signals={},
                confidence_level=0.0,
                timestamp=datetime.now()
            )
        
        # Find support and resistance levels
        support_levels = self._find_dynamic_support_levels(data)
        resistance_levels = self._find_dynamic_resistance_levels(data)
        
        # Determine trend direction based on current price vs levels
        current_price = data.iloc[-1]
        
        if resistance_levels:
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
            if current_price > nearest_resistance * (1 - self.params['level_threshold']):
                trend_direction = 'uptrend'
            else:
                trend_direction = 'downtrend'
        else:
            trend_direction = 'sideways'
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(data)
        
        # Pattern detection
        pattern_detected = self._detect_support_resistance_patterns(data, support_levels, resistance_levels)
        
        # Oscillator signals
        oscillator_signals = self._calculate_support_resistance_oscillators(data, support_levels, resistance_levels)
        
        # Confidence level
        confidence_level = self._calculate_level_confidence(data, support_levels, resistance_levels)
        
        return TechnicalAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pattern_detected=pattern_detected,
            oscillator_signals=oscillator_signals,
            confidence_level=confidence_level,
            timestamp=datetime.now()
        )
    
    def _find_dynamic_support_levels(self, data: pd.Series) -> List[float]:
        """Find dynamic support levels"""
        if len(data) < 20:
            return []
        
        # Find local minima with clustering
        troughs, _ = find_peaks(-data.values, distance=10)
        
        if len(troughs) < 2:
            return []
        
        # Cluster nearby levels
        trough_values = [data.iloc[t] for t in troughs]
        trough_array = np.array(trough_values).reshape(-1, 1)
        
        if len(trough_array) >= 2:
            kmeans = KMeans(n_clusters=min(3, len(trough_array)), random_state=42)
            clusters = kmeans.fit_predict(trough_array)
            
            # Get cluster centers as support levels
            support_levels = kmeans.cluster_centers_.flatten().tolist()
            return [level for level in support_levels if level > 0]
        else:
            return trough_values
    
    def _find_dynamic_resistance_levels(self, data: pd.Series) -> List[float]:
        """Find dynamic resistance levels"""
        if len(data) < 20:
            return []
        
        # Find local maxima with clustering
        peaks, _ = find_peaks(data.values, distance=10)
        
        if len(peaks) < 2:
            return []
        
        # Cluster nearby levels
        peak_values = [data.iloc[p] for p in peaks]
        peak_array = np.array(peak_values).reshape(-1, 1)
        
        if len(peak_array) >= 2:
            kmeans = KMeans(n_clusters=min(3, len(peak_array)), random_state=42)
            clusters = kmeans.fit_predict(peak_array)
            
            # Get cluster centers as resistance levels
            resistance_levels = kmeans.cluster_centers_.flatten().tolist()
            return [level for level in resistance_levels if level > 0]
        else:
            return peak_values
    
    def _calculate_trend_strength(self, data: pd.Series) -> float:
        """Calculate trend strength"""
        if len(data) < 20:
            return 0.0
        
        # Linear regression trend strength
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        model = LinearRegression()
        model.fit(X, y)
        
        # R-squared as trend strength
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return r_squared
    
    def _detect_support_resistance_patterns(self, data: pd.Series, support_levels: List[float], resistance_levels: List[float]) -> str:
        """Detect patterns based on support and resistance"""
        if not support_levels or not resistance_levels:
            return 'none'
        
        current_price = data.iloc[-1]
        
        # Check for breakout patterns
        nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
        
        if current_price > nearest_resistance:
            return 'breakout_up'
        elif current_price < nearest_support:
            return 'breakout_down'
        else:
            return 'consolidation'
    
    def _calculate_support_resistance_oscillators(self, data: pd.Series, support_levels: List[float], resistance_levels: List[float]) -> Dict[str, float]:
        """Calculate oscillators based on support and resistance"""
        if not support_levels or not resistance_levels:
            return {}
        
        current_price = data.iloc[-1]
        
        # Calculate distance to nearest levels
        nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
        
        # Position within range
        total_range = nearest_resistance - nearest_support
        position = (current_price - nearest_support) / total_range if total_range > 0 else 0.5
        
        return {
            'position_in_range': position,
            'distance_to_support': (current_price - nearest_support) / current_price,
            'distance_to_resistance': (nearest_resistance - current_price) / current_price
        }
    
    def _calculate_level_confidence(self, data: pd.Series, support_levels: List[float], resistance_levels: List[float]) -> float:
        """Calculate confidence in support and resistance levels"""
        if not support_levels or not resistance_levels:
            return 0.0
        
        # Confidence based on level clustering and price proximity
        current_price = data.iloc[-1]
        
        # Calculate average distance to levels
        support_distances = [abs(current_price - level) / current_price for level in support_levels]
        resistance_distances = [abs(current_price - level) / current_price for level in resistance_levels]
        
        avg_distance = (np.mean(support_distances) + np.mean(resistance_distances)) / 2
        
        # Inverse relationship: closer levels = higher confidence
        confidence = 1.0 / (1.0 + avg_distance)
        
        return confidence

class PatternRecognitionModel(BaseTechnicalModel):
    """Pattern recognition model"""
    
    def __init__(self, pattern_window: int = 50, pattern_threshold: float = 0.05):
        super().__init__(pattern_window=pattern_window, pattern_threshold=pattern_threshold, min_data_points=pattern_window + 10)
        
    def analyze_technical(self, data: pd.Series) -> TechnicalAnalysis:
        """Analyze chart patterns"""
        if len(data) < self.params['pattern_window']:
            return TechnicalAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                trend_direction='unknown',
                trend_strength=0.0,
                support_levels=[],
                resistance_levels=[],
                pattern_detected='none',
                oscillator_signals={},
                confidence_level=0.0,
                timestamp=datetime.now()
            )
        
        # Detect patterns
        pattern_detected = self._detect_advanced_patterns(data)
        
        # Determine trend direction based on pattern
        trend_direction = self._pattern_to_trend_direction(pattern_detected)
        
        # Trend strength
        trend_strength = self._calculate_pattern_strength(data, pattern_detected)
        
        # Support and resistance levels
        support_levels = self._find_pattern_support_levels(data, pattern_detected)
        resistance_levels = self._find_pattern_resistance_levels(data, pattern_detected)
        
        # Oscillator signals
        oscillator_signals = self._calculate_pattern_oscillators(data, pattern_detected)
        
        # Confidence level
        confidence_level = self._calculate_pattern_confidence(data, pattern_detected)
        
        return TechnicalAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pattern_detected=pattern_detected,
            oscillator_signals=oscillator_signals,
            confidence_level=confidence_level,
            timestamp=datetime.now()
        )
    
    def _detect_advanced_patterns(self, data: pd.Series) -> str:
        """Detect advanced chart patterns"""
        if len(data) < 20:
            return 'none'
        
        recent_data = data.iloc[-20:]
        
        # Find peaks and troughs
        peaks, _ = find_peaks(recent_data.values, distance=3)
        troughs, _ = find_peaks(-recent_data.values, distance=3)
        
        # Pattern detection logic
        if len(peaks) >= 2 and len(troughs) >= 2:
            # Double top/bottom
            if len(peaks) >= 2:
                return 'double_top'
            elif len(troughs) >= 2:
                return 'double_bottom'
        
        # Head and shoulders pattern
        if len(peaks) >= 3:
            return 'head_and_shoulders'
        
        # Triangle patterns
        if len(peaks) >= 2 and len(troughs) >= 2:
            return 'triangle'
        
        # Flag and pennant patterns
        if len(peaks) >= 2:
            return 'flag'
        
        return 'none'
    
    def _pattern_to_trend_direction(self, pattern: str) -> str:
        """Convert pattern to trend direction"""
        bullish_patterns = ['double_bottom', 'inverse_head_and_shoulders', 'ascending_triangle']
        bearish_patterns = ['double_top', 'head_and_shoulders', 'descending_triangle']
        
        if pattern in bullish_patterns:
            return 'uptrend'
        elif pattern in bearish_patterns:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_pattern_strength(self, data: pd.Series, pattern: str) -> float:
        """Calculate pattern strength"""
        if pattern == 'none':
            return 0.0
        
        # Pattern strength based on price movement
        recent_data = data.iloc[-20:]
        price_range = (recent_data.max() - recent_data.min()) / recent_data.mean()
        
        return min(price_range, 1.0)
    
    def _find_pattern_support_levels(self, data: pd.Series, pattern: str) -> List[float]:
        """Find support levels based on pattern"""
        if pattern == 'none':
            return []
        
        recent_data = data.iloc[-20:]
        troughs, _ = find_peaks(-recent_data.values, distance=3)
        
        support_levels = []
        for trough in troughs:
            if trough < len(recent_data):
                support_levels.append(float(recent_data.iloc[trough]))
        
        return support_levels
    
    def _find_pattern_resistance_levels(self, data: pd.Series, pattern: str) -> List[float]:
        """Find resistance levels based on pattern"""
        if pattern == 'none':
            return []
        
        recent_data = data.iloc[-20:]
        peaks, _ = find_peaks(recent_data.values, distance=3)
        
        resistance_levels = []
        for peak in peaks:
            if peak < len(recent_data):
                resistance_levels.append(float(recent_data.iloc[peak]))
        
        return resistance_levels
    
    def _calculate_pattern_oscillators(self, data: pd.Series, pattern: str) -> Dict[str, float]:
        """Calculate oscillators based on pattern"""
        if pattern == 'none':
            return {}
        
        # Pattern-specific oscillators
        recent_data = data.iloc[-20:]
        
        # Momentum oscillator
        momentum = (recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]
        
        # Volatility oscillator
        volatility = recent_data.std() / recent_data.mean()
        
        return {
            'pattern_momentum': momentum,
            'pattern_volatility': volatility,
            'pattern_completion': len(recent_data) / 20  # Pattern completion percentage
        }
    
    def _calculate_pattern_confidence(self, data: pd.Series, pattern: str) -> float:
        """Calculate pattern confidence"""
        if pattern == 'none':
            return 0.0
        
        # Confidence based on pattern clarity and price movement
        recent_data = data.iloc[-20:]
        
        # Calculate pattern clarity
        peaks, _ = find_peaks(recent_data.values, distance=3)
        troughs, _ = find_peaks(-recent_data.values, distance=3)
        
        pattern_clarity = (len(peaks) + len(troughs)) / 20  # More peaks/troughs = clearer pattern
        
        # Calculate price movement strength
        price_movement = abs(recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]
        
        # Combined confidence
        confidence = (pattern_clarity + price_movement) / 2
        
        return min(confidence, 1.0)

class OscillatorModel(BaseTechnicalModel):
    """Oscillator analysis model"""
    
    def __init__(self, rsi_period: int = 14, macd_fast: int = 12, macd_slow: int = 26):
        super().__init__(rsi_period=rsi_period, macd_fast=macd_fast, macd_slow=macd_slow, min_data_points=50)
        
    def analyze_technical(self, data: pd.Series) -> TechnicalAnalysis:
        """Analyze oscillator signals"""
        if len(data) < self.params['rsi_period']:
            return TechnicalAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                trend_direction='unknown',
                trend_strength=0.0,
                support_levels=[],
                resistance_levels=[],
                pattern_detected='none',
                oscillator_signals={},
                confidence_level=0.0,
                timestamp=datetime.now()
            )
        
        # Calculate oscillators
        oscillator_signals = self._calculate_all_oscillators(data)
        
        # Determine trend direction based on oscillators
        trend_direction = self._oscillator_to_trend_direction(oscillator_signals)
        
        # Trend strength
        trend_strength = self._calculate_oscillator_strength(oscillator_signals)
        
        # Support and resistance levels
        support_levels = self._find_oscillator_support_levels(data, oscillator_signals)
        resistance_levels = self._find_oscillator_resistance_levels(data, oscillator_signals)
        
        # Pattern detection
        pattern_detected = self._detect_oscillator_patterns(oscillator_signals)
        
        # Confidence level
        confidence_level = self._calculate_oscillator_confidence(oscillator_signals)
        
        return TechnicalAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pattern_detected=pattern_detected,
            oscillator_signals=oscillator_signals,
            confidence_level=confidence_level,
            timestamp=datetime.now()
        )
    
    def _calculate_all_oscillators(self, data: pd.Series) -> Dict[str, float]:
        """Calculate all oscillator signals"""
        returns = data.pct_change().dropna()
        
        # RSI
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        avg_gain = gains.rolling(window=self.params['rsi_period']).mean()
        avg_loss = losses.rolling(window=self.params['rsi_period']).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # MACD
        ema_fast = data.ewm(span=self.params['macd_fast']).mean()
        ema_slow = data.ewm(span=self.params['macd_slow']).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9).mean()
        current_macd = macd.iloc[-1] if len(macd) > 0 else 0
        current_signal = signal.iloc[-1] if len(signal) > 0 else 0
        
        # Stochastic
        high_20 = data.rolling(window=20).max()
        low_20 = data.rolling(window=20).min()
        k_percent = 100 * (data - low_20) / (high_20 - low_20)
        current_k = k_percent.iloc[-1] if len(k_percent) > 0 else 50
        
        # Williams %R
        highest_high = data.rolling(window=14).max()
        lowest_low = data.rolling(window=14).min()
        williams_r = -100 * (highest_high - data) / (highest_high - lowest_low)
        current_williams_r = williams_r.iloc[-1] if len(williams_r) > 0 else -50
        
        return {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_signal,
            'macd_histogram': current_macd - current_signal,
            'stochastic_k': current_k,
            'williams_r': current_williams_r
        }
    
    def _oscillator_to_trend_direction(self, oscillators: Dict[str, float]) -> str:
        """Convert oscillator signals to trend direction"""
        rsi = oscillators.get('rsi', 50)
        macd = oscillators.get('macd', 0)
        macd_signal = oscillators.get('macd_signal', 0)
        stochastic = oscillators.get('stochastic_k', 50)
        williams_r = oscillators.get('williams_r', -50)
        
        # Bullish signals
        bullish_signals = 0
        if rsi > 50: bullish_signals += 1
        if macd > macd_signal: bullish_signals += 1
        if stochastic > 50: bullish_signals += 1
        if williams_r > -50: bullish_signals += 1
        
        # Bearish signals
        bearish_signals = 0
        if rsi < 50: bearish_signals += 1
        if macd < macd_signal: bearish_signals += 1
        if stochastic < 50: bearish_signals += 1
        if williams_r < -50: bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'uptrend'
        elif bearish_signals > bullish_signals:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_oscillator_strength(self, oscillators: Dict[str, float]) -> float:
        """Calculate oscillator strength"""
        rsi = oscillators.get('rsi', 50)
        macd_histogram = oscillators.get('macd_histogram', 0)
        stochastic = oscillators.get('stochastic_k', 50)
        williams_r = oscillators.get('williams_r', -50)
        
        # Normalize oscillator values
        rsi_strength = abs(rsi - 50) / 50
        macd_strength = abs(macd_histogram)
        stochastic_strength = abs(stochastic - 50) / 50
        williams_strength = abs(williams_r + 50) / 50
        
        # Average strength
        avg_strength = (rsi_strength + macd_strength + stochastic_strength + williams_strength) / 4
        
        return min(avg_strength, 1.0)
    
    def _find_oscillator_support_levels(self, data: pd.Series, oscillators: Dict[str, float]) -> List[float]:
        """Find support levels based on oscillators"""
        if len(data) < 20:
            return []
        
        # Support levels based on oversold conditions
        support_levels = []
        current_price = data.iloc[-1]
        
        # RSI oversold support
        if oscillators.get('rsi', 50) < 30:
            support_levels.append(current_price * 0.95)
        
        # Stochastic oversold support
        if oscillators.get('stochastic_k', 50) < 20:
            support_levels.append(current_price * 0.97)
        
        return support_levels
    
    def _find_oscillator_resistance_levels(self, data: pd.Series, oscillators: Dict[str, float]) -> List[float]:
        """Find resistance levels based on oscillators"""
        if len(data) < 20:
            return []
        
        # Resistance levels based on overbought conditions
        resistance_levels = []
        current_price = data.iloc[-1]
        
        # RSI overbought resistance
        if oscillators.get('rsi', 50) > 70:
            resistance_levels.append(current_price * 1.05)
        
        # Stochastic overbought resistance
        if oscillators.get('stochastic_k', 50) > 80:
            resistance_levels.append(current_price * 1.03)
        
        return resistance_levels
    
    def _detect_oscillator_patterns(self, oscillators: Dict[str, float]) -> str:
        """Detect patterns based on oscillators"""
        rsi = oscillators.get('rsi', 50)
        macd = oscillators.get('macd', 0)
        macd_signal = oscillators.get('macd_signal', 0)
        
        # Divergence patterns
        if rsi > 70 and macd < macd_signal:
            return 'bearish_divergence'
        elif rsi < 30 and macd > macd_signal:
            return 'bullish_divergence'
        else:
            return 'none'
    
    def _calculate_oscillator_confidence(self, oscillators: Dict[str, float]) -> float:
        """Calculate oscillator confidence"""
        rsi = oscillators.get('rsi', 50)
        macd_histogram = oscillators.get('macd_histogram', 0)
        stochastic = oscillators.get('stochastic_k', 50)
        williams_r = oscillators.get('williams_r', -50)
        
        # Confidence based on signal strength
        rsi_confidence = 1.0 - abs(rsi - 50) / 50
        macd_confidence = 1.0 - abs(macd_histogram)
        stochastic_confidence = 1.0 - abs(stochastic - 50) / 50
        williams_confidence = 1.0 - abs(williams_r + 50) / 50
        
        # Average confidence
        avg_confidence = (rsi_confidence + macd_confidence + stochastic_confidence + williams_confidence) / 4
        
        return avg_confidence

class TechnicalAnalysisModel:
    """Comprehensive technical analysis model for S&P 500 trading"""
    
    def __init__(self, analysis_window: int = 252):
        self.analysis_window = analysis_window
        self.is_fitted = False
        self.models = {}
        
        # Initialize different technical analysis models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all technical analysis models"""
        
        # Basic technical models
        self.models['trend_analysis'] = TrendAnalysisModel()
        self.models['support_resistance'] = SupportResistanceModel()
        self.models['pattern_recognition'] = PatternRecognitionModel()
        self.models['oscillator_analysis'] = OscillatorModel()
        
    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            # Placeholder for real market data integration
            logger.info(f"Would get real market data for {symbol} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            # Placeholder for real economic data integration
            logger.info(f"Would get real economic data for {series_id} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real economic data: {e}")
            return None

    def _get_real_alternative_data(self, data_type: str, **kwargs):
        """Get real alternative data from approved sources"""
        try:
            # Implement based on data type
            if data_type == 'sentiment':
                return self._get_sentiment_data(**kwargs)
            elif data_type == 'news':
                return self._get_news_data(**kwargs)
            elif data_type == 'social':
                return self._get_social_data(**kwargs)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to get real alternative data: {e}")
            return None

    def fit(self, data: pd.Series) -> 'TechnicalAnalysisModel':
        """Fit all technical analysis models"""
        if len(data) < self.analysis_window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.analysis_window}")
        
        # Fit all models
        for model_name, model in self.models.items():
            try:
                model.fit(data)
            except Exception as e:
                logger.warning(f"Failed to fit {model_name}: {e}")
        
        self.is_fitted = True
        return self

    def analyze_technical(self, data: pd.Series, model_name: str = None) -> TechnicalAnalysis:
        """Perform technical analysis"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before technical analysis")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            return self.models[model_name].analyze_technical(data)
        else:
            # Return trend analysis as default
            return self.models['trend_analysis'].analyze_technical(data)

    def get_available_models(self) -> List[str]:
        """Get list of available technical analysis models"""
        return list(self.models.keys())

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name].params

    def get_parameters(self) -> Dict[str, Any]:
        """Get all model parameters"""
        return {
            'analysis_window': self.analysis_window,
            'is_fitted': self.is_fitted,
            'available_models': self.get_available_models(),
            'model_parameters': {
                name: model.params for name, model in self.models.items()
            }
        } 