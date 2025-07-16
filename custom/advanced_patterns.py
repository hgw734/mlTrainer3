"""
Advanced Pattern Recognition Models
===================================
Implements sophisticated pattern recognition strategies including
Elliott Wave, Harmonic patterns, and price action methodologies.
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
class PatternSignal:
    """Pattern detection signal"""
    pattern_name: str
    pattern_type: str  # bullish/bearish
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    timestamp: datetime
    pattern_data: Dict[str, Any] = None


class BasePatternModel(ABC):
    """Base class for pattern recognition models"""
    
    def __init__(self, min_pattern_bars: int = 5):
        self.min_pattern_bars = min_pattern_bars
        self.detected_patterns = []
        
    @abstractmethod
    def detect_pattern(self, data: pd.DataFrame, **kwargs) -> Optional[PatternSignal]:
        """Detect specific pattern in data"""
        pass
    
    def validate_data(self, data: pd.DataFrame, required_bars: int = None) -> bool:
        """Validate input data"""
        required = required_bars or self.min_pattern_bars
        if data is None or data.empty:
            return False
        if len(data) < required:
            return False
        return True


class ElliottWaveModel(BasePatternModel):
    """
    Wave counting:
    - 5-wave impulses
    - ABC corrections
    - Fibonacci ratios
    - Degree analysis
    """
    
    def __init__(self, min_wave_bars: int = 13):
        super().__init__(min_pattern_bars=min_wave_bars)
        self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
        
    def detect_pattern(self, data: pd.DataFrame, **kwargs) -> Optional[PatternSignal]:
        """Detect Elliott Wave patterns"""
        if not self.validate_data(data, 50):  # Need more data for waves
            return None
        
        # Find pivots (local highs and lows)
        pivots = self._find_pivots(data)
        
        if len(pivots) < 6:  # Need at least 6 pivots for 5-wave pattern
            return None
        
        # Try to identify 5-wave impulse
        impulse = self._identify_impulse_wave(pivots)
        if impulse:
            return self._create_impulse_signal(impulse, data)
        
        # Try to identify ABC correction
        correction = self._identify_correction(pivots)
        if correction:
            return self._create_correction_signal(correction, data)
        
        return None
    
    def _find_pivots(self, data: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find pivot points (local highs and lows)"""
        pivots = []
        highs = data['high'].rolling(window=window*2+1, center=True).max()
        lows = data['low'].rolling(window=window*2+1, center=True).min()
        
        for i in range(window, len(data) - window):
            # Check for pivot high
            if data['high'].iloc[i] == highs.iloc[i]:
                pivots.append({
                    'index': i,
                    'price': data['high'].iloc[i],
                    'type': 'high',
                    'time': data.index[i] if hasattr(data.index, '__iter__') else i
                })
            # Check for pivot low
            elif data['low'].iloc[i] == lows.iloc[i]:
                pivots.append({
                    'index': i,
                    'price': data['low'].iloc[i],
                    'type': 'low',
                    'time': data.index[i] if hasattr(data.index, '__iter__') else i
                })
        
        return pivots
    
    def _identify_impulse_wave(self, pivots: List[Dict]) -> Optional[Dict]:
        """Identify 5-wave impulse pattern"""
        # Look for alternating high-low pattern with proper ratios
        for i in range(len(pivots) - 5):
            # Check for proper alternation: low-high-low-high-low-high
            if (pivots[i]['type'] == 'low' and 
                pivots[i+1]['type'] == 'high' and
                pivots[i+2]['type'] == 'low' and
                pivots[i+3]['type'] == 'high' and
                pivots[i+4]['type'] == 'low' and
                pivots[i+5]['type'] == 'high'):
                
                # Waves
                wave1 = pivots[i+1]['price'] - pivots[i]['price']
                wave2 = pivots[i+1]['price'] - pivots[i+2]['price']
                wave3 = pivots[i+3]['price'] - pivots[i+2]['price']
                wave4 = pivots[i+3]['price'] - pivots[i+4]['price']
                wave5 = pivots[i+5]['price'] - pivots[i+4]['price']
                
                # Elliott Wave rules
                # 1. Wave 2 cannot retrace more than 100% of wave 1
                if wave2 >= wave1:
                    continue
                
                # 2. Wave 3 cannot be the shortest
                if wave3 < wave1 or wave3 < wave5:
                    continue
                
                # 3. Wave 4 cannot overlap wave 1 price territory
                if pivots[i+4]['price'] < pivots[i+1]['price']:
                    continue
                
                # Check Fibonacci relationships
                wave2_ratio = wave2 / wave1
                wave3_ratio = wave3 / wave1
                wave4_ratio = wave4 / wave3
                wave5_ratio = wave5 / wave1
                
                # Common ratios
                valid_pattern = (
                    0.3 < wave2_ratio < 0.8 and  # Wave 2 typically 38-62% of wave 1
                    1.0 < wave3_ratio < 3.0 and  # Wave 3 often 1.618x wave 1
                    0.2 < wave4_ratio < 0.6 and  # Wave 4 typically 38-50% of wave 3
                    0.5 < wave5_ratio < 1.7      # Wave 5 often equals wave 1
                )
                
                if valid_pattern:
                    return {
                        'type': 'impulse',
                        'direction': 'bullish',
                        'waves': pivots[i:i+6],
                        'wave_lengths': [wave1, wave2, wave3, wave4, wave5],
                        'confidence': self._calculate_pattern_confidence(
                            [wave2_ratio, wave3_ratio, wave4_ratio, wave5_ratio]
                        )
                    }
        
        return None
    
    def _identify_correction(self, pivots: List[Dict]) -> Optional[Dict]:
        """Identify ABC correction pattern"""
        for i in range(len(pivots) - 3):
            # Look for 3-wave pattern
            if (pivots[i]['type'] == 'high' and 
                pivots[i+1]['type'] == 'low' and
                pivots[i+2]['type'] == 'high' and
                pivots[i+3]['type'] == 'low'):
                
                # Waves
                waveA = pivots[i]['price'] - pivots[i+1]['price']
                waveB = pivots[i+2]['price'] - pivots[i+1]['price']
                waveC = pivots[i+2]['price'] - pivots[i+3]['price']
                
                # Check ratios
                waveB_ratio = waveB / waveA
                waveC_ratio = waveC / waveA
                
                # Common ABC ratios
                valid_correction = (
                    0.3 < waveB_ratio < 0.9 and  # B typically 38-78% of A
                    0.6 < waveC_ratio < 1.7      # C often equals A
                )
                
                if valid_correction:
                    return {
                        'type': 'correction',
                        'direction': 'bearish',
                        'waves': pivots[i:i+4],
                        'wave_lengths': [waveA, waveB, waveC],
                        'confidence': self._calculate_pattern_confidence(
                            [waveB_ratio, waveC_ratio]
                        )
                    }
        
        return None
    
    def _calculate_pattern_confidence(self, ratios: List[float]) -> float:
        """Calculate pattern confidence based on Fibonacci ratios"""
        confidence_scores = []
        
        for ratio in ratios:
            # Find closest Fibonacci ratio
            distances = [abs(ratio - fib) for fib in self.fibonacci_ratios]
            min_distance = min(distances)
            
            # Convert distance to confidence (closer = higher confidence)
            confidence = max(0, 1 - min_distance * 2)
            confidence_scores.append(confidence)
        
        return np.mean(confidence_scores)
    
    def _create_impulse_signal(self, impulse: Dict, data: pd.DataFrame) -> PatternSignal:
        """Create signal from impulse wave"""
        last_wave = impulse['waves'][-1]
        current_price = data['close'].iloc[-1]
        
        # Project next move based on wave 5 completion
        wave1_length = impulse['wave_lengths'][0]
        target = last_wave['price'] + wave1_length * 0.618  # Common retracement
        stop_loss = impulse['waves'][-2]['price']  # Below wave 4 low
        
        return PatternSignal(
            pattern_name='Elliott Wave Impulse',
            pattern_type='bullish',
            confidence=impulse['confidence'],
            entry_price=current_price,
            target_price=target,
            stop_loss=stop_loss,
            timestamp=datetime.now(),
            pattern_data=impulse
        )
    
    def _create_correction_signal(self, correction: Dict, data: pd.DataFrame) -> PatternSignal:
        """Create signal from ABC correction"""
        last_wave = correction['waves'][-1]
        current_price = data['close'].iloc[-1]
        
        # Project next move after ABC completion
        correction_size = correction['waves'][0]['price'] - last_wave['price']
        target = last_wave['price'] + correction_size * 0.618
        stop_loss = last_wave['price'] * 0.98  # 2% below C wave low
        
        return PatternSignal(
            pattern_name='Elliott Wave ABC',
            pattern_type='bullish',  # Bullish after bearish correction
            confidence=correction['confidence'],
            entry_price=current_price,
            target_price=target,
            stop_loss=stop_loss,
            timestamp=datetime.now(),
            pattern_data=correction
        )


class GartleyPatternModel(BasePatternModel):
    """
    Harmonic patterns:
    - XA, AB, BC, CD legs
    - Fibonacci zones
    - PRZ calculation
    - Risk/reward setup
    """
    
    def __init__(self):
        super().__init__(min_pattern_bars=5)
        # Gartley pattern ratios
        self.ratios = {
            'AB': (0.618, 0.618),  # AB = 61.8% of XA
            'BC': (0.382, 0.886),  # BC = 38.2% to 88.6% of AB
            'CD': (1.272, 1.618),  # CD = 127.2% to 161.8% of BC
            'AD': (0.786, 0.786)   # AD = 78.6% of XA
        }
        
    def detect_pattern(self, data: pd.DataFrame, **kwargs) -> Optional[PatternSignal]:
        """Detect Gartley patterns"""
        if not self.validate_data(data, 20):
            return None
        
        pivots = self._find_pivots(data)
        
        if len(pivots) < 5:
            return None
        
        # Look for potential XABCD patterns
        for i in range(len(pivots) - 4):
            pattern = self._check_gartley_pattern(pivots[i:i+5])
            if pattern:
                return self._create_gartley_signal(pattern, data)
        
        return None
    
    def _find_pivots(self, data: pd.DataFrame, window: int = 3) -> List[Dict]:
        """Find pivot points for harmonic patterns"""
        pivots = []
        
        for i in range(window, len(data) - window):
            # High pivot
            if (data['high'].iloc[i] == data['high'].iloc[i-window:i+window+1].max()):
                pivots.append({
                    'index': i,
                    'price': data['high'].iloc[i],
                    'type': 'high'
                })
            # Low pivot
            elif (data['low'].iloc[i] == data['low'].iloc[i-window:i+window+1].min()):
                pivots.append({
                    'index': i,
                    'price': data['low'].iloc[i],
                    'type': 'low'
                })
        
        return sorted(pivots, key=lambda x: x['index'])
    
    def _check_gartley_pattern(self, pivots: List[Dict]) -> Optional[Dict]:
        """Check if pivots form a valid Gartley pattern"""
        if len(pivots) != 5:
            return None
        
        # Label points as X, A, B, C, D
        X, A, B, C, D = pivots
        
        # Check for bullish Gartley (X is low)
        if X['type'] == 'low' and A['type'] == 'high' and B['type'] == 'low':
            # Calculate legs
            XA = A['price'] - X['price']
            AB = A['price'] - B['price']
            BC = C['price'] - B['price']
            CD = C['price'] - D['price']
            AD = A['price'] - D['price']
            
            # Check ratios
            AB_ratio = AB / XA
            BC_ratio = BC / AB
            CD_ratio = CD / BC
            AD_ratio = AD / XA
            
            # Validate Gartley ratios
            if (self._check_ratio(AB_ratio, self.ratios['AB']) and
                self._check_ratio(BC_ratio, self.ratios['BC']) and
                self._check_ratio(CD_ratio, self.ratios['CD']) and
                self._check_ratio(AD_ratio, self.ratios['AD'])):
                
                confidence = self._calculate_harmonic_confidence(
                    AB_ratio, BC_ratio, CD_ratio, AD_ratio
                )
                
                return {
                    'type': 'bullish_gartley',
                    'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D},
                    'ratios': {
                        'AB': AB_ratio,
                        'BC': BC_ratio,
                        'CD': CD_ratio,
                        'AD': AD_ratio
                    },
                    'confidence': confidence,
                    'prz': self._calculate_prz(X, A, B, C, XA)
                }
        
        return None
    
    def _check_ratio(self, actual: float, expected: Tuple[float, float]) -> bool:
        """Check if ratio falls within expected range"""
        return expected[0] <= actual <= expected[1]
    
    def _calculate_harmonic_confidence(self, *ratios) -> float:
        """Calculate pattern confidence based on ratio accuracy"""
        confidence_scores = []
        expected_ratios = [0.618, 0.5, 1.414, 0.786]  # Ideal ratios
        
        for actual, expected in zip(ratios, expected_ratios):
            deviation = abs(actual - expected) / expected
            confidence = max(0, 1 - deviation)
            confidence_scores.append(confidence)
        
        return np.mean(confidence_scores)
    
    def _calculate_prz(self, X: Dict, A: Dict, B: Dict, C: Dict, XA: float) -> Dict:
        """Calculate Potential Reversal Zone"""
        # Multiple Fibonacci projections converge at PRZ
        prz_levels = []
        
        # 78.6% retracement of XA
        prz_levels.append(A['price'] - 0.786 * XA)
        
        # 1.272 extension of BC
        BC = C['price'] - B['price']
        prz_levels.append(C['price'] - 1.272 * BC)
        
        # 161.8% extension of AB
        AB = A['price'] - B['price']
        prz_levels.append(B['price'] - 1.618 * AB)
        
        prz_high = max(prz_levels)
        prz_low = min(prz_levels)
        
        return {
            'high': prz_high,
            'low': prz_low,
            'center': (prz_high + prz_low) / 2,
            'levels': prz_levels
        }
    
    def _create_gartley_signal(self, pattern: Dict, data: pd.DataFrame) -> PatternSignal:
        """Create trading signal from Gartley pattern"""
        D = pattern['points']['D']
        prz = pattern['prz']
        current_price = data['close'].iloc[-1]
        
        # Entry at PRZ center
        entry = prz['center']
        
        # Target at 61.8% of AD move
        A = pattern['points']['A']
        AD = A['price'] - D['price']
        target = D['price'] + 0.618 * AD
        
        # Stop below X
        X = pattern['points']['X']
        stop_loss = X['price'] * 0.99
        
        return PatternSignal(
            pattern_name='Gartley Pattern',
            pattern_type='bullish',
            confidence=pattern['confidence'],
            entry_price=entry,
            target_price=target,
            stop_loss=stop_loss,
            timestamp=datetime.now(),
            pattern_data=pattern
        )


class ButterflyPatternModel(BasePatternModel):
    """
    Extended harmonics:
    - 1.27 XA extension
    - Deep retracements
    - Reversal zones
    - Stop placement
    """
    
    def __init__(self):
        super().__init__(min_pattern_bars=5)
        # Butterfly pattern ratios
        self.ratios = {
            'AB': (0.786, 0.786),   # AB = 78.6% of XA
            'BC': (0.382, 0.886),   # BC = 38.2% to 88.6% of AB
            'CD': (1.618, 2.618),   # CD = 161.8% to 261.8% of BC
            'AD': (1.272, 1.618)    # AD = 127.2% to 161.8% of XA (extension)
        }
        
    def detect_pattern(self, data: pd.DataFrame, **kwargs) -> Optional[PatternSignal]:
        """Detect Butterfly patterns"""
        if not self.validate_data(data, 20):
            return None
        
        pivots = self._find_pivots(data)
        
        if len(pivots) < 5:
            return None
        
        # Look for Butterfly patterns
        for i in range(len(pivots) - 4):
            pattern = self._check_butterfly_pattern(pivots[i:i+5])
            if pattern:
                return self._create_butterfly_signal(pattern, data)
        
        return None
    
    def _find_pivots(self, data: pd.DataFrame, window: int = 3) -> List[Dict]:
        """Find pivot points"""
        # Reuse similar logic from Gartley
        pivots = []
        
        for i in range(window, len(data) - window):
            if (data['high'].iloc[i] == data['high'].iloc[i-window:i+window+1].max()):
                pivots.append({
                    'index': i,
                    'price': data['high'].iloc[i],
                    'type': 'high'
                })
            elif (data['low'].iloc[i] == data['low'].iloc[i-window:i+window+1].min()):
                pivots.append({
                    'index': i,
                    'price': data['low'].iloc[i],
                    'type': 'low'
                })
        
        return sorted(pivots, key=lambda x: x['index'])
    
    def _check_butterfly_pattern(self, pivots: List[Dict]) -> Optional[Dict]:
        """Check if pivots form a valid Butterfly pattern"""
        if len(pivots) != 5:
            return None
        
        X, A, B, C, D = pivots
        
        # Check for bullish Butterfly
        if X['type'] == 'low' and A['type'] == 'high':
            XA = A['price'] - X['price']
            AB = A['price'] - B['price']
            BC = C['price'] - B['price']
            CD = C['price'] - D['price']
            AD = D['price'] - A['price']  # Note: D extends beyond X
            
            # Check ratios
            AB_ratio = AB / XA
            BC_ratio = BC / AB
            CD_ratio = CD / BC
            AD_ratio = abs(AD) / XA  # Extension ratio
            
            # Validate Butterfly ratios
            if (self._check_ratio(AB_ratio, self.ratios['AB']) and
                self._check_ratio(BC_ratio, self.ratios['BC']) and
                self._check_ratio(CD_ratio, self.ratios['CD']) and
                self._check_ratio(AD_ratio, self.ratios['AD'])):
                
                confidence = self._calculate_harmonic_confidence(
                    AB_ratio, BC_ratio, CD_ratio, AD_ratio
                )
                
                return {
                    'type': 'bullish_butterfly',
                    'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D},
                    'ratios': {
                        'AB': AB_ratio,
                        'BC': BC_ratio,
                        'CD': CD_ratio,
                        'AD': AD_ratio
                    },
                    'confidence': confidence,
                    'prz': self._calculate_butterfly_prz(X, A, B, C, XA)
                }
        
        return None
    
    def _check_ratio(self, actual: float, expected: Tuple[float, float]) -> bool:
        """Check if ratio falls within expected range"""
        return expected[0] <= actual <= expected[1]
    
    def _calculate_harmonic_confidence(self, *ratios) -> float:
        """Calculate pattern confidence"""
        confidence_scores = []
        expected_ratios = [0.786, 0.618, 2.0, 1.272]
        
        for actual, expected in zip(ratios, expected_ratios):
            deviation = abs(actual - expected) / expected
            confidence = max(0, 1 - deviation * 0.5)
            confidence_scores.append(confidence)
        
        return np.mean(confidence_scores)
    
    def _calculate_butterfly_prz(self, X: Dict, A: Dict, B: Dict, C: Dict, XA: float) -> Dict:
        """Calculate Potential Reversal Zone for Butterfly"""
        prz_levels = []
        
        # 127.2% extension of XA
        prz_levels.append(X['price'] - 1.272 * XA)
        
        # 161.8% extension of XA
        prz_levels.append(X['price'] - 1.618 * XA)
        
        # 2.618 extension of BC
        BC = C['price'] - B['price']
        prz_levels.append(C['price'] - 2.618 * BC)
        
        prz_high = max(prz_levels)
        prz_low = min(prz_levels)
        
        return {
            'high': prz_high,
            'low': prz_low,
            'center': (prz_high + prz_low) / 2,
            'levels': prz_levels
        }
    
    def _create_butterfly_signal(self, pattern: Dict, data: pd.DataFrame) -> PatternSignal:
        """Create trading signal from Butterfly pattern"""
        D = pattern['points']['D']
        prz = pattern['prz']
        
        # Entry at PRZ
        entry = prz['center']
        
        # Target at 38.2% retracement of AD
        A = pattern['points']['A']
        AD = abs(D['price'] - A['price'])
        target = D['price'] + 0.382 * AD
        
        # Stop beyond 161.8% extension
        X = pattern['points']['X']
        XA = A['price'] - X['price']
        stop_loss = X['price'] - 1.8 * XA
        
        return PatternSignal(
            pattern_name='Butterfly Pattern',
            pattern_type='bullish',
            confidence=pattern['confidence'],
            entry_price=entry,
            target_price=target,
            stop_loss=stop_loss,
            timestamp=datetime.now(),
            pattern_data=pattern
        )


class WyckoffModel(BasePatternModel):
    """
    Market phases:
    - Accumulation
    - Mark-up
    - Distribution
    - Mark-down
    - Volume analysis
    """
    
    def __init__(self):
        super().__init__(min_pattern_bars=20)
        self.phases = ['accumulation', 'markup', 'distribution', 'markdown']
        
    def detect_pattern(self, data: pd.DataFrame, **kwargs) -> Optional[PatternSignal]:
        """Detect Wyckoff phases"""
        if not self.validate_data(data, 50):
            return None
        
        # Identify current phase
        phase = self._identify_phase(data)
        
        if phase:
            return self._create_wyckoff_signal(phase, data)
        
        return None
    
    def _identify_phase(self, data: pd.DataFrame) -> Optional[Dict]:
        """Identify current Wyckoff phase"""
        # Calculate price and volume characteristics
        close = data['close']
        volume = data['volume']
        
        # Price range analysis
        price_range = close.rolling(window=20).max() - close.rolling(window=20).min()
        avg_range = price_range.mean()
        current_range = price_range.iloc[-1]
        
        # Volume analysis
        avg_volume = volume.rolling(window=20).mean()
        recent_volume = volume.tail(5).mean()
        
        # Trend analysis
        sma20 = close.rolling(window=20).mean()
        sma50 = close.rolling(window=50).mean()
        
        current_price = close.iloc[-1]
        
        # Phase identification logic
        if current_range < avg_range * 0.7 and recent_volume > avg_volume.iloc[-1]:
            # Narrow range with increasing volume = potential accumulation
            if self._check_accumulation_signs(data):
                return {
                    'phase': 'accumulation',
                    'confidence': 0.75,
                    'characteristics': {
                        'range_compression': current_range / avg_range,
                        'volume_expansion': recent_volume / avg_volume.iloc[-1],
                        'support_level': close.tail(20).min(),
                        'resistance_level': close.tail(20).max()
                    }
                }
        
        elif current_price > sma20.iloc[-1] > sma50.iloc[-1] and recent_volume > avg_volume.iloc[-1]:
            # Uptrend with volume = markup phase
            return {
                'phase': 'markup',
                'confidence': 0.8,
                'characteristics': {
                    'trend_strength': (current_price - sma50.iloc[-1]) / sma50.iloc[-1],
                    'volume_confirmation': recent_volume / avg_volume.iloc[-1]
                }
            }
        
        elif current_range < avg_range * 0.7 and current_price < sma20.iloc[-1]:
            # Range-bound after uptrend = potential distribution
            if self._check_distribution_signs(data):
                return {
                    'phase': 'distribution',
                    'confidence': 0.7,
                    'characteristics': {
                        'range_compression': current_range / avg_range,
                        'selling_pressure': self._calculate_selling_pressure(data)
                    }
                }
        
        elif current_price < sma20.iloc[-1] < sma50.iloc[-1]:
            # Downtrend = markdown phase
            return {
                'phase': 'markdown',
                'confidence': 0.75,
                'characteristics': {
                    'trend_weakness': (sma50.iloc[-1] - current_price) / sma50.iloc[-1]
                }
            }
        
        return None
    
    def _check_accumulation_signs(self, data: pd.DataFrame) -> bool:
        """Check for accumulation characteristics"""
        close = data['close'].tail(30)
        volume = data['volume'].tail(30)
        
        # Springs and tests
        lows = close.rolling(window=5).min()
        support = lows.min()
        tests = (close < support * 1.02).sum()
        
        # Volume on down days vs up days
        returns = close.pct_change()
        down_volume = volume[returns < 0].mean()
        up_volume = volume[returns > 0].mean()
        
        return tests >= 2 and up_volume > down_volume
    
    def _check_distribution_signs(self, data: pd.DataFrame) -> bool:
        """Check for distribution characteristics"""
        close = data['close'].tail(30)
        volume = data['volume'].tail(30)
        
        # Upthrusts
        highs = close.rolling(window=5).max()
        resistance = highs.max()
        tests = (close > resistance * 0.98).sum()
        
        # Volume analysis
        returns = close.pct_change()
        down_volume = volume[returns < 0].mean()
        up_volume = volume[returns > 0].mean()
        
        return tests >= 2 and down_volume > up_volume
    
    def _calculate_selling_pressure(self, data: pd.DataFrame) -> float:
        """Calculate selling pressure indicator"""
        close = data['close'].tail(20)
        volume = data['volume'].tail(20)
        
        # Volume on down moves
        returns = close.pct_change()
        selling_volume = volume[returns < 0].sum()
        total_volume = volume.sum()
        
        return selling_volume / total_volume if total_volume > 0 else 0
    
    def _create_wyckoff_signal(self, phase: Dict, data: pd.DataFrame) -> PatternSignal:
        """Create signal based on Wyckoff phase"""
        current_price = data['close'].iloc[-1]
        
        if phase['phase'] == 'accumulation':
            # Buy signal at end of accumulation
            entry = current_price
            resistance = phase['characteristics']['resistance_level']
            support = phase['characteristics']['support_level']
            target = resistance + (resistance - support) * 0.618
            stop_loss = support * 0.98
            pattern_type = 'bullish'
            
        elif phase['phase'] == 'markup':
            # Trend following in markup
            entry = current_price
            target = current_price * 1.05  # 5% target
            stop_loss = data['close'].rolling(window=10).min().iloc[-1]
            pattern_type = 'bullish'
            
        elif phase['phase'] == 'distribution':
            # Sell signal in distribution
            entry = current_price
            resistance = data['close'].tail(20).max()
            support = data['close'].tail(20).min()
            target = support - (resistance - support) * 0.382
            stop_loss = resistance * 1.02
            pattern_type = 'bearish'
            
        else:  # markdown
            # Short or stay out
            entry = current_price
            target = current_price * 0.95
            stop_loss = data['close'].rolling(window=10).max().iloc[-1]
            pattern_type = 'bearish'
        
        return PatternSignal(
            pattern_name=f'Wyckoff {phase["phase"].title()}',
            pattern_type=pattern_type,
            confidence=phase['confidence'],
            entry_price=entry,
            target_price=target,
            stop_loss=stop_loss,
            timestamp=datetime.now(),
            pattern_data=phase
        )


class PointFigureModel(BasePatternModel):
    """
    Column patterns:
    - Double tops/bottoms
    - Triple tops/bottoms
    - Catapult patterns
    - Price objectives
    """
    
    def __init__(self, box_size: float = None, reversal_size: int = 3):
        super().__init__(min_pattern_bars=20)
        self.box_size = box_size
        self.reversal_size = reversal_size
        
    def detect_pattern(self, data: pd.DataFrame, **kwargs) -> Optional[PatternSignal]:
        """Detect Point & Figure patterns"""
        if not self.validate_data(data, 30):
            return None
        
        # Convert to P&F chart
        pf_data = self._create_pf_chart(data)
        
        if not pf_data or len(pf_data['columns']) < 3:
            return None
        
        # Look for patterns
        patterns = []
        
        # Double top/bottom
        double_pattern = self._find_double_pattern(pf_data)
        if double_pattern:
            patterns.append(double_pattern)
        
        # Triple top/bottom
        triple_pattern = self._find_triple_pattern(pf_data)
        if triple_pattern:
            patterns.append(triple_pattern)
        
        # Catapult
        catapult = self._find_catapult_pattern(pf_data)
        if catapult:
            patterns.append(catapult)
        
        # Return highest confidence pattern
        if patterns:
            best_pattern = max(patterns, key=lambda x: x['confidence'])
            return self._create_pf_signal(best_pattern, data)
        
        return None
    
    def _create_pf_chart(self, data: pd.DataFrame) -> Dict:
        """Convert price data to Point & Figure format"""
        close = data['close']
        
        # Calculate box size if not provided
        if self.box_size is None:
            atr = self._calculate_atr(data, period=14)
            self.box_size = atr * 0.5
        
        # Initialize P&F data
        pf_columns = []
        current_column = {'type': None, 'boxes': [], 'start_price': None}
        
        # Starting price
        current_price = close.iloc[0]
        current_direction = None
        
        for price in close:
            price_change = price - current_price
            
            if current_direction is None:
                # First column
                if abs(price_change) >= self.box_size:
                    current_direction = 'X' if price_change > 0 else 'O'
                    current_column = {
                        'type': current_direction,
                        'boxes': [current_price],
                        'start_price': current_price
                    }
                    current_price = price
            
            elif current_direction == 'X':  # Rising column
                # Continue rising
                while price >= current_price + self.box_size:
                    current_price += self.box_size
                    current_column['boxes'].append(current_price)
                
                # Check for reversal
                if price <= current_price - (self.box_size * self.reversal_size):
                    # Start new falling column
                    pf_columns.append(current_column)
                    current_direction = 'O'
                    current_column = {
                        'type': 'O',
                        'boxes': [],
                        'start_price': current_price
                    }
                    
                    while price <= current_price - self.box_size:
                        current_price -= self.box_size
                        current_column['boxes'].append(current_price)
            
            else:  # Falling column
                # Continue falling
                while price <= current_price - self.box_size:
                    current_price -= self.box_size
                    current_column['boxes'].append(current_price)
                
                # Check for reversal
                if price >= current_price + (self.box_size * self.reversal_size):
                    # Start new rising column
                    pf_columns.append(current_column)
                    current_direction = 'X'
                    current_column = {
                        'type': 'X',
                        'boxes': [],
                        'start_price': current_price
                    }
                    
                    while price >= current_price + self.box_size:
                        current_price += self.box_size
                        current_column['boxes'].append(current_price)
        
        # Add last column
        if current_column['boxes']:
            pf_columns.append(current_column)
        
        return {
            'columns': pf_columns,
            'box_size': self.box_size,
            'reversal_size': self.reversal_size
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def _find_double_pattern(self, pf_data: Dict) -> Optional[Dict]:
        """Find double top or bottom patterns"""
        columns = pf_data['columns']
        
        if len(columns) < 3:
            return None
        
        # Look for double top (X-O-X with same high)
        for i in range(len(columns) - 2):
            if (columns[i]['type'] == 'X' and 
                columns[i+1]['type'] == 'O' and 
                columns[i+2]['type'] == 'X'):
                
                high1 = max(columns[i]['boxes']) if columns[i]['boxes'] else 0
                high2 = max(columns[i+2]['boxes']) if columns[i+2]['boxes'] else 0
                
                if abs(high1 - high2) < pf_data['box_size']:
                    return {
                        'pattern': 'double_top',
                        'type': 'bearish',
                        'resistance': high1,
                        'support': min(columns[i+1]['boxes']) if columns[i+1]['boxes'] else high1,
                        'confidence': 0.7,
                        'columns': [i, i+1, i+2]
                    }
        
        # Look for double bottom (O-X-O with same low)
        for i in range(len(columns) - 2):
            if (columns[i]['type'] == 'O' and 
                columns[i+1]['type'] == 'X' and 
                columns[i+2]['type'] == 'O'):
                
                low1 = min(columns[i]['boxes']) if columns[i]['boxes'] else float('inf')
                low2 = min(columns[i+2]['boxes']) if columns[i+2]['boxes'] else float('inf')
                
                if abs(low1 - low2) < pf_data['box_size']:
                    return {
                        'pattern': 'double_bottom',
                        'type': 'bullish',
                        'support': low1,
                        'resistance': max(columns[i+1]['boxes']) if columns[i+1]['boxes'] else low1,
                        'confidence': 0.7,
                        'columns': [i, i+1, i+2]
                    }
        
        return None
    
    def _find_triple_pattern(self, pf_data: Dict) -> Optional[Dict]:
        """Find triple top or bottom patterns"""
        columns = pf_data['columns']
        
        if len(columns) < 5:
            return None
        
        # Triple top
        for i in range(len(columns) - 4):
            if (columns[i]['type'] == 'X' and 
                columns[i+2]['type'] == 'X' and 
                columns[i+4]['type'] == 'X'):
                
                highs = []
                for j in [i, i+2, i+4]:
                    if columns[j]['boxes']:
                        highs.append(max(columns[j]['boxes']))
                
                if len(highs) == 3:
                    avg_high = np.mean(highs)
                    if all(abs(h - avg_high) < pf_data['box_size'] for h in highs):
                        return {
                            'pattern': 'triple_top',
                            'type': 'bearish',
                            'resistance': avg_high,
                            'confidence': 0.85,
                            'columns': [i, i+1, i+2, i+3, i+4]
                        }
        
        return None
    
    def _find_catapult_pattern(self, pf_data: Dict) -> Optional[Dict]:
        """Find catapult patterns (triple top breakout)"""
        triple = self._find_triple_pattern(pf_data)
        
        if triple and len(pf_data['columns']) > triple['columns'][-1] + 2:
            last_col_idx = triple['columns'][-1]
            next_col = pf_data['columns'][last_col_idx + 2]
            
            if next_col['type'] == 'X' and next_col['boxes']:
                if max(next_col['boxes']) > triple['resistance'] + pf_data['box_size']:
                    return {
                        'pattern': 'catapult',
                        'type': 'bullish',
                        'breakout_level': triple['resistance'],
                        'confidence': 0.9,
                        'base_pattern': triple
                    }
        
        return None
    
    def _create_pf_signal(self, pattern: Dict, data: pd.DataFrame) -> PatternSignal:
        """Create signal from P&F pattern"""
        current_price = data['close'].iloc[-1]
        
        if pattern['type'] == 'bullish':
            entry = current_price
            
            if 'support' in pattern:
                stop_loss = pattern['support'] - self.box_size * 2
                # Price objective using horizontal count
                pattern_width = len(pattern.get('columns', [])) * self.box_size
                target = pattern.get('resistance', current_price) + pattern_width
            else:
                stop_loss = current_price * 0.95
                target = current_price * 1.1
                
        else:  # bearish
            entry = current_price
            
            if 'resistance' in pattern:
                stop_loss = pattern['resistance'] + self.box_size * 2
                pattern_width = len(pattern.get('columns', [])) * self.box_size
                target = pattern.get('support', current_price) - pattern_width
            else:
                stop_loss = current_price * 1.05
                target = current_price * 0.9
        
        return PatternSignal(
            pattern_name=f"P&F {pattern['pattern'].replace('_', ' ').title()}",
            pattern_type=pattern['type'],
            confidence=pattern['confidence'],
            entry_price=entry,
            target_price=target,
            stop_loss=stop_loss,
            timestamp=datetime.now(),
            pattern_data=pattern
        )


class RenkoModel(BasePatternModel):
    """
    Brick patterns:
    - Trend identification
    - Support/resistance
    - Momentum shifts
    - Clean signals
    """
    
    def __init__(self, brick_size: float = None, use_atr: bool = True):
        super().__init__(min_pattern_bars=20)
        self.brick_size = brick_size
        self.use_atr = use_atr
        
    def detect_pattern(self, data: pd.DataFrame, **kwargs) -> Optional[PatternSignal]:
        """Detect Renko patterns"""
        if not self.validate_data(data, 30):
            return None
        
        # Create Renko chart
        renko_data = self._create_renko_chart(data)
        
        if not renko_data or len(renko_data['bricks']) < 3:
            return None
        
        # Detect patterns
        patterns = []
        
        # Trend patterns
        trend = self._identify_renko_trend(renko_data)
        if trend:
            patterns.append(trend)
        
        # Reversal patterns
        reversal = self._identify_renko_reversal(renko_data)
        if reversal:
            patterns.append(reversal)
        
        # Consolidation breakout
        breakout = self._identify_renko_breakout(renko_data)
        if breakout:
            patterns.append(breakout)
        
        if patterns:
            best_pattern = max(patterns, key=lambda x: x['confidence'])
            return self._create_renko_signal(best_pattern, data)
        
        return None
    
    def _create_renko_chart(self, data: pd.DataFrame) -> Dict:
        """Convert price data to Renko format"""
        close = data['close']
        
        # Calculate brick size
        if self.brick_size is None:
            if self.use_atr:
                atr = self._calculate_atr(data, period=14)
                self.brick_size = atr
            else:
                # Use percentage of price
                self.brick_size = close.mean() * 0.001
        
        # Build Renko bricks
        bricks = []
        current_price = close.iloc[0]
        
        for price in close:
            # Check for new bricks
            while price >= current_price + self.brick_size:
                # Up brick
                bricks.append({
                    'type': 'up',
                    'open': current_price,
                    'close': current_price + self.brick_size,
                    'index': len(bricks)
                })
                current_price += self.brick_size
            
            while price <= current_price - self.brick_size:
                # Down brick
                bricks.append({
                    'type': 'down',
                    'open': current_price,
                    'close': current_price - self.brick_size,
                    'index': len(bricks)
                })
                current_price -= self.brick_size
        
        return {
            'bricks': bricks,
            'brick_size': self.brick_size,
            'last_price': close.iloc[-1]
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def _identify_renko_trend(self, renko_data: Dict) -> Optional[Dict]:
        """Identify trend patterns in Renko"""
        bricks = renko_data['bricks']
        
        if len(bricks) < 5:
            return None
        
        # Count consecutive bricks
        last_5 = bricks[-5:]
        up_count = sum(1 for b in last_5 if b['type'] == 'up')
        down_count = sum(1 for b in last_5 if b['type'] == 'down')
        
        if up_count >= 4:
            # Strong uptrend
            return {
                'pattern': 'strong_uptrend',
                'type': 'bullish',
                'strength': up_count / 5,
                'confidence': 0.8,
                'consecutive_bricks': up_count
            }
        elif down_count >= 4:
            # Strong downtrend
            return {
                'pattern': 'strong_downtrend',
                'type': 'bearish',
                'strength': down_count / 5,
                'confidence': 0.8,
                'consecutive_bricks': down_count
            }
        
        return None
    
    def _identify_renko_reversal(self, renko_data: Dict) -> Optional[Dict]:
        """Identify reversal patterns"""
        bricks = renko_data['bricks']
        
        if len(bricks) < 3:
            return None
        
        # Look for color change after trend
        last_3 = bricks[-3:]
        
        # Bullish reversal: down-down-up
        if (last_3[0]['type'] == 'down' and 
            last_3[1]['type'] == 'down' and 
            last_3[2]['type'] == 'up'):
            
            # Check for prior downtrend
            prior_trend = bricks[-10:-3] if len(bricks) >= 10 else bricks[:-3]
            down_ratio = sum(1 for b in prior_trend if b['type'] == 'down') / len(prior_trend)
            
            if down_ratio > 0.6:
                return {
                    'pattern': 'bullish_reversal',
                    'type': 'bullish',
                    'confidence': 0.7 + (down_ratio - 0.6),
                    'prior_trend_strength': down_ratio
                }
        
        # Bearish reversal: up-up-down
        elif (last_3[0]['type'] == 'up' and 
              last_3[1]['type'] == 'up' and 
              last_3[2]['type'] == 'down'):
            
            prior_trend = bricks[-10:-3] if len(bricks) >= 10 else bricks[:-3]
            up_ratio = sum(1 for b in prior_trend if b['type'] == 'up') / len(prior_trend)
            
            if up_ratio > 0.6:
                return {
                    'pattern': 'bearish_reversal',
                    'type': 'bearish',
                    'confidence': 0.7 + (up_ratio - 0.6),
                    'prior_trend_strength': up_ratio
                }
        
        return None
    
    def _identify_renko_breakout(self, renko_data: Dict) -> Optional[Dict]:
        """Identify consolidation breakouts"""
        bricks = renko_data['bricks']
        
        if len(bricks) < 10:
            return None
        
        # Look for consolidation (alternating bricks)
        consolidation_zone = bricks[-10:-2]
        changes = sum(1 for i in range(1, len(consolidation_zone)) 
                     if consolidation_zone[i]['type'] != consolidation_zone[i-1]['type'])
        
        # High number of changes indicates consolidation
        if changes >= 5:
            # Check for breakout
            last_2 = bricks[-2:]
            
            if last_2[0]['type'] == last_2[1]['type']:
                # Breakout confirmed
                return {
                    'pattern': 'consolidation_breakout',
                    'type': 'bullish' if last_2[0]['type'] == 'up' else 'bearish',
                    'confidence': 0.75,
                    'consolidation_changes': changes,
                    'breakout_direction': last_2[0]['type']
                }
        
        return None
    
    def _create_renko_signal(self, pattern: Dict, data: pd.DataFrame) -> PatternSignal:
        """Create signal from Renko pattern"""
        current_price = data['close'].iloc[-1]
        brick_size = self.brick_size
        
        if pattern['type'] == 'bullish':
            entry = current_price
            stop_loss = current_price - brick_size * 2
            
            if 'strong' in pattern['pattern']:
                # Strong trend, larger target
                target = current_price + brick_size * 5
            else:
                # Reversal or breakout
                target = current_price + brick_size * 3
                
        else:  # bearish
            entry = current_price
            stop_loss = current_price + brick_size * 2
            
            if 'strong' in pattern['pattern']:
                target = current_price - brick_size * 5
            else:
                target = current_price - brick_size * 3
        
        return PatternSignal(
            pattern_name=f"Renko {pattern['pattern'].replace('_', ' ').title()}",
            pattern_type=pattern['type'],
            confidence=pattern['confidence'],
            entry_price=entry,
            target_price=target,
            stop_loss=stop_loss,
            timestamp=datetime.now(),
            pattern_data={
                **pattern,
                'brick_size': brick_size
            }
        )