#!/usr/bin/env python3
"""
Enhanced Technical Indicators Trading Models
Implements RSI, MACD, Bollinger, Stochastic, and more
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
class RSIModel:
    """RSI-based trading model"""
    
    period: int = 14
    oversold_threshold: float = 30
    overbought_threshold: float = 70
    use_divergence: bool = True
    confirmation_periods: int = 2
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on RSI"""
        # Calculate RSI
        rsi = self._calculate_rsi(data['close'], self.period)
        
        signals = pd.Series(0, index=data.index)
        
        # Basic RSI signals
        oversold = rsi < self.oversold_threshold
        overbought = rsi > self.overbought_threshold
        
        # Look for RSI exits from extreme zones
        for i in range(self.confirmation_periods, len(signals)):
            # Buy signal: RSI was oversold and now crossing above
            if (rsi.iloc[i] > self.oversold_threshold and 
                oversold.iloc[i-self.confirmation_periods:i].any()):
                signals.iloc[i] = 1
                
            # Sell signal: RSI was overbought and now crossing below
            elif (rsi.iloc[i] < self.overbought_threshold and 
                  overbought.iloc[i-self.confirmation_periods:i].any()):
                signals.iloc[i] = -1
        
        # Divergence detection
        if self.use_divergence and len(data) > 50:
            divergence_signals = self._detect_divergence(data['close'], rsi)
            signals = signals + divergence_signals
            signals = signals.clip(-1, 1)
        
        return signals.astype(int)
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _detect_divergence(self, prices: pd.Series, rsi: pd.Series) -> pd.Series:
        """Detect bullish and bearish divergences"""
        signals = pd.Series(0, index=prices.index)
        
        # Find local extrema
        price_highs = (prices.shift(1) < prices) & (prices.shift(-1) < prices)
        price_lows = (prices.shift(1) > prices) & (prices.shift(-1) > prices)
        
        rsi_highs = (rsi.shift(1) < rsi) & (rsi.shift(-1) < rsi)
        rsi_lows = (rsi.shift(1) > rsi) & (rsi.shift(-1) > rsi)
        
        # Look for divergences in recent history
        lookback = 20
        
        for i in range(lookback, len(prices)):
            # Bearish divergence: price makes higher high, RSI makes lower high
            if price_highs.iloc[i]:
                prev_highs = price_highs.iloc[i-lookback:i]
                if prev_highs.any():
                    prev_high_idx = prev_highs[prev_highs].index[-1]
                    if (prices.iloc[i] > prices.loc[prev_high_idx] and 
                        rsi.iloc[i] < rsi.loc[prev_high_idx]):
                        signals.iloc[i] = -1
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            if price_lows.iloc[i]:
                prev_lows = price_lows.iloc[i-lookback:i]
                if prev_lows.any():
                    prev_low_idx = prev_lows[prev_lows].index[-1]
                    if (prices.iloc[i] < prices.loc[prev_low_idx] and 
                        rsi.iloc[i] > rsi.loc[prev_low_idx]):
                        signals.iloc[i] = 1
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'oversold_threshold': self.oversold_threshold,
            'overbought_threshold': self.overbought_threshold,
            'use_divergence': self.use_divergence
        }


@dataclass
class MACDModel:
    """MACD-based trading model"""
    
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    use_histogram: bool = True
    use_zero_cross: bool = True
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on MACD"""
        # Calculate MACD
        macd_line, signal_line, histogram = self._calculate_macd(
            data['close'], 
            self.fast_period, 
            self.slow_period, 
            self.signal_period
        )
        
        signals = pd.Series(0, index=data.index)
        
        # MACD line crosses signal line
        macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        signals[macd_cross_up] = 1
        signals[macd_cross_down] = -1
        
        # Zero line crosses
        if self.use_zero_cross:
            zero_cross_up = (macd_line > 0) & (macd_line.shift(1) <= 0)
            zero_cross_down = (macd_line < 0) & (macd_line.shift(1) >= 0)
            
            # Only use zero crosses as confirmation
            signals[(signals == 0) & zero_cross_up] = 1
            signals[(signals == 0) & zero_cross_down] = -1
        
        # Histogram reversals
        if self.use_histogram:
            hist_reversal_up = (histogram > histogram.shift(1)) & (histogram.shift(1) < histogram.shift(2))
            hist_reversal_down = (histogram < histogram.shift(1)) & (histogram.shift(1) > histogram.shift(2))
            
            # Use histogram for early signals
            signals[(signals == 0) & hist_reversal_up & (histogram < 0)] = 1
            signals[(signals == 0) & hist_reversal_down & (histogram > 0)] = -1
        
        return signals.astype(int)
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'use_histogram': self.use_histogram,
            'use_zero_cross': self.use_zero_cross
        }


@dataclass 
class BollingerBreakoutModel:
    """Bollinger Bands breakout trading model"""
    
    period: int = 20
    num_std: float = 2.0
    use_squeeze: bool = True
    volume_confirmation: bool = True
    breakout_threshold: float = 0.02  # 2% beyond band
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on Bollinger Bands breakouts"""
        # Calculate Bollinger Bands
        sma = data['close'].rolling(self.period).mean()
        std = data['close'].rolling(self.period).std()
        
        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)
        band_width = (upper_band - lower_band) / sma
        
        signals = pd.Series(0, index=data.index)
        
        # Breakout signals
        upper_breakout = data['close'] > upper_band * (1 + self.breakout_threshold)
        lower_breakout = data['close'] < lower_band * (1 - self.breakout_threshold)
        
        # Volume confirmation
        if self.volume_confirmation and 'volume' in data.columns:
            volume_surge = data['volume'] > data['volume'].rolling(20).mean() * 1.5
        else:
            volume_surge = pd.Series(True, index=data.index)
        
        # Squeeze detection
        if self.use_squeeze:
            squeeze = band_width < band_width.rolling(50).quantile(0.2)
            
            # Buy on squeeze release with upward breakout
            squeeze_release_up = (~squeeze) & (squeeze.shift(1)) & upper_breakout
            signals[squeeze_release_up & volume_surge] = 1
            
            # Sell on squeeze release with downward breakout
            squeeze_release_down = (~squeeze) & (squeeze.shift(1)) & lower_breakout
            signals[squeeze_release_down & volume_surge] = -1
        
        # Regular breakouts (when not using squeeze)
        signals[(upper_breakout & volume_surge & (signals == 0))] = 1
        signals[(lower_breakout & volume_surge & (signals == 0))] = -1
        
        # Band touch reversals
        band_touch_upper = (data['high'] >= upper_band) & (data['close'] < upper_band)
        band_touch_lower = (data['low'] <= lower_band) & (data['close'] > lower_band)
        
        signals[(band_touch_upper & (signals == 0))] = -1
        signals[(band_touch_lower & (signals == 0))] = 1
        
        return signals.astype(int)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'num_std': self.num_std,
            'use_squeeze': self.use_squeeze,
            'volume_confirmation': self.volume_confirmation,
            'breakout_threshold': self.breakout_threshold
        }


@dataclass
class StochasticModel:
    """Stochastic oscillator trading model"""
    
    k_period: int = 14
    d_period: int = 3
    smooth_k: int = 3
    oversold: float = 20
    overbought: float = 80
    use_divergence: bool = True
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on Stochastic"""
        # Calculate Stochastic
        k_percent, d_percent = self._calculate_stochastic(
            data['high'], 
            data['low'], 
            data['close'],
            self.k_period,
            self.d_period,
            self.smooth_k
        )
        
        signals = pd.Series(0, index=data.index)
        
        # Oversold/Overbought crosses
        k_cross_oversold_up = (k_percent > self.oversold) & (k_percent.shift(1) <= self.oversold)
        k_cross_overbought_down = (k_percent < self.overbought) & (k_percent.shift(1) >= self.overbought)
        
        # K crosses D
        k_cross_d_up = (k_percent > d_percent) & (k_percent.shift(1) <= d_percent.shift(1))
        k_cross_d_down = (k_percent < d_percent) & (k_percent.shift(1) >= d_percent.shift(1))
        
        # Combined signals
        signals[k_cross_oversold_up & k_cross_d_up] = 1
        signals[k_cross_overbought_down & k_cross_d_down] = -1
        
        # Divergence
        if self.use_divergence:
            divergence_signals = self._detect_stoch_divergence(
                data['close'], k_percent, self.oversold, self.overbought
            )
            signals = signals + divergence_signals
            signals = signals.clip(-1, 1)
        
        return signals.astype(int)
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                             k_period: int, d_period: int, smooth_k: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic K% and D%"""
        # Calculate raw K%
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        
        k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 1)
        
        # Smooth K%
        k_percent = k_raw.rolling(smooth_k).mean()
        
        # Calculate D%
        d_percent = k_percent.rolling(d_period).mean()
        
        return k_percent, d_percent
    
    def _detect_stoch_divergence(self, prices: pd.Series, stoch: pd.Series, 
                                oversold: float, overbought: float) -> pd.Series:
        """Detect divergences between price and Stochastic"""
        signals = pd.Series(0, index=prices.index)
        
        # Only look for divergences in extreme zones
        for i in range(20, len(prices)):
            window_prices = prices.iloc[i-20:i]
            window_stoch = stoch.iloc[i-20:i]
            
            # Bullish divergence in oversold zone
            if stoch.iloc[i] < oversold:
                price_lows = (window_prices == window_prices.min())
                if price_lows.any() and i - price_lows.idxmax() > 5:
                    if (prices.iloc[i] < window_prices.min() and 
                        stoch.iloc[i] > window_stoch[price_lows].min()):
                        signals.iloc[i] = 1
            
            # Bearish divergence in overbought zone
            if stoch.iloc[i] > overbought:
                price_highs = (window_prices == window_prices.max())
                if price_highs.any() and i - price_highs.idxmax() > 5:
                    if (prices.iloc[i] > window_prices.max() and 
                        stoch.iloc[i] < window_stoch[price_highs].max()):
                        signals.iloc[i] = -1
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'k_period': self.k_period,
            'd_period': self.d_period,
            'smooth_k': self.smooth_k,
            'oversold': self.oversold,
            'overbought': self.overbought,
            'use_divergence': self.use_divergence
        }


@dataclass
class WilliamsRModel:
    """Williams %R trading model"""
    
    period: int = 14
    oversold: float = -80
    overbought: float = -20
    smoothing_period: int = 3
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on Williams %R"""
        # Calculate Williams %R
        williams_r = self._calculate_williams_r(
            data['high'], 
            data['low'], 
            data['close'], 
            self.period
        )
        
        # Smooth the indicator
        williams_r_smooth = williams_r.rolling(self.smoothing_period).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # Signal generation
        for i in range(1, len(signals)):
            # Buy signal: Exit from oversold
            if (williams_r_smooth.iloc[i] > self.oversold and 
                williams_r_smooth.iloc[i-1] <= self.oversold):
                signals.iloc[i] = 1
            
            # Sell signal: Exit from overbought  
            elif (williams_r_smooth.iloc[i] < self.overbought and 
                  williams_r_smooth.iloc[i-1] >= self.overbought):
                signals.iloc[i] = -1
        
        return signals.astype(int)
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, 
                             close: pd.Series, period: int) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, 1)
        
        return williams_r
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'oversold': self.oversold,
            'overbought': self.overbought,
            'smoothing_period': self.smoothing_period
        }


@dataclass
class CCIEnsembleModel:
    """Commodity Channel Index ensemble model"""
    
    periods: List[int] = None
    overbought: float = 100
    oversold: float = -100
    extreme_overbought: float = 200
    extreme_oversold: float = -200
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = [14, 20, 50]
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on CCI ensemble"""
        signals = pd.Series(0, index=data.index)
        
        # Calculate CCI for each period
        cci_signals = []
        
        for period in self.periods:
            cci = self._calculate_cci(data, period)
            period_signals = pd.Series(0, index=data.index)
            
            # Basic CCI signals
            for i in range(1, len(cci)):
                # Buy signals
                if (cci.iloc[i] > self.oversold and cci.iloc[i-1] <= self.oversold):
                    period_signals.iloc[i] = 1
                elif cci.iloc[i] < self.extreme_oversold:
                    period_signals.iloc[i] = 1
                
                # Sell signals
                elif (cci.iloc[i] < self.overbought and cci.iloc[i-1] >= self.overbought):
                    period_signals.iloc[i] = -1
                elif cci.iloc[i] > self.extreme_overbought:
                    period_signals.iloc[i] = -1
            
            cci_signals.append(period_signals)
        
        # Ensemble: majority vote
        ensemble_signals = pd.concat(cci_signals, axis=1).sum(axis=1)
        signals[ensemble_signals >= 2] = 1
        signals[ensemble_signals <= -2] = -1
        
        return signals.astype(int)
    
    def _calculate_cci(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(period).mean()
        mean_deviation = typical_price.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'periods': self.periods,
            'overbought': self.overbought,
            'oversold': self.oversold,
            'extreme_overbought': self.extreme_overbought,
            'extreme_oversold': self.extreme_oversold
        }


@dataclass
class ParabolicSARModel:
    """Parabolic SAR trading model"""
    
    initial_af: float = 0.02
    max_af: float = 0.2
    af_increment: float = 0.02
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on Parabolic SAR"""
        sar, trend = self._calculate_parabolic_sar(data)
        
        signals = pd.Series(0, index=data.index)
        
        # Trend changes
        for i in range(1, len(trend)):
            if trend.iloc[i] == 1 and trend.iloc[i-1] == -1:
                signals.iloc[i] = 1  # Buy on uptrend start
            elif trend.iloc[i] == -1 and trend.iloc[i-1] == 1:
                signals.iloc[i] = -1  # Sell on downtrend start
        
        return signals.astype(int)
    
    def _calculate_parabolic_sar(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Parabolic SAR and trend"""
        high = data['high']
        low = data['low']
        
        sar = pd.Series(index=data.index, dtype=float)
        trend = pd.Series(index=data.index, dtype=int)
        
        # Initialize
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1
        ep = high.iloc[0]  # Extreme point
        af = self.initial_af
        
        for i in range(1, len(data)):
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                
                # Check for reversal
                if low.iloc[i] <= sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = self.initial_af
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + self.af_increment, self.max_af)
                        
            else:  # Downtrend
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                
                # Check for reversal
                if high.iloc[i] >= sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = self.initial_af
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + self.af_increment, self.max_af)
        
        return sar, trend
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'initial_af': self.initial_af,
            'max_af': self.max_af,
            'af_increment': self.af_increment
        }


@dataclass
class EMACrossoverModel:
    """Exponential Moving Average crossover model"""
    
    fast_period: int = 12
    slow_period: int = 26
    signal_smoothing: int = 9
    use_triple_ema: bool = False
    trend_period: int = 50
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on EMA crossovers"""
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # Basic crossover signals
        golden_cross = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        death_cross = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        
        signals[golden_cross] = 1
        signals[death_cross] = -1
        
        # Triple EMA confirmation
        if self.use_triple_ema:
            ema_trend = data['close'].ewm(span=self.trend_period, adjust=False).mean()
            
            # Only take signals in direction of trend
            signals[(signals == 1) & (data['close'] < ema_trend)] = 0
            signals[(signals == -1) & (data['close'] > ema_trend)] = 0
        
        # Signal line smoothing
        if self.signal_smoothing > 0:
            signal_line = signals.rolling(self.signal_smoothing).mean()
            signals = pd.Series(0, index=data.index)
            signals[signal_line > 0.5] = 1
            signals[signal_line < -0.5] = -1
        
        return signals.astype(int)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_smoothing': self.signal_smoothing,
            'use_triple_ema': self.use_triple_ema,
            'trend_period': self.trend_period
        }


@dataclass
class TechnicalIndicatorEnsemble:
    """Ensemble of all technical indicators"""
    
    def __init__(self):
        self.models = {
            'rsi': RSIModel(),
            'macd': MACDModel(),
            'bollinger': BollingerBreakoutModel(),
            'stochastic': StochasticModel(),
            'williams_r': WilliamsRModel(),
            'cci': CCIEnsembleModel(),
            'sar': ParabolicSARModel(),
            'ema': EMACrossoverModel()
        }
        
        self.weights = {
            'rsi': 0.15,
            'macd': 0.15,
            'bollinger': 0.15,
            'stochastic': 0.10,
            'williams_r': 0.10,
            'cci': 0.10,
            'sar': 0.15,
            'ema': 0.10
        }
    
    def predict(self, data: pd.DataFrame, voting_threshold: float = 0.3) -> pd.Series:
        """Generate ensemble signals from all indicators"""
        all_signals = {}
        
        # Get signals from each model
        for name, model in self.models.items():
            try:
                signals = model.predict(data)
                all_signals[name] = signals
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                all_signals[name] = pd.Series(0, index=data.index)
        
        # Weighted voting
        weighted_sum = pd.Series(0.0, index=data.index)
        
        for name, signals in all_signals.items():
            weighted_sum += signals * self.weights.get(name, 0.1)
        
        # Generate final signals
        final_signals = pd.Series(0, index=data.index)
        final_signals[weighted_sum > voting_threshold] = 1
        final_signals[weighted_sum < -voting_threshold] = -1
        
        return final_signals.astype(int)
    
    def get_individual_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get signals from each indicator separately"""
        results = {}
        for name, model in self.models.items():
            try:
                results[name] = model.predict(data)
            except Exception as e:
                logger.error(f"Model {name} failed: {e}")
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