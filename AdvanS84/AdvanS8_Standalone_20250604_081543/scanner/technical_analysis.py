import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """
    Resource-efficient technical analysis with institutional-grade indicators
    and multi-timeframe momentum calculations using native Python.
    """
    
    def __init__(self):
        """Initialize technical analyzer with institutional parameters"""
        self.timeframes = {
            'short': [3, 5],
            'medium': [10, 20], 
            'long': [20, 50],
            'all': [3, 5, 10, 20, 50]
        }
        
        # Institutional-grade parameters
        self.rsi_periods = [2, 14]
        self.macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
        self.bb_period = 20
        self.bb_std = 2
        self.stoch_params = {'k_period': 14, 'd_period': 3}
        
    def analyze(self, data: pd.DataFrame, timeframe: str = 'all') -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis
        
        Args:
            data: OHLCV DataFrame
            timeframe: Analysis timeframe strategy
            
        Returns:
            Dictionary of technical scores and indicators
        """
        try:
            if data is None or data.empty or len(data) < 50:
                return self._empty_result()
            
            results = {}
            
            # Multi-timeframe momentum analysis
            momentum_scores = self._calculate_momentum_analysis(data, timeframe)
            results.update(momentum_scores)
            
            # RSI analysis (2-period and 14-period)
            rsi_scores = self._calculate_rsi_analysis(data)
            results.update(rsi_scores)
            
            # MACD analysis with histogram
            macd_scores = self._calculate_macd_analysis(data)
            results.update(macd_scores)
            
            # Bollinger Bands position and width
            bb_scores = self._calculate_bollinger_analysis(data)
            results.update(bb_scores)
            
            # Stochastic oscillator
            stoch_scores = self._calculate_stochastic_analysis(data)
            results.update(stoch_scores)
            
            # Williams %R
            williams_scores = self._calculate_williams_analysis(data)
            results.update(williams_scores)
            
            # CCI (Commodity Channel Index)
            cci_scores = self._calculate_cci_analysis(data)
            results.update(cci_scores)
            
            # ADX with Directional Movement
            adx_scores = self._calculate_adx_analysis(data)
            results.update(adx_scores)
            
            # Volume analysis
            volume_scores = self._calculate_volume_analysis(data)
            results.update(volume_scores)
            
            # Price acceleration (second derivative)
            acceleration_scores = self._calculate_acceleration_analysis(data)
            results.update(acceleration_scores)
            
            # Composite technical score
            results['composite_score'] = self._calculate_composite_score(results, timeframe)
            
            return results
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return self._empty_result()
    
    def _calculate_momentum_analysis(self, data: pd.DataFrame, timeframe: str) -> Dict[str, float]:
        """Calculate multi-timeframe momentum with quality scoring"""
        try:
            results = {}
            periods = self.timeframes.get(timeframe, self.timeframes['all'])
            
            for period in periods:
                if len(data) > period:
                    # Price momentum
                    momentum = ((data['close'].iloc[-1] / data['close'].iloc[-period-1]) - 1) * 100
                    results[f'momentum_{period}d'] = momentum
                    
                    # Momentum persistence (% positive days)
                    returns = data['close'].pct_change().tail(period)
                    persistence = (returns > 0).sum() / len(returns) * 100
                    results[f'momentum_persistence_{period}d'] = persistence
            
            # Momentum quality composite
            if results:
                momentum_values = [v for k, v in results.items() if 'momentum_' in k and 'd' in k]
                persistence_values = [v for k, v in results.items() if 'persistence_' in k]
                
                if momentum_values:
                    results['momentum_quality'] = np.mean(momentum_values) * (np.mean(persistence_values) / 100)
            
            return results
            
        except Exception as e:
            logger.error(f"Momentum analysis failed: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using native Python"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return pd.Series()
    
    def _calculate_rsi_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate multi-timeframe RSI analysis"""
        try:
            results = {}
            
            for period in self.rsi_periods:
                if len(data) > period * 2:
                    rsi = self._calculate_rsi(data['close'], period)
                    current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
                    results[f'rsi_{period}'] = current_rsi
                    
                    # Enhanced RSI with trend confirmation for 70%+ accuracy
                    if period == 2:
                        # Calculate trend confirmation
                        ema_5 = data['close'].ewm(span=5).mean().iloc[-1]
                        ema_20 = data['close'].ewm(span=20).mean().iloc[-1]
                        trend_confirmed = ema_5 > ema_20
                        
                        if current_rsi > 95:
                            results['rsi_2_signal'] = 'EXTREME_OVERBOUGHT'
                        elif current_rsi > 85:
                            results['rsi_2_signal'] = 'REJECT_OVERBOUGHT'  # Filter out overbought
                        elif current_rsi < 10 and trend_confirmed:
                            results['rsi_2_signal'] = 'EXTREME_OVERSOLD_BUY'
                        elif current_rsi < 25 and trend_confirmed:
                            results['rsi_2_signal'] = 'OVERSOLD_BUY'
                        elif current_rsi < 30 and not trend_confirmed:
                            results['rsi_2_signal'] = 'OVERSOLD_NO_TREND'  # Weaker signal
                        else:
                            results['rsi_2_signal'] = 'NEUTRAL'
                        
                        results['trend_confirmed'] = trend_confirmed
                        results['ema_5'] = ema_5
                        results['ema_20'] = ema_20
                    
                    elif period == 14:
                        # Apply 70%+ accuracy threshold - reject overbought conditions
                        if current_rsi > 80:
                            results['rsi_14_signal'] = 'REJECT_OVERBOUGHT'
                        elif current_rsi < 25:
                            results['rsi_14_signal'] = 'STRONG_OVERSOLD_BUY'
                        elif current_rsi < 35:
                            results['rsi_14_signal'] = 'OVERSOLD_BUY'
                        else:
                            results['rsi_14_signal'] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            logger.error(f"RSI analysis failed: {e}")
            return {}
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD using native Python"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=signal).mean()
            macd_histogram = macd - macd_signal
            
            return macd, macd_signal, macd_histogram
            
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return pd.Series(), pd.Series(), pd.Series()
    
    def _calculate_macd_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Professional MACD with histogram analysis"""
        try:
            if len(data) < self.macd_params['slow'] + 10:
                return {}
            
            # Calculate MACD
            macd, macd_signal, macd_hist = self._calculate_macd(
                data['close'],
                fast=self.macd_params['fast'],
                slow=self.macd_params['slow'],
                signal=self.macd_params['signal']
            )
            
            results = {
                'macd': macd.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'macd_histogram': macd_hist.iloc[-1]
            }
            
            # MACD trend analysis
            if len(macd_hist) >= 3:
                if macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3]:
                    results['macd_trend'] = 'BULLISH_ACCELERATION'
                elif macd_hist.iloc[-1] < macd_hist.iloc[-2] < macd_hist.iloc[-3]:
                    results['macd_trend'] = 'BEARISH_ACCELERATION'
                elif macd_hist.iloc[-1] > 0 and macd.iloc[-1] > macd_signal.iloc[-1]:
                    results['macd_trend'] = 'BULLISH'
                elif macd_hist.iloc[-1] < 0 and macd.iloc[-1] < macd_signal.iloc[-1]:
                    results['macd_trend'] = 'BEARISH'
                else:
                    results['macd_trend'] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            logger.error(f"MACD analysis failed: {e}")
            return {}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands using native Python"""
        try:
            rolling_mean = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            
            return upper_band, rolling_mean, lower_band
            
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            return pd.Series(), pd.Series(), pd.Series()
    
    def _calculate_bollinger_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Bollinger Band position and width analysis"""
        try:
            if len(data) < self.bb_period + 5:
                return {}
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                data['close'],
                period=self.bb_period,
                std_dev=self.bb_std
            )
            
            current_price = data['close'].iloc[-1]
            
            # Band position (0 = lower band, 1 = upper band)
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Band width (volatility measure)
            bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1] * 100
            
            results = {
                'bb_position': bb_position,
                'bb_width': bb_width,
                'bb_upper': bb_upper.iloc[-1],
                'bb_middle': bb_middle.iloc[-1],
                'bb_lower': bb_lower.iloc[-1]
            }
            
            # Bollinger Band signals
            if bb_position > 0.8:
                results['bb_signal'] = 'NEAR_UPPER_BAND'
            elif bb_position < 0.2:
                results['bb_signal'] = 'NEAR_LOWER_BAND'
            elif bb_width < 10:  # Squeeze
                results['bb_signal'] = 'SQUEEZE'
            else:
                results['bb_signal'] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            logger.error(f"Bollinger Bands analysis failed: {e}")
            return {}
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator using native Python"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return k_percent, d_percent
            
        except Exception as e:
            logger.error(f"Stochastic calculation failed: {e}")
            return pd.Series(), pd.Series()
    
    def _calculate_stochastic_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Stochastic Oscillator (K% and D%)"""
        try:
            if len(data) < self.stoch_params['k_period'] + 5:
                return {}
            
            slowk, slowd = self._calculate_stochastic(
                data['high'],
                data['low'],
                data['close'],
                k_period=self.stoch_params['k_period'],
                d_period=self.stoch_params['d_period']
            )
            
            results = {
                'stoch_k': slowk.iloc[-1],
                'stoch_d': slowd.iloc[-1]
            }
            
            # Stochastic signals
            if slowk.iloc[-1] > 80 and slowd.iloc[-1] > 80:
                results['stoch_signal'] = 'OVERBOUGHT'
            elif slowk.iloc[-1] < 20 and slowd.iloc[-1] < 20:
                results['stoch_signal'] = 'OVERSOLD'
            elif len(slowk) >= 2 and slowk.iloc[-1] > slowd.iloc[-1] and slowk.iloc[-2] <= slowd.iloc[-2]:
                results['stoch_signal'] = 'BULLISH_CROSSOVER'
            elif len(slowk) >= 2 and slowk.iloc[-1] < slowd.iloc[-1] and slowk.iloc[-2] >= slowd.iloc[-2]:
                results['stoch_signal'] = 'BEARISH_CROSSOVER'
            else:
                results['stoch_signal'] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            logger.error(f"Stochastic analysis failed: {e}")
            return {}
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R using native Python"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            return williams_r
            
        except Exception as e:
            logger.error(f"Williams %R calculation failed: {e}")
            return pd.Series()
    
    def _calculate_williams_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Williams %R analysis"""
        try:
            if len(data) < 14:
                return {}
            
            williams_r = self._calculate_williams_r(
                data['high'],
                data['low'],
                data['close'],
                period=14
            )
            
            current_wr = williams_r.iloc[-1]
            
            results = {'williams_r': current_wr}
            
            # Williams %R signals
            if current_wr > -20:
                results['williams_signal'] = 'OVERBOUGHT'
            elif current_wr < -80:
                results['williams_signal'] = 'OVERSOLD'
            else:
                results['williams_signal'] = 'NEUTRAL'
            
            return results
            
        except Exception as e:
            logger.error(f"Williams %R analysis failed: {e}")
            return {}
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index using native Python"""
        try:
            typical_price = (high + low + close) / 3
            sma = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma) / (0.015 * mean_deviation)
            
            return cci
            
        except Exception as e:
            logger.error(f"CCI calculation failed: {e}")
            return pd.Series()
    
    def _calculate_cci_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Commodity Channel Index analysis"""
        try:
            if len(data) < 20:
                return {}
            
            cci = self._calculate_cci(
                data['high'],
                data['low'],
                data['close'],
                period=20
            )
            
            current_cci = cci.iloc[-1]
            
            results = {'cci': current_cci}
            
            # CCI signals
            if current_cci > 100:
                results['cci_signal'] = 'STRONG_UPTREND'
            elif current_cci < -100:
                results['cci_signal'] = 'STRONG_DOWNTREND'
            elif current_cci > 0:
                results['cci_signal'] = 'BULLISH'
            else:
                results['cci_signal'] = 'BEARISH'
            
            return results
            
        except Exception as e:
            logger.error(f"CCI analysis failed: {e}")
            return {}
    
    def _calculate_adx_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """ADX with Directional Movement analysis"""
        try:
            if len(data) < 20:
                return {}
            
            # Simplified ADX calculation
            high_diff = data['high'].diff()
            low_diff = data['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            tr = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)),
                                    abs(data['low'] - data['close'].shift(1))))
            
            # Simple moving averages for demonstration
            plus_di = (pd.Series(plus_dm).rolling(14).sum() / pd.Series(tr).rolling(14).sum()) * 100
            minus_di = (pd.Series(minus_dm).rolling(14).sum() / pd.Series(tr).rolling(14).sum()) * 100
            
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = dx.rolling(14).mean()
            
            results = {
                'adx': adx.iloc[-1] if len(adx) > 0 else 25,
                'plus_di': plus_di.iloc[-1] if len(plus_di) > 0 else 25,
                'minus_di': minus_di.iloc[-1] if len(minus_di) > 0 else 25
            }
            
            # ADX trend strength
            adx_val = results['adx']
            if adx_val > 50:
                results['adx_strength'] = 'VERY_STRONG'
            elif adx_val > 25:
                results['adx_strength'] = 'STRONG'
            elif adx_val > 20:
                results['adx_strength'] = 'MODERATE'
            else:
                results['adx_strength'] = 'WEAK'
            
            # Directional movement
            if results['plus_di'] > results['minus_di']:
                results['dm_direction'] = 'BULLISH'
            else:
                results['dm_direction'] = 'BEARISH'
            
            return results
            
        except Exception as e:
            logger.error(f"ADX analysis failed: {e}")
            return {}
    
    def _calculate_volume_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Institutional volume analysis"""
        try:
            if len(data) < 20:
                return {}
            
            results = {}
            
            # On-Balance Volume (simplified)
            obv = (data['volume'] * np.sign(data['close'].diff())).cumsum()
            results['obv'] = obv.iloc[-1]
            results['obv_trend'] = obv.diff().tail(10).mean()  # 10-day OBV trend
            
            # Volume-Weighted Average Price (VWAP)
            vwap = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()
            results['vwap'] = vwap.iloc[-1]
            results['price_vs_vwap'] = (data['close'].iloc[-1] / vwap.iloc[-1] - 1) * 100
            
            # Volume breakout detection
            avg_volume = data['volume'].tail(20).mean()
            current_volume = data['volume'].iloc[-1]
            results['volume_ratio'] = current_volume / avg_volume
            
            # Volume trend confirmation
            volume_trend = data['volume'].rolling(5).mean().diff().tail(5).mean()
            results['volume_trend'] = volume_trend
            
            # Volume signals
            if results['volume_ratio'] > 3.0:
                results['volume_signal'] = 'HIGH_VOLUME_BREAKOUT'
            elif results['volume_ratio'] > 1.5:
                results['volume_signal'] = 'ABOVE_AVERAGE'
            elif results['volume_ratio'] < 0.5:
                results['volume_signal'] = 'LOW_VOLUME'
            else:
                results['volume_signal'] = 'NORMAL'
            
            return results
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return {}
    
    def _calculate_acceleration_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Price acceleration (second derivative) analysis"""
        try:
            if len(data) < 10:
                return {}
            
            # Calculate price velocity (first derivative)
            velocity = data['close'].pct_change()
            
            # Calculate acceleration (second derivative)
            acceleration = velocity.diff()
            
            results = {
                'price_velocity': velocity.tail(5).mean(),
                'price_acceleration': acceleration.tail(5).mean(),
                'acceleration_trend': acceleration.tail(10).mean()
            }
            
            # Acceleration signals
            if results['price_acceleration'] > 0.01:  # 1% acceleration
                results['acceleration_signal'] = 'ACCELERATING_UP'
            elif results['price_acceleration'] < -0.01:
                results['acceleration_signal'] = 'ACCELERATING_DOWN'
            else:
                results['acceleration_signal'] = 'STABLE'
            
            return results
            
        except Exception as e:
            logger.error(f"Acceleration analysis failed: {e}")
            return {}
    
    def _calculate_composite_score(self, results: Dict[str, Any], timeframe: str) -> float:
        """Calculate composite technical score"""
        try:
            score = 0.0
            max_score = 100.0
            
            # Momentum scoring (30 points)
            momentum_values = []
            for period in ['3d', '5d', '10d', '20d', '50d']:
                key = f'momentum_{period}'
                if key in results:
                    momentum_values.append(results[key])
            
            if momentum_values:
                avg_momentum = np.mean(momentum_values)
                if avg_momentum > 15:  # >15% momentum
                    score += 30
                elif avg_momentum > 10:
                    score += 25
                elif avg_momentum > 5:
                    score += 20
                elif avg_momentum > 0:
                    score += 15
                else:
                    score += 5
            
            # RSI scoring (20 points)
            rsi_14 = results.get('rsi_14', 50)
            if 40 <= rsi_14 <= 70:  # Healthy momentum zone
                score += 20
            elif 30 <= rsi_14 <= 80:
                score += 15
            elif rsi_14 > 80:
                score += 10  # Overbought penalty
            else:
                score += 5
            
            # MACD scoring (15 points)
            macd_trend = results.get('macd_trend', 'NEUTRAL')
            if macd_trend == 'BULLISH_ACCELERATION':
                score += 15
            elif macd_trend == 'BULLISH':
                score += 12
            elif macd_trend == 'NEUTRAL':
                score += 8
            else:
                score += 3
            
            # Volume scoring (15 points)
            volume_signal = results.get('volume_signal', 'NORMAL')
            if volume_signal == 'HIGH_VOLUME_BREAKOUT':
                score += 15
            elif volume_signal == 'ABOVE_AVERAGE':
                score += 12
            elif volume_signal == 'NORMAL':
                score += 8
            else:
                score += 3
            
            # Bollinger Band position (10 points)
            bb_position = results.get('bb_position', 0.5)
            if 0.6 <= bb_position <= 0.9:  # Strong position without extreme
                score += 10
            elif 0.4 <= bb_position <= 0.6:
                score += 8
            else:
                score += 5
            
            # ADX strength bonus (10 points)
            adx_strength = results.get('adx_strength', 'WEAK')
            dm_direction = results.get('dm_direction', 'NEUTRAL')
            
            if dm_direction == 'BULLISH':
                if adx_strength == 'VERY_STRONG':
                    score += 10
                elif adx_strength == 'STRONG':
                    score += 8
                elif adx_strength == 'MODERATE':
                    score += 6
                else:
                    score += 3
            else:
                score += 2  # Bearish direction penalty
            
            return min(max_score, max(0, score))
            
        except Exception as e:
            logger.error(f"Composite score calculation failed: {e}")
            return 50.0
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'composite_score': 0,
            'momentum_3d': 0,
            'momentum_5d': 0,
            'momentum_10d': 0,
            'momentum_20d': 0,
            'momentum_50d': 0,
            'rsi_14': 50,
            'rsi_2': 50,
            'macd': 0,
            'macd_signal': 0,
            'macd_histogram': 0,
            'macd_trend': 'NEUTRAL',
            'bb_position': 0.5,
            'bb_width': 10,
            'stoch_k': 50,
            'stoch_d': 50,
            'williams_r': -50,
            'cci': 0,
            'adx': 25,
            'plus_di': 25,
            'minus_di': 25,
            'adx_strength': 'WEAK',
            'dm_direction': 'NEUTRAL',
            'volume_ratio': 1.0,
            'volume_signal': 'NORMAL'
        }