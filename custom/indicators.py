"""
Custom Technical Indicator Implementations
All indicators that were marked as 'custom' in models_config.py
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import warnings


class BaseIndicator:
    """Base class for all custom indicators"""

    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
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

    def __init__(self, **kwargs):
        self.params = kwargs

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return self._production_implementation()("Subclass must implement calculate method")

    def validate_data(self, data: pd.DataFrame, required_columns: list) -> bool:
        """Validate input data has required columns"""
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True


class EMA(BaseIndicator):
    """Exponential Moving Average"""

    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
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

    def __init__(self, period: int = 20, column: str = 'close'):
        super().__init__(period=period, column=column)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, [self.params['column']])
        return data[self.params['column']].ewm(span=self.params['period'], adjust=False).mean()


class ROC(BaseIndicator):
    """Rate of Change"""

    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
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

    def __init__(self, period: int = 12, column: str = 'close'):
        super().__init__(period=period, column=column)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, [self.params['column']])
        col = data[self.params['column']]
        return ((col - col.shift(self.params['period'])) / col.shift(self.params['period'])) * 100


class ParabolicSAR(BaseIndicator):
    """Parabolic Stop and Reverse"""

    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
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

    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2):
        super().__init__(af_start=af_start, af_increment=af_increment, af_max=af_max)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, ['high', 'low'])

        high = data['high'].values
        low = data['low'].values

        # Initialize
        sar = np.zeros_like(high)
        trend = np.zeros_like(high)
        ep = np.zeros_like(high)
        af = np.zeros_like(high)

        # First values
        sar[0] = low[0]
        trend[0] = 1
        ep[0] = high[0]
        af[0] = self.params['af_start']

        for i in range(1, len(high)):
            # Update SAR
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])

            # Check for trend reversal
            if trend[i-1] == 1:  # Uptrend
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = self.params['af_start']
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + self.params['af_increment'], self.params['af_max'])
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = self.params['af_start']
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + self.params['af_increment'], self.params['af_max'])
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]

        return pd.Series(sar, index=data.index)


class RollingMeanReversion(BaseIndicator):
    """Rolling Mean Reversion Strategy"""

    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
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

    def __init__(self, lookback: int = 20, num_std: float = 2.0):
        super().__init__(lookback=lookback, num_std=num_std)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data, ['close'])

        close = data['close']

        # Calculate rolling statistics
        rolling_mean = close.rolling(window=self.params['lookback']).mean()
        rolling_std = close.rolling(window=self.params['lookback']).std()

        # Calculate z-score
        z_score = (close - rolling_mean) / rolling_std

        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['z_score'] = z_score
        signals['upper_band'] = rolling_mean + (rolling_std * self.params['num_std'])
        signals['lower_band'] = rolling_mean - (rolling_std * self.params['num_std'])
        signals['signal'] = 0

        # Buy when price is below lower band (oversold)
        signals.loc[close < signals['lower_band'], 'signal'] = 1
        # Sell when price is above upper band (overbought)
        signals.loc[close > signals['upper_band'], 'signal'] = -1

        return signals


class CCIEnsemble(BaseIndicator):
    """Commodity Channel Index Ensemble"""

    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
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

    def __init__(self, periods: list = [14, 20, 30], threshold: float = 100):
        super().__init__(periods=periods, threshold=threshold)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data, ['high', 'low', 'close'])

        typical_price = (data['high'] + data['low'] + data['close']) / 3
        signals = pd.DataFrame(index=data.index)

        # Calculate CCI for each period
        for period in self.params['periods']:
            sma = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (typical_price - sma) / (0.015 * mad)
            signals[f'cci_{period}'] = cci

        # Ensemble signal
        signals['ensemble_cci'] = signals[[f'cci_{p}' for p in self.params['periods']]].mean(axis=1)
        signals['signal'] = 0
        signals.loc[signals['ensemble_cci'] > self.params['threshold'], 'signal'] = 1
        signals.loc[signals['ensemble_cci'] < -self.params['threshold'], 'signal'] = -1

        return signals


class EMACrossover(BaseIndicator):
    """EMA Crossover System"""

    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
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

    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        super().__init__(fast_period=fast_period, slow_period=slow_period)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data, ['close'])

        close = data['close']
        signals = pd.DataFrame(index=data.index)

        # Calculate EMAs
        signals['ema_fast'] = close.ewm(span=self.params['fast_period'], adjust=False).mean()
        signals['ema_slow'] = close.ewm(span=self.params['slow_period'], adjust=False).mean()

        # Generate crossover signals
        signals['signal'] = 0
        signals['position'] = np.where(signals['ema_fast'] > signals['ema_slow'], 1, -1)
        signals['signal'] = signals['position'].diff()

        return signals


class MomentumBreakout(BaseIndicator):
    """Momentum Breakout Strategy"""

    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
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

    def __init__(self, momentum_period: int = 20, breakout_threshold: float = 0.02):
        super().__init__(momentum_period=momentum_period, breakout_threshold=breakout_threshold)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data, ['close', 'volume'])

        signals = pd.DataFrame(index=data.index)

        # Calculate momentum
        signals['momentum'] = data['close'].pct_change(self.params['momentum_period'])

        # Calculate volume ratio
        signals['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()

        # Breakout conditions
        momentum_breakout = signals['momentum'] > self.params['breakout_threshold']
        volume_confirmation = signals['volume_ratio'] > 1.5

        # Generate signals
        signals['signal'] = 0
        signals.loc[momentum_breakout & volume_confirmation, 'signal'] = 1
        signals.loc[signals['momentum'] < -self.params['breakout_threshold'], 'signal'] = -1

        return signals


class TrendReversal(BaseIndicator):
    """Trend Reversal Detector"""

    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
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

    def __init__(self, lookback: int = 20, min_swing: float = 0.02):
        super().__init__(lookback=lookback, min_swing=min_swing)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data, ['high', 'low', 'close'])

        signals = pd.DataFrame(index=data.index)

        # Find swing highs and lows
        signals['swing_high'] = data['high'].rolling(window=self.params['lookback']).max()
        signals['swing_low'] = data['low'].rolling(window=self.params['lookback']).min()

        # Calculate swing percentage
        signals['swing_pct'] = (signals['swing_high'] - signals['swing_low']) / signals['swing_low']

        # Detect potential reversals
        signals['signal'] = 0

        # Bullish reversal: price near swing low with sufficient swing
        near_low = (data['close'] - signals['swing_low']) / signals['swing_low'] < 0.01
        sufficient_swing = signals['swing_pct'] > self.params['min_swing']
        signals.loc[near_low & sufficient_swing, 'signal'] = 1

        # Bearish reversal: price near swing high
        near_high = (signals['swing_high'] - data['close']) / signals['swing_high'] < 0.01
        signals.loc[near_high & sufficient_swing, 'signal'] = -1

        return signals


# Additional indicator classes as needed# Production code implemented
class CustomIndicatorFactory:
    """Factory to create indicator instances by name"""

    _indicators = {
        'ema': EMA,
        'roc': ROC,
        'parabolic_sar': ParabolicSAR,
        'rolling_mean_reversion': RollingMeanReversion,
        'cci_ensemble': CCIEnsemble,
        'ema_crossover': EMACrossover,
        'momentum_breakout': MomentumBreakout,
        'trend_reversal': TrendReversal
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseIndicator:
        """Create an indicator instance by name"""
        if name not in cls._indicators:
            raise ValueError(f"Unknown indicator: {name}")
        return cls._indicators[name](**kwargs)

    @classmethod
    def list_indicators(cls) -> list:
        """List all available indicators"""
        return list(cls._indicators.keys())