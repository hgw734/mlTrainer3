"""
Custom Trading System Implementations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class BaseTradingSystem:
    """Base class for trading systems"""

    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real economic data: {e}")
            return None

    def _get_real_alternative_data(self, data_type: str, **kwargs):
        try:
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

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclass must implement generate_signals method")

class TurtleTrading(BaseTradingSystem):
    """Original Turtle Trading System"""
    def __init__(self, entry_breakout: int = 20, exit_breakout: int = 10, atr_period: int = 20, risk_per_trade: float = 0.02):
        super().__init__(entry_breakout=entry_breakout, exit_breakout=exit_breakout, atr_period=atr_period, risk_per_trade=risk_per_trade)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['upper_channel'] = data['high'].rolling(window=self.params['entry_breakout']).max()
        signals['lower_channel'] = data['low'].rolling(window=self.params['entry_breakout']).min()
        signals['exit_upper'] = data['high'].rolling(window=self.params['exit_breakout']).max()
        signals['exit_lower'] = data['low'].rolling(window=self.params['exit_breakout']).min()
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        signals['atr'] = true_range.rolling(window=self.params['atr_period']).mean()
        signals['long_entry'] = data['close'] > signals['upper_channel'].shift(1)
        signals['short_entry'] = data['close'] < signals['lower_channel'].shift(1)
        signals['long_exit'] = data['close'] < signals['exit_lower'].shift(1)
        signals['short_exit'] = data['close'] > signals['exit_upper'].shift(1)
        signals['position'] = 0
        position = 0
        for i in range(1, len(signals)):
            if position == 0:
                if signals['long_entry'].iloc[i]:
                    position = 1
                elif signals['short_entry'].iloc[i]:
                    position = -1
            elif position == 1:
                if signals['long_exit'].iloc[i]:
                    position = 0
            elif position == -1:
                if signals['short_exit'].iloc[i]:
                    position = 0
            signals['position'].iloc[i] = position
        signals['position_size'] = self.params['risk_per_trade'] / (2 * signals['atr'])
        return signals

class MeanReversionPairs(BaseTradingSystem):
    """Mean Reversion Pairs Trading System"""
    def __init__(self, lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5):
        super().__init__(lookback=lookback, entry_z=entry_z, exit_z=exit_z)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data.columns) < 2:
            raise ValueError("Pairs trading requires at least 2 assets")
        signals = pd.DataFrame(index=data.index)
        asset1 = data.iloc[:, 0]
        asset2 = data.iloc[:, 1]
        signals['hedge_ratio'] = self._calculate_hedge_ratio(asset1, asset2)
        signals['spread'] = asset1 - signals['hedge_ratio'] * asset2
        spread_mean = signals['spread'].rolling(window=self.params['lookback']).mean()
        spread_std = signals['spread'].rolling(window=self.params['lookback']).std()
        signals['z_score'] = (signals['spread'] - spread_mean) / spread_std
        signals['position'] = 0
        signals.loc[signals['z_score'] < -self.params['entry_z'], 'position'] = 1
        signals.loc[signals['z_score'] > self.params['entry_z'], 'position'] = -1
        exit_condition = (abs(signals['z_score']) < self.params['exit_z'])
        signals.loc[exit_condition, 'position'] = 0
        signals['position'] = signals['position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        return signals

    def _calculate_hedge_ratio(self, asset1: pd.Series, asset2: pd.Series) -> pd.Series:
        hedge_ratios = pd.Series(index=asset1.index)
        for i in range(self.params['lookback'], len(asset1)):
            window_asset1 = asset1.iloc[i-self.params['lookback']:i]
            window_asset2 = asset2.iloc[i-self.params['lookback']:i]
            cov = np.cov(window_asset1, window_asset2)[0, 1]
            var = np.var(window_asset2)
            hedge_ratios.iloc[i] = cov / var if var != 0 else 1.0
        return hedge_ratios

class DualMomentum(BaseTradingSystem):
    """Dual Momentum Trading System"""
    def __init__(self, lookback: int = 252, rebalance_freq: str = 'monthly'):
        super().__init__(lookback=lookback, rebalance_freq=rebalance_freq)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        returns = data.pct_change()
        abs_momentum = returns.rolling(window=self.params['lookback']).mean() * 252
        rel_momentum_rank = abs_momentum.rank(axis=1, pct=True)
        positive_momentum = abs_momentum > 0
        n_assets = max(1, len(data.columns) // 3)
        if self.params['rebalance_freq'] == 'monthly':
            rebalance_dates = data.resample('M').last().index
        elif self.params['rebalance_freq'] == 'weekly':
            rebalance_dates = data.resample('W').last().index
        else:
            rebalance_dates = data.index
        signals['position'] = 0
        for date in rebalance_dates:
            if date < data.index[self.params['lookback']]:
                continue
            eligible = positive_momentum.loc[date]
            top_assets = rel_momentum_rank.loc[date][eligible].nlargest(n_assets).index
            signals.loc[date, 'position'] = 0
            signals.loc[date, 'position'] = signals.columns.isin(top_assets).astype(int)
        return signals

@dataclass
class MomentumBreakout:
    window: int = 20
    threshold: float = 0.02
    is_fitted: bool = False

    def fit(self, data: pd.Series) -> 'MomentumBreakout':
        if len(data) < self.window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.window}")
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        signals = pd.Series(0, index=data.index)
        for i in range(self.window, len(data)):
            window_data = data.iloc[i-self.window:i+1]
            momentum = (data.iloc[i] - window_data.iloc[0]) / window_data.iloc[0]
            if momentum > self.threshold:
                signals.iloc[i] = 1
            elif momentum < -self.threshold:
                signals.iloc[i] = -1
        return signals

    def get_parameters(self) -> Dict[str, Any]:
        return {'window': self.window, 'threshold': self.threshold, 'is_fitted': self.is_fitted}

@dataclass
class EMACrossover:
    short_window: int = 12
    long_window: int = 26
    is_fitted: bool = False

    def fit(self, data: pd.Series) -> 'EMACrossover':
        if len(data) < self.long_window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.long_window}")
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        short_ema = data.ewm(span=self.short_window, adjust=False).mean()
        long_ema = data.ewm(span=self.long_window, adjust=False).mean()
        signals = pd.Series(0, index=data.index)
        signals[short_ema > long_ema] = 1
        signals[short_ema < long_ema] = -1
        return signals

    def get_parameters(self) -> Dict[str, Any]:
        return {'short_window': self.short_window, 'long_window': self.long_window, 'is_fitted': self.is_fitted} 