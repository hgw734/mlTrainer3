"""
Custom Risk Management Model Implementations
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class BaseRiskModel:
    """Base class for risk management models"""

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

    def calculate(self, data: Any) -> Any:
        raise NotImplementedError("Subclass must implement calculate method")

class DynamicRiskParity(BaseRiskModel):
    """Dynamic Risk Parity Model"""
    def __init__(self, lookback: int = 252, target_vol: float = 0.15, rebalance_freq: str = 'monthly'):
        super().__init__(lookback=lookback, target_vol=target_vol, rebalance_freq=rebalance_freq)

    def calculate(self, returns: pd.DataFrame) -> pd.DataFrame:
        weights = pd.DataFrame(index=returns.index, columns=returns.columns)
        if self.params['rebalance_freq'] == 'monthly':
            rebalance_dates = returns.resample('M').last().index
        elif self.params['rebalance_freq'] == 'weekly':
            rebalance_dates = returns.resample('W').last().index
        else:
            rebalance_dates = returns.index
        for date in rebalance_dates:
            if date < returns.index[self.params['lookback']]:
                continue
            hist_returns = returns.loc[:date].tail(self.params['lookback'])
            weights_at_date = self._calculate_risk_parity_weights(hist_returns)
            next_dates = rebalance_dates[rebalance_dates > date]
            next_date = next_dates[0] if len(next_dates) > 0 else returns.index[-1]
            weights.loc[date:next_date] = weights_at_date
        return weights

    def _calculate_risk_parity_weights(self, returns: pd.DataFrame) -> pd.Series:
        cov_matrix = returns.cov()
        n_assets = len(returns.columns)
        weights = np.ones(n_assets) / n_assets
        for _ in range(100):
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = (cov_matrix @ weights) / port_vol
            contrib = weights * marginal_contrib
            target_contrib = port_vol / n_assets
            weights = weights * (target_contrib / contrib)
            weights = weights / weights.sum()
            current_vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
            weights = weights * (self.params['target_vol'] / current_vol)
        return pd.Series(weights, index=returns.columns)

class EWMARiskMetrics(BaseRiskModel):
    """Exponentially Weighted Moving Average Risk Metrics"""
    def __init__(self, lambda_param: float = 0.94, min_periods: int = 20):
        super().__init__(lambda_param=lambda_param, min_periods=min_periods)

    def calculate(self, returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        results = {}
        results['volatility'] = returns.ewm(
            alpha=1-self.params['lambda_param'],
            min_periods=self.params['min_periods']
        ).std() * np.sqrt(252)
        cov_ewma = pd.DataFrame(index=returns.index, columns=returns.columns)
        corr_ewma = pd.DataFrame(index=returns.index, columns=returns.columns)
        for i in range(self.params['min_periods'], len(returns)):
            ret_to_date = returns.iloc[:i+1]
            cov = ret_to_date.ewm(
                alpha=1-self.params['lambda_param'],
                min_periods=self.params['min_periods']
            ).cov().iloc[-len(returns.columns):]
            date = returns.index[i]
            cov_ewma.loc[date] = cov.values.flatten()[:len(returns.columns)]
            std = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std, std)
            corr_ewma.loc[date] = corr.flatten()[:len(returns.columns)]
        results['covariance'] = cov_ewma
        results['correlation'] = corr_ewma
        results['var_95'] = -returns.rolling(window=self.params['min_periods']).quantile(0.05)
        results['var_99'] = -returns.rolling(window=self.params['min_periods']).quantile(0.01)
        results['cvar_95'] = self._calculate_cvar(returns, 0.05)
        results['cvar_99'] = self._calculate_cvar(returns, 0.01)
        return results

    def _calculate_cvar(self, returns: pd.DataFrame, alpha: float) -> pd.DataFrame:
        cvar = pd.DataFrame(index=returns.index, columns=returns.columns)
        for col in returns.columns:
            for i in range(self.params['min_periods'], len(returns)):
                window = returns[col].iloc[max(0, i-252):i+1]
                var_threshold = window.quantile(alpha)
                cvar.loc[returns.index[i], col] = -window[window <= var_threshold].mean()
        return cvar

class RegimeSwitchingVolatility(BaseRiskModel):
    """Regime Switching Volatility Model"""
    def __init__(self, n_regimes: int = 2, lookback: int = 252):
        super().__init__(n_regimes=n_regimes, lookback=lookback)

    def calculate(self, returns: pd.Series) -> pd.DataFrame:
        results = pd.DataFrame(index=returns.index)
        results['volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
        vol_percentiles = np.linspace(0, 100, self.params['n_regimes'] + 1)
        for i in range(self.params['lookback'], len(returns)):
            hist_vol = results['volatility'].iloc[max(0, i-self.params['lookback']):i]
            thresholds = np.percentile(hist_vol.dropna(), vol_percentiles)
            current_vol = results['volatility'].iloc[i]
            if pd.notna(current_vol):
                regime = np.digitize(current_vol, thresholds[1:-1])
                results.loc[returns.index[i], 'regime'] = regime
        for regime in range(self.params['n_regimes']):
            mask = results['regime'] == regime + 1
            results.loc[mask, 'regime_mean_vol'] = results.loc[mask, 'volatility'].mean()
            results.loc[mask, 'regime_std_vol'] = results.loc[mask, 'volatility'].std()
        return results

class VAR(BaseRiskModel):
    """Value at Risk Model"""
    def __init__(self, confidence_levels: list = [0.95, 0.99], method: str = 'historical'):
        super().__init__(confidence_levels=confidence_levels, method=method)

    def calculate(self, returns: pd.DataFrame, portfolio_weights: Optional[pd.Series] = None) -> pd.DataFrame:
        results = pd.DataFrame(index=returns.index)
        if portfolio_weights is not None:
            returns = (returns * portfolio_weights).sum(axis=1)
        for cl in self.params['confidence_levels']:
            if self.params['method'] == 'historical':
                results[f'VaR_{int(cl*100)}'] = -returns.rolling(window=252).quantile(1-cl)
            elif self.params['method'] == 'parametric':
                mu = returns.rolling(window=252).mean()
                sigma = returns.rolling(window=252).std()
                z = stats.norm.ppf(1-cl)
                results[f'VaR_{int(cl*100)}'] = -(mu + z * sigma)
        return results

@dataclass
class InformationRatio:
    benchmark_return: float = 0.0
    is_fitted: bool = False

    def fit(self, data: pd.Series) -> 'InformationRatio':
        if len(data) < 50:
            raise ValueError(f"Insufficient data: {len(data)} < 50")
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        returns = data.pct_change().dropna()
        info_ratios = pd.Series(0.0, index=data.index)
        for i in range(50, len(returns)):
            window_returns = returns.iloc[i-50:i+1]
            excess_return = window_returns.mean() - self.benchmark_return
            tracking_error = window_returns.std()
            if tracking_error > 0:
                info_ratios.iloc[i] = excess_return / tracking_error
        return info_ratios

    def get_parameters(self) -> Dict[str, Any]:
        return {'benchmark_return': self.benchmark_return, 'is_fitted': self.is_fitted}

@dataclass
class ExpectedShortfall:
    confidence_level: float = 0.95
    is_fitted: bool = False

    def fit(self, data: pd.Series) -> 'ExpectedShortfall':
        if len(data) < 50:
            raise ValueError(f"Insufficient data: {len(data)} < 50")
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        returns = data.pct_change().dropna()
        es_values = pd.Series(0.0, index=data.index)
        for i in range(50, len(returns)):
            window_returns = returns.iloc[i-50:i+1]
            var_threshold = np.percentile(window_returns, (1 - self.confidence_level) * 100)
            tail_returns = window_returns[window_returns <= var_threshold]
            if len(tail_returns) > 0:
                es_values.iloc[i] = tail_returns.mean()
        return es_values

    def get_parameters(self) -> Dict[str, Any]:
        return {'confidence_level': self.confidence_level, 'is_fitted': self.is_fitted}

@dataclass
class MaximumDrawdown:
    window: int = 252
    is_fitted: bool = False

    def fit(self, data: pd.Series) -> 'MaximumDrawdown':
        if len(data) < self.window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.window}")
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        drawdowns = pd.Series(0.0, index=data.index)
        for i in range(self.window, len(data)):
            window_data = data.iloc[i-self.window:i+1]
            peak = window_data.expanding().max()
            drawdown = (window_data - peak) / peak
            max_drawdown = drawdown.min()
            drawdowns.iloc[i] = max_drawdown
        return drawdowns

    def get_parameters(self) -> Dict[str, Any]:
        return {'window': self.window, 'is_fitted': self.is_fitted} 