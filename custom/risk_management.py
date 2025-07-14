#!/usr/bin/env python3
"""
Advanced Risk Management Model Implementations for S&P 500 Trading
=================================================================

Advanced risk management models including:
- Value at Risk (VaR) calculations
- Expected Shortfall (ES)
- Portfolio risk models
- Stress testing
- Dynamic risk adjustment
- Risk parity models
- Maximum drawdown protection
- Volatility targeting
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
import cvxpy as cp

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics result"""
    symbol: str
    var_95: float
    var_99: float
    expected_shortfall: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation: float
    timestamp: datetime

class BaseRiskModel:
    """Base class for risk management models"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
        
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

    def fit(self, data: pd.Series) -> 'BaseRiskModel':
        """Fit the risk model"""
        if len(data) < self.params.get('min_data_points', 100):
            raise ValueError(f"Insufficient data: {len(data)} < {self.params.get('min_data_points', 100)}")
        self.is_fitted = True
        return self

    def calculate_risk_metrics(self, data: pd.Series) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before risk calculation")
        raise NotImplementedError("Subclasses must implement calculate_risk_metrics method")

class VaRModel(BaseRiskModel):
    """Value at Risk (VaR) calculation model"""
    
    def __init__(self, confidence_level: float = 0.95, window: int = 252):
        super().__init__(confidence_level=confidence_level, window=window, min_data_points=window + 10)
        
    def calculate_risk_metrics(self, data: pd.Series) -> RiskMetrics:
        """Calculate VaR and other risk metrics"""
        returns = data.pct_change().dropna()
        
        # Calculate VaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Calculate Expected Shortfall (Conditional VaR)
        es_threshold = np.percentile(returns, (1 - self.params['confidence_level']) * 100)
        expected_shortfall = returns[returns <= es_threshold].mean()
        
        # Calculate other metrics
        volatility = returns.std()
        sharpe_ratio = returns.mean() / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = returns.mean() / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate Calmar ratio
        calmar_ratio = returns.mean() / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return RiskMetrics(
            symbol=data.name if hasattr(data, 'name') else 'Unknown',
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            beta=0.0,  # Will be calculated separately
            correlation=0.0,  # Will be calculated separately
            timestamp=datetime.now()
        )

class PortfolioRiskModel(BaseRiskModel):
    """Portfolio risk management model"""
    
    def __init__(self, risk_free_rate: float = 0.02, target_volatility: float = 0.15):
        super().__init__(risk_free_rate=risk_free_rate, target_volatility=target_volatility, min_data_points=100)
        
    def calculate_portfolio_risk(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        # Portfolio return
        portfolio_return = (returns * weights).sum(axis=1)
        
        # Portfolio volatility
        portfolio_vol = portfolio_return.std()
        
        # Portfolio VaR
        portfolio_var_95 = np.percentile(portfolio_return, 5)
        portfolio_var_99 = np.percentile(portfolio_return, 1)
        
        # Portfolio Sharpe ratio
        excess_return = portfolio_return.mean() - self.params['risk_free_rate']
        sharpe_ratio = excess_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_return).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'portfolio_volatility': portfolio_vol,
            'portfolio_var_95': portfolio_var_95,
            'portfolio_var_99': portfolio_var_99,
            'portfolio_sharpe': sharpe_ratio,
            'portfolio_max_drawdown': max_drawdown,
            'portfolio_return': portfolio_return.mean()
        }
    
    def optimize_portfolio(self, returns: pd.DataFrame, method: str = 'sharpe') -> Tuple[np.ndarray, Dict[str, float]]:
        """Optimize portfolio weights"""
        n_assets = returns.shape[1]
        
        if method == 'sharpe':
            # Maximize Sharpe ratio
            def objective(weights):
                portfolio_return = (returns * weights).sum(axis=1)
                excess_return = portfolio_return.mean() - self.params['risk_free_rate']
                portfolio_vol = portfolio_return.std()
                return -excess_return / portfolio_vol if portfolio_vol > 0 else 0
            
        elif method == 'min_variance':
            # Minimize variance
            def objective(weights):
                portfolio_return = (returns * weights).sum(axis=1)
                return portfolio_return.var()
            
        elif method == 'risk_parity':
            # Risk parity optimization
            def objective(weights):
                portfolio_return = (returns * weights).sum(axis=1)
                portfolio_vol = portfolio_return.std()
                target_vol = self.params['target_volatility']
                return abs(portfolio_vol - target_vol)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            risk_metrics = self.calculate_portfolio_risk(returns, optimal_weights)
            return optimal_weights, risk_metrics
        else:
            logger.warning("Portfolio optimization failed")
            return initial_weights, {}

class StressTestingModel(BaseRiskModel):
    """Stress testing model"""
    
    def __init__(self, stress_scenarios: Dict[str, float] = None):
        super().__init__(stress_scenarios=stress_scenarios or {
            'market_crash': -0.20,
            'volatility_spike': 0.50,
            'correlation_breakdown': 0.80,
            'liquidity_crisis': -0.15
        }, min_data_points=100)
        
    def stress_test_portfolio(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Perform stress testing on portfolio"""
        stress_results = {}
        
        # Market crash scenario
        if 'market_crash' in self.params['stress_scenarios']:
            crash_return = self.params['stress_scenarios']['market_crash']
            portfolio_crash_return = (returns * weights).sum(axis=1) * (1 + crash_return)
            stress_results['market_crash_loss'] = portfolio_crash_return.min()
        
        # Volatility spike scenario
        if 'volatility_spike' in self.params['stress_scenarios']:
            volatility_multiplier = 1 + self.params['stress_scenarios']['volatility_spike']
            stressed_returns = returns * volatility_multiplier
            portfolio_stressed_return = (stressed_returns * weights).sum(axis=1)
            stress_results['volatility_spike_var'] = np.percentile(portfolio_stressed_return, 5)
        
        # Correlation breakdown scenario
        if 'correlation_breakdown' in self.params['stress_scenarios']:
            correlation_threshold = self.params['stress_scenarios']['correlation_breakdown']
            # Simulate correlation breakdown by increasing diversification penalty
            portfolio_return = (returns * weights).sum(axis=1)
            correlation_penalty = 1 - correlation_threshold
            stressed_return = portfolio_return * (1 - correlation_penalty)
            stress_results['correlation_breakdown_loss'] = stressed_return.min()
        
        # Liquidity crisis scenario
        if 'liquidity_crisis' in self.params['stress_scenarios']:
            liquidity_shock = self.params['stress_scenarios']['liquidity_crisis']
            portfolio_return = (returns * weights).sum(axis=1)
            liquidity_impact = portfolio_return * liquidity_shock
            stress_results['liquidity_crisis_loss'] = liquidity_impact.min()
        
        return stress_results

class DynamicRiskAdjustmentModel(BaseRiskModel):
    """Dynamic risk adjustment model"""
    
    def __init__(self, base_volatility: float = 0.15, adjustment_window: int = 60):
        super().__init__(base_volatility=base_volatility, adjustment_window=adjustment_window, min_data_points=adjustment_window + 10)
        
    def calculate_dynamic_weights(self, returns: pd.DataFrame, target_volatility: float = None) -> np.ndarray:
        """Calculate dynamically adjusted portfolio weights"""
        if target_volatility is None:
            target_volatility = self.params['base_volatility']
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.params['adjustment_window']).std()
        
        # Calculate volatility adjustment factor
        vol_adjustment = target_volatility / rolling_vol.mean().mean()
        
        # Adjust weights based on volatility
        n_assets = returns.shape[1]
        base_weights = np.ones(n_assets) / n_assets
        
        # Apply volatility adjustment
        adjusted_weights = base_weights * vol_adjustment
        adjusted_weights = adjusted_weights / adjusted_weights.sum()  # Renormalize
        
        return adjusted_weights
    
    def calculate_risk_metrics(self, data: pd.Series) -> RiskMetrics:
        """Calculate risk metrics with dynamic adjustment"""
        returns = data.pct_change().dropna()
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.params['adjustment_window']).std()
        
        # Dynamic volatility adjustment
        current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else returns.std()
        vol_adjustment = self.params['base_volatility'] / current_vol if current_vol > 0 else 1
        
        # Adjust returns for risk targeting
        adjusted_returns = returns * vol_adjustment
        
        # Calculate adjusted metrics
        var_95 = np.percentile(adjusted_returns, 5)
        var_99 = np.percentile(adjusted_returns, 1)
        expected_shortfall = adjusted_returns[adjusted_returns <= var_95].mean()
        volatility = adjusted_returns.std()
        sharpe_ratio = adjusted_returns.mean() / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + adjusted_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sortino ratio
        downside_returns = adjusted_returns[adjusted_returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = adjusted_returns.mean() / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate Calmar ratio
        calmar_ratio = adjusted_returns.mean() / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return RiskMetrics(
            symbol=data.name if hasattr(data, 'name') else 'Unknown',
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            beta=0.0,
            correlation=0.0,
            timestamp=datetime.now()
        )

class RiskParityModel(BaseRiskModel):
    """Risk parity model"""
    
    def __init__(self, target_volatility: float = 0.15):
        super().__init__(target_volatility=target_volatility, min_data_points=100)
        
    def calculate_risk_parity_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate risk parity weights"""
        n_assets = returns.shape[1]
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Risk parity objective: equal risk contribution
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            risk_contributions = (weights * (cov_matrix @ weights)) / portfolio_vol
            return np.sum((risk_contributions - risk_contributions.mean())**2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(risk_parity_objective, initial_weights, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning("Risk parity optimization failed")
            return initial_weights

class MaximumDrawdownProtectionModel(BaseRiskModel):
    """Maximum drawdown protection model"""
    
    def __init__(self, max_drawdown_threshold: float = -0.10, protection_window: int = 20):
        super().__init__(max_drawdown_threshold=max_drawdown_threshold, protection_window=protection_window, min_data_points=protection_window + 10)
        
    def calculate_protection_signal(self, data: pd.Series) -> pd.Series:
        """Calculate drawdown protection signals"""
        signals = pd.Series(0.0, index=data.index)
        
        for i in range(self.params['protection_window'], len(data)):
            window_data = data.iloc[i-self.params['protection_window']:i+1]
            
            # Calculate drawdown
            cumulative = (1 + window_data.pct_change().dropna()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            current_drawdown = drawdown.iloc[-1]
            
            # Generate protection signal
            if current_drawdown < self.params['max_drawdown_threshold']:
                signals.iloc[i] = -1.0  # Reduce exposure
            elif current_drawdown > -0.05:  # Recovery signal
                signals.iloc[i] = 1.0  # Increase exposure
                
        return signals

class VolatilityTargetingModel(BaseRiskModel):
    """Volatility targeting model"""
    
    def __init__(self, target_volatility: float = 0.15, rebalance_window: int = 20):
        super().__init__(target_volatility=target_volatility, rebalance_window=rebalance_window, min_data_points=rebalance_window + 10)
        
    def calculate_volatility_adjustment(self, data: pd.Series) -> pd.Series:
        """Calculate volatility targeting adjustments"""
        adjustments = pd.Series(1.0, index=data.index)
        
        for i in range(self.params['rebalance_window'], len(data)):
            window_data = data.iloc[i-self.params['rebalance_window']:i+1]
            
            # Calculate rolling volatility
            returns = window_data.pct_change().dropna()
            current_vol = returns.std()
            
            # Calculate adjustment factor
            if current_vol > 0:
                adjustment = self.params['target_volatility'] / current_vol
                # Limit adjustment to reasonable bounds
                adjustment = np.clip(adjustment, 0.5, 2.0)
                adjustments.iloc[i] = adjustment
                
        return adjustments

class RiskManagementModel:
    """Comprehensive risk management model for S&P 500 trading"""
    
    def __init__(self, risk_window: int = 252):
        self.risk_window = risk_window
        self.is_fitted = False
        self.models = {}
        
        # Initialize different risk management models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all risk management models"""
        
        # Basic risk models
        self.models['var'] = VaRModel()
        self.models['portfolio_risk'] = PortfolioRiskModel()
        self.models['stress_testing'] = StressTestingModel()
        self.models['dynamic_adjustment'] = DynamicRiskAdjustmentModel()
        self.models['risk_parity'] = RiskParityModel()
        self.models['drawdown_protection'] = MaximumDrawdownProtectionModel()
        self.models['volatility_targeting'] = VolatilityTargetingModel()
        
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

    def fit(self, data: pd.Series) -> 'RiskManagementModel':
        """Fit all risk management models"""
        if len(data) < self.risk_window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.risk_window}")
        
        # Fit all models
        for model_name, model in self.models.items():
            try:
                model.fit(data)
            except Exception as e:
                logger.warning(f"Failed to fit {model_name}: {e}")
        
        self.is_fitted = True
        return self

    def calculate_risk_metrics(self, data: pd.Series, model_name: str = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before risk calculation")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            return self.models[model_name].calculate_risk_metrics(data)
        else:
            # Return VaR model metrics as default
            return self.models['var'].calculate_risk_metrics(data)

    def get_available_models(self) -> List[str]:
        """Get list of available risk management models"""
        return list(self.models.keys())

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name].params

    def get_parameters(self) -> Dict[str, Any]:
        """Get all model parameters"""
        return {
            'risk_window': self.risk_window,
            'is_fitted': self.is_fitted,
            'available_models': self.get_available_models(),
            'model_parameters': {
                name: model.params for name, model in self.models.items()
            }
        }