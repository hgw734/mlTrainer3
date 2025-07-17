#!/usr/bin/env python3
"""
Advanced Position Sizing Model Implementations for S&P 500 Trading
=================================================================

Advanced position sizing models including:
- Kelly Criterion
- Volatility targeting
- Risk parity sizing
- Dynamic position sizing
- Portfolio optimization
- Risk-adjusted sizing
- Maximum drawdown protection
- Correlation-based sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
from scipy.optimize import minimize
from scipy import stats
import cvxpy as cp

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class PositionSize:
    """Position sizing result"""
    symbol: str
    position_size: float
    position_value: float
    allocation_percentage: float
    risk_adjusted_size: float
    confidence_level: float
    sizing_method: str
    timestamp: datetime

class BasePositionSizingModel:
    """Base class for position sizing models"""
    
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

    def fit(self, data: pd.Series) -> 'BasePositionSizingModel':
        """Fit the position sizing model"""
        if len(data) < self.params.get('min_data_points', 100):
            raise ValueError(f"Insufficient data: {len(data)} < {self.params.get('min_data_points', 100)}")
        self.is_fitted = True
        return self

    def calculate_position_size(self, data: pd.Series, capital: float, signal: float) -> PositionSize:
        """Calculate position size"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before position sizing")
        raise NotImplementedError("Subclasses must implement calculate_position_size method")

class KellyCriterionModel(BasePositionSizingModel):
    """Kelly Criterion position sizing model"""
    
    def __init__(self, window: int = 252, risk_free_rate: float = 0.02):
        super().__init__(window=window, risk_free_rate=risk_free_rate, min_data_points=window + 10)
        
    def calculate_position_size(self, data: pd.Series, capital: float, signal: float) -> PositionSize:
        """Calculate position size using Kelly Criterion"""
        returns = data.pct_change().dropna()
        
        if len(returns) < self.params['window']:
            return PositionSize(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                position_size=0.0,
                position_value=0.0,
                allocation_percentage=0.0,
                risk_adjusted_size=0.0,
                confidence_level=0.0,
                sizing_method='kelly_criterion',
                timestamp=datetime.now()
            )
        
        # Calculate Kelly Criterion parameters
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
        
        # Kelly fraction
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.0
        
        # Apply signal strength
        position_size = kelly_fraction * signal
        position_value = capital * position_size
        allocation_percentage = position_size * 100
        
        return PositionSize(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            position_size=position_size,
            position_value=position_value,
            allocation_percentage=allocation_percentage,
            risk_adjusted_size=position_size,
            confidence_level=win_rate,
            sizing_method='kelly_criterion',
            timestamp=datetime.now()
        )

class VolatilityTargetingModel(BasePositionSizingModel):
    """Volatility targeting position sizing model"""
    
    def __init__(self, target_volatility: float = 0.15, window: int = 60):
        super().__init__(target_volatility=target_volatility, window=window, min_data_points=window + 10)
        
    def calculate_position_size(self, data: pd.Series, capital: float, signal: float) -> PositionSize:
        """Calculate position size using volatility targeting"""
        returns = data.pct_change().dropna()
        
        if len(returns) < self.params['window']:
            return PositionSize(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                position_size=0.0,
                position_value=0.0,
                allocation_percentage=0.0,
                risk_adjusted_size=0.0,
                confidence_level=0.0,
                sizing_method='volatility_targeting',
                timestamp=datetime.now()
            )
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.params['window']).std()
        current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else returns.std()
        
        # Calculate volatility adjustment
        if current_vol > 0:
            vol_adjustment = self.params['target_volatility'] / current_vol
            vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)  # Limit adjustment
        else:
            vol_adjustment = 1.0
        
        # Base position size
        base_size = 0.1  # 10% base allocation
        position_size = base_size * vol_adjustment * signal
        position_size = np.clip(position_size, -0.5, 0.5)  # Limit position size
        
        position_value = capital * position_size
        allocation_percentage = position_size * 100
        
        return PositionSize(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            position_size=position_size,
            position_value=position_value,
            allocation_percentage=allocation_percentage,
            risk_adjusted_size=position_size * vol_adjustment,
            confidence_level=1.0 - current_vol,
            sizing_method='volatility_targeting',
            timestamp=datetime.now()
        )

class RiskParityModel(BasePositionSizingModel):
    """Risk parity position sizing model"""
    
    def __init__(self, target_risk: float = 0.02, window: int = 60):
        super().__init__(target_risk=target_risk, window=window, min_data_points=window + 10)
        
    def calculate_position_size(self, data: pd.Series, capital: float, signal: float) -> PositionSize:
        """Calculate position size using risk parity"""
        returns = data.pct_change().dropna()
        
        if len(returns) < self.params['window']:
            return PositionSize(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                position_size=0.0,
                position_value=0.0,
                allocation_percentage=0.0,
                risk_adjusted_size=0.0,
                confidence_level=0.0,
                sizing_method='risk_parity',
                timestamp=datetime.now()
            )
        
        # Calculate volatility
        volatility = returns.std()
        
        # Risk parity sizing
        if volatility > 0:
            risk_parity_size = self.params['target_risk'] / volatility
            risk_parity_size = np.clip(risk_parity_size, 0, 0.5)  # Limit position size
        else:
            risk_parity_size = 0.0
        
        # Apply signal
        position_size = risk_parity_size * signal
        position_value = capital * position_size
        allocation_percentage = position_size * 100
        
        return PositionSize(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            position_size=position_size,
            position_value=position_value,
            allocation_percentage=allocation_percentage,
            risk_adjusted_size=position_size,
            confidence_level=1.0 - volatility,
            sizing_method='risk_parity',
            timestamp=datetime.now()
        )

class DynamicPositionSizingModel(BasePositionSizingModel):
    """Dynamic position sizing model"""
    
    def __init__(self, base_size: float = 0.1, max_size: float = 0.5, 
                 momentum_weight: float = 0.3, volatility_weight: float = 0.3):
        super().__init__(
            base_size=base_size,
            max_size=max_size,
            momentum_weight=momentum_weight,
            volatility_weight=volatility_weight,
            min_data_points=100
        )
        
    def calculate_position_size(self, data: pd.Series, capital: float, signal: float) -> PositionSize:
        """Calculate dynamic position size"""
        returns = data.pct_change().dropna()
        
        if len(returns) < 60:
            return PositionSize(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                position_size=0.0,
                position_value=0.0,
                allocation_percentage=0.0,
                risk_adjusted_size=0.0,
                confidence_level=0.0,
                sizing_method='dynamic_sizing',
                timestamp=datetime.now()
            )
        
        # Calculate momentum factor
        momentum_window = 20
        momentum = (data.iloc[-1] - data.iloc[-momentum_window]) / data.iloc[-momentum_window]
        momentum_factor = np.clip(momentum, -1, 1)
        
        # Calculate volatility factor
        volatility = returns.std()
        volatility_factor = 1.0 / (1.0 + volatility)  # Inverse relationship
        
        # Calculate trend factor
        trend_window = 50
        trend = (data.iloc[-1] - data.iloc[-trend_window]) / data.iloc[-trend_window]
        trend_factor = np.clip(trend, -1, 1)
        
        # Combine factors
        dynamic_factor = (
            self.params['momentum_weight'] * momentum_factor +
            self.params['volatility_weight'] * volatility_factor +
            (1 - self.params['momentum_weight'] - self.params['volatility_weight']) * trend_factor
        )
        
        # Calculate position size
        position_size = self.params['base_size'] * dynamic_factor * signal
        position_size = np.clip(position_size, -self.params['max_size'], self.params['max_size'])
        
        position_value = capital * position_size
        allocation_percentage = position_size * 100
        
        return PositionSize(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            position_size=position_size,
            position_value=position_value,
            allocation_percentage=allocation_percentage,
            risk_adjusted_size=position_size * volatility_factor,
            confidence_level=abs(dynamic_factor),
            sizing_method='dynamic_sizing',
            timestamp=datetime.now()
        )

class PortfolioOptimizationModel(BasePositionSizingModel):
    """Portfolio optimization position sizing model"""
    
    def __init__(self, target_return: float = 0.10, target_volatility: float = 0.15):
        super().__init__(target_return=target_return, target_volatility=target_volatility, min_data_points=100)
        
    def calculate_position_size(self, data: pd.Series, capital: float, signal: float, 
                              portfolio_returns: pd.Series = None) -> PositionSize:
        """Calculate position size using portfolio optimization"""
        returns = data.pct_change().dropna()
        
        if len(returns) < 60:
            return PositionSize(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                position_size=0.0,
                position_value=0.0,
                allocation_percentage=0.0,
                risk_adjusted_size=0.0,
                confidence_level=0.0,
                sizing_method='portfolio_optimization',
                timestamp=datetime.now()
            )
        
        # Calculate expected return and volatility
        expected_return = returns.mean()
        volatility = returns.std()
        
        # Calculate Sharpe ratio
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        # Portfolio optimization sizing
        if portfolio_returns is not None and len(portfolio_returns) > 0:
            # Calculate correlation with portfolio
            correlation = returns.corr(portfolio_returns) if len(portfolio_returns) == len(returns) else 0
            # Adjust size based on correlation
            correlation_adjustment = 1 - abs(correlation) * 0.5
        else:
            correlation_adjustment = 1.0
        
        # Calculate optimal position size
        if volatility > 0:
            optimal_size = (expected_return - self.params['target_return']) / (volatility ** 2)
            optimal_size *= correlation_adjustment
            optimal_size = np.clip(optimal_size, -0.5, 0.5)
        else:
            optimal_size = 0.0
        
        # Apply signal
        position_size = optimal_size * signal
        position_value = capital * position_size
        allocation_percentage = position_size * 100
        
        return PositionSize(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            position_size=position_size,
            position_value=position_value,
            allocation_percentage=allocation_percentage,
            risk_adjusted_size=position_size * correlation_adjustment,
            confidence_level=sharpe_ratio,
            sizing_method='portfolio_optimization',
            timestamp=datetime.now()
        )

class RiskAdjustedSizingModel(BasePositionSizingModel):
    """Risk-adjusted position sizing model"""
    
    def __init__(self, max_drawdown: float = 0.10, var_confidence: float = 0.95):
        super().__init__(max_drawdown=max_drawdown, var_confidence=var_confidence, min_data_points=100)
        
    def calculate_position_size(self, data: pd.Series, capital: float, signal: float) -> PositionSize:
        """Calculate risk-adjusted position size"""
        returns = data.pct_change().dropna()
        
        if len(returns) < 60:
            return PositionSize(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                position_size=0.0,
                position_value=0.0,
                allocation_percentage=0.0,
                risk_adjusted_size=0.0,
                confidence_level=0.0,
                sizing_method='risk_adjusted',
                timestamp=datetime.now()
            )
        
        # Calculate risk metrics
        volatility = returns.std()
        var_95 = np.percentile(returns, 5)
        max_drawdown = self._calculate_max_drawdown(data)
        
        # Risk-adjusted sizing
        risk_score = 1.0
        if volatility > 0:
            risk_score *= (1.0 / (1.0 + volatility))
        if abs(var_95) > 0:
            risk_score *= (1.0 / (1.0 + abs(var_95)))
        if abs(max_drawdown) > 0:
            risk_score *= (1.0 / (1.0 + abs(max_drawdown)))
        
        # Base position size
        base_size = 0.1
        position_size = base_size * risk_score * signal
        position_size = np.clip(position_size, -0.5, 0.5)
        
        position_value = capital * position_size
        allocation_percentage = position_size * 100
        
        return PositionSize(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            position_size=position_size,
            position_value=position_value,
            allocation_percentage=allocation_percentage,
            risk_adjusted_size=position_size * risk_score,
            confidence_level=risk_score,
            sizing_method='risk_adjusted',
            timestamp=datetime.now()
        )
    
    def _calculate_max_drawdown(self, data: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + data.pct_change().dropna()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

class CorrelationBasedSizingModel(BasePositionSizingModel):
    """Correlation-based position sizing model"""
    
    def __init__(self, correlation_threshold: float = 0.7, base_size: float = 0.1):
        super().__init__(correlation_threshold=correlation_threshold, base_size=base_size, min_data_points=100)
        
    def calculate_position_size(self, data: pd.Series, capital: float, signal: float,
                              market_data: pd.Series = None) -> PositionSize:
        """Calculate position size based on correlation"""
        returns = data.pct_change().dropna()
        
        if len(returns) < 60:
            return PositionSize(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                position_size=0.0,
                position_value=0.0,
                allocation_percentage=0.0,
                risk_adjusted_size=0.0,
                confidence_level=0.0,
                sizing_method='correlation_based',
                timestamp=datetime.now()
            )
        
        # Calculate correlation with market
        if market_data is not None and len(market_data) > 0:
            market_returns = market_data.pct_change().dropna()
            if len(market_returns) == len(returns):
                correlation = returns.corr(market_returns)
            else:
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Adjust position size based on correlation
        if abs(correlation) > self.params['correlation_threshold']:
            # High correlation - reduce size
            correlation_adjustment = 1.0 - abs(correlation) * 0.5
        else:
            # Low correlation - maintain size
            correlation_adjustment = 1.0
        
        # Calculate position size
        position_size = self.params['base_size'] * correlation_adjustment * signal
        position_size = np.clip(position_size, -0.5, 0.5)
        
        position_value = capital * position_size
        allocation_percentage = position_size * 100
        
        return PositionSize(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            position_size=position_size,
            position_value=position_value,
            allocation_percentage=allocation_percentage,
            risk_adjusted_size=position_size * correlation_adjustment,
            confidence_level=1.0 - abs(correlation),
            sizing_method='correlation_based',
            timestamp=datetime.now()
        )

class PositionSizingModel:
    """Comprehensive position sizing model for S&P 500 trading"""
    
    def __init__(self, sizing_window: int = 252):
        self.sizing_window = sizing_window
        self.is_fitted = False
        self.models = {}
        
        # Initialize different position sizing models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all position sizing models"""
        
        # Basic sizing models
        self.models['kelly_criterion'] = KellyCriterionModel()
        self.models['volatility_targeting'] = VolatilityTargetingModel()
        self.models['risk_parity'] = RiskParityModel()
        self.models['dynamic_sizing'] = DynamicPositionSizingModel()
        self.models['portfolio_optimization'] = PortfolioOptimizationModel()
        self.models['risk_adjusted'] = RiskAdjustedSizingModel()
        self.models['correlation_based'] = CorrelationBasedSizingModel()
        
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

    def fit(self, data: pd.Series) -> 'PositionSizingModel':
        """Fit all position sizing models"""
        if len(data) < self.sizing_window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.sizing_window}")
        
        # Fit all models
        for model_name, model in self.models.items():
            try:
                model.fit(data)
            except Exception as e:
                logger.warning(f"Failed to fit {model_name}: {e}")
        
        self.is_fitted = True
        return self

    def calculate_position_size(self, data: pd.Series, capital: float, signal: float, 
                              model_name: str = None, **kwargs) -> PositionSize:
        """Calculate position size"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before position sizing")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            return self.models[model_name].calculate_position_size(data, capital, signal, **kwargs)
        else:
            # Return Kelly Criterion as default
            return self.models['kelly_criterion'].calculate_position_size(data, capital, signal)

    def get_available_models(self) -> List[str]:
        """Get list of available position sizing models"""
        return list(self.models.keys())

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name].params

    def get_parameters(self) -> Dict[str, Any]:
        """Get all model parameters"""
        return {
            'sizing_window': self.sizing_window,
            'is_fitted': self.is_fitted,
            'available_models': self.get_available_models(),
            'model_parameters': {
                name: model.params for name, model in self.models.items()
            }
        } 