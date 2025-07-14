#!/usr/bin/env python3
"""
Portfolio Optimization Model
Implements Modern Portfolio Theory with various optimization objectives
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt import HRPOpt, BlackLittermanModel
    from pypfopt.efficient_frontier import EfficientCVaR
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    logging.warning("PyPortfolioOpt not available. Install with: pip install pyportfolioopt")

from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)

@dataclass
class PortfolioOptimizer:
    """
    Advanced portfolio optimization using multiple methodologies:
    - Mean-Variance Optimization (Markowitz)
    - Hierarchical Risk Parity
    - Black-Litterman
    - Risk Parity
    - Maximum Sharpe Ratio
    - Minimum Volatility
    """
    
    optimization_method: str = 'max_sharpe'  # 'max_sharpe', 'min_vol', 'hrp', 'risk_parity'
    risk_free_rate: float = 0.02  # Annual risk-free rate
    target_return: Optional[float] = None  # For efficient frontier
    max_weight: float = 0.30  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    lookback_days: int = 252  # Days for calculating returns/covariance
    confidence_level: float = 0.95  # For CVaR calculations
    
    def __post_init__(self):
        """Initialize internal state"""
        if not PYPFOPT_AVAILABLE:
            raise ImportError("PyPortfolioOpt is required but not installed")
            
        self.is_fitted = False
        self.weights = None
        self.expected_returns = None
        self.cov_matrix = None
        self.performance_metrics = None
        self.assets = None
        
    def fit(self, prices_data: Dict[str, pd.DataFrame], 
            market_views: Optional[Dict[str, float]] = None) -> 'PortfolioOptimizer':
        """
        Fit the portfolio optimization model
        
        Args:
            prices_data: Dictionary of asset symbols to OHLCV DataFrames
            market_views: Optional market views for Black-Litterman (symbol -> expected return)
            
        Returns:
            Self for chaining
        """
        # Extract closing prices
        prices_df = self._prepare_price_data(prices_data)
        self.assets = list(prices_df.columns)
        
        # Calculate expected returns
        if self.optimization_method == 'black_litterman' and market_views:
            self.expected_returns = self._black_litterman_returns(prices_df, market_views)
        else:
            self.expected_returns = expected_returns.mean_historical_return(
                prices_df, 
                frequency=252
            )
        
        # Calculate covariance matrix
        self.cov_matrix = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()
        
        # Optimize portfolio
        self.weights = self._optimize_portfolio(prices_df)
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_metrics()
        
        self.is_fitted = True
        logger.info(f"Portfolio optimized using {self.optimization_method} method")
        
        return self
    
    def predict(self, current_prices: Dict[str, float], 
               portfolio_value: float = 100000) -> Dict[str, Any]:
        """
        Generate portfolio allocation recommendations
        
        Args:
            current_prices: Current prices for each asset
            portfolio_value: Total portfolio value to allocate
            
        Returns:
            Dictionary with allocation details
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get discrete allocation
        da = DiscreteAllocation(
            self.weights, 
            current_prices, 
            total_portfolio_value=portfolio_value
        )
        
        allocation, leftover = da.greedy_portfolio()
        
        # Calculate actual weights
        actual_weights = {}
        total_allocated = 0
        
        for asset, shares in allocation.items():
            value = shares * current_prices[asset]
            actual_weights[asset] = value / portfolio_value
            total_allocated += value
        
        return {
            'target_weights': self.weights,
            'shares_to_buy': allocation,
            'actual_weights': actual_weights,
            'leftover_cash': leftover,
            'total_allocated': total_allocated,
            'allocation_efficiency': total_allocated / portfolio_value
        }
    
    def rebalance_signals(self, prices_data: Dict[str, pd.DataFrame], 
                         current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Generate rebalancing signals based on drift from target weights
        
        Args:
            prices_data: Historical price data
            current_weights: Current portfolio weights
            
        Returns:
            Dictionary of rebalancing trades (positive = buy, negative = sell)
        """
        # Refit if needed based on rebalance frequency
        if self._should_rebalance(prices_data):
            self.fit(prices_data)
        
        # Calculate weight differences
        trades = {}
        for asset in self.assets:
            target = self.weights.get(asset, 0)
            current = current_weights.get(asset, 0)
            trades[asset] = target - current
        
        return trades
    
    def _prepare_price_data(self, prices_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare price data for optimization"""
        prices_list = []
        
        for symbol, df in prices_data.items():
            if 'close' in df.columns:
                prices_list.append(df['close'].rename(symbol))
        
        prices_df = pd.concat(prices_list, axis=1).dropna()
        
        # Use only recent data based on lookback
        if len(prices_df) > self.lookback_days:
            prices_df = prices_df.iloc[-self.lookback_days:]
        
        return prices_df
    
    def _optimize_portfolio(self, prices_df: pd.DataFrame) -> Dict[str, float]:
        """Optimize portfolio based on selected method"""
        
        if self.optimization_method == 'max_sharpe':
            return self._max_sharpe_optimization()
        
        elif self.optimization_method == 'min_vol':
            return self._min_volatility_optimization()
        
        elif self.optimization_method == 'hrp':
            return self._hierarchical_risk_parity(prices_df)
        
        elif self.optimization_method == 'risk_parity':
            return self._risk_parity_optimization()
        
        elif self.optimization_method == 'efficient_frontier':
            return self._efficient_frontier_optimization()
        
        elif self.optimization_method == 'cvar':
            return self._cvar_optimization(prices_df)
        
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _max_sharpe_optimization(self) -> Dict[str, float]:
        """Maximum Sharpe ratio optimization"""
        ef = EfficientFrontier(
            self.expected_returns, 
            self.cov_matrix,
            weight_bounds=(self.min_weight, self.max_weight)
        )
        
        weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        return ef.clean_weights()
    
    def _min_volatility_optimization(self) -> Dict[str, float]:
        """Minimum volatility optimization"""
        ef = EfficientFrontier(
            self.expected_returns, 
            self.cov_matrix,
            weight_bounds=(self.min_weight, self.max_weight)
        )
        
        weights = ef.min_volatility()
        return ef.clean_weights()
    
    def _hierarchical_risk_parity(self, prices_df: pd.DataFrame) -> Dict[str, float]:
        """Hierarchical Risk Parity optimization"""
        hrp = HRPOpt(prices_df)
        weights = hrp.optimize()
        
        # Apply weight constraints
        cleaned_weights = {}
        for asset, weight in weights.items():
            cleaned_weights[asset] = np.clip(weight, self.min_weight, self.max_weight)
        
        # Renormalize
        total = sum(cleaned_weights.values())
        return {k: v/total for k, v in cleaned_weights.items()}
    
    def _risk_parity_optimization(self) -> Dict[str, float]:
        """Risk parity optimization - equal risk contribution"""
        n_assets = len(self.assets)
        
        # Objective: minimize difference between risk contributions
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            marginal_contrib = self.cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal contribution
            target_contrib = 1.0 / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return dict(zip(self.assets, result.x))
    
    def _efficient_frontier_optimization(self) -> Dict[str, float]:
        """Efficient frontier optimization for target return"""
        ef = EfficientFrontier(
            self.expected_returns, 
            self.cov_matrix,
            weight_bounds=(self.min_weight, self.max_weight)
        )
        
        if self.target_return:
            weights = ef.efficient_return(target_return=self.target_return)
        else:
            # Default to max Sharpe if no target return specified
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        
        return ef.clean_weights()
    
    def _cvar_optimization(self, prices_df: pd.DataFrame) -> Dict[str, float]:
        """Conditional Value at Risk (CVaR) optimization"""
        # Calculate returns
        returns = prices_df.pct_change().dropna()
        
        ef_cvar = EfficientCVaR(
            self.expected_returns,
            returns,
            weight_bounds=(self.min_weight, self.max_weight),
            alpha=self.confidence_level
        )
        
        weights = ef_cvar.min_cvar()
        return dict(zip(self.assets, weights))
    
    def _black_litterman_returns(self, prices_df: pd.DataFrame, 
                                market_views: Dict[str, float]) -> pd.Series:
        """Calculate Black-Litterman expected returns"""
        # Market equilibrium returns
        market_weights = self._get_market_weights(prices_df)
        implied_returns = BlackLittermanModel(
            self.cov_matrix,
            pi="market",
            market_caps=market_weights,
            risk_aversion=1
        ).bl_implied_returns()
        
        # Convert views to required format
        viewdict = {}
        for asset, view in market_views.items():
            if asset in self.assets:
                viewdict[asset] = view
        
        # Black-Litterman model
        bl = BlackLittermanModel(
            self.cov_matrix,
            pi=implied_returns,
            absolute_views=viewdict
        )
        
        return bl.bl_returns()
    
    def _get_market_weights(self, prices_df: pd.DataFrame) -> Dict[str, float]:
        """Get market cap weights (simplified - uses equal weights)"""
        # In practice, would fetch actual market caps
        n_assets = len(self.assets)
        return {asset: 1/n_assets for asset in self.assets}
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        weights_array = np.array([self.weights.get(asset, 0) for asset in self.assets])
        
        # Expected return
        portfolio_return = weights_array @ self.expected_returns.values
        
        # Volatility
        portfolio_vol = np.sqrt(weights_array @ self.cov_matrix @ weights_array)
        
        # Sharpe ratio
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Diversification ratio
        weighted_avg_vol = np.sum(
            weights_array * np.sqrt(np.diag(self.cov_matrix))
        )
        div_ratio = weighted_avg_vol / portfolio_vol
        
        # Maximum drawdown estimation (using normal approximation)
        time_horizon = 252  # 1 year
        max_dd_estimate = -2.24 * portfolio_vol * np.sqrt(time_horizon/252)
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'diversification_ratio': div_ratio,
            'max_drawdown_estimate': max_dd_estimate,
            'effective_assets': 1 / np.sum(weights_array**2),  # Herfindahl index
            'max_weight': max(self.weights.values()),
            'min_weight': min([w for w in self.weights.values() if w > 0])
        }
    
    def _should_rebalance(self, prices_data: Dict[str, pd.DataFrame]) -> bool:
        """Check if rebalancing is needed based on frequency"""
        # Simplified - always rebalance for now
        # In practice, would check dates and frequency
        return True
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        params = {
            'optimization_method': self.optimization_method,
            'risk_free_rate': self.risk_free_rate,
            'max_weight': self.max_weight,
            'min_weight': self.min_weight,
            'lookback_days': self.lookback_days,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            params['weights'] = self.weights
            params['performance_metrics'] = self.performance_metrics
            params['assets'] = self.assets
            
        return params
    
    def efficient_frontier_plot_data(self, n_points: int = 100) -> Dict[str, List[float]]:
        """Generate data for efficient frontier visualization"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Generate efficient frontier
        ef = EfficientFrontier(
            self.expected_returns, 
            self.cov_matrix,
            weight_bounds=(0, 1)  # Relaxed for visualization
        )
        
        # Get range of returns
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        risks = []
        returns = []
        
        for target in target_returns:
            try:
                ef.efficient_return(target_return=target)
                ret, vol, _ = ef.portfolio_performance()
                returns.append(ret)
                risks.append(vol)
            except:
                continue
        
        # Add current portfolio point
        current_perf = self.performance_metrics
        
        return {
            'frontier_returns': returns,
            'frontier_risks': risks,
            'current_return': current_perf['expected_return'],
            'current_risk': current_perf['volatility'],
            'asset_returns': self.expected_returns.tolist(),
            'asset_risks': np.sqrt(np.diag(self.cov_matrix)).tolist(),
            'asset_names': self.assets
        }