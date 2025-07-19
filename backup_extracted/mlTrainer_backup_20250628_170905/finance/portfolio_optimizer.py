
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% default
    
    def modern_portfolio_theory(self, returns: pd.DataFrame, target_return: float = None) -> Dict:
        """
        Implement Markowitz Mean-Variance Optimization
        """
        logger.info("🎯 Running Modern Portfolio Theory optimization")
        
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252
        n_assets = len(mean_returns)
        
        def portfolio_stats(weights, returns, cov_matrix):
            portfolio_return = np.sum(returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
            return portfolio_return, portfolio_std, sharpe_ratio
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Maximize Sharpe ratio
        def negative_sharpe(weights):
            return -portfolio_stats(weights, mean_returns, cov_matrix)[2]
        
        result = minimize(negative_sharpe, n_assets * [1. / n_assets], 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        ret, std, sharpe = portfolio_stats(optimal_weights, mean_returns, cov_matrix)
        
        return {
            "weights": dict(zip(returns.columns, optimal_weights)),
            "expected_return": ret,
            "volatility": std,
            "sharpe_ratio": sharpe,
            "method": "modern_portfolio_theory"
        }
    
    def black_litterman(self, returns: pd.DataFrame, market_caps: Dict, 
                       views: Dict = None, confidence: float = 0.5) -> Dict:
        """
        Black-Litterman model with investor views
        """
        logger.info("🧠 Running Black-Litterman optimization")
        
        # Market equilibrium returns
        cov_matrix = returns.cov() * 252
        market_weights = np.array([market_caps.get(asset, 1) for asset in returns.columns])
        market_weights = market_weights / market_weights.sum()
        
        # Risk aversion parameter (typical value)
        risk_aversion = 3.0
        
        # Equilibrium returns
        equilibrium_returns = risk_aversion * np.dot(cov_matrix, market_weights)
        
        if views:
            # Incorporate investor views (simplified implementation)
            P = np.identity(len(returns.columns))  # View matrix
            Q = np.array([views.get(asset, 0) for asset in returns.columns])  # View returns
            tau = 0.025  # Uncertainty scaling factor
            omega = np.identity(len(Q)) * confidence  # Confidence in views
            
            # Black-Litterman formula
            M1 = np.linalg.inv(tau * cov_matrix)
            M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
            M3 = np.dot(np.linalg.inv(tau * cov_matrix), equilibrium_returns)
            M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
            
            mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
            cov_bl = np.linalg.inv(M1 + M2)
        else:
            mu_bl = equilibrium_returns
            cov_bl = cov_matrix
        
        # Optimize with Black-Litterman inputs
        def negative_utility(weights):
            return -(np.dot(weights, mu_bl) - 0.5 * risk_aversion * np.dot(weights.T, np.dot(cov_bl, weights)))
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(returns.columns)))
        
        result = minimize(negative_utility, market_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        expected_return = np.dot(optimal_weights, mu_bl)
        volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_bl, optimal_weights)))
        
        return {
            "weights": dict(zip(returns.columns, optimal_weights)),
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": (expected_return - self.risk_free_rate) / volatility,
            "method": "black_litterman"
        }
    
    def risk_parity(self, returns: pd.DataFrame) -> Dict:
        """
        Risk parity portfolio - equal risk contribution
        """
        logger.info("⚖️ Running Risk Parity optimization")
        
        cov_matrix = returns.cov() * 252
        n_assets = len(returns.columns)
        
        def risk_budget_objective(weights, cov_matrix):
            """Minimize the difference between marginal risk contributions"""
            weights = np.array(weights)
            sigma = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            MRC = np.dot(cov_matrix, weights) / sigma
            TRC = weights * MRC
            risk_target = np.ones(n_assets) / n_assets
            return np.sum((TRC - risk_target) ** 2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 0.99) for _ in range(n_assets))
        
        result = minimize(risk_budget_objective, n_assets * [1. / n_assets],
                         method='SLSQP', args=(cov_matrix,), bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        portfolio_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
        
        return {
            "weights": dict(zip(returns.columns, optimal_weights)),
            "volatility": portfolio_vol,
            "method": "risk_parity"
        }

portfolio_optimizer = PortfolioOptimizer()
