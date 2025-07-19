
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RiskModels:
    def __init__(self):
        pass
    
    def value_at_risk(self, returns: pd.Series, confidence_level: float = 0.05, 
                     method: str = "historical") -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR)
        Methods: historical, parametric, monte_carlo
        """
        logger.info(f"📊 Calculating VaR using {method} method")
        
        if method == "historical":
            var = np.percentile(returns, confidence_level * 100)
        
        elif method == "parametric":
            mu = returns.mean()
            sigma = returns.std()
            var = mu + sigma * stats.norm.ppf(confidence_level)
        
        elif method == "monte_carlo":
            n_simulations = 10000
            simulated_returns = np.random.normal(returns.mean(), returns.std(), n_simulations)
            var = np.percentile(simulated_returns, confidence_level * 100)
        
        else:
            raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")
        
        return {
            "var": var,
            "confidence_level": confidence_level,
            "method": method,
            "var_absolute": abs(var) if var < 0 else var
        }
    
    def expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.05) -> Dict[str, float]:
        """
        Calculate Expected Shortfall (Conditional VaR)
        """
        logger.info("📉 Calculating Expected Shortfall (CVaR)")
        
        var_result = self.value_at_risk(returns, confidence_level, method="historical")
        var_threshold = var_result["var"]
        
        # Calculate average of losses beyond VaR
        tail_losses = returns[returns <= var_threshold]
        expected_shortfall = tail_losses.mean() if len(tail_losses) > 0 else var_threshold
        
        return {
            "expected_shortfall": expected_shortfall,
            "var": var_threshold,
            "confidence_level": confidence_level,
            "tail_observations": len(tail_losses)
        }
    
    def maximum_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics
        """
        logger.info("📈 Calculating Maximum Drawdown")
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_duration = self._calculate_drawdown_duration(drawdown)
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "current_drawdown": drawdown.iloc[-1],
            "recovery_factor": returns.sum() / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate the longest drawdown duration in periods"""
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for dd in is_drawdown:
            if dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def stress_testing(self, portfolio_returns: pd.Series, stress_scenarios: Dict[str, float]) -> Dict[str, Dict]:
        """
        Perform stress testing on portfolio
        stress_scenarios: Dict of scenario names to return shock multipliers
        """
        logger.info("⚠️ Running stress testing scenarios")
        
        results = {}
        base_var = self.value_at_risk(portfolio_returns, 0.05)["var"]
        
        for scenario_name, shock_multiplier in stress_scenarios.items():
            stressed_returns = portfolio_returns * shock_multiplier
            
            stressed_var = self.value_at_risk(stressed_returns, 0.05)["var"]
            stressed_es = self.expected_shortfall(stressed_returns, 0.05)["expected_shortfall"]
            
            results[scenario_name] = {
                "shock_multiplier": shock_multiplier,
                "stressed_var": stressed_var,
                "stressed_es": stressed_es,
                "var_change": stressed_var - base_var,
                "worst_case_loss": stressed_returns.min()
            }
        
        return results
    
    def risk_attribution(self, portfolio_weights: Dict[str, float], 
                        asset_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk contribution by asset
        """
        logger.info("🎯 Calculating risk attribution")
        
        weights = np.array([portfolio_weights.get(asset, 0) for asset in asset_returns.columns])
        cov_matrix = asset_returns.cov()
        
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Marginal risk contribution
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
        
        # Component risk contribution
        component_contrib = weights * marginal_contrib
        
        # Percentage contribution
        percentage_contrib = component_contrib / portfolio_variance
        
        risk_attribution = {}
        for i, asset in enumerate(asset_returns.columns):
            risk_attribution[asset] = {
                "weight": weights[i],
                "marginal_contribution": marginal_contrib[i],
                "component_contribution": component_contrib[i],
                "percentage_contribution": percentage_contrib[i]
            }
        
        return risk_attribution

risk_models = RiskModels()
