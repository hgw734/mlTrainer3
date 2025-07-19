"""
Risk metrics module for institutional-grade risk assessment.
Includes volatility analysis, drawdown metrics, and risk-adjusted returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """
    Institutional-grade risk analyzer for volatility, drawdown,
    and risk-adjusted performance metrics.
    """
    
    def __init__(self):
        """Initialize risk analyzer"""
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.trading_days = 252
        
    def analyze(self, data: pd.DataFrame, symbol: str = '') -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis
        
        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol
            
        Returns:
            Dictionary of risk metrics and scores
        """
        try:
            if data is None or data.empty or len(data) < 20:
                return self._empty_result()
            
            results = {
                'symbol': symbol,
                'risk_score': 50.0
            }
            
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            # Volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(returns)
            results.update(volatility_metrics)
            
            # Drawdown analysis
            drawdown_metrics = self._calculate_drawdown_metrics(data['close'])
            results.update(drawdown_metrics)
            
            # Risk-adjusted returns
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(returns)
            results.update(risk_adjusted_metrics)
            
            # Value at Risk (VaR)
            var_metrics = self._calculate_var_metrics(returns)
            results.update(var_metrics)
            
            # Calculate composite risk score
            results['risk_score'] = self._calculate_risk_score(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Risk analysis failed for {symbol}: {e}")
            return self._empty_result()
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility-based risk metrics"""
        try:
            if len(returns) < 10:
                return {}
            
            # Annualized volatility
            volatility = returns.std() * np.sqrt(self.trading_days)
            
            # Rolling volatility (20-day)
            rolling_vol = returns.rolling(20).std() * np.sqrt(self.trading_days)
            current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else volatility
            
            # Volatility trend
            vol_trend = rolling_vol.diff().tail(5).mean() if len(rolling_vol) > 5 else 0
            
            return {
                'volatility': volatility,
                'current_volatility': current_vol,
                'volatility_trend': vol_trend,
                'volatility_percentile': self._calculate_percentile(rolling_vol, current_vol)
            }
            
        except Exception as e:
            logger.error(f"Volatility metrics calculation failed: {e}")
            return {}
    
    def _calculate_drawdown_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-based risk metrics"""
        try:
            if len(prices) < 10:
                return {}
            
            # Calculate running maximum
            running_max = prices.expanding().max()
            
            # Calculate drawdowns
            drawdowns = (prices - running_max) / running_max
            
            # Maximum drawdown
            max_drawdown = drawdowns.min()
            
            # Current drawdown
            current_drawdown = drawdowns.iloc[-1]
            
            # Average drawdown
            negative_drawdowns = drawdowns[drawdowns < 0]
            avg_drawdown = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0
            
            # Drawdown duration (days in current drawdown)
            drawdown_duration = 0
            for i in range(len(drawdowns)-1, -1, -1):
                if drawdowns.iloc[i] < 0:
                    drawdown_duration += 1
                else:
                    break
            
            return {
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'avg_drawdown': avg_drawdown,
                'drawdown_duration': drawdown_duration
            }
            
        except Exception as e:
            logger.error(f"Drawdown metrics calculation failed: {e}")
            return {}
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        try:
            if len(returns) < 10:
                return {}
            
            # Annualized return
            annual_return = (1 + returns.mean()) ** self.trading_days - 1
            
            # Sharpe ratio
            excess_return = annual_return - self.risk_free_rate
            volatility = returns.std() * np.sqrt(self.trading_days)
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = returns[returns < 0]
            downside_dev = negative_returns.std() * np.sqrt(self.trading_days)
            sortino_ratio = excess_return / downside_dev if downside_dev > 0 else 0
            
            # Calmar ratio (return to max drawdown)
            # We'll estimate max drawdown from returns
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_dd = abs(drawdowns.min())
            calmar_ratio = annual_return / max_dd if max_dd > 0 else 0
            
            return {
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio
            }
            
        except Exception as e:
            logger.error(f"Risk-adjusted metrics calculation failed: {e}")
            return {}
    
    def _calculate_var_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk metrics"""
        try:
            if len(returns) < 20:
                return {}
            
            # 1-day VaR at 95% confidence
            var_95 = np.percentile(returns, 5)
            
            # 1-day VaR at 99% confidence
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall (Conditional VaR)
            var_95_threshold = returns <= var_95
            expected_shortfall = returns[var_95_threshold].mean() if var_95_threshold.any() else var_95
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall': expected_shortfall
            }
            
        except Exception as e:
            logger.error(f"VaR metrics calculation failed: {e}")
            return {}
    
    def _calculate_percentile(self, series: pd.Series, value: float) -> float:
        """Calculate percentile ranking of a value within a series"""
        try:
            if len(series) < 2:
                return 50.0
            
            cleaned_series = series.dropna()
            if len(cleaned_series) == 0:
                return 50.0
            
            percentile = (cleaned_series < value).sum() / len(cleaned_series) * 100
            return percentile
            
        except Exception as e:
            logger.error(f"Percentile calculation failed: {e}")
            return 50.0
    
    def _calculate_risk_score(self, results: Dict[str, Any]) -> float:
        """Calculate composite risk score (higher = less risky)"""
        try:
            score = 50.0
            
            # Volatility scoring (25 points)
            volatility = results.get('volatility', 0.3)
            if volatility < 0.15:  # Low volatility
                score += 12
            elif volatility < 0.25:
                score += 8
            elif volatility < 0.4:
                score += 4
            elif volatility > 0.6:
                score -= 12
            elif volatility > 0.5:
                score -= 8
            
            # Drawdown scoring (25 points)
            max_drawdown = results.get('max_drawdown', -0.2)
            if max_drawdown > -0.05:  # Small drawdowns
                score += 12
            elif max_drawdown > -0.1:
                score += 8
            elif max_drawdown > -0.2:
                score += 4
            elif max_drawdown < -0.4:
                score -= 12
            elif max_drawdown < -0.3:
                score -= 8
            
            # Sharpe ratio scoring (25 points)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            if sharpe_ratio > 2.0:
                score += 12
            elif sharpe_ratio > 1.5:
                score += 10
            elif sharpe_ratio > 1.0:
                score += 8
            elif sharpe_ratio > 0.5:
                score += 4
            elif sharpe_ratio < -0.5:
                score -= 8
            
            # VaR scoring (25 points)
            var_95 = results.get('var_95', -0.05)
            if var_95 > -0.02:  # Low downside risk
                score += 12
            elif var_95 > -0.03:
                score += 8
            elif var_95 > -0.05:
                score += 4
            elif var_95 < -0.1:
                score -= 8
            elif var_95 < -0.08:
                score -= 4
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Risk score calculation failed: {e}")
            return 50.0
    
    def get_risk_category(self, risk_score: float) -> str:
        """Get risk category based on risk score"""
        if risk_score >= 80:
            return 'LOW_RISK'
        elif risk_score >= 65:
            return 'MODERATE_LOW_RISK'
        elif risk_score >= 50:
            return 'MODERATE_RISK'
        elif risk_score >= 35:
            return 'MODERATE_HIGH_RISK'
        else:
            return 'HIGH_RISK'
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'risk_score': 50.0,
            'volatility': 0.3,
            'current_volatility': 0.3,
            'max_drawdown': -0.2,
            'current_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'var_95': -0.05,
            'var_99': -0.08
        }