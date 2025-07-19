"""
Continuous optimization engine that monitors performance and automatically
adjusts all parameters throughout the application for maximum profitability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Track performance metrics for parameter optimization"""
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    timestamp: datetime

class ContinuousOptimizer:
    """
    Continuously monitors performance and optimizes all parameters
    across the entire application for maximum profitability.
    """
    
    def __init__(self, optimization_window: int = 100):
        """Initialize continuous optimizer"""
        self.optimization_window = optimization_window
        self.performance_history = deque(maxlen=optimization_window)
        self.parameter_history = deque(maxlen=optimization_window)
        
        # Current optimized parameters
        self.current_params = self._initialize_baseline_params()
        
        # Performance tracking
        self.last_optimization = datetime.now()
        self.optimization_frequency = timedelta(hours=1)  # Optimize every hour
        
        # Learning rates for different parameter types
        self.learning_rates = {
            'threshold_params': 0.05,      # Conservative for thresholds
            'weight_params': 0.03,         # Very conservative for weights
            'exit_params': 0.08,           # More aggressive for exit timing
            'volume_params': 0.06          # Moderate for volume requirements
        }
        
    def _initialize_baseline_params(self) -> Dict[str, Any]:
        """Initialize baseline parameters for optimization"""
        return {
            # Scoring thresholds (continuously optimized)
            'min_score_multipliers': {
                'bull_market': 0.70,
                'bear_market': 1.20,
                'volatile_market': 1.10,
                'neutral_market': 1.00
            },
            
            # Component weights (continuously optimized)
            'adaptive_weights': {
                'bull_market': {
                    'momentum': 0.45, 'technical': 0.30, 'fundamental': 0.15, 'sentiment': 0.10
                },
                'bear_market': {
                    'momentum': 0.25, 'technical': 0.40, 'fundamental': 0.25, 'sentiment': 0.10
                },
                'volatile_market': {
                    'momentum': 0.35, 'technical': 0.35, 'fundamental': 0.20, 'sentiment': 0.10
                },
                'neutral_market': {
                    'momentum': 0.30, 'technical': 0.40, 'fundamental': 0.25, 'sentiment': 0.05
                }
            },
            
            # Exit parameters (continuously optimized)
            'exit_optimization': {
                'bull_market': {
                    'stop_loss_multiplier': 1.25,      # -2.5% base * 1.25 = -3.125%
                    'momentum_exit_multiplier': 1.20,
                    'profit_protection_multiplier': 1.30,
                    'trailing_stop_multiplier': 1.40
                },
                'bear_market': {
                    'stop_loss_multiplier': 0.75,      # Tighter stops
                    'momentum_exit_multiplier': 0.60,
                    'profit_protection_multiplier': 0.60,
                    'trailing_stop_multiplier': 0.50
                },
                'volatile_market': {
                    'stop_loss_multiplier': 1.00,
                    'momentum_exit_multiplier': 0.80,
                    'profit_protection_multiplier': 0.90,
                    'trailing_stop_multiplier': 0.80
                },
                'neutral_market': {
                    'stop_loss_multiplier': 1.00,
                    'momentum_exit_multiplier': 0.90,
                    'profit_protection_multiplier': 0.80,
                    'trailing_stop_multiplier': 0.90
                }
            },
            
            # Volume requirements (continuously optimized)
            'volume_multipliers': {
                'bull_market': 2.5,
                'bear_market': 3.5,
                'volatile_market': 4.0,
                'neutral_market': 3.0
            },
            
            # Technical thresholds (continuously optimized)
            'technical_thresholds': {
                'rsi_overbought': {'bull': 75, 'bear': 65, 'volatile': 75, 'neutral': 70},
                'rsi_oversold': {'bull': 25, 'bear': 30, 'volatile': 25, 'neutral': 30},
                'momentum_strength': {'bull': 4.0, 'bear': 6.0, 'volatile': 5.0, 'neutral': 4.5}
            }
        }
    
    def update_performance(self, trades_data: List[Dict], market_regime: str) -> None:
        """Update performance metrics and trigger optimization if needed"""
        if not trades_data:
            return
            
        # Calculate current performance metrics
        metrics = self._calculate_performance_metrics(trades_data)
        
        # Store performance and parameter snapshot
        self.performance_history.append(metrics)
        self.parameter_history.append({
            'timestamp': datetime.now(),
            'regime': market_regime,
            'params': self.current_params.copy()
        })
        
        # Check if optimization is needed
        if self._should_optimize():
            self._optimize_parameters(market_regime)
            self.last_optimization = datetime.now()
    
    def _calculate_performance_metrics(self, trades_data: List[Dict]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not trades_data:
            return PerformanceMetrics(0, 0, 0, 0, 0, datetime.now())
        
        df = pd.DataFrame(trades_data)
        
        # Win rate
        winning_trades = len(df[df['return_pct'] > 0])
        win_rate = winning_trades / len(df) if len(df) > 0 else 0
        
        # Average return
        avg_return = df['return_pct'].mean()
        
        # Sharpe ratio (annualized)
        returns = df['return_pct'] / 100  # Convert to decimal
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return PerformanceMetrics(
            win_rate=win_rate,
            avg_return=avg_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=len(df),
            timestamp=datetime.now()
        )
    
    def _should_optimize(self) -> bool:
        """Determine if optimization should be triggered"""
        # Time-based optimization
        if datetime.now() - self.last_optimization >= self.optimization_frequency:
            return True
            
        # Performance-based optimization
        if len(self.performance_history) >= 20:
            recent_performance = list(self.performance_history)[-10:]
            older_performance = list(self.performance_history)[-20:-10]
            
            recent_avg = np.mean([p.avg_return for p in recent_performance])
            older_avg = np.mean([p.avg_return for p in older_performance])
            
            # Optimize if performance is declining
            if recent_avg < older_avg * 0.95:  # 5% decline threshold
                return True
        
        return False
    
    def _optimize_parameters(self, market_regime: str) -> None:
        """Optimize parameters based on recent performance to maximize profits and minimize risk"""
        if len(self.performance_history) < 10:
            return
            
        logger.info(f"Starting aggressive profit-risk optimization for {market_regime} market")
        
        # Get recent performance trends
        recent_metrics = list(self.performance_history)[-10:]
        
        # Calculate risk-adjusted performance metrics
        recent_returns = [m.avg_return for m in recent_metrics]
        recent_drawdowns = [abs(m.max_drawdown) for m in recent_metrics]
        recent_sharpe = [m.sharpe_ratio for m in recent_metrics]
        
        avg_return = np.mean(recent_returns)
        avg_drawdown = np.mean(recent_drawdowns)
        avg_sharpe = np.mean(recent_sharpe)
        
        # Risk-reward optimization targets
        target_return = 5.0  # 5% average return target
        max_acceptable_drawdown = 0.08  # 8% max drawdown
        min_sharpe_ratio = 1.5  # Minimum Sharpe ratio
        
        # Aggressive optimization based on risk-reward profile
        self._optimize_threshold_parameters(recent_metrics, market_regime, avg_return, avg_drawdown)
        self._optimize_weight_parameters(recent_metrics, market_regime, avg_sharpe)
        self._optimize_exit_parameters(recent_metrics, market_regime, avg_return, avg_drawdown)
        self._optimize_volume_parameters(recent_metrics, market_regime, avg_return)
        self._optimize_risk_management_parameters(recent_metrics, market_regime, avg_drawdown, avg_sharpe)
        
        logger.info(f"Profit-risk optimization completed: Return={avg_return:.2f}%, Drawdown={avg_drawdown:.2f}%, Sharpe={avg_sharpe:.2f}")
    
    def _optimize_threshold_parameters(self, metrics: List[PerformanceMetrics], regime: str, avg_return: float, avg_drawdown: float) -> None:
        """Optimize score threshold parameters to maximize profit-risk ratio"""
        win_rate = np.mean([m.win_rate for m in metrics])
        current_multiplier = self.current_params['min_score_multipliers'][regime]
        
        # Calculate profit-risk ratio
        profit_risk_ratio = avg_return / max(avg_drawdown, 0.01)  # Avoid division by zero
        
        # Aggressive optimization for maximum profit with controlled risk
        if profit_risk_ratio < 30:  # Poor profit-risk ratio
            if win_rate < 0.5:  # Low win rate - be much more selective
                adjustment = self.learning_rates['threshold_params'] * 2.0  # Double the adjustment
                self.current_params['min_score_multipliers'][regime] = min(2.0, current_multiplier + adjustment)
            elif avg_drawdown > 0.10:  # High risk - increase selectivity
                adjustment = self.learning_rates['threshold_params'] * 1.5
                self.current_params['min_score_multipliers'][regime] = min(2.0, current_multiplier + adjustment)
        
        elif profit_risk_ratio > 60 and win_rate > 0.7:  # Excellent profit-risk ratio
            # Lower thresholds to capture more profitable opportunities
            adjustment = -self.learning_rates['threshold_params'] * 1.5
            self.current_params['min_score_multipliers'][regime] = max(0.3, current_multiplier + adjustment)
        
        elif avg_return > 8.0 and avg_drawdown < 0.05:  # High return, low risk
            # Aggressively lower thresholds for more volume
            adjustment = -self.learning_rates['threshold_params'] * 2.0
            self.current_params['min_score_multipliers'][regime] = max(0.2, current_multiplier + adjustment)
    
    def _optimize_weight_parameters(self, metrics: List[PerformanceMetrics], regime: str) -> None:
        """Optimize component weight parameters"""
        avg_return = np.mean([m.avg_return for m in metrics])
        
        weights = self.current_params['adaptive_weights'][regime]
        learning_rate = self.learning_rates['weight_params']
        
        # Adjust weights based on regime-specific performance patterns
        if regime == 'bull_market' and avg_return < 3.0:
            # Increase momentum weight in bull markets if underperforming
            weights['momentum'] = min(0.60, weights['momentum'] + learning_rate)
            weights['technical'] = max(0.20, weights['technical'] - learning_rate/2)
            
        elif regime == 'bear_market' and avg_return < 1.0:
            # Increase technical weight in bear markets if underperforming
            weights['technical'] = min(0.50, weights['technical'] + learning_rate)
            weights['momentum'] = max(0.15, weights['momentum'] - learning_rate/2)
            
        elif regime == 'volatile_market':
            # Balance weights more evenly in volatile markets
            target_balance = 0.30
            for component in ['momentum', 'technical']:
                if weights[component] > target_balance + 0.10:
                    weights[component] = max(target_balance, weights[component] - learning_rate)
                elif weights[component] < target_balance - 0.10:
                    weights[component] = min(target_balance + 0.10, weights[component] + learning_rate)
        
        # Ensure weights sum to 1.0
        total = sum(weights.values())
        for key in weights:
            weights[key] = weights[key] / total
    
    def _optimize_exit_parameters(self, metrics: List[PerformanceMetrics], regime: str) -> None:
        """Optimize exit timing parameters"""
        avg_return = np.mean([m.avg_return for m in metrics])
        max_drawdown = np.mean([m.max_drawdown for m in metrics])
        
        exit_params = self.current_params['exit_optimization'][regime]
        learning_rate = self.learning_rates['exit_params']
        
        # If drawdown is too high, tighten stops
        if max_drawdown < -0.15:  # More than 15% drawdown
            exit_params['stop_loss_multiplier'] = max(0.5, exit_params['stop_loss_multiplier'] - learning_rate)
            exit_params['trailing_stop_multiplier'] = max(0.4, exit_params['trailing_stop_multiplier'] - learning_rate)
            
        # If returns are low, adjust profit protection
        elif avg_return < 1.5:
            exit_params['profit_protection_multiplier'] = max(0.5, exit_params['profit_protection_multiplier'] - learning_rate)
            
        # If performance is good, allow more room
        elif avg_return > 4.0 and max_drawdown > -0.08:
            exit_params['stop_loss_multiplier'] = min(1.5, exit_params['stop_loss_multiplier'] + learning_rate/2)
            exit_params['profit_protection_multiplier'] = min(1.5, exit_params['profit_protection_multiplier'] + learning_rate/2)
    
    def _optimize_volume_parameters(self, metrics: List[PerformanceMetrics], regime: str) -> None:
        """Optimize volume requirement parameters"""
        win_rate = np.mean([m.win_rate for m in metrics])
        total_trades = np.mean([m.total_trades for m in metrics])
        
        current_multiplier = self.current_params['volume_multipliers'][regime]
        learning_rate = self.learning_rates['volume_params']
        
        # If too few trades, lower volume requirements
        if total_trades < 5:
            self.current_params['volume_multipliers'][regime] = max(1.5, current_multiplier - learning_rate)
            
        # If win rate is low, increase volume requirements
        elif win_rate < 0.4:
            self.current_params['volume_multipliers'][regime] = min(6.0, current_multiplier + learning_rate)
            
        # If performance is good, slight relaxation
        elif win_rate > 0.6 and total_trades > 10:
            self.current_params['volume_multipliers'][regime] = max(1.5, current_multiplier - learning_rate/2)
    
    def get_optimized_parameters(self, market_regime: str) -> Dict[str, Any]:
        """Get current optimized parameters for the specified market regime"""
        return {
            'min_score_multiplier': self.current_params['min_score_multipliers'].get(market_regime, 1.0),
            'component_weights': self.current_params['adaptive_weights'].get(market_regime, {}),
            'exit_multipliers': self.current_params['exit_optimization'].get(market_regime, {}),
            'volume_multiplier': self.current_params['volume_multipliers'].get(market_regime, 3.0),
            'technical_thresholds': self.current_params['technical_thresholds'],
            'last_optimization': self.last_optimization,
            'optimization_stats': {
                'total_optimizations': len(self.performance_history),
                'current_performance': self.performance_history[-1] if self.performance_history else None
            }
        }
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and performance trends"""
        if not self.performance_history:
            return {'status': 'initializing', 'message': 'Collecting initial performance data'}
        
        recent_performance = list(self.performance_history)[-5:] if len(self.performance_history) >= 5 else list(self.performance_history)
        
        avg_return = np.mean([p.avg_return for p in recent_performance])
        avg_win_rate = np.mean([p.win_rate for p in recent_performance])
        
        return {
            'status': 'optimizing',
            'last_optimization': self.last_optimization,
            'next_optimization': self.last_optimization + self.optimization_frequency,
            'recent_performance': {
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'total_trades': sum([p.total_trades for p in recent_performance])
            },
            'optimization_frequency': str(self.optimization_frequency),
            'parameters_tracked': len(self.current_params)
        }

# Global optimizer instance
global_optimizer = ContinuousOptimizer()