"""
Adaptive Feedback System - Automatically adjusts trading parameters based on actual outcomes
The system learns from every trade to continuously improve performance
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from performance_tracker import PerformanceTracker
import os
from typing import Dict, List

class AdaptiveFeedbackSystem:
    """
    Automatically adjusts trading parameters based on actual trade outcomes
    Implements continuous learning from real performance data
    """
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.parameter_state_file = 'adaptive_parameters.json'
        self.load_parameter_state()
        
    def load_parameter_state(self):
        """Load current adaptive parameters"""
        try:
            with open(self.parameter_state_file, 'r') as f:
                self.parameters = json.load(f)
        except FileNotFoundError:
            # Initialize with default parameters
            self.parameters = {
                'base_target_multiplier': 1.0,
                'confidence_weight': 1.0,
                'momentum_weight': 1.0,
                'score_threshold': 70.0,
                'stop_loss_multiplier': 1.0,
                'regime_bull_adjustment': 1.3,
                'regime_bear_adjustment': 0.7,
                'last_update': datetime.now().isoformat(),
                'update_count': 0,
                'performance_history': []
            }
            
    def save_parameter_state(self):
        """Save current parameter state"""
        self.parameters['last_update'] = datetime.now().isoformat()
        with open(self.parameter_state_file, 'w') as f:
            json.dump(self.parameters, f, indent=2)
            
    def process_trade_outcome(self, trade_data: Dict):
        """
        Process a completed trade and update parameters accordingly
        This is the core learning mechanism
        """
        # Record the trade outcome
        self.performance_tracker.record_trade_outcome(trade_data)
        
        # Analyze if we should adjust parameters
        self.analyze_and_adjust_parameters()
        
    def analyze_and_adjust_parameters(self):
        """
        Main adaptation logic - adjusts parameters based on recent performance
        """
        # Get recent performance metrics
        recent_metrics = self.performance_tracker.calculate_performance_metrics(days_back=14)
        
        # Need minimum trades for reliable adjustments
        if recent_metrics['total_trades'] < 10:
            return
            
        # Analyze target accuracy
        self.adjust_target_parameters(recent_metrics)
        
        # Analyze signal threshold effectiveness
        self.adjust_signal_threshold(recent_metrics)
        
        # Analyze stop loss effectiveness
        self.adjust_stop_loss_parameters(recent_metrics)
        
        # Analyze regime adjustments
        self.adjust_regime_parameters(recent_metrics)
        
        # Save updated parameters
        self.parameters['update_count'] += 1
        self.save_parameter_state()
        
        # Log the adjustment
        self.log_parameter_adjustment(recent_metrics)
        
    def adjust_target_parameters(self, metrics: Dict):
        """
        Adjust target calculation parameters based on target hit rates
        """
        target_hit_rate = metrics.get('target_hit_rate', 0)
        
        # If hitting targets too often, increase them
        if target_hit_rate > 0.8:
            adjustment = 1.05  # Increase targets by 5%
            self.parameters['base_target_multiplier'] *= adjustment
            self.parameters['base_target_multiplier'] = min(2.0, self.parameters['base_target_multiplier'])
            
        # If hitting targets too rarely, decrease them
        elif target_hit_rate < 0.4:
            adjustment = 0.95  # Decrease targets by 5%
            self.parameters['base_target_multiplier'] *= adjustment
            self.parameters['base_target_multiplier'] = max(0.5, self.parameters['base_target_multiplier'])
            
    def adjust_signal_threshold(self, metrics: Dict):
        """
        Adjust signal threshold based on win rate and signal accuracy
        """
        win_rate = metrics.get('win_rate', 0)
        signal_accuracy = metrics.get('signal_accuracy', 0)
        
        # If win rate is too low, raise the threshold (be more selective)
        if win_rate < 0.55:
            self.parameters['score_threshold'] += 2.0
            self.parameters['score_threshold'] = min(90.0, self.parameters['score_threshold'])
            
        # If win rate is very high, we can lower threshold (less selective)
        elif win_rate > 0.75:
            self.parameters['score_threshold'] -= 1.0
            self.parameters['score_threshold'] = max(50.0, self.parameters['score_threshold'])
            
    def adjust_stop_loss_parameters(self, metrics: Dict):
        """
        Adjust stop loss parameters based on stop hit rates and drawdown
        """
        stop_hit_rate = metrics.get('stop_hit_rate', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        
        # If stops are hit too often, widen them
        if stop_hit_rate > 0.4:
            self.parameters['stop_loss_multiplier'] *= 1.1
            self.parameters['stop_loss_multiplier'] = min(2.0, self.parameters['stop_loss_multiplier'])
            
        # If drawdown is excessive, tighten stops
        elif abs(max_drawdown) > 0.15:  # More than 15% drawdown
            self.parameters['stop_loss_multiplier'] *= 0.9
            self.parameters['stop_loss_multiplier'] = max(0.5, self.parameters['stop_loss_multiplier'])
            
    def adjust_regime_parameters(self, metrics: Dict):
        """
        Adjust regime-specific parameters based on regime performance
        """
        # This would require regime-specific performance tracking
        # For now, adjust based on overall Sharpe ratio
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        # If Sharpe ratio is poor, be more conservative in all regimes
        if sharpe_ratio < 0.5:
            self.parameters['regime_bull_adjustment'] *= 0.95
            self.parameters['regime_bear_adjustment'] *= 0.95
        elif sharpe_ratio > 2.0:
            self.parameters['regime_bull_adjustment'] *= 1.02
            self.parameters['regime_bear_adjustment'] *= 1.02
            
    def get_current_parameters(self) -> Dict:
        """
        Get current adaptive parameters for use in trading calculations
        """
        return self.parameters.copy()
        
    def get_adaptive_target_multiplier(self) -> float:
        """Get current target multiplier"""
        return self.parameters['base_target_multiplier']
        
    def get_adaptive_score_threshold(self) -> float:
        """Get current score threshold"""
        return self.parameters['score_threshold']
        
    def get_adaptive_stop_multiplier(self) -> float:
        """Get current stop loss multiplier"""
        return self.parameters['stop_loss_multiplier']
        
    def get_regime_adjustments(self) -> Dict:
        """Get current regime adjustment factors"""
        return {
            'bull': self.parameters['regime_bull_adjustment'],
            'bear': self.parameters['regime_bear_adjustment'],
            'neutral': 1.0  # Always neutral baseline
        }
        
    def log_parameter_adjustment(self, metrics: Dict):
        """Log parameter adjustments for transparency"""
        adjustment_log = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'new_parameters': self.parameters.copy(),
            'reason': self.generate_adjustment_reason(metrics)
        }
        
        # Add to history
        if 'adjustment_history' not in self.parameters:
            self.parameters['adjustment_history'] = []
            
        self.parameters['adjustment_history'].append(adjustment_log)
        
        # Keep only last 50 adjustments
        if len(self.parameters['adjustment_history']) > 50:
            self.parameters['adjustment_history'] = self.parameters['adjustment_history'][-50:]
            
    def generate_adjustment_reason(self, metrics: Dict) -> str:
        """Generate human-readable reason for parameter adjustment"""
        reasons = []
        
        if metrics.get('target_hit_rate', 0) > 0.8:
            reasons.append("Target hit rate too high - increased targets")
        elif metrics.get('target_hit_rate', 0) < 0.4:
            reasons.append("Target hit rate too low - decreased targets")
            
        if metrics.get('win_rate', 0) < 0.55:
            reasons.append("Win rate below 55% - raised signal threshold")
        elif metrics.get('win_rate', 0) > 0.75:
            reasons.append("Win rate above 75% - lowered signal threshold")
            
        if metrics.get('stop_hit_rate', 0) > 0.4:
            reasons.append("Stop losses hit too often - widened stops")
            
        if abs(metrics.get('max_drawdown', 0)) > 0.15:
            reasons.append("Excessive drawdown - tightened stops")
            
        return "; ".join(reasons) if reasons else "Routine optimization"
        
    def get_learning_summary(self) -> Dict:
        """
        Get summary of how the system has learned and adapted
        """
        improvement_status = self.performance_tracker.is_system_improving()
        
        return {
            'current_parameters': self.parameters,
            'total_adjustments': self.parameters['update_count'],
            'last_adjustment': self.parameters['last_update'],
            'performance_status': improvement_status['status'],
            'recent_adjustments': self.parameters.get('adjustment_history', [])[-5:],
            'parameter_evolution': self.get_parameter_evolution()
        }
        
    def get_parameter_evolution(self) -> Dict:
        """
        Show how parameters have evolved over time
        """
        history = self.parameters.get('adjustment_history', [])
        if len(history) < 2:
            return {'evolution': 'insufficient_data'}
            
        first = history[0]['new_parameters']
        current = self.parameters
        
        evolution = {
            'target_multiplier': {
                'initial': first.get('base_target_multiplier', 1.0),
                'current': current['base_target_multiplier'],
                'change': current['base_target_multiplier'] - first.get('base_target_multiplier', 1.0)
            },
            'score_threshold': {
                'initial': first.get('score_threshold', 70.0),
                'current': current['score_threshold'],
                'change': current['score_threshold'] - first.get('score_threshold', 70.0)
            },
            'stop_multiplier': {
                'initial': first.get('stop_loss_multiplier', 1.0),
                'current': current['stop_loss_multiplier'],
                'change': current['stop_loss_multiplier'] - first.get('stop_loss_multiplier', 1.0)
            }
        }
        
        return evolution

def main():
    """Test the adaptive feedback system"""
    feedback_system = AdaptiveFeedbackSystem()
    
    # Simulate some trade outcomes
    example_trades = [
        {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'AAPL',
            'signal_score': 85,
            'ml_confidence': 0.82,
            'entry_price': 150.0,
            'exit_price': 157.5,
            'return_pct': 0.05,
            'hold_days': 3,
            'outcome': 'win',
            'market_regime': 'bull',
            'target_hit': True,
            'stop_hit': False
        },
        {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'MSFT',
            'signal_score': 72,
            'ml_confidence': 0.65,
            'entry_price': 300.0,
            'exit_price': 285.0,
            'return_pct': -0.05,
            'hold_days': 2,
            'outcome': 'loss',
            'market_regime': 'neutral',
            'target_hit': False,
            'stop_hit': True
        }
    ]
    
    for trade in example_trades:
        feedback_system.process_trade_outcome(trade)
        
    summary = feedback_system.get_learning_summary()
    print("Adaptive Learning Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()