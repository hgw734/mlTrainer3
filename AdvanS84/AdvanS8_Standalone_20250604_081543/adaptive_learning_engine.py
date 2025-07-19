"""
Adaptive Learning Engine - Self-Learning Trading System
Automatically adjusts parameters based on actual trade outcomes
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3

class AdaptiveLearningEngine:
    """
    Self-learning system that automatically optimizes trading parameters
    based on actual trade performance and market conditions
    """
    
    def __init__(self, db_path: str = "data/trade_history.db"):
        """Initialize the adaptive learning engine"""
        self.db_path = db_path
        self.learning_state = self._load_learning_state()
        self._ensure_database_exists()
        
    def _ensure_database_exists(self):
        """Create database tables if they don't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    return_pct REAL,
                    hold_days INTEGER,
                    exit_reason TEXT,
                    score_at_entry REAL,
                    vix_at_entry REAL,
                    market_regime TEXT,
                    threshold_used REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS parameter_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    parameter_name TEXT,
                    old_value REAL,
                    new_value REAL,
                    reason TEXT,
                    performance_metric REAL
                )
            ''')
    
    def record_trade(self, trade_data: Dict):
        """Record a completed trade for learning"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trades (
                    timestamp, symbol, entry_price, exit_price, return_pct,
                    hold_days, exit_reason, score_at_entry, vix_at_entry,
                    market_regime, threshold_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('timestamp', datetime.now().isoformat()),
                trade_data.get('symbol', ''),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price', 0),
                trade_data.get('return_pct', 0),
                trade_data.get('hold_days', 0),
                trade_data.get('exit_reason', ''),
                trade_data.get('score_at_entry', 0),
                trade_data.get('vix_at_entry', 20),
                trade_data.get('market_regime', 'moderate'),
                trade_data.get('threshold_used', 75)
            ))
    
    def get_adaptive_threshold(self, current_market_state: Dict) -> float:
        """
        Calculate optimal threshold based on recent performance
        Uses reinforcement learning principles
        """
        recent_performance = self._analyze_recent_performance()
        
        if recent_performance['total_trades'] < 10:
            # Bootstrap mode - use conservative defaults
            return self._get_bootstrap_threshold(current_market_state)
        
        # Calculate reward signal from recent trades
        reward = self._calculate_reward_signal(recent_performance)
        
        # Get current threshold or default
        current_threshold = self.learning_state.get('current_threshold', 75.0)
        
        # Apply learning update
        new_threshold = self._update_threshold_rl(
            current_threshold, reward, recent_performance, current_market_state
        )
        
        # Record parameter change
        if abs(new_threshold - current_threshold) > 1.0:
            self._record_parameter_change(
                'threshold', current_threshold, new_threshold, 
                reward, "Performance-based adaptation"
            )
        
        # Update learning state
        self.learning_state['current_threshold'] = new_threshold
        self.learning_state['last_update'] = datetime.now().isoformat()
        self._save_learning_state()
        
        return max(45.0, min(95.0, new_threshold))
    
    def _analyze_recent_performance(self, lookback_days: int = 30) -> Dict:
        """Analyze performance over recent period"""
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            trades_df = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            ''', conn, params=[cutoff_date])
        
        if len(trades_df) == 0:
            return {'total_trades': 0, 'win_rate': 0.5, 'avg_return': 0.0, 'sharpe': 0.0}
        
        win_rate = (trades_df['return_pct'] > 0).mean()
        avg_return = trades_df['return_pct'].mean()
        return_std = trades_df['return_pct'].std()
        sharpe = avg_return / return_std if return_std > 0 else 0.0
        
        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe': sharpe,
            'return_std': return_std,
            'best_return': trades_df['return_pct'].max(),
            'worst_return': trades_df['return_pct'].min()
        }
    
    def _calculate_reward_signal(self, performance: Dict) -> float:
        """
        Calculate reward signal for reinforcement learning
        Combines multiple performance metrics
        """
        if performance['total_trades'] == 0:
            return 0.0
        
        # Multi-objective reward function
        win_rate_component = (performance['win_rate'] - 0.5) * 2.0  # Target 50%+ win rate
        return_component = performance['avg_return'] * 10.0  # Scale returns
        sharpe_component = performance['sharpe'] * 0.5  # Risk-adjusted returns
        
        # Penalize high volatility
        volatility_penalty = -abs(performance['return_std']) if performance['return_std'] > 0.15 else 0
        
        total_reward = win_rate_component + return_component + sharpe_component + volatility_penalty
        
        # Scale by confidence (number of trades)
        confidence = min(1.0, performance['total_trades'] / 20.0)
        
        return total_reward * confidence
    
    def _update_threshold_rl(self, current_threshold: float, reward: float, 
                           performance: Dict, market_state: Dict) -> float:
        """
        Update threshold using reinforcement learning principles
        """
        # Learning rate decreases as we get more data
        base_learning_rate = 0.1
        learning_rate = base_learning_rate / (1 + performance['total_trades'] / 50.0)
        
        # Policy gradient-style update
        if reward > 0.2:  # Good performance
            # Lower threshold to capture more opportunities
            adjustment = -3.0 * learning_rate
        elif reward < -0.1:  # Poor performance
            # Raise threshold to be more selective
            adjustment = +5.0 * learning_rate
        else:
            # Fine-tune based on win rate
            if performance['win_rate'] > 0.6:
                adjustment = -1.0 * learning_rate  # Slightly more aggressive
            elif performance['win_rate'] < 0.45:
                adjustment = +2.0 * learning_rate  # More conservative
            else:
                adjustment = 0.0
        
        # Market regime adaptation
        vix_level = market_state.get('vix', 20.0)
        if vix_level > 25:  # High volatility
            adjustment -= 2.0 * learning_rate  # Lower threshold in volatile markets
        elif vix_level < 15:  # Low volatility
            adjustment += 1.0 * learning_rate  # Higher threshold in calm markets
        
        return current_threshold + adjustment
    
    def _get_bootstrap_threshold(self, market_state: Dict) -> float:
        """Conservative threshold during initial learning phase"""
        vix_level = market_state.get('vix', 20.0)
        
        if vix_level > 30:
            return 65.0  # More aggressive in crisis
        elif vix_level > 25:
            return 70.0  # Moderately aggressive in volatility
        elif vix_level < 15:
            return 80.0  # Conservative in calm markets
        else:
            return 75.0  # Balanced default
    
    def _record_parameter_change(self, param_name: str, old_value: float, 
                               new_value: float, performance_metric: float, reason: str):
        """Record parameter evolution for analysis"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO parameter_evolution (
                    timestamp, parameter_name, old_value, new_value, 
                    reason, performance_metric
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(), param_name, old_value, 
                new_value, reason, performance_metric
            ))
    
    def _load_learning_state(self) -> Dict:
        """Load persistent learning state"""
        state_file = "data/learning_state.json"
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'current_threshold': 75.0,
            'learning_phase': 'bootstrap',
            'total_trades_processed': 0,
            'last_update': datetime.now().isoformat()
        }
    
    def _save_learning_state(self):
        """Save persistent learning state"""
        os.makedirs("data", exist_ok=True)
        with open("data/learning_state.json", 'w') as f:
            json.dump(self.learning_state, f, indent=2)
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress and current state"""
        recent_perf = self._analyze_recent_performance()
        
        return {
            'current_threshold': self.learning_state.get('current_threshold', 75.0),
            'learning_phase': 'adaptive' if recent_perf['total_trades'] >= 10 else 'bootstrap',
            'recent_performance': recent_perf,
            'total_trades_processed': recent_perf['total_trades'],
            'last_parameter_update': self.learning_state.get('last_update', 'Never'),
            'learning_confidence': min(1.0, recent_perf['total_trades'] / 30.0)
        }
    
    def should_take_trade(self, signal_score: float, market_state: Dict) -> bool:
        """
        Decide whether to take a trade based on adaptive threshold
        """
        adaptive_threshold = self.get_adaptive_threshold(market_state)
        
        # Add some randomness for exploration (epsilon-greedy)
        exploration_rate = max(0.05, 0.2 - self.learning_state.get('total_trades_processed', 0) / 500.0)
        
        if False:  # Disabled synthetic exploration
            # Exploration: sometimes take lower-scoring trades to learn
            exploration_threshold = adaptive_threshold * 0.85
            return signal_score >= exploration_threshold
        else:
            # Exploitation: use learned threshold
            return signal_score >= adaptive_threshold