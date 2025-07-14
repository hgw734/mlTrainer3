#!/usr/bin/env python3
"""
Walk-Forward Analysis Trial Launcher
Uses REAL historical data from approved sources
NO FAKE DATA OR MOCK PATTERNS
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Import real components
from core.data_pipeline import DataPipeline
from polygon_connector import PolygonConnector
from mltrainer_models import MLTrainerModelManager

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardStep:
    """Represents a single walk-forward step"""
    step_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: pd.DataFrame
    test_data: pd.DataFrame

class WalkForwardAnalyzer:
    """
    Proper walk-forward analysis using real historical data
    NO MOCK DATA - REAL BACKTESTING ONLY
    """
    
    def __init__(self, 
                 train_window_days: int = 252,  # 1 year
                 test_window_days: int = 63,    # 3 months
                 step_size_days: int = 21):     # 1 month
        """
        Initialize walk-forward analyzer with time windows
        
        Args:
            train_window_days: Trading days for training period
            test_window_days: Trading days for testing period
            step_size_days: Trading days to step forward each iteration
        """
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.step_size = step_size_days
        self.data_pipeline = DataPipeline()
        self.model_manager = MLTrainerModelManager()
        
    def prepare_walk_forward_data(self, symbol: str, start_date: str = None, 
                                 total_days: int = 756) -> List[WalkForwardStep]:
        """
        Prepare walk-forward steps with real market data
        
        Args:
            symbol: Stock symbol to analyze
            start_date: Start date for analysis
            total_days: Total days of data to use (default 3 years)
            
        Returns:
            List of WalkForwardStep objects with real data
        """
        # Fetch real historical data
        logger.info(f"Fetching {total_days} days of historical data for {symbol}")
        all_data = self.data_pipeline.fetch_historical_data(symbol, days=total_days)
        
        if len(all_data) < self.train_window + self.test_window:
            raise ValueError(f"Insufficient data for walk-forward analysis. Need at least "
                           f"{self.train_window + self.test_window} days, got {len(all_data)}")
        
        steps = []
        step_number = 0
        
        # Walk forward through the data
        current_idx = 0
        while current_idx + self.train_window + self.test_window <= len(all_data):
            # Define train and test periods
            train_start_idx = current_idx
            train_end_idx = current_idx + self.train_window
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.test_window
            
            # Slice the data
            train_data = all_data.iloc[train_start_idx:train_end_idx].copy()
            test_data = all_data.iloc[test_start_idx:test_end_idx].copy()
            
            # Create step
            step = WalkForwardStep(
                step_number=step_number,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                train_data=train_data,
                test_data=test_data
            )
            
            steps.append(step)
            
            # Move forward
            current_idx += self.step_size
            step_number += 1
            
        logger.info(f"Created {len(steps)} walk-forward steps")
        return steps
    
    def execute_walk_forward_step(self, step: WalkForwardStep, model_class, 
                                 model_params: Dict = None) -> Dict[str, Any]:
        """
        Execute a single walk-forward step with real data
        
        Args:
            step: WalkForwardStep with train/test data
            model_class: Model class to train and test
            model_params: Model parameters
            
        Returns:
            Dictionary with real performance metrics
        """
        try:
            # Initialize model
            model = model_class(**(model_params or {}))
            
            # Train on training data
            logger.info(f"Training model on data from {step.train_start} to {step.train_end}")
            model.fit(step.train_data)
            
            # Generate predictions on test data
            predictions = model.predict(step.test_data)
            
            # Calculate real performance metrics
            metrics = self._calculate_real_metrics(step.test_data, predictions)
            
            # Add step information
            metrics.update({
                'step_number': step.step_number,
                'train_period': f"{step.train_start} to {step.train_end}",
                'test_period': f"{step.test_start} to {step.test_end}",
                'train_samples': len(step.train_data),
                'test_samples': len(step.test_data)
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in walk-forward step {step.step_number}: {e}")
            raise
    
    def _calculate_real_metrics(self, test_data: pd.DataFrame, 
                               predictions: pd.Series) -> Dict[str, float]:
        """
        Calculate real trading metrics from actual predictions
        
        Args:
            test_data: Real test data with OHLCV
            predictions: Model predictions (1=buy, 0=hold, -1=sell)
            
        Returns:
            Dictionary with real performance metrics
        """
        # Calculate returns based on predictions
        returns = test_data['close'].pct_change()
        signal_returns = returns.shift(-1) * predictions.shift(1)
        signal_returns = signal_returns.dropna()
        
        # Real trading metrics
        total_return = (1 + signal_returns).prod() - 1
        
        # Sharpe ratio (annualized)
        if signal_returns.std() > 0:
            sharpe_ratio = (signal_returns.mean() * 252) / (signal_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Win rate
        winning_trades = signal_returns[signal_returns > 0]
        losing_trades = signal_returns[signal_returns < 0]
        total_trades = len(signal_returns[signal_returns != 0])
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + signal_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Volatility (annualized)
        volatility = signal_returns.std() * np.sqrt(252)
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'volatility': volatility,
            'calmar_ratio': calmar_ratio,
            'total_trades': total_trades,
            'profitable_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
    
    def run_complete_walk_forward(self, symbol: str, model_configs: List[Dict], 
                                 total_days: int = 756) -> pd.DataFrame:
        """
        Run complete walk-forward analysis for multiple models
        
        Args:
            symbol: Stock symbol to analyze
            model_configs: List of model configurations
            total_days: Total days of historical data to use
            
        Returns:
            DataFrame with all results
        """
        # Prepare walk-forward steps
        steps = self.prepare_walk_forward_data(symbol, total_days=total_days)
        
        all_results = []
        
        for config in model_configs:
            model_id = config['model_id']
            model_class = self.model_manager.get_model_class(model_id)
            model_params = config.get('params', {})
            
            logger.info(f"Running walk-forward for {model_id}")
            
            for step in steps:
                try:
                    # Execute step
                    results = self.execute_walk_forward_step(step, model_class, model_params)
                    results['model_id'] = model_id
                    all_results.append(results)
                    
                except Exception as e:
                    logger.error(f"Failed step {step.step_number} for {model_id}: {e}")
                    continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Add summary statistics
        self._add_summary_statistics(results_df)
        
        return results_df
    
    def _add_summary_statistics(self, results_df: pd.DataFrame):
        """Add summary statistics to results DataFrame"""
        # Group by model
        for model_id in results_df['model_id'].unique():
            model_results = results_df[results_df['model_id'] == model_id]
            
            logger.info(f"\nSummary for {model_id}:")
            logger.info(f"  Average Sharpe: {model_results['sharpe_ratio'].mean():.3f}")
            logger.info(f"  Average Return: {model_results['total_return'].mean():.3%}")
            logger.info(f"  Average Drawdown: {model_results['max_drawdown'].mean():.3%}")
            logger.info(f"  Average Win Rate: {model_results['win_rate'].mean():.3%}")
            logger.info(f"  Consistency (% positive): {(model_results['total_return'] > 0).mean():.3%}")
    
    def analyze_regime_performance(self, symbol: str, model_configs: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Analyze model performance across different market regimes
        Uses real historical data to identify regimes
        """
        # Get market regime features
        regime_features = self.data_pipeline.create_market_regime_features(symbol)
        
        # Define regimes based on real market conditions
        regimes = self._identify_market_regimes(regime_features)
        
        regime_results = {}
        
        for regime_name, regime_periods in regimes.items():
            logger.info(f"Analyzing performance in {regime_name} regime")
            
            # Run walk-forward for each regime period
            regime_df_list = []
            
            for start_date, end_date in regime_periods:
                # Get data for this regime period
                regime_data = self.data_pipeline.fetch_historical_data(
                    symbol, 
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if len(regime_data) < self.train_window + self.test_window:
                    continue
                
                # Run models on this regime
                for config in model_configs:
                    # ... implement regime-specific analysis
                    pass
            
            regime_results[regime_name] = pd.concat(regime_df_list) if regime_df_list else pd.DataFrame()
        
        return regime_results
    
    def _identify_market_regimes(self, regime_features: pd.DataFrame) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """Identify market regimes from real data"""
        regimes = {
            'bull': [],
            'bear': [],
            'high_volatility': [],
            'low_volatility': []
        }
        
        # Simple regime identification based on trends and volatility
        # This is a simplified version - real implementation would be more sophisticated
        
        # Bull regime: price above 50-day MA
        bull_mask = regime_features['trend_50'] > 0
        
        # Bear regime: price below 50-day MA
        bear_mask = regime_features['trend_50'] < -0.05
        
        # High volatility: vol_ratio > 1.2
        high_vol_mask = regime_features['vol_ratio'] > 1.2
        
        # Convert masks to date ranges
        # ... (implementation details)
        
        return regimes


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = WalkForwardAnalyzer(
        train_window_days=252,  # 1 year training
        test_window_days=63,    # 3 months testing
        step_size_days=21       # 1 month steps
    )
    
    # Define models to test
    model_configs = [
        {
            'model_id': 'momentum_breakout',
            'params': {'lookback_period': 20, 'breakout_threshold': 2.0}
        },
        {
            'model_id': 'mean_reversion',
            'params': {'lookback_period': 10, 'entry_threshold': -2.0}
        }
    ]
    
    # Run walk-forward analysis with real data
    try:
        results = analyzer.run_complete_walk_forward(
            symbol='AAPL',
            model_configs=model_configs,
            total_days=756  # 3 years of data
        )
        
        # Save results
        results.to_csv('walk_forward_results.csv', index=False)
        logger.info("Walk-forward analysis complete. Results saved to walk_forward_results.csv")
        
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}")
        raise
