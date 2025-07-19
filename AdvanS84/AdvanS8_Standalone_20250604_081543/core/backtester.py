"""
Real Backtester Module
ML-based exit prediction with comprehensive trade simulation
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from .exit_model import hybrid_exit_strategy, predict_exit_strategy
from .regime_logic import get_market_regime

logger = logging.getLogger(__name__)

class RealBacktester:
    """
    Real backtester with ML-predicted exits and comprehensive performance tracking
    """
    
    def __init__(self, initial_capital=100000):
        """Initialize backtester with starting capital"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_log = []
        self.equity_curve = []
        self.performance_metrics = {}
        
    def evaluate_strategy(self, params, market_data, universe, exit_predictor=None, label_encoder=None):
        """
        Evaluate trading strategy with ML-predicted exits using authentic 15-minute data
        
        Args:
            params: Strategy parameters
            market_data: Authentic market data from Polygon API
            universe: List of stock symbols
            exit_predictor: Trained ML exit predictor
            label_encoder: Label encoder for exit strategies
        
        Returns:
            dict: Comprehensive performance results
        """
        logger.info("Running real backtester with ML exit predictions...")
        
        # Reset backtester state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trade_log = []
        self.equity_curve = []
        
        # Get VIX data for market regime detection
        vix_data = market_data.get('VIX')
        
        # Process each symbol independently
        for symbol in universe:
            if symbol == 'VIX' or symbol not in market_data:
                continue
                
            df = market_data[symbol]
            if len(df) < 500:  # Need sufficient data for 15-minute bars
                continue
            
            logger.info(f"Backtesting {symbol} with {len(df)} data points")
            self._simulate_symbol_trading(df, symbol, params, vix_data, exit_predictor, label_encoder)
        
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        
        # Save trade log
        self._save_trade_log()
        
        return self.performance_metrics
    
    def _simulate_symbol_trading(self, df, symbol, params, vix_data, exit_predictor, label_encoder):
        """
        Simulate trading for a single symbol with ML exit prediction
        
        Args:
            df: Stock price data with technical indicators
            symbol: Stock symbol
            params: Strategy parameters
            vix_data: VIX data for regime detection
            exit_predictor: ML model for exit prediction
            label_encoder: Label encoder
        """
        position = None
        
        # Use 15-minute bars for realistic intraday simulation
        for i in range(100, len(df) - 50):  # Leave buffer for indicators and exits
            current_time = df.index[i]
            current_row = df.iloc[i]
            
            # Update equity curve
            self._update_equity_curve(current_time, df.iloc[i])
            
            # Check for exit signal if we have a position
            if position is not None:
                should_exit, exit_reason = self._check_exit_condition(
                    df, i, position, params, symbol, vix_data, exit_predictor, label_encoder
                )
                
                if should_exit:
                    self._close_position(position, current_row, exit_reason, current_time)
                    position = None
            
            # Check for entry signal if no position
            if position is None:
                should_enter = self._check_entry_condition(current_row, params, vix_data, current_time)
                
                if should_enter:
                    position = self._open_position(symbol, current_row, current_time)
        
        # Close any remaining position at the end
        if position is not None:
            final_row = df.iloc[-1]
            self._close_position(position, final_row, 'end_of_data', df.index[-1])
    
    def _check_entry_condition(self, current_row, params, vix_data, current_time):
        """
        Check if entry conditions are met using authentic market data
        
        Args:
            current_row: Current market data row
            params: Strategy parameters
            vix_data: VIX data for regime detection
            current_time: Current timestamp
        
        Returns:
            bool: True if entry conditions are met
        """
        # Skip if essential data is missing
        if pd.isna(current_row['momentum_5']) or pd.isna(current_row['rsi_14']):
            return False
        
        # Get current market regime
        market_regime = get_market_regime(current_time, vix_data) if vix_data is not None else 'moderate_volatility_normal'
        
        # Entry criteria with regime adaptation
        momentum_ok = current_row['momentum_5'] >= params.get('momentum_threshold', 0.02)
        volume_ok = current_row['volume_ratio'] >= params.get('volume_multiplier', 2.0)
        trend_ok = current_row['close'] > current_row['sma_20']
        
        # Regime-adaptive RSI threshold
        from .regime_logic import get_regime_adaptive_rsi_threshold
        rsi_threshold = get_regime_adaptive_rsi_threshold(market_regime)
        rsi_ok = current_row['rsi_14'] > rsi_threshold
        
        # Risk management
        risk_ok = True
        if not pd.isna(current_row['atr_pct_14']):
            risk_tolerance = params.get('risk_tolerance', 0.03)
            if 'crisis' in market_regime:
                risk_tolerance *= 0.7
            risk_ok = current_row['atr_pct_14'] < risk_tolerance
        
        return momentum_ok and volume_ok and trend_ok and rsi_ok and risk_ok
    
    def _check_exit_condition(self, df, current_idx, position, params, symbol, vix_data, exit_predictor, label_encoder):
        """
        Check exit condition using ML prediction and market conditions
        
        Args:
            df: Price data
            current_idx: Current index
            position: Current position
            params: Strategy parameters
            symbol: Stock symbol
            vix_data: VIX data
            exit_predictor: ML exit predictor
            label_encoder: Label encoder
        
        Returns:
            tuple: (should_exit: bool, exit_reason: str)
        """
        current_time = df.index[current_idx]
        current_row = df.iloc[current_idx]
        
        # Calculate position metrics
        days_held = (current_time - position['entry_time']).days
        current_return = (current_row['close'] - position['entry_price']) / position['entry_price']
        
        # Get market regime
        market_regime = get_market_regime(current_time, vix_data) if vix_data is not None else 'moderate_volatility_normal'
        
        # Use ML prediction for exit strategy if available
        if exit_predictor is not None and label_encoder is not None:
            try:
                # Use hybrid exit strategy with ML prediction
                predicted_return = hybrid_exit_strategy(
                    df, position['entry_index'], params, symbol, market_regime, 
                    exit_predictor, label_encoder
                )
                
                if predicted_return is not None:
                    return True, 'ml_predicted_exit'
            except Exception as e:
                logger.warning(f"Error in ML exit prediction: {e}")
        
        # Fallback exit conditions
        
        # Stop loss
        if current_return < -0.08:
            return True, 'stop_loss'
        
        # Take profit
        take_profit_threshold = 0.15 if 'low_volatility' in market_regime else 0.25
        if current_return > take_profit_threshold:
            return True, 'take_profit'
        
        # Time-based exit (regime adaptive)
        max_hold_days = params.get('hold_period', 12)
        if 'crisis' in market_regime:
            max_hold_days = max(3, int(max_hold_days * 0.6))
        elif 'high_volatility' in market_regime:
            max_hold_days = max(5, int(max_hold_days * 0.8))
        
        if days_held >= max_hold_days:
            return True, 'time_exit'
        
        return False, None
    
    def _open_position(self, symbol, current_row, current_time):
        """
        Open new position
        
        Args:
            symbol: Stock symbol
            current_row: Current market data
            current_time: Current timestamp
        
        Returns:
            dict: Position information
        """
        position_size = self.current_capital * 0.1  # 10% position sizing
        shares = int(position_size / current_row['close'])
        
        if shares > 0:
            position = {
                'symbol': symbol,
                'entry_time': current_time,
                'entry_index': len(self.trade_log),
                'entry_price': current_row['close'],
                'shares': shares,
                'capital_allocated': shares * current_row['close']
            }
            
            self.current_capital -= position['capital_allocated']
            logger.debug(f"Opened position: {symbol} @ ${current_row['close']:.2f}")
            return position
        
        return None
    
    def _close_position(self, position, current_row, exit_reason, current_time):
        """
        Close position and log trade
        
        Args:
            position: Position to close
            current_row: Current market data
            exit_reason: Reason for exit
            current_time: Current timestamp
        """
        exit_value = position['shares'] * current_row['close']
        trade_return = (exit_value - position['capital_allocated']) / position['capital_allocated']
        
        # Update capital
        self.current_capital += exit_value
        
        # Log trade
        trade_record = {
            'symbol': position['symbol'],
            'entry_time': position['entry_time'],
            'exit_time': current_time,
            'entry_price': position['entry_price'],
            'exit_price': current_row['close'],
            'shares': position['shares'],
            'trade_return': trade_return,
            'exit_reason': exit_reason,
            'hold_time_days': (current_time - position['entry_time']).days,
            'profit_loss': exit_value - position['capital_allocated']
        }
        
        self.trade_log.append(trade_record)
        logger.debug(f"Closed position: {position['symbol']} - Return: {trade_return:.2%}")
    
    def _update_equity_curve(self, current_time, current_row):
        """Update equity curve with current portfolio value"""
        total_value = self.current_capital
        
        # Add value of open positions
        for position in self.positions.values():
            if position is not None:
                total_value += position['shares'] * current_row['close']
        
        self.equity_curve.append({
            'timestamp': current_time,
            'equity': total_value,
            'returns': (total_value - self.initial_capital) / self.initial_capital
        })
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trade_log:
            self.performance_metrics = {'error': 'No trades executed'}
            return
        
        trades_df = pd.DataFrame(self.trade_log)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['trade_return'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        returns = trades_df['trade_return'].values
        avg_return = np.mean(returns)
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Risk metrics
        volatility = np.std(returns)
        sharpe_ratio = avg_return / (volatility + 1e-6)
        
        # Drawdown calculation
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 0:
            equity_values = equity_df['equity'].values
            running_max = np.maximum.accumulate(equity_values)
            drawdown = (equity_values - running_max) / running_max
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': self.current_capital,
            'profit_factor': self._calculate_profit_factor(trades_df),
            'avg_hold_time': trades_df['hold_time_days'].mean() if total_trades > 0 else 0
        }
        
        logger.info(f"Backtest complete - Total return: {total_return:.2%}, Win rate: {win_rate:.2%}")
    
    def _calculate_profit_factor(self, trades_df):
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum()
        gross_loss = abs(trades_df[trades_df['profit_loss'] < 0]['profit_loss'].sum())
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _save_trade_log(self):
        """Save trade log to CSV file"""
        if not self.trade_log:
            return
        
        trades_dir = 'data/trades'
        os.makedirs(trades_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{trades_dir}/trades_{timestamp}.csv"
        
        trades_df = pd.DataFrame(self.trade_log)
        trades_df.to_csv(filename, index=False)
        
        logger.info(f"Trade log saved to {filename}")
        
        # Save equity curve
        if self.equity_curve:
            equity_filename = f"{trades_dir}/equity_curve_{timestamp}.csv"
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(equity_filename, index=False)
            logger.info(f"Equity curve saved to {equity_filename}")

def create_backtester(initial_capital=100000):
    """Factory function to create backtester"""
    return RealBacktester(initial_capital)