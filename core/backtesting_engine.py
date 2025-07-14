#!/usr/bin/env python3
"""
Comprehensive Backtesting Engine for mlTrainer
Realistic simulation with transaction costs, slippage, and proper metrics
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000
    position_size: float = 0.95  # Use 95% of capital per trade
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    min_trade_size: float = 100  # Minimum trade size in dollars
    max_positions: int = 1  # Maximum simultaneous positions
    stop_loss: Optional[float] = 0.02  # 2% stop loss (optional)
    take_profit: Optional[float] = 0.05  # 5% take profit (optional)
    allow_shorting: bool = True
    rebalance_frequency: Optional[str] = None  # 'daily', 'weekly', 'monthly'

@dataclass
class Trade:
    """Record of a single trade"""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    entry_price: float = 0
    exit_price: float = 0
    position_size: float = 0
    direction: int = 0  # 1 for long, -1 for short
    commission_paid: float = 0
    slippage_cost: float = 0
    pnl: float = 0
    return_pct: float = 0
    exit_reason: str = ""
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.exit_date:
            return self.exit_date - self.entry_date
        return None

class BacktestingEngine:
    """
    Professional backtesting engine with realistic simulation
    """
    
    def __init__(self, config: BacktestConfig = None):
        """Initialize backtesting engine"""
        self.config = config or BacktestConfig()
        self.reset()
        
    def reset(self):
        """Reset engine state"""
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.positions = []
        self.cash = self.config.initial_capital
        self.current_position = None
        self.metrics = {}
        
    def run(self, data: pd.DataFrame, signals: pd.Series, 
            start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest on historical data with signals
        
        Args:
            data: OHLCV data
            signals: Trading signals (1: buy, 0: hold, -1: sell)
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            
        Returns:
            Dictionary with backtest results and metrics
        """
        self.reset()
        
        # Filter date range if specified
        if start_date:
            data = data[data.index >= start_date]
            signals = signals[signals.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            signals = signals[signals.index <= end_date]
            
        # Align data and signals
        common_index = data.index.intersection(signals.index)
        data = data.loc[common_index]
        signals = signals.loc[common_index]
        
        # Initialize tracking arrays
        equity = np.zeros(len(data))
        positions = np.zeros(len(data))
        
        # Simulate trading
        for i, (date, signal) in enumerate(signals.items()):
            current_price = data.loc[date, 'close']
            
            # Update current position value
            if self.current_position and self.current_position.is_open:
                self._update_position_value(date, current_price)
                
                # Check stop loss and take profit
                if self.config.stop_loss or self.config.take_profit:
                    self._check_exit_conditions(date, data.loc[date])
            
            # Process new signals
            if signal != 0 and i < len(data) - 1:  # Don't trade on last bar
                self._process_signal(date, signal, current_price)
            
            # Record equity and position
            equity[i] = self._calculate_equity(current_price)
            positions[i] = self._get_position_size()
            
        # Close any open positions at end
        if self.current_position and self.current_position.is_open:
            self._close_position(
                data.index[-1], 
                data.iloc[-1]['close'], 
                'end_of_backtest'
            )
        
        # Calculate metrics
        equity_series = pd.Series(equity, index=data.index)
        self.equity_curve = equity_series
        self.positions = pd.Series(positions, index=data.index)
        
        # Generate comprehensive metrics
        self.metrics = self._calculate_metrics(data, signals, equity_series)
        
        return {
            'metrics': self.metrics,
            'equity_curve': self.equity_curve,
            'positions': self.positions,
            'trades': self._trades_to_dataframe(),
            'config': self.config
        }
    
    def _process_signal(self, date: pd.Timestamp, signal: int, price: float):
        """Process trading signal"""
        # Check if we need to close current position
        if self.current_position and self.current_position.is_open:
            if signal != self.current_position.direction:
                # Close current position
                self._close_position(date, price, 'signal_reversal')
                
                # Open new position if not neutral signal
                if signal != 0:
                    self._open_position(date, signal, price)
        else:
            # No current position, open if signal is not neutral
            if signal != 0:
                self._open_position(date, signal, price)
    
    def _open_position(self, date: pd.Timestamp, direction: int, price: float):
        """Open a new position"""
        # Calculate position size
        available_capital = self.cash * self.config.position_size
        
        # Apply slippage to entry price
        slippage_mult = 1 + self.config.slippage if direction == 1 else 1 - self.config.slippage
        entry_price = price * slippage_mult
        
        # Calculate shares (considering commission)
        shares = available_capital / (entry_price * (1 + self.config.commission))
        
        # Check minimum trade size
        if shares * entry_price < self.config.min_trade_size:
            return
        
        # Calculate costs
        commission = shares * entry_price * self.config.commission
        slippage_cost = shares * price * self.config.slippage
        
        # Create trade
        trade = Trade(
            entry_date=date,
            entry_price=entry_price,
            position_size=shares,
            direction=direction,
            commission_paid=commission,
            slippage_cost=slippage_cost
        )
        
        # Update cash
        self.cash -= (shares * entry_price + commission)
        
        # Set as current position
        self.current_position = trade
        self.trades.append(trade)
        
        logger.debug(f"Opened {direction} position: {shares:.2f} shares at {entry_price:.2f}")
    
    def _close_position(self, date: pd.Timestamp, price: float, reason: str):
        """Close current position"""
        if not self.current_position or not self.current_position.is_open:
            return
        
        trade = self.current_position
        
        # Apply slippage to exit price
        slippage_mult = 1 - self.config.slippage if trade.direction == 1 else 1 + self.config.slippage
        exit_price = price * slippage_mult
        
        # Calculate commission
        commission = trade.position_size * exit_price * self.config.commission
        
        # Update trade
        trade.exit_date = date
        trade.exit_price = exit_price
        trade.commission_paid += commission
        trade.exit_reason = reason
        
        # Calculate P&L
        if trade.direction == 1:  # Long position
            gross_pnl = trade.position_size * (exit_price - trade.entry_price)
        else:  # Short position
            gross_pnl = trade.position_size * (trade.entry_price - exit_price)
        
        trade.pnl = gross_pnl - trade.commission_paid - trade.slippage_cost
        trade.return_pct = trade.pnl / (trade.position_size * trade.entry_price)
        
        # Update cash
        self.cash += trade.position_size * exit_price - commission
        
        # Clear current position
        self.current_position = None
        
        logger.debug(f"Closed position: P&L = {trade.pnl:.2f} ({trade.return_pct*100:.2f}%)")
    
    def _update_position_value(self, date: pd.Timestamp, price: float):
        """Update unrealized P&L for current position"""
        if not self.current_position or not self.current_position.is_open:
            return
        
        trade = self.current_position
        if trade.direction == 1:
            unrealized_pnl = trade.position_size * (price - trade.entry_price)
        else:
            unrealized_pnl = trade.position_size * (trade.entry_price - price)
        
        # Store for metrics calculation
        trade.unrealized_pnl = unrealized_pnl
    
    def _check_exit_conditions(self, date: pd.Timestamp, bar_data: pd.Series):
        """Check stop loss and take profit conditions"""
        if not self.current_position or not self.current_position.is_open:
            return
        
        trade = self.current_position
        entry_price = trade.entry_price
        
        # Use high/low for more realistic exit prices
        high = bar_data['high']
        low = bar_data['low']
        close = bar_data['close']
        
        if trade.direction == 1:  # Long position
            # Check stop loss
            if self.config.stop_loss and low <= entry_price * (1 - self.config.stop_loss):
                exit_price = entry_price * (1 - self.config.stop_loss)
                self._close_position(date, exit_price, 'stop_loss')
                return
            
            # Check take profit
            if self.config.take_profit and high >= entry_price * (1 + self.config.take_profit):
                exit_price = entry_price * (1 + self.config.take_profit)
                self._close_position(date, exit_price, 'take_profit')
                return
        else:  # Short position
            # Check stop loss
            if self.config.stop_loss and high >= entry_price * (1 + self.config.stop_loss):
                exit_price = entry_price * (1 + self.config.stop_loss)
                self._close_position(date, exit_price, 'stop_loss')
                return
            
            # Check take profit
            if self.config.take_profit and low <= entry_price * (1 - self.config.take_profit):
                exit_price = entry_price * (1 - self.config.take_profit)
                self._close_position(date, exit_price, 'take_profit')
                return
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current total equity"""
        equity = self.cash
        
        if self.current_position and self.current_position.is_open:
            trade = self.current_position
            if trade.direction == 1:
                position_value = trade.position_size * current_price
            else:
                # Short position value
                position_value = trade.position_size * (2 * trade.entry_price - current_price)
            equity += position_value
        
        return equity
    
    def _get_position_size(self) -> float:
        """Get current position size (signed)"""
        if self.current_position and self.current_position.is_open:
            return self.current_position.position_size * self.current_position.direction
        return 0
    
    def _calculate_metrics(self, data: pd.DataFrame, signals: pd.Series, 
                          equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        returns = equity_curve.pct_change().fillna(0)
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        
        # Annualized metrics
        days = (data.index[-1] - data.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        drawdown_start = None
        max_dd_duration = 0
        current_dd_duration = 0
        
        for date, dd in drawdown.items():
            if dd < 0 and drawdown_start is None:
                drawdown_start = date
            elif dd == 0 and drawdown_start is not None:
                duration = (date - drawdown_start).days
                max_dd_duration = max(max_dd_duration, duration)
                drawdown_start = None
                current_dd_duration = 0
            elif drawdown_start is not None:
                current_dd_duration = (date - drawdown_start).days
        
        max_dd_duration = max(max_dd_duration, current_dd_duration)
        
        # Trade statistics
        closed_trades = [t for t in self.trades if not t.is_open]
        
        if closed_trades:
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(closed_trades)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(sum(t.pnl for t in winning_trades) / 
                               sum(t.pnl for t in losing_trades)) if losing_trades else np.inf
            
            avg_trade_return = np.mean([t.return_pct for t in closed_trades])
            best_trade = max(t.return_pct for t in closed_trades)
            worst_trade = min(t.return_pct for t in closed_trades)
            
            # Trade duration
            trade_durations = [t.duration.days for t in closed_trades if t.duration]
            avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            avg_trade_return = best_trade = worst_trade = avg_trade_duration = 0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
        
        # Market exposure
        positions_held = (self.positions != 0).sum()
        market_exposure = positions_held / len(self.positions) if len(self.positions) > 0 else 0
        
        metrics = {
            # Returns
            'total_return': total_return,
            'annual_return': annual_return,
            'total_pnl': equity_curve.iloc[-1] - self.config.initial_capital,
            
            # Risk
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration_days': max_dd_duration,
            
            # Trade statistics
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades) if closed_trades else 0,
            'losing_trades': len(losing_trades) if closed_trades else 0,
            'win_rate': win_rate,
            'profit_factor': profit_factor if profit_factor != np.inf else 999,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade_return': avg_trade_return,
            'best_trade_return': best_trade,
            'worst_trade_return': worst_trade,
            'avg_trade_duration_days': avg_trade_duration,
            
            # Other
            'market_exposure': market_exposure,
            'final_equity': equity_curve.iloc[-1],
            'peak_equity': equity_curve.max(),
            
            # Commission and slippage impact
            'total_commission': sum(t.commission_paid for t in self.trades),
            'total_slippage': sum(t.slippage_cost for t in self.trades),
        }
        
        return metrics
    
    def _trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades list to DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position_size': trade.position_size,
                'direction': 'long' if trade.direction == 1 else 'short',
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
                'commission': trade.commission_paid,
                'slippage': trade.slippage_cost,
                'exit_reason': trade.exit_reason,
                'duration_days': trade.duration.days if trade.duration else None
            })
        
        return pd.DataFrame(trades_data)
    
    def plot_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plot data for visualization"""
        equity_curve = results['equity_curve']
        positions = results['positions']
        
        # Calculate drawdown series
        cumulative = equity_curve / self.config.initial_capital
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        # Prepare plot data
        plot_data = {
            'equity': {
                'dates': equity_curve.index.tolist(),
                'values': equity_curve.tolist(),
                'label': 'Equity Curve'
            },
            'returns': {
                'dates': equity_curve.index.tolist(),
                'values': (equity_curve.pct_change() * 100).tolist(),
                'label': 'Daily Returns %'
            },
            'drawdown': {
                'dates': drawdown.index.tolist(),
                'values': drawdown.tolist(),
                'label': 'Drawdown %'
            },
            'positions': {
                'dates': positions.index.tolist(),
                'values': positions.tolist(),
                'label': 'Position Size'
            }
        }
        
        return plot_data