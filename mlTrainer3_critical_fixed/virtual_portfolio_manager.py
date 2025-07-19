"""
Virtual Portfolio Manager
========================

Manages paper trading positions for the recommendation system.
Tracks virtual buys/sells and calculates performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
from dataclasses import dataclass
from enum import Enum

# Import data connectors
from polygon_connector import PolygonConnector
from fred_connector import FREDConnector

# Import database connection (would need to be implemented)
# from database.connection import get_db_connection

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED_OUT = "STOPPED_OUT"


class ExitReason(Enum):
    TARGET_HIT = "TARGET_HIT"
    STOP_LOSS = "STOP_LOSS"
    TIMEOUT = "TIMEOUT"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    MANUAL_CLOSE = "MANUAL_CLOSE"


@dataclass
class VirtualPosition:
    """Represents a virtual trading position"""
    recommendation_id: int
    symbol: str
    entry_price: float
    shares: int
    entry_time: datetime
    target_price: float
    stop_loss: float
    timeframe: str
    current_price: float = None
    exit_price: float = None
    exit_time: datetime = None
    status: PositionStatus = PositionStatus.OPEN
    exit_reason: ExitReason = None

    @property
    def current_value(self) -> float:
        """Calculate current position value"""
        price = self.current_price or self.entry_price
        return price * self.shares

    @property
    def profit_loss(self) -> float:
        """Calculate current P&L"""
        if self.exit_price:
            return (self.exit_price - self.entry_price) * self.shares
        elif self.current_price:
            return (self.current_price - self.entry_price) * self.shares
        return 0.0

    @property
    def profit_loss_pct(self) -> float:
        """Calculate P&L percentage"""
        if self.entry_price > 0:
            if self.exit_price:
                return ((self.exit_price - self.entry_price) /
                        self.entry_price) * 100
            elif self.current_price:
                return ((self.current_price - self.entry_price) /
                        self.entry_price) * 100
        return 0.0

    @property
    def days_held(self) -> int:
        """Calculate days position has been held"""
        end_time = self.exit_time or datetime.now()
        return (end_time - self.entry_time).days


class VirtualPortfolioManager:
    """Manages virtual portfolio for paper trading"""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        # recommendation_id -> position
        self.positions: Dict[int, VirtualPosition] = {}
        self.closed_positions: List[VirtualPosition] = []
        self.polygon_client = PolygonConnector()
        self.max_positions = 20  # Maximum concurrent positions
        self.position_size_pct = 0.05  # 5% of portfolio per position

        logger.info(
            f"Virtual Portfolio Manager initialized with ${initial_capital:,.2f}")

    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        open_positions = sum(1 for p in self.positions.values()
                             if p.status == PositionStatus.OPEN)
        return open_positions < self.max_positions and self.cash > 0

    def calculate_position_size(self, price: float) -> int:
        """Calculate number of shares for position sizing"""
        position_value = self.cash * self.position_size_pct
        shares = int(position_value / price)
        return max(shares, 1)  # At least 1 share

    def open_position(self, recommendation: Dict) -> Optional[VirtualPosition]:
        """Open a new virtual position from recommendation"""
        if not self.can_open_position():
            logger.warning(
                "Cannot open new position - limit reached or insufficient cash")
            return None

        try:
            # Calculate position size
            shares = self.calculate_position_size(
                recommendation['entry_price'])
            cost = shares * recommendation['entry_price']

            if cost > self.cash:
                shares = int(self.cash / recommendation['entry_price'])
                cost = shares * recommendation['entry_price']

            if shares == 0:
                logger.warning("Insufficient cash for position")
                return None

            # Create position
            position = VirtualPosition(
                recommendation_id=recommendation['id'],
                symbol=recommendation['symbol'],
                entry_price=recommendation['entry_price'],
                shares=shares,
                entry_time=datetime.now(),
                target_price=recommendation['target_price'],
                stop_loss=recommendation['stop_loss'],
                timeframe=recommendation['timeframe'],
                current_price=recommendation['entry_price']
            )

            # Update portfolio
            self.cash -= cost
            self.positions[recommendation['id']] = position

            # Log to database
            self._save_position_to_db(position)

            logger.info(
                f"Opened virtual position: {shares} shares of {position.symbol} at ${position.entry_price:.2f}")
            return position

        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None

    def update_positions(self):
        """Update all open positions with current prices"""
        open_positions = [
            p for p in self.positions.values() if p.status == PositionStatus.OPEN]

        if not open_positions:
            return

        # Get current prices for all symbols
        symbols = list(set(p.symbol for p in open_positions))
        current_prices = self._get_current_prices(symbols)

        for position in open_positions:
            if position.symbol in current_prices:
                old_price = position.current_price
                position.current_price = current_prices[position.symbol]

                # Check exit conditions
                self._check_exit_conditions(position)

                # Update database
                self._update_position_in_db(position)

                if old_price != position.current_price:
                    logger.debug(
                        f"Updated {position.symbol}: ${old_price:.2f} -> ${position.current_price:.2f}")

    def _check_exit_conditions(self, position: VirtualPosition):
        """Check if position should be closed"""
        if position.status != PositionStatus.OPEN:
            return

        # Check stop loss
        if position.current_price <= position.stop_loss:
            self.close_position(
                position.recommendation_id,
                ExitReason.STOP_LOSS)
            return

        # Check target
        if position.current_price >= position.target_price:
            self.close_position(
                position.recommendation_id,
                ExitReason.TARGET_HIT)
            return

        # Check timeout based on timeframe
        days_held = position.days_held
        if position.timeframe == "7-12 days" and days_held > 12:
            self.close_position(position.recommendation_id, ExitReason.TIMEOUT)
        elif position.timeframe == "50-70 days" and days_held > 70:
            self.close_position(position.recommendation_id, ExitReason.TIMEOUT)

    def close_position(
            self,
            recommendation_id: int,
            reason: ExitReason) -> Optional[VirtualPosition]:
        """Close a virtual position"""
        if recommendation_id not in self.positions:
            logger.warning(f"Position {recommendation_id} not found")
            return None

        position = self.positions[recommendation_id]
        if position.status != PositionStatus.OPEN:
            logger.warning(f"Position {recommendation_id} already closed")
            return None

        # Update position
        position.exit_price = position.current_price
        position.exit_time = datetime.now()
        position.status = PositionStatus.CLOSED if reason != ExitReason.STOP_LOSS else PositionStatus.STOPPED_OUT
        position.exit_reason = reason

        # Update cash
        self.cash += position.exit_price * position.shares

        # Move to closed positions
        self.closed_positions.append(position)

        # Update database
        self._close_position_in_db(position)

        logger.info(
            f"Closed position: {position.symbol} at ${position.exit_price:.2f} "
            f"({position.profit_loss_pct:+.2f}%) - Reason: {reason.value}")

        return position

    def get_portfolio_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        # Current portfolio value
        positions_value = sum(p.current_value for p in self.positions.values()
                              if p.status == PositionStatus.OPEN)
        total_value = self.cash + positions_value

        # Calculate returns
        total_return = total_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Win/loss statistics
        closed = self.closed_positions
        if closed:
            wins = [p for p in closed if p.profit_loss > 0]
            losses = [p for p in closed if p.profit_loss < 0]

            win_rate = len(wins) / len(closed) * 100 if closed else 0
            avg_win = np.mean([p.profit_loss_pct for p in wins]) if wins else 0
            avg_loss = np.mean(
                [p.profit_loss_pct for p in losses]) if losses else 0

            # Calculate Sharpe ratio (simplified)
            returns = [p.profit_loss_pct for p in closed]
            if len(returns) > 1:
                sharpe = np.mean(returns) / \
                    (np.std(returns) + 1e-6) * np.sqrt(252)

                # Calculate Sortino ratio (downside deviation)
                negative_returns = [r for r in returns if r < 0]
                if negative_returns:
                    downside_deviation = np.std(negative_returns)
                    sortino = np.mean(returns) / \
                        (downside_deviation + 1e-6) * np.sqrt(252)
                else:
                    sortino = np.mean(
                        returns) * np.sqrt(252) if np.mean(returns) > 0 else 0
            else:
                sharpe = 0
                sortino = 0
        else:
            win_rate = avg_win = avg_loss = sharpe = sortino = 0

        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'open_positions': len([p for p in self.positions.values() if p.status == PositionStatus.OPEN]),
            'closed_positions': len(closed),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'best_trade': max((p.profit_loss_pct for p in closed), default=0),
            'worst_trade': min((p.profit_loss_pct for p in closed), default=0)
        }

    def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols"""
        prices = {}
        for symbol in symbols:
            try:
                quote = self.polygon_client.get_quote(symbol)
                if quote:
                    prices[symbol] = quote.price
                else:
                    logger.warning(f"No quote available for {symbol}")
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
        return prices

    def _save_position_to_db(self, position: VirtualPosition):
        """Save position to database"""
        # TODO: Implement database save
        logger.debug(
            f"Would save position {position.recommendation_id} to database")

    def _update_position_in_db(self, position: VirtualPosition):
        """Update position in database"""
        # TODO: Implement database update
        logger.debug(
            f"Would update position {position.recommendation_id} in database")

    def _close_position_in_db(self, position: VirtualPosition):
        """Mark position as closed in database"""
        # TODO: Implement database close
        logger.debug(
            f"Would close position {position.recommendation_id} in database")

    def generate_performance_report(self) -> str:
        """Generate a performance report"""
        metrics = self.get_portfolio_metrics()

        report = f"""
Virtual Portfolio Performance Report
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Portfolio Value: ${metrics['total_value']:,.2f}
Cash Balance: ${metrics['cash']:,.2f}
Positions Value: ${metrics['positions_value']:,.2f}

Total Return: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:+.2f}%)

Trading Statistics:
- Open Positions: {metrics['open_positions']}
- Closed Trades: {metrics['closed_positions']}
- Win Rate: {metrics['win_rate']:.1f}%
- Average Win: {metrics['avg_win_pct']:+.2f}%
- Average Loss: {metrics['avg_loss_pct']:+.2f}%
- Best Trade: {metrics['best_trade']:+.2f}%
- Worst Trade: {metrics['worst_trade']:+.2f}%
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
- Sortino Ratio: {metrics['sortino_ratio']:.2f}
"""
        return report


# Singleton instance
_portfolio_manager = None


def get_virtual_portfolio_manager() -> VirtualPortfolioManager:
    """Get or create the virtual portfolio manager instance"""
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = VirtualPortfolioManager()
    return _portfolio_manager
