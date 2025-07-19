"""
Portfolio Exit Strategy Backtesting System
Tests the dual exit approach: Cut losers fast vs Let winners run through pullbacks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a portfolio position with entry details"""
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    scanner_score: float
    momentum_strength: float
    current_price: float = 0.0
    days_held: int = 0
    max_gain: float = 0.0
    max_loss: float = 0.0
    exit_reason: str = ""
    
    @property
    def current_return(self) -> float:
        if self.current_price > 0:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        return 0.0
    
    @property
    def is_winner(self) -> bool:
        return self.current_return > 0.0

class PortfolioExitBacktest:
    """
    Comprehensive backtesting for portfolio exit strategies
    Tests: Cut losers fast vs sophisticated winner exit timing
    """
    
    def __init__(self):
        """Initialize portfolio exit backtesting system"""
        try:
            from polygon import RESTClient
            from config.api_keys import get_polygon_key
            polygon_key = get_polygon_key()
            self.polygon_client = RESTClient(polygon_key) if polygon_key else None
        except ImportError:
            self.polygon_client = None
            logger.warning("Polygon client not available")
        
        self.positions = {}  # Current positions
        self.closed_positions = []  # Historical closed positions
        self.results = {
            'total_positions': 0,
            'winners_closed': 0,
            'losers_closed': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_winner_return': 0.0,
            'avg_loser_return': 0.0,
            'avg_winner_hold_days': 0.0,
            'avg_loser_hold_days': 0.0,
            'pullback_resilience': 0.0,  # Winners that survived pullbacks
            'momentum_exit_accuracy': 0.0  # Accuracy of momentum-based exits
        }
    
    def run_portfolio_exit_backtest(self, 
                                  test_positions: List[Dict],
                                  start_date: str = "2023-01-01",
                                  end_date: str = "2024-01-01") -> Dict:
        """
        Run comprehensive portfolio exit strategy backtest
        
        Args:
            test_positions: List of position dictionaries with entry data
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Comprehensive backtest results
        """
        
        if not self.polygon_client:
            return {"error": "Polygon API key required for portfolio exit backtesting"}
        
        logger.info(f"Starting portfolio exit backtest from {start_date} to {end_date}")
        
        # Initialize positions from test data
        for pos_data in test_positions:
            position = Position(
                symbol=pos_data['symbol'],
                entry_date=datetime.strptime(pos_data['entry_date'], "%Y-%m-%d"),
                entry_price=pos_data['entry_price'],
                quantity=pos_data.get('quantity', 100),
                scanner_score=pos_data.get('scanner_score', 50.0),
                momentum_strength=pos_data.get('momentum_strength', 0.5)
            )
            self.positions[pos_data['symbol']] = position
        
        # Run daily monitoring loop
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        current_date = start_dt
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Trading days only
                self._process_daily_positions(current_date)
            current_date += timedelta(days=1)
        
        # Close remaining positions
        self._close_all_remaining_positions(end_dt)
        
        # Calculate final results
        self._calculate_portfolio_metrics()
        
        # Save results
        self._save_portfolio_backtest_results()
        
        return self.results
    
    def _process_daily_positions(self, current_date: datetime):
        """Process all positions for daily exit decisions"""
        
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            try:
                # Update position with current market data
                self._update_position_data(position, current_date)
                
                # Check exit conditions
                exit_decision = self._evaluate_exit_conditions(position, current_date)
                
                if exit_decision['should_exit']:
                    positions_to_close.append((symbol, exit_decision))
                    
            except Exception as e:
                logger.error(f"Error processing position {symbol}: {e}")
                continue
        
        # Execute exits
        for symbol, exit_decision in positions_to_close:
            self._close_position(symbol, exit_decision, current_date)
    
    def _update_position_data(self, position: Position, current_date: datetime):
        """Update position with current market data"""
        
        try:
            # Get current price from Polygon
            current_price = self._get_price_for_date(position.symbol, current_date)
            
            if current_price:
                position.current_price = current_price
                position.days_held = (current_date - position.entry_date).days
                
                # Track max gain/loss
                current_return = position.current_return
                if current_return > position.max_gain:
                    position.max_gain = current_return
                if current_return < position.max_loss:
                    position.max_loss = current_return
                    
        except Exception as e:
            logger.debug(f"Failed to update {position.symbol}: {e}")
    
    def _evaluate_exit_conditions(self, position: Position, current_date: datetime) -> Dict:
        """
        Sophisticated exit condition evaluation
        Different logic for winners vs losers
        """
        
        current_return = position.current_return
        days_held = position.days_held
        
        # LOSER EXIT CONDITIONS (Fast cuts)
        if current_return < 0:
            # Progressive loss-cutting thresholds
            if days_held == 1 and current_return <= -2.0:
                return {'should_exit': True, 'reason': 'Day 1 loss cut (-2%)', 'type': 'loser'}
            elif days_held == 2 and current_return <= -3.0:
                return {'should_exit': True, 'reason': 'Day 2 loss cut (-3%)', 'type': 'loser'}
            elif days_held >= 3 and current_return <= -4.0:
                return {'should_exit': True, 'reason': 'Day 3+ loss cut (-4%)', 'type': 'loser'}
        
        # WINNER EXIT CONDITIONS (Momentum-based)
        elif current_return > 0:
            momentum_signals = self._analyze_winner_momentum(position, current_date)
            
            # Don't exit on small pullbacks if momentum is strong
            if self._is_healthy_pullback(position, momentum_signals):
                return {'should_exit': False, 'reason': 'Healthy pullback - hold', 'type': 'winner'}
            
            # Exit if momentum clearly breaks down
            elif self._is_momentum_breakdown(position, momentum_signals):
                return {'should_exit': True, 'reason': 'Momentum breakdown confirmed', 'type': 'winner'}
            
            # Exit if extreme overextension
            elif current_return > 50 and momentum_signals['rsi'] > 80:
                return {'should_exit': True, 'reason': 'Extreme overextension', 'type': 'winner'}
        
        # Maximum hold period safety
        if days_held > 30:
            return {'should_exit': True, 'reason': 'Maximum hold period reached', 'type': 'time_limit'}
        
        return {'should_exit': False, 'reason': 'Hold position', 'type': 'hold'}
    
    def _analyze_winner_momentum(self, position: Position, current_date: datetime) -> Dict:
        """Analyze momentum signals for winning positions"""
        
        try:
            # Get recent price history for momentum analysis
            historical_data = self._get_historical_data(position.symbol, current_date, days=20)
            
            if historical_data is None or len(historical_data) < 10:
                return {'rsi': 50, 'momentum_score': 0.5, 'volume_trend': 'neutral'}
            
            # Calculate momentum indicators
            prices = historical_data['close']
            volume = historical_data['volume']
            
            # RSI calculation
            rsi = self._calculate_rsi(prices)
            
            # Momentum score (custom multi-factor)
            momentum_score = self._calculate_momentum_score(historical_data)
            
            # Volume trend analysis
            recent_volume = volume.tail(5).mean()
            older_volume = volume.head(10).mean()
            volume_trend = 'increasing' if recent_volume > older_volume * 1.2 else 'decreasing'
            
            return {
                'rsi': rsi,
                'momentum_score': momentum_score,
                'volume_trend': volume_trend,
                'price_trend': 'up' if prices.iloc[-1] > prices.iloc[-5] else 'down'
            }
            
        except Exception as e:
            logger.debug(f"Momentum analysis failed for {position.symbol}: {e}")
            return {'rsi': 50, 'momentum_score': 0.5, 'volume_trend': 'neutral'}
    
    def _is_healthy_pullback(self, position: Position, momentum_signals: Dict) -> bool:
        """Determine if current decline is a healthy pullback vs real weakness"""
        
        current_return = position.current_return
        max_gain = position.max_gain
        
        # Pullback characteristics that suggest holding
        pullback_depth = max_gain - current_return
        
        # Healthy pullback criteria
        conditions = [
            pullback_depth < 8.0,  # Less than 8% pullback from peak
            momentum_signals['momentum_score'] > 0.6,  # Strong underlying momentum
            momentum_signals['volume_trend'] != 'increasing',  # No volume spike on decline
            current_return > 5.0,  # Still significantly positive
            position.days_held < 20  # Not held too long
        ]
        
        # Must meet at least 3/5 criteria
        return sum(conditions) >= 3
    
    def _is_momentum_breakdown(self, position: Position, momentum_signals: Dict) -> bool:
        """Determine if momentum has truly broken down"""
        
        current_return = position.current_return
        max_gain = position.max_gain
        
        # Breakdown criteria
        breakdown_signals = [
            momentum_signals['momentum_score'] < 0.4,  # Weak momentum
            momentum_signals['rsi'] < 30,  # Oversold but in downtrend
            momentum_signals['volume_trend'] == 'increasing',  # Volume on decline
            (max_gain - current_return) > 12.0,  # Large pullback from peak
            momentum_signals['price_trend'] == 'down'  # Clear price downtrend
        ]
        
        # Must meet at least 3/5 breakdown signals
        return sum(breakdown_signals) >= 3
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate custom momentum score (0-1)"""
        try:
            prices = data['close']
            volume = data['volume']
            
            # Price momentum (recent vs older)
            recent_avg = prices.tail(5).mean()
            older_avg = prices.head(10).mean()
            price_momentum = (recent_avg - older_avg) / older_avg
            
            # Volume momentum
            recent_vol = volume.tail(5).mean()
            older_vol = volume.head(10).mean()
            volume_momentum = (recent_vol - older_vol) / older_vol
            
            # Combined score (0-1)
            combined = (price_momentum * 0.7 + volume_momentum * 0.3)
            normalized = max(0, min(1, (combined + 0.2) / 0.4))  # Normalize to 0-1
            
            return normalized
            
        except:
            return 0.5
    
    def _get_price_for_date(self, symbol: str, date: datetime) -> Optional[float]:
        """Get stock price for specific date"""
        try:
            if not self.polygon_client:
                return None
                
            date_str = date.strftime("%Y-%m-%d")
            aggs = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=date_str,
                to=date_str
            )
            
            if aggs and len(aggs) > 0:
                return float(aggs[0].close)
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get price for {symbol} on {date}: {e}")
            return None
    
    def _get_historical_data(self, symbol: str, end_date: datetime, days: int = 20) -> Optional[pd.DataFrame]:
        """Get historical data for momentum analysis"""
        try:
            if not self.polygon_client:
                return None
                
            start_date = end_date - timedelta(days=days+10)  # Extra buffer
            
            aggs = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            
            if not aggs or len(aggs) < 10:
                return None
            
            data = []
            for agg in aggs:
                data.append({
                    'date': pd.to_datetime(agg.timestamp, unit='ms'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
            
            df = pd.DataFrame(data)
            return df.tail(days) if len(df) >= days else df
            
        except Exception as e:
            logger.debug(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    def _close_position(self, symbol: str, exit_decision: Dict, exit_date: datetime):
        """Close position and record results"""
        
        position = self.positions.pop(symbol)
        position.exit_reason = exit_decision['reason']
        
        # Record final metrics
        final_return = position.current_return
        hold_days = position.days_held
        
        # Categorize the trade
        trade_result = {
            'symbol': symbol,
            'entry_date': position.entry_date.strftime("%Y-%m-%d"),
            'exit_date': exit_date.strftime("%Y-%m-%d"),
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'return_pct': final_return,
            'hold_days': hold_days,
            'exit_reason': exit_decision['reason'],
            'exit_type': exit_decision['type'],
            'max_gain': position.max_gain,
            'max_loss': position.max_loss,
            'scanner_score': position.scanner_score
        }
        
        self.closed_positions.append(trade_result)
        logger.info(f"Closed {symbol}: {final_return:.2f}% return, {hold_days} days, reason: {exit_decision['reason']}")
    
    def _close_all_remaining_positions(self, final_date: datetime):
        """Close all remaining positions at end of backtest"""
        
        remaining_symbols = list(self.positions.keys())
        for symbol in remaining_symbols:
            exit_decision = {'reason': 'Backtest end', 'type': 'forced'}
            self._close_position(symbol, exit_decision, final_date)
    
    def _calculate_portfolio_metrics(self):
        """Calculate comprehensive portfolio performance metrics"""
        
        if not self.closed_positions:
            return
        
        winners = [t for t in self.closed_positions if t['return_pct'] > 0]
        losers = [t for t in self.closed_positions if t['return_pct'] <= 0]
        
        # Basic metrics
        self.results.update({
            'total_positions': len(self.closed_positions),
            'winners_closed': len(winners),
            'losers_closed': len(losers),
            'win_rate': len(winners) / len(self.closed_positions) if self.closed_positions else 0,
            'total_return': sum(t['return_pct'] for t in self.closed_positions),
            'avg_winner_return': np.mean([t['return_pct'] for t in winners]) if winners else 0,
            'avg_loser_return': np.mean([t['return_pct'] for t in losers]) if losers else 0,
            'avg_winner_hold_days': np.mean([t['hold_days'] for t in winners]) if winners else 0,
            'avg_loser_hold_days': np.mean([t['hold_days'] for t in losers]) if losers else 0
        })
        
        # Advanced exit strategy metrics
        pullback_survivors = [t for t in winners if t['max_gain'] - t['return_pct'] > 5.0]
        momentum_exits = [t for t in winners if 'momentum' in t['exit_reason'].lower()]
        
        self.results.update({
            'pullback_resilience': len(pullback_survivors) / len(winners) if winners else 0,
            'momentum_exit_accuracy': len([t for t in momentum_exits if t['return_pct'] > 10]) / len(momentum_exits) if momentum_exits else 0
        })
        
        # Risk metrics
        returns = [t['return_pct'] for t in self.closed_positions]
        self.results['max_drawdown'] = min(returns) if returns else 0
        self.results['volatility'] = np.std(returns) if returns else 0
        
        # Loss cutting effectiveness
        early_cuts = [t for t in losers if t['hold_days'] <= 3]
        self.results['early_loss_cut_rate'] = len(early_cuts) / len(losers) if losers else 0
        self.results['avg_early_cut_loss'] = np.mean([t['return_pct'] for t in early_cuts]) if early_cuts else 0
    
    def _save_portfolio_backtest_results(self):
        """Save portfolio backtest results to file"""
        
        try:
            results_dir = "data/backtest_results"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_exit_backtest_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            full_results = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'test_type': 'portfolio_exit_strategy',
                    'description': 'Dual exit strategy: Cut losers fast vs momentum-based winner exits'
                },
                'summary_metrics': self.results,
                'individual_trades': self.closed_positions
            }
            
            with open(filepath, 'w') as f:
                json.dump(full_results, f, indent=2, default=str)
            
            # Also save as latest
            latest_filepath = os.path.join(results_dir, "latest_portfolio_backtest.json")
            with open(latest_filepath, 'w') as f:
                json.dump(full_results, f, indent=2, default=str)
            
            logger.info(f"Portfolio exit backtest results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save portfolio backtest results: {e}")

def create_sample_portfolio_positions() -> List[Dict]:
    """Create sample portfolio positions for testing"""
    
    # Sample positions based on typical scanner outputs
    sample_positions = [
        {'symbol': 'AAPL', 'entry_date': '2023-03-15', 'entry_price': 155.50, 'scanner_score': 65.2, 'momentum_strength': 0.75},
        {'symbol': 'NVDA', 'entry_date': '2023-03-20', 'entry_price': 220.30, 'scanner_score': 78.1, 'momentum_strength': 0.85},
        {'symbol': 'TSLA', 'entry_date': '2023-04-01', 'entry_price': 180.00, 'scanner_score': 45.8, 'momentum_strength': 0.45},
        {'symbol': 'MSFT', 'entry_date': '2023-04-10', 'entry_price': 280.75, 'scanner_score': 72.3, 'momentum_strength': 0.70},
        {'symbol': 'GOOGL', 'entry_date': '2023-04-15', 'entry_price': 105.20, 'scanner_score': 58.9, 'momentum_strength': 0.60},
        {'symbol': 'META', 'entry_date': '2023-05-01', 'entry_price': 240.50, 'scanner_score': 82.4, 'momentum_strength': 0.80},
        {'symbol': 'AMD', 'entry_date': '2023-05-10', 'entry_price': 85.30, 'scanner_score': 41.2, 'momentum_strength': 0.40},
        {'symbol': 'NFLX', 'entry_date': '2023-05-20', 'entry_price': 385.00, 'scanner_score': 69.7, 'momentum_strength': 0.65}
    ]
    
    return sample_positions