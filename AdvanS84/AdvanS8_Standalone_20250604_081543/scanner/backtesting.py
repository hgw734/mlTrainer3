"""
Comprehensive backtesting system for the momentum scanner using Polygon historical data.
Tests signal accuracy, performance metrics, and risk-adjusted returns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from polygon import RESTClient
import os
from .core import MomentumScanner
from .data_provider import DataProvider
from .dynamic_exit_system import DynamicExitSystem

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Comprehensive backtesting engine that validates scanner performance
    using authentic Polygon historical data.
    """
    
    def __init__(self):
        """Initialize backtesting engine"""
        self.polygon_client = None
        self.scanner = MomentumScanner()
        self.data_provider = DataProvider()
        self.exit_system = DynamicExitSystem()
        
        # Initialize Polygon client
        from config.api_keys import get_polygon_key
        api_key = get_polygon_key()
        if api_key:
            self.polygon_client = RESTClient(api_key)
        
        self.backtest_results = {}
        
    def run_historical_backtest(self, 
                              start_date: str = "2024-01-01",
                              end_date: str = "2024-11-01",
                              min_score: float = 50.0,
                              hold_period: int = 5,
                              max_positions: int = 20) -> Dict:
        """
        Run comprehensive backtest using historical Polygon data
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            min_score: Minimum scanner score to generate signals
            hold_period: Days to hold positions
            max_positions: Maximum concurrent positions
            
        Returns:
            Dictionary with comprehensive backtest results
        """
        
        if not self.polygon_client:
            return {"error": "Polygon API key required for backtesting"}
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Generate test dates (weekly scans)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        test_dates = []
        current_date = start_dt
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Weekdays only
                test_dates.append(current_date)
            current_date += timedelta(days=7)  # Weekly scans
        
        results = {
            'total_signals': 0,
            'profitable_signals': 0,
            'total_return': 0.0,
            'winning_trades': [],
            'losing_trades': [],
            'daily_returns': [],
            'signal_history': [],
            'performance_metrics': {}
        }
        
        current_positions = {}
        
        for scan_date in test_dates:
            try:
                # Run scanner for this date
                signals = self._run_historical_scan(scan_date, min_score)
                
                # Close expired positions
                self._close_expired_positions(current_positions, scan_date, hold_period, results)
                
                # Open new positions
                self._open_new_positions(signals, current_positions, scan_date, max_positions, results)
                
                logger.info(f"Processed {scan_date.strftime('%Y-%m-%d')}: {len(signals)} signals, {len(current_positions)} active positions")
                
            except Exception as e:
                logger.error(f"Error processing {scan_date}: {e}")
                continue
        
        # Close any remaining positions
        final_date = test_dates[-1] if test_dates else end_dt
        self._close_all_positions(current_positions, final_date, results)
        
        # Calculate performance metrics
        results['performance_metrics'] = self._calculate_performance_metrics(results)
        
        # Store results
        self.backtest_results = results
        
        # Automatically save results to file
        save_success = self.save_backtest_results(results)
        if save_success:
            logger.info("Backtest results saved to data/backtest_results/")
        
        logger.info(f"Backtest completed: {results['total_signals']} signals, {results['performance_metrics'].get('win_rate', 0):.1%} win rate")
        
        return results
    
    def _run_historical_scan(self, scan_date: datetime, min_score: float) -> List[Dict]:
        """Run scanner for historical date"""
        
        # Get a subset of symbols for testing (to manage API limits)
        from config.stock_universe import INSTITUTIONAL_UNIVERSE
        test_symbols = INSTITUTIONAL_UNIVERSE[:50]  # Test with 50 symbols
        
        signals = []
        
        for symbol in test_symbols:
            try:
                # Get historical data for this date
                historical_data = self._get_historical_data(symbol, scan_date)
                
                if historical_data is None or len(historical_data) < 50:
                    continue
                
                # Simulate scanner analysis for this historical point
                signal = self._simulate_scanner_signal(symbol, historical_data, scan_date, min_score)
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.debug(f"Error analyzing {symbol} for {scan_date}: {e}")
                continue
        
        return signals
    
    def _get_historical_data(self, symbol: str, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical price data from Polygon"""
        
        try:
            # Get 100 days of data before the scan date
            start_date = end_date - timedelta(days=150)
            
            aggs = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                adjusted=True,
                sort="asc",
                limit=50000
            )
            
            if not aggs or len(aggs) < 50:
                return None
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.debug(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    def _simulate_scanner_signal(self, symbol: str, data: pd.DataFrame, scan_date: datetime, min_score: float) -> Optional[Dict]:
        """Simulate scanner analysis for historical data point"""
        
        try:
            # Calculate basic momentum indicators
            current_price = data['close'].iloc[-1]
            
            # Price momentum (various timeframes)
            returns_3d = (current_price / data['close'].iloc[-4] - 1) * 100 if len(data) >= 4 else 0
            returns_5d = (current_price / data['close'].iloc[-6] - 1) * 100 if len(data) >= 6 else 0
            returns_10d = (current_price / data['close'].iloc[-11] - 1) * 100 if len(data) >= 11 else 0
            returns_20d = (current_price / data['close'].iloc[-21] - 1) * 100 if len(data) >= 21 else 0
            
            # Volume analysis
            avg_volume = data['volume'].tail(20).mean()
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Technical indicators
            rsi = self._calculate_rsi(data['close'], 14)
            sma_20 = data['close'].tail(20).mean()
            price_vs_sma = (current_price / sma_20 - 1) * 100
            
            # Simplified scoring (mimics actual scanner logic)
            momentum_score = (returns_3d * 0.3 + returns_5d * 0.25 + returns_10d * 0.25 + returns_20d * 0.2)
            technical_score = min(max((rsi - 30) / 40 * 100, 0), 100)
            volume_score = min(volume_ratio * 25, 100)
            trend_score = min(max(price_vs_sma + 50, 0), 100)
            
            # Composite score
            composite_score = (momentum_score * 0.4 + technical_score * 0.3 + volume_score * 0.2 + trend_score * 0.1)
            composite_score = max(min(composite_score, 100), 0)  # Clamp to 0-100
            
            if composite_score >= min_score:
                return {
                    'symbol': symbol,
                    'scan_date': scan_date,
                    'entry_price': current_price,
                    'composite_score': composite_score,
                    'momentum_score': momentum_score,
                    'volume_ratio': volume_ratio,
                    'returns_5d': returns_5d,
                    'rsi': rsi
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error simulating signal for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except:
            return 50.0
    
    def _close_expired_positions(self, positions: Dict, current_date: datetime, hold_period: int, results: Dict):
        """Close positions based on aggressive dynamic exit conditions"""
        
        symbols_to_close = []
        
        for symbol, position in positions.items():
            # Get current market data for the symbol
            current_data = self._get_current_market_data(symbol, current_date)
            
            if not current_data:
                # Fallback to time-based exit if no data available
                days_held = (current_date - position['entry_date']).days
                if days_held >= hold_period:
                    symbols_to_close.append((symbol, f"Time-based exit ({days_held} days)"))
                continue
            
            # Calculate current performance
            entry_price = position['entry_price']
            current_price = current_data['current_price']
            current_return = (current_price - entry_price) / entry_price
            days_held = (current_date - position['entry_date']).days
            
            # AGGRESSIVE LOSS-CUTTING SYSTEM
            should_exit = False
            exit_reason = ""
            
            if current_return < 0:  # LOSING TRADE - 25% confidence threshold
                if days_held == 1 and current_return <= -0.02:  # Day 1: -2% = IMMEDIATE EXIT
                    should_exit = True
                    exit_reason = "Day 1 aggressive loss cut (-2%)"
                elif days_held == 2 and current_return <= -0.03:  # Day 2: -3% = IMMEDIATE EXIT
                    should_exit = True
                    exit_reason = "Day 2 aggressive loss cut (-3%)"
                elif days_held == 3 and current_return <= -0.04:  # Day 3: -4% = IMMEDIATE EXIT
                    should_exit = True
                    exit_reason = "Day 3 aggressive loss cut (-4%)"
                elif days_held > 3 and current_return <= -0.015:  # After day 3: -1.5% = EXIT
                    should_exit = True
                    exit_reason = "Sustained loss cut (-1.5%)"
                elif current_return <= -0.06:  # Hard stop loss at -6%
                    should_exit = True
                    exit_reason = "Hard stop loss (-6%)"
                
                # Additional momentum checks for losing trades
                rsi = current_data.get('rsi', 50)
                volume_ratio = current_data.get('volume_ratio', 1.0)
                
                if rsi > 70 and volume_ratio < 0.5:  # Overbought + volume exhaustion
                    should_exit = True
                    exit_reason = "Momentum reversal on loss"
            
            else:  # WINNING TRADE - 75% confidence threshold (let winners run)
                # Only exit winners on severe deterioration
                rsi = current_data.get('rsi', 50)
                volume_ratio = current_data.get('volume_ratio', 1.0)
                
                # Much higher bar for exiting profitable trades
                if rsi > 80 and volume_ratio < 0.3 and current_return > 0.15:  # Severe overbought + volume collapse + good profit
                    should_exit = True
                    exit_reason = "Severe deterioration on winner (profit secured)"
                elif days_held >= 30:  # Maximum 30-day hold
                    should_exit = True
                    exit_reason = "Maximum hold period (30 days)"
            
            if should_exit:
                symbols_to_close.append((symbol, exit_reason))
        
        # Execute exits
        for symbol, exit_reason in symbols_to_close:
            self._execute_position_exit(symbol, positions[symbol], current_date, exit_reason, results)
            del positions[symbol]
    
    def _get_current_market_data(self, symbol: str, date: datetime) -> Dict:
        """Get current market data for dynamic exit analysis"""
        try:
            # Get recent price data
            end_date = date
            start_date = date - timedelta(days=30)  # Get 30 days of data for analysis
            
            if not self.polygon_client:
                return None
            
            # Get historical bars
            aggs = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            
            if not aggs or len(aggs) == 0:
                return None
            
            # Convert to lists for analysis
            prices = [bar.close for bar in aggs]
            volumes = [bar.volume for bar in aggs]
            
            if len(prices) < 5:
                return None
            
            # Calculate technical indicators
            current_price = prices[-1]
            rsi = self._calculate_rsi(prices[-14:]) if len(prices) >= 14 else 50
            volume_ratio = volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 else 1.0
            
            return {
                'current_price': current_price,
                'prices': prices,
                'volumes': volumes,
                'rsi': rsi,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _determine_market_regime(self, date: datetime) -> str:
        """Determine current market regime for exit decisions"""
        # Simplified regime detection - in practice, this would use more sophisticated analysis
        try:
            # Get SPY data for market regime
            spy_data = self._get_current_market_data('SPY', date)
            if not spy_data:
                return 'neutral_market'
            
            spy_prices = spy_data.get('prices', [])
            if len(spy_prices) < 20:
                return 'neutral_market'
            
            # Simple trend analysis
            short_trend = (spy_prices[-1] / spy_prices[-6] - 1) * 100  # 5-day trend
            medium_trend = (spy_prices[-1] / spy_prices[-21] - 1) * 100  # 20-day trend
            
            # Determine regime based on trends
            if short_trend > 2 and medium_trend > 5:
                return 'bull_market'
            elif short_trend < -2 and medium_trend < -5:
                return 'bear_market'
            elif abs(short_trend) > 3 or abs(medium_trend) > 8:
                return 'volatile_market'
            else:
                return 'neutral_market'
                
        except Exception:
            return 'neutral_market'
    
    def _get_max_hold_period(self, market_regime: str) -> int:
        """Get maximum hold period by market regime"""
        regime_limits = {
            'bull_market': 25,       # Hold longer in bull markets for momentum runs
            'bear_market': 10,       # Exit quickly in bear markets
            'volatile_market': 15,   # Moderate hold in volatile conditions
            'neutral_market': 20     # Standard hold period
        }
        return regime_limits.get(market_regime, 20)
    
    def _execute_position_exit(self, symbol: str, position: Dict, exit_date: datetime, exit_reason: str, results: Dict):
        """Execute position exit and record results"""
        exit_price = self._get_exit_price(symbol, exit_date)
        
        if exit_price:
            entry_price = position['entry_price']
            trade_return = (exit_price / entry_price - 1) * 100
            days_held = (exit_date - position['entry_date']).days
            
            trade_result = {
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': trade_return,
                'days_held': days_held,
                'score': position['score'],
                'exit_reason': exit_reason
            }
            
            results['total_return'] += trade_return
            
            if trade_return > 0:
                results['profitable_signals'] += 1
                results['winning_trades'].append(trade_result)
            else:
                results['losing_trades'].append(trade_result)
            
            results['daily_returns'].append(trade_return)
    
    def _open_new_positions(self, signals: List[Dict], positions: Dict, scan_date: datetime, max_positions: int, results: Dict):
        """Open new positions from signals"""
        
        # Sort signals by score
        signals.sort(key=lambda x: x['composite_score'], reverse=True)
        
        for signal in signals:
            if len(positions) >= max_positions:
                break
            
            symbol = signal['symbol']
            
            if symbol not in positions:
                positions[symbol] = {
                    'entry_date': scan_date,
                    'entry_price': signal['entry_price'],
                    'score': signal['composite_score']
                }
                
                results['total_signals'] += 1
                results['signal_history'].append(signal)
    
    def _close_all_positions(self, positions: Dict, final_date: datetime, results: Dict):
        """Close all remaining positions at end of backtest"""
        
        for symbol, position in positions.items():
            exit_price = self._get_exit_price(symbol, final_date)
            
            if exit_price:
                entry_price = position['entry_price']
                trade_return = (exit_price / entry_price - 1) * 100
                days_held = (final_date - position['entry_date']).days
                
                # Calculate advanced metrics
                profit_targets = self._calculate_profit_targets(trade_return, days_held)
                
                trade_result = {
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': final_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': trade_return,
                    'days_held': days_held,
                    'score': position['score'],
                    'hit_5pct_target': profit_targets['hit_5pct'],
                    'hit_10pct_target': profit_targets['hit_10pct'],
                    'hit_15pct_target': profit_targets['hit_15pct'],
                    'max_gain': profit_targets['max_gain'],
                    'max_drawdown': profit_targets['max_drawdown'],
                    'risk_reward_ratio': profit_targets['risk_reward']
                }
                
                results['total_return'] += trade_return
                
                if trade_return > 0:
                    results['profitable_signals'] += 1
                    results['winning_trades'].append(trade_result)
                else:
                    results['losing_trades'].append(trade_result)
                
                results['daily_returns'].append(trade_return)
    
    def _calculate_profit_targets(self, final_return: float, days_held: int) -> Dict[str, Any]:
        """Calculate whether profit targets were hit and risk metrics"""
        
        # For now, estimate based on final return and typical momentum patterns
        # In a real system, this would track intraday highs/lows
        
        # Estimate maximum gain during holding period
        # Momentum stocks typically see 60-80% of their total move in first 2-3 days
        if days_held <= 2:
            estimated_max_gain = final_return * 1.2  # Assume some additional upside was captured
        elif days_held <= 5:
            estimated_max_gain = final_return * 1.4  # More volatility in longer holds
        else:
            estimated_max_gain = final_return * 1.6  # Extended holds see more fluctuation
        
        # Estimate maximum drawdown (typical momentum stock behavior)
        if final_return > 0:
            # Winning trades usually see 2-5% temporary drawdown
            estimated_drawdown = min(-2.0, final_return * -0.3)
        else:
            # Losing trades often see additional downside beyond final loss
            estimated_drawdown = final_return * 1.2
        
        # Calculate target achievement
        hit_5pct = estimated_max_gain >= 5.0
        hit_10pct = estimated_max_gain >= 10.0
        hit_15pct = estimated_max_gain >= 15.0
        
        # Risk-reward ratio
        risk = abs(estimated_drawdown) if estimated_drawdown < 0 else 1.0
        reward = max(estimated_max_gain, 0)
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'hit_5pct': hit_5pct,
            'hit_10pct': hit_10pct,
            'hit_15pct': hit_15pct,
            'max_gain': estimated_max_gain,
            'max_drawdown': estimated_drawdown,
            'risk_reward': risk_reward
        }
    
    def _get_exit_price(self, symbol: str, exit_date: datetime) -> Optional[float]:
        """Get exit price for a symbol on given date"""
        
        try:
            # Get price data around exit date
            start_date = exit_date - timedelta(days=5)
            end_date = exit_date + timedelta(days=5)
            
            aggs = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                adjusted=True,
                sort="asc",
                limit=10
            )
            
            if aggs and len(aggs) > 0:
                # Find closest trading day
                target_timestamp = exit_date.timestamp() * 1000
                
                closest_agg = min(aggs, key=lambda x: abs(x.timestamp - target_timestamp))
                return closest_agg.close
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get exit price for {symbol}: {e}")
            return None
    
    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        total_trades = len(results['winning_trades']) + len(results['losing_trades'])
        
        if total_trades == 0:
            return {'error': 'No completed trades to analyze'}
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(results['winning_trades']),
            'losing_trades': len(results['losing_trades']),
            'win_rate': len(results['winning_trades']) / total_trades,
            'total_return': results['total_return'],
            'average_return': results['total_return'] / total_trades,
        }
        
        # Calculate additional metrics
        if results['daily_returns']:
            returns_array = np.array(results['daily_returns'])
            metrics.update({
                'std_dev': np.std(returns_array),
                'sharpe_ratio': (np.mean(returns_array) / np.std(returns_array)) if np.std(returns_array) > 0 else 0,
                'max_return': np.max(returns_array),
                'min_return': np.min(returns_array),
                'median_return': np.median(returns_array)
            })
        
        # Win/Loss analysis
        if results['winning_trades']:
            winning_returns = [trade['return_pct'] for trade in results['winning_trades']]
            metrics['avg_winning_trade'] = np.mean(winning_returns)
            metrics['max_winning_trade'] = np.max(winning_returns)
        
        if results['losing_trades']:
            losing_returns = [trade['return_pct'] for trade in results['losing_trades']]
            metrics['avg_losing_trade'] = np.mean(losing_returns)
            metrics['max_losing_trade'] = np.min(losing_returns)
        
        # Risk metrics
        if 'avg_winning_trade' in metrics and 'avg_losing_trade' in metrics:
            metrics['profit_factor'] = abs(metrics['avg_winning_trade'] / metrics['avg_losing_trade']) if metrics['avg_losing_trade'] != 0 else float('inf')
        
        return metrics
    
    def save_backtest_results(self, results: Dict, filename: str = None) -> bool:
        """Save backtest results to JSON file"""
        import json
        import os
        from datetime import datetime
        
        try:
            # Create results directory if it doesn't exist
            results_dir = "data/backtest_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backtest_{timestamp}.json"
            
            filepath = os.path.join(results_dir, filename)
            
            # Add metadata to results
            results_with_metadata = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'exit_strategy': 'aggressive_loss_cutting',
                    'description': 'Backtest with Day 1-3 progressive loss cuts (-2%, -3%, -4%)'
                },
                'results': results
            }
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(results_with_metadata, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to: {filepath}")
            
            # Also save as latest results
            latest_filepath = os.path.join(results_dir, "latest_backtest.json")
            with open(latest_filepath, 'w') as f:
                json.dump(results_with_metadata, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
            return False
    
    def load_backtest_results(self, filename: str = "latest_backtest.json") -> Dict:
        """Load saved backtest results from JSON file"""
        import json
        import os
        
        try:
            results_dir = "data/backtest_results"
            filepath = os.path.join(results_dir, filename)
            
            if not os.path.exists(filepath):
                return {'error': f'No saved results found at {filepath}'}
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load backtest results: {e}")
            return {'error': f'Failed to load results: {str(e)}'}
    
    def get_backtest_summary(self) -> Dict:
        """Get formatted backtest summary"""
        
        if not self.backtest_results:
            return {'error': 'No backtest results available'}
        
        results = self.backtest_results
        metrics = results.get('performance_metrics', {})
        
        summary = {
            'overview': {
                'total_signals': results.get('total_signals', 0),
                'completed_trades': metrics.get('total_trades', 0),
                'win_rate': f"{metrics.get('win_rate', 0):.1%}",
                'total_return': f"{results.get('total_return', 0):.2f}%"
            },
            'performance': {
                'average_return_per_trade': f"{metrics.get('average_return', 0):.2f}%",
                'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'max_winning_trade': f"{metrics.get('max_winning_trade', 0):.2f}%",
                'max_losing_trade': f"{metrics.get('max_losing_trade', 0):.2f}%"
            },
            'risk_metrics': {
                'volatility': f"{metrics.get('std_dev', 0):.2f}%",
                'profit_factor': f"{metrics.get('profit_factor', 0):.2f}",
                'winning_trades': metrics.get('winning_trades', 0),
                'losing_trades': metrics.get('losing_trades', 0)
            }
        }
        
        return summary