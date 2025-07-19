"""
Performance analysis system to identify weaknesses and optimization opportunities
for the momentum scanner across different market environments.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from .backtesting import BacktestEngine

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Analyzes scanner performance across different market conditions
    to identify weaknesses and optimization opportunities.
    """
    
    def __init__(self):
        self.backtest_engine = BacktestEngine()
        
    def analyze_market_regime_performance(self, 
                                        start_date: str = "2022-01-01",
                                        end_date: str = "2024-11-01") -> Dict:
        """
        Analyze performance across different market regimes to identify weaknesses
        
        Returns:
            Dictionary with performance by market regime
        """
        
        # Define market periods with different characteristics
        market_periods = {
            "Bull Market 2021": ("2021-01-01", "2021-12-31", "bull"),
            "Bear Market 2022": ("2022-01-01", "2022-12-31", "bear"), 
            "Recovery 2023": ("2023-01-01", "2023-12-31", "recovery"),
            "Mixed 2024": ("2024-01-01", "2024-10-31", "mixed")
        }
        
        results = {}
        
        for period_name, (start, end, regime) in market_periods.items():
            try:
                # Run backtest for this specific period
                backtest_results = self.backtest_engine.run_historical_backtest(
                    start_date=start,
                    end_date=end,
                    min_score=35.0,  # Lower threshold for more signals
                    hold_period=5,
                    max_positions=20
                )
                
                if 'error' not in backtest_results:
                    metrics = backtest_results.get('performance_metrics', {})
                    
                    results[period_name] = {
                        'regime': regime,
                        'period': f"{start} to {end}",
                        'total_signals': backtest_results.get('total_signals', 0),
                        'completed_trades': metrics.get('total_trades', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'total_return': backtest_results.get('total_return', 0),
                        'avg_return': metrics.get('average_return', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'profit_factor': metrics.get('profit_factor', 0)
                    }
                else:
                    results[period_name] = {'error': backtest_results['error']}
                    
            except Exception as e:
                results[period_name] = {'error': str(e)}
                
        return results
    
    def identify_weak_sectors(self, backtest_results: Dict) -> Dict:
        """
        Analyze which sectors/stocks perform poorly with current scoring
        """
        
        losing_trades = backtest_results.get('losing_trades', [])
        winning_trades = backtest_results.get('winning_trades', [])
        
        if not losing_trades and not winning_trades:
            return {'error': 'No trade data available for analysis'}
        
        # Analyze by symbol performance
        symbol_performance = {}
        
        for trade in losing_trades + winning_trades:
            symbol = trade['symbol']
            if symbol not in symbol_performance:
                symbol_performance[symbol] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_return': 0.0,
                    'avg_score': 0.0,
                    'scores': []
                }
            
            symbol_performance[symbol]['total_trades'] += 1
            symbol_performance[symbol]['total_return'] += trade['return_pct']
            symbol_performance[symbol]['scores'].append(trade['score'])
            
            if trade['return_pct'] > 0:
                symbol_performance[symbol]['winning_trades'] += 1
        
        # Calculate win rates and identify weak performers
        weak_performers = []
        strong_performers = []
        
        for symbol, stats in symbol_performance.items():
            if stats['total_trades'] >= 3:  # Only analyze symbols with sufficient trades
                win_rate = stats['winning_trades'] / stats['total_trades']
                avg_return = stats['total_return'] / stats['total_trades']
                avg_score = np.mean(stats['scores'])
                
                performance_data = {
                    'symbol': symbol,
                    'trades': stats['total_trades'],
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'avg_score': avg_score
                }
                
                if win_rate < 0.4 or avg_return < -1.0:
                    weak_performers.append(performance_data)
                elif win_rate > 0.7 and avg_return > 2.0:
                    strong_performers.append(performance_data)
        
        return {
            'weak_performers': sorted(weak_performers, key=lambda x: x['win_rate']),
            'strong_performers': sorted(strong_performers, key=lambda x: x['win_rate'], reverse=True),
            'total_symbols_analyzed': len(symbol_performance)
        }
    
    def analyze_score_thresholds(self, 
                               start_date: str = "2023-01-01",
                               end_date: str = "2024-06-01") -> Dict:
        """
        Test different minimum score thresholds to find optimal accuracy vs opportunity balance
        """
        
        thresholds_to_test = [30, 35, 40, 45, 50, 55, 60]
        results = {}
        
        for threshold in thresholds_to_test:
            try:
                backtest_results = self.backtest_engine.run_historical_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    min_score=float(threshold),
                    hold_period=5,
                    max_positions=15
                )
                
                if 'error' not in backtest_results:
                    metrics = backtest_results.get('performance_metrics', {})
                    
                    results[f"threshold_{threshold}"] = {
                        'min_score': threshold,
                        'total_signals': backtest_results.get('total_signals', 0),
                        'completed_trades': metrics.get('total_trades', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'avg_return': metrics.get('average_return', 0),
                        'total_return': backtest_results.get('total_return', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0)
                    }
                    
            except Exception as e:
                logger.error(f"Error testing threshold {threshold}: {e}")
                
        return results
    
    def get_optimization_recommendations(self, analysis_results: Dict) -> List[str]:
        """
        Generate specific recommendations based on performance analysis
        """
        recommendations = []
        
        # Analyze market regime performance
        if 'market_regimes' in analysis_results:
            regime_data = analysis_results['market_regimes']
            
            best_regime = None
            worst_regime = None
            best_win_rate = 0
            worst_win_rate = 1
            
            for period, data in regime_data.items():
                if 'win_rate' in data:
                    if data['win_rate'] > best_win_rate:
                        best_win_rate = data['win_rate']
                        best_regime = (period, data['regime'])
                    if data['win_rate'] < worst_win_rate:
                        worst_win_rate = data['win_rate']
                        worst_regime = (period, data['regime'])
            
            if worst_regime:
                recommendations.append(f"Scanner struggles in {worst_regime[1]} markets ({worst_regime[0]}: {worst_win_rate:.1%} win rate)")
                
                if worst_regime[1] == 'bear':
                    recommendations.append("Consider adding defensive filters for bear markets: lower RSI thresholds, stronger volume confirmation")
                elif worst_regime[1] == 'mixed':
                    recommendations.append("In mixed markets, consider shorter hold periods and stricter momentum requirements")
        
        # Analyze score threshold optimization
        if 'score_thresholds' in analysis_results:
            threshold_data = analysis_results['score_thresholds']
            
            best_combo = None
            best_score = 0
            
            for threshold_key, data in threshold_data.items():
                if data.get('completed_trades', 0) > 0:
                    # Combined score: win_rate * signal_count (balancing accuracy with opportunity)
                    combined_score = data.get('win_rate', 0) * data.get('total_signals', 0)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_combo = data
            
            if best_combo:
                recommendations.append(f"Optimal threshold appears to be {best_combo['min_score']} (Win rate: {best_combo['win_rate']:.1%}, Signals: {best_combo['total_signals']})")
        
        # Analyze weak sectors
        if 'sector_analysis' in analysis_results:
            weak_performers = analysis_results['sector_analysis'].get('weak_performers', [])
            
            if weak_performers:
                weak_symbols = [p['symbol'] for p in weak_performers[:5]]
                recommendations.append(f"Consider excluding or applying stricter filters to: {', '.join(weak_symbols)}")
                
                # Check if there are patterns in weak performers
                avg_scores = [p['avg_score'] for p in weak_performers]
                if avg_scores:
                    avg_weak_score = np.mean(avg_scores)
                    recommendations.append(f"Weak performers average score: {avg_weak_score:.1f} - consider raising minimum threshold")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Run more comprehensive analysis with different time periods to identify patterns")
            
        recommendations.append("Consider adding additional filters: earnings proximity, sector rotation, macro indicators")
        recommendations.append("Implement dynamic scoring based on market volatility (VIX levels)")
        
        return recommendations

def run_comprehensive_analysis() -> Dict:
    """
    Run complete performance analysis to identify improvement opportunities
    """
    
    analyzer = PerformanceAnalyzer()
    
    results = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'market_regimes': {},
        'score_thresholds': {},
        'sector_analysis': {},
        'recommendations': []
    }
    
    try:
        # Analyze market regime performance
        logger.info("Analyzing performance across market regimes...")
        results['market_regimes'] = analyzer.analyze_market_regime_performance()
        
        # Test score thresholds
        logger.info("Testing optimal score thresholds...")
        results['score_thresholds'] = analyzer.analyze_score_thresholds()
        
        # Run one comprehensive backtest for sector analysis
        logger.info("Running sector performance analysis...")
        comprehensive_backtest = analyzer.backtest_engine.run_historical_backtest(
            start_date="2023-01-01",
            end_date="2024-06-01",
            min_score=35.0,
            hold_period=5,
            max_positions=20
        )
        
        if 'error' not in comprehensive_backtest:
            results['sector_analysis'] = analyzer.identify_weak_sectors(comprehensive_backtest)
        
        # Generate recommendations
        results['recommendations'] = analyzer.get_optimization_recommendations(results)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        results['error'] = str(e)
    
    return results