"""
Comprehensive stock analyzer that calculates ALL criteria for EVERY stock 
before applying any filters, with detailed reporting of filter analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json

from .data_provider import DataProvider
from .scoring_engine import ScoringEngine
from .money_flow_analyzer import MoneyFlowAnalyzer
from .signal_filter import AdvancedSignalFilter

logger = logging.getLogger(__name__)

class ComprehensiveStockAnalyzer:
    """
    Analyzes every stock completely before applying any filters.
    Provides detailed breakdown of why stocks pass or fail each filter.
    """
    
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self.scoring_engine = ScoringEngine()
        self.money_flow_analyzer = MoneyFlowAnalyzer()
        self.signal_filter = AdvancedSignalFilter()
        
        # Analysis tracking
        self.complete_analysis = []
        self.filter_analysis = {}
        self.analysis_timestamp = None
        
    def analyze_all_stocks(self, symbols: List[str], timeframe: str = "1D") -> Dict[str, Any]:
        """
        Analyze ALL stocks completely before applying any filters.
        
        Returns:
            Complete analysis results with detailed filtering breakdown
        """
        logger.info(f"Starting comprehensive analysis of {len(symbols)} stocks")
        self.analysis_timestamp = datetime.now()
        
        # Reset analysis tracking
        self.complete_analysis = []
        self.filter_analysis = {
            'total_analyzed': 0,
            'filter_failures': {
                'price_filter': {'count': 0, 'stocks': []},
                'volume_filter': {'count': 0, 'stocks': []},
                'market_cap_filter': {'count': 0, 'stocks': []},
                'rsi_filter': {'count': 0, 'stocks': []},
                'momentum_filter': {'count': 0, 'stocks': []},
                'score_filter': {'count': 0, 'stocks': []},
                'quality_filter': {'count': 0, 'stocks': []}
            },
            'pass_rates': {},
            'analysis_details': []
        }
        
        # Step 1: Calculate ALL criteria for EVERY stock
        for i, symbol in enumerate(symbols):
            logger.info(f"Analyzing {symbol} ({i+1}/{len(symbols)})")
            stock_analysis = self._analyze_single_stock(symbol, timeframe)
            if stock_analysis:
                self.complete_analysis.append(stock_analysis)
                self.filter_analysis['total_analyzed'] += 1
        
        # Step 2: Apply filters with detailed tracking
        filtered_results = self._apply_filters_with_tracking()
        
        # Step 3: Generate comprehensive reports
        analysis_report = self._generate_analysis_report()
        filter_report = self._generate_filter_report()
        
        return {
            'complete_analysis': self.complete_analysis,
            'filtered_results': filtered_results,
            'analysis_report': analysis_report,
            'filter_report': filter_report,
            'summary_stats': self._generate_summary_stats()
        }
    
    def _analyze_single_stock(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Analyze single stock completely - ALL criteria calculated"""
        try:
            # Get market data
            df = self.data_provider.get_market_data(symbol, timespan="day", limit=100)
            if df is None or df.empty or len(df) < 20:
                return None
            
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(20).mean()
            
            # Calculate ALL technical indicators
            rsi = self._calculate_rsi(df['close'])
            
            # Calculate momentum metrics
            returns_1d = ((current_price / df['close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
            returns_3d = ((current_price / df['close'].iloc[-4]) - 1) * 100 if len(df) > 3 else 0
            returns_5d = ((current_price / df['close'].iloc[-6]) - 1) * 100 if len(df) > 5 else 0
            returns_10d = ((current_price / df['close'].iloc[-11]) - 1) * 100 if len(df) > 10 else 0
            returns_20d = ((current_price / df['close'].iloc[-21]) - 1) * 100 if len(df) > 20 else 0
            
            # Calculate volume metrics
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            volume_surge = volume_ratio > 1.5
            
            # Calculate Money Flow Analysis
            money_flow_data = self.money_flow_analyzer.analyze_money_flow(df)
            
            # Calculate market cap (estimated)
            shares_outstanding = 1000000000  # Default estimate
            market_cap = current_price * shares_outstanding
            
            # Calculate all scoring components directly
            momentum_score = self._calculate_momentum_score_direct(returns_1d, returns_3d, returns_5d, returns_10d, returns_20d)
            technical_score = self._calculate_technical_score_direct(rsi, volume_ratio)
            fundamental_score = self._calculate_fundamental_score_direct(market_cap, current_price)
            
            sentiment_score = 50.0  # Default neutral
            
            # Calculate composite score
            weights = {
                'momentum': 0.35,
                'technical': 0.30,
                'fundamental': 0.20,
                'sentiment': 0.10,
                'money_flow': 0.15  # MarketStructureEdge weight
            }
            
            money_flow_score = money_flow_data.get('composite_score', 50.0)
            
            composite_score = (
                momentum_score * weights['momentum'] +
                technical_score * weights['technical'] +
                fundamental_score * weights['fundamental'] +
                sentiment_score * weights['sentiment'] +
                money_flow_score * weights['money_flow']
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(df, money_flow_data)
            
            # Complete stock analysis
            analysis = {
                'symbol': symbol,
                'timestamp': self.analysis_timestamp,
                
                # Price data
                'current_price': round(current_price, 2),
                'price_change_1d': round(returns_1d, 2),
                
                # Volume data
                'current_volume': int(current_volume),
                'avg_volume_20d': int(avg_volume),
                'volume_ratio': round(volume_ratio, 2),
                'volume_surge': volume_surge,
                
                # Market metrics
                'market_cap': market_cap,
                'market_cap_billions': round(market_cap / 1e9, 2),
                
                # Technical indicators
                'rsi': round(rsi, 2),
                'rsi_signal': self._get_rsi_signal(rsi),
                
                # Momentum metrics
                'returns_1d': round(returns_1d, 2),
                'returns_3d': round(returns_3d, 2),
                'returns_5d': round(returns_5d, 2),
                'returns_10d': round(returns_10d, 2),
                'returns_20d': round(returns_20d, 2),
                'momentum_trend': self._analyze_momentum_trend(returns_1d, returns_3d, returns_5d),
                
                # Scoring components
                'momentum_score': round(momentum_score, 2),
                'technical_score': round(technical_score, 2),
                'fundamental_score': round(fundamental_score, 2),
                'sentiment_score': round(sentiment_score, 2),
                'money_flow_score': round(money_flow_score, 2),
                'composite_score': round(composite_score, 2),
                
                # Money Flow Analysis
                'money_flow_analysis': money_flow_data,
                'supply_demand_pressure': money_flow_data.get('supply_demand_pressure', 'neutral'),
                'institutional_flow': money_flow_data.get('institutional_flow_detected', False),
                
                # Quality metrics
                'quality_metrics': quality_metrics,
                'quality_score': quality_metrics.get('overall_quality', 50.0),
                
                # Analysis flags
                'analysis_complete': True,
                'data_quality': 'good' if len(df) >= 20 else 'limited'
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'analysis_complete': False,
                'error': str(e),
                'timestamp': self.analysis_timestamp
            }
    
    def _calculate_momentum_score_direct(self, returns_1d, returns_3d, returns_5d, returns_10d, returns_20d):
        """Calculate momentum score directly from returns"""
        try:
            scores = []
            returns = [returns_1d, returns_3d, returns_5d, returns_10d, returns_20d]
            
            for ret in returns:
                if ret > 10:
                    scores.append(90)
                elif ret > 5:
                    scores.append(80)
                elif ret > 2:
                    scores.append(70)
                elif ret > 0:
                    scores.append(60)
                elif ret > -2:
                    scores.append(40)
                elif ret > -5:
                    scores.append(30)
                else:
                    scores.append(20)
            
            return sum(scores) / len(scores) if scores else 50.0
        except:
            return 50.0
    
    def _calculate_technical_score_direct(self, rsi, volume_ratio):
        """Calculate technical score from RSI and volume"""
        try:
            rsi_score = 50.0
            if 30 <= rsi <= 70:
                rsi_score = 80.0
            elif rsi < 30:
                rsi_score = 90.0  # Oversold - potential bounce
            elif rsi > 70:
                rsi_score = 30.0  # Overbought
            
            volume_score = 50.0
            if volume_ratio > 2.0:
                volume_score = 90.0
            elif volume_ratio > 1.5:
                volume_score = 75.0
            elif volume_ratio > 1.0:
                volume_score = 60.0
            
            return (rsi_score + volume_score) / 2
        except:
            return 50.0
    
    def _calculate_fundamental_score_direct(self, market_cap, price):
        """Calculate fundamental score from market cap and price"""
        try:
            # Basic fundamental scoring
            if market_cap > 100_000_000_000:  # Large cap
                cap_score = 80.0
            elif market_cap > 10_000_000_000:  # Mid cap
                cap_score = 70.0
            elif market_cap > 1_000_000_000:  # Small cap
                cap_score = 60.0
            else:
                cap_score = 40.0
            
            # Price reasonableness (simple check)
            if 10 <= price <= 1000:
                price_score = 70.0
            elif 1 <= price <= 2000:
                price_score = 60.0
            else:
                price_score = 50.0
            
            return (cap_score + price_score) / 2
        except:
            return 50.0
    
    def _apply_filters_with_tracking(self) -> List[Dict[str, Any]]:
        """Apply filters while tracking exactly why each stock passes/fails"""
        logger.info("Applying filters with detailed tracking...")
        
        passed_stocks = []
        
        for stock in self.complete_analysis:
            if not stock.get('analysis_complete', False):
                continue
                
            filter_results = {}
            passed_all = True
            
            # Price filter
            price_pass = stock['current_price'] >= 5.0  # Min $5
            filter_results['price_filter'] = {
                'passed': price_pass,
                'value': stock['current_price'],
                'threshold': 5.0,
                'reason': f"Price ${stock['current_price']:.2f} {'≥' if price_pass else '<'} $5.00"
            }
            if not price_pass:
                self.filter_analysis['filter_failures']['price_filter']['count'] += 1
                self.filter_analysis['filter_failures']['price_filter']['stocks'].append(stock['symbol'])
                passed_all = False
            
            # Volume filter
            volume_pass = stock['avg_volume_20d'] >= 500000  # Min 500K avg volume
            filter_results['volume_filter'] = {
                'passed': volume_pass,
                'value': stock['avg_volume_20d'],
                'threshold': 500000,
                'reason': f"Avg volume {stock['avg_volume_20d']:,} {'≥' if volume_pass else '<'} 500K"
            }
            if not volume_pass:
                self.filter_analysis['filter_failures']['volume_filter']['count'] += 1
                self.filter_analysis['filter_failures']['volume_filter']['stocks'].append(stock['symbol'])
                passed_all = False
            
            # Market cap filter
            mcap_pass = stock['market_cap'] >= 1e9  # Min $1B market cap
            filter_results['market_cap_filter'] = {
                'passed': mcap_pass,
                'value': stock['market_cap_billions'],
                'threshold': 1.0,
                'reason': f"Market cap ${stock['market_cap_billions']:.1f}B {'≥' if mcap_pass else '<'} $1.0B"
            }
            if not mcap_pass:
                self.filter_analysis['filter_failures']['market_cap_filter']['count'] += 1
                self.filter_analysis['filter_failures']['market_cap_filter']['stocks'].append(stock['symbol'])
                passed_all = False
            
            # RSI filter - Aggressive momentum trading: extreme oversold = strong buy signals
            rsi_pass = stock['rsi'] <= 80  # Only reject overbought (>80), accept all oversold levels
            rsi_signal = ""
            if stock['rsi'] < 20:
                rsi_signal = " (EXTREME Oversold - Strong Buy Signal)"
            elif stock['rsi'] < 30:
                rsi_signal = " (Oversold - Buy Signal)"
            elif stock['rsi'] > 70:
                rsi_signal = " (Approaching Overbought)"
            elif stock['rsi'] > 80:
                rsi_signal = " (Overbought - High Risk)"
            
            filter_results['rsi_filter'] = {
                'passed': rsi_pass,
                'value': stock['rsi'],
                'threshold': '≤80',
                'reason': f"RSI {stock['rsi']:.1f} {'≤' if rsi_pass else '>'} 80{rsi_signal}"
            }
            if not rsi_pass:
                self.filter_analysis['filter_failures']['rsi_filter']['count'] += 1
                self.filter_analysis['filter_failures']['rsi_filter']['stocks'].append(stock['symbol'])
                passed_all = False
            
            # Momentum filter
            momentum_pass = stock['returns_5d'] > -10  # Not falling too hard
            filter_results['momentum_filter'] = {
                'passed': momentum_pass,
                'value': stock['returns_5d'],
                'threshold': -10.0,
                'reason': f"5-day return {stock['returns_5d']:.1f}% {'>' if momentum_pass else '≤'} -10%"
            }
            if not momentum_pass:
                self.filter_analysis['filter_failures']['momentum_filter']['count'] += 1
                self.filter_analysis['filter_failures']['momentum_filter']['stocks'].append(stock['symbol'])
                passed_all = False
            
            # Score filter
            score_pass = stock['composite_score'] >= 25.0  # Min score threshold
            filter_results['score_filter'] = {
                'passed': score_pass,
                'value': stock['composite_score'],
                'threshold': 25.0,
                'reason': f"Score {stock['composite_score']:.1f} {'≥' if score_pass else '<'} 25.0"
            }
            if not score_pass:
                self.filter_analysis['filter_failures']['score_filter']['count'] += 1
                self.filter_analysis['filter_failures']['score_filter']['stocks'].append(stock['symbol'])
                passed_all = False
            
            # Quality filter
            quality_pass = stock['quality_score'] >= 40.0  # Quality threshold
            filter_results['quality_filter'] = {
                'passed': quality_pass,
                'value': stock['quality_score'],
                'threshold': 40.0,
                'reason': f"Quality {stock['quality_score']:.1f} {'≥' if quality_pass else '<'} 40.0"
            }
            if not quality_pass:
                self.filter_analysis['filter_failures']['quality_filter']['count'] += 1
                self.filter_analysis['filter_failures']['quality_filter']['stocks'].append(stock['symbol'])
                passed_all = False
            
            # Add filter results to stock data
            stock['filter_analysis'] = filter_results
            stock['passed_all_filters'] = passed_all
            
            # Track detailed analysis
            self.filter_analysis['analysis_details'].append({
                'symbol': stock['symbol'],
                'passed_all': passed_all,
                'filters': filter_results,
                'composite_score': stock['composite_score'],
                'quality_score': stock['quality_score']
            })
            
            if passed_all:
                passed_stocks.append(stock)
        
        return passed_stocks
    
    def _calculate_quality_metrics(self, df: pd.DataFrame, money_flow_data: Dict) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        try:
            # Data consistency
            price_volatility = df['close'].pct_change().std() * 100
            volume_consistency = 1 - (df['volume'].std() / df['volume'].mean()) if df['volume'].mean() > 0 else 0
            
            # Trend strength
            price_trend = np.polyfit(range(len(df)), df['close'], 1)[0]
            trend_strength = abs(price_trend) / df['close'].mean() * 100
            
            # Money flow quality
            mf_strength = money_flow_data.get('flow_strength', 0.5)
            
            # Overall quality score
            quality_score = (
                min(100 - price_volatility, 100) * 0.3 +  # Lower volatility = higher quality
                volume_consistency * 100 * 0.2 +
                min(trend_strength * 10, 100) * 0.3 +
                mf_strength * 100 * 0.2
            )
            
            return {
                'price_volatility': round(price_volatility, 2),
                'volume_consistency': round(volume_consistency, 3),
                'trend_strength': round(trend_strength, 2),
                'money_flow_strength': round(mf_strength, 3),
                'overall_quality': round(quality_score, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {'overall_quality': 50.0}
    
    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        total = len(self.complete_analysis)
        successful = len([s for s in self.complete_analysis if s.get('analysis_complete', False)])
        
        # Score distribution
        scores = [s['composite_score'] for s in self.complete_analysis if s.get('analysis_complete', False)]
        
        return {
            'analysis_summary': {
                'total_stocks': total,
                'successful_analysis': successful,
                'failed_analysis': total - successful,
                'success_rate': round(successful / total * 100, 1) if total > 0 else 0
            },
            'score_distribution': {
                'mean_score': round(np.mean(scores), 2) if scores else 0,
                'median_score': round(np.median(scores), 2) if scores else 0,
                'min_score': round(min(scores), 2) if scores else 0,
                'max_score': round(max(scores), 2) if scores else 0,
                'std_score': round(np.std(scores), 2) if scores else 0
            },
            'top_performers': sorted(
                [s for s in self.complete_analysis if s.get('analysis_complete', False)],
                key=lambda x: x['composite_score'],
                reverse=True
            )[:5],
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }
    
    def _generate_filter_report(self) -> Dict[str, Any]:
        """Generate detailed filter analysis report"""
        total = self.filter_analysis['total_analyzed']
        
        # Calculate pass rates
        for filter_name, failure_data in self.filter_analysis['filter_failures'].items():
            pass_rate = ((total - failure_data['count']) / total * 100) if total > 0 else 0
            self.filter_analysis['pass_rates'][filter_name] = round(pass_rate, 1)
        
        return self.filter_analysis
    
    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_analyzed = len(self.complete_analysis)
        successful_analysis = len([s for s in self.complete_analysis if s.get('analysis_complete', False)])
        passed_filters = len([s for s in self.complete_analysis if s.get('passed_all_filters', False)])
        
        return {
            'total_stocks_analyzed': total_analyzed,
            'successful_analysis': successful_analysis,
            'passed_all_filters': passed_filters,
            'overall_pass_rate': round(passed_filters / total_analyzed * 100, 1) if total_analyzed > 0 else 0,
            'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None
        }
    
    def save_analysis_report(self, filepath: str = None) -> str:
        """Save complete analysis to file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"data/comprehensive_analysis_{timestamp}.json"
        
        report_data = {
            'complete_analysis': self.complete_analysis,
            'filter_analysis': self.filter_analysis,
            'analysis_report': self._generate_analysis_report(),
            'summary_stats': self._generate_summary_stats()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            logger.info(f"Analysis report saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _get_rsi_signal(self, rsi: float) -> str:
        """Get RSI signal interpretation"""
        if rsi >= 70:
            return "Overbought"
        elif rsi <= 30:
            return "Oversold"
        elif rsi >= 60:
            return "Strong"
        elif rsi <= 40:
            return "Weak"
        else:
            return "Neutral"
    
    def _analyze_momentum_trend(self, ret_1d: float, ret_3d: float, ret_5d: float) -> str:
        """Analyze momentum trend pattern"""
        if ret_1d > 0 and ret_3d > 0 and ret_5d > 0:
            return "Accelerating Up"
        elif ret_1d > 0 and ret_3d > 0:
            return "Gaining Momentum"
        elif ret_1d < 0 and ret_3d < 0 and ret_5d < 0:
            return "Accelerating Down"
        elif ret_1d < 0 and ret_3d < 0:
            return "Losing Momentum"
        else:
            return "Mixed"