"""
Core momentum scanner with institutional-grade analysis capabilities.
Simplified version for initial deployment.
"""

import logging
import time
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

from .data_provider import DataProvider
from .technical_analysis import TechnicalAnalyzer
from .fundamental_analysis import FundamentalAnalyzer
from .sentiment_analysis import SentimentAnalyzer
from .scoring_engine import ScoringEngine
from .market_intelligence import MarketIntelligenceEngine
from .adaptive_parameters import AdaptiveParameterEngine
from .signal_filter import AdvancedSignalFilter
from .money_flow_analyzer import MoneyFlowAnalyzer
from .comprehensive_analyzer import ComprehensiveStockAnalyzer
from .continuous_optimizer import global_optimizer
from .layered_enhancements import LayeredEnhancementSystem

# Import full 500-stock institutional universe
from config.stock_universe import get_universe_by_type
INSTITUTIONAL_UNIVERSE = get_universe_by_type("institutional")

logger = logging.getLogger(__name__)

class MomentumScanner:
    """
    Institutional-grade momentum scanner combining technical, fundamental, 
    and sentiment analysis with professional risk metrics.
    """
    
    def __init__(self):
        """Initialize the momentum scanner with all components"""
        self.data_provider = DataProvider()
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.scoring_engine = ScoringEngine()
        self.market_intelligence = MarketIntelligenceEngine()
        
        # Initialize adaptive parameter engine and signal filter
        self.adaptive_engine = AdaptiveParameterEngine()
        self.signal_filter = AdvancedSignalFilter()
        
        # Initialize money flow analyzer for MarketStructureEdge analysis
        self.money_flow_analyzer = MoneyFlowAnalyzer()
        
        # Initialize comprehensive analyzer for complete stock analysis
        self.comprehensive_analyzer = ComprehensiveStockAnalyzer(self.data_provider)
        
        # Initialize layered enhancement system for amplitude-based adjustments
        self.layered_enhancements = LayeredEnhancementSystem(self.data_provider)
        
        self.scan_statistics = {
            'total_scanned': 0,
            'successful_scans': 0,
            'failed_scans': 0,
            'scan_duration': 0,
            'last_scan_time': None
        }
    
    def run_comprehensive_analysis(self, 
                                 symbols: Optional[List[str]] = None,
                                 save_report: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive analysis of ALL stocks before any filtering.
        Calculates ALL criteria for EVERY stock and provides detailed filter analysis.
        
        Args:
            symbols: List of symbols to analyze (defaults to institutional universe)
            save_report: Whether to save detailed analysis report
            
        Returns:
            Complete analysis results with detailed filtering breakdown
        """
        try:
            start_time = time.time()
            
            # Use provided symbols or default universe
            analyze_symbols = symbols or INSTITUTIONAL_UNIVERSE[:10]  # Analyze 10 stocks for comprehensive report
            
            logger.info(f"Starting comprehensive analysis of {len(analyze_symbols)} stocks")
            
            # Run complete analysis using comprehensive analyzer
            analysis_results = self.comprehensive_analyzer.analyze_all_stocks(
                symbols=analyze_symbols,
                timeframe="1D"
            )
            
            # Save detailed report if requested
            if save_report:
                report_path = self.comprehensive_analyzer.save_analysis_report()
                analysis_results['report_path'] = report_path
            
            # Update scan statistics
            scan_duration = time.time() - start_time
            self.scan_statistics.update({
                'total_scanned': self.scan_statistics['total_scanned'] + len(analyze_symbols),
                'scan_duration': scan_duration,
                'last_scan_time': time.time()
            })
            
            logger.info(f"Comprehensive analysis completed in {scan_duration:.2f}s")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                'error': str(e),
                'complete_analysis': [],
                'filtered_results': [],
                'analysis_report': {},
                'filter_report': {},
                'summary_stats': {}
            }
    
    def run_full_scan(self, 
                     symbols: Optional[List[str]] = None,
                     timeframe: str = "all",
                     min_score: float = 30.0,
                     limit: int = 50,
                     progress_callback=None) -> pd.DataFrame:
        """
        Run complete institutional-grade momentum scan
        
        Args:
            symbols: List of symbols to scan (defaults to institutional universe)
            timeframe: Scanning timeframe ('short', 'medium', 'long', 'all')
            min_score: Minimum composite score threshold
            limit: Maximum number of results to return
            
        Returns:
            DataFrame with scan results and scores
        """
        try:
            start_time = time.time()
            
            # Use provided symbols or default universe (full 500 stocks for comprehensive scan)
            scan_symbols = symbols or INSTITUTIONAL_UNIVERSE  # Full institutional universe
            
            logger.info(f"Starting momentum scan for {len(scan_symbols)} symbols")
            
            # Initialize results list
            results = []
            
            # Determine market regime (simplified)
            market_regime = self._get_market_regime()
            
            # Process symbols in smaller batches to respect API rate limits
            batch_size = 10  # Smaller batch size to stay within 80 req/sec limit
            total_symbols = len(scan_symbols)
            processed_count = 0
            
            for i in range(0, len(scan_symbols), batch_size):
                batch = scan_symbols[i:i + batch_size]
                
                # Process each symbol in batch with individual progress updates
                for j, symbol in enumerate(batch):
                    try:
                        # Update progress with current symbol
                        current_processed = processed_count + j + 1
                        if progress_callback:
                            progress_callback(current_processed, total_symbols, symbol)
                        
                        # Process single symbol
                        symbol_result = self._analyze_symbol(symbol, timeframe, market_regime)
                        if symbol_result:
                            results.append(symbol_result)
                            
                    except Exception as e:
                        logger.warning(f"Failed to analyze {symbol}: {e}")
                        continue
                
                # Update total processed count
                processed_count += len(batch)
                    
                # Log progress less frequently to reduce overhead
                if processed_count % 100 == 0 or processed_count == total_symbols:
                    logger.info(f"Scanning progress: {processed_count}/{total_symbols} stocks processed ({processed_count/total_symbols*100:.1f}%)")
            
            # Convert to DataFrame
            if results:
                df = pd.DataFrame(results)
                
                # Apply composite scoring to ALL stocks without initial filtering
                df = self._apply_composite_scoring(df, market_regime, timeframe)
                
                # Show ALL results from complete universe scan
                logger.info(f"Complete scan results: {len(df)} stocks analyzed from {len(scan_symbols)} total")
                if 'composite_score' in df.columns:
                    df_sorted = df.sort_values('composite_score', ascending=False)
                    logger.info(f"Top 20 scores: {df_sorted['composite_score'].head(20).tolist()}")
                
                # Apply filtering ONLY after all stocks are fully analyzed
                optimized_params = global_optimizer.get_optimized_parameters(market_regime)
                threshold_multiplier = optimized_params.get('min_score_multiplier', 1.0)
                adaptive_min_score = max(min_score * threshold_multiplier, 32.0)
                
                logger.info(f"Market regime: {market_regime}, threshold multiplier: {threshold_multiplier:.2f}")
                logger.info(f"Adaptive min score: {adaptive_min_score:.1f} (original: {min_score})")
                
                # Filter by score threshold
                if not df.empty and 'composite_score' in df.columns:
                    score_filtered = df[df['composite_score'] >= adaptive_min_score].copy()
                    logger.info(f"After adaptive score filtering: {len(score_filtered)} signals")
                    
                    # Apply quality filtering only if we have score-qualified signals
                    if len(score_filtered) > 0:
                        market_context = {
                            'regime': market_regime,
                            'vix_level': 20.0,
                            'market_trend': 'neutral'
                        }
                        
                        signals_list = score_filtered.to_dict('records')
                        high_quality_signals = self.signal_filter.apply_advanced_filtering(signals_list, market_context)
                        logger.info(f"After quality filtering: {len(high_quality_signals)} high-quality signals")
                        
                        if high_quality_signals:
                            df = pd.DataFrame(high_quality_signals)
                            if 'quality_score' in df.columns:
                                df = df.sort_values(['quality_score', 'composite_score'], ascending=[False, False])
                            else:
                                df = df.sort_values('composite_score', ascending=False)
                        else:
                            # Return top score-filtered signals if quality filter eliminates all
                            df = score_filtered.sort_values('composite_score', ascending=False)
                            logger.info(f"Quality filter too restrictive, returning {len(df)} score-filtered signals")
                    else:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()
                
                # Apply final limit
                df = df.head(limit)
                
                # Add institutional grades
                if len(df) > 0 and 'composite_score' in df.columns:
                    # Ensure we're working with a DataFrame and Series
                    if isinstance(df, pd.DataFrame):
                        df['grade'] = df['composite_score'].apply(self._assign_grade)
                    else:
                        # Convert to DataFrame if needed
                        df = pd.DataFrame(df)
                        if 'composite_score' in df.columns:
                            df['grade'] = df['composite_score'].apply(self._assign_grade)
                        else:
                            df['grade'] = 'C'
                elif len(df) > 0:
                    df['grade'] = 'C'
                
            else:
                # Return empty DataFrame with expected columns
                expected_columns = [
                    'symbol', 'company_name', 'current_price', 'change_pct',
                    'composite_score', 'technical_score', 'momentum_score',
                    'fundamental_score', 'sentiment_score', 'money_flow_score',
                    'demand_pressure', 'supply_pressure', 'flow_imbalance',
                    'flow_grade', 'institutional_signal', 'grade',
                    'recommendation', 'volume_signal', 'risk_category'
                ]
                df = pd.DataFrame(columns=expected_columns)
            
            # Update statistics
            self.scan_statistics.update({
                'total_scanned': len(scan_symbols),
                'successful_scans': len(df),
                'failed_scans': len(scan_symbols) - len(df),
                'scan_duration': time.time() - start_time,
                'last_scan_time': time.time()
            })
            
            logger.info(f"Scan completed: {len(df)} results in {self.scan_statistics['scan_duration']:.2f}s")
            
            # Ensure we always return a DataFrame
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Full scan failed: {e}")
            return pd.DataFrame()
    
    def run_enhanced_scan(self, 
                         symbols: Optional[List[str]] = None,
                         timeframe: str = "medium",
                         progress_callback=None) -> pd.DataFrame:
        """
        Run enhanced scan with layered amplitude-based adjustments
        
        Args:
            symbols: List of symbols to scan (defaults to institutional universe)
            timeframe: Scanning timeframe ('short', 'medium', 'long')
            progress_callback: Function to call with progress updates
            
        Returns:
            DataFrame with enhanced scoring and regime adjustments
        """
        try:
            start_time = time.time()
            scan_symbols = symbols or INSTITUTIONAL_UNIVERSE[:100]  # Limit for testing
            
            # Get current market regime and amplitude
            market_regime = self.layered_enhancements.get_current_regime()
            sector_momentum = self.layered_enhancements.get_sector_momentum()
            
            logger.info(f"Enhanced scan starting: {len(scan_symbols)} symbols")
            logger.info(f"Market regime: {market_regime.get('regime', 'unknown')} "
                       f"(amplitude: {market_regime.get('amplitude', 0):.2f})")
            
            # Process symbols with enhanced scoring
            enhanced_results = []
            current_positions = 0  # Track positions for regime-based limits
            
            for i, symbol in enumerate(scan_symbols):
                try:
                    if progress_callback:
                        progress_callback(i, len(scan_symbols), symbol)
                    
                    # Get base analysis
                    base_result = self._analyze_symbol(symbol, timeframe, market_regime.get('regime', 'neutral'))
                    
                    if base_result and base_result.get('composite_score', 0) > 0:
                        base_score = base_result['composite_score']
                        
                        # Apply layered enhancements
                        should_enter, enhanced_score, entry_reason = self.layered_enhancements.should_enter_position(
                            symbol, base_score, current_positions
                        )
                        
                        # Update result with enhanced data
                        base_result['enhanced_score'] = enhanced_score
                        base_result['entry_approved'] = should_enter
                        base_result['entry_reason'] = entry_reason
                        base_result['market_regime'] = market_regime.get('regime', 'neutral')
                        base_result['regime_amplitude'] = market_regime.get('amplitude', 0.5)
                        
                        # Add exit parameters for approved entries
                        if should_enter:
                            exit_params = self.layered_enhancements.get_exit_parameters(symbol, 0, 0.0)
                            base_result['dynamic_stop'] = exit_params['stop_loss']
                            base_result['trailing_trigger'] = exit_params['trailing_trigger']
                            current_positions += 1
                        
                        enhanced_results.append(base_result)
                        
                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Convert to DataFrame and sort by enhanced score
            if enhanced_results:
                df = pd.DataFrame(enhanced_results)
                df = df.sort_values('enhanced_score', ascending=False)
                
                # Add enhanced grading
                df['enhanced_grade'] = df['enhanced_score'].apply(self._assign_enhanced_grade)
                
            else:
                # Return empty DataFrame with expected columns
                expected_columns = [
                    'symbol', 'company_name', 'current_price', 'change_pct',
                    'composite_score', 'enhanced_score', 'entry_approved', 'entry_reason',
                    'market_regime', 'regime_amplitude', 'dynamic_stop', 'trailing_trigger',
                    'enhanced_grade', 'technical_score', 'momentum_score',
                    'fundamental_score', 'sentiment_score', 'money_flow_score',
                    'demand_pressure', 'supply_pressure', 'flow_imbalance',
                    'flow_grade', 'institutional_signal', 'grade',
                    'recommendation', 'volume_signal', 'risk_category'
                ]
                df = pd.DataFrame(columns=expected_columns)
            
            # Update statistics
            approved_entries = len(df[df.get('entry_approved', False)]) if len(df) > 0 else 0
            self.scan_statistics.update({
                'total_scanned': len(scan_symbols),
                'successful_scans': len(df),
                'approved_entries': approved_entries,
                'scan_duration': time.time() - start_time,
                'last_scan_time': time.time(),
                'market_regime': market_regime.get('regime', 'neutral'),
                'regime_amplitude': market_regime.get('amplitude', 0.5)
            })
            
            logger.info(f"Enhanced scan completed: {len(df)} analyzed, {approved_entries} approved "
                       f"in {self.scan_statistics['scan_duration']:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Enhanced scan failed: {e}")
            return pd.DataFrame()
    
    def _process_batch(self, symbols: List[str], timeframe: str, market_regime: str) -> List[Dict]:
        """Process a batch of symbols with analysis"""
        batch_results = []
        
        for symbol in symbols:
            try:
                result = self._analyze_symbol(symbol, timeframe, market_regime)
                if result:
                    batch_results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                
        return batch_results
    
    def _analyze_symbol(self, symbol: str, timeframe: str, market_regime: str) -> Optional[Dict]:
        """Perform comprehensive analysis on a single symbol"""
        try:
            # Get market data
            market_data = self.data_provider.get_market_data(symbol, limit=100)
            if market_data is None or market_data.empty:
                return None
            
            # Basic symbol info
            current_price = market_data['close'].iloc[-1]
            change_pct = ((current_price / market_data['close'].iloc[-2]) - 1) * 100
            
            # Technical analysis
            technical_results = self.technical_analyzer.analyze(market_data, timeframe)
            
            # Fundamental analysis (simplified for demo)
            company_data = self.data_provider.get_company_info(symbol)
            fundamental_results = self.fundamental_analyzer.analyze(
                symbol, company_data
            )
            
            # Sentiment analysis (simplified for demo)  
            news_data = self.data_provider.get_news_data(symbol, limit=10)
            sentiment_results = self.sentiment_analyzer.analyze(
                symbol, news_data
            )
            
            # Money Flow Analysis (MarketStructureEdge methodology)
            money_flow_results = self.money_flow_analyzer.analyze_money_flow(market_data)
            
            # Composite scoring (now includes money flow)
            scoring_results = self.scoring_engine.calculate_composite_score(
                technical_results, fundamental_results, sentiment_results,
                market_regime, timeframe, money_flow_results
            )
            
            # Compile result
            result = {
                'symbol': symbol,
                'company_name': company_data.get('name', symbol) if company_data else symbol,
                'current_price': current_price,
                'change_pct': change_pct,
                'volume': market_data['volume'].iloc[-1],
                'market_cap': company_data.get('market_cap', 0) if company_data else 0,
                
                # Core scores
                'composite_score': scoring_results.get('composite_score', 50),
                'technical_score': scoring_results.get('technical_score', 50),
                'momentum_score': scoring_results.get('momentum_score', 50),
                'fundamental_score': scoring_results.get('fundamental_score', 50),
                'sentiment_score': scoring_results.get('sentiment_score', 50),
                'confidence_score': scoring_results.get('confidence_score', 50),
                
                # Technical indicators
                'rsi_14': technical_results.get('rsi_14', 50),
                'macd_signal': technical_results.get('macd_trend', 'NEUTRAL'),
                'bb_position': technical_results.get('bb_position', 0.5),
                'volume_signal': technical_results.get('volume_signal', 'NORMAL'),
                
                # Fundamental metrics
                'pe_ratio': fundamental_results.get('pe_ratio', 0),
                'analyst_rating': fundamental_results.get('analyst_rating', 'HOLD'),
                'earnings_growth': fundamental_results.get('earnings_growth', 0),
                
                # Sentiment metrics
                'sentiment_strength': sentiment_results.get('sentiment_strength', 'NEUTRAL'),
                'news_sentiment': sentiment_results.get('news_sentiment', 50),
                
                # Money Flow Analysis (MarketStructureEdge)
                'money_flow_score': scoring_results.get('money_flow_score', 50),
                'demand_pressure': money_flow_results.get('demand_pressure', 0.5),
                'supply_pressure': money_flow_results.get('supply_pressure', 0.5),
                'flow_imbalance': money_flow_results.get('imbalance_score', 0.0),
                'flow_grade': money_flow_results.get('flow_grade', 'C'),
                'institutional_signal': money_flow_results.get('institutional_flow', {}).get('institutional_signal', False),
                
                # Risk and recommendation
                'recommendation': scoring_results.get('recommendation', 'HOLD'),
                'institutional_grade': scoring_results.get('institutional_grade', 'C'),
                'market_regime': market_regime,
                'timeframe': timeframe
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Symbol analysis failed for {symbol}: {e}")
            return None
    
    def _apply_composite_scoring(self, df: pd.DataFrame, market_regime: str, timeframe: str) -> pd.DataFrame:
        """Apply institutional-grade composite scoring"""
        try:
            # Ensure composite_score exists
            if 'composite_score' not in df.columns:
                df['composite_score'] = 50.0
            
            # Add risk category based on volatility and other factors
            df['risk_category'] = 'MODERATE'
            
            return df
            
        except Exception as e:
            logger.error(f"Composite scoring failed: {e}")
            return df
    
    def _get_market_regime(self) -> str:
        """Determine current market regime for adaptive scoring"""
        try:
            # Simplified market regime detection
            # In a real implementation, this would use VIX data and market indicators
            return 'neutral_market'
            
        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            return 'neutral_market'
    
    def _assign_grade(self, score: float) -> str:
        """Convert numerical score to institutional grade"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        elif score >= 40:
            return 'D'
        else:
            return 'F'
    
    def _assign_enhanced_grade(self, score: float) -> str:
        """Assign enhanced letter grade based on layered enhancement score"""
        if score >= 85:
            return 'A+++'
        elif score >= 80:
            return 'A++'
        elif score >= 75:
            return 'A+'
        elif score >= 70:
            return 'A'
        elif score >= 65:
            return 'A-'
        elif score >= 60:
            return 'B+'
        elif score >= 55:
            return 'B'
        elif score >= 50:
            return 'B-'
        elif score >= 45:
            return 'C+'
        elif score >= 40:
            return 'C'
        elif score >= 35:
            return 'C-'
        elif score >= 30:
            return 'D+'
        elif score >= 25:
            return 'D'
        else:
            return 'F'
    
    def get_scan_statistics(self) -> Dict:
        """Get statistics about the last scan"""
        return self.scan_statistics.copy()
    
    def export_results(self, df: pd.DataFrame, format: str = "csv") -> str:
        """Export scan results for institutional reporting"""
        try:
            if format.lower() == "csv":
                filename = f"momentum_scan_{int(time.time())}.csv"
                df.to_csv(filename, index=False)
                return filename
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ""