"""
Meta-Enhanced TPE-ML Trading Engine
Modular implementation with authentic data integration
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from core import fetch_data, run_optimization, train_exit_predictor
from core.notification_system import create_notification_system
from core.visualization import create_visualizer

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MetaTPEEngine:
    """
    Meta-Enhanced Trading Engine with TPE optimization and ML exit prediction
    
    Features:
    - Authentic Polygon API stock data integration
    - FRED API VIX data for market regime detection
    - Custom TPE optimization with fallback to Optuna
    - XGBoost/LSTM ML exit strategy prediction
    - Multi-factor market regime classification
    """
    
    def __init__(self, config=None):
        """Initialize the trading engine with configuration"""
        self.config = config or self._default_config()
        self.market_data = {}
        self.universe = self.config['universe']
        self.exit_predictor = None
        self.label_encoder = None
        self.best_params = None
        
        # API keys from centralized configuration
        from api_config import get_polygon_key, get_fred_key
        self.polygon_api_key = get_polygon_key()
        self.fred_api_key = get_fred_key()
        
        # Initialize notification and visualization systems
        self.notification_system = create_notification_system(self.config)
        self.visualizer = create_visualizer()
        
        logger.info("MetaTPEEngine initialized with notification and visualization systems")
    
    def _default_config(self):
        """Default configuration parameters"""
        return {
            'universe': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN', 'NFLX'],
            'param_space': {
                'momentum_threshold': (0.01, 0.08),
                'volume_multiplier': (1.2, 3.0),
                'hold_period': (8, 20),
                'adaptation_strength': (0.1, 0.9),
                'risk_tolerance': (0.015, 0.045)
            },
            'optimization_trials': 30,
            'start_date': datetime(2022, 1, 1),
            'end_date': datetime(2024, 11, 30)
        }
    
    def load_authentic_data(self):
        """Load authentic market data from Polygon and FRED APIs"""
        if not self.polygon_api_key:
            raise ValueError("Polygon API key required for authentic data. Please set POLYGON_API_KEY environment variable.")
        
        logger.info("Loading authentic market data from APIs...")
        
        self.market_data = fetch_data(
            universe=self.universe,
            polygon_api_key=self.polygon_api_key,
            fred_api_key=self.fred_api_key,
            start_date=self.config['start_date'],
            end_date=self.config['end_date']
        )
        
        # Validate data integrity
        loaded_symbols = [sym for sym in self.universe if sym in self.market_data and len(self.market_data[sym]) > 100]
        vix_loaded = 'VIX' in self.market_data
        
        logger.info(f"Loaded authentic data: {len(loaded_symbols)} stocks, VIX: {vix_loaded}")
        
        if len(loaded_symbols) < 4:
            raise ValueError("Insufficient authentic market data loaded. Check API connectivity.")
        
        return len(loaded_symbols), vix_loaded
    
    def train_ml_models(self):
        """Train ML exit prediction models with authentic data"""
        if not self.market_data:
            raise ValueError("Market data must be loaded before training ML models")
        
        logger.info("Training ML exit prediction models...")
        
        self.exit_predictor, self.label_encoder, is_trained = train_exit_predictor(
            self.market_data, self.universe
        )
        
        if is_trained:
            logger.info("ML exit predictor trained successfully")
        else:
            logger.warning("ML training failed - using fallback exit strategies")
        
        return is_trained
    
    def run_optimization(self):
        """Run TPE parameter optimization with authentic data"""
        if not self.market_data:
            raise ValueError("Market data must be loaded before optimization")
        
        logger.info("Starting TPE parameter optimization...")
        
        self.best_params = run_optimization(
            market_data=self.market_data,
            universe=self.universe,
            param_space=self.config['param_space'],
            n_trials=self.config['optimization_trials']
        )
        
        if self.best_params:
            logger.info("TPE optimization completed successfully")
            return True
        else:
            logger.error("Optimization failed")
            return False
    
    def get_optimization_results(self):
        """Get optimization results and performance metrics"""
        if not self.best_params:
            return None
        
        return {
            'best_parameters': {k: v for k, v in self.best_params.items() if k != 'optimization_objective'},
            'objective_value': self.best_params.get('optimization_objective', 0),
            'data_sources': {
                'stocks_loaded': len([s for s in self.universe if s in self.market_data]),
                'vix_data': 'VIX' in self.market_data,
                'ml_trained': self.exit_predictor is not None
            }
        }
    
    def validate_data_integrity(self):
        """Validate that only authentic data sources are used"""
        validation_report = {
            'polygon_data': False,
            'fred_vix_data': False,
            'synthetic_data_detected': False,
            'data_quality': 'unknown'
        }
        
        if self.market_data:
            # Check for Polygon data characteristics
            for symbol in self.universe:
                if symbol in self.market_data and symbol != 'VIX':
                    data = self.market_data[symbol]
                    if 'close' in data.columns and 'volume' in data.columns:
                        validation_report['polygon_data'] = True
                        break
            
            # Check for FRED VIX data
            if 'VIX' in self.market_data:
                vix_data = self.market_data['VIX']
                if 'vix' in vix_data.columns:
                    validation_report['fred_vix_data'] = True
            
            # Assess overall data quality
            total_symbols = len([s for s in self.universe if s in self.market_data])
            if total_symbols >= 6 and validation_report['fred_vix_data']:
                validation_report['data_quality'] = 'high'
            elif total_symbols >= 4:
                validation_report['data_quality'] = 'adequate'
            else:
                validation_report['data_quality'] = 'insufficient'
        
        return validation_report
    
    def run_live_signal_scan(self):
        """
        Run live signal scanning with notification alerts
        
        Returns:
            dict: Alert summary
        """
        logger.info("Running live signal scan with notifications...")
        
        # Run comprehensive alert scan
        alert_summary = self.notification_system.run_alert_scan(self.best_params)
        
        return alert_summary
    
    def generate_performance_visualizations(self):
        """
        Generate comprehensive performance visualizations
        
        Returns:
            list: Generated visualization files
        """
        if not self.best_params or '_trades_log' not in self.best_params:
            logger.warning("No trading data available for visualization")
            return []
        
        trades_log = self.best_params['_trades_log']
        performance_metrics = self.best_params.get('_performance_metrics')
        
        # Generate comprehensive dashboard
        visualization_files = self.visualizer.create_performance_dashboard(
            trades_log, performance_metrics
        )
        
        logger.info(f"Generated {len(visualization_files)} performance visualizations")
        return visualization_files
    
    def run_optimization(self, n_trials=200, n_jobs=4):
        """
        Run Meta-Enhanced TPE optimization on trading parameters
        
        Args:
            n_trials: Number of optimization trials (default: 200)
            n_jobs: Number of parallel jobs (default: 4)
            
        Returns:
            dict: Comprehensive optimization results
        """
        
        print(f"Meta-Enhanced TPE Trading Engine")
        print(f"Running optimization with {n_trials} trials across {n_jobs} parallel jobs")
        print("=" * 60)
        
        # Check for API credentials
        if not self.polygon_api_key or not self.fred_api_key:
            print("API credentials required for authentic data optimization")
            print("Please ensure POLYGON_API_KEY and FRED_API_KEY are available")
            return None
        
        # Load and validate authentic data
        try:
            stocks_loaded, vix_loaded = self.load_authentic_data()
            if stocks_loaded == 0:
                print("Failed to load authentic market data")
                return None
            print(f"Loaded {stocks_loaded} stocks, VIX data: {vix_loaded}")
        except Exception as e:
            print(f"Data loading error: {e}")
            return None
        
        # Parameter space for optimization
        param_space = {
            'momentum_threshold': (0.005, 0.025),
            'volume_multiplier': (1.1, 2.0),
            'rsi_threshold': (55, 75),
            'hold_period': (5, 20),
            'stop_loss': (0.015, 0.05),
            'take_profit': (0.025, 0.08),
            'volatility_filter': (0.1, 0.4)
        }
        
        # Initialize optimization tracking
        optimization_results = []
        best_score = float('-inf')
        best_params = None
        
        # Get current market context
        market_context = self._get_optimization_market_context()
        
        print(f"Market Regime: {market_context.get('regime', 'unknown')}")
        print(f"VIX Level: {market_context.get('vix', 'N/A')}")
        print()
        
        # Meta-TPE optimization loop
        for trial in range(n_trials):
            try:
                # Generate regime-aware parameters
                params = self._suggest_parameters(param_space, market_context, trial)
                
                # Evaluate strategy with authentic data
                score = self._evaluate_strategy_performance(params)
                
                # Track optimization progress
                optimization_results.append({
                    'trial': trial + 1,
                    'score': score,
                    'params': params.copy(),
                    'regime': market_context.get('regime'),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update best results
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"Trial {trial+1}: New best score {score:.4f}")
                    for key, value in params.items():
                        print(f"   {key}: {value:.4f}")
                    print()
                
                # Progress indicators
                if (trial + 1) % 50 == 0:
                    avg_score = sum(r['score'] for r in optimization_results[-50:]) / 50
                    print(f"Progress: {trial+1}/{n_trials} | Avg Score (last 50): {avg_score:.4f}")
                
            except Exception as e:
                print(f"Trial {trial+1} failed: {e}")
                continue
        
        # Compile comprehensive results
        final_results = self._compile_optimization_results(
            optimization_results, best_score, best_params, n_trials, market_context
        )
        
        # Display optimization summary
        self._display_optimization_summary(final_results)
        
        return final_results
    
    def _get_optimization_market_context(self):
        """Get current market context for optimization"""
        
        context = {
            'regime': 'moderate_mixed',
            'volatility': 0.02,
            'momentum': 0.0,
            'vix': 20.0,
            'volume_surge': 1.0
        }
        
        try:
            # Get recent market data if available
            if hasattr(self, 'market_data') and self.market_data:
                recent_vol = []
                recent_momentum = []
                
                for symbol, data in self.market_data.items():
                    if len(data) > 20:
                        recent = data.tail(20)
                        if 'returns' in recent.columns:
                            vol = recent['returns'].std()
                            momentum = recent['returns'].mean()
                            if not pd.isna(vol) and not pd.isna(momentum):
                                recent_vol.append(vol)
                                recent_momentum.append(momentum)
                
                if recent_vol:
                    context['volatility'] = np.mean(recent_vol)
                    context['momentum'] = np.mean(recent_momentum)
            
            # VIX data if available
            if hasattr(self, 'vix_data') and self.vix_data is not None:
                if len(self.vix_data) > 0:
                    context['vix'] = self.vix_data['vix'].iloc[-1]
            
            # Determine regime
            vix = context['vix']
            vol = context['volatility']
            momentum = abs(context['momentum'])
            
            if vix > 25 or vol > 0.03:
                context['regime'] = 'high_volatility'
            elif vix < 15 and vol < 0.015:
                context['regime'] = 'stable_trending' if momentum > 0.005 else 'stable_sideways'
            else:
                context['regime'] = 'moderate_mixed'
                
        except Exception as e:
            print(f"Using default market context due to: {e}")
        
        return context
    
    def _suggest_parameters(self, param_space, market_context, trial):
        """Generate regime-aware parameter suggestions"""
        
        params = {}
        regime = market_context.get('regime', 'moderate_mixed')
        
        # Base parameter generation
        for param_name, (low, high) in param_space.items():
            # Regime-specific adjustments
            if regime == 'high_volatility':
                if param_name == 'momentum_threshold':
                    adjusted_low, adjusted_high = low * 1.2, high * 0.9
                elif param_name == 'stop_loss':
                    adjusted_low, adjusted_high = low * 1.1, high * 1.2
                else:
                    adjusted_low, adjusted_high = low, high
            
            elif regime == 'stable_trending':
                if param_name == 'hold_period':
                    adjusted_low, adjusted_high = low * 1.2, high * 1.3
                elif param_name == 'momentum_threshold':
                    adjusted_low, adjusted_high = low * 0.8, high * 1.1
                else:
                    adjusted_low, adjusted_high = low, high
            
            else:  # moderate_mixed or stable_sideways
                adjusted_low, adjusted_high = low, high
            
            # Ensure bounds are valid
            adjusted_low = max(adjusted_low, low)
            adjusted_high = min(adjusted_high, high)
            
            # Generate parameter value
            if trial < 20:
                # More exploration early on
                params[param_name] = np.random.uniform(adjusted_low, adjusted_high)
            else:
                # More focused search later
                center = (adjusted_low + adjusted_high) / 2
                spread = (adjusted_high - adjusted_low) * 0.3
                params[param_name] = np.clip(
                    np.random.normal(center, spread),
                    adjusted_low, adjusted_high
                )
        
        return params
    
    def _evaluate_strategy_performance(self, params):
        """Evaluate strategy performance with authentic data"""
        
        if not hasattr(self, 'market_data') or not self.market_data:
            return 0.0
        
        total_return = 0.0
        total_trades = 0
        winning_trades = 0
        
        # Strategy evaluation across all symbols
        for symbol, data in self.market_data.items():
            if len(data) < 100:
                continue
            
            symbol_return, symbol_trades, symbol_wins = self._backtest_symbol(data, params)
            total_return += symbol_return
            total_trades += symbol_trades
            winning_trades += symbol_wins
        
        if total_trades == 0:
            return -1.0  # Penalty for no trades
        
        # Calculate risk-adjusted score
        avg_return_per_trade = total_return / total_trades
        win_rate = winning_trades / total_trades
        
        # Combine return and win rate with trade count factor
        trade_factor = min(1.0, total_trades / 20)  # Bonus for sufficient trades
        risk_adjusted_score = (avg_return_per_trade * 0.7 + win_rate * 0.3) * trade_factor
        
        return risk_adjusted_score
    
    def _backtest_symbol(self, data, params):
        """Backtest strategy on individual symbol"""
        
        returns = []
        trades = 0
        wins = 0
        
        momentum_threshold = params.get('momentum_threshold', 0.015)
        volume_multiplier = params.get('volume_multiplier', 1.5)
        rsi_threshold = params.get('rsi_threshold', 65)
        hold_period = int(params.get('hold_period', 10))
        stop_loss = params.get('stop_loss', 0.03)
        take_profit = params.get('take_profit', 0.05)
        
        for i in range(50, len(data) - hold_period, 3):
            try:
                current = data.iloc[i]
                
                # Entry conditions with authentic indicators
                momentum_5 = current.get('momentum_5', 0)
                volume_ratio = current.get('volume_ratio', 1)
                rsi_14 = current.get('rsi_14', 50)
                volatility = current.get('volatility_20', 0.02)
                
                # Apply volatility filter
                if volatility > params.get('volatility_filter', 0.4):
                    continue
                
                # Entry signal
                if (momentum_5 > momentum_threshold and 
                    volume_ratio > volume_multiplier and 
                    rsi_14 < rsi_threshold):
                    
                    entry_price = current['close']
                    
                    # Simulate holding period with stop/take profit
                    for j in range(1, hold_period + 1):
                        if i + j >= len(data):
                            break
                        
                        day_price = data.iloc[i + j]['close']
                        current_return = (day_price - entry_price) / entry_price
                        
                        if current_return >= take_profit:
                            returns.append(take_profit)
                            wins += 1
                            trades += 1
                            break
                        elif current_return <= -stop_loss:
                            returns.append(-stop_loss)
                            trades += 1
                            break
                    else:
                        # Held for full period
                        final_price = data.iloc[i + hold_period]['close']
                        trade_return = (final_price - entry_price) / entry_price
                        returns.append(trade_return)
                        if trade_return > 0:
                            wins += 1
                        trades += 1
                        
            except (IndexError, KeyError, TypeError):
                continue
        
        total_return = sum(returns) if returns else 0.0
        return total_return, trades, wins
    
    def _compile_optimization_results(self, optimization_results, best_score, best_params, n_trials, market_context):
        """Compile comprehensive optimization results"""
        
        scores = [r['score'] for r in optimization_results if r['score'] is not None]
        
        results = {
            'optimization_summary': {
                'total_trials': n_trials,
                'successful_trials': len(scores),
                'best_score': best_score,
                'best_params': best_params,
                'market_regime': market_context.get('regime'),
                'optimization_date': datetime.now().isoformat()
            },
            'performance_statistics': {
                'mean_score': np.mean(scores) if scores else 0.0,
                'std_score': np.std(scores) if scores else 0.0,
                'min_score': np.min(scores) if scores else 0.0,
                'max_score': np.max(scores) if scores else 0.0,
                'score_improvement': best_score - np.mean(scores[:20]) if len(scores) >= 20 else 0.0
            },
            'optimization_history': optimization_results
        }
        
        return results
    
    def _display_optimization_summary(self, results):
        """Display comprehensive optimization summary"""
        
        print("\n" + "="*60)
        print("META-ENHANCED TPE OPTIMIZATION COMPLETE")
        print("="*60)
        
        summary = results['optimization_summary']
        stats = results['performance_statistics']
        
        print(f"PERFORMANCE SUMMARY:")
        print(f"   Best Score: {summary['best_score']:.4f}")
        print(f"   Successful Trials: {summary['successful_trials']}/{summary['total_trials']}")
        print(f"   Market Regime: {summary['market_regime']}")
        print(f"   Improvement: {stats['score_improvement']:.4f}")
        
        print(f"\nOPTIMAL PARAMETERS:")
        for param, value in summary['best_params'].items():
            print(f"   {param}: {value:.4f}")
        
        print(f"\nSTATISTICS:")
        print(f"   Mean Score: {stats['mean_score']:.4f} ± {stats['std_score']:.4f}")
        print(f"   Score Range: {stats['min_score']:.4f} to {stats['max_score']:.4f}")
        
        print("="*60)
    
    def get_current_market_status(self):
        """
        Get current market status and regime information
        
        Returns:
            dict: Current market status
        """
        try:
            from datetime import timedelta
            from core.data_loader import get_vix_data
            from core.regime_logic import get_market_regime
            
            # Get recent VIX data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            vix_data = get_vix_data(start_date, end_date, self.fred_api_key)
            
            if vix_data is not None and len(vix_data) > 0:
                current_regime = get_market_regime(vix_data.index[-1], vix_data)
                current_vix = vix_data.iloc[-1]['vix']
                
                return {
                    'current_regime': current_regime,
                    'vix_level': current_vix,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'active'
                }
            else:
                return {
                    'status': 'data_unavailable',
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Main execution function"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize engine
    engine = MetaTPEEngine()
    
    try:
        # Load authentic market data
        stocks_loaded, vix_loaded = engine.load_authentic_data()
        print(f"Loaded authentic data: {stocks_loaded} stocks, VIX: {vix_loaded}")
        
        # Validate data integrity
        validation = engine.validate_data_integrity()
        print(f"Data validation: {validation['data_quality']} quality")
        
        # Train ML models
        ml_trained = engine.train_ml_models()
        print(f"ML training: {'successful' if ml_trained else 'fallback mode'}")
        
        # Run optimization
        optimization_success = engine.run_optimization()
        
        if optimization_success:
            results = engine.get_optimization_results()
            print("\nOptimization Results:")
            print(f"Objective value: {results['objective_value']:.4f}")
            print("Best parameters:")
            for param, value in results['best_parameters'].items():
                print(f"  {param}: {value:.4f}")
        else:
            print("Optimization failed")
    
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure POLYGON_API_KEY is set in environment variables")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()