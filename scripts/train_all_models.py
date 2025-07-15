#!/usr/bin/env python3
"""
Comprehensive Model Training Script for mlTrainer
Trains all implemented models using real market data
NO MOCK DATA
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# Import data pipeline
from core.data_pipeline import DataPipeline

# Import models
from models.momentum_breakout_enhanced import MomentumBreakoutEnhanced
from models.mean_reversion_enhanced import MeanReversionEnhanced
from models.volatility_regime_enhanced import VolatilityRegimeEnhanced
from models.random_forest_enhanced import RandomForestEnhanced
from models.xgboost_enhanced import XGBoostEnhanced
from models.lstm_enhanced import LSTMEnhanced
from models.pairs_trading_enhanced import PairsTradingEnhanced
from models.market_regime_detector import MarketRegimeDetector
from models.portfolio_optimizer import PortfolioOptimizer

# Import new model suites
from models.technical_indicators_enhanced import (
    TechnicalIndicatorEnsemble, RSIModel, MACDModel, 
    BollingerBreakoutModel, StochasticModel
)
from models.volume_analysis_enhanced import (
    VolumeAnalysisEnsemble, OBVModel, VolumeSpikeModel,
    VolumePriceAnalysisModel, VolumeWeightedPriceModel
)
from models.pattern_recognition_enhanced import (
    PatternRecognitionEnsemble, CandlestickPatternsModel,
    SupportResistanceModel, ChartPatternRecognitionModel
)

# Import backtesting
from core.backtesting_engine import BacktestingEngine, BacktestConfig

# Import training service
from services.model_trainer import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models/trained',
        'results/backtests',
        'results/metrics',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def train_rule_based_models(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train rule-based trading models"""
    logger.info("Training rule-based models...")
    results = {}
    
    # Momentum Breakout
    logger.info("Training Momentum Breakout model...")
    momentum_model = MomentumBreakoutEnhanced(
        lookback_period=config.get('momentum_lookback', 20),
        volume_multiplier=config.get('volume_multiplier', 1.5)
    )
    momentum_signals = momentum_model.predict(data)
    results['momentum_breakout'] = {
        'model': momentum_model,
        'signals': momentum_signals,
        'params': momentum_model.get_parameters()
    }
    
    # Mean Reversion
    logger.info("Training Mean Reversion model...")
    mean_rev_model = MeanReversionEnhanced(
        bb_period=config.get('bb_period', 20),
        bb_std=config.get('bb_std', 2.0)
    )
    mean_rev_signals = mean_rev_model.predict(data)
    results['mean_reversion'] = {
        'model': mean_rev_model,
        'signals': mean_rev_signals,
        'params': mean_rev_model.get_parameters()
    }
    
    # Volatility Regime
    logger.info("Training Volatility Regime model...")
    vol_model = VolatilityRegimeEnhanced(
        vol_lookback=config.get('vol_lookback', 20),
        regime_threshold=config.get('regime_threshold', 0.02)
    )
    vol_signals = vol_model.predict(data)
    results['volatility_regime'] = {
        'model': vol_model,
        'signals': vol_signals,
        'params': vol_model.get_parameters()
    }
    
    return results

def train_technical_indicator_models(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train technical indicator models"""
    logger.info("Training technical indicator models...")
    results = {}
    
    # Technical Indicator Ensemble
    logger.info("Training Technical Indicator Ensemble...")
    tech_ensemble = TechnicalIndicatorEnsemble()
    tech_signals = tech_ensemble.predict(data)
    results['technical_indicators_ensemble'] = {
        'model': tech_ensemble,
        'signals': tech_signals,
        'params': tech_ensemble.get_parameters()
    }
    
    # Individual indicator models (for comparison)
    if config.get('train_individual_indicators', False):
        logger.info("Training individual technical indicators...")
        
        # RSI
        rsi_model = RSIModel()
        results['rsi'] = {
            'model': rsi_model,
            'signals': rsi_model.predict(data),
            'params': rsi_model.get_parameters()
        }
        
        # MACD
        macd_model = MACDModel()
        results['macd'] = {
            'model': macd_model,
            'signals': macd_model.predict(data),
            'params': macd_model.get_parameters()
        }
    
    return results

def train_volume_models(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train volume analysis models"""
    logger.info("Training volume analysis models...")
    results = {}
    
    # Volume Analysis Ensemble
    logger.info("Training Volume Analysis Ensemble...")
    volume_ensemble = VolumeAnalysisEnsemble()
    volume_signals = volume_ensemble.predict(data)
    results['volume_analysis_ensemble'] = {
        'model': volume_ensemble,
        'signals': volume_signals,
        'params': volume_ensemble.get_parameters()
    }
    
    # Individual volume models (for comparison)
    if config.get('train_individual_volume', False):
        logger.info("Training individual volume models...")
        
        # OBV
        obv_model = OBVModel()
        results['obv'] = {
            'model': obv_model,
            'signals': obv_model.predict(data),
            'params': obv_model.get_parameters()
        }
        
        # VWAP
        vwap_model = VolumeWeightedPriceModel()
        results['vwap'] = {
            'model': vwap_model,
            'signals': vwap_model.predict(data),
            'params': vwap_model.get_parameters()
        }
    
    return results

def train_pattern_models(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train pattern recognition models"""
    logger.info("Training pattern recognition models...")
    results = {}
    
    # Pattern Recognition Ensemble
    logger.info("Training Pattern Recognition Ensemble...")
    pattern_ensemble = PatternRecognitionEnsemble()
    pattern_signals = pattern_ensemble.predict(data)
    results['pattern_recognition_ensemble'] = {
        'model': pattern_ensemble,
        'signals': pattern_signals,
        'params': pattern_ensemble.get_parameters()
    }
    
    # Individual pattern models (for comparison)
    if config.get('train_individual_patterns', False):
        logger.info("Training individual pattern models...")
        
        # Candlestick
        candle_model = CandlestickPatternsModel()
        results['candlestick'] = {
            'model': candle_model,
            'signals': candle_model.predict(data),
            'params': candle_model.get_parameters()
        }
        
        # Support/Resistance
        sr_model = SupportResistanceModel()
        results['support_resistance'] = {
            'model': sr_model,
            'signals': sr_model.predict(data),
            'params': sr_model.get_parameters()
        }
    
    return results

def train_advanced_models(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train advanced models (market regime, portfolio optimization)"""
    logger.info("Training advanced models...")
    results = {}
    
    # Market Regime Detector
    logger.info("Training Market Regime Detector...")
    try:
        regime_model = MarketRegimeDetector()
        regime_model.fit(data)
        regime_signals = regime_model.predict(data)
        results['market_regime'] = {
            'model': regime_model,
            'signals': regime_signals,
            'params': regime_model.get_parameters(),
            'regimes': regime_model.get_regimes(data).to_dict()
        }
    except Exception as e:
        logger.error(f"Market regime detector failed: {e}")
    
    # Note: Portfolio Optimizer requires multiple assets, so skipping for single asset
    
    return results

def train_ml_models(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train machine learning models"""
    logger.info("Training ML models...")
    results = {}
    
    # Random Forest
    logger.info("Training Random Forest model...")
    rf_model = RandomForestEnhanced(
        n_estimators=config.get('rf_n_estimators', 100),
        max_depth=config.get('rf_max_depth', 10),
        prediction_type='classification'
    )
    rf_model.fit(data)
    rf_signals = rf_model.predict(data)
    
    # Cross-validation
    cv_results = rf_model.cross_validate(data, n_splits=5)
    
    results['random_forest'] = {
        'model': rf_model,
        'signals': rf_signals,
        'params': rf_model.get_parameters(),
        'cv_results': cv_results,
        'feature_importance': rf_model.feature_importance_.head(10).to_dict('records')
    }
    
    # XGBoost
    if config.get('train_xgboost', True):
        logger.info("Training XGBoost model...")
        try:
            xgb_model = XGBoostEnhanced(
                n_estimators=config.get('xgb_n_estimators', 100),
                max_depth=config.get('xgb_max_depth', 6),
                learning_rate=config.get('xgb_learning_rate', 0.1)
            )
            xgb_model.fit(data)
            xgb_signals = xgb_model.predict(data)
            
            results['xgboost'] = {
                'model': xgb_model,
                'signals': xgb_signals,
                'params': xgb_model.get_parameters(),
                'feature_importance': xgb_model.get_feature_importance().head(10).to_dict('records')
            }
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
    
    # LSTM (if TensorFlow available)
    if config.get('train_lstm', False):
        logger.info("Training LSTM model...")
        try:
            lstm_model = LSTMEnhanced(
                sequence_length=config.get('lstm_sequence_length', 60),
                lstm_units=config.get('lstm_units', [128, 64, 32]),
                epochs=config.get('lstm_epochs', 50)
            )
            lstm_model.fit(data)
            lstm_signals = lstm_model.predict(data)
            
            results['lstm'] = {
                'model': lstm_model,
                'signals': lstm_signals,
                'params': lstm_model.get_parameters()
            }
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
    
    return results

def run_backtests(data: pd.DataFrame, model_results: Dict[str, Any], 
                 config: Dict[str, Any]) -> Dict[str, Any]:
    """Run backtests for all models"""
    logger.info("Running backtests...")
    
    backtest_config = BacktestConfig(
        initial_capital=config.get('initial_capital', 100000),
        commission=config.get('commission', 0.001),
        slippage=config.get('slippage', 0.0005),
        stop_loss=config.get('stop_loss', 0.02),
        take_profit=config.get('take_profit', 0.05)
    )
    
    engine = BacktestingEngine(backtest_config)
    backtest_results = {}
    
    # Run backtest for each model
    for model_name, model_data in model_results.items():
        logger.info(f"Backtesting {model_name}...")
        
        try:
            signals = model_data['signals']
            results = engine.run(data, signals)
            
            backtest_results[model_name] = {
                'metrics': results['metrics'],
                'equity_curve': results['equity_curve'].to_dict(),
                'trades': results['trades'].to_dict('records') if not results['trades'].empty else [],
                'plot_data': engine.plot_results(results)
            }
            
            # Log key metrics
            metrics = results['metrics']
            logger.info(f"{model_name} - Total Return: {metrics['total_return']*100:.2f}%, "
                       f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                       f"Max DD: {metrics['max_drawdown']*100:.2f}%, "
                       f"Win Rate: {metrics['win_rate']*100:.2f}%")
            
        except Exception as e:
            logger.error(f"Backtest failed for {model_name}: {e}")
            backtest_results[model_name] = {'error': str(e)}
    
    return backtest_results

def create_ensemble_signals(model_results: Dict[str, Any], weights: Dict[str, float] = None) -> pd.Series:
    """Create ensemble signals from multiple models"""
    logger.info("Creating ensemble signals...")
    
    # Default equal weights
    if weights is None:
        weights = {model: 1.0 / len(model_results) for model in model_results}
    
    # Collect all signals
    all_signals = {}
    for model_name, model_data in model_results.items():
        if 'signals' in model_data:
            all_signals[model_name] = model_data['signals']
    
    if not all_signals:
        raise ValueError("No signals found in model results")
    
    # Find common index
    common_index = None
    for signals in all_signals.values():
        if common_index is None:
            common_index = signals.index
        else:
            common_index = common_index.intersection(signals.index)
    
    # Create weighted ensemble
    ensemble = pd.Series(0.0, index=common_index)
    
    for model_name, signals in all_signals.items():
        weight = weights.get(model_name, 0)
        ensemble += weight * signals.reindex(common_index).fillna(0)
    
    # Convert to signals
    ensemble_signals = pd.Series(0, index=common_index)
    ensemble_signals[ensemble > 0.2] = 1
    ensemble_signals[ensemble < -0.2] = -1
    
    return ensemble_signals

def save_results(results: Dict[str, Any], output_dir: str = 'results'):
    """Save training results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save backtest results
    backtest_file = os.path.join(output_dir, 'backtests', f'backtest_results_{timestamp}.json')
    with open(backtest_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        json.dump(results, f, indent=2, default=convert_types)
    
    logger.info(f"Results saved to {backtest_file}")
    
    # Save summary metrics
    summary = []
    for model_name, backtest in results.get('backtests', {}).items():
        if 'metrics' in backtest:
            metrics = backtest['metrics']
            summary.append({
                'model': model_name,
                'total_return': metrics.get('total_return', 0),
                'annual_return': metrics.get('annual_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'total_trades': metrics.get('total_trades', 0)
            })
    
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(output_dir, 'metrics', f'summary_{timestamp}.csv')
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary saved to {summary_file}")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train all mlTrainer models')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to train on')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date for data')
    parser.add_argument('--end-date', type=str, help='End date for data (default: today)')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--skip-ml', action='store_true', help='Skip ML models')
    parser.add_argument('--skip-backtest', action='store_true', help='Skip backtesting')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Load config
    config = {
        'symbol': args.symbol,
        'start_date': args.start_date,
        'end_date': args.end_date or datetime.now().strftime('%Y-%m-%d'),
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0005,
        'train_lstm': False,  # Disabled by default (requires TensorFlow)
        'train_xgboost': True
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    logger.info(f"Training configuration: {config}")
    
    try:
        # Initialize data pipeline
        logger.info("Initializing data pipeline...")
        pipeline = DataPipeline()
        
        # Fetch data
        logger.info(f"Fetching data for {config['symbol']}...")
        data = pipeline.fetch_stock_data(
            config['symbol'],
            start_date=config['start_date'],
            end_date=config['end_date']
        )
        
        if data is None or data.empty:
            raise ValueError(f"No data fetched for {config['symbol']}")
        
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Add features
        logger.info("Adding technical features...")
        data = pipeline.add_all_features(data)
        
        # Train models
        all_results = {}
        
        # Rule-based models
        rule_results = train_rule_based_models(data, config)
        all_results.update(rule_results)
        
        # Technical indicator models
        tech_results = train_technical_indicator_models(data, config)
        all_results.update(tech_results)

        # Volume models
        volume_results = train_volume_models(data, config)
        all_results.update(volume_results)

        # Pattern models
        pattern_results = train_pattern_models(data, config)
        all_results.update(pattern_results)

        # Advanced models
        advanced_results = train_advanced_models(data, config)
        all_results.update(advanced_results)

        # ML models
        if not args.skip_ml:
            ml_results = train_ml_models(data, config)
            all_results.update(ml_results)
        
        # Create ensemble
        try:
            ensemble_signals = create_ensemble_signals(all_results)
            all_results['ensemble'] = {
                'signals': ensemble_signals,
                'params': {'type': 'equal_weight_ensemble', 'n_models': len(all_results)}
            }
        except Exception as e:
            logger.error(f"Failed to create ensemble: {e}")
        
        # Run backtests
        if not args.skip_backtest:
            backtest_results = run_backtests(data, all_results, config)
        else:
            backtest_results = {}
        
        # Prepare final results
        final_results = {
            'config': config,
            'data_info': {
                'symbol': config['symbol'],
                'start_date': str(data.index[0]),
                'end_date': str(data.index[-1]),
                'n_samples': len(data),
                'n_features': data.shape[1]
            },
            'models': {name: res.get('params', {}) for name, res in all_results.items()},
            'backtests': backtest_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        save_results(final_results)
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Symbol: {config['symbol']}")
        print(f"Models trained: {len(all_results)}")
        
        if backtest_results:
            print("\nBacktest Summary:")
            for model_name, results in backtest_results.items():
                if 'metrics' in results:
                    metrics = results['metrics']
                    print(f"\n{model_name}:")
                    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
                    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
                    print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()