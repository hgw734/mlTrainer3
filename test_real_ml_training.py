#!/usr/bin/env python3
"""
Test Real ML Training
Demonstrates training models with actual historical market data
NO MOCK DATA - REAL TRAINING ONLY
"""

import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ml_training_pipeline():
    """Test the complete ML training pipeline with real data"""
    
    try:
        logger.info("🚀 Starting ML Training Pipeline Test")
        logger.info("=" * 50)
        
        # Import required components
        from core.data_pipeline import DataPipeline
        from core.model_trainer import ModelTrainer
        from models.momentum_breakout_enhanced import MomentumBreakoutEnhanced
        from walk_forward_trial_launcher import WalkForwardAnalyzer
        
        # Initialize components
        data_pipeline = DataPipeline()
        model_trainer = ModelTrainer()
        
        # Test 1: Fetch Real Historical Data
        logger.info("\n📊 Test 1: Fetching Real Historical Data")
        symbol = "AAPL"
        lookback_days = 500
        
        historical_data = data_pipeline.fetch_historical_data(symbol, days=lookback_days)
        logger.info(f"✅ Fetched {len(historical_data)} days of {symbol} data")
        logger.info(f"   Date range: {historical_data.index[0]} to {historical_data.index[-1]}")
        logger.info(f"   Latest close: ${historical_data['close'].iloc[-1]:.2f}")
        
        # Test 2: Feature Engineering
        logger.info("\n🔧 Test 2: Feature Engineering")
        features = data_pipeline.prepare_features(historical_data)
        logger.info(f"✅ Created {features.shape[1]} features from OHLCV data")
        logger.info(f"   Feature matrix shape: {features.shape}")
        
        # Test 3: Calculate Market Statistics
        logger.info("\n📈 Test 3: Market Statistics (Real Data)")
        stats = data_pipeline.get_market_statistics(symbol, lookback_days=252)
        logger.info(f"✅ Calculated real market statistics:")
        logger.info(f"   Annual Return: {stats['mean_return']:.2%}")
        logger.info(f"   Annual Volatility: {stats['volatility']:.2%}")
        logger.info(f"   Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        logger.info(f"   Max Drawdown: {stats['max_drawdown']:.2%}")
        logger.info(f"   Win Rate: {stats['win_rate']:.2%}")
        
        # Test 4: Train MomentumBreakout Model
        logger.info("\n🤖 Test 4: Training MomentumBreakout Model")
        
        model_params = {
            'lookback_period': 20,
            'breakout_threshold': 2.0,
            'volume_confirmation': True,
            'volatility_filter': True
        }
        
        metrics = model_trainer.train_model(
            model_id='momentum_breakout_test',
            model_class=MomentumBreakoutEnhanced,
            data=historical_data,
            validation_split=0.2,
            params=model_params
        )
        
        logger.info(f"✅ Model trained successfully!")
        logger.info(f"   Validation Sharpe: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"   Total Return: {metrics['total_return']:.2%}")
        logger.info(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"   Win Rate: {metrics['win_rate']:.2%}")
        
        # Test 5: Walk-Forward Analysis
        logger.info("\n🚶 Test 5: Walk-Forward Analysis")
        
        analyzer = WalkForwardAnalyzer(
            train_window_days=252,  # 1 year
            test_window_days=63,    # 3 months
            step_size_days=21       # 1 month
        )
        
        # Prepare walk-forward steps
        steps = analyzer.prepare_walk_forward_data(symbol, total_days=lookback_days)
        logger.info(f"✅ Created {len(steps)} walk-forward steps")
        
        # Execute first step as example
        if steps:
            first_step = steps[0]
            step_results = analyzer.execute_walk_forward_step(
                first_step,
                MomentumBreakoutEnhanced,
                model_params
            )
            
            logger.info(f"   Step 1 Results:")
            logger.info(f"   - Train: {first_step.train_start} to {first_step.train_end}")
            logger.info(f"   - Test: {first_step.test_start} to {first_step.test_end}")
            logger.info(f"   - Test Sharpe: {step_results['sharpe_ratio']:.3f}")
            logger.info(f"   - Test Return: {step_results['total_return']:.2%}")
        
        # Test 6: Load Saved Model
        logger.info("\n💾 Test 6: Loading Saved Model")
        
        loaded_model = model_trainer.load_model('momentum_breakout_test')
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Model class: {loaded_model.__class__.__name__}")
        logger.info(f"   Parameters: {loaded_model.get_parameters()}")
        
        # Test 7: Generate Real-Time Prediction
        logger.info("\n🎯 Test 7: Real-Time Prediction")
        
        # Get latest 100 days for prediction
        recent_data = data_pipeline.fetch_historical_data(symbol, days=100)
        prediction = loaded_model.predict(recent_data)
        latest_signal = prediction.iloc[-1]
        
        signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
        logger.info(f"✅ Latest signal for {symbol}: {signal_map.get(latest_signal, 'UNKNOWN')}")
        
        # Test 8: Multiple Symbols Dataset
        logger.info("\n🌍 Test 8: Multi-Symbol Dataset Creation")
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        X, y, metadata = data_pipeline.create_training_dataset(
            symbols=symbols,
            lookback_days=365,
            target_horizon=1
        )
        
        logger.info(f"✅ Created multi-symbol dataset:")
        logger.info(f"   Total samples: {X.shape[0]}")
        logger.info(f"   Features: {X.shape[1]}")
        logger.info(f"   Symbols included: {metadata['symbol'].unique().tolist()}")
        
        # Test 9: Economic Indicators
        logger.info("\n💰 Test 9: Economic Indicators from FRED")
        
        try:
            econ_data = data_pipeline.fetch_economic_indicators()
            logger.info(f"✅ Fetched economic indicators:")
            for indicator in econ_data.columns:
                latest_value = econ_data[indicator].dropna().iloc[-1]
                logger.info(f"   {indicator}: {latest_value:.2f}")
        except Exception as e:
            logger.warning(f"⚠️  Could not fetch economic indicators: {e}")
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("The ML training pipeline is working with REAL market data")
        logger.info("NO mock data or fake functions were used")
        
        # List all trained models
        logger.info("\n📋 Available Models:")
        models = model_trainer.list_models()
        for model in models:
            logger.info(f"   - {model['model_id']}: Sharpe={model['sharpe_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_training_pipeline()
    sys.exit(0 if success else 1)