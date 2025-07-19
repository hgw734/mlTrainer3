"""
Scheduled ML Signal Generation System
LSTM: Every 15 minutes | Transformer: Every hour
"""

import time
import json
import pandas as pd
from datetime import datetime, timedelta
import requests
from threading import Thread
import schedule
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_config import get_polygon_key
from ml_engine.lstm_transformer_models import MLSignalGenerator
from ml_engine.market_regime_classifier import classify_regime
from utils.indicators import compute_indicators

class ScheduledMLRunner:
    """Manages scheduled execution of LSTM and Transformer models"""
    
    def __init__(self):
        self.polygon_api_key = get_polygon_key()
        self.ml_generator = MLSignalGenerator()
        self.current_signals = {}
        self.signal_history = []
        self.is_trained = False
        
        # Load optimized parameters from TPE results
        self.optimized_params = self.load_optimized_parameters()
        
        # Load full Elite 500 universe for live trading
        try:
            with open('elites_500_universe.json', 'r') as f:
                universe_data = json.load(f)
                self.live_universe = universe_data.get('elites_500_universe', [])
                print(f"ML Scheduler: Loaded Elite 500 universe with {len(self.live_universe)} symbols")
        except:
            # Fallback to core symbols if file not found
            self.live_universe = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JNJ"
            ]
            print(f"ML Scheduler: Using fallback universe with {len(self.live_universe)} symbols")
        
        self.data_cache = {}
        self.last_lstm_run = None
        self.last_transformer_run = None
    
    def load_optimized_parameters(self):
        """Load optimized parameters from TPE optimization results"""
        try:
            with open('robust_optimization_results.json', 'r') as f:
                results = json.load(f)
                params = results['optimization_summary']['best_params']
                print(f"ML Scheduler: Loaded optimized parameters - Sharpe: {results['optimization_summary']['best_sharpe']:.4f}")
                return params
        except Exception as e:
            print(f"Warning: Could not load optimized parameters: {e}")
            # Fallback to default parameters
            return {
                'momentum_period': 9,
                'volume_threshold': 2.12,
                'stop_loss': 0.0583,
                'take_profit': 0.2245,
                'momentum_threshold': 0.0529
            }
    
    def fetch_live_data(self, symbol, days=100):
        """Fetch recent data for live analysis"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': days,
                'apikey': self.polygon_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('results') and len(data['results']) > 50:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.rename(columns={'c': 'close', 'v': 'volume'})
                    df = df.set_index('date').sort_index()
                    
                    return df['close']
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Live data fetch error for {symbol}: {e}")
            return None
    
    def initialize_ml_models(self):
        """Initialize and train ML models with recent data"""
        print("Initializing ML models with authentic market data...")
        
        training_data = {}
        for symbol in self.live_universe:
            # Fetch 6 months of data for training
            prices = self.fetch_live_data(symbol, days=180)
            if prices is not None and len(prices) > 100:
                training_data[symbol] = prices
                print(f"Training data loaded for {symbol}: {len(prices)} periods")
        
        if not training_data:
            print("ERROR: No training data available")
            return False
        
        # Train models individually on each symbol's data
        training_success = False
        
        params = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'bollinger_window': 20,
            'bollinger_std': 2.0,
            'momentum_threshold': 0.02
        }
        
        # Use the largest dataset for training
        largest_symbol = max(training_data.keys(), key=lambda k: len(training_data[k]))
        training_prices = training_data[largest_symbol]
        
        print(f"Training ML models on {largest_symbol} data: {len(training_prices)} periods")
        
        indicators = compute_indicators(training_prices, params)
        
        # Train ML models
        success = self.ml_generator.train_models(training_prices, indicators)
        if success:
            self.is_trained = True
            print("ML models successfully trained on authentic data")
            return True
        else:
            print("ML model training failed")
            return False
    
    def run_lstm_cycle(self):
        """Execute LSTM analysis - runs every 15 minutes"""
        if not self.is_trained:
            print("LSTM: Models not trained, skipping cycle")
            return
        
        timestamp = datetime.now()
        print(f"LSTM Cycle - {timestamp.strftime('%H:%M:%S')}")
        
        lstm_signals = {}
        
        for symbol in self.live_universe:
            try:
                # Get recent data
                prices = self.fetch_live_data(symbol, days=60)
                if prices is None or len(prices) < 30:
                    continue
                
                # Generate indicators using optimized parameters
                params = {
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'bollinger_window': 20,
                    'bollinger_std': 2.0,
                    'momentum_threshold': self.optimized_params['momentum_threshold'],
                    'momentum_period': self.optimized_params['momentum_period'],
                    'volume_threshold': self.optimized_params['volume_threshold']
                }
                
                indicators = compute_indicators(prices, params)
                
                # Get market regime
                regime_code = classify_regime(prices)
                regime_name = {-1: "volatile", 0: "bear", 1: "neutral", 2: "bull"}.get(regime_code, "neutral")
                
                # Generate LSTM signal
                lstm_signal = self.ml_generator.lstm.predict(prices, indicators)
                
                lstm_signals[symbol] = {
                    'signal': lstm_signal,
                    'regime': regime_name,
                    'price': float(prices.iloc[-1]),
                    'rsi': float(indicators['rsi'].iloc[-1]) if len(indicators) > 0 else None,
                    'timestamp': timestamp.isoformat()
                }
                
                print(f"LSTM {symbol}: {lstm_signal} (regime: {regime_name})")
                
            except Exception as e:
                print(f"LSTM error for {symbol}: {e}")
                continue
        
        # Update signals
        self.current_signals['lstm'] = lstm_signals
        self.last_lstm_run = timestamp
        
        # Save to history
        self.signal_history.append({
            'timestamp': timestamp.isoformat(),
            'type': 'lstm',
            'signals': lstm_signals
        })
        
        # Save to file
        self.save_signals()
    
    def run_transformer_cycle(self):
        """Execute Transformer analysis - runs every hour"""
        if not self.is_trained:
            print("Transformer: Models not trained, skipping cycle")
            return
        
        timestamp = datetime.now()
        print(f"Transformer Cycle - {timestamp.strftime('%H:%M:%S')}")
        
        transformer_signals = {}
        
        for symbol in self.live_universe:
            try:
                # Get extended data for pattern analysis
                prices = self.fetch_live_data(symbol, days=90)
                if prices is None or len(prices) < 50:
                    continue
                
                # Generate indicators using optimized parameters
                params = {
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'bollinger_window': 20,
                    'bollinger_std': 2.0,
                    'momentum_threshold': self.optimized_params['momentum_threshold'],
                    'momentum_period': self.optimized_params['momentum_period'],
                    'volume_threshold': self.optimized_params['volume_threshold']
                }
                
                indicators = compute_indicators(prices, params)
                
                # Get market regime
                regime_code = classify_regime(prices)
                regime_name = {-1: "volatile", 0: "bear", 1: "neutral", 2: "bull"}.get(regime_code, "neutral")
                
                # Generate Transformer signal
                transformer_signal = self.ml_generator.transformer.predict(prices, indicators)
                
                transformer_signals[symbol] = {
                    'signal': transformer_signal,
                    'regime': regime_name,
                    'price': float(prices.iloc[-1]),
                    'momentum': float(indicators['momentum'].iloc[-1]) if len(indicators) > 0 else None,
                    'timestamp': timestamp.isoformat()
                }
                
                print(f"Transformer {symbol}: {transformer_signal} (regime: {regime_name})")
                
            except Exception as e:
                print(f"Transformer error for {symbol}: {e}")
                continue
        
        # Update signals
        self.current_signals['transformer'] = transformer_signals
        self.last_transformer_run = timestamp
        
        # Save to history
        self.signal_history.append({
            'timestamp': timestamp.isoformat(),
            'type': 'transformer',
            'signals': transformer_signals
        })
        
        # Save to file
        self.save_signals()
    
    def generate_combined_signals(self):
        """Generate combined LSTM + Transformer signals"""
        if 'lstm' not in self.current_signals or 'transformer' not in self.current_signals:
            return {}
        
        combined_signals = {}
        lstm_signals = self.current_signals['lstm']
        transformer_signals = self.current_signals['transformer']
        
        for symbol in self.live_universe:
            if symbol in lstm_signals and symbol in transformer_signals:
                lstm_signal = lstm_signals[symbol]['signal']
                transformer_signal = transformer_signals[symbol]['signal']
                regime = lstm_signals[symbol]['regime']
                
                # Combined signal logic
                final_signal = self.ml_generator.generate_signal(
                    None, None, regime
                ) if lstm_signal == transformer_signal else "hold"
                
                combined_signals[symbol] = {
                    'final_signal': final_signal,
                    'lstm_signal': lstm_signal,
                    'transformer_signal': transformer_signal,
                    'regime': regime,
                    'agreement': lstm_signal == transformer_signal,
                    'timestamp': datetime.now().isoformat()
                }
        
        self.current_signals['combined'] = combined_signals
        return combined_signals
    
    def save_signals(self):
        """Save current signals to file"""
        signal_data = {
            'current_signals': self.current_signals,
            'last_lstm_run': self.last_lstm_run.isoformat() if self.last_lstm_run else None,
            'last_transformer_run': self.last_transformer_run.isoformat() if self.last_transformer_run else None,
            'signal_count': len(self.signal_history),
            'updated': datetime.now().isoformat()
        }
        
        with open('live_ml_signals.json', 'w') as f:
            json.dump(signal_data, f, indent=2, default=str)
        
        # Save signal history
        if len(self.signal_history) > 100:  # Keep last 100 signals
            self.signal_history = self.signal_history[-100:]
        
        with open('ml_signal_history.json', 'w') as f:
            json.dump(self.signal_history, f, indent=2, default=str)
    
    def start_scheduler(self):
        """Start the scheduled ML execution"""
        print("Starting ML Scheduler:")
        print("- LSTM: Every 15 minutes")
        print("- Transformer: Every hour")
        
        # Initialize models
        if not self.initialize_ml_models():
            print("Failed to initialize ML models")
            return
        
        # Schedule jobs
        schedule.every(15).minutes.do(self.run_lstm_cycle)
        schedule.every().hour.do(self.run_transformer_cycle)
        
        # Run initial cycles
        self.run_lstm_cycle()
        self.run_transformer_cycle()
        
        print("ML Scheduler started successfully")
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def get_status(self):
        """Get current ML system status"""
        return {
            'ml_trained': self.is_trained,
            'last_lstm_run': self.last_lstm_run.isoformat() if self.last_lstm_run else None,
            'last_transformer_run': self.last_transformer_run.isoformat() if self.last_transformer_run else None,
            'active_symbols': len(self.live_universe),
            'signal_history_count': len(self.signal_history),
            'current_time': datetime.now().isoformat()
        }

def main():
    """Start the scheduled ML runner"""
    runner = ScheduledMLRunner()
    
    print("ML Scheduled Runner")
    print("Using authentic Polygon data for live trading signals")
    print("=" * 50)
    
    try:
        runner.start_scheduler()
    except KeyboardInterrupt:
        print("ML Scheduler stopped by user")
    except Exception as e:
        print(f"ML Scheduler error: {e}")

if __name__ == "__main__":
    main()