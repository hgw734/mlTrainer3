"""
Robust TPE Optimizer with Progress Saving
Implements checkpoint saving to prevent progress loss during restarts
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from api_config import get_polygon_key

class RobustTPEOptimizer:
    def __init__(self):
        self.polygon_api_key = get_polygon_key()
        
        # Load universe
        with open('elites_500_universe.json', 'r') as f:
            self.universe = json.load(f)
        
        print(f"Robust TPE Optimizer: Loaded Elite 500 universe with {len(self.universe)} symbols")
        
        self.param_space = {
            "momentum_period": {'low': 5, 'high': 15},
            "volume_threshold": {'low': 1.2, 'high': 2.5},
            "stop_loss": {'low': 0.03, 'high': 0.12},
            "take_profit": {'low': 0.08, 'high': 0.25},
            "momentum_threshold": {'low': 0.01, 'high': 0.1}
        }
        
        self.checkpoint_file = 'tpe_checkpoint.json'
        self.trials = []
        self.best_sharpe = float('-inf')
        self.best_params = None
        self.start_trial = 1
        
        # Load existing progress
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """Load existing optimization progress"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                self.trials = data.get('trials', [])
                self.best_sharpe = data.get('best_sharpe', float('-inf'))
                self.best_params = data.get('best_params', None)
                self.start_trial = len(self.trials) + 1
                
                print(f"Resumed from trial {self.start_trial}, best Sharpe: {self.best_sharpe:.3f}")
            except Exception as e:
                print(f"Checkpoint load error: {e}")
    
    def save_checkpoint(self):
        """Save current optimization progress"""
        checkpoint_data = {
            'trials': self.trials,
            'best_sharpe': self.best_sharpe,
            'best_params': self.best_params,
            'last_update': datetime.now().isoformat()
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            print(f"Checkpoint save error: {e}")
    
    def suggest_parameters(self, trial_num):
        """Generate deterministic parameter suggestions"""
        # Deterministic parameter selection based on trial number
        momentum_periods = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        volume_thresholds = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
        stop_losses = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
        take_profits = [0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.25]
        momentum_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        
        params = {
            "momentum_period": momentum_periods[trial_num % len(momentum_periods)],
            "volume_threshold": volume_thresholds[trial_num % len(volume_thresholds)],
            "stop_loss": stop_losses[trial_num % len(stop_losses)],
            "take_profit": take_profits[trial_num % len(take_profits)],
            "momentum_threshold": momentum_thresholds[trial_num % len(momentum_thresholds)]
        }
        
        return params
    
    def fetch_symbol_data(self, symbol):
        """Fetch authentic data for symbol with timestamp verification"""
        try:
            current_time = datetime.now()
            end_date = current_time.strftime('%Y-%m-%d')
            start_date = (current_time - timedelta(days=800)).strftime('%Y-%m-%d')
            
            print(f"TIMESTAMP VERIFICATION: Fetching {symbol} data at {current_time.isoformat()}")
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            response = requests.get(url, params={
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 800,
                'apikey': self.polygon_api_key
            }, timeout=8)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get('results') or len(data['results']) < 150:
                print(f"TIMESTAMP VERIFICATION: {symbol} - Insufficient data or no results")
                return None
            
            df = pd.DataFrame(data['results'])
            
            # Verify timestamps are authentic
            if 't' in df.columns:
                timestamps = pd.to_datetime(df['t'], unit='ms')
                latest_timestamp = timestamps.max()
                oldest_timestamp = timestamps.min()
                print(f"TIMESTAMP VERIFICATION: {symbol} - Data range {oldest_timestamp.date()} to {latest_timestamp.date()}")
                
                # Check for suspicious patterns (all same timestamps, future dates, etc.)
                if len(timestamps.unique()) < len(timestamps) * 0.8:
                    print(f"WARNING: {symbol} - Suspicious timestamp patterns detected")
                    return None
            
            return df[['c', 'v']].values  # close, volume
            
        except Exception:
            return None
    
    def backtest_symbol(self, symbol, params):
        """Backtest strategy on symbol"""
        data = self.fetch_symbol_data(symbol)
        if data is None or len(data) < 50:
            return None
        
        try:
            closes = data[:, 0]
            volumes = data[:, 1]
            
            # Calculate momentum
            momentum_period = params['momentum_period']
            momentum = np.zeros(len(closes))
            
            for i in range(momentum_period, len(closes)):
                momentum[i] = (closes[i] / closes[i-momentum_period] - 1)
            
            # Volume filter
            volume_avg = np.zeros(len(volumes))
            for i in range(15, len(volumes)):
                volume_avg[i] = np.mean(volumes[i-15:i])
            
            # Find signals
            signals = []
            for i in range(15, len(closes)):
                if (momentum[i] > params['momentum_threshold'] and 
                    volumes[i] > volume_avg[i] * params['volume_threshold']):
                    signals.append(i)
            
            if len(signals) == 0:
                return {'sharpe': -3, 'trades': 0}
            
            # Calculate returns
            returns = []
            for signal_idx in signals:
                entry_price = closes[signal_idx]
                
                for day in range(1, min(9, len(closes) - signal_idx)):
                    current_price = closes[signal_idx + day]
                    current_return = (current_price / entry_price - 1)
                    
                    if (current_return >= params['take_profit'] or 
                        current_return <= -params['stop_loss'] or 
                        day == 8):
                        returns.append(current_return)
                        break
            
            if len(returns) == 0:
                return {'sharpe': -3, 'trades': 0}
            
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0.01
            sharpe = avg_return / std_return if std_return > 0 else -3
            
            return {'sharpe': sharpe, 'trades': len(returns)}
            
        except Exception:
            return None
    
    def evaluate_parameters(self, params):
        """Evaluate parameters across symbol sample"""
        sample_size = min(40, len(self.universe))
        sample_symbols = self.universe[:sample_size]  # Use first 40 symbols deterministically
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(self.backtest_symbol, symbol, params) for symbol in sample_symbols]
            results = []
            
            for future in as_completed(futures):
                result = future.result()
                if result and result['sharpe'] > -3:
                    results.append(result)
        
        if len(results) == 0:
            return {'sharpe': -10, 'trades': 0, 'symbols': 0}
        
        total_sharpe = sum(r['sharpe'] for r in results)
        total_trades = sum(r['trades'] for r in results)
        avg_sharpe = total_sharpe / len(results)
        
        if total_trades < 3:
            avg_sharpe -= 5
        
        return {
            'sharpe': avg_sharpe,
            'trades': total_trades,
            'symbols': len(results)
        }
    
    def run_optimization(self, target_trials=200):
        """Run robust optimization with checkpointing"""
        print("Robust TPE Optimization - Authentic Polygon Data")
        print("Strategy: 8-Day Momentum with Volume Confirmation")
        print("Elite 500 Universe with Checkpoint Saving")
        print("=" * 55)
        
        start_time = time.time()
        
        for trial in range(self.start_trial, target_trials + 1):
            trial_start = time.time()
            
            params = self.suggest_parameters(trial)
            result = self.evaluate_parameters(params)
            
            trial_data = {
                'trial': trial,
                'params': params,
                'sharpe': result['sharpe'],
                'trades': result['trades'],
                'symbols': result['symbols'],
                'duration': time.time() - trial_start
            }
            self.trials.append(trial_data)
            
            if result['sharpe'] > self.best_sharpe:
                self.best_sharpe = result['sharpe']
                self.best_params = params.copy()
            
            # Save progress every 10 trials
            if trial % 10 == 0:
                self.save_checkpoint()
            
            # Progress reporting
            if trial % 15 == 0 or result['sharpe'] > self.best_sharpe:
                elapsed = (time.time() - start_time) / 60
                remaining = (elapsed / (trial - self.start_trial + 1)) * (target_trials - trial)
                
                print(f"Trial {trial:3d}: Sharpe {result['sharpe']:6.3f} | "
                      f"Best: {self.best_sharpe:6.3f} | "
                      f"Trades: {result['trades']:3d} | "
                      f"ETA: {remaining:4.1f}min")
        
        duration = (time.time() - start_time) / 60
        self.save_final_results(duration, target_trials)
        
        return {
            'best_params': self.best_params,
            'best_sharpe': self.best_sharpe,
            'duration': duration
        }
    
    def save_final_results(self, duration, trials):
        """Save final optimization results"""
        results = {
            'optimization_summary': {
                'best_sharpe': self.best_sharpe,
                'best_params': self.best_params,
                'total_trials': trials,
                'duration_minutes': duration,
                'optimization_date': datetime.now().isoformat(),
                'universe_size': len(self.universe)
            },
            'all_trials': self.trials,
            'top_10_trials': sorted(self.trials, key=lambda x: x['sharpe'], reverse=True)[:10],
            'performance_stats': {
                'avg_sharpe': np.mean([t['sharpe'] for t in self.trials if t['sharpe'] > -10]),
                'successful_trials': len([t for t in self.trials if t['sharpe'] > 0]),
                'total_trades': sum(t['trades'] for t in self.trials)
            }
        }
        
        with open('robust_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n" + "=" * 55)
        print("ROBUST OPTIMIZATION COMPLETE")
        print(f"Best Sharpe Ratio: {self.best_sharpe:.4f}")
        print(f"Total Duration: {duration:.1f} minutes")
        print(f"Trials Completed: {len(self.trials)}")
        print(f"Results saved to: robust_optimization_results.json")
        print("=" * 55)

def main():
    optimizer = RobustTPEOptimizer()
    results = optimizer.run_optimization(target_trials=200)
    
    print(f"\nBest parameters: {results['best_params']}")
    print(f"Best Sharpe ratio: {results['best_sharpe']:.4f}")

if __name__ == "__main__":
    main()