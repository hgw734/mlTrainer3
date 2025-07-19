"""
Signal Transition Manager
Converts insufficient_data signals to real trading signals as data accumulates
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from api_config import get_polygon_key, get_polygon_params
import requests

class SignalTransitionManager:
    """
    Manages transition from insufficient_data to real trading signals
    """
    
    def __init__(self):
        self.signals_path = Path("signals_live.json")
        self.polygon_api_key = get_polygon_key()
        self.base_url = "https://api.polygon.io"
        self.required_bars = 30  # Reduced for faster transitions
        
        # Load optimized parameters
        self.load_optimized_parameters()
        
        logging.info("Signal Transition Manager initialized")
    
    def load_optimized_parameters(self):
        """Load TPE-optimized parameters"""
        try:
            with open('robust_optimization_results.json', 'r') as f:
                data = json.load(f)
                self.best_params = data['optimization_summary']['best_params']
        except:
            self.best_params = {
                'momentum_period': 9,
                'volume_threshold': 2.12,
                'momentum_threshold': 0.053
            }
    
    def load_signals(self):
        """Load current live signals"""
        if not self.signals_path.exists():
            return {}
        try:
            with open(self.signals_path) as f:
                data = json.load(f)
                return data.get('signals', {})
        except:
            return {}
    
    def save_signals(self, signals):
        """Save updated signals"""
        try:
            # Load existing data structure
            if self.signals_path.exists():
                with open(self.signals_path) as f:
                    data = json.load(f)
            else:
                data = {}
            
            # Update signals and metadata
            data['signals'] = signals
            data['updated'] = datetime.now().isoformat()
            data['total_symbols'] = len(signals)
            
            with open(self.signals_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            logging.info(f"Saved signals for {len(signals)} symbols")
        except Exception as e:
            logging.error(f"Error saving signals: {e}")
    
    def fetch_symbol_history(self, symbol, days=5):
        """Fetch recent price history for a symbol with timestamp verification"""
        try:
            current_time = datetime.now()
            print(f"TIMESTAMP VERIFICATION: Fetching {symbol} history at {current_time.isoformat()}")
            
            end_time = current_time
            start_time = end_time - timedelta(days=days)
            
            # Use minute bars for better granularity
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{start_time.date()}/{end_time.date()}"
            params = get_polygon_params()
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    bars = data['results']
                    
                    df = pd.DataFrame(bars)
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    
                    # Verify timestamp authenticity
                    if not df.empty:
                        latest_timestamp = df['timestamp'].max()
                        oldest_timestamp = df['timestamp'].min()
                        print(f"TIMESTAMP VERIFICATION: {symbol} - Data range {oldest_timestamp} to {latest_timestamp}")
                        
                        # Check for suspicious patterns
                        if latest_timestamp > current_time:
                            print(f"WARNING: {symbol} - Future timestamp detected: {latest_timestamp}")
                            return None
                        
                        if len(df['timestamp'].unique()) < len(df) * 0.8:
                            print(f"WARNING: {symbol} - Suspicious timestamp patterns detected")
                            return None
                    
                    df = df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                    })
                    
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return None
            
        except Exception as e:
            logging.warning(f"Failed to fetch history for {symbol}: {e}")
            return None
    
    def has_sufficient_history(self, df):
        """Check if symbol has enough data for inference"""
        if df is None or len(df) < self.required_bars:
            return False
        
        # Check data quality
        clean_data = df.dropna()
        return len(clean_data) >= self.required_bars
    
    def generate_features(self, df):
        """Generate technical features for inference"""
        if len(df) < 20:
            return df
        
        # Momentum features
        for period in [3, 5, 10]:
            if len(df) > period:
                df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Volume features
        if len(df) > 10:
            df['volume_sma'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        if len(df) > 10:
            df['volatility'] = df['returns'].rolling(10).std()
        
        # RSI
        if len(df) > 14:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        return df.fillna(method='bfill').fillna(0)
    
    def infer_signal(self, symbol, df):
        """Generate trading signal using enhanced logic"""
        try:
            # Generate features
            df = self.generate_features(df)
            
            # LSTM-style inference
            lstm_signal = self.lstm_inference(df, symbol)
            
            # Transformer-style inference
            transformer_signal = self.transformer_inference(df, symbol)
            
            # Combine signals
            final_signal, confidence = self.combine_signals(lstm_signal, transformer_signal)
            
            return final_signal, confidence
            
        except Exception as e:
            logging.error(f"Inference error for {symbol}: {e}")
            return 'hold', 0.5
    
    def lstm_inference(self, df, symbol):
        """LSTM-style signal generation"""
        try:
            # Extract momentum features
            momentum_features = ['momentum_3', 'momentum_5', 'momentum_10']
            available_features = [f for f in momentum_features if f in df.columns]
            
            if len(available_features) < 2:
                return {'signal': 'hold', 'confidence': 0.5}
            
            # Recent momentum analysis
            recent_data = df[available_features].tail(10)
            momentum_score = recent_data.mean().mean()
            momentum_trend = recent_data.tail(3).mean().mean() - recent_data.head(3).mean().mean()
            
            # Volume confirmation
            volume_factor = 1.0
            if 'volume_ratio' in df.columns:
                volume_factor = df['volume_ratio'].iloc[-1] if len(df) > 0 else 1.0
                volume_factor = min(2.0, max(0.5, volume_factor))
            
            # Signal strength
            signal_strength = momentum_score * volume_factor
            confidence = min(0.95, abs(signal_strength) * 30 + 0.5)
            
            # Generate signal
            if signal_strength > self.best_params['momentum_threshold']:
                signal = 'buy'
            elif signal_strength < -self.best_params['momentum_threshold']:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'momentum_score': momentum_score
            }
            
        except Exception as e:
            logging.error(f"LSTM inference error for {symbol}: {e}")
            return {'signal': 'hold', 'confidence': 0.3}
    
    def transformer_inference(self, df, symbol):
        """Transformer-style pattern recognition"""
        try:
            if len(df) < 20:
                return {'signal': 'hold', 'confidence': 0.5}
            
            # Pattern analysis
            price_sequence = df['close'].tail(20).values
            returns_sequence = df['returns'].tail(20).values
            
            # Trend detection
            long_trend = np.polyfit(range(20), price_sequence, 1)[0]
            short_trend = np.polyfit(range(5), price_sequence[-5:], 1)[0]
            
            # Pattern consistency
            pattern_strength = abs(long_trend) * 1000  # Scale for readability
            trend_alignment = 1 if (long_trend > 0 and short_trend > 0) or (long_trend < 0 and short_trend < 0) else 0.5
            
            # Volume pattern
            volume_trend = 1.0
            if 'volume_ratio' in df.columns:
                volume_trend = df['volume_ratio'].tail(5).mean()
            
            # Confidence calculation
            confidence = min(0.95, pattern_strength * trend_alignment * 0.5 + 0.4)
            
            # Signal generation
            if long_trend > 0.001 and trend_alignment > 0.8:
                signal = 'buy'
            elif long_trend < -0.001 and trend_alignment > 0.8:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'pattern_strength': pattern_strength
            }
            
        except Exception as e:
            logging.error(f"Transformer inference error for {symbol}: {e}")
            return {'signal': 'hold', 'confidence': 0.3}
    
    def combine_signals(self, lstm_result, transformer_result):
        """Combine LSTM and Transformer signals"""
        lstm_signal = lstm_result['signal']
        transformer_signal = transformer_result['signal']
        lstm_conf = lstm_result['confidence']
        transformer_conf = transformer_result['confidence']
        
        # Signal agreement logic
        if lstm_signal == transformer_signal:
            # Both models agree
            final_signal = lstm_signal
            final_confidence = (lstm_conf + transformer_conf) / 2
        elif 'buy' in [lstm_signal, transformer_signal] and 'sell' in [lstm_signal, transformer_signal]:
            # Conflicting signals - hold
            final_signal = 'hold'
            final_confidence = min(lstm_conf, transformer_conf)
        else:
            # One hold, one signal - use higher confidence
            if lstm_conf > transformer_conf:
                final_signal = lstm_signal
                final_confidence = lstm_conf * 0.8  # Reduce for disagreement
            else:
                final_signal = transformer_signal
                final_confidence = transformer_conf * 0.8
        
        return final_signal, final_confidence
    
    def classify_regime(self, df):
        """Classify market regime"""
        if len(df) < 20:
            return 'neutral'
        
        # Calculate regime indicators
        recent_return = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        volatility = df['returns'].tail(20).std() if 'returns' in df.columns else 0.02
        
        # Regime classification
        if recent_return > 0.05 and volatility < 0.025:
            return 'bull_stable'
        elif recent_return > 0.02:
            return 'bull'
        elif recent_return < -0.05 and volatility < 0.025:
            return 'bear_stable'
        elif recent_return < -0.02:
            return 'bear'
        elif volatility > 0.04:
            return 'volatile'
        else:
            return 'neutral'
    
    def transition_signals(self):
        """Main transition function - convert insufficient_data to real signals"""
        signals = self.load_signals()
        transitions_made = 0
        
        print(f"Checking {len(signals)} symbols for signal transitions...")
        
        for symbol, entry in signals.items():
            try:
                # Check if needs transition
                if (isinstance(entry.get('lstm'), dict) and 
                    entry['lstm'].get('reason') == 'insufficient_data'):
                    
                    # Fetch fresh data
                    df = self.fetch_symbol_history(symbol)
                    
                    if self.has_sufficient_history(df):
                        # Generate real signal
                        signal, confidence = self.infer_signal(symbol, df)
                        regime = self.classify_regime(df)
                        
                        # Update signal entry
                        signals[symbol] = {
                            'lstm': {
                                'signal': signal,
                                'confidence': confidence,
                                'model': 'lstm_transition'
                            },
                            'transformer': {
                                'signal': signal,
                                'confidence': confidence,
                                'model': 'transformer_transition'
                            },
                            'regime': regime,
                            'timestamp': datetime.now().isoformat(),
                            'price': df['close'].iloc[-1] if len(df) > 0 else 0
                        }
                        
                        transitions_made += 1
                        print(f"[TRANSITION] {symbol}: {signal} (confidence: {confidence:.3f}, regime: {regime})")
                        
                        # Rate limiting
                        import time
                        time.sleep(0.1)
            
            except Exception as e:
                logging.error(f"Transition error for {symbol}: {e}")
        
        if transitions_made > 0:
            self.save_signals(signals)
            print(f"Completed {transitions_made} signal transitions")
        else:
            print("No signals ready for transition")
        
        return transitions_made

def main():
    """Main function for signal transition management"""
    print("Signal Transition Manager")
    print("=" * 40)
    
    manager = SignalTransitionManager()
    transitions = manager.transition_signals()
    
    print(f"Signal transition cycle complete: {transitions} transitions made")

if __name__ == "__main__":
    main()