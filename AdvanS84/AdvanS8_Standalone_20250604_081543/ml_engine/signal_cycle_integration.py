"""
Signal Cycle Integration - Unified ML Signal Processing
Connects LSTM + Transformer outputs with trading decisions
"""

import json
import asyncio
import aiohttp
import pandas as pd
import time
import sys
import os
from datetime import datetime
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_config import get_polygon_key
from ml_engine.market_regime_classifier import classify_regime
from ml_engine.lstm_transformer_models import MLSignalGenerator

# Cache and rate limiting
RATE_LIMIT = 100
BATCH_SIZE = 95
CACHE = OrderedDict()
MAX_CACHE_SIZE = 300
RETRY_LIMIT = 3
RETRY_DELAY = 1.5

class SignalCycleProcessor:
    """Processes ML signals and generates trading decisions"""
    
    def __init__(self):
        self.polygon_api_key = get_polygon_key()
        self.ml_generator = MLSignalGenerator()
        self.signal_decisions = {}
        self.performance_log = []
        
        # Load full Elite 500 universe
        try:
            with open('elites_500_universe.json', 'r') as f:
                universe_data = json.load(f)
                self.universe = universe_data.get('elites_500_universe', [])
                print(f"Loaded Elite 500 universe with {len(self.universe)} symbols")
        except:
            # Fallback to core symbols if file not found
            self.universe = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JNJ",
                "UNH", "JPM", "XOM", "WMT", "PG", "MA", "HD", "BAC", "LLY", "DIS"
            ]
            print(f"Using fallback universe with {len(self.universe)} symbols")
    
    async def get_historical_prices_async(self, session, symbol: str, days: int = 730) -> pd.Series:
        """Fetch historical prices with caching and retry logic"""
        if symbol in CACHE:
            return CACHE[symbol]
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2020-01-01/2024-12-31"
        params = {
            'adjusted': 'true',
            'sort': 'desc',
            'limit': days,
            'apikey': self.polygon_api_key
        }
        
        for attempt in range(RETRY_LIMIT):
            try:
                start = time.perf_counter()
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    elapsed = time.perf_counter() - start
                    
                    if data.get('results'):
                        prices = [bar['c'] for bar in data['results']]
                        series = pd.Series(prices[::-1])  # Reverse for chronological order
                        
                        # Cache management
                        CACHE[symbol] = series
                        if len(CACHE) > MAX_CACHE_SIZE:
                            CACHE.popitem(last=False)
                        
                        print(f"Signal data for {symbol} fetched in {elapsed:.2f}s")
                        return series
                        
            except Exception as e:
                print(f"Retry {attempt+1} for {symbol}: {e}")
                await asyncio.sleep(RETRY_DELAY)
        
        return pd.Series(dtype=float)
    
    async def get_lstm_signal(self, symbol: str) -> str:
        """Get LSTM signal from live ML signals"""
        try:
            with open('live_ml_signals.json', 'r') as f:
                signals = json.load(f)
            
            lstm_signals = signals.get('current_signals', {}).get('lstm', {})
            if symbol in lstm_signals:
                return lstm_signals[symbol]['signal']
            
            return "hold"
            
        except Exception:
            return "hold"
    
    async def get_transformer_signal(self, symbol: str) -> str:
        """Get Transformer signal from live ML signals"""
        try:
            with open('live_ml_signals.json', 'r') as f:
                signals = json.load(f)
            
            transformer_signals = signals.get('current_signals', {}).get('transformer', {})
            if symbol in transformer_signals:
                return transformer_signals[symbol]['signal']
            
            return "hold"
            
        except Exception:
            return "hold"
    
    async def fetch_predictions(self, session, symbol):
        """Combine all predictions and regime classification per symbol"""
        lstm_pred = await self.get_lstm_signal(symbol)
        transformer_pred = await self.get_transformer_signal(symbol)
        prices = await self.get_historical_prices_async(session, symbol)
        
        regime_code = classify_regime(prices) if len(prices) > 50 else 0
        regime = {-1: "volatile", 0: "bear", 1: "neutral", 2: "bull"}.get(regime_code, "neutral")
        
        return symbol, lstm_pred, transformer_pred, regime
    
    async def fetch_all_predictions(self, symbols):
        """Fetch all signals in batches respecting API rate limits"""
        async with aiohttp.ClientSession() as session:
            results = []
            for i in range(0, len(symbols), BATCH_SIZE):
                batch = symbols[i:i + BATCH_SIZE]
                tasks = [self.fetch_predictions(session, symbol) for symbol in batch]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                
                if i + BATCH_SIZE < len(symbols):
                    await asyncio.sleep(1.1)  # Stay under 100 requests/sec
            
            return results
    
    def run_signal_cycle(self, symbol: str, lstm_pred: str, transformer_pred: str, regime: str):
        """Process individual signal and make trading decision"""
        timestamp = datetime.now().isoformat()
        
        # Combined signal logic
        final_signal = "hold"
        confidence = 0.0
        
        # Both models agree
        if lstm_pred == transformer_pred and lstm_pred != "hold":
            final_signal = lstm_pred
            confidence = 0.9
        # High confidence single model in favorable regime
        elif regime == "bull" and (lstm_pred == "buy" or transformer_pred == "buy"):
            final_signal = "buy"
            confidence = 0.7
        elif regime == "bear" and (lstm_pred == "sell" or transformer_pred == "sell"):
            final_signal = "sell"
            confidence = 0.7
        # Conservative approach in uncertain regimes
        elif regime in ["neutral", "volatile"]:
            if lstm_pred == transformer_pred:
                final_signal = lstm_pred
                confidence = 0.6
        
        # Store signal decision
        self.signal_decisions[symbol] = {
            'symbol': symbol,
            'final_signal': final_signal,
            'lstm_signal': lstm_pred,
            'transformer_signal': transformer_pred,
            'regime': regime,
            'confidence': confidence,
            'timestamp': timestamp,
            'agreement': lstm_pred == transformer_pred
        }
        
        # Log performance
        self.performance_log.append({
            'symbol': symbol,
            'signal': final_signal,
            'confidence': confidence,
            'regime': regime,
            'timestamp': timestamp
        })
        
        print(f"Signal: {symbol} -> {final_signal} ({confidence:.1%} confidence, {regime} regime)")
    
    def get_signal_summary(self):
        """Get summary of current signals"""
        if not self.signal_decisions:
            return {}
        
        buy_signals = [s for s in self.signal_decisions.values() if s['final_signal'] == 'buy']
        sell_signals = [s for s in self.signal_decisions.values() if s['final_signal'] == 'sell']
        hold_signals = [s for s in self.signal_decisions.values() if s['final_signal'] == 'hold']
        
        return {
            'total_signals': len(self.signal_decisions),
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'hold_count': len(hold_signals),
            'high_confidence': len([s for s in self.signal_decisions.values() if s['confidence'] > 0.8]),
            'model_agreement': len([s for s in self.signal_decisions.values() if s['agreement']]),
            'latest_update': datetime.now().isoformat()
        }
    
    def save_signal_decisions(self):
        """Save signal decisions to file"""
        output = {
            'signal_decisions': self.signal_decisions,
            'summary': self.get_signal_summary(),
            'performance_log': self.performance_log[-100:],  # Keep last 100 entries
            'updated': datetime.now().isoformat()
        }
        
        with open('signal_cycle_decisions.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
    
    async def run_full_cycle(self, symbol_limit=None):
        """Execute complete signal cycle for Elite 500 universe"""
        if symbol_limit is None:
            symbol_limit = len(self.universe)
        
        print(f"Running signal cycle for {symbol_limit} symbols from Elite 500 universe...")
        
        symbols = self.universe[:symbol_limit]
        results = await self.fetch_all_predictions(symbols)
        
        processed = 0
        for symbol, lstm_pred, transformer_pred, regime in results:
            self.run_signal_cycle(symbol, lstm_pred, transformer_pred, regime)
            processed += 1
        
        self.save_signal_decisions()
        
        summary = self.get_signal_summary()
        print(f"Signal cycle complete: {processed} symbols processed")
        print(f"Signals: {summary['buy_count']} buy, {summary['sell_count']} sell, {summary['hold_count']} hold")
        print(f"High confidence: {summary['high_confidence']}, Model agreement: {summary['model_agreement']}")
        
        return summary

def main():
    """Main execution entry for signal cycle"""
    processor = SignalCycleProcessor()
    
    # Run signal cycle for top 50 symbols
    results = asyncio.run(processor.run_full_cycle(symbol_limit=50))
    
    print("Signal cycle integration complete")
    print(f"Results saved to signal_cycle_decisions.json")
    
    return results

if __name__ == "__main__":
    main()