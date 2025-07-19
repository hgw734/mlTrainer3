# AdvanS8 Market Regime Integration with GMM
# Integrates GMM regime detection as a feature for LSTM/Transformer models,
# adds it to signal logs, and prepares the system for dashboard integration and model arbitration.

import numpy as np
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
from utils.config_loader import CONFIG
from utils.symbol_mapper import normalize_symbol
import requests
from datetime import datetime

# === 1. Fetch historical prices from Polygon ===
def get_historical_prices(symbol: str, days: int = 60) -> pd.Series:
    key = CONFIG['polygon_api_key']
    norm = normalize_symbol(symbol)
    url = f"https://api.polygon.io/v2/aggs/ticker/{norm}/range/1/day/2023-01-01/2024-12-31?adjusted=true&sort=desc&limit={days}&apiKey={key}"
    try:
        response = requests.get(url)
        data = response.json().get("results", [])
        prices = [bar['c'] for bar in data]
        return pd.Series(prices[::-1])  # chronological order
    except Exception as e:
        print(f"[ERROR] Price fetch failed for {symbol}: {e}")
        return pd.Series(dtype=float)

# === 2. Classify market regime using GMM ===
def classify_regime(prices: pd.Series, n_states=3) -> int:
    if prices.empty or len(prices) < 10:
        return -1

    log_returns = np.log(prices / prices.shift(1)).dropna().values.reshape(-1, 1)
    model = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
    model.fit(log_returns)
    states = model.predict(log_returns)
    return states[-1]

# === 3. Enhanced signal cycle with regime-aware decision making ===
def run_signal_cycle(symbol: str, lstm_prediction: str, transformer_prediction: str, 
                    momentum_period=9, volume_threshold=2.12, stop_loss=0.0583, 
                    take_profit=0.2245, momentum_threshold=0.0529):
    """
    Run enhanced signal cycle with GMM regime detection and optimized parameters
    
    Args:
        symbol: Stock symbol
        lstm_prediction: LSTM model prediction (buy/sell/hold)
        transformer_prediction: Transformer model prediction (buy/sell/hold)
        momentum_period: Optimized momentum period (default: 9)
        volume_threshold: Optimized volume threshold (default: 2.12)
        stop_loss: Optimized stop loss level (default: 0.0583)
        take_profit: Optimized take profit level (default: 0.2245)
        momentum_threshold: Optimized momentum threshold (default: 0.0529)
    
    Returns:
        dict: Signal decision with regime context and optimized parameters
    """
    prices = get_historical_prices(symbol)
    regime = classify_regime(prices)

    # Regime-aware decision logic
    final_decision = "hold"
    confidence = 0.5
    
    # Model agreement check
    models_agree = lstm_prediction == transformer_prediction
    
    if models_agree:
        if regime == 2 and lstm_prediction == "buy":  # Bull regime + buy signal
            final_decision = "buy"
            confidence = 0.8
        elif regime == 0 and lstm_prediction == "sell":  # Bear regime + sell signal
            final_decision = "sell"
            confidence = 0.8
        elif regime == 1 and lstm_prediction in ["buy", "sell"]:  # Neutral regime
            final_decision = lstm_prediction
            confidence = 0.6
    else:
        # Models disagree - use regime as tiebreaker
        if regime == 2:  # Bull regime favors buy
            final_decision = "buy" if "buy" in [lstm_prediction, transformer_prediction] else "hold"
            confidence = 0.4
        elif regime == 0:  # Bear regime favors sell
            final_decision = "sell" if "sell" in [lstm_prediction, transformer_prediction] else "hold"
            confidence = 0.4
        else:  # Neutral regime - conservative approach
            final_decision = "hold"
            confidence = 0.3

    # Create enhanced signal record with optimized parameters
    signal_record = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "regime": regime,
        "regime_name": ["Bear", "Neutral", "Bull"][regime] if regime >= 0 else "Unknown",
        "lstm_prediction": lstm_prediction,
        "transformer_prediction": transformer_prediction,
        "models_agree": models_agree,
        "final_decision": final_decision,
        "confidence": confidence,
        "momentum_period": momentum_period,
        "volume_threshold": volume_threshold,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "momentum_threshold": momentum_threshold,
        "optimized_params": True
    }

    # Log to CSV
    log_signal_to_csv(signal_record)
    
    print(f"AdvanS8 Signal: {symbol} -> {final_decision} (Regime: {signal_record['regime_name']}, Confidence: {confidence:.1%})")
    
    return signal_record

# === 4. Enhanced logging system ===
def log_signal_to_csv(signal_record: dict):
    """Log signal to CSV with proper headers"""
    os.makedirs("logs", exist_ok=True)
    
    log_df = pd.DataFrame([signal_record])
    log_file = "logs/signal_log.csv"
    
    # Add header if file doesn't exist
    if not os.path.exists(log_file):
        log_df.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, mode='a', header=False, index=False)

# === 5. Batch processing for multiple symbols ===
def process_symbol_batch(symbols: list, lstm_predictions: dict, transformer_predictions: dict):
    """
    Process multiple symbols with regime-aware decisions
    
    Args:
        symbols: List of stock symbols
        lstm_predictions: Dict of {symbol: prediction}
        transformer_predictions: Dict of {symbol: prediction}
    
    Returns:
        list: List of signal records
    """
    results = []
    
    for symbol in symbols:
        lstm_pred = lstm_predictions.get(symbol, "hold")
        transformer_pred = transformer_predictions.get(symbol, "hold")
        
        try:
            signal_record = run_signal_cycle(symbol, lstm_pred, transformer_pred)
            results.append(signal_record)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    return results

# === 6. Regime summary for dashboard ===
def get_regime_summary():
    """Get summary of recent regime classifications for dashboard"""
    try:
        if os.path.exists("logs/signal_log.csv"):
            df = pd.read_csv("logs/signal_log.csv")
            if not df.empty:
                recent_df = df.tail(50)  # Last 50 signals
                
                regime_counts = recent_df['regime_name'].value_counts()
                decision_counts = recent_df['final_decision'].value_counts()
                avg_confidence = recent_df['confidence'].mean()
                
                return {
                    'regime_distribution': regime_counts.to_dict(),
                    'decision_distribution': decision_counts.to_dict(),
                    'average_confidence': avg_confidence,
                    'total_signals': len(recent_df),
                    'last_updated': datetime.now().isoformat()
                }
    except Exception as e:
        print(f"Error generating regime summary: {e}")
    
    return {
        'regime_distribution': {},
        'decision_distribution': {},
        'average_confidence': 0,
        'total_signals': 0,
        'last_updated': datetime.now().isoformat()
    }

# === 7. Example usage ===
def example_usage():
    """Example of how to use the regime integration"""
    # Example LSTM and Transformer predictions
    lstm_predictions = {
        "AAPL": "buy",
        "MSFT": "hold",
        "GOOGL": "buy",
        "AMZN": "sell"
    }
    
    transformer_predictions = {
        "AAPL": "buy",
        "MSFT": "buy", 
        "GOOGL": "hold",
        "AMZN": "sell"
    }
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    # Process batch
    results = process_symbol_batch(symbols, lstm_predictions, transformer_predictions)
    
    # Get summary
    summary = get_regime_summary()
    
    print("\nRegime Integration Example Complete:")
    print(f"Processed {len(results)} signals")
    print(f"Summary: {summary}")
    
    return results

if __name__ == "__main__":
    example_usage()