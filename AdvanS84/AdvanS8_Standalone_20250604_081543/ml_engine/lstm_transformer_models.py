"""
LSTM and Transformer Models for Trading Signal Generation
Lightweight implementation using authentic market data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor:
    """LSTM-style predictor using gradient boosting for sequential patterns"""
    
    def __init__(self, lookback_window=20):
        self.lookback_window = lookback_window
        self.scaler = MinMaxScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.is_trained = False
    
    def prepare_sequences(self, prices, indicators):
        """Prepare sequential data for LSTM-style training"""
        if len(prices) < self.lookback_window + 10:
            return None, None
        
        # Ensure indicators have required columns
        required_cols = ['rsi', 'macd', 'momentum']
        for col in required_cols:
            if col not in indicators.columns:
                indicators[col] = 0.0
        
        # Align data lengths
        min_length = min(len(prices), len(indicators))
        if min_length < self.lookback_window + 10:
            return None, None
        
        # Use last min_length data points
        aligned_prices = prices.iloc[-min_length:]
        aligned_indicators = indicators.iloc[-min_length:]
        
        # Combine price and indicator features
        features = np.column_stack([
            aligned_prices.values,
            aligned_indicators['rsi'].fillna(50.0).values,
            aligned_indicators['macd'].fillna(0.0).values,
            aligned_indicators['momentum'].fillna(0.0).values
        ])
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_window, len(features) - 5):
            # Input sequence
            sequence = features[i-self.lookback_window:i]
            X.append(sequence.flatten())
            
            # Target: price direction in next 5 days
            future_return = (prices.iloc[i+5] / prices.iloc[i] - 1)
            y.append(1 if future_return > 0.02 else 0)  # 2% threshold
        
        if len(X) == 0:
            return None, None
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train(self, prices, indicators):
        """Train LSTM-style model on historical data"""
        X, y = self.prepare_sequences(prices, indicators)
        if X is None or len(X) < 20:
            return False
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except Exception:
            return False
    
    def predict(self, prices, indicators):
        """Generate LSTM-style prediction"""
        if not self.is_trained or len(prices) < self.lookback_window:
            return "hold"
        
        try:
            # Ensure indicators have required columns
            required_cols = ['rsi', 'macd', 'momentum']
            for col in required_cols:
                if col not in indicators.columns:
                    indicators[col] = 0.0
            
            # Prepare current sequence with proper data handling
            rsi_values = indicators['rsi'].fillna(50.0).values[-self.lookback_window:]
            macd_values = indicators['macd'].fillna(0.0).values[-self.lookback_window:]
            momentum_values = indicators['momentum'].fillna(0.0).values[-self.lookback_window:]
            price_values = prices.values[-self.lookback_window:]
            
            # Ensure all arrays have the same length
            min_len = min(len(price_values), len(rsi_values), len(macd_values), len(momentum_values))
            if min_len < self.lookback_window:
                return "hold"
            
            features = np.column_stack([
                price_values[-min_len:],
                rsi_values[-min_len:],
                macd_values[-min_len:],
                momentum_values[-min_len:]
            ])
            
            sequence = features.flatten().reshape(1, -1)
            sequence = self.scaler.transform(sequence)
            
            # Get prediction probability
            prob = self.model.predict_proba(sequence)[0]
            confidence = max(prob)
            prediction = self.model.predict(sequence)[0]
            
            if confidence > 0.65:
                return "buy" if prediction == 1 else "sell"
            else:
                return "hold"
                
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return "hold"

class TransformerPredictor:
    """Transformer-style predictor using attention-like pattern analysis"""
    
    def __init__(self, attention_window=30):
        self.attention_window = attention_window
        self.pattern_models = {}
        self.is_trained = False
    
    def extract_patterns(self, prices, indicators):
        """Extract transformer-style attention patterns"""
        if len(prices) < self.attention_window:
            return {}
        
        patterns = {}
        
        # Price momentum patterns
        returns = prices.pct_change()
        patterns['momentum_trend'] = returns.rolling(5).mean().iloc[-1]
        patterns['momentum_acceleration'] = returns.rolling(5).mean().diff().iloc[-1]
        
        # Volume-price relationship patterns
        if 'volume_ratio' in indicators.columns:
            patterns['volume_price_sync'] = np.corrcoef(
                returns.dropna().iloc[-20:],
                indicators['volume_ratio'].dropna().iloc[-20:]
            )[0, 1] if len(returns.dropna()) >= 20 else 0
        else:
            patterns['volume_price_sync'] = 0
        
        # Technical indicator patterns
        patterns['rsi_momentum'] = indicators['rsi'].diff().iloc[-1]
        patterns['macd_divergence'] = indicators['macd'].iloc[-1] - indicators['macd'].rolling(10).mean().iloc[-1]
        
        # Multi-timeframe attention
        patterns['short_trend'] = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
        patterns['medium_trend'] = (prices.iloc[-1] / prices.iloc[-15] - 1) if len(prices) >= 15 else 0
        patterns['long_trend'] = (prices.iloc[-1] / prices.iloc[-30] - 1) if len(prices) >= 30 else 0
        
        # Pattern consistency (attention weights)
        consistency_score = 0
        if patterns['short_trend'] > 0 and patterns['medium_trend'] > 0:
            consistency_score += 0.3
        if patterns['momentum_trend'] > 0 and patterns['momentum_acceleration'] > 0:
            consistency_score += 0.3
        if patterns['rsi_momentum'] > 0 and patterns['macd_divergence'] > 0:
            consistency_score += 0.4
        
        patterns['attention_score'] = consistency_score
        
        return patterns
    
    def train(self, prices, indicators):
        """Train transformer-style pattern recognition"""
        if len(prices) < self.attention_window + 10:
            return False
        
        try:
            # Train on multiple time windows
            for i in range(self.attention_window, len(prices) - 10, 10):
                window_prices = prices.iloc[i-self.attention_window:i]
                window_indicators = indicators.iloc[i-self.attention_window:i]
                
                patterns = self.extract_patterns(window_prices, window_indicators)
                
                # Future performance
                future_return = prices.iloc[i+10] / prices.iloc[i] - 1
                success = future_return > 0.03  # 3% target
                
                # Store successful patterns
                if patterns['attention_score'] > 0.5:
                    pattern_key = f"pattern_{len(self.pattern_models)}"
                    self.pattern_models[pattern_key] = {
                        'patterns': patterns,
                        'success': success,
                        'return': future_return
                    }
            
            self.is_trained = len(self.pattern_models) > 5
            return self.is_trained
            
        except Exception:
            return False
    
    def predict(self, prices, indicators):
        """Generate transformer-style prediction using attention patterns"""
        if not self.is_trained:
            return "hold"
        
        try:
            current_patterns = self.extract_patterns(prices, indicators)
            
            if current_patterns['attention_score'] < 0.3:
                return "hold"
            
            # Match against successful historical patterns
            similarity_scores = []
            for pattern_data in self.pattern_models.values():
                if not pattern_data['success']:
                    continue
                
                stored_patterns = pattern_data['patterns']
                
                # Calculate pattern similarity (attention mechanism)
                similarity = 0
                for key in ['momentum_trend', 'short_trend', 'medium_trend']:
                    if key in current_patterns and key in stored_patterns:
                        diff = abs(current_patterns[key] - stored_patterns[key])
                        similarity += max(0, 1 - diff) * 0.33
                
                if similarity > 0.7:
                    similarity_scores.append((similarity, pattern_data['return']))
            
            if not similarity_scores:
                return "hold"
            
            # Weighted prediction based on pattern similarity
            weighted_return = sum(sim * ret for sim, ret in similarity_scores) / len(similarity_scores)
            avg_similarity = sum(sim for sim, _ in similarity_scores) / len(similarity_scores)
            
            if avg_similarity > 0.8 and weighted_return > 0.02:
                return "buy"
            elif avg_similarity > 0.8 and weighted_return < -0.02:
                return "sell"
            else:
                return "hold"
                
        except Exception:
            return "hold"

class MLSignalGenerator:
    """Combined LSTM + Transformer signal generation"""
    
    def __init__(self):
        self.lstm = LSTMPredictor()
        self.transformer = TransformerPredictor()
        self.is_trained = False
    
    def train_models(self, prices, indicators):
        """Train both LSTM and Transformer models"""
        lstm_success = self.lstm.train(prices, indicators)
        transformer_success = self.transformer.train(prices, indicators)
        
        self.is_trained = lstm_success or transformer_success
        return self.is_trained
    
    def generate_signal(self, prices, indicators, regime="neutral"):
        """Generate combined ML signal"""
        if not self.is_trained:
            return "hold"
        
        lstm_signal = self.lstm.predict(prices, indicators)
        transformer_signal = self.transformer.predict(prices, indicators)
        
        # Signal combination logic
        if lstm_signal == transformer_signal:
            # Both models agree
            confidence_boost = 1.2 if regime in ["bull", "neutral"] else 0.8
            return lstm_signal if np.random.random() < confidence_boost * 0.8 else "hold"
        
        elif lstm_signal == "buy" and transformer_signal == "hold":
            return "buy" if regime in ["bull", "neutral"] else "hold"
        
        elif transformer_signal == "buy" and lstm_signal == "hold":
            return "buy" if regime == "bull" else "hold"
        
        elif lstm_signal == "sell" or transformer_signal == "sell":
            return "sell" if regime in ["bear", "volatile"] else "hold"
        
        else:
            return "hold"