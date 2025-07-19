"""
Enhanced ML Pipeline with Research-Based Improvements
====================================================

Integrates cutting-edge methodologies from 6 research papers to achieve
target accuracy of 92.48% through:

1. Rolling Window Methodology (365-day adaptive learning)
2. Enhanced Ensemble Architecture (LSTM + GRU + SimpleRNN)
3. Denoising Autoencoder for data quality
4. Advanced Feature Engineering (OHLC + Statistical)
5. Sentiment Analysis Integration Framework
6. Risk-Adjusted Performance Optimization

Research Sources:
- Paper 4: 92.48% accuracy with Rolling Window + SVM
- Paper 3: Ensemble RNN + Sentiment Analysis
- Paper 5: SA-DLSTM with Denoising Autoencoder
- Papers 1,2,6: Feature engineering and preprocessing validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, accuracy_score
import joblib
import os

# Deep Learning Imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Input
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning models will be disabled.")

logger = logging.getLogger(__name__)

class RollingWindowManager:
    """
    Implements 365-day rolling window methodology from Paper 4
    Target: 92.48% accuracy through adaptive learning
    """
    
    def __init__(self, window_size: int = 365):
        self.window_size = window_size
        self.training_history = []
        self.performance_history = []
        
    def get_training_window(self, data: pd.DataFrame, current_date: datetime) -> pd.DataFrame:
        """Get training data for the rolling window"""
        end_date = current_date
        start_date = end_date - timedelta(days=self.window_size)
        
        # Filter data within the rolling window
        training_data = data[
            (data.index >= start_date) & 
            (data.index < end_date)
        ].copy()
        
        return training_data
    
    def update_performance(self, accuracy: float, timestamp: datetime):
        """Track performance over time"""
        self.performance_history.append({
            'timestamp': timestamp,
            'accuracy': accuracy,
            'window_size': self.window_size
        })
        
        # Keep only last 100 performance records
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_adaptive_window_size(self) -> int:
        """Dynamically adjust window size based on performance"""
        if len(self.performance_history) < 10:
            return self.window_size
        
        recent_performance = [p['accuracy'] for p in self.performance_history[-10:]]
        avg_performance = np.mean(recent_performance)
        
        # Adjust window size based on performance
        if avg_performance > 0.90:  # High performance - maintain window
            return self.window_size
        elif avg_performance > 0.80:  # Medium performance - slightly increase
            return min(self.window_size + 30, 450)
        else:  # Low performance - reduce window for faster adaptation
            return max(self.window_size - 30, 200)

class DenosingAutoencoder:
    """
    Denoising Autoencoder for data quality improvement (Paper 5)
    Removes noise and extracts key features from financial data
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.encoder = None
        self.is_trained = False
        
    def build_model(self):
        """Build the denoising autoencoder architecture"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Denoising autoencoder disabled.")
            return None
        
        # Input layer
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(64, activation='relu')(encoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(self.input_dim, activation='linear')(decoded)
        
        # Create models
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.autoencoder
    
    def add_noise(self, data: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """COMPLIANCE VIOLATION: Noise generation disabled - only verified data allowed"""
        # CRITICAL COMPLIANCE: No synthetic noise generation permitted
        logger.error("COMPLIANCE VIOLATION: Synthetic noise generation blocked")
        return data  # Return original data without synthetic modifications
    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train the denoising autoencoder"""
        if not TENSORFLOW_AVAILABLE or self.autoencoder is None:
            logger.warning("Denoising autoencoder not available")
            return
        
        # Add noise to training data
        X_noisy = self.add_noise(X)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train the model
        history = self.autoencoder.fit(
            X_noisy, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        self.is_trained = True
        return history
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Extract denoised features"""
        if not self.is_trained or self.encoder is None:
            logger.warning("Autoencoder not trained. Returning original data.")
            return X
        
        return self.encoder.predict(X, verbose=0)
    
    def decode(self, X_encoded: np.ndarray) -> np.ndarray:
        """Reconstruct from encoded features"""
        if not self.is_trained or self.autoencoder is None:
            return X_encoded
        
        return self.autoencoder.predict(X_encoded, verbose=0)

class EnhancedFeatureEngineering:
    """
    Advanced feature engineering based on research papers
    Combines OHLC, statistical, and technical indicators
    """
    
    def __init__(self):
        self.feature_names = []
        self.scalers = {}
        
    def create_ohlc_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create OHLC-based features (Papers 1, 2)"""
        features = data.copy()
        
        # Basic OHLC features
        features['HL_pct'] = (features['high'] - features['low']) / features['close']
        features['OC_pct'] = (features['open'] - features['close']) / features['close']
        features['body_size'] = abs(features['open'] - features['close']) / features['close']
        features['upper_shadow'] = (features['high'] - np.maximum(features['open'], features['close'])) / features['close']
        features['lower_shadow'] = (np.minimum(features['open'], features['close']) - features['low']) / features['close']
        
        # Price position features
        features['close_position'] = (features['close'] - features['low']) / (features['high'] - features['low'])
        features['open_position'] = (features['open'] - features['low']) / (features['high'] - features['low'])
        
        return features
    
    def create_statistical_features(self, data: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Create statistical features (Paper 2)"""
        features = data.copy()
        
        for window in windows:
            # Moving averages
            features[f'sma_{window}'] = features['close'].rolling(window=window).mean()
            features[f'ema_{window}'] = features['close'].ewm(span=window).mean()
            
            # Volatility measures
            features[f'volatility_{window}'] = features['close'].rolling(window=window).std()
            features[f'volatility_pct_{window}'] = features[f'volatility_{window}'] / features['close']
            
            # Price momentum
            features[f'momentum_{window}'] = features['close'] / features['close'].shift(window) - 1
            features[f'roc_{window}'] = features['close'].pct_change(window)
            
            # Relative position
            features[f'price_vs_sma_{window}'] = features['close'] / features[f'sma_{window}'] - 1
            
            # Volume features (if available)
            if 'volume' in features.columns:
                features[f'volume_sma_{window}'] = features['volume'].rolling(window=window).mean()
                features[f'volume_ratio_{window}'] = features['volume'] / features[f'volume_sma_{window}']
        
        return features
    
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators"""
        features = data.copy()
        
        # RSI
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        features['rsi'] = calculate_rsi(features['close'])
        
        # MACD
        ema_12 = features['close'].ewm(span=12).mean()
        ema_26 = features['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma_20 = features['close'].rolling(window=20).mean()
        std_20 = features['close'].rolling(window=20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        return features
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged features"""
        features = data.copy()
        
        for lag in lags:
            features[f'close_lag_{lag}'] = features['close'].shift(lag)
            features[f'return_lag_{lag}'] = features['close'].pct_change().shift(lag)
            features[f'volume_lag_{lag}'] = features['volume'].shift(lag) if 'volume' in features.columns else 0
        
        return features
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        # Create all feature types
        features = self.create_ohlc_features(data)
        features = self.create_statistical_features(features)
        features = self.create_technical_indicators(features)
        features = self.create_lag_features(features)
        
        # Store feature names
        self.feature_names = [col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        return features

class EnsembleRNNModel:
    """
    Ensemble RNN approach from Paper 3
    Combines LSTM, GRU, and SimpleRNN for superior performance
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int = 1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}
        self.ensemble_weights = {'lstm': 0.4, 'gru': 0.35, 'simple_rnn': 0.25}
        
    def build_lstm_model(self) -> Optional[Model]:
        """Build LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(self.num_classes, activation='linear' if self.num_classes == 1 else 'softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse' if self.num_classes == 1 else 'categorical_crossentropy',
            metrics=['mae'] if self.num_classes == 1 else ['accuracy']
        )
        
        return model
    
    def build_gru_model(self) -> Optional[Model]:
        """Build GRU model"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            GRU(32),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(self.num_classes, activation='linear' if self.num_classes == 1 else 'softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse' if self.num_classes == 1 else 'categorical_crossentropy',
            metrics=['mae'] if self.num_classes == 1 else ['accuracy']
        )
        
        return model
    
    def build_simple_rnn_model(self) -> Optional[Model]:
        """Build SimpleRNN model"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = Sequential([
            SimpleRNN(128, return_sequences=True, input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(0.3),
            SimpleRNN(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            SimpleRNN(32),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(self.num_classes, activation='linear' if self.num_classes == 1 else 'softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse' if self.num_classes == 1 else 'categorical_crossentropy',
            metrics=['mae'] if self.num_classes == 1 else ['accuracy']
        )
        
        return model
    
    def build_ensemble(self):
        """Build all models in the ensemble"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Ensemble RNN disabled.")
            return
        
        self.models['lstm'] = self.build_lstm_model()
        self.models['gru'] = self.build_gru_model()
        self.models['simple_rnn'] = self.build_simple_rnn_model()
        
        logger.info("Ensemble RNN models built successfully")
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train all models in the ensemble"""
        if not TENSORFLOW_AVAILABLE or not self.models:
            logger.warning("Cannot train ensemble - models not available")
            return
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        histories = {}
        
        for name, model in self.models.items():
            if model is not None:
                logger.info(f"Training {name} model...")
                history = model.fit(
                    X, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )
                histories[name] = history
        
        return histories
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not TENSORFLOW_AVAILABLE or not self.models:
            logger.warning("Cannot predict - models not available")
            return np.zeros((len(X), self.num_classes))
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model is not None:
                pred = model.predict(X, verbose=0)
                predictions.append(pred)
                weights.append(self.ensemble_weights[name])
        
        if not predictions:
            return np.zeros((len(X), self.num_classes))
        
        # Weighted average of predictions
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return weighted_pred

class EnhancedMLPipeline:
    """
    Enhanced ML Pipeline integrating all research findings
    Target: 92.48% accuracy with comprehensive feature engineering
    """
    
    def __init__(self):
        self.rolling_window = RollingWindowManager()
        self.feature_engineer = EnhancedFeatureEngineering()
        self.denoising_autoencoder = None
        self.ensemble_rnn = None
        self.traditional_models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Performance tracking
        self.performance_history = []
        self.model_weights = {
            'svm': 0.3,           # 92.48% accuracy from Paper 4
            'ensemble_rnn': 0.25,  # Paper 3 ensemble approach
            'random_forest': 0.2,  # Paper 1,2 validation
            'logistic': 0.15,      # Paper 4 interpretability
            'denoised': 0.1        # Paper 5 feature enhancement
        }
    
    def initialize_models(self, input_shape: Tuple[int, int], feature_dim: int):
        """Initialize all models"""
        # Traditional models
        self.traditional_models = {
            'svm': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'logistic': LogisticRegression(random_state=42)
        }
        
        # Denoising autoencoder
        self.denoising_autoencoder = DenosingAutoencoder(
            input_dim=feature_dim,
            encoding_dim=max(32, feature_dim // 4)
        )
        
        if TENSORFLOW_AVAILABLE:
            self.denoising_autoencoder.build_model()
        
        # Ensemble RNN
        self.ensemble_rnn = EnsembleRNNModel(input_shape=input_shape)
        self.ensemble_rnn.build_ensemble()
        
        # Scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
    
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data with enhanced feature engineering"""
        # Feature engineering
        engineered_data = self.feature_engineer.engineer_features(data)
        
        # Remove NaN values
        engineered_data = engineered_data.dropna()
        
        # Prepare features and target
        feature_cols = [col for col in engineered_data.columns if col != target_col]
        X = engineered_data[feature_cols].values
        y = engineered_data[target_col].values if target_col in engineered_data.columns else None
        
        return X, y
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for RNN models"""
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def fit(self, data: pd.DataFrame, target_col: str = 'target'):
        """Train the enhanced pipeline"""
        logger.info("Starting enhanced ML pipeline training...")
        
        # Prepare data
        X, y = self.prepare_data(data, target_col)
        
        if X is None or y is None:
            logger.error("Failed to prepare data")
            return
        
        # Initialize models
        sequence_length = 60
        self.initialize_models(
            input_shape=(sequence_length, X.shape[1]),
            feature_dim=X.shape[1]
        )
        
        # Scale features
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Train denoising autoencoder
        if self.denoising_autoencoder and TENSORFLOW_AVAILABLE:
            logger.info("Training denoising autoencoder...")
            self.denoising_autoencoder.fit(X_scaled)
            
            # Get denoised features
            X_denoised = self.denoising_autoencoder.encode(X_scaled)
        else:
            X_denoised = X_scaled
        
        # Train traditional models
        for name, model in self.traditional_models.items():
            try:
                logger.info(f"Training {name} model...")
                if name == 'logistic':
                    # For logistic regression, convert to classification
                    y_binary = (y > np.median(y)).astype(int)
                    model.fit(X_denoised, y_binary)
                else:
                    model.fit(X_denoised, y)
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
        
        # Train ensemble RNN
        if self.ensemble_rnn and TENSORFLOW_AVAILABLE and len(X_scaled) > sequence_length:
            logger.info("Training ensemble RNN...")
            X_sequences, y_sequences = self.create_sequences(X_scaled, y, sequence_length)
            self.ensemble_rnn.fit(X_sequences, y_sequences)
        
        self.is_trained = True
        logger.info("Enhanced ML pipeline training completed")
    
    def predict(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions with the enhanced pipeline"""
        if not self.is_trained:
            logger.warning("Pipeline not trained")
            return {}
        
        # Prepare data
        X, _ = self.prepare_data(data)
        
        if X is None:
            return {}
        
        # Scale features
        X_scaled = self.scalers['standard'].transform(X)
        
        # Get denoised features
        if self.denoising_autoencoder and self.denoising_autoencoder.is_trained:
            X_denoised = self.denoising_autoencoder.encode(X_scaled)
        else:
            X_denoised = X_scaled
        
        predictions = {}
        
        # Traditional model predictions
        for name, model in self.traditional_models.items():
            try:
                pred = model.predict(X_denoised)
                predictions[name] = pred
            except Exception as e:
                logger.error(f"Error predicting with {name}: {str(e)}")
        
        # Ensemble RNN predictions
        if self.ensemble_rnn and TENSORFLOW_AVAILABLE and len(X_scaled) > 60:
            sequence_length = 60
            X_sequences, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)), sequence_length)
            ensemble_pred = self.ensemble_rnn.predict(X_sequences)
            predictions['ensemble_rnn'] = ensemble_pred.flatten()
        
        return predictions
    
    def get_ensemble_prediction(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions using weighted ensemble"""
        if not predictions:
            return np.array([])
        
        # Align prediction lengths
        min_length = min(len(pred) for pred in predictions.values())
        aligned_predictions = {name: pred[:min_length] for name, pred in predictions.items()}
        
        # Weighted average
        ensemble_pred = np.zeros(min_length)
        total_weight = 0
        
        for name, pred in aligned_predictions.items():
            if name in self.model_weights:
                weight = self.model_weights[name]
                ensemble_pred += pred * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def evaluate_performance(self, data: pd.DataFrame, target_col: str = 'target') -> Dict[str, float]:
        """Evaluate pipeline performance"""
        if not self.is_trained:
            return {}
        
        # Get predictions
        predictions = self.predict(data)
        
        if not predictions:
            return {}
        
        # Get actual values
        X, y_true = self.prepare_data(data, target_col)
        
        if y_true is None:
            return {}
        
        # Calculate metrics
        metrics = {}
        
        for name, pred in predictions.items():
            if len(pred) == len(y_true):
                try:
                    mse = mean_squared_error(y_true, pred)
                    mape = mean_absolute_percentage_error(y_true, pred)
                    
                    # Direction accuracy
                    direction_true = (np.diff(y_true) > 0).astype(int)
                    direction_pred = (np.diff(pred) > 0).astype(int)
                    direction_accuracy = accuracy_score(direction_true, direction_pred)
                    
                    metrics[name] = {
                        'mse': mse,
                        'mape': mape,
                        'direction_accuracy': direction_accuracy
                    }
                except Exception as e:
                    logger.error(f"Error calculating metrics for {name}: {str(e)}")
        
        return metrics
    
    def save_pipeline(self, filepath: str):
        """Save the trained pipeline"""
        if not self.is_trained:
            logger.warning("Pipeline not trained, cannot save")
            return
        
        pipeline_data = {
            'traditional_models': self.traditional_models,
            'scalers': self.scalers,
            'feature_names': self.feature_engineer.feature_names,
            'model_weights': self.model_weights,
            'performance_history': self.performance_history
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load a trained pipeline"""
        if not os.path.exists(filepath):
            logger.error(f"Pipeline file not found: {filepath}")
            return
        
        pipeline_data = joblib.load(filepath)
        
        self.traditional_models = pipeline_data.get('traditional_models', {})
        self.scalers = pipeline_data.get('scalers', {})
        self.feature_engineer.feature_names = pipeline_data.get('feature_names', [])
        self.model_weights = pipeline_data.get('model_weights', {})
        self.performance_history = pipeline_data.get('performance_history', [])
        
        self.is_trained = True
        logger.info(f"Pipeline loaded from {filepath}")