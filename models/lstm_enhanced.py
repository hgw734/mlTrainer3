#!/usr/bin/env python3
"""
Enhanced LSTM Model for Financial Time Series Prediction
Deep learning approach using TensorFlow/Keras
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

logger = logging.getLogger(__name__)

@dataclass
class LSTMEnhanced:
    """
    LSTM neural network for financial market prediction
    Uses sequence learning for time series patterns
    """
    
    sequence_length: int = 60  # Days of history to use
    lstm_units: List[int] = None  # Units in each LSTM layer
    dense_units: List[int] = None  # Units in dense layers
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    prediction_type: str = 'regression'  # 'regression' or 'classification'
    n_features: Optional[int] = None  # Auto-determined from data
    
    def __post_init__(self):
        """Initialize internal state"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not installed")
            
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]
        if self.dense_units is None:
            self.dense_units = [16]
            
        self.is_fitted = False
        self.model = None
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = None
        self.history = None
        
    def build_model(self):
        """Build LSTM architecture"""
        model = keras.Sequential()
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            if i == 0:
                model.add(layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    input_shape=(self.sequence_length, self.n_features)
                ))
            else:
                model.add(layers.LSTM(units, return_sequences=return_sequences))
            
            model.add(layers.Dropout(self.dropout_rate))
        
        # Dense layers
        for units in self.dense_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        if self.prediction_type == 'classification':
            model.add(layers.Dense(3, activation='softmax'))  # 3 classes: down, neutral, up
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def fit(self, data: pd.DataFrame, validation_split: float = 0.2) -> 'LSTMEnhanced':
        """
        Fit the LSTM model using historical data
        
        Args:
            data: DataFrame with OHLCV data
            validation_split: Fraction of data for validation
            
        Returns:
            Self for chaining
        """
        # Prepare sequences
        X, y, feature_names = self._prepare_sequences(data)
        self.feature_names = feature_names
        self.n_features = X.shape[2]
        
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        
        # Scale target if regression
        if self.prediction_type == 'regression':
            y_scaled = self.scaler_target.fit_transform(y.reshape(-1, 1))
        else:
            # Convert to categorical for classification
            y_scaled = keras.utils.to_categorical(y + 1, num_classes=3)  # -1,0,1 -> 0,1,2
        
        # Split data (temporal split)
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train = X_scaled[:split_idx]
        y_train = y_scaled[:split_idx]
        X_val = X_scaled[split_idx:]
        y_val = y_scaled[split_idx:]
        
        # Build model
        self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info(f"LSTM fitted with {len(X_train)} samples, "
                   f"{self.n_features} features, sequence length {self.sequence_length}")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using LSTM
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of signals: 1 (buy), 0 (hold), -1 (sell)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare sequences
        X = self._prepare_prediction_sequences(data)
        X_scaled = self._scale_features(X, fit=False)
        
        # Get predictions
        predictions = self.model.predict(X_scaled, verbose=0)
        
        if self.prediction_type == 'classification':
            # Get class with highest probability
            predicted_classes = np.argmax(predictions, axis=1) - 1  # 0,1,2 -> -1,0,1
            signals = pd.Series(
                predicted_classes,
                index=data.index[self.sequence_length:]
            )
        else:
            # Inverse transform predictions
            predictions_original = self.scaler_target.inverse_transform(predictions)
            
            # Convert to signals based on predicted returns
            signals = pd.Series(0, index=data.index[self.sequence_length:])
            signals[predictions_original.flatten() > 0.001] = 1
            signals[predictions_original.flatten() < -0.001] = -1
        
        return signals.astype(int)
    
    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get prediction probabilities (classification only)
        
        Returns DataFrame with probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.prediction_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        X = self._prepare_prediction_sequences(data)
        X_scaled = self._scale_features(X, fit=False)
        
        proba = self.model.predict(X_scaled, verbose=0)
        
        # Create DataFrame with class probabilities
        proba_df = pd.DataFrame(
            proba,
            columns=['prob_sell', 'prob_hold', 'prob_buy'],
            index=data.index[self.sequence_length:]
        )
        
        return proba_df
    
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare sequences for training"""
        # Calculate features
        features_dict = self._calculate_features(data)
        
        # Create feature matrix
        feature_names = []
        feature_arrays = []
        
        for name, values in features_dict.items():
            feature_names.append(name)
            feature_arrays.append(values.values)
        
        features = np.column_stack(feature_arrays)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.sequence_length, len(features) - 1):
            X.append(features[i-self.sequence_length:i])
            
            # Target is next day return
            if self.prediction_type == 'regression':
                next_return = data['close'].pct_change().iloc[i+1]
                y.append(next_return)
            else:
                # Classification: -1, 0, 1
                next_return = data['close'].pct_change().iloc[i+1]
                if next_return > 0.001:
                    y.append(1)
                elif next_return < -0.001:
                    y.append(-1)
                else:
                    y.append(0)
        
        return np.array(X), np.array(y), feature_names
    
    def _prepare_prediction_sequences(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare sequences for prediction"""
        features_dict = self._calculate_features(data)
        
        # Use same features as training
        feature_arrays = []
        for name in self.feature_names:
            if name in features_dict:
                feature_arrays.append(features_dict[name].values)
            else:
                # Handle missing features
                feature_arrays.append(np.zeros(len(data)))
        
        features = np.column_stack(feature_arrays)
        
        # Create sequences
        X = []
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
        
        return np.array(X)
    
    def _calculate_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate features for LSTM"""
        features = {}
        
        # Price features (normalized)
        features['close_norm'] = data['close'] / data['close'].shift(1) - 1
        features['high_norm'] = data['high'] / data['close'] - 1
        features['low_norm'] = data['low'] / data['close'] - 1
        features['open_norm'] = data['open'] / data['close'] - 1
        
        # Returns at different scales
        for period in [1, 5, 10, 20]:
            features[f'return_{period}d'] = data['close'].pct_change(period)
        
        # Moving average features
        for period in [10, 20, 50]:
            if len(data) >= period:
                ma = data['close'].rolling(period).mean()
                features[f'ma_{period}_ratio'] = data['close'] / ma - 1
        
        # Volatility
        returns = data['close'].pct_change()
        for period in [5, 10, 20]:
            if len(data) >= period:
                features[f'volatility_{period}d'] = returns.rolling(period).std()
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_norm'] = data['volume'] / data['volume'].rolling(20).mean() - 1
            features['volume_change'] = data['volume'].pct_change()
        
        # Technical indicators
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        features['rsi'] = (100 - (100 / (1 + rs))) / 100  # Normalized to 0-1
        
        # MACD
        ema_12 = data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = (ema_12 - ema_26) / data['close']
        
        # Bollinger Bands position
        ma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        features['bb_position'] = (data['close'] - ma_20) / (2 * std_20)
        
        # Fill NaN values
        for name in features:
            features[name] = features[name].fillna(0)
        
        return features
    
    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features to 0-1 range"""
        # Reshape for scaling
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        if fit:
            X_scaled = self.scaler_features.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler_features.transform(X_reshaped)
        
        # Reshape back
        return X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        params = {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'prediction_type': self.prediction_type,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            params['n_features'] = self.n_features
            params['total_parameters'] = self.model.count_params()
            
            # Training history
            if self.history:
                params['final_loss'] = self.history.history['loss'][-1]
                params['final_val_loss'] = self.history.history['val_loss'][-1]
                params['epochs_trained'] = len(self.history.history['loss'])
        
        return params
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        X = self._prepare_prediction_sequences(data)
        X_scaled = self._scale_features(X, fit=False)
        
        # Get true values
        y_true = data['close'].pct_change().shift(-1).iloc[self.sequence_length:-1]
        
        if self.prediction_type == 'regression':
            y_true_scaled = self.scaler_target.transform(y_true.values.reshape(-1, 1))
            loss, mae = self.model.evaluate(X_scaled, y_true_scaled, verbose=0)
            
            # Get predictions for additional metrics
            predictions = self.model.predict(X_scaled, verbose=0)
            predictions_original = self.scaler_target.inverse_transform(predictions).flatten()
            
            # Calculate R²
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true.values, predictions_original)
            
            return {
                'loss': loss,
                'mae': mae,
                'r2_score': r2
            }
        else:
            # Convert to categorical
            y_categorical = np.zeros(len(y_true))
            y_categorical[y_true > 0.001] = 2
            y_categorical[y_true < -0.001] = 0
            y_categorical[(y_true >= -0.001) & (y_true <= 0.001)] = 1
            y_categorical = keras.utils.to_categorical(y_categorical, num_classes=3)
            
            loss, accuracy = self.model.evaluate(X_scaled, y_categorical, verbose=0)
            
            return {
                'loss': loss,
                'accuracy': accuracy
            }
    
    def save_model(self, filepath: str):
        """Save the fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Save Keras model
        model_path = filepath.replace('.pkl', '_model.h5')
        self.model.save(model_path)
        
        # Save other components
        import joblib
        model_data = {
            'model_path': model_path,
            'scaler_features': self.scaler_features,
            'scaler_target': self.scaler_target,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'parameters': self.get_parameters()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted model"""
        import joblib
        model_data = joblib.load(filepath)
        
        # Load Keras model
        self.model = keras.models.load_model(model_data['model_path'])
        
        self.scaler_features = model_data['scaler_features']
        self.scaler_target = model_data['scaler_target']
        self.feature_names = model_data['feature_names']
        self.n_features = model_data['n_features']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")