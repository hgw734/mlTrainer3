#!/usr/bin/env python3
"""
Enhanced XGBoost Model for Financial Prediction
High-performance gradient boosting with real market features
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

logger = logging.getLogger(__name__)

@dataclass
class XGBoostEnhanced:
    """
    Enhanced XGBoost for financial market prediction
    Optimized for time series with proper validation
    """
    
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    objective: str = 'multi:softprob'  # For classification
    prediction_type: str = 'classification'
    target_threshold: float = 0.001
    random_state: int = 42
    early_stopping_rounds: int = 10
    
    def __post_init__(self):
        """Initialize internal state"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required but not installed")
            
        self.is_fitted = False
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance_ = None
        self.best_iteration = None
        
    def fit(self, data: pd.DataFrame, validation_split: float = 0.2) -> 'XGBoostEnhanced':
        """
        Fit the XGBoost model using historical data
        
        Args:
            data: DataFrame with OHLCV data
            validation_split: Fraction of data for validation
            
        Returns:
            Self for chaining
        """
        # Prepare features and target
        X, y, feature_names = self._prepare_training_data(data)
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation (temporal split)
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train = X_scaled[:split_idx]
        y_train = y[:split_idx]
        X_val = X_scaled[split_idx:]
        y_val = y[split_idx:]
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        # Set parameters based on prediction type
        if self.prediction_type == 'classification':
            num_classes = len(np.unique(y))
            params = {
                'objective': self.objective,
                'num_class': num_classes,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'min_child_weight': self.min_child_weight,
                'gamma': self.gamma,
                'alpha': self.reg_alpha,
                'lambda': self.reg_lambda,
                'random_state': self.random_state,
                'eval_metric': 'mlogloss',
                'tree_method': 'hist',  # Fast histogram method
                'device': 'cpu'
            }
        else:
            params = {
                'objective': 'reg:squarederror',
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'min_child_weight': self.min_child_weight,
                'gamma': self.gamma,
                'alpha': self.reg_alpha,
                'lambda': self.reg_lambda,
                'random_state': self.random_state,
                'eval_metric': 'rmse',
                'tree_method': 'hist',
                'device': 'cpu'
            }
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        
        self.best_iteration = self.model.best_iteration
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        self.feature_importance_ = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        logger.info(f"XGBoost fitted with {len(X)} samples, {len(feature_names)} features. "
                   f"Best iteration: {self.best_iteration}")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using XGBoost
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of signals: 1 (buy), 0 (hold), -1 (sell)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X = self._prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        # Create DMatrix
        dtest = xgb.DMatrix(X_scaled, feature_names=self.feature_names)
        
        if self.prediction_type == 'classification':
            # Get predictions
            predictions = self.model.predict(dtest)
            
            # For multi-class, get the class with highest probability
            if len(predictions.shape) > 1:
                predicted_classes = np.argmax(predictions, axis=1)
                # Map back to -1, 0, 1
                class_mapping = {0: -1, 1: 0, 2: 1}
                predicted_classes = np.array([class_mapping.get(c, 0) for c in predicted_classes])
            else:
                predicted_classes = predictions
            
            signals = pd.Series(predicted_classes, index=data.index[len(data) - len(X):])
        else:
            # Regression: convert return predictions to signals
            return_predictions = self.model.predict(dtest)
            signals = pd.Series(0, index=data.index[len(data) - len(X):])
            signals[return_predictions > self.target_threshold] = 1
            signals[return_predictions < -self.target_threshold] = -1
        
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
        
        X = self._prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        dtest = xgb.DMatrix(X_scaled, feature_names=self.feature_names)
        proba = self.model.predict(dtest)
        
        # Create DataFrame with class probabilities
        proba_df = pd.DataFrame(
            proba,
            columns=['prob_sell', 'prob_hold', 'prob_buy'],
            index=data.index[len(data) - len(X):]
        )
        
        return proba_df
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """Prepare features and target for training"""
        # Calculate all features
        features_dict = self._calculate_all_features(data)
        
        # Create feature matrix
        feature_names = []
        feature_arrays = []
        
        for feature_name, feature_values in features_dict.items():
            if len(feature_values) > 0:
                feature_names.append(feature_name)
                feature_arrays.append(feature_values.values)
        
        X = np.column_stack(feature_arrays)
        
        # Create target
        if self.prediction_type == 'classification':
            # Classify next day return direction
            returns = data['close'].pct_change().shift(-1)
            y = np.ones(len(returns))  # Default to hold (1)
            y[returns > self.target_threshold] = 2  # Buy
            y[returns < -self.target_threshold] = 0  # Sell
        else:
            # Predict actual returns
            y = data['close'].pct_change().shift(-1).values
        
        # Remove NaN rows
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        
        return X[mask], y[mask], feature_names
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction"""
        features_dict = self._calculate_all_features(data)
        
        # Use same features as training
        feature_arrays = []
        for feature_name in self.feature_names:
            if feature_name in features_dict:
                feature_arrays.append(features_dict[feature_name].values)
            else:
                # Handle missing features with zeros
                feature_arrays.append(np.zeros(len(data)))
        
        X = np.column_stack(feature_arrays)
        
        # Remove NaN rows
        mask = ~np.isnan(X).any(axis=1)
        
        return X[mask]
    
    def _calculate_all_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate comprehensive feature set"""
        features = {}
        
        # Price-based features
        features['returns_1d'] = data['close'].pct_change()
        features['returns_2d'] = data['close'].pct_change(2)
        features['returns_5d'] = data['close'].pct_change(5)
        features['returns_10d'] = data['close'].pct_change(10)
        features['returns_20d'] = data['close'].pct_change(20)
        
        # Log returns for better distribution
        features['log_returns_1d'] = np.log(data['close'] / data['close'].shift(1))
        features['log_returns_5d'] = np.log(data['close'] / data['close'].shift(5))
        
        # Intraday features
        features['intraday_return'] = (data['close'] - data['open']) / data['open']
        features['daily_range'] = (data['high'] - data['low']) / data['close']
        features['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
        features['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']
        features['body_size'] = abs(data['close'] - data['open']) / data['close']
        
        # Moving averages and slopes
        for period in [5, 10, 20, 50, 100]:
            if len(data) >= period:
                ma = data['close'].rolling(period).mean()
                features[f'ma_{period}_ratio'] = data['close'] / ma
                features[f'ma_{period}_slope'] = ma.pct_change(5)  # 5-day slope
                
                # Moving average crossovers
                if period > 5:
                    ma_short = data['close'].rolling(5).mean()
                    features[f'ma_cross_5_{period}'] = (ma_short - ma) / ma
        
        # Volatility features
        returns = data['close'].pct_change()
        for period in [5, 10, 20, 60]:
            if len(data) >= period:
                features[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
                features[f'volatility_ratio_{period}d'] = (
                    returns.rolling(period).std() / returns.rolling(period * 2).std()
                )
                
                # Realized volatility ratios
                if period > 5:
                    features[f'vol_ratio_5_{period}'] = (
                        returns.rolling(5).std() / returns.rolling(period).std()
                    )
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            features['volume_trend'] = data['volume'].rolling(5).mean().pct_change(5)
            features['volume_volatility'] = data['volume'].pct_change().rolling(20).std()
            
            # Price-volume features
            features['price_volume_corr'] = (
                data['close'].pct_change().rolling(20)
                .corr(data['volume'].pct_change())
            )
            features['volume_price_trend'] = (
                (data['close'].pct_change() * data['volume'].pct_change()).rolling(10).mean()
            )
        
        # Technical indicators
        # RSI variants
        for period in [9, 14, 21]:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, 1)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = (ema_12 - ema_26) / data['close']
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        for period in [10, 20, 30]:
            ma = data['close'].rolling(period).mean()
            std = data['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = (data['close'] - (ma + 2*std)) / data['close']
            features[f'bb_lower_{period}'] = (data['close'] - (ma - 2*std)) / data['close']
            features[f'bb_width_{period}'] = (4 * std) / ma
            features[f'bb_position_{period}'] = (data['close'] - ma) / (2 * std)
        
        # Market microstructure
        features['high_low_spread'] = (data['high'] - data['low']) / data['low']
        features['close_to_high'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-10)
        features['close_to_low'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-10)
        
        # Momentum indicators
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            features[f'roc_{period}'] = data['close'].pct_change(period)
        
        # Time-based features
        if isinstance(data.index, pd.DatetimeIndex):
            features['day_of_week'] = data.index.dayofweek
            features['day_of_month'] = data.index.day
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['day_of_year'] = data.index.dayofyear
            
            # Seasonal patterns
            features['is_monday'] = (data.index.dayofweek == 0).astype(int)
            features['is_friday'] = (data.index.dayofweek == 4).astype(int)
            features['is_month_end'] = (data.index.day >= 28).astype(int)
            features['is_month_start'] = (data.index.day <= 3).astype(int)
        
        return features
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'prediction_type': self.prediction_type,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            params['best_iteration'] = self.best_iteration
            params['n_features'] = len(self.feature_names)
            if self.feature_importance_ is not None and len(self.feature_importance_) > 0:
                params['feature_importance_top5'] = (
                    self.feature_importance_.head(5).to_dict('records')
                )
        
        return params
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get detailed feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance = self.model.get_score(importance_type=importance_type)
        
        return pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str):
        """Save the fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Save XGBoost model separately
        xgb_path = filepath.replace('.pkl', '_xgb.json')
        self.model.save_model(xgb_path)
        
        # Save other components
        model_data = {
            'xgb_path': xgb_path,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_,
            'best_iteration': self.best_iteration,
            'parameters': self.get_parameters()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted model"""
        model_data = joblib.load(filepath)
        
        # Load XGBoost model
        xgb_path = model_data['xgb_path']
        self.model = xgb.Booster()
        self.model.load_model(xgb_path)
        
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance_ = model_data['feature_importance']
        self.best_iteration = model_data['best_iteration']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")