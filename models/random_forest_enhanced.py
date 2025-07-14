#!/usr/bin/env python3
"""
Enhanced Random Forest Model for Financial Prediction
Uses scikit-learn with real market features
NO MOCK DATA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

logger = logging.getLogger(__name__)

@dataclass
class RandomForestEnhanced:
    """
    Enhanced Random Forest for financial market prediction
    Supports both classification (direction) and regression (returns)
    """
    
    n_estimators: int = 100
    max_depth: Optional[int] = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = 'sqrt'
    prediction_type: str = 'classification'  # 'classification' or 'regression'
    target_threshold: float = 0.001  # 0.1% for classification
    random_state: int = 42
    
    def __post_init__(self):
        """Initialize internal state"""
        self.is_fitted = False
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance_ = None
        
    def fit(self, data: pd.DataFrame) -> 'RandomForestEnhanced':
        """
        Fit the Random Forest model using historical data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Self for chaining
        """
        # Prepare features and target
        X, y, feature_names = self._prepare_training_data(data)
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model based on prediction type
        if self.prediction_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Fit the model
        self.model.fit(X_scaled, y)
        
        # Store feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        logger.info(f"Random Forest fitted with {len(X)} samples and {len(feature_names)} features")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using Random Forest
        
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
        
        if self.prediction_type == 'classification':
            # Direct prediction of direction
            predictions = self.model.predict(X_scaled)
            signals = pd.Series(predictions, index=data.index[len(data) - len(X):])
        else:
            # Regression: convert return predictions to signals
            return_predictions = self.model.predict(X_scaled)
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
        
        proba = self.model.predict_proba(X_scaled)
        
        # Create DataFrame with class probabilities
        classes = self.model.classes_
        proba_df = pd.DataFrame(
            proba,
            columns=[f'prob_class_{c}' for c in classes],
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
            y = np.zeros(len(returns))
            y[returns > self.target_threshold] = 1
            y[returns < -self.target_threshold] = -1
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
        features['returns_5d'] = data['close'].pct_change(5)
        features['returns_20d'] = data['close'].pct_change(20)
        
        # Intraday features
        features['intraday_return'] = (data['close'] - data['open']) / data['open']
        features['daily_range'] = (data['high'] - data['low']) / data['close']
        features['upper_shadow'] = (data['high'] - data['close']) / data['close']
        features['lower_shadow'] = (data['close'] - data['low']) / data['close']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(data) >= period:
                ma = data['close'].rolling(period).mean()
                features[f'ma_{period}_ratio'] = data['close'] / ma
                features[f'ma_{period}_slope'] = ma.pct_change()
        
        # Volatility features
        returns = data['close'].pct_change()
        for period in [5, 10, 20]:
            if len(data) >= period:
                features[f'volatility_{period}d'] = returns.rolling(period).std()
                features[f'volatility_ratio_{period}d'] = (
                    returns.rolling(period).std() / returns.rolling(period * 2).std()
                )
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            features['volume_trend'] = data['volume'].rolling(5).mean().pct_change()
            
            # Price-volume correlation
            features['price_volume_corr'] = (
                data['close'].pct_change().rolling(20)
                .corr(data['volume'].pct_change())
            )
        
        # Technical indicators
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = (ema_12 - ema_26) / data['close']
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        ma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        features['bb_upper_ratio'] = (data['close'] - (ma_20 + 2*std_20)) / data['close']
        features['bb_lower_ratio'] = (data['close'] - (ma_20 - 2*std_20)) / data['close']
        features['bb_width'] = (4 * std_20) / ma_20
        
        # Market microstructure
        features['high_low_spread'] = (data['high'] - data['low']) / data['low']
        features['close_to_high'] = (data['high'] - data['close']) / (data['high'] - data['low'])
        
        # Time-based features (if index is datetime)
        if isinstance(data.index, pd.DatetimeIndex):
            features['day_of_week'] = data.index.dayofweek
            features['day_of_month'] = data.index.day
            features['month'] = data.index.month
        
        return features
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'prediction_type': self.prediction_type,
            'target_threshold': self.target_threshold,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            params['n_features'] = len(self.feature_names)
            params['feature_importance_top5'] = (
                self.feature_importance_.head(5).to_dict('records')
            )
        
        return params
    
    def cross_validate(self, data: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform time series cross-validation
        
        Args:
            data: Historical data
            n_splits: Number of CV splits
            
        Returns:
            Dictionary with CV results
        """
        X, y, feature_names = self._prepare_training_data(data)
        X_scaled = self.scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train temporary model
            if self.prediction_type == 'classification':
                temp_model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                temp_model.fit(X_train, y_train)
                score = temp_model.score(X_test, y_test)
            else:
                temp_model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                temp_model.fit(X_train, y_train)
                score = temp_model.score(X_test, y_test)
            
            scores.append(score)
        
        return {
            'cv_scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_splits': n_splits
        }
    
    def save_model(self, filepath: str):
        """Save the fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_,
            'parameters': self.get_parameters()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance_ = model_data['feature_importance']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")