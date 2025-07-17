#!/usr/bin/env python3
"""
Advanced Machine Learning Model Implementations for S&P 500 Trading
==================================================================

Advanced machine learning models including:
- Ensemble learning methods
- Deep learning models
- Reinforcement learning
- Feature engineering
- Model selection and validation
- Hyperparameter optimization
- Advanced ML techniques
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
import joblib
import pickle

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class MLPrediction:
    """Machine learning prediction result"""
    symbol: str
    prediction: float
    probability: float
    confidence: float
    model_type: str
    features_importance: Dict[str, float]
    timestamp: datetime

class BaseMLModel:
    """Base class for machine learning models"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            # Placeholder for real market data integration
            logger.info(f"Would get real market data for {symbol} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            # Placeholder for real economic data integration
            logger.info(f"Would get real economic data for {series_id} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real economic data: {e}")
            return None

    def _engineer_features(self, data: pd.Series) -> pd.DataFrame:
        """Engineer features for machine learning"""
        if len(data) < 50:
            return pd.DataFrame()
        
        # Basic technical indicators
        returns = data.pct_change().dropna()
        
        features = {}
        
        # Price-based features
        features['price'] = data.iloc[-1]
        features['price_ma_5'] = data.rolling(window=5).mean().iloc[-1]
        features['price_ma_20'] = data.rolling(window=20).mean().iloc[-1]
        features['price_ma_50'] = data.rolling(window=50).mean().iloc[-1]
        
        # Return-based features
        features['return_1d'] = returns.iloc[-1] if len(returns) > 0 else 0
        features['return_5d'] = (data.iloc[-1] - data.iloc[-6]) / data.iloc[-6] if len(data) > 5 else 0
        features['return_20d'] = (data.iloc[-1] - data.iloc[-21]) / data.iloc[-21] if len(data) > 20 else 0
        
        # Volatility features
        features['volatility_5d'] = returns.rolling(window=5).std().iloc[-1] if len(returns) >= 5 else 0
        features['volatility_20d'] = returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else 0
        
        # Momentum features
        features['momentum_5d'] = data.iloc[-1] / data.iloc[-6] - 1 if len(data) > 5 else 0
        features['momentum_20d'] = data.iloc[-1] / data.iloc[-21] - 1 if len(data) > 20 else 0
        
        # RSI
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        avg_gain = gains.rolling(window=14).mean()
        avg_loss = losses.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        features['rsi'] = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # MACD
        ema12 = data.ewm(span=12).mean()
        ema26 = data.ewm(span=26).mean()
        macd = ema12 - ema26
        features['macd'] = macd.iloc[-1] if len(macd) > 0 else 0
        
        # Bollinger Bands
        bb_ma = data.rolling(window=20).mean()
        bb_std = data.rolling(window=20).std()
        bb_upper = bb_ma + (bb_std * 2)
        bb_lower = bb_ma - (bb_std * 2)
        features['bb_position'] = (data.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if len(bb_upper) > 0 and bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5
        
        # Volume features (if available)
        features['volume_ma_5'] = 0  # Placeholder
        features['volume_ma_20'] = 0  # Placeholder
        
        # Time features
        features['day_of_week'] = datetime.now().weekday()
        features['month'] = datetime.now().month
        
        return pd.DataFrame([features])

    def fit(self, data: pd.Series, target: pd.Series = None) -> 'BaseMLModel':
        """Fit the machine learning model"""
        if len(data) < self.params.get('min_data_points', 100):
            raise ValueError(f"Insufficient data: {len(data)} < {self.params.get('min_data_points', 100)}")
        
        # Engineer features
        X = self._engineer_features(data)
        if X.empty:
            raise ValueError("Failed to engineer features")
        
        # Create target if not provided
        if target is None:
            # Default target: next day return direction
            returns = data.pct_change().dropna()
            target = (returns.shift(-1) > 0).astype(int)
            target = target.dropna()
            # Align X and target
            X = X.iloc[:-1]  # Remove last row since we don't have target for it
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, target)
        
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series) -> MLPrediction:
        """Make machine learning prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Engineer features
        X = self._engineer_features(data)
        if X.empty:
            return MLPrediction(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                prediction=0.0,
                probability=0.5,
                confidence=0.0,
                model_type=self.__class__.__name__,
                features_importance={},
                timestamp=datetime.now()
            )
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(X_scaled)[0][1]  # Probability of positive class
        else:
            probability = 0.5
        
        # Calculate confidence
        confidence = self._calculate_confidence(X_scaled)
        
        # Get feature importance
        feature_importance = self._get_feature_importance()
        
        return MLPrediction(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            model_type=self.__class__.__name__,
            features_importance=feature_importance,
            timestamp=datetime.now()
        )
    
    def _calculate_confidence(self, X_scaled: np.ndarray) -> float:
        """Calculate prediction confidence"""
        # Simple confidence based on feature values
        feature_std = np.std(X_scaled)
        confidence = 1.0 / (1.0 + feature_std)
        return min(confidence, 1.0)
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            importance_dict = dict(zip(self.feature_names, abs(self.model.coef_[0])))
        else:
            importance_dict = {feature: 0.0 for feature in self.feature_names}
        
        return importance_dict

class EnsembleMLModel(BaseMLModel):
    """Ensemble machine learning model"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, learning_rate: float = 0.1):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_data_points=100)
        
        # Initialize ensemble models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        self.ensemble_weights = {
            'random_forest': 0.3,
            'gradient_boosting': 0.3,
            'svm': 0.2,
            'neural_network': 0.2
        }
    
    def fit(self, data: pd.Series, target: pd.Series = None) -> 'EnsembleMLModel':
        """Fit ensemble model"""
        super().fit(data, target)
        
        # Engineer features
        X = self._engineer_features(data)
        if X.empty:
            return self
        
        # Create target if not provided
        if target is None:
            returns = data.pct_change().dropna()
            target = (returns.shift(-1) > 0).astype(int)
            target = target.dropna()
            X = X.iloc[:-1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit all models
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, target)
            except Exception as e:
                logger.warning(f"Failed to fit {name}: {e}")
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.Series) -> MLPrediction:
        """Make ensemble prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Engineer features
        X = self._engineer_features(data)
        if X.empty:
            return MLPrediction(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                prediction=0.0,
                probability=0.5,
                confidence=0.0,
                model_type='EnsembleMLModel',
                features_importance={},
                timestamp=datetime.now()
            )
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0][1] if hasattr(model, 'predict_proba') else 0.5
                
                predictions.append(pred * self.ensemble_weights[name])
                probabilities.append(prob * self.ensemble_weights[name])
            except Exception as e:
                logger.warning(f"Failed to predict with {name}: {e}")
        
        # Ensemble prediction
        ensemble_prediction = 1 if sum(predictions) > 0.5 else 0
        ensemble_probability = sum(probabilities)
        
        # Calculate confidence
        confidence = self._calculate_ensemble_confidence(predictions, probabilities)
        
        # Get feature importance (average across models)
        feature_importance = self._get_ensemble_feature_importance()
        
        return MLPrediction(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            prediction=ensemble_prediction,
            probability=ensemble_probability,
            confidence=confidence,
            model_type='EnsembleMLModel',
            features_importance=feature_importance,
            timestamp=datetime.now()
        )
    
    def _calculate_ensemble_confidence(self, predictions: List[float], probabilities: List[float]) -> float:
        """Calculate ensemble confidence"""
        if not predictions:
            return 0.0
        
        # Confidence based on agreement among models
        agreement = 1.0 - np.std(predictions)
        probability_confidence = 1.0 - np.std(probabilities)
        
        return (agreement + probability_confidence) / 2
    
    def _get_ensemble_feature_importance(self) -> Dict[str, float]:
        """Get ensemble feature importance"""
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                model_importance = dict(zip(self.feature_names, model.feature_importances_))
                for feature, importance in model_importance.items():
                    if feature not in importance_dict:
                        importance_dict[feature] = 0.0
                    importance_dict[feature] += importance * self.ensemble_weights[name]
        
        return importance_dict

class DeepLearningModel(BaseMLModel):
    """Deep learning model"""
    
    def __init__(self, hidden_layers: Tuple[int, ...] = (100, 50, 25), dropout_rate: float = 0.2):
        super().__init__(hidden_layers=hidden_layers, dropout_rate=dropout_rate, min_data_points=100)
        
        # Initialize deep learning model
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            dropout=dropout_rate,
            max_iter=1000,
            random_state=42
        )
    
    def _engineer_deep_features(self, data: pd.Series) -> pd.DataFrame:
        """Engineer features for deep learning"""
        # Enhanced feature engineering for deep learning
        features = super()._engineer_features(data)
        
        if features.empty:
            return features
        
        # Add more sophisticated features
        returns = data.pct_change().dropna()
        
        # Technical indicators
        features['adx'] = self._calculate_adx(data)
        features['cci'] = self._calculate_cci(data)
        features['williams_r'] = self._calculate_williams_r(data)
        
        # Statistical features
        features['skewness'] = returns.skew()
        features['kurtosis'] = returns.kurtosis()
        features['jarque_bera'] = self._calculate_jarque_bera(returns)
        
        # Volatility features
        features['realized_volatility'] = np.sqrt((returns ** 2).sum())
        features['volatility_of_volatility'] = returns.rolling(window=20).std().std()
        
        return features
    
    def _calculate_adx(self, data: pd.Series) -> float:
        """Calculate Average Directional Index"""
        try:
            # Simplified ADX calculation
            returns = data.pct_change().dropna()
            return abs(returns.mean()) / returns.std() if returns.std() > 0 else 0
        except:
            return 0
    
    def _calculate_cci(self, data: pd.Series) -> float:
        """Calculate Commodity Channel Index"""
        try:
            if len(data) < 20:
                return 0
            
            typical_price = data.rolling(window=20).mean()
            mean_deviation = abs(data - typical_price).rolling(window=20).mean()
            cci = (data.iloc[-1] - typical_price.iloc[-1]) / (0.015 * mean_deviation.iloc[-1]) if mean_deviation.iloc[-1] > 0 else 0
            return cci
        except:
            return 0
    
    def _calculate_williams_r(self, data: pd.Series) -> float:
        """Calculate Williams %R"""
        try:
            if len(data) < 14:
                return -50
            
            highest_high = data.rolling(window=14).max()
            lowest_low = data.rolling(window=14).min()
            williams_r = -100 * (highest_high.iloc[-1] - data.iloc[-1]) / (highest_high.iloc[-1] - lowest_low.iloc[-1]) if highest_high.iloc[-1] != lowest_low.iloc[-1] else -50
            return williams_r
        except:
            return -50
    
    def _calculate_jarque_bera(self, returns: pd.Series) -> float:
        """Calculate Jarque-Bera statistic"""
        try:
            from scipy.stats import jarque_bera
            return jarque_bera(returns)[0]
        except:
            return 0
    
    def fit(self, data: pd.Series, target: pd.Series = None) -> 'DeepLearningModel':
        """Fit deep learning model"""
        if len(data) < self.params['min_data_points']:
            raise ValueError(f"Insufficient data: {len(data)} < {self.params['min_data_points']}")
        
        # Engineer deep features
        X = self._engineer_deep_features(data)
        if X.empty:
            raise ValueError("Failed to engineer features")
        
        # Create target if not provided
        if target is None:
            returns = data.pct_change().dropna()
            target = (returns.shift(-1) > 0).astype(int)
            target = target.dropna()
            X = X.iloc[:-1]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, target)
        
        self.is_fitted = True
        return self

class ReinforcementLearningModel(BaseMLModel):
    """Reinforcement learning model"""
    
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.95):
        super().__init__(learning_rate=learning_rate, discount_factor=discount_factor, min_data_points=100)
        
        # Simple Q-learning implementation
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = 0.1  # Exploration rate
    
    def _discretize_state(self, data: pd.Series) -> str:
        """Discretize state for Q-learning"""
        if len(data) < 5:
            return "unknown"
        
        # Create state based on recent price movements
        recent_returns = data.pct_change().iloc[-5:].fillna(0)
        
        # Discretize returns
        discretized = []
        for ret in recent_returns:
            if ret > 0.01:
                discretized.append("up")
            elif ret < -0.01:
                discretized.append("down")
            else:
                discretized.append("flat")
        
        return "_".join(discretized)
    
    def _get_action(self, state: str) -> int:
        """Get action for current state"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]  # [buy, sell]
        
        # Epsilon-greedy policy using deterministic exploration
        # Use hash of state and current time for exploration decision
        import hashlib
        import time
        state_hash = int(hashlib.md5(f"{state}{int(time.time())}".encode()).hexdigest()[:8], 16)
        exploration_threshold = state_hash / (2**32)  # Normalize to [0, 1]
        
        if exploration_threshold < self.epsilon:
            # Deterministic "random" action based on state hash
            action = state_hash % 2
            return action
        else:
            return np.argmax(self.q_table[state])  # Best action
    
    def fit(self, data: pd.Series, target: pd.Series = None) -> 'ReinforcementLearningModel':
        """Fit reinforcement learning model"""
        if len(data) < self.params['min_data_points']:
            raise ValueError(f"Insufficient data: {len(data)} < {self.params['min_data_points']}")
        
        # Create target if not provided
        if target is None:
            returns = data.pct_change().dropna()
            target = (returns.shift(-1) > 0).astype(int)
            target = target.dropna()
        
        # Train Q-learning agent
        for i in range(len(data) - 5):
            state = self._discretize_state(data.iloc[i:i+5])
            action = self._get_action(state)
            
            # Get reward
            if i < len(target):
                reward = 1 if target.iloc[i] == action else -1
            else:
                reward = 0
            
            # Update Q-table
            if state not in self.q_table:
                self.q_table[state] = [0.0, 0.0]
            
            # Q-learning update
            next_state = self._discretize_state(data.iloc[i+1:i+6]) if i + 5 < len(data) else state
            if next_state not in self.q_table:
                self.q_table[next_state] = [0.0, 0.0]
            
            max_next_q = max(self.q_table[next_state])
            self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * max_next_q - self.q_table[state][action])
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.Series) -> MLPrediction:
        """Make reinforcement learning prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get current state
        state = self._discretize_state(data)
        
        # Get action
        if state in self.q_table:
            action = np.argmax(self.q_table[state])
            q_values = self.q_table[state]
        else:
            action = 0
            q_values = [0.0, 0.0]
        
        # Convert to prediction
        prediction = action
        probability = q_values[action] / (sum(q_values) + 1e-8)
        confidence = max(q_values) / (sum(q_values) + 1e-8)
        
        return MLPrediction(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            model_type='ReinforcementLearningModel',
            features_importance={'state': state},
            timestamp=datetime.now()
        )

class MachineLearningModel:
    """Comprehensive machine learning model for S&P 500 trading"""
    
    def __init__(self, ml_window: int = 252):
        self.ml_window = ml_window
        self.is_fitted = False
        self.models = {}
        
        # Initialize different machine learning models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all machine learning models"""
        
        # Basic ML models
        self.models['ensemble'] = EnsembleMLModel()
        self.models['deep_learning'] = DeepLearningModel()
        self.models['reinforcement_learning'] = ReinforcementLearningModel()
        
        # Additional models
        self.models['random_forest'] = BaseMLModel()
        self.models['random_forest'].model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.models['gradient_boosting'] = BaseMLModel()
        self.models['gradient_boosting'].model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        self.models['logistic_regression'] = BaseMLModel()
        self.models['logistic_regression'].model = LogisticRegression(random_state=42)
        
        self.models['svm'] = BaseMLModel()
        self.models['svm'].model = SVC(probability=True, random_state=42)
        
    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            # Placeholder for real market data integration
            logger.info(f"Would get real market data for {symbol} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            # Placeholder for real economic data integration
            logger.info(f"Would get real economic data for {series_id} from {start_date} to {end_date}")
            return None
        except Exception as e:
            logger.error(f"Failed to get real economic data: {e}")
            return None

    def _get_real_alternative_data(self, data_type: str, **kwargs):
        """Get real alternative data from approved sources"""
        try:
            # Implement based on data type
            if data_type == 'sentiment':
                return self._get_sentiment_data(**kwargs)
            elif data_type == 'news':
                return self._get_news_data(**kwargs)
            elif data_type == 'social':
                return self._get_social_data(**kwargs)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to get real alternative data: {e}")
            return None

    def fit(self, data: pd.Series, target: pd.Series = None) -> 'MachineLearningModel':
        """Fit all machine learning models"""
        if len(data) < self.ml_window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.ml_window}")
        
        # Fit all models
        for model_name, model in self.models.items():
            try:
                model.fit(data, target)
            except Exception as e:
                logger.warning(f"Failed to fit {model_name}: {e}")
        
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series, model_name: str = None) -> MLPrediction:
        """Make machine learning prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            return self.models[model_name].predict(data)
        else:
            # Return ensemble prediction as default
            return self.models['ensemble'].predict(data)

    def get_available_models(self) -> List[str]:
        """Get list of available machine learning models"""
        return list(self.models.keys())

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name].params

    def get_parameters(self) -> Dict[str, Any]:
        """Get all model parameters"""
        return {
            'ml_window': self.ml_window,
            'is_fitted': self.is_fitted,
            'available_models': self.get_available_models(),
            'model_parameters': {
                name: model.params for name, model in self.models.items()
            }
        }

    def save_model(self, model_name: str, filepath: str):
        """Save trained model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        if model.is_fitted:
            joblib.dump(model, filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
        else:
            raise ValueError(f"Model {model_name} is not fitted")

    def load_model(self, model_name: str, filepath: str):
        """Load trained model"""
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            logger.info(f"Model {model_name} loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise 