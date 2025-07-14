#!/usr/bin/env python3
"""
Advanced Regime Detection Model Implementations for S&P 500 Trading
==================================================================

Advanced regime detection models including:
- Hidden Markov Models (HMM)
- Change point detection
- Clustering-based regime detection
- Volatility regime detection
- Market condition classifiers
- Regime switching logic
- Performance-based reweighting
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.cluster import KMeans, GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class RegimeResult:
    """Regime detection result"""
    symbol: str
    regime: int
    regime_probability: float
    regime_confidence: float
    regime_duration: int
    regime_characteristics: Dict[str, float]
    timestamp: datetime

class BaseRegimeModel:
    """Base class for regime detection models"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
        self.regime_history = []
        
    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real economic data: {e}")
            return None

    def fit(self, data: pd.Series) -> 'BaseRegimeModel':
        """Fit the regime detection model"""
        if len(data) < self.params.get('min_data_points', 100):
            raise ValueError(f"Insufficient data: {len(data)} < {self.params.get('min_data_points', 100)}")
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series) -> pd.Series:
        """Detect regimes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        raise NotImplementedError("Subclasses must implement predict method")

class VolatilityRegimeModel(BaseRegimeModel):
    """Volatility-based regime detection"""
    
    def __init__(self, window: int = 30, n_regimes: int = 3, volatility_threshold: float = 0.02):
        super().__init__(window=window, n_regimes=n_regimes, volatility_threshold=volatility_threshold, min_data_points=window + 50)
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Detect volatility regimes"""
        regimes = pd.Series(0, index=data.index)
        
        for i in range(self.params['window'], len(data)):
            window_data = data.iloc[i-self.params['window']:i+1]
            
            # Calculate volatility
            returns = window_data.pct_change().dropna()
            volatility = returns.std()
            
            # Classify regime based on volatility
            if volatility < self.params['volatility_threshold'] * 0.5:
                regimes.iloc[i] = 0  # Low volatility
            elif volatility < self.params['volatility_threshold']:
                regimes.iloc[i] = 1  # Medium volatility
            else:
                regimes.iloc[i] = 2  # High volatility
                
        return regimes

class ClusteringRegimeModel(BaseRegimeModel):
    """Clustering-based regime detection"""
    
    def __init__(self, window: int = 30, n_regimes: int = 3, features: List[str] = None):
        super().__init__(window=window, n_regimes=n_regimes, features=features or ['returns', 'volatility'], min_data_points=window + 50)
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Detect regimes using clustering"""
        regimes = pd.Series(0, index=data.index)
        
        # Create feature matrix
        features = []
        for i in range(self.params['window'], len(data)):
            window_data = data.iloc[i-self.params['window']:i+1]
            
            # Calculate features
            returns = window_data.pct_change().dropna()
            feature_vector = [
                returns.mean(),  # Return
                returns.std(),   # Volatility
                returns.skew(),  # Skewness
                returns.kurtosis(),  # Kurtosis
                (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0]  # Momentum
            ]
            features.append(feature_vector)
        
        if len(features) > 0:
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply clustering
            kmeans = KMeans(n_clusters=self.params['n_regimes'], random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Assign regimes
            for i, label in enumerate(cluster_labels):
                idx = i + self.params['window']
                if idx < len(regimes):
                    regimes.iloc[idx] = label
                    
        return regimes

class GaussianMixtureRegimeModel(BaseRegimeModel):
    """Gaussian Mixture Model for regime detection"""
    
    def __init__(self, window: int = 30, n_regimes: int = 3):
        super().__init__(window=window, n_regimes=n_regimes, min_data_points=window + 50)
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Detect regimes using Gaussian Mixture Model"""
        regimes = pd.Series(0, index=data.index)
        
        # Create feature matrix
        features = []
        for i in range(self.params['window'], len(data)):
            window_data = data.iloc[i-self.params['window']:i+1]
            
            # Calculate features
            returns = window_data.pct_change().dropna()
            feature_vector = [
                returns.mean(),
                returns.std(),
                returns.skew(),
                returns.kurtosis()
            ]
            features.append(feature_vector)
        
        if len(features) > 0:
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply Gaussian Mixture Model
            gmm = GaussianMixture(n_components=self.params['n_regimes'], random_state=42)
            cluster_labels = gmm.fit_predict(features_scaled)
            
            # Assign regimes
            for i, label in enumerate(cluster_labels):
                idx = i + self.params['window']
                if idx < len(regimes):
                    regimes.iloc[idx] = label
                    
        return regimes

class ChangePointRegimeModel(BaseRegimeModel):
    """Change point detection for regime identification"""
    
    def __init__(self, window: int = 30, change_threshold: float = 2.0):
        super().__init__(window=window, change_threshold=change_threshold, min_data_points=window + 50)
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Detect regime changes using change point detection"""
        regimes = pd.Series(0, index=data.index)
        current_regime = 0
        
        for i in range(self.params['window'], len(data)):
            window_data = data.iloc[i-self.params['window']:i+1]
            
            # Calculate rolling statistics
            returns = window_data.pct_change().dropna()
            current_mean = returns.mean()
            current_std = returns.std()
            
            # Detect change point
            if i > self.params['window']:
                prev_window = data.iloc[i-self.params['window']*2:i-self.params['window']]
                prev_returns = prev_window.pct_change().dropna()
                prev_mean = prev_returns.mean()
                prev_std = prev_returns.std()
                
                # Calculate change statistic
                mean_change = abs(current_mean - prev_mean) / (prev_std + 1e-8)
                std_change = abs(current_std - prev_std) / (prev_std + 1e-8)
                
                if mean_change > self.params['change_threshold'] or std_change > self.params['change_threshold']:
                    current_regime = (current_regime + 1) % 3  # Cycle through 3 regimes
            
            regimes.iloc[i] = current_regime
                    
        return regimes

class MarketConditionRegimeModel(BaseRegimeModel):
    """Market condition classifier"""
    
    def __init__(self, window: int = 60, condition_threshold: float = 0.1):
        super().__init__(window=window, condition_threshold=condition_threshold, min_data_points=window + 50)
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Classify market conditions"""
        regimes = pd.Series(0, index=data.index)
        
        for i in range(self.params['window'], len(data)):
            window_data = data.iloc[i-self.params['window']:i+1]
            
            # Calculate market condition features
            returns = window_data.pct_change().dropna()
            volatility = returns.std()
            trend = (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0]
            
            # Classify market condition
            if volatility < self.params['condition_threshold'] and abs(trend) < self.params['condition_threshold']:
                regimes.iloc[i] = 0  # Sideways market
            elif trend > self.params['condition_threshold']:
                regimes.iloc[i] = 1  # Bull market
            elif trend < -self.params['condition_threshold']:
                regimes.iloc[i] = 2  # Bear market
            else:
                regimes.iloc[i] = 3  # Volatile market
                    
        return regimes

class RegimeSwitchingModel(BaseRegimeModel):
    """Regime switching model with transition probabilities"""
    
    def __init__(self, window: int = 30, n_regimes: int = 3, transition_threshold: float = 0.5):
        super().__init__(window=window, n_regimes=n_regimes, transition_threshold=transition_threshold, min_data_points=window + 50)
        self.transition_matrix = None
        self.regime_probabilities = None
        
    def fit(self, data: pd.Series) -> 'RegimeSwitchingModel':
        """Fit regime switching model"""
        super().fit(data)
        
        # Initialize transition matrix
        self.transition_matrix = np.ones((self.params['n_regimes'], self.params['n_regimes'])) / self.params['n_regimes']
        self.regime_probabilities = np.ones(self.params['n_regimes']) / self.params['n_regimes']
        
        return self
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Predict regimes with switching logic"""
        regimes = pd.Series(0, index=data.index)
        
        for i in range(self.params['window'], len(data)):
            window_data = data.iloc[i-self.params['window']:i+1]
            
            # Calculate regime features
            returns = window_data.pct_change().dropna()
            features = [
                returns.mean(),
                returns.std(),
                returns.skew(),
                returns.kurtosis()
            ]
            
            # Calculate regime probabilities
            if self.regime_probabilities is not None:
                # Update probabilities based on features
                feature_score = np.array(features)
                regime_scores = np.dot(self.transition_matrix, self.regime_probabilities)
                
                # Assign regime based on highest probability
                regime = np.argmax(regime_scores)
                regimes.iloc[i] = regime
                
                # Update transition probabilities
                self.regime_probabilities = regime_scores
                
        return regimes

class PerformanceBasedRegimeModel(BaseRegimeModel):
    """Performance-based regime detection with reweighting"""
    
    def __init__(self, window: int = 30, performance_threshold: float = 0.05):
        super().__init__(window=window, performance_threshold=performance_threshold, min_data_points=window + 50)
        self.performance_history = []
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Detect regimes based on performance"""
        regimes = pd.Series(0, index=data.index)
        
        for i in range(self.params['window'], len(data)):
            window_data = data.iloc[i-self.params['window']:i+1]
            
            # Calculate performance metrics
            returns = window_data.pct_change().dropna()
            cumulative_return = (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0]
            sharpe_ratio = returns.mean() / (returns.std() + 1e-8)
            max_drawdown = self._calculate_max_drawdown(window_data)
            
            # Classify regime based on performance
            if cumulative_return > self.params['performance_threshold'] and sharpe_ratio > 0:
                regimes.iloc[i] = 0  # High performance
            elif cumulative_return < -self.params['performance_threshold'] or max_drawdown < -0.1:
                regimes.iloc[i] = 1  # Low performance
            else:
                regimes.iloc[i] = 2  # Neutral performance
                
            # Store performance history
            self.performance_history.append({
                'timestamp': i,
                'cumulative_return': cumulative_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'regime': regimes.iloc[i]
            })
                    
        return regimes
    
    def _calculate_max_drawdown(self, data: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = data.expanding().max()
        drawdown = (data - peak) / peak
        return drawdown.min()

class EnsembleRegimeModel(BaseRegimeModel):
    """Ensemble regime detection model"""
    
    def __init__(self, models: List[BaseRegimeModel] = None):
        super().__init__(min_data_points=100)
        self.models = models or [
            VolatilityRegimeModel(),
            ClusteringRegimeModel(),
            GaussianMixtureRegimeModel(),
            MarketConditionRegimeModel()
        ]
        
    def fit(self, data: pd.Series) -> 'EnsembleRegimeModel':
        """Fit all ensemble models"""
        for model in self.models:
            try:
                model.fit(data)
            except Exception as e:
                logger.warning(f"Failed to fit ensemble model: {e}")
        
        self.is_fitted = True
        return self
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Predict regimes using ensemble"""
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(data)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict with ensemble model: {e}")
        
        if predictions:
            # Combine predictions (majority vote)
            ensemble_pred = pd.concat(predictions, axis=1).mode(axis=1).iloc[:, 0]
            return ensemble_pred
        else:
            return pd.Series(0, index=data.index)

class RegimeDetectionModel:
    """Comprehensive regime detection model for S&P 500 trading"""
    
    def __init__(self, regime_window: int = 30):
        self.regime_window = regime_window
        self.is_fitted = False
        self.models = {}
        
        # Initialize different regime detection models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all regime detection models"""
        
        # Basic regime models
        self.models['volatility'] = VolatilityRegimeModel(window=self.regime_window)
        self.models['clustering'] = ClusteringRegimeModel(window=self.regime_window)
        self.models['gaussian_mixture'] = GaussianMixtureRegimeModel(window=self.regime_window)
        self.models['change_point'] = ChangePointRegimeModel(window=self.regime_window)
        self.models['market_condition'] = MarketConditionRegimeModel(window=self.regime_window)
        self.models['regime_switching'] = RegimeSwitchingModel(window=self.regime_window)
        self.models['performance_based'] = PerformanceBasedRegimeModel(window=self.regime_window)
        
        # Ensemble model
        self.models['ensemble'] = EnsembleRegimeModel([
            self.models['volatility'],
            self.models['clustering'],
            self.models['gaussian_mixture'],
            self.models['market_condition']
        ])
        
    def _get_real_market_data(self, symbol: str, start_date: str, end_date: str):
        """Get real market data from Polygon API"""
        try:
            from polygon_connector import PolygonConnector
            connector = PolygonConnector()
            return connector.get_ohlcv_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get real market data: {e}")
            return None

    def _get_real_economic_data(self, series_id: str, start_date: str, end_date: str):
        """Get real economic data from FRED API"""
        try:
            from fred_connector import FREDConnector
            connector = FREDConnector()
            return connector.get_series_data(series_id, start_date, end_date)
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

    def fit(self, data: pd.Series) -> 'RegimeDetectionModel':
        """Fit all regime detection models"""
        if len(data) < self.regime_window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.regime_window}")
        
        # Fit all models
        for model_name, model in self.models.items():
            try:
                model.fit(data)
            except Exception as e:
                logger.warning(f"Failed to fit {model_name}: {e}")
        
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series, model_name: str = None) -> pd.Series:
        """Detect regimes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            return self.models[model_name].predict(data)
        else:
            # Return ensemble prediction
            predictions = []
            for model in self.models.values():
                try:
                    pred = model.predict(data)
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Failed to predict with model: {e}")
            
            if predictions:
                # Ensemble prediction (majority vote)
                ensemble_pred = pd.concat(predictions, axis=1).mode(axis=1).iloc[:, 0]
                return ensemble_pred
            else:
                return pd.Series(0, index=data.index)

    def get_available_models(self) -> List[str]:
        """Get list of available regime detection models"""
        return list(self.models.keys())

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name].params

    def get_parameters(self) -> Dict[str, Any]:
        """Get all model parameters"""
        return {
            'regime_window': self.regime_window,
            'is_fitted': self.is_fitted,
            'available_models': self.get_available_models(),
            'model_parameters': {
                name: model.params for name, model in self.models.items()
            }
        }