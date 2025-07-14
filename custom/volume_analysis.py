#!/usr/bin/env python3
"""
Advanced Volume Analysis Model Implementations for S&P 500 Trading
=================================================================

Advanced volume analysis models including:
- Volume-weighted indicators
- Volume-price relationships
- Volume momentum analysis
- Volume-based trading signals
- Volume profile analysis
- Volume divergence detection
- Volume clustering
- Volume-based risk metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class VolumeAnalysis:
    """Volume analysis result"""
    symbol: str
    volume_momentum: float
    volume_trend: float
    volume_divergence: float
    volume_signal: float
    volume_confidence: float
    volume_profile: Dict[str, float]
    timestamp: datetime

class BaseVolumeModel:
    """Base class for volume analysis models"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
        
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

    def fit(self, data: pd.Series, volume_data: pd.Series) -> 'BaseVolumeModel':
        """Fit the volume analysis model"""
        if len(data) < self.params.get('min_data_points', 100):
            raise ValueError(f"Insufficient data: {len(data)} < {self.params.get('min_data_points', 100)}")
        self.is_fitted = True
        return self

    def analyze_volume(self, data: pd.Series, volume_data: pd.Series) -> VolumeAnalysis:
        """Analyze volume patterns"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before volume analysis")
        raise NotImplementedError("Subclasses must implement analyze_volume method")

class VolumeWeightedModel(BaseVolumeModel):
    """Volume-weighted analysis model"""
    
    def __init__(self, window: int = 20, volume_threshold: float = 1.5):
        super().__init__(window=window, volume_threshold=volume_threshold, min_data_points=window + 10)
        
    def analyze_volume(self, data: pd.Series, volume_data: pd.Series) -> VolumeAnalysis:
        """Analyze volume-weighted patterns"""
        if len(data) < self.params['window']:
            return VolumeAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                volume_momentum=0.0,
                volume_trend=0.0,
                volume_divergence=0.0,
                volume_signal=0.0,
                volume_confidence=0.0,
                volume_profile={},
                timestamp=datetime.now()
            )
        
        # Calculate volume-weighted metrics
        returns = data.pct_change().dropna()
        volume_returns = volume_data.pct_change().dropna()
        
        # Volume-weighted price momentum
        vw_momentum = (returns * volume_data.iloc[1:]).rolling(window=self.params['window']).mean()
        current_vw_momentum = vw_momentum.iloc[-1] if len(vw_momentum) > 0 else 0
        
        # Volume trend
        volume_trend = volume_data.rolling(window=self.params['window']).mean()
        current_volume_trend = volume_trend.iloc[-1] if len(volume_trend) > 0 else volume_data.mean()
        
        # Volume divergence
        price_trend = data.rolling(window=self.params['window']).mean()
        volume_divergence = self._calculate_volume_divergence(data, volume_data)
        
        # Volume signal
        volume_signal = self._calculate_volume_signal(data, volume_data)
        
        # Volume confidence
        volume_confidence = self._calculate_volume_confidence(volume_data)
        
        # Volume profile
        volume_profile = self._calculate_volume_profile(volume_data)
        
        return VolumeAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            volume_momentum=current_vw_momentum,
            volume_trend=current_volume_trend,
            volume_divergence=volume_divergence,
            volume_signal=volume_signal,
            volume_confidence=volume_confidence,
            volume_profile=volume_profile,
            timestamp=datetime.now()
        )
    
    def _calculate_volume_divergence(self, data: pd.Series, volume_data: pd.Series) -> float:
        """Calculate volume-price divergence"""
        if len(data) < self.params['window']:
            return 0.0
        
        # Calculate price and volume trends
        price_trend = data.rolling(window=self.params['window']).mean()
        volume_trend = volume_data.rolling(window=self.params['window']).mean()
        
        # Calculate correlation
        if len(price_trend) > 0 and len(volume_trend) > 0:
            correlation = price_trend.corr(volume_trend)
            return 1.0 - abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            return 0.0
    
    def _calculate_volume_signal(self, data: pd.Series, volume_data: pd.Series) -> float:
        """Calculate volume-based trading signal"""
        if len(data) < self.params['window']:
            return 0.0
        
        # Calculate volume ratio
        current_volume = volume_data.iloc[-1]
        avg_volume = volume_data.rolling(window=self.params['window']).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate price momentum
        returns = data.pct_change().dropna()
        price_momentum = returns.rolling(window=self.params['window']).mean().iloc[-1]
        
        # Generate signal
        if volume_ratio > self.params['volume_threshold'] and price_momentum > 0:
            return 1.0  # Strong buy signal
        elif volume_ratio > self.params['volume_threshold'] and price_momentum < 0:
            return -1.0  # Strong sell signal
        else:
            return 0.0  # Neutral
    
    def _calculate_volume_confidence(self, volume_data: pd.Series) -> float:
        """Calculate volume confidence level"""
        if len(volume_data) < self.params['window']:
            return 0.0
        
        # Calculate volume stability
        volume_std = volume_data.rolling(window=self.params['window']).std()
        volume_mean = volume_data.rolling(window=self.params['window']).mean()
        
        if len(volume_std) > 0 and len(volume_mean) > 0:
            current_cv = volume_std.iloc[-1] / volume_mean.iloc[-1] if volume_mean.iloc[-1] > 0 else 1.0
            return 1.0 / (1.0 + current_cv)  # Inverse relationship
        else:
            return 0.0
    
    def _calculate_volume_profile(self, volume_data: pd.Series) -> Dict[str, float]:
        """Calculate volume profile metrics"""
        if len(volume_data) < self.params['window']:
            return {}
        
        profile = {
            'mean_volume': volume_data.mean(),
            'std_volume': volume_data.std(),
            'max_volume': volume_data.max(),
            'min_volume': volume_data.min(),
            'volume_trend': volume_data.rolling(window=self.params['window']).mean().iloc[-1] if len(volume_data) >= self.params['window'] else volume_data.mean()
        }
        
        return profile

class VolumePriceRelationshipModel(BaseVolumeModel):
    """Volume-price relationship analysis model"""
    
    def __init__(self, window: int = 20, relationship_threshold: float = 0.7):
        super().__init__(window=window, relationship_threshold=relationship_threshold, min_data_points=window + 10)
        
    def analyze_volume(self, data: pd.Series, volume_data: pd.Series) -> VolumeAnalysis:
        """Analyze volume-price relationships"""
        if len(data) < self.params['window']:
            return VolumeAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                volume_momentum=0.0,
                volume_trend=0.0,
                volume_divergence=0.0,
                volume_signal=0.0,
                volume_confidence=0.0,
                volume_profile={},
                timestamp=datetime.now()
            )
        
        # Calculate volume-price relationship
        returns = data.pct_change().dropna()
        volume_returns = volume_data.pct_change().dropna()
        
        # Volume-price correlation
        correlation = returns.corr(volume_returns) if len(returns) == len(volume_returns) else 0
        correlation = 0.0 if np.isnan(correlation) else correlation
        
        # Volume momentum
        volume_momentum = volume_returns.rolling(window=self.params['window']).mean().iloc[-1] if len(volume_returns) >= self.params['window'] else 0
        volume_momentum = 0.0 if np.isnan(volume_momentum) else volume_momentum
        
        # Volume trend
        volume_trend = volume_data.rolling(window=self.params['window']).mean().iloc[-1] if len(volume_data) >= self.params['window'] else volume_data.mean()
        volume_trend = volume_data.mean() if np.isnan(volume_trend) else volume_trend
        
        # Volume divergence
        volume_divergence = 1.0 - abs(correlation)
        
        # Volume signal
        volume_signal = self._calculate_relationship_signal(data, volume_data, correlation)
        
        # Volume confidence
        volume_confidence = abs(correlation)
        
        # Volume profile
        volume_profile = {
            'price_volume_correlation': correlation,
            'volume_momentum': volume_momentum,
            'volume_trend': volume_trend,
            'relationship_strength': abs(correlation)
        }
        
        return VolumeAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            volume_momentum=volume_momentum,
            volume_trend=volume_trend,
            volume_divergence=volume_divergence,
            volume_signal=volume_signal,
            volume_confidence=volume_confidence,
            volume_profile=volume_profile,
            timestamp=datetime.now()
        )
    
    def _calculate_relationship_signal(self, data: pd.Series, volume_data: pd.Series, correlation: float) -> float:
        """Calculate signal based on volume-price relationship"""
        if len(data) < self.params['window']:
            return 0.0
        
        # Calculate price and volume momentum
        returns = data.pct_change().dropna()
        volume_returns = volume_data.pct_change().dropna()
        
        price_momentum = returns.rolling(window=self.params['window']).mean().iloc[-1] if len(returns) >= self.params['window'] else 0
        volume_momentum = volume_returns.rolling(window=self.params['window']).mean().iloc[-1] if len(volume_returns) >= self.params['window'] else 0
        
        price_momentum = 0.0 if np.isnan(price_momentum) else price_momentum
        volume_momentum = 0.0 if np.isnan(volume_momentum) else volume_momentum
        
        # Generate signal based on relationship strength and momentum alignment
        if abs(correlation) > self.params['relationship_threshold']:
            if price_momentum > 0 and volume_momentum > 0:
                return 1.0  # Strong positive relationship
            elif price_momentum < 0 and volume_momentum < 0:
                return -1.0  # Strong negative relationship
            else:
                return 0.0  # Divergence
        else:
            return 0.0  # Weak relationship

class VolumeMomentumModel(BaseVolumeModel):
    """Volume momentum analysis model"""
    
    def __init__(self, short_window: int = 10, long_window: int = 30, momentum_threshold: float = 0.1):
        super().__init__(short_window=short_window, long_window=long_window, momentum_threshold=momentum_threshold, min_data_points=long_window + 10)
        
    def analyze_volume(self, data: pd.Series, volume_data: pd.Series) -> VolumeAnalysis:
        """Analyze volume momentum patterns"""
        if len(data) < self.params['long_window']:
            return VolumeAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                volume_momentum=0.0,
                volume_trend=0.0,
                volume_divergence=0.0,
                volume_signal=0.0,
                volume_confidence=0.0,
                volume_profile={},
                timestamp=datetime.now()
            )
        
        # Calculate volume momentum
        volume_returns = volume_data.pct_change().dropna()
        
        short_momentum = volume_returns.rolling(window=self.params['short_window']).mean().iloc[-1] if len(volume_returns) >= self.params['short_window'] else 0
        long_momentum = volume_returns.rolling(window=self.params['long_window']).mean().iloc[-1] if len(volume_returns) >= self.params['long_window'] else 0
        
        short_momentum = 0.0 if np.isnan(short_momentum) else short_momentum
        long_momentum = 0.0 if np.isnan(long_momentum) else long_momentum
        
        # Combined momentum
        volume_momentum = (short_momentum + long_momentum) / 2
        
        # Volume trend
        volume_trend = volume_data.rolling(window=self.params['long_window']).mean().iloc[-1] if len(volume_data) >= self.params['long_window'] else volume_data.mean()
        volume_trend = volume_data.mean() if np.isnan(volume_trend) else volume_trend
        
        # Volume divergence
        price_returns = data.pct_change().dropna()
        price_momentum = price_returns.rolling(window=self.params['long_window']).mean().iloc[-1] if len(price_returns) >= self.params['long_window'] else 0
        price_momentum = 0.0 if np.isnan(price_momentum) else price_momentum
        
        volume_divergence = abs(volume_momentum - price_momentum)
        
        # Volume signal
        volume_signal = self._calculate_momentum_signal(volume_momentum, price_momentum)
        
        # Volume confidence
        volume_confidence = abs(volume_momentum)
        
        # Volume profile
        volume_profile = {
            'short_momentum': short_momentum,
            'long_momentum': long_momentum,
            'combined_momentum': volume_momentum,
            'price_momentum': price_momentum,
            'momentum_divergence': volume_divergence
        }
        
        return VolumeAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            volume_momentum=volume_momentum,
            volume_trend=volume_trend,
            volume_divergence=volume_divergence,
            volume_signal=volume_signal,
            volume_confidence=volume_confidence,
            volume_profile=volume_profile,
            timestamp=datetime.now()
        )
    
    def _calculate_momentum_signal(self, volume_momentum: float, price_momentum: float) -> float:
        """Calculate signal based on volume momentum"""
        if abs(volume_momentum) > self.params['momentum_threshold']:
            if volume_momentum > 0 and price_momentum > 0:
                return 1.0  # Strong positive momentum
            elif volume_momentum < 0 and price_momentum < 0:
                return -1.0  # Strong negative momentum
            elif volume_momentum > 0 and price_momentum < 0:
                return 0.5  # Volume increasing, price decreasing (potential reversal)
            elif volume_momentum < 0 and price_momentum > 0:
                return -0.5  # Volume decreasing, price increasing (potential reversal)
            else:
                return 0.0
        else:
            return 0.0

class VolumeDivergenceModel(BaseVolumeModel):
    """Volume divergence detection model"""
    
    def __init__(self, window: int = 20, divergence_threshold: float = 0.3):
        super().__init__(window=window, divergence_threshold=divergence_threshold, min_data_points=window + 10)
        
    def analyze_volume(self, data: pd.Series, volume_data: pd.Series) -> VolumeAnalysis:
        """Analyze volume divergence patterns"""
        if len(data) < self.params['window']:
            return VolumeAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                volume_momentum=0.0,
                volume_trend=0.0,
                volume_divergence=0.0,
                volume_signal=0.0,
                volume_confidence=0.0,
                volume_profile={},
                timestamp=datetime.now()
            )
        
        # Calculate divergences
        price_returns = data.pct_change().dropna()
        volume_returns = volume_data.pct_change().dropna()
        
        # Price and volume trends
        price_trend = price_returns.rolling(window=self.params['window']).mean().iloc[-1] if len(price_returns) >= self.params['window'] else 0
        volume_trend = volume_returns.rolling(window=self.params['window']).mean().iloc[-1] if len(volume_returns) >= self.params['window'] else 0
        
        price_trend = 0.0 if np.isnan(price_trend) else price_trend
        volume_trend = 0.0 if np.isnan(volume_trend) else volume_trend
        
        # Volume momentum
        volume_momentum = volume_trend
        
        # Volume divergence
        volume_divergence = abs(price_trend - volume_trend)
        
        # Volume signal
        volume_signal = self._calculate_divergence_signal(price_trend, volume_trend)
        
        # Volume confidence
        volume_confidence = 1.0 - volume_divergence
        
        # Volume profile
        volume_profile = {
            'price_trend': price_trend,
            'volume_trend': volume_trend,
            'divergence_magnitude': volume_divergence,
            'divergence_type': 'positive' if price_trend > 0 and volume_trend < 0 else 'negative' if price_trend < 0 and volume_trend > 0 else 'none'
        }
        
        return VolumeAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            volume_momentum=volume_momentum,
            volume_trend=volume_trend,
            volume_divergence=volume_divergence,
            volume_signal=volume_signal,
            volume_confidence=volume_confidence,
            volume_profile=volume_profile,
            timestamp=datetime.now()
        )
    
    def _calculate_divergence_signal(self, price_trend: float, volume_trend: float) -> float:
        """Calculate signal based on volume divergence"""
        divergence = abs(price_trend - volume_trend)
        
        if divergence > self.params['divergence_threshold']:
            if price_trend > 0 and volume_trend < 0:
                return -1.0  # Bearish divergence
            elif price_trend < 0 and volume_trend > 0:
                return 1.0  # Bullish divergence
            else:
                return 0.0
        else:
            return 0.0

class VolumeClusteringModel(BaseVolumeModel):
    """Volume clustering analysis model"""
    
    def __init__(self, n_clusters: int = 3, window: int = 20):
        super().__init__(n_clusters=n_clusters, window=window, min_data_points=window + 10)
        
    def analyze_volume(self, data: pd.Series, volume_data: pd.Series) -> VolumeAnalysis:
        """Analyze volume clustering patterns"""
        if len(data) < self.params['window']:
            return VolumeAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                volume_momentum=0.0,
                volume_trend=0.0,
                volume_divergence=0.0,
                volume_signal=0.0,
                volume_confidence=0.0,
                volume_profile={},
                timestamp=datetime.now()
            )
        
        # Prepare features for clustering
        returns = data.pct_change().dropna()
        volume_returns = volume_data.pct_change().dropna()
        
        if len(returns) < self.params['window'] or len(volume_returns) < self.params['window']:
            return VolumeAnalysis(
                symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
                volume_momentum=0.0,
                volume_trend=0.0,
                volume_divergence=0.0,
                volume_signal=0.0,
                volume_confidence=0.0,
                volume_profile={},
                timestamp=datetime.now()
            )
        
        # Create feature matrix
        features = []
        for i in range(self.params['window'], len(returns)):
            feature_vector = [
                returns.iloc[i-self.params['window']:i].mean(),  # Price momentum
                returns.iloc[i-self.params['window']:i].std(),   # Price volatility
                volume_returns.iloc[i-self.params['window']:i].mean(),  # Volume momentum
                volume_returns.iloc[i-self.params['window']:i].std()    # Volume volatility
            ]
            features.append(feature_vector)
        
        if len(features) > 0:
            # Normalize features
            features_array = np.array(features)
            features_normalized = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-8)
            
            # Apply clustering
            kmeans = KMeans(n_clusters=self.params['n_clusters'], random_state=42)
            cluster_labels = kmeans.fit_predict(features_normalized)
            
            # Analyze current cluster
            current_features = features_normalized[-1] if len(features_normalized) > 0 else np.zeros(4)
            current_cluster = cluster_labels[-1] if len(cluster_labels) > 0 else 0
            
            # Calculate cluster-based metrics
            cluster_centers = kmeans.cluster_centers_
            current_center = cluster_centers[current_cluster]
            
            # Volume momentum (based on cluster characteristics)
            volume_momentum = current_center[2]  # Volume momentum component
            
            # Volume trend
            volume_trend = volume_data.rolling(window=self.params['window']).mean().iloc[-1] if len(volume_data) >= self.params['window'] else volume_data.mean()
            volume_trend = volume_data.mean() if np.isnan(volume_trend) else volume_trend
            
            # Volume divergence
            volume_divergence = abs(current_center[0] - current_center[2])  # Price vs volume momentum difference
            
            # Volume signal
            volume_signal = self._calculate_cluster_signal(current_center)
            
            # Volume confidence
            volume_confidence = 1.0 - np.linalg.norm(current_features - current_center)
            
            # Volume profile
            volume_profile = {
                'cluster_id': int(current_cluster),
                'cluster_center': current_center.tolist(),
                'distance_to_center': float(np.linalg.norm(current_features - current_center)),
                'cluster_size': int(np.sum(cluster_labels == current_cluster))
            }
            
        else:
            # Default values if clustering fails
            volume_momentum = 0.0
            volume_trend = volume_data.mean()
            volume_divergence = 0.0
            volume_signal = 0.0
            volume_confidence = 0.0
            volume_profile = {}
        
        return VolumeAnalysis(
            symbol=str(data.name) if hasattr(data, 'name') and data.name is not None else 'Unknown',
            volume_momentum=volume_momentum,
            volume_trend=volume_trend,
            volume_divergence=volume_divergence,
            volume_signal=volume_signal,
            volume_confidence=volume_confidence,
            volume_profile=volume_profile,
            timestamp=datetime.now()
        )
    
    def _calculate_cluster_signal(self, cluster_center: np.ndarray) -> float:
        """Calculate signal based on cluster characteristics"""
        price_momentum = cluster_center[0]
        volume_momentum = cluster_center[2]
        
        # Signal based on momentum alignment
        if price_momentum > 0 and volume_momentum > 0:
            return 1.0  # Strong positive
        elif price_momentum < 0 and volume_momentum < 0:
            return -1.0  # Strong negative
        elif price_momentum > 0 and volume_momentum < 0:
            return -0.5  # Bearish divergence
        elif price_momentum < 0 and volume_momentum > 0:
            return 0.5  # Bullish divergence
        else:
            return 0.0  # Neutral

class VolumeAnalysisModel:
    """Comprehensive volume analysis model for S&P 500 trading"""
    
    def __init__(self, analysis_window: int = 252):
        self.analysis_window = analysis_window
        self.is_fitted = False
        self.models = {}
        
        # Initialize different volume analysis models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all volume analysis models"""
        
        # Basic volume models
        self.models['volume_weighted'] = VolumeWeightedModel()
        self.models['volume_price_relationship'] = VolumePriceRelationshipModel()
        self.models['volume_momentum'] = VolumeMomentumModel()
        self.models['volume_divergence'] = VolumeDivergenceModel()
        self.models['volume_clustering'] = VolumeClusteringModel()
        
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

    def fit(self, data: pd.Series, volume_data: pd.Series) -> 'VolumeAnalysisModel':
        """Fit all volume analysis models"""
        if len(data) < self.analysis_window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.analysis_window}")
        
        # Fit all models
        for model_name, model in self.models.items():
            try:
                model.fit(data, volume_data)
            except Exception as e:
                logger.warning(f"Failed to fit {model_name}: {e}")
        
        self.is_fitted = True
        return self

    def analyze_volume(self, data: pd.Series, volume_data: pd.Series, model_name: str = None) -> VolumeAnalysis:
        """Analyze volume patterns"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before volume analysis")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            return self.models[model_name].analyze_volume(data, volume_data)
        else:
            # Return volume-weighted analysis as default
            return self.models['volume_weighted'].analyze_volume(data, volume_data)

    def get_available_models(self) -> List[str]:
        """Get list of available volume analysis models"""
        return list(self.models.keys())

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name].params

    def get_parameters(self) -> Dict[str, Any]:
        """Get all model parameters"""
        return {
            'analysis_window': self.analysis_window,
            'is_fitted': self.is_fitted,
            'available_models': self.get_available_models(),
            'model_parameters': {
                name: model.params for name, model in self.models.items()
            }
        } 
 