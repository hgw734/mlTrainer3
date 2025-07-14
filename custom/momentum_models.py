#!/usr/bin/env python3
"""
Comprehensive Momentum Model Implementations for S&P 500 Trading
==============================================================

Advanced momentum models including:
- Multiple window lengths (7-12, 50-70 day windows)
- Dual momentum (absolute + relative)
- Cross-sectional momentum
- Risk-adjusted momentum
- Time series momentum
- Momentum with volume confirmation
- Sector rotation momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class MomentumResult:
    """Momentum calculation result"""
    symbol: str
    momentum_score: float
    momentum_rank: int
    signal: float  # -1, 0, 1
    confidence: float
    window: int
    method: str
    timestamp: datetime

class BaseMomentumModel:
    """Base class for momentum models"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
        
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

    def fit(self, data: pd.Series) -> 'BaseMomentumModel':
        """Fit the momentum model"""
        if len(data) < self.params.get('min_data_points', 50):
            raise ValueError(f"Insufficient data: {len(data)} < {self.params.get('min_data_points', 50)}")
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series) -> pd.Series:
        """Make momentum predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        raise NotImplementedError("Subclasses must implement predict method")

class SimpleMomentumModel(BaseMomentumModel):
    """Simple price momentum model"""
    
    def __init__(self, window: int = 20, threshold: float = 0.01):
        super().__init__(window=window, threshold=threshold, min_data_points=window + 10)
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Calculate simple momentum signals"""
        signals = pd.Series(0.0, index=data.index)
        
        for i in range(self.params['window'], len(data)):
            window_data = data.iloc[i-self.params['window']:i+1]
            
            # Calculate momentum
            momentum = (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0]
            
            # Generate signals
            if momentum > self.params['threshold']:
                signals.iloc[i] = 1.0
            elif momentum < -self.params['threshold']:
                signals.iloc[i] = -1.0
                
        return signals

class DualMomentumModel(BaseMomentumModel):
    """Dual momentum model (absolute + relative)"""
    
    def __init__(self, absolute_window: int = 20, relative_window: int = 60, 
                 absolute_threshold: float = 0.01, relative_threshold: float = 0.05):
        super().__init__(
            absolute_window=absolute_window,
            relative_window=relative_window,
            absolute_threshold=absolute_threshold,
            relative_threshold=relative_threshold,
            min_data_points=max(absolute_window, relative_window) + 10
        )
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Calculate dual momentum signals"""
        signals = pd.Series(0.0, index=data.index)
        
        for i in range(self.params['relative_window'], len(data)):
            # Absolute momentum
            abs_window = data.iloc[i-self.params['absolute_window']:i+1]
            abs_momentum = (abs_window.iloc[-1] - abs_window.iloc[0]) / abs_window.iloc[0]
            
            # Relative momentum (vs longer-term average)
            rel_window = data.iloc[i-self.params['relative_window']:i+1]
            rel_momentum = (rel_window.iloc[-1] - rel_window.mean()) / rel_window.mean()
            
            # Combined signal
            if (abs_momentum > self.params['absolute_threshold'] and 
                rel_momentum > self.params['relative_threshold']):
                signals.iloc[i] = 1.0
            elif (abs_momentum < -self.params['absolute_threshold'] and 
                  rel_momentum < -self.params['relative_threshold']):
                signals.iloc[i] = -1.0
                
        return signals

class CrossSectionalMomentumModel(BaseMomentumModel):
    """Cross-sectional momentum model"""
    
    def __init__(self, window: int = 20, percentile_threshold: float = 0.7):
        super().__init__(window=window, percentile_threshold=percentile_threshold, min_data_points=window + 10)
        
    def predict(self, data: pd.Series, universe_data: pd.DataFrame) -> pd.Series:
        """Calculate cross-sectional momentum signals"""
        signals = pd.Series(0.0, index=data.index)
        
        for i in range(self.params['window'], len(data)):
            # Calculate momentum for current stock
            window_data = data.iloc[i-self.params['window']:i+1]
            momentum = (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0]
            
            # Calculate momentum for universe at same time point
            if i < len(universe_data):
                universe_momentums = []
                for col in universe_data.columns:
                    if col != data.name and i < len(universe_data[col]):
                        stock_data = universe_data[col].iloc[i-self.params['window']:i+1]
                        if len(stock_data) == self.params['window']:
                            stock_momentum = (stock_data.iloc[-1] - stock_data.iloc[0]) / stock_data.iloc[0]
                            universe_momentums.append(stock_momentum)
                
                if universe_momentums:
                    # Rank momentum within universe
                    momentum_rank = np.percentile(universe_momentums, momentum * 100)
                    
                    if momentum_rank > self.params['percentile_threshold']:
                        signals.iloc[i] = 1.0
                    elif momentum_rank < (1 - self.params['percentile_threshold']):
                        signals.iloc[i] = -1.0
                        
        return signals

class RiskAdjustedMomentumModel(BaseMomentumModel):
    """Risk-adjusted momentum model"""
    
    def __init__(self, window: int = 20, volatility_window: int = 60, 
                 momentum_threshold: float = 0.01, volatility_threshold: float = 0.02):
        super().__init__(
            window=window,
            volatility_window=volatility_window,
            momentum_threshold=momentum_threshold,
            volatility_threshold=volatility_threshold,
            min_data_points=max(window, volatility_window) + 10
        )
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Calculate risk-adjusted momentum signals"""
        signals = pd.Series(0.0, index=data.index)
        
        for i in range(self.params['volatility_window'], len(data)):
            # Calculate momentum
            mom_window = data.iloc[i-self.params['window']:i+1]
            momentum = (mom_window.iloc[-1] - mom_window.iloc[0]) / mom_window.iloc[0]
            
            # Calculate volatility
            vol_window = data.iloc[i-self.params['volatility_window']:i+1]
            volatility = vol_window.pct_change().std()
            
            # Risk-adjusted momentum
            risk_adj_momentum = momentum / volatility if volatility > 0 else 0
            
            # Generate signals
            if (risk_adj_momentum > self.params['momentum_threshold'] and 
                volatility < self.params['volatility_threshold']):
                signals.iloc[i] = 1.0
            elif (risk_adj_momentum < -self.params['momentum_threshold'] and 
                  volatility < self.params['volatility_threshold']):
                signals.iloc[i] = -1.0
                
        return signals

class TimeSeriesMomentumModel(BaseMomentumModel):
    """Time series momentum model"""
    
    def __init__(self, short_window: int = 7, long_window: int = 50, 
                 signal_threshold: float = 0.02):
        super().__init__(
            short_window=short_window,
            long_window=long_window,
            signal_threshold=signal_threshold,
            min_data_points=long_window + 10
        )
        
    def predict(self, data: pd.Series) -> pd.Series:
        """Calculate time series momentum signals"""
        signals = pd.Series(0.0, index=data.index)
        
        for i in range(self.params['long_window'], len(data)):
            # Short-term momentum
            short_window = data.iloc[i-self.params['short_window']:i+1]
            short_momentum = (short_window.iloc[-1] - short_window.iloc[0]) / short_window.iloc[0]
            
            # Long-term momentum
            long_window = data.iloc[i-self.params['long_window']:i+1]
            long_momentum = (long_window.iloc[-1] - long_window.iloc[0]) / long_window.iloc[0]
            
            # Combined signal
            combined_momentum = (short_momentum + long_momentum) / 2
            
            if combined_momentum > self.params['signal_threshold']:
                signals.iloc[i] = 1.0
            elif combined_momentum < -self.params['signal_threshold']:
                signals.iloc[i] = -1.0
                
        return signals

class VolumeConfirmedMomentumModel(BaseMomentumModel):
    """Momentum model with volume confirmation"""
    
    def __init__(self, momentum_window: int = 20, volume_window: int = 20,
                 momentum_threshold: float = 0.01, volume_threshold: float = 1.5):
        super().__init__(
            momentum_window=momentum_window,
            volume_window=volume_window,
            momentum_threshold=momentum_threshold,
            volume_threshold=volume_threshold,
            min_data_points=max(momentum_window, volume_window) + 10
        )
        
    def predict(self, price_data: pd.Series, volume_data: pd.Series) -> pd.Series:
        """Calculate volume-confirmed momentum signals"""
        signals = pd.Series(0.0, index=price_data.index)
        
        for i in range(self.params['momentum_window'], len(price_data)):
            # Calculate momentum
            price_window = price_data.iloc[i-self.params['momentum_window']:i+1]
            momentum = (price_window.iloc[-1] - price_window.iloc[0]) / price_window.iloc[0]
            
            # Calculate volume confirmation
            volume_window = volume_data.iloc[i-self.params['volume_window']:i+1]
            avg_volume = volume_window.mean()
            current_volume = volume_data.iloc[i]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Generate signals with volume confirmation
            if (momentum > self.params['momentum_threshold'] and 
                volume_ratio > self.params['volume_threshold']):
                signals.iloc[i] = 1.0
            elif (momentum < -self.params['momentum_threshold'] and 
                  volume_ratio > self.params['volume_threshold']):
                signals.iloc[i] = -1.0
                
        return signals

class SectorRotationMomentumModel(BaseMomentumModel):
    """Sector rotation momentum model"""
    
    def __init__(self, window: int = 60, sector_threshold: float = 0.05):
        super().__init__(window=window, sector_threshold=sector_threshold, min_data_points=window + 10)
        
    def predict(self, sector_data: Dict[str, pd.Series], stock_sector: str) -> pd.Series:
        """Calculate sector rotation momentum signals"""
        if stock_sector not in sector_data:
            return pd.Series(0.0, index=list(sector_data.values())[0].index)
            
        stock_data = sector_data[stock_sector]
        signals = pd.Series(0.0, index=stock_data.index)
        
        for i in range(self.params['window'], len(stock_data)):
            # Calculate sector momentum
            sector_window = stock_data.iloc[i-self.params['window']:i+1]
            sector_momentum = (sector_window.iloc[-1] - sector_window.iloc[0]) / sector_window.iloc[0]
            
            # Calculate relative sector performance
            sector_performances = []
            for sector_name, sector_series in sector_data.items():
                if i < len(sector_series):
                    sector_window = sector_series.iloc[i-self.params['window']:i+1]
                    if len(sector_window) == self.params['window']:
                        sector_perf = (sector_window.iloc[-1] - sector_window.iloc[0]) / sector_window.iloc[0]
                        sector_performances.append(sector_perf)
            
            if sector_performances:
                # Rank sector performance
                sector_rank = np.percentile(sector_performances, sector_momentum * 100)
                
                if sector_rank > self.params['sector_threshold']:
                    signals.iloc[i] = 1.0
                elif sector_rank < -self.params['sector_threshold']:
                    signals.iloc[i] = -1.0
                    
        return signals

class MomentumModelFactory:
    """Factory for creating momentum models"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseMomentumModel:
        """Create a momentum model by type"""
        models = {
            'simple': SimpleMomentumModel,
            'dual': DualMomentumModel,
            'cross_sectional': CrossSectionalMomentumModel,
            'risk_adjusted': RiskAdjustedMomentumModel,
            'time_series': TimeSeriesMomentumModel,
            'volume_confirmed': VolumeConfirmedMomentumModel,
            'sector_rotation': SectorRotationMomentumModel
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown momentum model type: {model_type}")
            
        return models[model_type](**kwargs)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available momentum model types"""
        return ['simple', 'dual', 'cross_sectional', 'risk_adjusted', 
                'time_series', 'volume_confirmed', 'sector_rotation']
    
    @staticmethod
    def get_model_parameters(model_type: str) -> Dict[str, Any]:
        """Get default parameters for a model type"""
        default_params = {
            'simple': {'window': 20, 'threshold': 0.01},
            'dual': {'absolute_window': 20, 'relative_window': 60, 
                    'absolute_threshold': 0.01, 'relative_threshold': 0.05},
            'cross_sectional': {'window': 20, 'percentile_threshold': 0.7},
            'risk_adjusted': {'window': 20, 'volatility_window': 60,
                            'momentum_threshold': 0.01, 'volatility_threshold': 0.02},
            'time_series': {'short_window': 7, 'long_window': 50, 'signal_threshold': 0.02},
            'volume_confirmed': {'momentum_window': 20, 'volume_window': 20,
                               'momentum_threshold': 0.01, 'volume_threshold': 1.5},
            'sector_rotation': {'window': 60, 'sector_threshold': 0.05}
        }
        
        return default_params.get(model_type, {})

class MomentumModel:
    """Comprehensive momentum model for S&P 500 trading"""
    
    def __init__(self, momentum_window: int = 20):
        self.momentum_window = momentum_window
        self.is_fitted = False
        self.models = {}
        
        # Initialize different momentum models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all momentum models"""
        factory = MomentumModelFactory()
        
        # Create models with different window lengths
        windows = [7, 12, 20, 50, 70]
        
        for window in windows:
            # Simple momentum models
            self.models[f'simple_{window}'] = factory.create_model('simple', window=window)
            
            # Time series momentum models
            if window >= 50:
                self.models[f'time_series_{window}'] = factory.create_model(
                    'time_series', short_window=7, long_window=window
                )
            
            # Risk-adjusted momentum models
            self.models[f'risk_adjusted_{window}'] = factory.create_model(
                'risk_adjusted', window=window, volatility_window=window*2
            )
        
        # Specialized models
        self.models['dual_momentum'] = factory.create_model('dual')
        self.models['cross_sectional'] = factory.create_model('cross_sectional')
        self.models['volume_confirmed'] = factory.create_model('volume_confirmed')
        self.models['sector_rotation'] = factory.create_model('sector_rotation')
        
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

    def fit(self, data: pd.Series) -> 'MomentumModel':
        """Fit all momentum models"""
        if len(data) < self.momentum_window:
            raise ValueError(f"Insufficient data: {len(data)} < {self.momentum_window}")
        
        # Fit all models
        for model_name, model in self.models.items():
            try:
                model.fit(data)
            except Exception as e:
                logger.warning(f"Failed to fit {model_name}: {e}")
        
        self.is_fitted = True
        return self

    def predict(self, data: pd.Series, model_name: str = None) -> pd.Series:
        """Make momentum predictions"""
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
                # Simple ensemble (average)
                ensemble_pred = pd.concat(predictions, axis=1).mean(axis=1)
                return ensemble_pred
            else:
                return pd.Series(0.0, index=data.index)

    def get_available_models(self) -> List[str]:
        """Get list of available momentum models"""
        return list(self.models.keys())

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name].params

    def get_parameters(self) -> Dict[str, Any]:
        """Get all model parameters"""
        return {
            'momentum_window': self.momentum_window,
            'is_fitted': self.is_fitted,
            'available_models': self.get_available_models(),
            'model_parameters': {
                name: model.params for name, model in self.models.items()
            }
        }
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

                                                        def __init__(self, momentum_window: int = 20):
                                                            self.momentum_window = momentum_window
                                                            self.is_fitted = False

                                                            def fit(self, data: pd.Series) -> 'MomentumModel':
                                                                if len(data) < self.momentum_window:
                                                                    raise ValueError(f"Insufficient data: {len(data)} < {self.momentum_window}")
                                                                    self.is_fitted = True
                                                                    return self

                                                                    def predict(self, data: pd.Series) -> pd.Series:
                                                                        if not self.is_fitted:
                                                                            raise ValueError("Model must be fitted before prediction")

                                                                            signals = pd.Series(0.0, index=data.index)

                                                                            for i in range(self.momentum_window, len(data)):
                                                                                window_data = data.iloc[i-self.momentum_window:i+1]

                                                                                # Calculate momentum
                                                                                momentum = (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0]

                                                                                # Simple momentum signal
                                                                                if momentum > 0.01:
                                                                                    signals.iloc[i] = 1.0
                                                                                    elif momentum < -0.01:
                                                                                        signals.iloc[i] = -1.0

                                                                                        return signals

                                                                                        def get_parameters(self) -> Dict[str, Any]:
                                                                                            return {'momentum_window': self.momentum_window, 'is_fitted': self.is_fitted}