#!/usr/bin/env python3
"""
Prediction Service for mlTrainer
Handles real-time predictions using trained models
USES REAL MODELS AND REAL DATA ONLY
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

from core.data_pipeline import DataPipeline
from core.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Service for generating real-time predictions using trained models
    All predictions based on real market data
    """
    
    def __init__(self):
        """Initialize prediction service"""
        self.data_pipeline = DataPipeline()
        self.model_trainer = ModelTrainer()
        self.active_models = {}
        self.prediction_cache = {}
        self.cache_duration = timedelta(minutes=5)
        
    def load_model(self, model_id: str) -> bool:
        """
        Load a trained model into memory
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model = self.model_trainer.load_model(model_id)
            self.active_models[model_id] = {
                'model': model,
                'loaded_at': datetime.now(),
                'predictions_count': 0
            }
            logger.info(f"Model {model_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def unload_model(self, model_id: str):
        """Remove model from memory"""
        if model_id in self.active_models:
            del self.active_models[model_id]
            # Clear related cache
            self.prediction_cache = {
                k: v for k, v in self.prediction_cache.items() 
                if not k.startswith(model_id)
            }
            logger.info(f"Model {model_id} unloaded")
    
    def get_prediction(self, model_id: str, symbol: str, 
                      use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate prediction for a symbol using specified model
        
        Args:
            model_id: Model to use for prediction
            symbol: Stock symbol
            use_cache: Whether to use cached predictions
            
        Returns:
            Dictionary with prediction details
        """
        # Check cache first
        cache_key = f"{model_id}_{symbol}"
        if use_cache and cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            if datetime.now() - cached['timestamp'] < self.cache_duration:
                logger.info(f"Returning cached prediction for {symbol}")
                return cached
        
        # Ensure model is loaded
        if model_id not in self.active_models:
            if not self.load_model(model_id):
                raise ValueError(f"Failed to load model {model_id}")
        
        try:
            # Fetch recent data
            lookback_days = 100  # Enough for most models
            data = self.data_pipeline.fetch_historical_data(symbol, days=lookback_days)
            
            if data is None or len(data) < 50:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Get model and generate prediction
            model = self.active_models[model_id]['model']
            
            # Generate raw predictions
            predictions = model.predict(data)
            
            # Get latest signal
            latest_signal = int(predictions.iloc[-1]) if len(predictions) > 0 else 0
            
            # Calculate signal strength if available
            signal_strength = 0.0
            if hasattr(model, 'calculate_signal_strength'):
                strength_series = model.calculate_signal_strength(data)
                signal_strength = float(strength_series.iloc[-1]) if len(strength_series) > 0 else 0.0
            
            # Get additional analysis if available
            analysis = {}
            if hasattr(model, 'get_entry_points'):
                entry_points = model.get_entry_points(data)
                if len(entry_points) > 0:
                    latest_entry = entry_points.iloc[-1]
                    analysis = latest_entry.to_dict()
            elif hasattr(model, 'get_regime_analysis'):
                regime_analysis = model.get_regime_analysis(data)
                if len(regime_analysis) > 0:
                    analysis = regime_analysis.iloc[-1].to_dict()
            
            # Build prediction result
            result = {
                'model_id': model_id,
                'symbol': symbol,
                'signal': latest_signal,
                'signal_description': self._get_signal_description(latest_signal),
                'signal_strength': signal_strength,
                'confidence': self._calculate_confidence(model, data, predictions),
                'analysis': analysis,
                'data_timestamp': data.index[-1].isoformat(),
                'prediction_timestamp': datetime.now().isoformat(),
                'latest_price': float(data['close'].iloc[-1]),
                'price_change_1d': float(data['close'].pct_change().iloc[-1])
            }
            
            # Update cache
            if use_cache:
                self.prediction_cache[cache_key] = {
                    **result,
                    'timestamp': datetime.now()
                }
            
            # Update model stats
            self.active_models[model_id]['predictions_count'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol} using {model_id}: {e}")
            raise
    
    def get_bulk_predictions(self, model_id: str, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Generate predictions for multiple symbols
        
        Args:
            model_id: Model to use
            symbols: List of symbols
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for symbol in symbols:
            try:
                pred = self.get_prediction(model_id, symbol)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to predict {symbol}: {e}")
                predictions.append({
                    'model_id': model_id,
                    'symbol': symbol,
                    'error': str(e),
                    'signal': 0,
                    'signal_description': 'ERROR'
                })
        
        return predictions
    
    def get_ensemble_prediction(self, model_ids: List[str], symbol: str, 
                               weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate ensemble prediction using multiple models
        
        Args:
            model_ids: List of model IDs to ensemble
            symbol: Stock symbol
            weights: Optional weights for each model
            
        Returns:
            Ensemble prediction dictionary
        """
        if not model_ids:
            raise ValueError("No models specified for ensemble")
        
        # Default equal weights
        if weights is None:
            weights = {model_id: 1.0 / len(model_ids) for model_id in model_ids}
        
        # Get individual predictions
        individual_predictions = []
        weighted_signal = 0.0
        weighted_strength = 0.0
        
        for model_id in model_ids:
            try:
                pred = self.get_prediction(model_id, symbol)
                individual_predictions.append(pred)
                
                weight = weights.get(model_id, 0.0)
                weighted_signal += pred['signal'] * weight
                weighted_strength += pred.get('signal_strength', 0.5) * weight
                
            except Exception as e:
                logger.error(f"Failed to get prediction from {model_id}: {e}")
        
        if not individual_predictions:
            raise ValueError("No successful predictions from ensemble models")
        
        # Determine ensemble signal
        if weighted_signal > 0.3:
            ensemble_signal = 1
        elif weighted_signal < -0.3:
            ensemble_signal = -1
        else:
            ensemble_signal = 0
        
        # Build ensemble result
        result = {
            'ensemble_models': model_ids,
            'symbol': symbol,
            'signal': ensemble_signal,
            'signal_description': self._get_signal_description(ensemble_signal),
            'signal_strength': weighted_strength,
            'weighted_signal': weighted_signal,
            'individual_predictions': individual_predictions,
            'weights': weights,
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _get_signal_description(self, signal: int) -> str:
        """Convert signal to description"""
        signal_map = {
            1: "BUY",
            0: "HOLD",
            -1: "SELL"
        }
        return signal_map.get(signal, "UNKNOWN")
    
    def _calculate_confidence(self, model, data: pd.DataFrame, 
                             predictions: pd.Series) -> float:
        """
        Calculate confidence score for prediction
        
        Simple heuristic based on:
        - Model's historical performance
        - Current market conditions
        - Signal consistency
        """
        try:
            # Base confidence from model info
            model_info = self.model_trainer.get_model_info(
                self.active_models[next(k for k, v in self.active_models.items() if v['model'] == model)]
            )
            
            if model_info and 'metrics' in model_info:
                sharpe = model_info['metrics'].get('sharpe_ratio', 0)
                win_rate = model_info['metrics'].get('win_rate', 0.5)
                
                # Base confidence from historical performance
                confidence = 0.3 * min(sharpe / 2.0, 1.0) + 0.3 * win_rate
            else:
                confidence = 0.5
            
            # Adjust for signal consistency
            recent_signals = predictions.tail(5)
            if len(recent_signals) == 5:
                consistency = abs(recent_signals.mean())
                confidence += 0.2 * consistency
            
            # Adjust for market conditions
            volatility = data['close'].pct_change().tail(20).std()
            normal_vol = 0.02  # 2% daily vol is "normal"
            vol_factor = min(normal_vol / (volatility + 0.001), 1.0)
            confidence += 0.2 * vol_factor
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Could not calculate confidence: {e}")
            return 0.5
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        status = {
            'loaded_models': {},
            'cache_size': len(self.prediction_cache),
            'service_uptime': datetime.now().isoformat()
        }
        
        for model_id, info in self.active_models.items():
            status['loaded_models'][model_id] = {
                'loaded_at': info['loaded_at'].isoformat(),
                'predictions_count': info['predictions_count'],
                'model_class': info['model'].__class__.__name__
            }
        
        return status
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")