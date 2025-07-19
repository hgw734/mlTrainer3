
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

class AdaptiveEnsemble:
    """
    Adaptive ensemble that adjusts model weights based on regime performance
    Implements online learning for non-stationary financial markets
    """
    
    def __init__(self, models: Dict[str, BaseEstimator], adaptation_rate: float = 0.1):
        self.models = models
        self.adaptation_rate = adaptation_rate
        self.weights = {name: 1.0/len(models) for name in models.keys()}
        self.performance_history = {name: [] for name in models.keys()}
        self.regime_performance = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all base models"""
        logger.info("🎯 Training Adaptive Ensemble")
        
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                logger.info(f"  ✅ {name} trained successfully")
            except Exception as e:
                logger.error(f"  ❌ {name} training failed: {e}")
                
    def predict(self, X: np.ndarray, regime_info: Dict = None) -> np.ndarray:
        """Make ensemble predictions with regime-adaptive weighting"""
        predictions = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"⚠️ {name} prediction failed: {e}")
                continue
        
        if not predictions:
            logger.error("❌ No model predictions available")
            return np.zeros(len(X))
        
        # Adjust weights based on regime if provided
        effective_weights = self._get_regime_adjusted_weights(regime_info)
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = effective_weights.get(name, 0)
            ensemble_pred += weight * pred
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
            
        return ensemble_pred
    
    def online_update(self, X_new: np.ndarray, y_new: np.ndarray, 
                     regime_info: Dict = None):
        """
        Online learning update based on new data
        Adjusts model weights based on recent performance
        """
        logger.info("🔄 Performing online ensemble update")
        
        # Get predictions for new data
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_new)
                predictions[name] = pred
                
                # Calculate performance
                mse = mean_squared_error(y_new, pred)
                self.performance_history[name].append(mse)
                
                # Keep only recent history
                if len(self.performance_history[name]) > 100:
                    self.performance_history[name] = self.performance_history[name][-100:]
                    
            except Exception as e:
                logger.warning(f"⚠️ {name} online update failed: {e}")
                continue
        
        # Update weights based on recent performance
        self._update_weights()
        
        # Update regime-specific performance if regime info provided
        if regime_info:
            self._update_regime_performance(predictions, y_new, regime_info)
    
    def _update_weights(self):
        """Update model weights based on recent performance"""
        if not any(self.performance_history.values()):
            return
        
        # Calculate inverse performance scores (lower MSE = higher weight)
        scores = {}
        for name, history in self.performance_history.items():
            if history:
                recent_performance = np.mean(history[-10:])  # Last 10 observations
                scores[name] = 1.0 / (recent_performance + 1e-6)
            else:
                scores[name] = 1.0
        
        # Normalize scores to get new weights
        total_score = sum(scores.values())
        if total_score > 0:
            new_weights = {name: score / total_score for name, score in scores.items()}
            
            # Smooth update (exponential moving average)
            for name in self.weights:
                if name in new_weights:
                    self.weights[name] = (1 - self.adaptation_rate) * self.weights[name] + \
                                       self.adaptation_rate * new_weights[name]
        
        logger.info(f"📊 Updated weights: {self.weights}")
    
    def _get_regime_adjusted_weights(self, regime_info: Dict) -> Dict[str, float]:
        """Adjust weights based on current regime"""
        if not regime_info or not self.regime_performance:
            return self.weights
        
        # Extract regime characteristics
        vol_score = regime_info.get('volatility_score', 50)
        stress_score = regime_info.get('market_stress', 50)
        
        # Determine regime category
        if stress_score > 80:
            regime_category = 'crisis'
        elif vol_score > 70:
            regime_category = 'high_vol'
        elif vol_score < 30:
            regime_category = 'low_vol'
        else:
            regime_category = 'normal'
        
        # Get regime-specific weights if available
        if regime_category in self.regime_performance:
            regime_weights = self.regime_performance[regime_category]
            
            # Blend with base weights
            adjusted_weights = {}
            for name in self.weights:
                base_weight = self.weights[name]
                regime_weight = regime_weights.get(name, base_weight)
                adjusted_weights[name] = 0.7 * base_weight + 0.3 * regime_weight
            
            return adjusted_weights
        
        return self.weights
    
    def _update_regime_performance(self, predictions: Dict, y_true: np.ndarray, 
                                  regime_info: Dict):
        """Update regime-specific model performance"""
        vol_score = regime_info.get('volatility_score', 50)
        stress_score = regime_info.get('market_stress', 50)
        
        # Determine regime category
        if stress_score > 80:
            regime_category = 'crisis'
        elif vol_score > 70:
            regime_category = 'high_vol'
        elif vol_score < 30:
            regime_category = 'low_vol'
        else:
            regime_category = 'normal'
        
        # Initialize regime performance tracking
        if regime_category not in self.regime_performance:
            self.regime_performance[regime_category] = {name: [] for name in self.models.keys()}
        
        # Calculate and store performance for each model in this regime
        for name, pred in predictions.items():
            mse = mean_squared_error(y_true, pred)
            self.regime_performance[regime_category][name].append(mse)
            
            # Keep only recent regime performance
            if len(self.regime_performance[regime_category][name]) > 50:
                self.regime_performance[regime_category][name] = \
                    self.regime_performance[regime_category][name][-50:]
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about model performance across regimes"""
        insights = {
            'current_weights': self.weights,
            'regime_performance_summary': {},
            'best_models_by_regime': {},
            'overall_performance': {}
        }
        
        # Overall performance summary
        for name, history in self.performance_history.items():
            if history:
                insights['overall_performance'][name] = {
                    'avg_mse': np.mean(history),
                    'recent_mse': np.mean(history[-10:]) if len(history) >= 10 else np.mean(history),
                    'trend': 'improving' if len(history) > 10 and np.mean(history[-5:]) < np.mean(history[-10:-5]) else 'stable'
                }
        
        # Regime-specific performance
        for regime, performance in self.regime_performance.items():
            regime_summary = {}
            best_model = None
            best_performance = float('inf')
            
            for name, mse_history in performance.items():
                if mse_history:
                    avg_mse = np.mean(mse_history)
                    regime_summary[name] = avg_mse
                    
                    if avg_mse < best_performance:
                        best_performance = avg_mse
                        best_model = name
            
            insights['regime_performance_summary'][regime] = regime_summary
            insights['best_models_by_regime'][regime] = best_model
        
        return insights

def create_adaptive_ensemble_pipeline(models: Dict[str, BaseEstimator]) -> AdaptiveEnsemble:
    """
    Create an adaptive ensemble pipeline
    
    Args:
        models: Dictionary of model name -> model instance
    """
    logger.info("🎭 Creating Adaptive Ensemble Pipeline")
    
    ensemble = AdaptiveEnsemble(models, adaptation_rate=0.15)
    
    logger.info(f"✅ Adaptive ensemble created with {len(models)} models")
    return ensemble
