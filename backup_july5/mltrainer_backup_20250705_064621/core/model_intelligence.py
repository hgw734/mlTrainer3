"""
mlTrainer - Model Technical Registry
===================================

Purpose: Technical registry of available ML models and their implementation characteristics.
Provides mlTrainer with factual information about model capabilities and requirements.

This system provides mlTrainer with:
- Available models and their technical specifications
- Implementation requirements (data, compute, time)
- Model strengths and limitations
- Technical integration capabilities

All strategy decisions, combinations, and applications are made by mlTrainer.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ModelIntelligence:
    """Complete intelligence system for all models - uses centralized registry as source of truth"""
    
    def __init__(self):
        # Get centralized model registry - SINGLE SOURCE OF TRUTH
        from .model_registry import get_model_registry
        self.model_registry = get_model_registry()
        
        # Build intelligence on top of centralized registry
        self.model_knowledge = self._build_intelligence_from_registry()
        self.combination_strategies = self._initialize_combination_strategies()
        self.market_condition_mappings = self._initialize_market_mappings()
        self.ensemble_recipes = self._initialize_ensemble_recipes()
        
        logger.info("ModelIntelligence initialized with complete model knowledge")
    
    def _build_intelligence_from_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build intelligence layer on top of centralized model registry"""
        intelligence = {}
        
        # Get all models from centralized registry
        for model_name in self.model_registry.get_all_models():
            model_info = self.model_registry.get_model_info(model_name)
            
            # Add intelligence layer to registry data
            intelligence[model_name] = {
                "category": model_info.get("category", "Unknown"),
                "strength": self._get_model_strength(model_name, model_info),
                "weakness": self._get_model_weakness(model_name, model_info),
                "best_for": self._get_model_best_for(model_name, model_info),
                "market_conditions": self._get_market_conditions(model_name, model_info),
                "training_time": model_info.get("training_time", "medium"),
                "interpretability": model_info.get("interpretability", "medium"),
                "data_requirements": model_info.get("data_requirements", "medium"),
                "ensemble_weight": self._calculate_ensemble_weight(model_name, model_info)
            }
        
        return intelligence
    
    def _get_model_strength(self, model_name: str, model_info: Dict) -> str:
        """Get model strength based on type and category"""
        category = model_info.get("category", "")
        model_type = model_info.get("type", "")
        
        strength_map = {
            "RandomForest": "Robust to overfitting, handles mixed data types",
            "XGBoost": "High performance, handles missing values well",
            "LightGBM": "Fast training, memory efficient",
            "LSTM": "Captures long-term dependencies, sequential patterns",
            "ARIMA": "Statistical rigor, time series forecasting",
            "Prophet": "Handles seasonality, missing data robust"
        }
        
        return strength_map.get(model_name, f"General {category.lower()} capabilities")
    
    def _get_model_weakness(self, model_name: str, model_info: Dict) -> str:
        """Get model weakness based on type and category"""
        weakness_map = {
            "RandomForest": "Can overfit with very noisy data",
            "XGBoost": "Prone to overfitting without proper tuning",
            "LightGBM": "Sensitive to hyperparameters",
            "LSTM": "Requires large datasets, slower training",
            "ARIMA": "Requires stationarity assumptions",
            "Prophet": "Less flexible for complex patterns"
        }
        
        return weakness_map.get(model_name, "Requires proper tuning and validation")
    
    def _get_model_best_for(self, model_name: str, model_info: Dict) -> List[str]:
        """Get what the model is best for"""
        best_for_map = {
            "RandomForest": ["Medium-term predictions", "Feature importance analysis", "Stable markets"],
            "XGBoost": ["High accuracy requirements", "Competitions", "Complex patterns"],
            "LightGBM": ["Large datasets", "Fast training", "Memory constraints"],
            "LSTM": ["Time series prediction", "Long-term trends", "Sequential data"],
            "ARIMA": ["Traditional forecasting", "Statistical analysis", "Short-term predictions"],
            "Prophet": ["Business forecasting", "Holiday effects", "Trend analysis"]
        }
        
        return best_for_map.get(model_name, ["General machine learning tasks"])
    
    def _get_market_conditions(self, model_name: str, model_info: Dict) -> List[str]:
        """Get suitable market conditions for the model"""
        conditions_map = {
            "RandomForest": ["stable", "trending"],
            "XGBoost": ["volatile", "complex"],
            "LightGBM": ["trending", "stable"],
            "LSTM": ["trending", "cyclical"],
            "ARIMA": ["stable", "predictable"],
            "Prophet": ["trending", "seasonal"]
        }
        
        return conditions_map.get(model_name, ["general"])
    
    def _calculate_ensemble_weight(self, model_name: str, model_info: Dict) -> float:
        """Calculate ensemble weight based on model characteristics"""
        # Base weights by category
        category_weights = {
            "Tree-Based": 0.15,
            "Deep Learning": 0.20,
            "Time Series": 0.18,
            "Traditional ML": 0.12,
            "Ensemble": 0.25,
            "Advanced Models": 0.16
        }
        
        category = model_info.get("category", "Traditional ML")
        return category_weights.get(category, 0.15)
    
    def _initialize_combination_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Strategic model combinations for different objectives"""
        return {
            "maximum_accuracy": {
                "models": ["Transformer", "StackingEnsemble", "MultiHeadAttention", "LSTM", "XGBoost"],
                "weights": [0.25, 0.25, 0.20, 0.15, 0.15],
                "description": "Highest possible accuracy using deep learning and ensemble methods",
                "use_cases": ["Competition settings", "Maximum performance required", "Complex patterns"],
                "training_time": "very_slow",
                "data_requirements": "very_large"
            },
            
            "robust_trading": {
                "models": ["RandomForest", "XGBoost", "LightGBM", "ARIMA", "Prophet"],
                "weights": [0.25, 0.30, 0.20, 0.15, 0.10],
                "description": "Balanced approach for consistent trading performance",
                "use_cases": ["Live trading", "Robust predictions", "Mixed market conditions"],
                "training_time": "medium",
                "data_requirements": "medium"
            },
            
            "fast_execution": {
                "models": ["LinearRegression", "RandomForest", "LightGBM", "Prophet"],
                "weights": [0.20, 0.30, 0.35, 0.15],
                "description": "Fast training and inference for real-time applications",
                "use_cases": ["Real-time trading", "High-frequency", "Quick decisions"],
                "training_time": "fast",
                "data_requirements": "small"
            },
            
            "interpretable_predictions": {
                "models": ["LinearRegression", "DecisionTree", "RandomForest", "ARIMA"],
                "weights": [0.30, 0.25, 0.25, 0.20],
                "description": "Highly interpretable models for understanding decisions",
                "use_cases": ["Regulatory compliance", "Risk management", "Explanation required"],
                "training_time": "fast",
                "data_requirements": "small"
            },
            
            "volatile_markets": {
                "models": ["XGBoost", "LightGBM", "LSTM", "GRU", "Transformer"],
                "weights": [0.25, 0.20, 0.25, 0.15, 0.15],
                "description": "Specialized for high volatility and rapid changes",
                "use_cases": ["Volatile markets", "Crisis periods", "High uncertainty"],
                "training_time": "slow",
                "data_requirements": "large"
            },
            
            "trend_following": {
                "models": ["LSTM", "GRU", "ARIMA", "Prophet", "XGBoost"],
                "weights": [0.30, 0.20, 0.20, 0.15, 0.15],
                "description": "Optimized for trend detection and following",
                "use_cases": ["Trend following", "Momentum strategies", "Long-term positions"],
                "training_time": "medium",
                "data_requirements": "medium"
            },
            
            "regime_detection": {
                "models": ["HiddenMarkovModel", "BayesianChangePointDetection", "KMeansClustering", "LSTM"],
                "weights": [0.35, 0.25, 0.20, 0.20],
                "description": "Specialized for detecting market regime changes",
                "use_cases": ["Regime analysis", "Market phase detection", "Strategy switching"],
                "training_time": "medium",
                "data_requirements": "large"
            }
        }
    
    def _initialize_market_mappings(self) -> Dict[str, List[str]]:
        """Technical specifications for model characteristics under different market conditions"""
        return {
            "stable_market": ["RandomForest", "LinearRegression", "Ridge", "ARIMA", "DecisionTree"],
            "trending_market": ["LSTM", "GRU", "Prophet", "XGBoost", "LightGBM"],
            "volatile_market": ["XGBoost", "LightGBM", "Transformer", "SVR", "GaussianMixtureModel"],
            "complex_patterns": ["Transformer", "CNN_LSTM", "StackingEnsemble", "DuelingDQN", "GraphNeuralNetworks"],
            "high_frequency": ["LightGBM", "LinearRegression", "RandomForest", "KNearestNeighbors"],
            "interpretable_required": ["LinearRegression", "DecisionTree", "RandomForest", "LogisticRegression", "ARIMA"],
            "large_datasets": ["LightGBM", "XGBoost", "LSTM", "Transformer", "NeuralArchitectureSearch"],
            "small_datasets": ["LinearRegression", "Ridge", "Lasso", "DecisionTree", "KNearestNeighbors"],
            "real_time": ["LightGBM", "RandomForest", "LinearRegression", "OnlineLearningAlgorithms"],
            "regime_change": ["HiddenMarkovModel", "BayesianChangePointDetection", "ThresholdModels", "DynamicEnsemble"]
        }
    
    def _initialize_ensemble_recipes(self) -> Dict[str, Dict[str, Any]]:
        """Pre-configured ensemble recipes for different accuracy targets"""
        return {
            "ultra_high_accuracy": {
                "models": ["StackingEnsemble", "Transformer", "DiffusionModels", "NeuralArchitectureSearch"],
                "ensemble_method": "stacking",
                "meta_learner": "XGBoost",
                "description": "Maximum possible accuracy using cutting-edge methods",
                "expected_improvement": "30-50% over single models",
                "computational_cost": "very_high",
                "suitable_for": ["Competition", "Research", "Maximum performance requirements"]
            },
            
            "production_ready": {
                "models": ["RandomForest", "XGBoost", "LightGBM", "LSTM"],
                "ensemble_method": "voting",
                "meta_learner": None,
                "description": "Balanced accuracy and reliability for production",
                "expected_improvement": "10-20% over single models",
                "computational_cost": "medium",
                "suitable_for": ["Live trading", "Production systems", "Reliable performance"]
            },
            
            "fast_ensemble": {
                "models": ["RandomForest", "LightGBM", "LinearRegression"],
                "ensemble_method": "simple_average",
                "meta_learner": None,
                "description": "Fast ensemble for real-time applications",
                "expected_improvement": "5-15% over single models", 
                "computational_cost": "low",
                "suitable_for": ["Real-time trading", "High-frequency", "Speed critical"]
            }
        }
    
    def get_model_recommendations(self, market_condition: str, accuracy_target: float = 0.85, 
                                speed_priority: str = "medium") -> Dict[str, Any]:
        """Get intelligent model recommendations based on conditions"""
        
        # Get suitable models for market condition
        suitable_models = self.market_condition_mappings.get(market_condition, [])
        
        if not suitable_models:
            # Fallback to general recommendations
            if speed_priority == "high":
                suitable_models = self.market_condition_mappings["high_frequency"]
            elif accuracy_target > 0.90:
                suitable_models = self.market_condition_mappings["complex_patterns"]
            else:
                suitable_models = ["RandomForest", "XGBoost", "LightGBM"]
        
        # Get ensemble strategy
        ensemble_strategy = self._get_optimal_ensemble_strategy(accuracy_target, market_condition)
        
        # Generate recommendation
        recommendation = {
            "primary_models": suitable_models[:5],  # Top 5 recommendations
            "ensemble_strategy": ensemble_strategy,
            "reasoning": self._generate_recommendation_reasoning(market_condition, accuracy_target, suitable_models),
            "market_condition": market_condition,
            "accuracy_target": accuracy_target,
            "speed_priority": speed_priority
        }
        
        return recommendation
    
    def get_ensemble_strategy(self, objective: str) -> Dict[str, Any]:
        """Get specific ensemble strategy"""
        return self.combination_strategies.get(objective, self.combination_strategies["robust_trading"])
    
    def get_high_accuracy_recipe(self, target_accuracy: float = 0.65) -> Dict[str, Any]:
        """Get recipe for achieving specific accuracy targets"""
        if target_accuracy >= 0.95:
            return self.ensemble_recipes["ultra_high_accuracy"]
        elif target_accuracy >= 0.80:
            return self.ensemble_recipes["production_ready"]
        else:
            return self.ensemble_recipes["fast_ensemble"]
    
    def _get_optimal_ensemble_strategy(self, accuracy_target: float, market_condition: str) -> str:
        """Determine optimal ensemble strategy"""
        if accuracy_target >= 0.95:
            return "maximum_accuracy"
        elif market_condition in ["volatile_market", "complex_patterns"]:
            return "volatile_markets"
        elif market_condition == "trending_market":
            return "trend_following"
        elif market_condition == "regime_change":
            return "regime_detection"
        else:
            return "robust_trading"
    
    def _generate_recommendation_reasoning(self, market_condition: str, accuracy_target: float, 
                                         models: List[str]) -> str:
        """Generate human-readable reasoning for recommendations"""
        reasoning = f"For {market_condition} conditions with {accuracy_target:.1%} accuracy target:\n\n"
        
        reasoning += f"Selected {len(models)} models based on:\n"
        reasoning += f"• Market condition compatibility\n"
        reasoning += f"• Performance requirements\n"
        reasoning += f"• Computational constraints\n\n"
        
        reasoning += f"Primary models: {', '.join(models[:3])}\n"
        reasoning += f"These models are optimized for the specified conditions and requirements."
        
        return reasoning
    
    def get_all_models_summary(self) -> Dict[str, Any]:
        """Get complete summary of all models with intelligent categorization"""
        
        summary = {
            "total_models": len(self.model_knowledge),
            "categories": {},
            "speed_tiers": {"very_fast": [], "fast": [], "medium": [], "slow": [], "very_slow": []},
            "interpretability_levels": {"very_high": [], "high": [], "medium": [], "low": [], "very_low": []},
            "data_requirements": {"small": [], "medium": [], "large": [], "very_large": []},
            "ensemble_strategies": list(self.combination_strategies.keys()),
            "market_mappings": list(self.market_condition_mappings.keys())
        }
        
        # Categorize models by various criteria
        for model_name, model_info in self.model_knowledge.items():
            # Category grouping
            category = model_info["category"]
            if category not in summary["categories"]:
                summary["categories"][category] = []
            summary["categories"][category].append(model_name)
            
            # Speed grouping
            speed = model_info["training_time"]
            if speed in summary["speed_tiers"]:
                summary["speed_tiers"][speed].append(model_name)
            
            # Interpretability grouping
            interp = model_info["interpretability"]
            if interp in summary["interpretability_levels"]:
                summary["interpretability_levels"][interp].append(model_name)
            
            # Data requirements grouping
            data_req = model_info["data_requirements"]
            if data_req in summary["data_requirements"]:
                summary["data_requirements"][data_req].append(model_name)
        
        return summary

def get_model_intelligence() -> ModelIntelligence:
    """Get global ModelIntelligence instance"""
    global _model_intelligence_instance
    if '_model_intelligence_instance' not in globals():
        _model_intelligence_instance = ModelIntelligence()
    return _model_intelligence_instance