"""
mlTrainer - Model Router
=======================

Purpose: Advanced model routing system that dynamically selects and weights
ML models based on current market regime characteristics. Integrates with
regime detection to optimize model performance across different market conditions.

Features:
- Multi-dimensional regime analysis
- Dynamic model selection and weighting
- Real-time model activation/deactivation
- Ensemble coordination and optimization
- Performance-based routing decisions
"""

import yaml
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class ModelRouter:
    """Advanced model routing with regime-aware selection and dynamic weighting"""
    
    def __init__(self, routing_config_path: str = "model_routing.yaml", 
                 weight_config_path: str = "config/ensemble_weights.json"):
        """
        Initialize ModelRouter with configuration files
        
        Args:
            routing_config_path: Path to YAML routing configuration
            weight_config_path: Path to JSON ensemble weights configuration
        """
        self.routing_config_path = routing_config_path
        self.weight_config_path = weight_config_path
        
        # Load configurations
        self.routing_config = self._load_routing_config()
        self.weight_config = self._load_weight_config()
        
        # Routing state
        self.current_regime = None
        self.active_models = []
        self.current_weights = {}
        self.routing_history = []
        
        # Performance tracking
        self.model_performance = {}
        self.regime_performance = {}
        
        # Advanced routing parameters
        self.routing_params = {
            "confidence_threshold": 0.7,
            "min_models_per_regime": 2,
            "max_models_per_regime": 5,
            "weight_adaptation_rate": 0.1,
            "performance_window_days": 30,
            "regime_stability_threshold": 0.8
        }
        
        logger.info("ModelRouter initialized with advanced regime-aware selection")
    
    def _load_routing_config(self) -> Dict[str, Any]:
        """Load routing configuration from YAML file"""
        try:
            if os.path.exists(self.routing_config_path):
                with open(self.routing_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded routing configuration from {self.routing_config_path}")
                return config
            else:
                logger.warning(f"Routing config not found: {self.routing_config_path}")
                return self._get_default_routing_config()
        except Exception as e:
            logger.error(f"Failed to load routing config: {e}")
            return self._get_default_routing_config()
    
    def _load_weight_config(self) -> Dict[str, Any]:
        """Load ensemble weights configuration from JSON file"""
        try:
            if os.path.exists(self.weight_config_path):
                with open(self.weight_config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded weight configuration from {self.weight_config_path}")
                return config
            else:
                logger.warning(f"Weight config not found: {self.weight_config_path}")
                return self._get_default_weight_config()
        except Exception as e:
            logger.error(f"Failed to load weight config: {e}")
            return self._get_default_weight_config()
    
    def _get_default_routing_config(self) -> Dict[str, Any]:
        """Default routing configuration if file not available"""
        return {
            "regimes": {
                "stable_low_vol": {
                    "volatility": "low",
                    "macro_signal": "neutral",
                    "score_range": [0, 30],
                    "models": ["RandomForest", "Prophet", "LinearRegression"],
                    "description": "Stable market with low volatility",
                    "confidence_multiplier": 1.2
                },
                "trending_medium_vol": {
                    "volatility": "medium", 
                    "macro_signal": "trending",
                    "score_range": [30, 60],
                    "models": ["LSTM", "XGBoost", "LightGBM", "CatBoost"],
                    "description": "Trending market with moderate volatility",
                    "confidence_multiplier": 1.0
                },
                "volatile_high_stress": {
                    "volatility": "high",
                    "macro_signal": ["shock", "irregular"],
                    "score_range": [60, 100],
                    "models": ["GRU", "CNN_LSTM", "Transformer", "DQN", "AdaptiveEnsemble"],
                    "description": "High volatility crisis conditions",
                    "confidence_multiplier": 0.8
                },
                "transition_regime": {
                    "volatility": "high",
                    "macro_signal": "macro_shift",
                    "models": ["Autoencoder", "MetaLearner", "AdaptiveEnsemble"],
                    "description": "Market regime transition phase",
                    "confidence_multiplier": 0.6
                },
                "always_active": {
                    "models": ["EnsembleVoting", "MetaLearner"],
                    "description": "Models that are always active for robustness",
                    "weight_factor": 0.3
                }
            },
            "model_characteristics": {
                "RandomForest": {
                    "regime_preference": ["stable"],
                    "volatility_tolerance": "low",
                    "complexity": "medium",
                    "interpretability": "high"
                },
                "XGBoost": {
                    "regime_preference": ["trending", "volatile"],
                    "volatility_tolerance": "medium",
                    "complexity": "high",
                    "interpretability": "medium"
                },
                "LSTM": {
                    "regime_preference": ["trending", "volatile"],
                    "volatility_tolerance": "high",
                    "complexity": "high",
                    "interpretability": "low"
                },
                "Transformer": {
                    "regime_preference": ["volatile", "transition"],
                    "volatility_tolerance": "very_high",
                    "complexity": "very_high",
                    "interpretability": "low"
                }
            },
            "routing_rules": {
                "min_confidence_threshold": 0.6,
                "max_models_per_prediction": 5,
                "ensemble_weight_normalization": True,
                "adaptive_weighting": True,
                "performance_based_selection": True
            }
        }
    
    def _get_default_weight_config(self) -> Dict[str, Any]:
        """Default ensemble weights configuration"""
        return {
            "stable_conditions": {
                "RandomForest": 0.35,
                "Prophet": 0.25,
                "LinearRegression": 0.20,
                "EnsembleVoting": 0.20
            },
            "trending_conditions": {
                "LSTM": 0.30,
                "XGBoost": 0.25,
                "LightGBM": 0.25,
                "CatBoost": 0.20
            },
            "high_volatility": {
                "Transformer": 0.35,
                "GRU": 0.25,
                "CNN_LSTM": 0.20,
                "DQN": 0.20
            },
            "crisis_conditions": {
                "Autoencoder": 0.40,
                "AdaptiveEnsemble": 0.30,
                "MetaLearner": 0.30
            },
            "transition_conditions": {
                "MetaLearner": 0.40,
                "Autoencoder": 0.30,
                "AdaptiveEnsemble": 0.30
            },
            "default": {
                "LSTM": 0.25,
                "XGBoost": 0.25,
                "EnsembleVoting": 0.25,
                "MetaLearner": 0.25
            }
        }
    
    def select_models(self, volatility: str, macro_signal: str, 
                     regime_score: float, confidence: float = 1.0) -> List[str]:
        """
        Select optimal models based on regime characteristics with advanced logic
        
        Args:
            volatility: Market volatility level (low/medium/high)
            macro_signal: Macro economic signal (neutral/trending/shock/macro_shift/irregular)
            regime_score: Regime score from 0-100
            confidence: Confidence in regime classification (0-1)
            
        Returns:
            List of selected model names
        """
        selected_models = set()
        regime_matches = []
        
        # Find matching regimes
        regimes = self.routing_config.get("regimes", {})
        for regime_name, regime_config in regimes.items():
            if regime_name == "always_active":
                continue  # Handle separately
                
            match_score = self._calculate_regime_match_score(
                regime_config, volatility, macro_signal, regime_score
            )
            
            if match_score > 0:
                regime_matches.append((regime_name, regime_config, match_score))
        
        # Sort by match score
        regime_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Select models from best matching regimes
        max_models = self.routing_params["max_models_per_regime"]
        min_models = self.routing_params["min_models_per_regime"]
        
        for regime_name, regime_config, match_score in regime_matches:
            regime_models = regime_config.get("models", [])
            
            # Apply confidence multiplier
            confidence_mult = regime_config.get("confidence_multiplier", 1.0)
            adjusted_confidence = confidence * confidence_mult
            
            if adjusted_confidence >= self.routing_params["confidence_threshold"]:
                # Add models weighted by match score
                num_models = min(len(regime_models), max_models)
                selected_models.update(regime_models[:num_models])
            
            # Stop if we have enough models
            if len(selected_models) >= max_models:
                break
        
        # Always include always_active models
        always_active = regimes.get("always_active", {})
        if always_active:
            always_models = always_active.get("models", [])
            selected_models.update(always_models)
        
        # Ensure minimum number of models
        final_models = list(selected_models)
        if len(final_models) < min_models:
            # Add default models if needed
            default_models = ["RandomForest", "XGBoost", "EnsembleVoting"]
            for model in default_models:
                if model not in final_models:
                    final_models.append(model)
                    if len(final_models) >= min_models:
                        break
        
        # Update routing state
        self.active_models = final_models
        self.current_regime = {
            "volatility": volatility,
            "macro_signal": macro_signal,
            "regime_score": regime_score,
            "confidence": confidence
        }
        
        # Log routing decision
        self._log_routing_decision(final_models, regime_matches)
        
        return final_models
    
    def _calculate_regime_match_score(self, regime_config: Dict[str, Any], 
                                    volatility: str, macro_signal: str, 
                                    regime_score: float) -> float:
        """Calculate how well current conditions match a regime configuration"""
        match_score = 0.0
        
        # Volatility match
        if "volatility" in regime_config:
            if regime_config["volatility"] == volatility:
                match_score += 0.4
        
        # Macro signal match
        if "macro_signal" in regime_config:
            config_macro = regime_config["macro_signal"]
            if isinstance(config_macro, list):
                if macro_signal in config_macro:
                    match_score += 0.4
            elif config_macro == macro_signal:
                match_score += 0.4
        
        # Score range match
        if "score_range" in regime_config:
            min_score, max_score = regime_config["score_range"]
            if min_score <= regime_score <= max_score:
                match_score += 0.2
                # Bonus for being in the center of the range
                range_center = (min_score + max_score) / 2
                distance_from_center = abs(regime_score - range_center)
                max_distance = (max_score - min_score) / 2
                if max_distance > 0:
                    center_bonus = 0.1 * (1 - distance_from_center / max_distance)
                    match_score += center_bonus
        
        return match_score
    
    def get_ensemble_weights(self, regime_type: str = None, 
                           selected_models: List[str] = None) -> Dict[str, float]:
        """
        Get optimized ensemble weights for selected models
        
        Args:
            regime_type: Type of regime for weight selection
            selected_models: List of models to weight
            
        Returns:
            Dictionary of model names to weights
        """
        if regime_type is None:
            regime_type = self._determine_regime_type()
        
        if selected_models is None:
            selected_models = self.active_models
        
        # Get base weights from configuration
        base_weights = self.weight_config.get(
            regime_type, 
            self.weight_config.get("default", {})
        )
        
        # Filter weights for selected models
        model_weights = {}
        total_weight = 0.0
        
        for model in selected_models:
            weight = base_weights.get(model, 0.0)
            if weight > 0:
                model_weights[model] = weight
                total_weight += weight
        
        # Handle models not in base weights
        unweighted_models = [m for m in selected_models if m not in model_weights]
        if unweighted_models:
            default_weight = 1.0 / len(selected_models)
            for model in unweighted_models:
                model_weights[model] = default_weight
                total_weight += default_weight
        
        # Normalize weights
        if total_weight > 0:
            for model in model_weights:
                model_weights[model] = model_weights[model] / total_weight
        
        # Apply performance-based adjustments
        if self.routing_config.get("routing_rules", {}).get("performance_based_selection", True):
            model_weights = self._adjust_weights_by_performance(model_weights)
        
        self.current_weights = model_weights
        return model_weights
    
    def _adjust_weights_by_performance(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust ensemble weights based on recent model performance"""
        if not self.model_performance:
            return base_weights
        
        adjusted_weights = base_weights.copy()
        adjustment_rate = self.routing_params["weight_adaptation_rate"]
        
        for model, base_weight in base_weights.items():
            if model in self.model_performance:
                performance = self.model_performance[model]
                accuracy = performance.get("accuracy", 0.5)
                
                # Calculate performance adjustment (-0.5 to +0.5)
                performance_adjustment = (accuracy - 0.5) * adjustment_rate
                
                # Apply adjustment
                new_weight = base_weight * (1 + performance_adjustment)
                adjusted_weights[model] = max(0.01, new_weight)  # Minimum weight
        
        # Renormalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for model in adjusted_weights:
                adjusted_weights[model] = adjusted_weights[model] / total_weight
        
        return adjusted_weights
    
    def _determine_regime_type(self) -> str:
        """Determine regime type for weight selection from current regime"""
        if not self.current_regime:
            return "default"
        
        volatility = self.current_regime.get("volatility", "medium")
        macro_signal = self.current_regime.get("macro_signal", "neutral")
        regime_score = self.current_regime.get("regime_score", 50)
        
        # Mapping logic
        if volatility == "high" and macro_signal in ["shock", "irregular"]:
            return "crisis_conditions"
        elif volatility == "high":
            return "high_volatility"
        elif volatility == "low" and macro_signal == "neutral":
            return "stable_conditions"
        elif macro_signal == "trending":
            return "trending_conditions"
        elif macro_signal == "macro_shift":
            return "transition_conditions"
        else:
            return "default"
    
    def explain_routing(self, volatility: str, macro_signal: str, 
                       regime_score: float, confidence: float = 1.0) -> Dict[str, Any]:
        """
        Provide detailed explanation of routing decision
        
        Args:
            volatility: Market volatility level
            macro_signal: Macro economic signal
            regime_score: Regime score from 0-100
            confidence: Confidence in regime classification
            
        Returns:
            Comprehensive routing explanation
        """
        # Select models
        selected_models = self.select_models(volatility, macro_signal, regime_score, confidence)
        
        # Determine regime type and get weights
        regime_type = self._determine_regime_type()
        ensemble_weights = self.get_ensemble_weights(regime_type, selected_models)
        
        # Generate reasoning
        routing_reasoning = self._generate_detailed_reasoning(
            volatility, macro_signal, regime_score, confidence, selected_models
        )
        
        # Find matching regimes for explanation
        regimes = self.routing_config.get("regimes", {})
        matching_regimes = []
        
        for regime_name, regime_config in regimes.items():
            if regime_name == "always_active":
                continue
            match_score = self._calculate_regime_match_score(
                regime_config, volatility, macro_signal, regime_score
            )
            if match_score > 0:
                matching_regimes.append({
                    "name": regime_name,
                    "match_score": round(match_score, 3),
                    "description": regime_config.get("description", ""),
                    "models": regime_config.get("models", [])
                })
        
        matching_regimes.sort(key=lambda x: x["match_score"], reverse=True)
        
        explanation = {
            "input_parameters": {
                "volatility": volatility,
                "macro_signal": macro_signal,
                "regime_score": regime_score,
                "confidence": confidence
            },
            "activated_models": selected_models,
            "regime_type": regime_type,
            "ensemble_weights": ensemble_weights,
            "routing_reasoning": routing_reasoning,
            "matching_regimes": matching_regimes[:3],  # Top 3 matches
            "model_selection_criteria": {
                "min_confidence_threshold": self.routing_params["confidence_threshold"],
                "max_models": self.routing_params["max_models_per_regime"],
                "performance_based": True,
                "adaptive_weighting": True
            },
            "routing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "router_version": "2.0",
                "config_source": self.routing_config_path
            }
        }
        
        # Store in routing history
        self.routing_history.append(explanation.copy())
        
        return explanation
    
    def _generate_detailed_reasoning(self, volatility: str, macro_signal: str, 
                                   regime_score: float, confidence: float, 
                                   selected_models: List[str]) -> str:
        """Generate detailed human-readable routing reasoning"""
        reasoning_parts = []
        
        # Volatility analysis
        if volatility == "high":
            reasoning_parts.append(
                "High volatility environment detected - prioritizing adaptive and robust models "
                "capable of handling rapid market changes and non-linear patterns"
            )
        elif volatility == "low":
            reasoning_parts.append(
                "Low volatility conditions - traditional statistical models and "
                "trend-following approaches are well-suited for this stable environment"
            )
        else:
            reasoning_parts.append(
                "Moderate volatility detected - employing balanced model selection "
                "with both traditional and advanced machine learning approaches"
            )
        
        # Macro signal analysis
        macro_explanations = {
            "shock": "Market shock conditions require defensive models with strong risk management",
            "trending": "Trending markets favor momentum-based models and pattern recognition systems",
            "neutral": "Neutral macro environment allows for diverse model approaches",
            "macro_shift": "Regime transition detected - employing adaptive models for changing conditions",
            "irregular": "Irregular market patterns require sophisticated non-linear models"
        }
        
        if macro_signal in macro_explanations:
            reasoning_parts.append(macro_explanations[macro_signal])
        
        # Regime score analysis
        if regime_score > 80:
            reasoning_parts.append(
                f"Very high regime score ({regime_score}) indicates complex market conditions "
                "requiring advanced models with sophisticated pattern recognition capabilities"
            )
        elif regime_score > 60:
            reasoning_parts.append(
                f"Elevated regime score ({regime_score}) suggests moderately complex conditions "
                "balancing traditional and advanced modeling approaches"
            )
        elif regime_score < 30:
            reasoning_parts.append(
                f"Low regime score ({regime_score}) indicates stable conditions "
                "where simpler, more interpretable models can be effective"
            )
        
        # Confidence impact
        if confidence < 0.7:
            reasoning_parts.append(
                f"Lower confidence ({confidence:.2f}) in regime classification "
                "results in more conservative model selection with higher diversity"
            )
        elif confidence > 0.9:
            reasoning_parts.append(
                f"High confidence ({confidence:.2f}) in regime classification "
                "allows for more specialized model selection optimized for current conditions"
            )
        
        # Model selection summary
        reasoning_parts.append(
            f"Selected {len(selected_models)} models: {', '.join(selected_models)} "
            f"based on regime-specific suitability and performance characteristics"
        )
        
        return ". ".join(reasoning_parts) + "."
    
    def _log_routing_decision(self, selected_models: List[str], 
                            regime_matches: List[Tuple[str, Dict, float]]):
        """Log routing decision for debugging and analysis"""
        logger.info(
            f"🎯 Model Router Decision: Selected {len(selected_models)} models - "
            f"{', '.join(selected_models)}"
        )
        
        if regime_matches:
            best_match = regime_matches[0]
            logger.info(
                f"🎯 Best regime match: {best_match[0]} (score: {best_match[2]:.3f})"
            )
        
        logger.debug(f"🎯 Full routing state: {self.current_regime}")
    
    def update_model_performance(self, model_name: str, performance_metrics: Dict[str, float]):
        """Update performance metrics for a model"""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {}
        
        self.model_performance[model_name].update(performance_metrics)
        self.model_performance[model_name]["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"🎯 Updated performance for {model_name}: {performance_metrics}")
    
    def get_routing_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get routing statistics and analytics"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_decisions = []
        for decision in self.routing_history:
            try:
                decision_time = datetime.fromisoformat(
                    decision["routing_metadata"]["timestamp"]
                )
                if decision_time > cutoff_date:
                    recent_decisions.append(decision)
            except:
                continue
        
        if not recent_decisions:
            return {
                "period_days": days,
                "total_decisions": 0,
                "message": "No routing decisions in specified period"
            }
        
        # Analyze decisions
        model_usage = {}
        regime_usage = {}
        
        for decision in recent_decisions:
            # Count model usage
            for model in decision.get("activated_models", []):
                model_usage[model] = model_usage.get(model, 0) + 1
            
            # Count regime usage
            regime = decision.get("regime_type", "unknown")
            regime_usage[regime] = regime_usage.get(regime, 0) + 1
        
        statistics = {
            "period_days": days,
            "total_decisions": len(recent_decisions),
            "model_usage_frequency": model_usage,
            "regime_distribution": regime_usage,
            "most_used_model": max(model_usage.items(), key=lambda x: x[1])[0] if model_usage else None,
            "most_common_regime": max(regime_usage.items(), key=lambda x: x[1])[0] if regime_usage else None,
            "avg_models_per_decision": np.mean([
                len(d.get("activated_models", [])) for d in recent_decisions
            ]) if recent_decisions else 0,
            "unique_models_used": len(model_usage),
            "unique_regimes_encountered": len(regime_usage),
            "timestamp": datetime.now().isoformat()
        }
        
        return statistics
    
    def save_routing_config(self):
        """Save current routing configuration to file"""
        try:
            with open(self.routing_config_path, 'w') as f:
                yaml.dump(self.routing_config, f, default_flow_style=False, indent=2)
            logger.info(f"Saved routing configuration to {self.routing_config_path}")
        except Exception as e:
            logger.error(f"Failed to save routing configuration: {e}")
    
    def save_weight_config(self):
        """Save current weight configuration to file"""
        try:
            with open(self.weight_config_path, 'w') as f:
                json.dump(self.weight_config, f, indent=2)
            logger.info(f"Saved weight configuration to {self.weight_config_path}")
        except Exception as e:
            logger.error(f"Failed to save weight configuration: {e}")


# Global router instance for easy importing
global_model_router = ModelRouter()

# Convenience functions for backward compatibility
def select_models(volatility: str, macro_signal: str, regime_score: float) -> List[str]:
    """Convenience function for model selection"""
    return global_model_router.select_models(volatility, macro_signal, regime_score)

def get_ensemble_weights(regime_type: str) -> Dict[str, float]:
    """Convenience function for getting ensemble weights"""
    return global_model_router.get_ensemble_weights(regime_type)

def explain_routing(volatility: str, macro_signal: str, regime_score: float) -> Dict[str, Any]:
    """Convenience function for routing explanation"""
    return global_model_router.explain_routing(volatility, macro_signal, regime_score)


# Example usage and testing
if __name__ == "__main__":
    # Initialize router
    router = ModelRouter()
    
    # Test routing scenarios
    test_scenarios = [
        {
            "name": "Stable Market",
            "volatility": "low",
            "macro_signal": "neutral", 
            "regime_score": 25,
            "confidence": 0.9
        },
        {
            "name": "Trending Market",
            "volatility": "medium",
            "macro_signal": "trending",
            "regime_score": 45,
            "confidence": 0.8
        },
        {
            "name": "Crisis Conditions",
            "volatility": "high", 
            "macro_signal": "shock",
            "regime_score": 85,
            "confidence": 0.7
        },
        {
            "name": "Regime Transition",
            "volatility": "high",
            "macro_signal": "macro_shift", 
            "regime_score": 75,
            "confidence": 0.6
        }
    ]
    
    print("🎯 ModelRouter Testing Results")
    print("=" * 50)
    
    for scenario in test_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        result = router.explain_routing(
            scenario["volatility"],
            scenario["macro_signal"], 
            scenario["regime_score"],
            scenario["confidence"]
        )
        
        print(f"Selected Models: {result['activated_models']}")
        print(f"Regime Type: {result['regime_type']}")
        print(f"Ensemble Weights: {result['ensemble_weights']}")
        print(f"Reasoning: {result['routing_reasoning'][:100]}...")
    
    # Show statistics
    print(f"\n📈 Routing Statistics:")
    stats = router.get_routing_statistics()
    print(f"Total Decisions: {stats['total_decisions']}")
    print(f"Most Used Model: {stats.get('most_used_model', 'N/A')}")
    print(f"Most Common Regime: {stats.get('most_common_regime', 'N/A')}")
