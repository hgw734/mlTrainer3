
# model_router.py

import yaml
import json
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

ROUTING_PATH = "model_routing.yaml"
WEIGHT_PATH = "ensemble_weights.json"


def load_yaml(filepath: str) -> dict:
    """Load YAML configuration file"""
    try:
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"⚠️ {filepath} not found, using default configuration")
        return {}
    except Exception as e:
        logger.error(f"❌ Error loading {filepath}: {e}")
        return {}


def load_json(filepath: str) -> dict:
    """Load JSON configuration file"""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"⚠️ {filepath} not found, using default weights")
        return {}
    except Exception as e:
        logger.error(f"❌ Error loading {filepath}: {e}")
        return {}


class ModelRouter:
    def __init__(
            self,
            routing_path: str = ROUTING_PATH,
            weight_path: str = WEIGHT_PATH):
        self.routing_config = load_yaml(routing_path)
        self.weight_config = load_json(weight_path)

        # Fallback configurations if files don't exist
        if not self.routing_config:
            self.routing_config = self._get_default_routing_config()
        if not self.weight_config:
            self.weight_config = self._get_default_weight_config()

    def select_models(self, volatility: str, macro: str,
                      regime_score: float) -> List[str]:
        """Select models based on regime characteristics"""
        selected_models = set()

        for name, rule in self.routing_config.get("regimes", {}).items():
            vol_match = "volatility" not in rule or rule["volatility"] == volatility
            macro_match = "macro" not in rule or rule["macro"] == macro
            score_match = True

            if "score_range" in rule:
                min_score, max_score = rule["score_range"]
                score_match = min_score <= regime_score <= max_score

            if vol_match and macro_match and score_match:
                selected_models.update(rule.get("models", []))

        # Always include "always_active" models
        always_models = self.routing_config.get(
            "regimes",
            {}).get(
            "always_active",
            {}).get(
            "models",
            [])
        selected_models.update(always_models)

        logger.info(
            f"🎯 Selected models for Vol={volatility}, Macro={macro}, Score={regime_score:.1f}: {sorted(selected_models)}")
        return sorted(selected_models)

    def get_ensemble_weights(self, regime_type: str) -> Dict[str, float]:
        """Get ensemble weights for specific regime type"""
        weights = self.weight_config.get(
            regime_type, self.weight_config.get(
                "default", {}))
        logger.info(f"⚖️ Ensemble weights for {regime_type}: {weights}")
        return weights

    def explain_routing(self, volatility: str, macro: str,
                        regime_score: float) -> Dict[str, Any]:
        """Explain model routing decision"""
        models = self.select_models(volatility, macro, regime_score)

        # Determine regime type for weight selection
        regime_type = self._determine_regime_type(
            volatility, macro, regime_score)
        weights = self.get_ensemble_weights(regime_type)

        return {
            "volatility": volatility,
            "macro_signal": macro,
            "regime_score": regime_score,
            "activated_models": models,
            "regime_type": regime_type,
            "ensemble_weights": weights,
            "routing_reasoning": self._get_routing_reasoning(
                volatility,
                macro,
                regime_score)}

    def _determine_regime_type(
            self,
            volatility: str,
            macro: str,
            regime_score: float) -> str:
        """Determine regime type for weight selection"""
        if volatility == "high" and macro in ["shock", "irregular"]:
            return "crisis_vol"
        elif volatility == "high":
            return "high_vol"
        elif volatility == "low" and macro == "neutral":
            return "stable_conditions"
        elif macro == "trending":
            return "trending_conditions"
        elif macro == "macro_shift":
            return "transition_conditions"
        else:
            return "default"

    def _get_routing_reasoning(
            self,
            volatility: str,
            macro: str,
            regime_score: float) -> str:
        """Generate human-readable routing reasoning"""
        reasoning_parts = []

        if volatility == "high":
            reasoning_parts.append(
                "High volatility detected - favoring adaptive models")
        elif volatility == "low":
            reasoning_parts.append(
                "Low volatility - traditional models suitable")

        if macro == "shock":
            reasoning_parts.append(
                "Market shock conditions - emphasizing defensive models")
        elif macro == "trending":
            reasoning_parts.append(
                "Trending market - momentum models activated")
        elif macro == "neutral":
            reasoning_parts.append(
                "Neutral conditions - balanced model selection")

        if regime_score > 70:
            reasoning_parts.append(
                "High regime score - complex models preferred")
        elif regime_score < 30:
            reasoning_parts.append(
                "Low regime score - simple models preferred")

        return ". ".join(reasoning_parts) + "."

    def _get_default_routing_config(self) -> dict:
        """Default routing configuration if YAML file not found"""
        return {
            "regimes": {
                "low_vol_neutral_macro": {
                    "volatility": "low",
                    "macro": "neutral",
                    "score_range": [0, 30],
                    "models": ["ARIMA", "Prophet", "RandomForest"]
                },
                "medium_vol_trending": {
                    "volatility": "medium",
                    "macro": "trending",
                    "score_range": [31, 60],
                    "models": ["LSTM", "XGBoost", "LightGBM", "CatBoost"]
                },
                "high_vol_shock": {
                    "volatility": "high",
                    "macro": "shock",
                    "score_range": [61, 100],
                    "models": ["GRU", "CNN_LSTM", "Transformer", "DQN"]
                },
                "regime_transition": {
                    "volatility": "high",
                    "macro": "macro_shift",
                    "models": ["Autoencoder", "MetaLearner"]
                },
                "always_active": {
                    "models": ["EnsembleVoting", "MetaLearner"]
                }
            }
        }

    def _get_default_weight_config(self) -> dict:
        """Default weight configuration if JSON file not found"""
        return {
            "stable_conditions": {
                "ARIMA": 0.3,
                "Prophet": 0.3,
                "RandomForest": 0.4
            },
            "trending_conditions": {
                "LSTM": 0.3,
                "XGBoost": 0.25,
                "LightGBM": 0.25,
                "CatBoost": 0.2
            },
            "high_vol": {
                "Transformer": 0.4,
                "GRU": 0.3,
                "CNN_LSTM": 0.2,
                "DQN": 0.1
            },
            "crisis_vol": {
                "Autoencoder": 0.4,
                "XGBoost": 0.3,
                "MetaLearner": 0.3
            },
            "default": {
                "LSTM": 0.3,
                "XGBoost": 0.3,
                "EnsembleVoting": 0.4
            }
        }


# Global router instance
model_router = ModelRouter()

# Integration function for existing model_activation.py


def get_routed_models(regime_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integration function to work with existing regime detection system
    """
    try:
        # Extract regime characteristics from multidimensional profile
        from model_activation import classify_regime_dimensions

        regime_dims = classify_regime_dimensions(regime_profile)
        volatility = regime_dims["volatility"]
        macro_signal = regime_dims["macro_signal"]
        regime_score = regime_dims["regime_score"]

        # Use model router for selection
        routing_result = model_router.explain_routing(
            volatility, macro_signal, regime_score)

        # Format for compatibility with existing system
        return {
            "active_models": routing_result["activated_models"],
            "model_weights": routing_result["ensemble_weights"],
            "regime_classification": routing_result["regime_type"].upper(),
            "activation_reasoning": routing_result["routing_reasoning"],
            "regime_dimensions": regime_dims,
            "confidence": regime_profile.get('confidence', 0.7)
        }

    except ImportError:
        logger.warning(
            "⚠️ model_activation.py not found, using router directly")
        # Fallback to simplified logic
        vol_score = regime_profile.get('volatility_score', 50)
        volatility = "low" if vol_score < 30 else "high" if vol_score > 70 else "medium"
        macro_signal = "neutral"
        regime_score = vol_score

        routing_result = model_router.explain_routing(
            volatility, macro_signal, regime_score)
        return {
            "active_models": routing_result["activated_models"],
            "model_weights": routing_result["ensemble_weights"],
            "regime_classification": routing_result["regime_type"].upper(),
            "activation_reasoning": routing_result["routing_reasoning"]
        }


# Example usage and testing
if __name__ == "__main__":
    router = ModelRouter()

    # Test scenario
    regime_info = {
        "volatility": "high",
        "macro": "shock",
        "regime_score": 84.5
    }

    result = router.explain_routing(**regime_info)

    print("📊 Model Routing Explanation:")
    for k, v in result.items():
        print(f"{k}: {v}")

    print(f"\n⚖️ Ensemble Weights for {result['regime_type']}:")
    for model, weight in result["ensemble_weights"].items():
        print(f"{model}: {weight}")
