
import logging
from typing import List, Dict, Any
import yaml
import os

logger = logging.getLogger(__name__)


def determine_models(
        volatility: str,
        macro_signal: str,
        regime_score: float) -> list:
    """
    Determine active models based on multidimensional regime analysis

    Args:
        volatility: "low", "medium", "high"
        macro_signal: "neutral", "trending", "irregular", "shock", "macro_shift"
        regime_score: float 0-100
    """
    models = []

    # Stable regime
    if volatility == "low" and macro_signal == "neutral" and regime_score <= 30:
        models += ["ARIMA", "Prophet", "RandomForest"]

    # Neutral / trend-following regime
    elif volatility == "medium" and macro_signal == "trending":
        models += ["LSTM", "XGBoost", "LightGBM", "CatBoost"]

    # Momentum breakout regime
    elif volatility == "high" and macro_signal in ["irregular", "shock"] and regime_score > 60:
        models += ["GRU", "CNN_LSTM", "Transformer", "DQN"]

    # Macro regime shifts
    if macro_signal == "macro_shift" or volatility == "high":
        models += ["Autoencoder"]

    # Always combine with ensemble/meta
    models += ["EnsembleVoting", "MetaLearner"]

    return list(set(models))  # Deduplicate


def classify_regime_dimensions(
        regime_profile: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert numerical regime scores to categorical classifications
    """
    vol_score = regime_profile.get('volatility_score', 50)
    trend_score = regime_profile.get('trend_score', 50)
    stress_score = regime_profile.get('market_stress', 50)
    stability_score = regime_profile.get('regime_stability', 50)

    # Classify volatility
    if vol_score < 30:
        volatility = "low"
    elif vol_score < 70:
        volatility = "medium"
    else:
        volatility = "high"

    # Classify macro signal based on trend, stress, and stability
    if stress_score > 80:
        macro_signal = "shock"
    elif stability_score < 30:
        macro_signal = "macro_shift"
    elif stress_score > 60:
        macro_signal = "irregular"
    elif trend_score > 60 or trend_score < 40:
        macro_signal = "trending"
    else:
        macro_signal = "neutral"

    # Create composite regime score (weighted average)
    composite_score = (vol_score * 0.3 + trend_score * 0.3 +
                       (100 - stress_score) * 0.2 + stability_score * 0.2)

    return {
        "volatility": volatility,
        "macro_signal": macro_signal,
        "regime_score": composite_score,
        "raw_scores": {
            "volatility_score": vol_score,
            "trend_score": trend_score,
            "stress_score": stress_score,
            "stability_score": stability_score
        }
    }


def get_active_models(regime_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced model activation using multidimensional regime classification with model router
    """
    try:
        # Try to use the new model router if available
        from model_router import get_routed_models
        logger.info("🎯 Using advanced model router for regime-based selection")
        return get_routed_models(regime_profile)

    except ImportError:
        logger.info("🎯 Using fallback multidimensional regime classification")
        # Fallback to original logic
        regime_dims = classify_regime_dimensions(regime_profile)

        volatility = regime_dims["volatility"]
        macro_signal = regime_dims["macro_signal"]
        regime_score = regime_dims["regime_score"]

        logger.info(
            f"🎯 Regime Classification: Vol={volatility}, Macro={macro_signal}, Score={regime_score:.1f}")

        # Get active models using the multidimensional logic
        active_models = determine_models(
            volatility, macro_signal, regime_score)

        # Calculate model weights based on regime characteristics
        model_weights = _calculate_regime_based_weights(
            active_models, regime_dims)

        # Determine regime classification for strategy
        regime_classification = _get_regime_classification(
            volatility, macro_signal, regime_score)

        # Generate activation reasoning
        activation_reasoning = _generate_activation_reasoning(
            volatility, macro_signal, regime_score, active_models)

        result = {
            "active_models": active_models,
            "model_weights": model_weights,
            "regime_classification": regime_classification,
            "regime_dimensions": regime_dims,
            "activation_reasoning": activation_reasoning,
            "confidence": regime_profile.get('confidence', 0.7)
        }

        logger.info(
            f"✅ Activated {len(active_models)} models for {regime_classification} regime")
        return result


def _calculate_regime_based_weights(
        models: List[str], regime_dims: Dict[str, str]) -> Dict[str, float]:
    """Calculate model weights based on regime suitability"""

    volatility = regime_dims["volatility"]
    macro_signal = regime_dims["macro_signal"]
    regime_dims["regime_score"]

    weights = {}

    # Base weights for different model types
    for model in models:
        if model in ["ARIMA", "Prophet"]:
            # Better in stable, low volatility conditions
            weights[model] = 0.4 if volatility == "low" else 0.1
        elif model in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
            # Better in medium volatility, trending conditions
            weights[model] = 0.3 if volatility == "medium" else 0.2
        elif model in ["LSTM", "GRU"]:
            # Good for time series in various conditions
            weights[model] = 0.25
        elif model in ["CNN_LSTM", "Transformer"]:
            # Better in high volatility, complex patterns
            weights[model] = 0.35 if volatility == "high" else 0.15
        elif model == "DQN":
            # Reinforcement learning for dynamic conditions
            weights[model] = 0.3 if macro_signal in [
                "irregular", "shock"] else 0.1
        elif model == "Autoencoder":
            # Anomaly detection during regime shifts
            weights[model] = 0.2
        elif model in ["EnsembleVoting", "MetaLearner"]:
            # Always important for robustness
            weights[model] = 0.15

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}

    return weights


def _get_regime_classification(
        volatility: str,
        macro_signal: str,
        regime_score: float) -> str:
    """Classify the overall regime type"""

    if macro_signal == "shock":
        return "CRISIS"
    elif volatility == "low" and macro_signal == "neutral":
        return "LOW_VOL_STABLE"
    elif volatility == "high" and macro_signal in ["irregular", "shock"]:
        return "HIGH_VOL_STRESS"
    elif macro_signal == "trending":
        return "TRENDING_MOMENTUM"
    elif macro_signal == "macro_shift":
        return "REGIME_TRANSITION"
    elif volatility == "medium":
        return "BALANCED_CONDITIONS"
    else:
        return "MIXED_CONDITIONS"


def _generate_activation_reasoning(
        volatility: str,
        macro_signal: str,
        regime_score: float,
        active_models: List[str]) -> str:
    """Generate human-readable reasoning for model activation"""

    reasoning = f"Regime Analysis: {volatility.title()} volatility with {macro_signal.replace('_', ' ')} macro conditions (score: {regime_score:.1f}). "

    if volatility == "low" and macro_signal == "neutral":
        reasoning += "Stable conditions favor mean-reversion models. "
    elif volatility == "medium" and macro_signal == "trending":
        reasoning += "Trending conditions with moderate volatility suit ensemble tree models and LSTM. "
    elif volatility == "high":
        reasoning += "High volatility requires adaptive models and anomaly detection. "

    if macro_signal in ["irregular", "shock"]:
        reasoning += "Market stress detected, emphasizing robust and defensive models. "
    elif macro_signal == "macro_shift":
        reasoning += "Regime transition detected, meta-learning models activated. "

    reasoning += f"Selected {len(active_models)} models: {', '.join(active_models[:3])}..."

    return reasoning


def get_regime_specific_parameters(
        regime_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Get regime-specific hyperparameters"""

    regime_dims = classify_regime_dimensions(regime_profile)
    volatility = regime_dims["volatility"]
    macro_signal = regime_dims["macro_signal"]
    regime_dims["regime_score"]

    # Base parameters
    base_params = {
        "learning_rate": 0.001,
        "lookback_days": 60,
        "batch_size": 64,
        "dropout_rate": 0.2,
        "regularization": 0.01,
        "rebalance_frequency": "weekly"
    }

    # Adjust parameters based on regime
    if volatility == "high":
        base_params["learning_rate"] = 0.0005  # Lower for stability
        base_params["dropout_rate"] = 0.3  # Higher to prevent overfitting
        base_params["rebalance_frequency"] = "daily"
    elif volatility == "low":
        base_params["learning_rate"] = 0.002  # Higher for faster learning
        base_params["lookback_days"] = 90  # Longer for stable patterns

    if macro_signal in ["shock", "irregular"]:
        base_params["regularization"] = 0.05  # Higher for robustness
        base_params["batch_size"] = 32  # Smaller for instability

    # Risk adjustments
    risk_multiplier = 1.0
    if volatility == "high" or macro_signal in ["shock", "irregular"]:
        risk_multiplier = 0.5
    elif volatility == "low" and macro_signal == "neutral":
        risk_multiplier = 1.5

    base_params["risk_multiplier"] = risk_multiplier
    base_params["stop_loss_factor"] = max(0.5, 2.0 - risk_multiplier)
    base_params["position_sizing_factor"] = risk_multiplier

    return base_params

# Optional: Load from YAML config if it exists


def load_model_routing_config() -> Dict[str, Any]:
    """Load model routing configuration from YAML file if available"""
    config_path = "model_routing.yaml"

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load model routing config: {e}")

    return {}

# Backward compatibility function


def get_active_models_legacy(regime_score: float) -> Dict[str, Any]:
    """Legacy function for backward compatibility with single regime score"""
    # Convert single score to multidimensional format
    regime_profile = {
        'volatility_score': regime_score,
        'trend_score': regime_score,
        'market_stress': 100 - regime_score,
        'regime_stability': regime_score,
        'confidence': 0.7
    }

    return get_active_models(regime_profile)
