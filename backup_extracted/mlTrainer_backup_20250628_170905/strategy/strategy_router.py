import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def select_strategy_multidimensional(regime_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Selects optimal ML model and config based on multidimensional regime analysis.
    
    Args:
        regime_profile: Multidimensional regime data with volatility, trend, 
                       distribution, and structure scores
    
    Returns:
        Dict with comprehensive model config and risk settings
    """
    
    logger.info(f"🧭 Selecting strategy for multidimensional regime:")
    logger.info(f"   Vol: {regime_profile.get('volatility_score', 0):.1f}, "
               f"Trend: {regime_profile.get('trend_score', 0):.1f}, "
               f"Stress: {regime_profile.get('market_stress', 0):.1f}")
    
    vol_score = regime_profile.get('volatility_score', 50)
    trend_score = regime_profile.get('trend_score', 50) 
    stress_score = regime_profile.get('market_stress', 50)
    stability_score = regime_profile.get('regime_stability', 50)
    
    # Multidimensional strategy selection
    config = _select_adaptive_config(vol_score, trend_score, stress_score, stability_score)
    
    # Add regime-specific optimizations
    config.update({
        "regime_vector": regime_profile.get('composite_regime_vector', [50, 50, 50, 50]),
        "regime_confidence": regime_profile.get('confidence', 0.7),
        "adaptive_parameters": _get_adaptive_parameters(regime_profile)
    })
    
    logger.info(f"✅ Strategy selected: {config['model']} with {config['risk_profile']} profile")
    return config

def _select_adaptive_config(vol_score: float, trend_score: float, 
                          stress_score: float, stability_score: float) -> Dict[str, Any]:
    """Select configuration based on regime dimensions"""
    
    # Crisis conditions - defensive approach
    if stress_score > 80 or vol_score > 90:
        return {
            "model": "XGBoost",
            "optimization_target": "sharpe_ratio",
            "risk_profile": "defensive",
            "lookback_window": 20,
            "position_size_multiplier": 0.3,
            "rebalance_frequency": "daily"
        }
    
    # High volatility bull market - aggressive but cautious
    elif vol_score > 70 and trend_score > 60:
        return {
            "model": "Ensemble", 
            "optimization_target": "risk_adjusted_return",
            "risk_profile": "aggressive",
            "lookback_window": 45,
            "position_size_multiplier": 0.7,
            "rebalance_frequency": "daily"
        }
    
    # Low volatility stable conditions - growth focused
    elif vol_score < 30 and stability_score > 70:
        return {
            "model": "LSTM",
            "optimization_target": "total_return", 
            "risk_profile": "growth",
            "lookback_window": 90,
            "position_size_multiplier": 0.9,
            "rebalance_frequency": "weekly"
        }
    
    # Strong trend regardless of volatility - momentum strategy
    elif trend_score > 80 or trend_score < 20:
        return {
            "model": "Transformer",
            "optimization_target": "momentum_capture",
            "risk_profile": "momentum", 
            "lookback_window": 60,
            "position_size_multiplier": 0.8,
            "rebalance_frequency": "bi-daily"
        }
    
    # Mixed/transitional conditions - balanced adaptive approach
    else:
        return {
            "model": "Adaptive_Ensemble",
            "optimization_target": "multi_objective",
            "risk_profile": "balanced",
            "lookback_window": int(60 + (stability_score - 50) * 0.6),
            "position_size_multiplier": 0.5 + (stability_score / 200),
            "rebalance_frequency": "daily" if vol_score > 60 else "weekly"
        }

def _get_adaptive_parameters(regime_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Generate adaptive parameters based on regime characteristics"""
    
    vol_score = regime_profile.get('volatility_score', 50)
    trend_score = regime_profile.get('trend_score', 50)
    stress_score = regime_profile.get('market_stress', 50)
    stability_score = regime_profile.get('regime_stability', 50)
    
    # Dynamic stop loss based on volatility and stability
    base_stop_loss = 0.05
    vol_adjustment = (vol_score - 50) / 1000  # ±0.05 max
    stability_adjustment = (50 - stability_score) / 2000  # Wider stops in unstable markets
    stop_loss = max(0.01, base_stop_loss + vol_adjustment + stability_adjustment)
    
    # Dynamic take profit based on trend strength
    base_take_profit = 0.10
    trend_adjustment = (trend_score - 50) / 500  # ±0.10 max
    take_profit = max(0.03, base_take_profit + trend_adjustment)
    
    # Dynamic learning rate based on market stress
    base_learning_rate = 0.001
    stress_adjustment = (100 - stress_score) / 100000  # Lower LR in high stress
    learning_rate = max(0.0001, base_learning_rate + stress_adjustment)
    
    return {
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4), 
        "learning_rate": learning_rate,
        "feature_importance_threshold": 0.01 + (stability_score / 10000),
        "ensemble_weights": _calculate_ensemble_weights(regime_profile),
        "risk_scaling_factor": max(0.1, min(2.0, stability_score / 50))
    }

def _calculate_ensemble_weights(regime_profile: Dict[str, Any]) -> Dict[str, float]:
    """Calculate model ensemble weights based on regime characteristics"""
    
    vol_score = regime_profile.get('volatility_score', 50)
    trend_score = regime_profile.get('trend_score', 50)
    stress_score = regime_profile.get('market_stress', 50)
    
    # Weight models based on regime suitability
    weights = {
        "xgboost": max(0.1, (100 - vol_score + stress_score) / 200),  # Better in stable/stress
        "lstm": max(0.1, (vol_score + trend_score) / 200),  # Better in trending/volatile
        "transformer": max(0.1, trend_score / 100),  # Better in strong trends
        "random_forest": max(0.1, (100 - stress_score) / 100)  # Better when not stressed
    }
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    return {k: v/total_weight for k, v in weights.items()}

# Legacy function for backward compatibility
def select_strategy(regime_score: float) -> Dict[str, Any]:
    """Legacy function - converts single score to multidimensional format"""
    # Convert single regime score to multidimensional profile
    regime_profile = {
        'volatility_score': regime_score,
        'trend_score': regime_score,
        'market_stress': 100 - regime_score,
        'regime_stability': regime_score,
        'composite_regime_vector': [regime_score] * 4,
        'confidence': 0.7
    }
    
    return select_strategy_multidimensional(regime_profile)
