#!/usr/bin/env python3
"""
🧠 SELF-LEARNING ENGINE HELPER FUNCTIONS
Supporting methods for the comprehensive self-learning system
"""

import numpy as np
import pandas as pd
import json

# import pickle  # Removed - not used
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SelfLearningEngineHelpers:
    """Helper methods for the SelfLearningEngine"""

    def _load_learning_memory(self) -> Dict[str, Any]:
        """Load persistent learning memory from disk"""
        memory_file = Path("logs/learning_memory.json")
        try:
            if memory_file.exists():
                with open(memory_file, "r") as f:
                    return json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load learning memory: {e}")

                        return {
                        "model_performance_cache": {},
                        "context_mappings": {},
                        "hyperparameter_successes": {},
                        "correction_effectiveness": {},
                        "ensemble_histories": {},
                        }

                        def _initialize_performance_tracker(self) -> Dict[str, Any]:
                            """Initialize performance tracking system"""
                            return {
                            "model_scores": {},
                            "trend_analysis": {},
                            "performance_baselines": {},
                            "improvement_tracking": {},
                            "degradation_alerts": [],
                            }

                            def _initialize_correction_engine(self) -> Dict[str, Any]:
                                """Initialize self-correction engine"""
                                return {
                                "correction_rules": self._load_correction_rules(),
                                "correction_history": [],
                                "effectiveness_tracking": {},
                                "trigger_thresholds": {
                                "performance_degradation": 0.1,
                                "drift_detection": 0.05,
                                "bias_threshold": 0.15,
                                "ensemble_imbalance": 0.2,
                                },
                                }

                                def _initialize_adaptation_engine(self) -> Dict[str, Any]:
                                    """Initialize adaptation engine"""
                                    return {
                                    "adaptation_history": [],
                                    "context_transitions": {},
                                    "adaptation_effectiveness": {},
                                    "learning_rate_adjustments": {},
                                    "model_transition_patterns": {},
                                    }

                                    def _initialize_drift_monitoring(self) -> Dict[str, Any]:
                                        """Initialize drift monitoring specifically for self-learning"""
                                        return {"drift_detectors": {}, "drift_history": [], "adaptation_triggers": {}, "drift_correction_mapping": {}}

                                        def _log_initialization(self):
                                            """Log initialization details with compliance"""
                                            from drift_protection import log_compliance_event

                                            initialization_data = {
                                            "timestamp": str(datetime.now()),
                                            "engine_type": "SELF_LEARNING_ENGINE",
                                            "models_loaded": len(self.system_models["all_models"]),
                                            "institutional_models": len(self.system_models["institutional_models"]),
                                            "adaptation_rules": len(self.meta_knowledge.adaptation_rules),
                                            "correction_strategies": len(self.meta_knowledge.correction_strategies),
                                            "compliance_active": self.compliance_active,
                                            }

                                            log_compliance_event("SELF_LEARNING_ENGINE_INITIALIZED", initialization_data)
                                            logger.info(f"🧠 Self-Learning Engine initialized: {initialization_data}")

                                            def _generate_model_hash(self, model_name: str, context) -> str:
                                                """Generate unique hash for model in context"""
                                                hash_data = {
                                                "model_name": model_name,
                                                "context_signature": self._generate_context_signature(context),
                                                "timestamp": str(datetime.now().date()),
                                                }
                                                hash_str = json.dumps(hash_data, sort_keys=True)
                                                return hashlib.sha256(hash_str.encode()).hexdigest()

                                                def _get_current_parameters(self, model_name: str) -> Dict[str, Any]:
                                                    """Get current parameters for a model"""
                                                    try:
                                                        import config

                                                        model_config = config.MATHEMATICAL_MODELS.get(model_name)
                                                        if model_config and model_config.parameters:
                                                            return {param_name: param.default_value for param_name, param in list(model_config.parameters.items())}
                                                            except Exception as e:
                                                                logger.warning(f"Could not get parameters for {model_name}: {e}")

                                                                # Default parameters for common models
                                                                default_params = {
                                                                "linear_regression": {},
                                                                "ridge_regression": {"alpha": 1.0},
                                                                "lasso_regression": {"alpha": 1.0},
                                                                "random_forest": {"n_estimators": 100, "max_depth": None},
                                                                "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1},
                                                                }

                                                                return default_params.get(model_name, {})

                                                                def _analyze_data_characteristics(self, X: np.ndarray) -> Dict[str, float]:
                                                                    """Analyze characteristics of input data"""
                                                                    try:
                                                                        return {
                                                                        "n_samples": float(X.shape[0]),
                                                                        "n_features": float(X.shape[1]) if len(X.shape) > 1 else 1.0,
                                                                        "mean_value": float(np.mean(X)),
                                                                        "std_value": float(np.std(X)),
                                                                        "skewness": float(np.mean(((X - np.mean(X)) / np.std(X)) ** 3)) if np.std(X) > 0 else 0.0,
                                                                        "kurtosis": float(np.mean(((X - np.mean(X)) / np.std(X)) ** 4)) if np.std(X) > 0 else 0.0,
                                                                        "missing_ratio": float(np.isnan(X).sum() / X.size),
                                                                        "sparsity": float((X == 0).sum() / X.size),
                                                                        }
                                                                        except Exception as e:
                                                                            logger.warning(f"Error analyzing data characteristics: {e}")
                                                                            return {
                                                                            "n_samples": 0.0,
                                                                            "n_features": 0.0,
                                                                            "mean_value": 0.0,
                                                                            "std_value": 0.0,
                                                                            "skewness": 0.0,
                                                                            "kurtosis": 0.0,
                                                                            "missing_ratio": 0.0,
                                                                            "sparsity": 0.0,
                                                                            }

                                                                            def _calculate_confidence_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
                                                                                """Calculate confidence score for predictions"""
                                                                                try:
                                                                                    from sklearn.metrics import r2_score

                                                                                    # Calculate prediction consistency
                                                                                    residuals = y_true - y_pred
                                                                                    residual_std = np.std(residuals)

                                                                                    # Calculate R² score
                                                                                    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0

                                                                                    # Combine metrics for confidence score
                                                                                    consistency_score = 1.0 / (1.0 + residual_std)
                                                                                    accuracy_score = max(0.0, r2)

                                                                                    confidence = (consistency_score + accuracy_score) / 2.0
                                                                                    return float(np.clip(confidence, 0.0, 1.0))

                                                                                    except Exception as e:
                                                                                        logger.warning(f"Error calculating confidence score: {e}")
                                                                                        return 0.0

                                                                                        def _update_meta_knowledge(self, record, context):
                                                                                            """Update meta-knowledge with new performance record"""
                                                                                            model_name = record.model_name

                                                                                            # Update performance history
                                                                                            if model_name not in self.meta_knowledge.model_performance_history:
                                                                                                self.meta_knowledge.model_performance_history[model_name] = []

                                                                                                self.meta_knowledge.model_performance_history[model_name].append(record)

                                                                                                # Keep only recent records (memory management)
                                                                                                max_records = 1000
                                                                                                if len(self.meta_knowledge.model_performance_history[model_name]) > max_records:
                                                                                                    self.meta_knowledge.model_performance_history[model_name] = self.meta_knowledge.model_performance_history[
                                                                                                    model_name
                                                                                                    ][-max_records:]

                                                                                                    # Update optimal model mappings
                                                                                                    context_signature = self._generate_context_signature(context)

                                                                                                    # Check if this is the best model for this context
                                                                                                    if (
                                                                                                    context_signature not in self.meta_knowledge.optimal_model_mappings
                                                                                                    or record.prediction_accuracy > self._get_best_accuracy_for_context(context_signature)
                                                                                                    ):
                                                                                                        self.meta_knowledge.optimal_model_mappings[context_signature] = model_name

                                                                                                        def _get_best_accuracy_for_context(self, context_signature: str) -> float:
                                                                                                            """Get best accuracy achieved for a given context"""
                                                                                                            best_accuracy = 0.0

                                                                                                            for model_name, records in list(self.meta_knowledge.model_performance_history.items()):
                                                                                                                for record in records:
                                                                                                                    if record.model_hash.endswith(context_signature[:8]):  # Context matching
                                                                                                                    best_accuracy = max(best_accuracy, record.prediction_accuracy)

                                                                                                                    return best_accuracy

                                                                                                                    def _check_for_drift(self, record, context) -> bool:
                                                                                                                        """Check if drift has been detected"""
                                                                                                                        try:
                                                                                                                            # Use existing drift detection system
                                                                                                                            from drift_protection import detect_distribution_drift, track_model_performance

                                                                                                                            # Use actual market data for drift detection
                                                                                                                            # Get recent returns from the same symbol
                                                                                                                            from polygon_connector import PolygonConnector
                                                                                                                            connector = PolygonConnector()
                                                                                                                            recent_data = connector.get_ohlcv_data(
                                                                                                                                record.context.get('symbol', 'SPY'),
                                                                                                                                (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                                                                                                                                datetime.now().strftime('%Y-%m-%d')
                                                                                                                            )
                                                                                                                            if recent_data is not None and not recent_data.empty:
                                                                                                                                actual_returns = recent_data['close'].pct_change().dropna().values
                                                                                                                            else:
                                                                                                                                # Fallback to zeros if no data available
                                                                                                                                actual_returns = np.zeros(100)

                                                                                                                            # Check for data drift
                                                                                                                            data_drift = detect_distribution_drift(
                                                                                                                            {
                                                                                                                            "mean": record.data_characteristics.get("mean_value", 0),
                                                                                                                            "std": record.data_characteristics.get("std_value", 1),
                                                                                                                            "skew": record.data_characteristics.get("skewness", 0),
                                                                                                                            "kurtosis": record.data_characteristics.get("kurtosis", 0),
                                                                                                                            },
                                                                                                                            record.model_name,
                                                                                                                            )

                                                                                                                            # Check for performance drift
                                                                                                                            performance_drift = record.prediction_accuracy < 0.5  # Simple threshold

                                                                                                                            return data_drift or performance_drift

                                                                                                                            except Exception as e:
                                                                                                                                logger.warning(f"Error checking for drift: {e}")
                                                                                                                                return False

                                                                                                                                def _apply_self_correction(self, record, context) -> Dict[str, Any]:
                                                                                                                                    """Apply self-correction based on detected issues"""
                                                                                                                                    correction_actions = []

                                                                                                                                    # Identify correction type needed
                                                                                                                                    if record.prediction_accuracy < 0.3:
                                                                                                                                        correction_actions.append(
                                                                                                                                        {"type": "performance_degradation", "severity": "high", "action": "model_replacement"}
                                                                                                                                        )
                                                                                                                                        elif record.prediction_accuracy < 0.7:
                                                                                                                                            correction_actions.append(
                                                                                                                                            {"type": "performance_degradation", "severity": "medium", "action": "hyperparameter_tuning"}
                                                                                                                                            )

                                                                                                                                            # Apply corrections
                                                                                                                                            for action in correction_actions:
                                                                                                                                                try:
                                                                                                                                                    if action["type"] in self.meta_knowledge.correction_strategies:
                                                                                                                                                        correction_func = self.meta_knowledge.correction_strategies[action["type"]]
                                                                                                                                                        correction_func(action, context)
                                                                                                                                                        except Exception as e:
                                                                                                                                                            logger.warning(f"Correction failed: {e}")

                                                                                                                                                            return {"corrections_applied": correction_actions, "correction_timestamp": str(datetime.now())}

                                                                                                                                                            def _update_learning_memory(self, record, context):
                                                                                                                                                                """Update persistent learning memory"""
                                                                                                                                                                try:
                                                                                                                                                                    # Update learning memory
                                                                                                                                                                    self.learning_memory["model_performance_cache"][record.model_name] = {
                                                                                                                                                                    "latest_performance": record.prediction_accuracy,
                                                                                                                                                                    "timestamp": str(record.timestamp),
                                                                                                                                                                    "context_signature": self._generate_context_signature(context),
                                                                                                                                                                    }

                                                                                                                                                                    # Save to disk periodically
                                                                                                                                                                    if self.learning_iterations % 10 == 0:
                                                                                                                                                                        self._save_learning_memory()

                                                                                                                                                                        except Exception as e:
                                                                                                                                                                            logger.warning(f"Error updating learning memory: {e}")

                                                                                                                                                                            def _save_learning_memory(self):
                                                                                                                                                                                """Save learning memory to disk"""
                                                                                                                                                                                try:
                                                                                                                                                                                    memory_file = Path("logs/learning_memory.json")
                                                                                                                                                                                    memory_file.parent.mkdir(exist_ok=True)

                                                                                                                                                                                    with open(memory_file, "w") as f:
                                                                                                                                                                                        json.dump(self.learning_memory, f, indent=2, default=str)

                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                            logger.warning(f"Error saving learning memory: {e}")

                                                                                                                                                                                            def _log_learning_event(self, record, context):
                                                                                                                                                                                                """Log learning event for audit and analysis"""
                                                                                                                                                                                                try:
                                                                                                                                                                                                    from drift_protection import log_compliance_event

                                                                                                                                                                                                    learning_event = {
                                                                                                                                                                                                    "model_name": record.model_name,
                                                                                                                                                                                                    "prediction_accuracy": record.prediction_accuracy,
                                                                                                                                                                                                    "learning_iteration": record.learning_iteration,
                                                                                                                                                                                                    "context_signature": self._generate_context_signature(context),
                                                                                                                                                                                                    "drift_detected": record.drift_detected,
                                                                                                                                                                                                    "correction_applied": record.correction_applied,
                                                                                                                                                                                                    "timestamp": str(record.timestamp),
                                                                                                                                                                                                    }

                                                                                                                                                                                                    log_compliance_event("SELF_LEARNING_EVENT", learning_event)

                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                        logger.warning(f"Error logging learning event: {e}")

                                                                                                                                                                                                        def _should_create_ensemble(self, top_models: List, context) -> bool:
                                                                                                                                                                                                            """Determine if ensemble would be beneficial"""
                                                                                                                                                                                                            if len(top_models) < 2:
                                                                                                                                                                                                                return False

                                                                                                                                                                                                                # Check if models are diverse enough
                                                                                                                                                                                                                score_variance = np.var([score for _, score in top_models])

                                                                                                                                                                                                                # Ensemble beneficial if models have similar but diverse performance
                                                                                                                                                                                                                return 0.01 < score_variance < 0.1 and context.data_quality_score > 0.6

                                                                                                                                                                                                                def _create_adaptive_ensemble(self, top_models: List, context) -> Dict[str, float]:
                                                                                                                                                                                                                    """Create adaptive ensemble strategy"""
                                                                                                                                                                                                                    total_score = sum(score for _, score in top_models)

                                                                                                                                                                                                                    if total_score > 0:
                                                                                                                                                                                                                        # Weight by performance with diversity bonus
                                                                                                                                                                                                                        weights = {}
                                                                                                                                                                                                                        for model_name, score in top_models:
                                                                                                                                                                                                                            base_weight = score / total_score
                                                                                                                                                                                                                            # Add small diversity bonus
                                                                                                                                                                                                                            diversity_bonus = 0.1 / len(top_models)
                                                                                                                                                                                                                            weights[model_name] = base_weight + diversity_bonus

                                                                                                                                                                                                                            # Normalize weights
                                                                                                                                                                                                                            total_weight = sum(weights.values())
                                                                                                                                                                                                                            weights = {k: v / total_weight for k, v in list(weights.items())}

                                                                                                                                                                                                                            return weights
                                                                                                                                                                                                                            else:
                                                                                                                                                                                                                                # Equal weights fallback
                                                                                                                                                                                                                                equal_weight = 1.0 / len(top_models)
                                                                                                                                                                                                                                return {model_name: equal_weight for model_name, _ in top_models}

                                                                                                                                                                                                                                def _analyze_performance_trends(self, performance_history: List[Dict]) -> Dict[str, Any]:
                                                                                                                                                                                                                                    """Analyze performance trends for correction opportunities"""
                                                                                                                                                                                                                                    if len(performance_history) < 3:
                                                                                                                                                                                                                                        return {"trend": "insufficient_data", "current_performance": 0.0}

                                                                                                                                                                                                                                        recent_performances = [p.get("accuracy", 0.0) for p in performance_history[-10:]]

                                                                                                                                                                                                                                        # Calculate trend
                                                                                                                                                                                                                                        x = np.arange(len(recent_performances))
                                                                                                                                                                                                                                        slope = np.polyfit(x, recent_performances, 1)[0]

                                                                                                                                                                                                                                        trend = "improving" if slope > 0.01 else "degrading" if slope < -0.01 else "stable"

                                                                                                                                                                                                                                        return {
                                                                                                                                                                                                                                        "trend": trend,
                                                                                                                                                                                                                                        "slope": slope,
                                                                                                                                                                                                                                        "current_performance": recent_performances[-1],
                                                                                                                                                                                                                                        "average_performance": np.mean(recent_performances),
                                                                                                                                                                                                                                        "performance_volatility": np.std(recent_performances),
                                                                                                                                                                                                                                        }

                                                                                                                                                                                                                                        def _identify_correction_opportunities(self, trend_analysis: Dict, context) -> List[Dict]:
                                                                                                                                                                                                                                            """Identify opportunities for self-correction"""
                                                                                                                                                                                                                                            opportunities = []

                                                                                                                                                                                                                                            if trend_analysis["trend"] == "degrading":
                                                                                                                                                                                                                                                opportunities.append(
                                                                                                                                                                                                                                                {
                                                                                                                                                                                                                                                "type": "performance_degradation",
                                                                                                                                                                                                                                                "severity": "high" if trend_analysis["slope"] < -0.05 else "medium",
                                                                                                                                                                                                                                                "cause": "unknown",
                                                                                                                                                                                                                                                "magnitude": abs(trend_analysis["slope"]),
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                )

                                                                                                                                                                                                                                                if trend_analysis["performance_volatility"] > 0.2:
                                                                                                                                                                                                                                                    opportunities.append(
                                                                                                                                                                                                                                                    {
                                                                                                                                                                                                                                                    "type": "prediction_bias",
                                                                                                                                                                                                                                                    "severity": "medium",
                                                                                                                                                                                                                                                    "bias_direction": "unknown",
                                                                                                                                                                                                                                                    "bias_magnitude": trend_analysis["performance_volatility"],
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                    if context.data_quality_score < 0.7:
                                                                                                                                                                                                                                                        opportunities.append({"type": "model_selection_error", "severity": "medium", "current_model": "unknown"})

                                                                                                                                                                                                                                                        return opportunities

                                                                                                                                                                                                                                                        def _estimate_improvement(self, corrections_applied: List[Dict]) -> float:
                                                                                                                                                                                                                                                            """Estimate expected improvement from corrections"""
                                                                                                                                                                                                                                                            total_improvement = 0.0

                                                                                                                                                                                                                                                            for correction in corrections_applied:
                                                                                                                                                                                                                                                                expected_improvement = correction.get("expected_improvement", 0.0)
                                                                                                                                                                                                                                                                total_improvement += expected_improvement

                                                                                                                                                                                                                                                                # Cap improvement estimate
                                                                                                                                                                                                                                                                return min(total_improvement, 0.5)

                                                                                                                                                                                                                                                                def _load_correction_rules(self) -> Dict[str, Any]:
                                                                                                                                                                                                                                                                    """Load correction rules"""
                                                                                                                                                                                                                                                                    return {
                                                                                                                                                                                                                                                                    "performance_thresholds": {"critical": 0.3, "warning": 0.6, "good": 0.8},
                                                                                                                                                                                                                                                                    "drift_thresholds": {"data_drift": 0.05, "performance_drift": 0.1, "concept_drift": 0.15},
                                                                                                                                                                                                                                                                    "correction_strategies": {
                                                                                                                                                                                                                                                                    "retrain": {"trigger_threshold": 0.3, "expected_improvement": 0.2},
                                                                                                                                                                                                                                                                    "tune_hyperparameters": {"trigger_threshold": 0.6, "expected_improvement": 0.1},
                                                                                                                                                                                                                                                                    "ensemble_rebalance": {"trigger_threshold": 0.7, "expected_improvement": 0.05},
                                                                                                                                                                                                                                                                    },
                                                                                                                                                                                                                                                                    }


                                                                                                                                                                                                                                                                    # Additional helper functions for completeness
                                                                                                                                                                                                                                                                    def create_learning_context(
                                                                                                                                                                                                                                                                    data_distribution: Dict = None,
                                                                                                                                                                                                                                                                    market_regime: str = "normal",
                                                                                                                                                                                                                                                                    volatility_level: str = "medium",
                                                                                                                                                                                                                                                                    data_quality_score: float = 0.8,
                                                                                                                                                                                                                                                                    prediction_horizon: int = 60,
                                                                                                                                                                                                                                                                    ) -> "LearningContext":
                                                                                                                                                                                                                                                                        """Create a learning context for the self-learning engine"""
                                                                                                                                                                                                                                                                        from self_learning_engine import LearningContext

                                                                                                                                                                                                                                                                        return LearningContext(
                                                                                                                                                                                                                                                                        data_distribution=data_distribution or {},
                                                                                                                                                                                                                                                                        market_regime=market_regime,
                                                                                                                                                                                                                                                                        volatility_level=volatility_level,
                                                                                                                                                                                                                                                                        data_quality_score=data_quality_score,
                                                                                                                                                                                                                                                                        prediction_horizon=prediction_horizon,
                                                                                                                                                                                                                                                                        feature_importance={},
                                                                                                                                                                                                                                                                        previous_predictions=[],
                                                                                                                                                                                                                                                                        actual_outcomes=[],
                                                                                                                                                                                                                                                                        performance_trend="stable",
                                                                                                                                                                                                                                                                        )


                                                                                                                                                                                                                                                                        def analyze_model_compatibility(model_name: str, context) -> Dict[str, Any]:
                                                                                                                                                                                                                                                                            """Analyze compatibility between model and context"""
                                                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                                                                import config

                                                                                                                                                                                                                                                                                model_config = config.MATHEMATICAL_MODELS.get(model_name)
                                                                                                                                                                                                                                                                                if not model_config:
                                                                                                                                                                                                                                                                                    return {"compatible": False, "reason": "model_not_found"}

                                                                                                                                                                                                                                                                                    compatibility_score = 0.5  # Base score

                                                                                                                                                                                                                                                                                    # Check interpretability requirements
                                                                                                                                                                                                                                                                                    if context.data_quality_score > 0.8 and model_config.interpretability_score > 7:
                                                                                                                                                                                                                                                                                        compatibility_score += 0.2

                                                                                                                                                                                                                                                                                        # Check computational requirements
                                                                                                                                                                                                                                                                                        if context.prediction_horizon > 300:  # Long horizon
                                                                                                                                                                                                                                                                                        if "O(n)" in model_config.computational_complexity:
                                                                                                                                                                                                                                                                                            compatibility_score += 0.2
                                                                                                                                                                                                                                                                                            else:  # Short horizon
                                                                                                                                                                                                                                                                                            if "O(n²)" in model_config.computational_complexity:
                                                                                                                                                                                                                                                                                                compatibility_score -= 0.2

                                                                                                                                                                                                                                                                                                return {
                                                                                                                                                                                                                                                                                                "compatible": compatibility_score > 0.6,
                                                                                                                                                                                                                                                                                                "compatibility_score": compatibility_score,
                                                                                                                                                                                                                                                                                                "reasons": ["interpretability_match", "computational_efficiency"],
                                                                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                                                    logger.warning(f"Error analyzing model compatibility: {e}")
                                                                                                                                                                                                                                                                                                    return {"compatible": False, "reason": "analysis_error"}


                                                                                                                                                                                                                                                                                                    # Export helper functions
                                                                                                                                                                                                                                                                                                    __all__ = ["SelfLearningEngineHelpers", "create_learning_context", "analyze_model_compatibility"]
