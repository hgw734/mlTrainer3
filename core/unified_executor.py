"""
Unified MLTrainer Executor
=========================

Bridges the advanced execution framework with compliance and model infrastructure.
Provides a single interface for all ML and financial model executions.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Import from current system
from mlagent_bridge import MLAgentBridge
from mlagent_model_integration import get_model_integration
from goal_system import GoalSystem
from config.immutable_compliance_gateway import ComplianceGateway

# These would come from advanced system
# from core.mltrainer_executor import MLTrainerExecutor
# from core.dynamic_executor import DynamicActionGenerator

logger = logging.getLogger(__name__)


class UnifiedMLTrainerExecutor:
    """
    Unified executor that combines:
        - Advanced execution capabilities from sophisticated version
        - Compliance and model management from current version
        - Goal-aware execution with audit trail
    """

    def __init__(self):
        # Current system components
        self.mlagent_bridge = MLAgentBridge()
        self.model_integration = get_model_integration()
        self.goal_system = GoalSystem()
        self.compliance_gateway = ComplianceGateway()

        # Execution tracking
        self.execution_history = []
        self.registered_actions = {}

        # Register all available actions
        self._register_all_actions()

        logger.info("UnifiedMLTrainerExecutor initialized with compliance and models")

    def _register_all_actions(self):
        """Register all ML and financial models as executable actions"""
        # Register ML model training actions
        for model_id in self.model_integration.ml_manager.get_available_models():
            action_name = f"train_{model_id}"
            self.register_action(
                action_name,
                lambda mid=model_id, **kwargs: self.execute_ml_model_training(mid, **kwargs),
                description=f"Train {model_id} model",
            )

        # Register financial model actions
        for model_id in self.model_integration.financial_manager.get_available_models():
            action_name = f"calculate_{model_id}"
            self.register_action(
                action_name,
                lambda mid=model_id, **kwargs: self.execute_financial_model(mid, **kwargs),
                description=f"Calculate {model_id}",
            )

        # Register standard actions
        self.register_action("momentum_screening", self.execute_momentum_screening)
        self.register_action("regime_detection", self.execute_regime_detection)
        self.register_action("portfolio_optimization", self.execute_portfolio_optimization)

        logger.info(f"Registered {len(self.registered_actions)} executable actions")

    def register_action(self, name: str, func: callable, description: str = ""):
        """Register an executable action"""
        self.registered_actions[name] = {
            "function": func,
            "description": description,
            "registered_at": datetime.now().isoformat(),
        }

    def parse_mltrainer_response(self, response: str) -> Dict[str, Any]:
        """Parse mlTrainer response for executable actions"""
        # Use existing mlAgent bridge
        base_parse = self.mlagent_bridge.parse_mltrainer_response(response)

        # Enhance with model detection
        model_request = self.model_integration.parse_model_request(response)

        # Check for registered actions in the response
        detected_actions = []
        for action_name in self.registered_actions:
            if action_name.replace("_", " ") in response.lower():
                detected_actions.append(action_name)

        # Combine results
        result = {
            "executable": base_parse.get("patterns_detected", []) or detected_actions or model_request["model_id"],
            "actions": detected_actions,
            "trial_suggestions": base_parse.get("trial_suggestions", []),
            "model_request": model_request,
            "models_mentioned": base_parse.get("models_mentioned", []),
            "parameters": base_parse.get("parameters", {}),
        }

        return result

    def execute_suggestion(self, mltrainer_response: str, user_approved: bool = False) -> Dict[str, Any]:
        """Execute suggestions from mlTrainer response"""
        if not user_approved:
            return {"success": False, "error": "User approval required for execution"}

        # Parse the response
        parsed = self.parse_mltrainer_response(mltrainer_response)

        if not parsed["executable"]:
            return {"success": False, "error": "No executable actions found"}

        # Compliance check
        current_goal = self.goal_system.get_current_goal()
        if not self._verify_execution_compliance(parsed, current_goal):
            return {"success": False, "error": "Execution blocked by compliance"}

        results = []

        # Execute detected actions
        for action in parsed["actions"]:
            if action in self.registered_actions:
                try:
                    result = self.registered_actions[action]["function"](**parsed["parameters"])
                    results.append({"action": action, "result": result, "timestamp": datetime.now().isoformat()})
                except Exception as e:
                    results.append(
                        {
                            "action": action,
                            "result": {"success": False, "error": str(e)},
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        # Execute model request if present
        if parsed["model_request"]["model_id"]:
            model_result = self.model_integration.execute_model_request(parsed["model_request"])
            results.append(
                {
                    "action": f"model_{parsed['model_request']['type']}",
                    "result": model_result,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Log execution
        self._log_execution(parsed, results)

        return {"success": len(results) > 0, "results": results, "executed_at": datetime.now().isoformat()}

    def execute_ml_model_training(self, model_id: str, symbol: str = "AAPL", **params) -> Dict[str, Any]:
        """Execute ML model training with compliance"""
        try:
            # Compliance verification
            data_source = params.get("data_source", "polygon")
            if not self.compliance_gateway.verify_data_source(data_source, f"train_{model_id}"):
                return {"success": False, "error": f"Data source {data_source} not compliant"}

            # Execute through model manager
            result = self.model_integration.ml_manager.train_model(
                model_id, symbol=symbol, data_source=data_source, **params
            )

            return {
                "success": result.compliance_status == "approved",
                "data": {
                    "model_id": model_id,
                    "performance": result.performance_metrics,
                    "training_time": result.training_time,
                    "feature_importance": (
                        result.feature_importance.tolist() if result.feature_importance is not None else None
                    ),
                },
            }
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
            return {"success": False, "error": str(e)}

    def execute_financial_model(self, model_id: str, **params) -> Dict[str, Any]:
        """Execute financial model with compliance"""
        try:
            # Execute through financial manager
            result = self.model_integration.financial_manager.run_model(model_id, **params)

            response = {
                "success": result.compliance_status == "approved",
                "data": {"model_id": model_id, "execution_time": result.execution_time},
            }

            # Add model-specific results
            if result.option_price is not None:
                response["data"]["option_price"] = result.option_price
                response["data"]["greeks"] = result.greeks

            if result.portfolio_weights is not None:
                response["data"]["portfolio_weights"] = result.portfolio_weights.tolist()
                response["data"]["performance"] = result.performance_metrics

            if result.risk_metrics:
                response["data"]["risk_metrics"] = result.risk_metrics

            return response
        except Exception as e:
            logger.error(f"Financial model execution failed: {e}")
            return {"success": False, "error": str(e)}

    def execute_momentum_screening(self, models: List[str] = None, **params) -> Dict[str, Any]:
        """Execute momentum screening analysis"""
        try:
            # Default to high-performing models
            if not models:
                models = ["random_forest_100", "gradient_boosting_100", "mlp_100"]

            results = {}
            for model_id in models:
                if model_id in self.model_integration.ml_manager.get_available_models():
                    result = self.execute_ml_model_training(model_id, **params)
                    results[model_id] = result

            return {"success": True, "data": {"screening_type": "momentum", "models_used": models, "results": results}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_regime_detection(self, **params) -> Dict[str, Any]:
        """Execute market regime detection"""
        try:
            # Use clustering models for regime detection
            clustering_models = self.model_integration.ml_manager.get_models_by_category("clustering")

            if not clustering_models:
                return {"success": False, "error": "No clustering models available"}

            # Train a clustering model
            model_id = clustering_models[0]  # Use first available
            result = self.execute_ml_model_training(model_id, **params)

            return {
                "success": result["success"],
                "data": {"detection_type": "regime", "model_used": model_id, "result": result},
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_portfolio_optimization(self, symbols: List[str] = None, **params) -> Dict[str, Any]:
        """Execute portfolio optimization"""
        try:
            if not symbols:
                symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]

            # Use mean-variance optimization
            result = self.execute_financial_model("mean_variance", symbols=symbols, **params)

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _verify_execution_compliance(self, parsed_data: Dict, current_goal: Dict) -> bool:
        """Verify execution meets compliance requirements"""
        # Check for prohibited terms
        prohibited_terms = ["synthetic", "random", "generate", "simulate"]

        for term in prohibited_terms:
            if term in str(parsed_data).lower():
                logger.warning(f"Compliance violation: prohibited term '{term}' detected")
                return False

        # Verify alignment with goal
        if current_goal and current_goal.get("goal"):
            # Basic goal alignment check
            if "no_data_generators" in current_goal.get("compliance_requirements", []):
                if any("generat" in str(action).lower() for action in parsed_data.get("actions", [])):
                    logger.warning("Compliance violation: data generation attempted")
                    return False

        return True

    def _log_execution(self, parsed_data: Dict, results: List[Dict]):
        """Log execution for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "parsed_data": parsed_data,
            "results": results,
            "goal_context": self.goal_system.get_current_goal(),
            "success": all(r.get("result", {}).get("success", False) for r in results),
        }

        self.execution_history.append(log_entry)

        # Save to file
        try:
            with open("logs/execution_history.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of executions"""
        if not self.execution_history:
            return {"message": "No executions yet"}

        successful = sum(1 for e in self.execution_history if e["success"])
        failed = len(self.execution_history) - successful

        return {
            "total_executions": len(self.execution_history),
            "successful": successful,
            "failed": failed,
            "registered_actions": len(self.registered_actions),
            "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None,
        }


# Create singleton instance
_unified_executor = None


def get_unified_executor() -> UnifiedMLTrainerExecutor:
    """Get the unified executor instance"""
    global _unified_executor
    if _unified_executor is None:
        _unified_executor = UnifiedMLTrainerExecutor()
    return _unified_executor
