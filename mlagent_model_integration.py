"""
mlAgent Model Integration
========================

Integration layer that connects ML and Financial model managers with mlAgent bridge.
Allows Claude to access and execute models through the chat interface.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# Import model managers
from mltrainer_models import get_ml_model_manager
from mltrainer_financial_models import get_financial_model_manager

# Import mlAgent bridge
from mlagent_bridge import MLAgentBridge

logger = logging.getLogger(__name__)


class MLAgentModelIntegration:
    """Integration layer between models and mlAgent"""

    def __init__(self):
        self.ml_manager = get_ml_model_manager()
        self.financial_manager = get_financial_model_manager()
        self.mlagent = MLAgentBridge()
        self.logger = logging.getLogger(__name__)

        # Register model handlers with mlAgent
        self._register_handlers()

        self.logger.info("MLAgentModelIntegration initialized")

        def _register_handlers(self):
            """Register model execution handlers with mlAgent"""
            # Register as additional patterns for mlAgent to recognize
            self.model_patterns = {
                "ml_model": r"(?:train|run|execute)\s+(?:ml\s+)?model\s+(\w+)",
                "financial_model": r"(?:run|execute|calculate)\s+(?:financial\s+)?model\s+(\w+)",
                "list_models": r"(?:list|show|what\s+are)\s+(?:available\s+)?models",
                "model_info": r"(?:info|details|describe)\s+(?:about\s+)?model\s+(\w+)",
                "best_models": r"(?:best|top)\s+(\d+)?\s*models",
            }

            def parse_model_request(
                    self, mltrainer_response: str) -> Dict[str, Any]:
                """Parse model execution requests from mlTrainer's response"""
                request = {
                    "type": None,
                    "model_id": None,
                    "parameters": {},
                    "action": None}

                # Check for ML model requests
                if "train model" in mltrainer_response.lower():
                    request["type"] = "ml"
                    request["action"] = "train"

                    # Extract model ID
                    for model_id in self.ml_manager.get_available_models():
                        if model_id in mltrainer_response.lower():
                            request["model_id"] = model_id
                            break

                        # Extract parameters
                        request["parameters"] = self._extract_ml_parameters(
                            mltrainer_response)

                        # Check for financial model requests
                        elif any(term in mltrainer_response.lower() for term in ["black-scholes", "portfolio", "risk", "var"]):
                            request["type"] = "financial"
                            request["action"] = "run"

                            # Extract model ID
                            for model_id in self.financial_manager.get_available_models():
                                if model_id.replace(
                                        "_", " ") in mltrainer_response.lower():
                                    request["model_id"] = model_id
                                    break

                                # Extract parameters
                                request["parameters"] = self._extract_financial_parameters(
                                    mltrainer_response)

                                # Check for model listing requests
                                elif "list models" in mltrainer_response.lower():
                                    request["type"] = "info"
                                    request["action"] = "list"

                                    # Check for best models request
                                    elif "best models" in mltrainer_response.lower():
                                        request["type"] = "info"
                                        request["action"] = "best"

                                        return request

                                        def _extract_ml_parameters(
                                                self, response: str) -> Dict[str, Any]:
                                            """Extract ML model parameters from response"""
                                            params = {}

                                            # Extract symbol
                                            if "symbol" in response or "stock" in response:
                                                import re

                                                symbol_match = re.search(
                                                    r"(?:symbol|stock)[:\s]+([A-Z]{1,5})", response)
                                                if symbol_match:
                                                    params["symbol"] = symbol_match.group(
                                                        1)

                                                    # Extract data source
                                                    if "polygon" in response.lower():
                                                        params["data_source"] = "polygon"
                                                        elif "fred" in response.lower():
                                                            params["data_source"] = "fred"

                                                            # Extract lookback
                                                            # period
                                                            lookback_match = re.search(
                                                                r"(\d+)\s*(?:days?|periods?)", response)
                                                            if lookback_match:
                                                                params["lookback_days"] = int(
                                                                    lookback_match.group(1))

                                                                return params

                                                                def _extract_financial_parameters(
                                                                        self, response: str) -> Dict[str, Any]:
                                                                    """Extract financial model parameters from response"""
                                                                    params = {}
                                                                    import re

                                                                    # Extract
                                                                    # numeric
                                                                    # values
                                                                    numbers = re.findall(
                                                                        r"[-+]?\d*\.?\d+", response)

                                                                    # Black-Scholes
                                                                    # parameters
                                                                    if "black-scholes" in response.lower():
                                                                        param_names = [
                                                                            "spot", "strike", "risk_free_rate", "volatility", "time_to_expiry"]
                                                                        for i, (name, value) in enumerate(
                                                                                zip(param_names, numbers[:5])):
                                                                            params[name] = float(
                                                                                value)

                                                                            # Option
                                                                            # type
                                                                            if "put" in response.lower():
                                                                                params["option_type"] = "put"
                                                                                else:
                                                                                    params["option_type"] = "call"

                                                                                    # Portfolio
                                                                                    # parameters
                                                                                    elif "portfolio" in response.lower():
                                                                                        # Extract
                                                                                        # symbols
                                                                                        symbols = re.findall(
                                                                                            r"\b[A-Z]{1,5}\b", response)
                                                                                        if symbols:
                                                                                            params["symbols"] = symbols

                                                                                            # Risk-free
                                                                                            # rate
                                                                                            if numbers:
                                                                                                params["risk_free_rate"] = float(
                                                                                                    numbers[0]) / 100 if float(numbers[0]) > 1 else float(numbers[0])

                                                                                                # VaR
                                                                                                # parameters
                                                                                                elif "var" in response.lower() or "value at risk" in response.lower():
                                                                                                    if numbers:
                                                                                                        # Confidence
                                                                                                        # level
                                                                                                        for num in numbers:
                                                                                                            if 0 < float(
                                                                                                                    num) < 1:
                                                                                                                params["confidence_level"] = float(
                                                                                                                    num)
                                                                                                                elif 90 <= float(num) <= 99:
                                                                                                                    params["confidence_level"] = float(
                                                                                                                        num) / 100

                                                                                                                    return params

                                                                                                                    def execute_model_request(
                                                                                                                            self, request: Dict[str, Any]) -> Dict[str, Any]:
                                                                                                                        """Execute a model request and return results"""
                                                                                                                        try:
                                                                                                                            if request["type"] == "ml" and request[
                                                                                                                                    "action"] == "train":
                                                                                                                                return self._execute_ml_model(
                                                                                                                                    request)

                                                                                                                                elif request["type"] == "financial" and request["action"] == "run":
                                                                                                                                    return self._execute_financial_model(
                                                                                                                                        request)

                                                                                                                                    elif request["type"] == "info":
                                                                                                                                        return self._handle_info_request(
                                                                                                                                            request)

                                                                                                                                        else:
                                                                                                                                            return {
                                                                                                                                                "success": False, "error": "Unknown request type"}

                                                                                                                                            except Exception as e:
                                                                                                                                                self.logger.error(
                                                                                                                                                    f"Model execution failed: {e}")
                                                                                                                                                return {
                                                                                                                                                    "success": False, "error": str(e)}

                                                                                                                                                def _execute_ml_model(
                                                                                                                                                        self, request: Dict[str, Any]) -> Dict[str, Any]:
                                                                                                                                                    """Execute ML model training"""
                                                                                                                                                    model_id = request[
                                                                                                                                                        "model_id"]
                                                                                                                                                    params = request[
                                                                                                                                                        "parameters"]

                                                                                                                                                    # Train
                                                                                                                                                    # model
                                                                                                                                                    result = self.ml_manager.train_model(
                                                                                                                                                        model_id,
                                                                                                                                                        symbol=params.get("symbol"),
                                                                                                                                                        data_source=params.get("data_source", "polygon"),
                                                                                                                                                        lookback_days=params.get("lookback_days", 365),
                                                                                                                                                    )

                                                                                                                                                    return {
                                                                                                                                                        "success": True,
                                                                                                                                                        "model_id": model_id,
                                                                                                                                                        "model_type": "ml",
                                                                                                                                                        "performance": result.performance_metrics,
                                                                                                                                                        "training_time": result.training_time,
                                                                                                                                                        "compliance_status": result.compliance_status,
                                                                                                                                                        "feature_importance": result.feature_importance.tolist() if result.feature_importance is not None else None,
                                                                                                                                                    }

                                                                                                                                                    def _execute_financial_model(
                                                                                                                                                            self, request: Dict[str, Any]) -> Dict[str, Any]:
                                                                                                                                                        """Execute financial model"""
                                                                                                                                                        model_id = request[
                                                                                                                                                            "model_id"]
                                                                                                                                                        params = request[
                                                                                                                                                            "parameters"]

                                                                                                                                                        # Run
                                                                                                                                                        # model
                                                                                                                                                        result = self.financial_manager.run_model(
                                                                                                                                                            model_id, **params)

                                                                                                                                                        response = {
                                                                                                                                                            "success": True,
                                                                                                                                                            "model_id": model_id,
                                                                                                                                                            "model_type": "financial",
                                                                                                                                                            "execution_time": result.execution_time,
                                                                                                                                                            "compliance_status": result.compliance_status,
                                                                                                                                                        }

                                                                                                                                                        # Add
                                                                                                                                                        # model-specific
                                                                                                                                                        # results
                                                                                                                                                        if result.option_price is not None:
                                                                                                                                                            response[
                                                                                                                                                                "option_price"] = result.option_price
                                                                                                                                                            response[
                                                                                                                                                                "greeks"] = result.greeks

                                                                                                                                                            if result.portfolio_weights is not None:
                                                                                                                                                                response["portfolio_weights"] = result.portfolio_weights.tolist(
                                                                                                                                                                )
                                                                                                                                                                response[
                                                                                                                                                                    "performance_metrics"] = result.performance_metrics

                                                                                                                                                                if result.risk_metrics:
                                                                                                                                                                    response[
                                                                                                                                                                        "risk_metrics"] = result.risk_metrics

                                                                                                                                                                    if result.predictions is not None:
                                                                                                                                                                        # For
                                                                                                                                                                        # large
                                                                                                                                                                        # arrays,
                                                                                                                                                                        # just
                                                                                                                                                                        # return
                                                                                                                                                                        # summary
                                                                                                                                                                        # statistics
                                                                                                                                                                        if isinstance(
                                                                                                                                                                                result.predictions, np.ndarray):
                                                                                                                                                                            response["predictions_summary"] = {
                                                                                                                                                                                "shape": result.predictions.shape,
                                                                                                                                                                                "mean": float(np.mean(result.predictions)),
                                                                                                                                                                                "std": float(np.std(result.predictions)),
                                                                                                                                                                                "min": float(np.min(result.predictions)),
                                                                                                                                                                                "max": float(np.max(result.predictions)),
                                                                                                                                                                            }

                                                                                                                                                                            return response

                                                                                                                                                                            def _handle_info_request(
                                                                                                                                                                                    self, request: Dict[str, Any]) -> Dict[str, Any]:
                                                                                                                                                                                """Handle information requests"""
                                                                                                                                                                                if request[
                                                                                                                                                                                        "action"] == "list":
                                                                                                                                                                                    return {
                                                                                                                                                                                        "success": True,
                                                                                                                                                                                        "ml_models": {
                                                                                                                                                                                            "total": len(self.ml_manager.get_available_models()),
                                                                                                                                                                                            "categories": list(self.ml_manager._get_category_summary().keys()),
                                                                                                                                                                                            "sample_models": self.ml_manager.get_available_models()[:10],
                                                                                                                                                                                        },
                                                                                                                                                                                        "financial_models": {
                                                                                                                                                                                            "total": len(self.financial_manager.get_available_models()),
                                                                                                                                                                                            "models": self.financial_manager.get_available_models(),
                                                                                                                                                                                        },
                                                                                                                                                                                    }

                                                                                                                                                                                    elif request["action"] == "best":
                                                                                                                                                                                        best_ml = self.ml_manager.get_best_models(
                                                                                                                                                                                            top_n=5)

                                                                                                                                                                                        return {"success": True, "best_ml_models": [
                                                                                                                                                                                            {"model_id": model_id, "r2_score": score} for model_id, score in best_ml], }

                                                                                                                                                                                        return {
                                                                                                                                                                                            "success": False, "error": "Unknown info request"}

                                                                                                                                                                                        def format_response_for_mltrainer(
                                                                                                                                                                                                self, execution_result: Dict[str, Any]) -> str:
                                                                                                                                                                                            """Format model execution results for mlTrainer"""
                                                                                                                                                                                            if not execution_result.get(
                                                                                                                                                                                                    "success"):
                                                                                                                                                                                                return f"Model execution failed: {execution_result.get('error', 'Unknown error')}"

                                                                                                                                                                                                lines = []

                                                                                                                                                                                                # ML
                                                                                                                                                                                                # model
                                                                                                                                                                                                # results
                                                                                                                                                                                                if execution_result.get(
                                                                                                                                                                                                        "model_type") == "ml":
                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                        f"✅ ML Model '{execution_result['model_id']}' trained successfully!")
                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                        f"Training time: {execution_result['training_time']:.2f} seconds")

                                                                                                                                                                                                    if execution_result.get(
                                                                                                                                                                                                            "performance"):
                                                                                                                                                                                                        perf = execution_result[
                                                                                                                                                                                                            "performance"]
                                                                                                                                                                                                        lines.append(
                                                                                                                                                                                                            "\nPerformance Metrics:")
                                                                                                                                                                                                        lines.append(
                                                                                                                                                                                                            f"  • R² Score: {perf.get('r2_score', 0):.4f}")
                                                                                                                                                                                                        lines.append(
                                                                                                                                                                                                            f"  • RMSE: {perf.get('rmse', 0):.4f}")
                                                                                                                                                                                                        lines.append(
                                                                                                                                                                                                            f"  • MAE: {perf.get('mae', 0):.4f}")
                                                                                                                                                                                                        lines.append(
                                                                                                                                                                                                            f"  • Directional Accuracy: {perf.get('directional_accuracy', 0):.2%}")

                                                                                                                                                                                                        # Financial
                                                                                                                                                                                                        # model
                                                                                                                                                                                                        # results
                                                                                                                                                                                                        elif execution_result.get("model_type") == "financial":
                                                                                                                                                                                                            lines.append(
                                                                                                                                                                                                                f"✅ Financial Model '{execution_result['model_id']}' executed successfully!")

                                                                                                                                                                                                            if "option_price" in execution_result:
                                                                                                                                                                                                                lines.append(
                                                                                                                                                                                                                    f"\nOption Price: ${execution_result['option_price']:.2f}")
                                                                                                                                                                                                                if execution_result.get(
                                                                                                                                                                                                                        "greeks"):
                                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                                        "Greeks:")
                                                                                                                                                                                                                    for greek, value in list(
                                                                                                                                                                                                                            execution_result["greeks"].items()):
                                                                                                                                                                                                                        lines.append(
                                                                                                                                                                                                                            f"  • {greek.capitalize()}: {value:.4f}")

                                                                                                                                                                                                                        if "portfolio_weights" in execution_result:
                                                                                                                                                                                                                            lines.append(
                                                                                                                                                                                                                                "\nOptimal Portfolio Weights:")
                                                                                                                                                                                                                            weights = execution_result[
                                                                                                                                                                                                                                "portfolio_weights"]
                                                                                                                                                                                                                            for i, weight in enumerate(
                                                                                                                                                                                                                                    weights):
                                                                                                                                                                                                                                lines.append(
                                                                                                                                                                                                                                    f"  • Asset {i+1}: {weight:.2%}")

                                                                                                                                                                                                                                if execution_result.get(
                                                                                                                                                                                                                                        "performance_metrics"):
                                                                                                                                                                                                                                    metrics = execution_result[
                                                                                                                                                                                                                                        "performance_metrics"]
                                                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                                                        "\nPortfolio Performance:")
                                                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                                                        f"  • Expected Return: {metrics.get('expected_return', 0):.2%}")
                                                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                                                        f"  • Volatility: {metrics.get('volatility', 0):.2%}")
                                                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                                                        f"  • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")

                                                                                                                                                                                                                                    if "risk_metrics" in execution_result:
                                                                                                                                                                                                                                        lines.append(
                                                                                                                                                                                                                                            "\nRisk Metrics:")
                                                                                                                                                                                                                                        for metric, value in list(
                                                                                                                                                                                                                                                execution_result["risk_metrics"].items()):
                                                                                                                                                                                                                                            if isinstance(
                                                                                                                                                                                                                                                    value, (int, float)):
                                                                                                                                                                                                                                                lines.append(
                                                                                                                                                                                                                                                    f"  • {metric.replace('_', ' ').title()}: {value:.4f}")

                                                                                                                                                                                                                                                # Info
                                                                                                                                                                                                                                                # results
                                                                                                                                                                                                                                                elif "ml_models" in execution_result:
                                                                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                                                                        "📊 Available Models:")
                                                                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                                                                        f"\nML Models: {execution_result['ml_models']['total']} models")
                                                                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                                                                        f"Categories: {', '.join(execution_result['ml_models']['categories'])}")
                                                                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                                                                        f"\nFinancial Models: {execution_result['financial_models']['total']} models")
                                                                                                                                                                                                                                                    lines.append(
                                                                                                                                                                                                                                                        "Available: " + ", ".join(execution_result["financial_models"]["models"]))

                                                                                                                                                                                                                                                    elif "best_ml_models" in execution_result:
                                                                                                                                                                                                                                                        lines.append(
                                                                                                                                                                                                                                                            "🏆 Best Performing ML Models:")
                                                                                                                                                                                                                                                        for model in execution_result[
                                                                                                                                                                                                                                                                "best_ml_models"]:
                                                                                                                                                                                                                                                            lines.append(
                                                                                                                                                                                                                                                                f"  • {model['model_id']}: R² = {model['r2_score']:.4f}")

                                                                                                                                                                                                                                                            return "\n".join(
                                                                                                                                                                                                                                                                lines)

                                                                                                                                                                                                                                                            def get_model_recommendations(
                                                                                                                                                                                                                                                                    self, context: Dict[str, Any]) -> List[str]:
                                                                                                                                                                                                                                                                """Get model recommendations based on context"""
                                                                                                                                                                                                                                                                recommendations = []

                                                                                                                                                                                                                                                                # Extract
                                                                                                                                                                                                                                                                # context
                                                                                                                                                                                                                                                                # information
                                                                                                                                                                                                                                                                objective = context.get(
                                                                                                                                                                                                                                                                    "objective", "")
                                                                                                                                                                                                                                                                data_type = context.get(
                                                                                                                                                                                                                                                                    "data_type", "")

                                                                                                                                                                                                                                                                # ML
                                                                                                                                                                                                                                                                # model
                                                                                                                                                                                                                                                                # recommendations
                                                                                                                                                                                                                                                                if "prediction" in objective or "forecast" in objective:
                                                                                                                                                                                                                                                                    recommendations.extend(
                                                                                                                                                                                                                                                                        [
                                                                                                                                                                                                                                                                            "For time series prediction, try: random_forest_100, gradient_boosting_100, or mlp_100",
                                                                                                                                                                                                                                                                            "For simple linear relationships, try: linear_regression, ridge_1.0, or lasso_0.1",
                                                                                                                                                                                                                                                                        ]
                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                    # Financial
                                                                                                                                                                                                                                                                    # model
                                                                                                                                                                                                                                                                    # recommendations
                                                                                                                                                                                                                                                                    if "portfolio" in objective:
                                                                                                                                                                                                                                                                        recommendations.append(
                                                                                                                                                                                                                                                                            "For portfolio optimization, use: mean_variance or risk_parity")

                                                                                                                                                                                                                                                                        if "option" in objective:
                                                                                                                                                                                                                                                                            recommendations.append(
                                                                                                                                                                                                                                                                                "For option pricing, use: black_scholes")

                                                                                                                                                                                                                                                                            if "risk" in objective:
                                                                                                                                                                                                                                                                                recommendations.append(
                                                                                                                                                                                                                                                                                    "For risk assessment, use: value_at_risk or stress_testing")

                                                                                                                                                                                                                                                                                if "trading" in objective or "strategy" in objective:
                                                                                                                                                                                                                                                                                    recommendations.append(
                                                                                                                                                                                                                                                                                        "For trading strategies, try: ma_crossover or rsi_strategy")

                                                                                                                                                                                                                                                                                    return recommendations

                                                                                                                                                                                                                                                                                    # Create
                                                                                                                                                                                                                                                                                    # integration
                                                                                                                                                                                                                                                                                    # instance
                                                                                                                                                                                                                                                                                    _model_integration = None

                                                                                                                                                                                                                                                                                    def get_model_integration() -> MLAgentModelIntegration:
                                                                                                                                                                                                                                                                                        """Get global model integration instance"""
                                                                                                                                                                                                                                                                                        global _model_integration
                                                                                                                                                                                                                                                                                        if _model_integration is None:
                                                                                                                                                                                                                                                                                            _model_integration = MLAgentModelIntegration()
                                                                                                                                                                                                                                                                                            return _model_integration

                                                                                                                                                                                                                                                                                            # Update
                                                                                                                                                                                                                                                                                            # mlAgent
                                                                                                                                                                                                                                                                                            # bridge
                                                                                                                                                                                                                                                                                            # to
                                                                                                                                                                                                                                                                                            # use
                                                                                                                                                                                                                                                                                            # model
                                                                                                                                                                                                                                                                                            # integration

                                                                                                                                                                                                                                                                                            def enhance_mlagent_with_models():
                                                                                                                                                                                                                                                                                                """Enhance mlAgent bridge with model execution capabilities"""
                                                                                                                                                                                                                                                                                                integration = get_model_integration()
                                                                                                                                                                                                                                                                                                mlagent = MLAgentBridge()

                                                                                                                                                                                                                                                                                                # Add
                                                                                                                                                                                                                                                                                                # model
                                                                                                                                                                                                                                                                                                # parsing
                                                                                                                                                                                                                                                                                                # to
                                                                                                                                                                                                                                                                                                # mlAgent
                                                                                                                                                                                                                                                                                                original_parse = mlagent.parse_mltrainer_response

                                                                                                                                                                                                                                                                                                def enhanced_parse(
                                                                                                                                                                                                                                                                                                        mltrainer_response: str) -> Dict[str, Any]:
                                                                                                                                                                                                                                                                                                    # First
                                                                                                                                                                                                                                                                                                    # try
                                                                                                                                                                                                                                                                                                    # original
                                                                                                                                                                                                                                                                                                    # parsing
                                                                                                                                                                                                                                                                                                    result = original_parse(
                                                                                                                                                                                                                                                                                                        mltrainer_response)

                                                                                                                                                                                                                                                                                                    # Then
                                                                                                                                                                                                                                                                                                    # check
                                                                                                                                                                                                                                                                                                    # for
                                                                                                                                                                                                                                                                                                    # model
                                                                                                                                                                                                                                                                                                    # requests
                                                                                                                                                                                                                                                                                                    model_request = integration.parse_model_request(
                                                                                                                                                                                                                                                                                                        mltrainer_response)
                                                                                                                                                                                                                                                                                                    if model_request[
                                                                                                                                                                                                                                                                                                            "model_id"] or model_request["action"]:
                                                                                                                                                                                                                                                                                                        # Execute
                                                                                                                                                                                                                                                                                                        # model
                                                                                                                                                                                                                                                                                                        # request
                                                                                                                                                                                                                                                                                                        execution_result = integration.execute_model_request(
                                                                                                                                                                                                                                                                                                            model_request)

                                                                                                                                                                                                                                                                                                        # Add
                                                                                                                                                                                                                                                                                                        # formatted
                                                                                                                                                                                                                                                                                                        # response
                                                                                                                                                                                                                                                                                                        # to
                                                                                                                                                                                                                                                                                                        # trials
                                                                                                                                                                                                                                                                                                        model_response = integration.format_response_for_mltrainer(
                                                                                                                                                                                                                                                                                                            execution_result)

                                                                                                                                                                                                                                                                                                        # Merge
                                                                                                                                                                                                                                                                                                        # with
                                                                                                                                                                                                                                                                                                        # original
                                                                                                                                                                                                                                                                                                        # result
                                                                                                                                                                                                                                                                                                        if "trial_suggestions" not in result:
                                                                                                                                                                                                                                                                                                            result["trial_suggestions"] = [
                                                                                                                                                                                                                                                                                                            ]

                                                                                                                                                                                                                                                                                                            result["trial_suggestions"].append(
                                                                                                                                                                                                                                                                                                                {
                                                                                                                                                                                                                                                                                                                    "description": f"Model execution: {model_request['model_id'] or model_request['action']}",
                                                                                                                                                                                                                                                                                                                    "model_response": model_response,
                                                                                                                                                                                                                                                                                                                    "execution_result": execution_result,
                                                                                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                                                                            )

                                                                                                                                                                                                                                                                                                            return result

                                                                                                                                                                                                                                                                                                            # Replace
                                                                                                                                                                                                                                                                                                            # the
                                                                                                                                                                                                                                                                                                            # parse
                                                                                                                                                                                                                                                                                                            # method
                                                                                                                                                                                                                                                                                                            mlagent.parse_mltrainer_response = enhanced_parse

                                                                                                                                                                                                                                                                                                            logger.info(
                                                                                                                                                                                                                                                                                                                "mlAgent enhanced with model execution capabilities")

                                                                                                                                                                                                                                                                                                            # Auto-enhance
                                                                                                                                                                                                                                                                                                            # on
                                                                                                                                                                                                                                                                                                            # import
                                                                                                                                                                                                                                                                                                            enhance_mlagent_with_models()
