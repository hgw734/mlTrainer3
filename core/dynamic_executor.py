"""
Dynamic Action Generator
Generates and executes actions from natural language descriptions
"""

import ast
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security violation exception"""
    pass


class DynamicActionGenerator:
    """
    Generates executable actions from natural language descriptions
    Uses templates and code generation for dynamic functionality
    """

    def __init__(self):
        self.generated_actions: Dict[str, Dict[str, Any]] = {}
        self.action_templates = {
            "data_fetch": '''
def fetch_data(data_type="{data_type}"):
    """Fetch data from approved sources"""
    # Implementation would connect to Polygon/FRED APIs
    return {{"data_type": data_type, "status": "fetched"}}
''',
            "technical_indicator": '''
def calculate_indicator(indicator="{indicator}", period={default_period}):
    """Calculate technical indicator"""
    # Implementation would calculate the specified indicator
    return {{"indicator": indicator, "period": period, "value": 0.0}}
''',
            "model_ensemble": '''
def create_ensemble(ensemble_type="{ensemble_type}"):
    """Create model ensemble"""
    # Implementation would combine multiple models
    return {{"ensemble_type": ensemble_type, "models": []}}
''',
            "risk_analysis": '''
def analyze_risk(risk_type="{risk_type}"):
    """Perform risk analysis"""
    # Implementation would calculate risk metrics
    return {{"risk_type": risk_type, "value": 0.0}}
''',
        }

    def generate_action(self, description: str, parameters: Dict[str, Any]) -> Optional[Callable]:
        """Generate an action from natural language description"""
        try:
            # Identify action type from description
            action_type = self._identify_action_type(description)

            if not action_type:
                logger.warning(f"Could not identify action type from: {description}")
                return None

            # Generate function code
            function_code = self._generate_function_code(action_type, parameters)

            if not function_code:
                return None

            # Compile and create function
            function = self._compile_function(function_code)

            if function:
                # Cache the generated action
                action_id = f"generated_{action_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.generated_actions[action_id] = {
                    "function": function,
                    "description": description,
                    "code": function_code,
                    "created_at": datetime.now().isoformat(),
                }

                logger.info(f"Generated action: {action_id}")
                return function

        except Exception as e:
            logger.error(f"Failed to generate action: {e}")

        return None

    def _identify_action_type(self, description: str) -> Optional[str]:
        """Identify the type of action from description"""
        description_lower = description.lower()

        # Map keywords to action types
        keyword_map = {
            "fetch": "data_fetch",
            "get data": "data_fetch",
            "download": "data_fetch",
            "calculate indicator": "technical_indicator",
            "technical analysis": "technical_indicator",
            "sma": "technical_indicator",
            "ema": "technical_indicator",
            "rsi": "technical_indicator",
            "ensemble": "model_ensemble",
            "combine models": "model_ensemble",
            "voting": "model_ensemble",
            "risk": "risk_analysis",
            "var": "risk_analysis",
            "sharpe": "risk_analysis",
            "drawdown": "risk_analysis",
        }

        for keyword, action_type in list(keyword_map.items()):
            if keyword in description_lower:
                return action_type

        return None

    def _generate_function_code(self, action_type: str, parameters: Dict[str, Any]) -> Optional[str]:
        """Generate function code based on action type and parameters"""
        template = self.action_templates.get(action_type)

        if not template:
            return None

        # Fill in template parameters
        if action_type == "data_fetch":
            code = template.format(data_type=parameters.get("data_type", "ohlcv"))
        elif action_type == "technical_indicator":
            code = template.format(
                indicator=parameters.get("indicator", "sma"),
                default_period=parameters.get("period", 20)
            )
        elif action_type == "model_ensemble":
            code = template.format(ensemble_type=parameters.get("ensemble_type", "voting"))
        elif action_type == "risk_analysis":
            code = template.format(risk_type=parameters.get("risk_type", "var"))
        else:
            code = template

        return code

    def _compile_function(self, code: str) -> Optional[Callable]:
        """Compile function code and return callable"""
        try:
            # Parse the code
            tree = ast.parse(code)

            # Compile to bytecode
            code_obj = compile(tree, "<dynamic>", "exec")

            # Execute in a namespace
            namespace = {}
            
            # Security check before execution
            if not self.is_safe_to_execute(code):
                raise SecurityError("Code failed safety check")
            
            # SECURITY: Dynamic execution disabled for compliance
            # execute() is disabled for compliance. If dynamic execution is needed, use a secure sandboxed alternative.
            return self._production_implementation()

        except Exception as e:
            logger.error(f"Failed to compile function: {e}")

        return None

    def is_safe_to_execute(self, code: str) -> bool:
        """Check if code is safe to execute"""
        # Basic security checks
        dangerous_patterns = [
            "import os",
            "import sys",
            "import subprocess",
            "eval(",
            "exec(",
            "__import__",
            "open(",
            "file(",
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False
        
        return True

    def _production_implementation(self) -> Callable:
        """Return a safe production implementation"""
        def safe_function(**kwargs):
            return {"status": "safe_execution", "message": "Dynamic execution disabled for compliance"}
        return safe_function

    def execute_generated_action(self, action_id: str, **kwargs) -> Any:
        """Execute a previously generated action"""
        if action_id not in self.generated_actions:
            raise ValueError(f"Action {action_id} not found")

        action = self.generated_actions[action_id]
        function = action["function"]

        try:
            result = function(**kwargs)

            # Log execution
            logger.info(f"Executed generated action {action_id}")

            return {
                "success": True,
                "result": result,
                "action_id": action_id,
                "executed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to execute generated action: {e}")
            return {"success": False, "error": str(e), "action_id": action_id}

    def list_generated_actions(self) -> List[Dict[str, Any]]:
        """List all generated actions"""
        return [
            {
                "action_id": action_id,
                "description": info["description"],
                "created_at": info["created_at"]
            }
            for action_id, info in list(self.generated_actions.items())
        ]

    def get_action_code(self, action_id: str) -> Optional[str]:
        """Get the generated code for an action"""
        if action_id in self.generated_actions:
            return self.generated_actions[action_id]["code"]
        return None

    def save_generated_actions(self, filepath: str = "generated_actions.json"):
        """Save generated actions to file"""
        try:
            # Convert functions to serializable format
            serializable = {}
            for action_id, info in list(self.generated_actions.items()):
                serializable[action_id] = {
                    "description": info["description"],
                    "code": info["code"],
                    "created_at": info["created_at"],
                }

            with open(filepath, "w") as f:
                json.dump(serializable, f, indent=2)

            logger.info(f"Saved {len(serializable)} generated actions")
            return True

        except Exception as e:
            logger.error(f"Failed to save generated actions: {e}")
            return False

    def load_generated_actions(self, filepath: str = "generated_actions.json") -> bool:
        """Load previously generated actions from file"""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            for action_id, info in list(data.items()):
                # Recompile the function
                function = self._compile_function(info["code"])
                if function:
                    self.generated_actions[action_id] = {
                        "function": function,
                        "description": info["description"],
                        "code": info["code"],
                        "created_at": info["created_at"],
                    }

            logger.info(f"Loaded {len(self.generated_actions)} generated actions")
            return True

        except Exception as e:
            logger.error(f"Failed to load generated actions: {e}")
            return False


# Global instance
_dynamic_generator = None


def get_dynamic_action_generator() -> DynamicActionGenerator:
    """Get the dynamic action generator instance"""
    global _dynamic_generator
    if _dynamic_generator is None:
        _dynamic_generator = DynamicActionGenerator()
    return _dynamic_generator
