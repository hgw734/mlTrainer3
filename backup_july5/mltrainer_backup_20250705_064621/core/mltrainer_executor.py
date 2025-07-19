"""
mlTrainer Executor - AI-to-System Bridge
=======================================

Purpose: Intermediary system that parses mlTrainer's suggestions and executes
them against the actual ML infrastructure. Bridges the gap between Claude's
text responses and real system operations.
"""

import logging
import json
import re
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MLTrainerExecutor:
    """
    Intermediary system that executes mlTrainer's suggestions on real infrastructure
    """
    
    def __init__(self, backend_url: str = "http://127.0.0.1:8000"):
        self.backend_url = backend_url
        self.execution_log = []
        self.active_trial_mode = False  # Tracks if we're in active trial vs pre-initialization
        
        # Initialize centralized model registry access - SINGLE SOURCE OF TRUTH
        from .model_registry import get_model_registry
        self.model_registry = get_model_registry()
        self.available_models = self.model_registry.get_all_models()
        
        # Enhanced action patterns that mlTrainer might suggest
        self.action_patterns = {
            'momentum_screening': [
                r'(?:initiate|start|begin|launch|suggest|recommend).{0,50}momentum.{0,50}screening',
                r'(?:screen|analyze|identify).{0,50}momentum.{0,50}stocks',
                r'momentum.{0,50}analysis.{0,50}(?:trial|test)',
                r'find.{0,50}momentum.{0,50}(?:stocks|opportunities)'
            ],
            'walk_forward_test': [
                r'(?:initiate|start|begin|launch).{0,50}walk.{0,20}forward',
                r'walk.forward.{0,50}(?:test|testing|analysis)',
                r'(?:paper|back).{0,20}test.{0,50}strategy'
            ],
            'regime_detection': [
                r'(?:detect|analyze|assess).{0,50}(?:market\s)?regime',
                r'regime.{0,50}(?:analysis|detection|classification)',
                r'market.{0,50}condition.{0,50}analysis'
            ],
            'model_execution': [
                r'(?:execute|run|use).{0,50}(?:model|ensemble)',
                r'(?:train|apply).{0,50}(?:model|ML)',
                r'(?:implement|deploy).{0,50}ML.{0,50}model'
            ],
            'trial_execution': [
                r'(?:run|start|initiate).{0,50}trial',
                r'(?:suggest|recommend).{0,50}(?:we|I).{0,50}(?:run|start)',
                r'ready.{0,50}(?:to|for).{0,50}(?:proceed|start)'
            ]
        }
        
        # Confidence indicators for stronger parsing (excluding "execute" to prevent false triggers)
        self.confidence_patterns = [
            r'I suggest we',
            r'I recommend',
            r'Let\'s (?:initiate|start|begin)',
            r'Ready to (?:proceed|start)',
            r'propose.{0,20}(?:trial|analysis|screening)',
            r'shall we (?:initiate|start|begin)',
            r'(?:should|could) we (?:run|start|initiate)'
        ]
        
        logger.info("MLTrainerExecutor initialized - ready to bridge AI to systems")
    
    def parse_mltrainer_response(self, mltrainer_text: str, trial_mode: bool = False) -> Dict[str, Any]:
        """
        Parse mlTrainer's text response to extract actionable suggestions
        
        Args:
            mltrainer_text: The text response from mlTrainer
            trial_mode: If True, allows full command authority (including "execute" patterns)
                       If False, restricts to suggestion patterns only (pre-initialization security)
        """
        suggestions = {
            'actions': [],
            'models_mentioned': [],
            'timeframes_mentioned': [],
            'stocks_mentioned': [],
            'confidence_levels': [],
            'executable': False
        }
        
        # Use different pattern sets based on mode
        if trial_mode:
            # ACTIVE TRIAL MODE: Full command authority - include all patterns including "execute"
            active_trial_patterns = self.action_patterns.copy()
            active_trial_patterns['trial_execution'].extend([
                r'(?:execute|run|start|initiate).{0,50}trial',
                r'execute.{0,50}(?:momentum|regime|analysis)',
                r'(?:command|order).{0,50}(?:execution|run)',
                r'proceed.{0,50}with.{0,50}execution'
            ])
            patterns_to_use = active_trial_patterns
        else:
            # PRE-INITIALIZATION MODE: Restricted patterns - exclude "execute" 
            patterns_to_use = self.action_patterns
        
        # Extract action suggestions using appropriate patterns
        for action_type, patterns in patterns_to_use.items():
            for pattern in patterns:
                if re.search(pattern, mltrainer_text, re.IGNORECASE):
                    suggestions['actions'].append(action_type)
                    break  # Found match for this action type
        
        # Check for confidence indicators to boost detection
        has_confidence_indicator = False
        for conf_pattern in self.confidence_patterns:
            if re.search(conf_pattern, mltrainer_text, re.IGNORECASE):
                has_confidence_indicator = True
                break
        
        # Extract model mentions - use centralized registry
        from .model_registry import get_model_registry
        registry = get_model_registry()
        model_names = registry.get_all_models()
        
        # Create pattern from registry models
        model_patterns = [re.escape(model) for model in model_names]
        model_patterns.extend([r'ensemble', r'neural network'])  # Add general terms
        
        for pattern in model_patterns:
            matches = re.findall(pattern, mltrainer_text, re.IGNORECASE)
            suggestions['models_mentioned'].extend(matches)
        
        # Remove case-insensitive duplicates from models mentioned
        seen_models = set()
        unique_models = []
        for model in suggestions['models_mentioned']:
            model_lower = model.lower()
            if model_lower not in seen_models:
                seen_models.add(model_lower)
                unique_models.append(model)
        suggestions['models_mentioned'] = unique_models
        
        # Extract timeframes
        timeframe_patterns = [
            r'7.{0,5}10.{0,5}day', r'3.{0,5}month', r'9.{0,5}month',
            r'short.{0,10}term', r'medium.{0,10}term', r'long.{0,10}term'
        ]
        for pattern in timeframe_patterns:
            matches = re.findall(pattern, mltrainer_text, re.IGNORECASE)
            suggestions['timeframes_mentioned'].extend(matches)
        
        # Extract confidence levels
        confidence_pattern = r'(\d{2,3})%?\s*confidence'
        confidence_matches = re.findall(confidence_pattern, mltrainer_text, re.IGNORECASE)
        suggestions['confidence_levels'].extend([int(x) for x in confidence_matches])
        
        # Determine if response contains executable suggestions
        suggestions['executable'] = len(suggestions['actions']) > 0
        
        return suggestions
    
    def execute_momentum_screening(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute momentum screening trial using specified models"""
        try:
            # Use default models from centralized registry if none specified
            if not models:
                # Get fast, reliable models for default screening from our registry
                all_models = self.available_models
                models = [model for model in all_models if model in ['RandomForest', 'XGBoost', 'LightGBM']]
                if not models:
                    models = all_models[:3]  # Fallback to first 3 models
            
            # Call momentum screening endpoint  
            response = requests.post(
                f"{self.backend_url}/api/facilitator/momentum-screening",
                json={"models": models},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "momentum_screening",
                    "status": "success",
                    "models": models,
                    "result": result
                })
                return {"success": True, "data": result}
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Momentum screening execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_regime_detection(self) -> Dict[str, Any]:
        """Execute regime detection analysis"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/regime-analysis",
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "regime_detection",
                    "status": "success",
                    "result": result
                })
                return {"success": True, "data": result}
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Regime detection execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_walk_forward_test(self, models: List[str] = None) -> Dict[str, Any]:
        """Execute walk-forward testing with specified models"""
        try:
            models_list = models if models else ['RandomForest', 'XGBoost']
            response = requests.post(
                f"{self.backend_url}/api/facilitator/walk-forward-test",
                json={"models": models_list},
                headers={"Content-Type": "application/json"},
                timeout=60  # Longer timeout for backtesting
            )
            
            if response.status_code == 200:
                result = response.json()
                self.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "walk_forward_test",
                    "status": "success",
                    "models": models_list,
                    "result": result
                })
                return {"success": True, "data": result}
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Walk-forward test execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_trial_execution(self, suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute general trial based on parsed suggestions"""
        try:
            # For trial_execution, we can delegate to momentum screening as the primary action
            if suggestions.get('models_mentioned'):
                return self.execute_momentum_screening(suggestions['models_mentioned'])
            else:
                # Execute with default configuration
                response = requests.post(
                    f"{self.backend_url}/api/facilitator/momentum-screening",
                    json={"models": ["RandomForest", "XGBoost"]},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.execution_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "action": "trial_execution",
                        "status": "success",
                        "result": result
                    })
                    return {"success": True, "data": result}
                else:
                    return {"success": False, "error": f"API returned {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Trial execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/facilitator/system-status",
                timeout=10
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}
                
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_suggestion(self, mltrainer_response: str, user_approved: bool = False) -> Dict[str, Any]:
        """
        Main execution method - parses mlTrainer response and executes if approved
        """
        suggestions = self.parse_mltrainer_response(mltrainer_response)
        
        if not suggestions['executable']:
            return {
                "success": False,
                "message": "No executable actions found in mlTrainer response",
                "suggestions": suggestions
            }
        
        if not user_approved:
            return {
                "success": False,
                "message": "User approval required before execution",
                "suggestions": suggestions,
                "awaiting_approval": True
            }
        
        # Execute the suggested actions
        results = []
        
        if 'momentum_screening' in suggestions['actions']:
            result = self.execute_momentum_screening(suggestions['models_mentioned'])
            results.append({"action": "momentum_screening", "result": result})
        
        if 'regime_detection' in suggestions['actions']:
            result = self.execute_regime_detection()
            results.append({"action": "regime_detection", "result": result})
        
        if 'walk_forward_test' in suggestions['actions']:
            result = self.execute_walk_forward_test(suggestions['models_mentioned'])
            results.append({"action": "walk_forward_test", "result": result})
        
        if 'trial_execution' in suggestions['actions']:
            result = self.execute_trial_execution(suggestions)
            results.append({"action": "trial_execution", "result": result})
        
        return {
            "success": True,
            "message": f"Executed {len(results)} actions",
            "results": results,
            "suggestions": suggestions
        }
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get log of all executed actions"""
        return self.execution_log
    
    def get_available_models(self) -> List[str]:
        """Get all available models from centralized registry - SINGLE SOURCE OF TRUTH"""
        return self.available_models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model from centralized registry"""
        return self.model_registry.get_model_info(model_name)
    
    def get_momentum_models(self) -> List[str]:
        """Get all momentum-specific models for trading analysis"""
        momentum_models = []
        for model_name in self.available_models:
            model_info = self.model_registry.get_model_info(model_name)
            category = model_info.get("category", "")
            if "Momentum" in category or "Volume" in category or "Pattern" in category:
                momentum_models.append(model_name)
        return momentum_models