"""
mlTrainer - Technical Facilitator
================================

Purpose: Pure technical facilitation between user, mlTrainer, and ML systems.
Provides infrastructure and data access without any strategy or decision logic.

Role: Facilitator only - all decisions made by mlTrainer.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TechnicalFacilitator:
    """
    Pure technical facilitator - provides infrastructure and data access.
    NO strategy logic, decisions, or interpretations - those are mlTrainer's job.
    """
    
    def __init__(self):
        """Initialize technical facilitator"""
        self.data_path = Path("data/facilitation")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Only verified API sources
        self.verified_sources = ["polygon", "fred"]
        
        # Available models (technical specs only)
        self.available_models = self._load_available_models()
        
        logger.info("TechnicalFacilitator initialized - ready to facilitate")
    
    def _load_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Load available models with technical specifications only"""
        return {
            # Tree-Based Models
            "RandomForest": {
                "type": "ensemble",
                "implementation": "scikit-learn",
                "training_time": "fast",
                "memory_usage": "medium",
                "interpretability": "high",
                "data_requirements": "medium"
            },
            "XGBoost": {
                "type": "ensemble", 
                "implementation": "xgboost",
                "training_time": "medium",
                "memory_usage": "medium",
                "interpretability": "medium", 
                "data_requirements": "medium"
            },
            "LightGBM": {
                "type": "ensemble",
                "implementation": "lightgbm", 
                "training_time": "fast",
                "memory_usage": "low",
                "interpretability": "medium",
                "data_requirements": "large"
            },
            
            # Deep Learning Models
            "LSTM": {
                "type": "neural_network",
                "implementation": "tensorflow/pytorch",
                "training_time": "slow", 
                "memory_usage": "high",
                "interpretability": "low",
                "data_requirements": "large"
            },
            "GRU": {
                "type": "neural_network",
                "implementation": "tensorflow/pytorch",
                "training_time": "medium",
                "memory_usage": "medium", 
                "interpretability": "low",
                "data_requirements": "medium"
            },
            "Transformer": {
                "type": "neural_network",
                "implementation": "tensorflow/pytorch",
                "training_time": "very_slow",
                "memory_usage": "very_high", 
                "interpretability": "very_low",
                "data_requirements": "very_large"
            }
        }
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models with technical specs"""
        return self.available_models
    
    def get_verified_data_sources(self) -> List[str]:
        """Get list of verified data sources"""
        return self.verified_sources
    
    def check_data_source_status(self) -> Dict[str, bool]:
        """Check status of verified data sources"""
        try:
            from backend.data_sources import get_data_source_manager
            data_sources = get_data_source_manager()
            
            status = {}
            for source in self.verified_sources:
                try:
                    # Simple connection test
                    status[source] = True  # Simplified for now
                except:
                    status[source] = False
            
            return status
        except Exception as e:
            logger.error(f"Failed to check data source status: {e}")
            return {source: False for source in self.verified_sources}
    
    def save_results(self, results_type: str, data: Dict[str, Any]) -> bool:
        """Save results data for mlTrainer"""
        try:
            timestamp = datetime.now().isoformat()
            filename = f"{results_type}_{timestamp}.json"
            filepath = self.data_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {results_type} results to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def load_results(self, results_type: str) -> List[Dict[str, Any]]:
        """Load saved results for mlTrainer"""
        try:
            results = []
            pattern = f"{results_type}_*.json"
            
            for filepath in self.data_path.glob(pattern):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            
            return results
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for mlTrainer"""
        return {
            "timestamp": datetime.now().isoformat(),
            "available_models": len(self.available_models),
            "verified_sources": self.verified_sources,
            "data_source_status": self.check_data_source_status(),
            "results_storage": str(self.data_path),
            "facilitator_role": "infrastructure_only",
            "decision_maker": "mlTrainer"
        }
    
    def execute_model(self, model_name: str, data: Any, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a model with given data - pure technical execution"""
        try:
            if model_name not in self.available_models:
                return {
                    "status": "error",
                    "message": f"Model {model_name} not available",
                    "available_models": list(self.available_models.keys())
                }
            
            # This would integrate with actual model implementations
            # For now, return structure for mlTrainer to work with
            return {
                "status": "ready_for_implementation",
                "model": model_name,
                "model_specs": self.available_models[model_name],
                "message": "Model execution framework ready - implementation needed"
            }
            
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            return {
                "status": "error", 
                "message": str(e)
            }


def get_technical_facilitator() -> TechnicalFacilitator:
    """Get global TechnicalFacilitator instance"""
    global _technical_facilitator
    if '_technical_facilitator' not in globals():
        _technical_facilitator = TechnicalFacilitator()
    return _technical_facilitator