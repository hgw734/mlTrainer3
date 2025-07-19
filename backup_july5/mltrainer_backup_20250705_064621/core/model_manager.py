"""
mlTrainer - Model Manager
========================

Purpose: Manages ML model lifecycle including loading, unloading, versioning,
and performance tracking. Handles regime-aware model selection and ensemble
coordination.

Features:
- Dynamic model loading/unloading
- Performance monitoring
- Model versioning
- Regime-specific model selection
- Ensemble coordination
"""

import logging
import os
import pickle
import json
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ML model lifecycle and performance"""
    
    def __init__(self):
        self.models_path = "models/"
        self.model_registry = {}
        self.performance_cache = {}
        self.model_metadata = {}
        
        # Ensure models directory exists
        Path(self.models_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing model registry
        self._load_model_registry()
        
        logger.info("ModelManager initialized")
    
    def _load_model_registry(self):
        """Load model registry from disk"""
        registry_path = os.path.join(self.models_path, "model_registry.json")
        
        try:
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
                logger.info(f"Loaded model registry with {len(self.model_registry)} entries")
            else:
                self.model_registry = {}
                logger.info("Created new model registry")
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            self.model_registry = {}
    
    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_path = os.path.join(self.models_path, "model_registry.json")
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=2, default=str)
            logger.info("Model registry saved")
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def register_model(self, model_name: str, model_type: str, regime_type: str = "default",
                      version: str = "1.0", metadata: Dict = None) -> str:
        """Register a new model in the registry"""
        model_id = f"{model_name}_{regime_type}_{version}"
        
        registration_data = {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "regime_type": regime_type,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "status": "registered",
            "file_path": None,
            "performance": {},
            "metadata": metadata or {}
        }
        
        self.model_registry[model_id] = registration_data
        self._save_model_registry()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def save_model(self, model, model_id: str, overwrite: bool = False) -> bool:
        """Save model to disk and update registry"""
        if model_id not in self.model_registry:
            logger.error(f"Model {model_id} not registered")
            return False
        
        model_path = os.path.join(self.models_path, f"{model_id}.pkl")
        
        if os.path.exists(model_path) and not overwrite:
            logger.error(f"Model file already exists: {model_path}")
            return False
        
        try:
            # Save model using joblib for better sklearn compatibility
            joblib.dump(model, model_path)
            
            # Update registry
            self.model_registry[model_id]["file_path"] = model_path
            self.model_registry[model_id]["status"] = "saved"
            self.model_registry[model_id]["saved_at"] = datetime.now().isoformat()
            self.model_registry[model_id]["file_size"] = os.path.getsize(model_path)
            
            self._save_model_registry()
            
            logger.info(f"Model saved: {model_id} to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            return False
    
    def load_model(self, model_id: str):
        """Load model from disk"""
        if model_id not in self.model_registry:
            logger.error(f"Model {model_id} not registered")
            return None
        
        model_info = self.model_registry[model_id]
        model_path = model_info.get("file_path")
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            model = joblib.load(model_path)
            
            # Update registry
            self.model_registry[model_id]["status"] = "loaded"
            self.model_registry[model_id]["loaded_at"] = datetime.now().isoformat()
            
            logger.info(f"Model loaded: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    def update_model_performance(self, model_id: str, performance_metrics: Dict):
        """Update model performance metrics"""
        if model_id not in self.model_registry:
            logger.error(f"Model {model_id} not registered")
            return False
        
        try:
            # Add timestamp to performance metrics
            performance_metrics["updated_at"] = datetime.now().isoformat()
            
            # Update registry
            self.model_registry[model_id]["performance"] = performance_metrics
            
            # Cache performance for quick access
            self.performance_cache[model_id] = performance_metrics
            
            self._save_model_registry()
            
            logger.info(f"Updated performance for model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update performance for {model_id}: {e}")
            return False
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all registered models"""
        status_data = {}
        
        for model_id, model_info in self.model_registry.items():
            status_data[model_info["model_name"]] = {
                "loaded": model_info.get("status") == "loaded",
                "accuracy": model_info.get("performance", {}).get("accuracy", 0),
                "last_updated": model_info.get("performance", {}).get("updated_at", "Never"),
                "training_samples": model_info.get("performance", {}).get("training_samples", 0),
                "regime_type": model_info.get("regime_type", "default"),
                "version": model_info.get("version", "1.0"),
                "model_type": model_info.get("model_type", "unknown"),
                "file_size": model_info.get("file_size", 0)
            }
        
        return status_data
    
    def get_models_by_regime(self, regime_type: str) -> List[str]:
        """Get all models suitable for a specific regime"""
        regime_models = []
        
        for model_id, model_info in self.model_registry.items():
            if (model_info.get("regime_type") == regime_type or 
                model_info.get("regime_type") == "default"):
                regime_models.append(model_id)
        
        return regime_models
    
    def get_best_models(self, regime_type: str, metric: str = "accuracy", top_n: int = 3) -> List[str]:
        """Get best performing models for a regime"""
        regime_models = self.get_models_by_regime(regime_type)
        
        # Sort by performance metric
        model_performance = []
        for model_id in regime_models:
            model_info = self.model_registry[model_id]
            performance = model_info.get("performance", {})
            metric_value = performance.get(metric, 0)
            model_performance.append((model_id, metric_value))
        
        # Sort by metric value (descending)
        model_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N model IDs
        return [model_id for model_id, _ in model_performance[:top_n]]
    
    def cleanup_old_models(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up old model files and registry entries"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        cleanup_results = {
            "models_removed": 0,
            "files_deleted": 0,
            "space_freed": 0,
            "errors": []
        }
        
        models_to_remove = []
        
        for model_id, model_info in self.model_registry.items():
            try:
                created_at = datetime.fromisoformat(model_info.get("created_at", "1970-01-01"))
                
                if created_at < cutoff_date:
                    # Remove file if exists
                    file_path = model_info.get("file_path")
                    if file_path and os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleanup_results["files_deleted"] += 1
                        cleanup_results["space_freed"] += file_size
                    
                    models_to_remove.append(model_id)
                    
            except Exception as e:
                cleanup_results["errors"].append(f"Error cleaning {model_id}: {e}")
        
        # Remove from registry
        for model_id in models_to_remove:
            del self.model_registry[model_id]
            cleanup_results["models_removed"] += 1
        
        # Save updated registry
        if models_to_remove:
            self._save_model_registry()
        
        logger.info(f"Cleanup completed: {cleanup_results['models_removed']} models removed")
        return cleanup_results
    
    def export_model_metadata(self, model_id: str) -> Optional[Dict]:
        """Export complete model metadata"""
        if model_id not in self.model_registry:
            return None
        
        model_info = self.model_registry[model_id].copy()
        
        # Add runtime information
        model_info["export_timestamp"] = datetime.now().isoformat()
        model_info["registry_version"] = "1.0"
        
        return model_info
    
    def import_model_metadata(self, metadata: Dict) -> bool:
        """Import model metadata from external source"""
        try:
            model_id = metadata.get("model_id")
            if not model_id:
                logger.error("No model_id in metadata")
                return False
            
            # Validate required fields
            required_fields = ["model_name", "model_type", "regime_type"]
            if not all(field in metadata for field in required_fields):
                logger.error("Missing required metadata fields")
                return False
            
            # Add import timestamp
            metadata["imported_at"] = datetime.now().isoformat()
            
            # Add to registry
            self.model_registry[model_id] = metadata
            self._save_model_registry()
            
            logger.info(f"Imported model metadata: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import model metadata: {e}")
            return False
    
    def validate_model_integrity(self, model_id: str) -> Dict[str, Any]:
        """Validate model file integrity and metadata consistency"""
        validation_results = {
            "model_id": model_id,
            "valid": False,
            "issues": [],
            "timestamp": datetime.now().isoformat()
        }
        
        if model_id not in self.model_registry:
            validation_results["issues"].append("Model not in registry")
            return validation_results
        
        model_info = self.model_registry[model_id]
        
        # Check file existence
        file_path = model_info.get("file_path")
        if not file_path:
            validation_results["issues"].append("No file path in registry")
        elif not os.path.exists(file_path):
            validation_results["issues"].append("Model file missing")
        else:
            try:
                # Try to load the model
                model = joblib.load(file_path)
                validation_results["model_loadable"] = True
                
                # Check file size consistency
                actual_size = os.path.getsize(file_path)
                registered_size = model_info.get("file_size", 0)
                
                if actual_size != registered_size:
                    validation_results["issues"].append("File size mismatch")
                
            except Exception as e:
                validation_results["issues"].append(f"Model loading failed: {e}")
        
        # Check metadata completeness
        required_fields = ["model_name", "model_type", "regime_type", "created_at"]
        for field in required_fields:
            if field not in model_info:
                validation_results["issues"].append(f"Missing metadata field: {field}")
        
        validation_results["valid"] = len(validation_results["issues"]) == 0
        
        return validation_results
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get model development lineage and history"""
        if model_id not in self.model_registry:
            return {"error": "Model not found"}
        
        model_info = self.model_registry[model_id]
        
        # Find related models (same name, different versions)
        model_name = model_info.get("model_name")
        regime_type = model_info.get("regime_type")
        
        related_models = []
        for mid, minfo in self.model_registry.items():
            if (minfo.get("model_name") == model_name and 
                minfo.get("regime_type") == regime_type and 
                mid != model_id):
                related_models.append({
                    "model_id": mid,
                    "version": minfo.get("version"),
                    "created_at": minfo.get("created_at"),
                    "performance": minfo.get("performance", {}).get("accuracy", 0)
                })
        
        # Sort by creation date
        related_models.sort(key=lambda x: x["created_at"])
        
        return {
            "current_model": model_info,
            "related_models": related_models,
            "total_versions": len(related_models) + 1,
            "lineage_timestamp": datetime.now().isoformat()
        }

