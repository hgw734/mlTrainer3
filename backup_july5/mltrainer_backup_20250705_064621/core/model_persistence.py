"""
mlTrainer - Model Persistence Manager
====================================

Purpose: Handles saving and loading of trained ML models to enable
persistence across sessions and efficient model reuse for predictions.

Features:
- Model serialization using joblib for sklearn models
- XGBoost and LightGBM native serialization
- Model metadata tracking (training date, samples, accuracy)
- Automatic model versioning and backup
"""

import os
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelPersistence:
    """Manages model persistence and loading operations"""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize model persistence manager
        
        Args:
            models_dir: Directory to store saved models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different model types
        self.sklearn_dir = self.models_dir / "sklearn"
        self.xgboost_dir = self.models_dir / "xgboost"
        self.lightgbm_dir = self.models_dir / "lightgbm"
        self.metadata_dir = self.models_dir / "metadata"
        
        for dir_path in [self.sklearn_dir, self.xgboost_dir, self.lightgbm_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
            
        logger.info(f"ModelPersistence initialized with models directory: {self.models_dir}")
    
    def save_model(self, model_name: str, model: Any, metadata: Dict[str, Any]) -> bool:
        """Save a trained model with metadata
        
        Args:
            model_name: Name of the model
            model: Trained model object
            metadata: Model metadata (accuracy, training_samples, etc.)
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Determine model type and save accordingly
            if model_name in ["RandomForest", "LinearRegression", "Ridge", "Lasso", "SVR"]:
                model_path = self.sklearn_dir / f"{model_name}_{timestamp}.joblib"
                joblib.dump(model, model_path)
                
            elif model_name == "XGBoost":
                model_path = self.xgboost_dir / f"{model_name}_{timestamp}.json"
                model.save_model(str(model_path))
                
            elif model_name == "LightGBM":
                model_path = self.lightgbm_dir / f"{model_name}_{timestamp}.txt"
                model.booster_.save_model(str(model_path))
                
            else:
                # Fallback to joblib for unknown model types
                model_path = self.sklearn_dir / f"{model_name}_{timestamp}.joblib"
                joblib.dump(model, model_path)
            
            # Save metadata
            metadata_enhanced = {
                **metadata,
                "model_name": model_name,
                "saved_timestamp": timestamp,
                "model_path": str(model_path),
                "model_type": self._get_model_type(model_name)
            }
            
            metadata_path = self.metadata_dir / f"{model_name}_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata_enhanced, f, indent=2)
            
            # Update latest model pointer
            latest_path = self.metadata_dir / f"{model_name}_latest.json"
            with open(latest_path, 'w') as f:
                json.dump(metadata_enhanced, f, indent=2)
            
            logger.info(f"Model {model_name} saved successfully to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str, timestamp: Optional[str] = None) -> Optional[tuple]:
        """Load a saved model
        
        Args:
            model_name: Name of the model to load
            timestamp: Specific timestamp, or None for latest
            
        Returns:
            tuple: (model_object, metadata) or None if not found
        """
        try:
            # Determine which metadata file to use
            if timestamp:
                metadata_path = self.metadata_dir / f"{model_name}_{timestamp}.json"
            else:
                metadata_path = self.metadata_dir / f"{model_name}_latest.json"
            
            if not metadata_path.exists():
                logger.warning(f"No saved model found for {model_name}")
                return None
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            model_path = Path(metadata["model_path"])
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Load model based on type
            model_type = metadata.get("model_type", "sklearn")
            
            if model_type == "sklearn":
                model = joblib.load(model_path)
                
            elif model_type == "xgboost":
                import xgboost as xgb
                model = xgb.XGBRegressor()
                model.load_model(str(model_path))
                
            elif model_type == "lightgbm":
                import lightgbm as lgb
                model = lgb.LGBMRegressor()
                model.booster_ = lgb.Booster(model_file=str(model_path))
                
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            logger.info(f"Model {model_name} loaded successfully from {model_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def list_saved_models(self) -> Dict[str, Dict[str, Any]]:
        """List all saved models with their metadata
        
        Returns:
            Dict mapping model names to their latest metadata
        """
        saved_models = {}
        
        for metadata_file in self.metadata_dir.glob("*_latest.json"):
            try:
                model_name = metadata_file.stem.replace("_latest", "")
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                saved_models[model_name] = {
                    "accuracy": metadata.get("accuracy", 0.0),
                    "training_samples": metadata.get("training_samples", 0),
                    "saved_timestamp": metadata.get("saved_timestamp", "unknown"),
                    "features_used": metadata.get("features_used", 0),
                    "model_path": metadata.get("model_path", "")
                }
                
            except Exception as e:
                logger.error(f"Error reading metadata for {metadata_file}: {e}")
        
        return saved_models
    
    def delete_model(self, model_name: str, timestamp: Optional[str] = None) -> bool:
        """Delete a saved model
        
        Args:
            model_name: Name of the model to delete
            timestamp: Specific timestamp, or None for latest
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            # Find and delete metadata file
            if timestamp:
                metadata_path = self.metadata_dir / f"{model_name}_{timestamp}.json"
            else:
                metadata_path = self.metadata_dir / f"{model_name}_latest.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Delete model file
                model_path = Path(metadata["model_path"])
                if model_path.exists():
                    model_path.unlink()
                
                # Delete metadata file
                metadata_path.unlink()
                
                logger.info(f"Model {model_name} deleted successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    def cleanup_old_models(self, keep_versions: int = 5) -> int:
        """Clean up old model versions, keeping only the most recent
        
        Args:
            keep_versions: Number of versions to keep for each model
            
        Returns:
            int: Number of files deleted
        """
        deleted_count = 0
        
        # Group models by name
        model_groups = {}
        for metadata_file in self.metadata_dir.glob("*.json"):
            if metadata_file.name.endswith("_latest.json"):
                continue
                
            # Extract model name and timestamp
            parts = metadata_file.stem.split("_")
            if len(parts) >= 3:
                model_name = "_".join(parts[:-2])
                timestamp = "_".join(parts[-2:])
                
                if model_name not in model_groups:
                    model_groups[model_name] = []
                model_groups[model_name].append((timestamp, metadata_file))
        
        # Delete old versions
        for model_name, versions in model_groups.items():
            if len(versions) > keep_versions:
                # Sort by timestamp (newest first)
                versions.sort(key=lambda x: x[0], reverse=True)
                
                # Delete old versions
                for timestamp, metadata_file in versions[keep_versions:]:
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Delete model file
                        model_path = Path(metadata["model_path"])
                        if model_path.exists():
                            model_path.unlink()
                            deleted_count += 1
                        
                        # Delete metadata file
                        metadata_file.unlink()
                        deleted_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error during cleanup for {metadata_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleanup completed: deleted {deleted_count} old model files")
        
        return deleted_count
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine the model type for serialization
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Model type (sklearn, xgboost, lightgbm, etc.)
        """
        if model_name == "XGBoost":
            return "xgboost"
        elif model_name == "LightGBM":
            return "lightgbm"
        else:
            return "sklearn"
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a saved model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with model information or None if not found
        """
        latest_path = self.metadata_dir / f"{model_name}_latest.json"
        
        if not latest_path.exists():
            return None
        
        try:
            with open(latest_path, 'r') as f:
                metadata = json.load(f)
            
            # Add file size information
            model_path = Path(metadata["model_path"])
            if model_path.exists():
                metadata["file_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None