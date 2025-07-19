"""
mlTrainer - Configuration Manager
================================

Purpose: Centralized configuration management for the mlTrainer system.
Handles loading, validation, and updating of configuration files for
APIs, ML models, compliance settings, and system parameters.

Features:
- YAML/JSON configuration loading
- Environment variable integration
- Configuration validation
- Hot reloading capabilities
- Secure API key management
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages system configuration from multiple sources"""
    
    def __init__(self):
        self.config_dir = "config/"
        self.configs = {}
        self.config_files = {
            "api": "api_config.yaml",
            "ml": "ml_config.yaml", 
            "compliance": "compliance_config.yaml"
        }
        
        # Ensure config directory exists
        Path(self.config_dir).mkdir(parents=True, exist_ok=True)
        
        # Load all configurations
        self._load_all_configs()
        
        logger.info("ConfigManager initialized")
    
    def _load_all_configs(self):
        """Load all configuration files"""
        for config_name, filename in self.config_files.items():
            config_path = os.path.join(self.config_dir, filename)
            
            if os.path.exists(config_path):
                self.configs[config_name] = self._load_config_file(config_path)
                logger.info(f"Loaded {config_name} configuration")
            else:
                self.configs[config_name] = self._get_default_config(config_name)
                self._save_config(config_name, self.configs[config_name])
                logger.info(f"Created default {config_name} configuration")
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    return yaml.safe_load(f) or {}
                elif file_path.endswith('.json'):
                    return json.load(f) or {}
                else:
                    logger.warning(f"Unknown config file format: {file_path}")
                    return {}
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            return {}
    
    def _get_default_config(self, config_name: str) -> Dict[str, Any]:
        """Get default configuration for specified config type"""
        defaults = {
            "api": {
                "polygon": {
                    "base_url": "https://api.polygon.io",
                    "api_key_env": "POLYGON_API_KEY",
                    "rate_limit": 100,
                    "timeout": 15
                },
                "fred": {
                    "base_url": "https://api.stlouisfed.org/fred",
                    "api_key_env": "FRED_API_KEY", 
                    "rate_limit": 120,
                    "timeout": 15
                },
                "quiverquant": {
                    "base_url": "https://api.quiverquant.com",
                    "api_key_env": "QUIVERQUANT_API_KEY",
                    "rate_limit": 60,
                    "timeout": 15
                },
                "anthropic": {
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "temperature": 0.1
                }
            },
            "ml": {
                "models": {
                    "RandomForest": {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "random_state": 42,
                        "enabled": True
                    },
                    "XGBoost": {
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 6,
                        "enabled": True
                    },
                    "LightGBM": {
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "num_leaves": 31,
                        "enabled": True
                    },
                    "LSTM": {
                        "units": 50,
                        "dropout": 0.2,
                        "epochs": 50,
                        "batch_size": 32,
                        "enabled": False
                    }
                },
                "training": {
                    "test_size": 0.2,
                    "validation_size": 0.1,
                    "min_samples": 100,
                    "retrain_threshold": 0.7,
                    "max_training_time": 3600
                },
                "features": {
                    "technical_indicators": True,
                    "volume_features": True,
                    "price_features": True,
                    "macro_features": True,
                    "lookback_periods": [5, 10, 20]
                }
            },
            "compliance": {
                "strict_mode": True,
                "verified_sources": ["polygon", "fred", "quiverquant"],
                "max_data_age_hours": 24,
                "require_timestamps": True,
                "allow_synthetic_data": False,
                "response_template": "I don't know. But based on the data, I would suggest {suggestion}.",
                "data_validation": {
                    "check_source_verification": True,
                    "check_timestamp_freshness": True,
                    "check_data_completeness": True
                }
            }
        }
        
        return defaults.get(config_name, {})
    
    def _save_config(self, config_name: str, config_data: Dict[str, Any]):
        """Save configuration to file"""
        if config_name not in self.config_files:
            logger.error(f"Unknown config name: {config_name}")
            return False
        
        filename = self.config_files[config_name]
        file_path = os.path.join(self.config_dir, filename)
        
        try:
            with open(file_path, 'w') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif filename.endswith('.json'):
                    json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Saved {config_name} configuration to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config {config_name}: {e}")
            return False
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get complete configuration by name"""
        return self.configs.get(config_name, {})
    
    def get_config_value(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """Get specific configuration value using dot notation"""
        config = self.configs.get(config_name, {})
        
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_config_value(self, config_name: str, key_path: str, value: Any) -> bool:
        """Set specific configuration value using dot notation"""
        if config_name not in self.configs:
            self.configs[config_name] = {}
        
        config = self.configs[config_name]
        keys = key_path.split('.')
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        
        # Save to file
        return self._save_config(config_name, self.configs[config_name])
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """Get API-specific configuration with environment variables"""
        api_config = self.get_config_value("api", api_name, {})
        
        if not api_config:
            return {}
        
        # Resolve environment variables
        resolved_config = api_config.copy()
        
        if "api_key_env" in api_config:
            env_var = api_config["api_key_env"]
            api_key = os.getenv(env_var, "")
            resolved_config["api_key"] = api_key
            resolved_config["api_key_configured"] = bool(api_key)
        
        return resolved_config
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get ML configuration"""
        return self.get_config("ml")
    
    def get_compliance_config(self) -> Dict[str, Any]:
        """Get compliance configuration"""
        return self.get_config("compliance")
    
    def validate_config(self, config_name: str) -> Dict[str, Any]:
        """Validate configuration completeness and correctness"""
        validation_results = {
            "config_name": config_name,
            "valid": True,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.now().isoformat()
        }
        
        config = self.configs.get(config_name, {})
        
        if not config:
            validation_results["valid"] = False
            validation_results["errors"].append("Configuration is empty")
            return validation_results
        
        # Validate API configurations
        if config_name == "api":
            required_apis = ["polygon", "fred", "quiverquant", "anthropic"]
            for api in required_apis:
                if api not in config:
                    validation_results["errors"].append(f"Missing {api} API configuration")
                else:
                    api_config = config[api]
                    if "api_key_env" in api_config:
                        env_var = api_config["api_key_env"]
                        if not os.getenv(env_var):
                            validation_results["warnings"].append(f"{api} API key not set in environment")
        
        # Validate ML configurations
        elif config_name == "ml":
            if "models" not in config:
                validation_results["errors"].append("No models configuration found")
            else:
                enabled_models = [name for name, conf in config["models"].items() 
                                if conf.get("enabled", True)]
                if not enabled_models:
                    validation_results["warnings"].append("No models are enabled")
        
        # Validate compliance configurations
        elif config_name == "compliance":
            required_fields = ["strict_mode", "verified_sources", "max_data_age_hours"]
            for field in required_fields:
                if field not in config:
                    validation_results["errors"].append(f"Missing compliance field: {field}")
        
        validation_results["valid"] = len(validation_results["errors"]) == 0
        
        return validation_results
    
    def reload_config(self, config_name: str) -> bool:
        """Reload specific configuration from file"""
        if config_name not in self.config_files:
            logger.error(f"Unknown config name: {config_name}")
            return False
        
        filename = self.config_files[config_name]
        file_path = os.path.join(self.config_dir, filename)
        
        if os.path.exists(file_path):
            self.configs[config_name] = self._load_config_file(file_path)
            logger.info(f"Reloaded {config_name} configuration")
            return True
        else:
            logger.error(f"Config file not found: {file_path}")
            return False
    
    def export_config(self, config_name: str, export_path: str) -> bool:
        """Export configuration to specified path"""
        if config_name not in self.configs:
            logger.error(f"Config {config_name} not found")
            return False
        
        try:
            export_data = {
                "config_name": config_name,
                "exported_at": datetime.now().isoformat(),
                "config": self.configs[config_name]
            }
            
            with open(export_path, 'w') as f:
                if export_path.endswith('.yaml') or export_path.endswith('.yml'):
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported {config_name} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export config: {e}")
            return False
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations"""
        return self.configs.copy()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations"""
        summary = {
            "total_configs": len(self.configs),
            "config_files": self.config_files,
            "config_status": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for config_name in self.configs:
            validation = self.validate_config(config_name)
            summary["config_status"][config_name] = {
                "valid": validation["valid"],
                "error_count": len(validation["errors"]),
                "warning_count": len(validation["warnings"])
            }
        
        return summary

