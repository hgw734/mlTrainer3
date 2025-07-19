"""
mlTrainer System Configuration Manager
====================================

Single source of truth for ALL system parameters, data sources, and configuration.
No hard coding anywhere - everything references this centralized configuration.

Purpose: Eliminate hard coding by providing centralized access to all system parameters
"""

import yaml
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemConfiguration:
    """Centralized configuration manager - single source of truth for all system parameters"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemConfiguration, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_configuration()
    
    def load_configuration(self, config_path: str = "config/system_configuration.yaml"):
        """Load system configuration from YAML file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.error(f"Configuration file not found: {config_path}")
                self._config = self._get_emergency_fallback_config()
                return
                
            with open(config_file, 'r') as f:
                self._config = yaml.safe_load(f)
                
            logger.info(f"✅ System configuration loaded from {config_path}")
            logger.info(f"📊 Configuration version: {self._config.get('version', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            self._config = self._get_emergency_fallback_config()
    
    def _get_emergency_fallback_config(self) -> Dict[str, Any]:
        """Emergency fallback configuration - minimal settings"""
        return {
            "version": "emergency_fallback",
            "core": {
                "max_training_samples": 1000,
                "min_training_samples": 20,
                "default_training_days": 90,
                "confidence_threshold": 0.85
            },
            "data_sources": {
                "verified_sources": ["polygon", "fred"],
                "primary_market_data": "polygon",
                "primary_economic_data": "fred"
            },
            "infrastructure": {
                "ports": {"streamlit": 5000, "flask_backend": 8000}
            }
        }
    
    # ==================
    # CORE SYSTEM ACCESS
    # ==================
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            "name": self._config.get("system_name", "mlTrainer"),
            "version": self._config.get("version", "unknown"),
            "description": self._config.get("system_description", "Trading Intelligence System")
        }
    
    def get_core_settings(self) -> Dict[str, Any]:
        """Get core system settings"""
        return self._config.get("core", {})
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """Get ML training parameters"""
        core = self.get_core_settings()
        model_config = self._config.get("models", {})
        
        return {
            "max_samples": core.get("max_training_samples", 1000),
            "min_samples": core.get("min_training_samples", 20), 
            "default_days": core.get("default_training_days", 90),
            "confidence_threshold": core.get("confidence_threshold", 0.85),
            "random_state": model_config.get("training_parameters", {}).get("random_state", 42),
            "test_size": model_config.get("training_parameters", {}).get("test_size", 0.2),
            "n_estimators": model_config.get("training_parameters", {}).get("n_estimators", 100),
            "cv_folds": model_config.get("training_parameters", {}).get("cross_validation_folds", 5)
        }
    
    def get_cpu_allocation(self) -> Dict[str, int]:
        """Get CPU allocation settings"""
        return self._config.get("core", {}).get("cpu_allocation", {"ml_training": 6, "system_operations": 2})
    
    # ==================
    # TRADING CONFIG
    # ==================
    
    def get_trading_objectives(self) -> Dict[str, Any]:
        """Get trading objectives and targets"""
        return self._config.get("trading", {})
    
    def get_target_timeframes(self) -> Dict[str, Any]:
        """Get target timeframes and returns"""
        trading = self.get_trading_objectives()
        return trading.get("timeframes", {})
    
    def get_universe_config(self) -> Dict[str, Any]:
        """Get trading universe configuration"""
        trading = self.get_trading_objectives()
        return trading.get("universe", {})
    
    def get_default_tickers(self) -> List[str]:
        """Get default test tickers"""
        universe = self.get_universe_config()
        return universe.get("default_test_tickers", ["AAPL", "MSFT", "GOOGL"])
    
    # ==================
    # DATA SOURCES
    # ==================
    
    def get_verified_sources(self) -> List[str]:
        """Get list of verified data sources"""
        return self._config.get("data_sources", {}).get("verified_sources", ["polygon", "fred"])
    
    def get_primary_market_source(self) -> str:
        """Get primary market data source"""
        return self._config.get("data_sources", {}).get("primary_market_data", "polygon")
    
    def get_primary_economic_source(self) -> str:
        """Get primary economic data source"""
        return self._config.get("data_sources", {}).get("primary_economic_data", "fred")
    
    def get_polygon_config(self) -> Dict[str, Any]:
        """Get Polygon API configuration"""
        return self._config.get("data_sources", {}).get("polygon", {})
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration"""
        polygon = self.get_polygon_config()
        return {
            "rps": polygon.get("rate_limit_rps", 50),
            "dropout_threshold": polygon.get("dropout_threshold", 0.15)
        }
    
    # ==================
    # AI PROVIDERS
    # ==================
    
    def get_ai_provider_config(self) -> Dict[str, Any]:
        """Get AI provider configuration"""
        return self._config.get("ai_providers", {})
    
    def get_primary_ai_provider(self) -> str:
        """Get primary AI provider"""
        return self._config.get("ai_providers", {}).get("primary", "anthropic")
    
    def get_anthropic_model(self) -> str:
        """Get Anthropic model name"""
        providers = self.get_ai_provider_config()
        return providers.get("anthropic", {}).get("model", "claude-sonnet-4-20250514")
    
    # ==================
    # MODEL CONFIGURATION
    # ==================
    
    def get_model_registry_info(self) -> Dict[str, Any]:
        """Get model registry information"""
        models = self._config.get("models", {})
        return {
            "total_count": models.get("total_registry_count", 120),
            "categories": models.get("categories", 20)
        }
    
    def get_authentic_models(self) -> Dict[str, List[str]]:
        """Get authentic model implementations"""
        models = self._config.get("models", {})
        return models.get("authentic_implementations", {})
    
    def get_blocked_categories(self) -> List[str]:
        """Get blocked model categories"""
        models = self._config.get("models", {})
        return models.get("blocked_categories", [])
    
    # ==================
    # COMPLIANCE CONFIG
    # ==================
    
    def get_compliance_config(self) -> Dict[str, Any]:
        """Get compliance configuration"""
        return self._config.get("compliance", {})
    
    def get_blocked_data_indicators(self) -> List[str]:
        """Get blocked data indicators"""
        compliance = self.get_compliance_config()
        return compliance.get("blocked_data_indicators", ["mock", "fake", "synthetic"])
    
    def get_exempted_patterns(self) -> List[str]:
        """Get exempted file patterns"""
        compliance = self.get_compliance_config()
        return compliance.get("exempted_patterns", ["chat_history", "mltrainer_chat"])
    
    def get_audit_schedule(self) -> Dict[str, Any]:
        """Get compliance audit schedule"""
        compliance = self.get_compliance_config()
        return compliance.get("audit_schedule", {})
    
    def get_data_quality_standards(self) -> Dict[str, Any]:
        """Get data quality standards"""
        compliance = self.get_compliance_config()
        return compliance.get("data_quality_standards", {})
    
    # ==================
    # INFRASTRUCTURE
    # ==================
    
    def get_ports(self) -> Dict[str, int]:
        """Get port configuration"""
        infra = self._config.get("infrastructure", {})
        return infra.get("ports", {"streamlit": 5000, "flask_backend": 8000})
    
    def get_directories(self) -> Dict[str, str]:
        """Get directory configuration"""
        infra = self._config.get("infrastructure", {})
        return infra.get("directories", {})
    
    def get_file_patterns(self) -> Dict[str, str]:
        """Get file naming patterns"""
        infra = self._config.get("infrastructure", {})
        return infra.get("file_patterns", {})
    
    # ==================
    # FEATURES & MONITORING
    # ==================
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self._config.get("features", {})
    
    def get_technical_indicators(self) -> List[str]:
        """Get technical indicators list"""
        features = self.get_feature_config()
        return features.get("technical_indicators", ["SMA", "EMA", "MACD", "RSI"])
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self._config.get("monitoring", {})
    
    def get_alert_types(self) -> List[str]:
        """Get alert types"""
        monitoring = self.get_monitoring_config()
        return monitoring.get("alert_types", ["regime_change", "entry_signal", "exit_signal"])
    
    # ==================
    # REGIME DETECTION
    # ==================
    
    def get_regime_config(self) -> Dict[str, Any]:
        """Get regime detection configuration"""
        return self._config.get("regime", {})
    
    def get_regime_thresholds(self) -> Dict[str, float]:
        """Get regime detection thresholds"""
        regime = self.get_regime_config()
        return regime.get("thresholds", {})
    
    # ==================
    # VALIDATION RULES
    # ==================
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration"""
        return self._config.get("validation", {})
    
    def get_trial_validation_checks(self) -> Dict[str, List[str]]:
        """Get trial validation check categories"""
        validation = self.get_validation_config()
        return validation.get("trial_validation", {})
    
    # ==================
    # UTILITY METHODS
    # ==================
    
    def get_config_section(self, section: str) -> Dict[str, Any]:
        """Get any configuration section by name"""
        return self._config.get(section, {})
    
    def get_nested_config(self, *keys) -> Any:
        """Get nested configuration value using dot notation"""
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration completeness"""
        required_sections = [
            "core", "trading", "data_sources", "ai_providers", 
            "models", "compliance", "infrastructure"
        ]
        
        validation_results = {
            "valid": True,
            "missing_sections": [],
            "warnings": []
        }
        
        for section in required_sections:
            if section not in self._config:
                validation_results["missing_sections"].append(section)
                validation_results["valid"] = False
        
        return validation_results
    
    def reload_configuration(self):
        """Reload configuration from file"""
        self._config = None
        self.load_configuration()
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration (for debugging)"""
        return self._config.copy()

# Global singleton instance
_system_config = None

def get_system_config() -> SystemConfiguration:
    """Get global system configuration instance"""
    global _system_config
    if _system_config is None:
        _system_config = SystemConfiguration()
    return _system_config

# Convenience functions for common access patterns
def get_training_params() -> Dict[str, Any]:
    """Quick access to training parameters"""
    return get_system_config().get_training_parameters()

def get_verified_sources() -> List[str]:
    """Quick access to verified data sources"""
    return get_system_config().get_verified_sources()

def get_default_tickers() -> List[str]:
    """Quick access to default test tickers"""
    return get_system_config().get_default_tickers()

def get_ports() -> Dict[str, int]:
    """Quick access to port configuration"""
    return get_system_config().get_ports()

def get_authentic_models() -> Dict[str, List[str]]:
    """Quick access to authentic model lists"""
    return get_system_config().get_authentic_models()