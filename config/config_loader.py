#!/usr/bin/env python3
"""
Central Configuration Loader - Single Source of Truth
No hardcoded values anywhere in the system
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing"""

    pass


@dataclass
class ConfigValue:
    """Wrapper for config values with metadata"""

    value: Any
    source: str
    validated: bool = False
    compliance_checked: bool = False


class ConfigLoader:
    """
    Central configuration management system
    ALL configuration values flow through here
    """

    def __init__(self, env: str = None):
        self.env = env or os.getenv("MLTRAINER_ENV", "development")
        self.config_dir = Path(__file__).parent
        self._cache = {}
        self._load_all_configs()

    def _load_all_configs(self):
        """Load all configuration files"""
        # Load base configs
        self._load_api_config()
        self._load_models_config()
        self._load_ai_config()
        self._load_compliance_config()

        # Apply environment overrides
        self._apply_environment_overrides()

        # Validate all configs
        self._validate_all_configs()

    def _load_api_config(self):
        """Load API configuration - NO HARDCODED VALUES"""
        try:
            # Import existing api_config.py
            from . import api_config

            # Ensure no hardcoded keys
            if not os.getenv("POLYGON_API_KEY"):
                raise ConfigurationError("POLYGON_API_KEY environment variable required")
            if not os.getenv("FRED_API_KEY"):
                raise ConfigurationError("FRED_API_KEY environment variable required")

            self._cache["api"] = {
                "polygon": {
                    "api_key": api_config.POLYGON_API_KEY,
                    "endpoints": api_config.APPROVED_ENDPOINTS.get(api_config.APISource.POLYGON, {}),
                    "rate_limit": 100,
                    "timeout": 30,
                },
                "fred": {
                    "api_key": api_config.FRED_API_KEY,
                    "endpoints": api_config.APPROVED_ENDPOINTS.get(api_config.APISource.FRED, {}),
                    "rate_limit": 120,
                    "timeout": 30,
                },
            }
        except ImportError as e:
            raise ConfigurationError(f"Failed to load API config: {e}")

    def _load_models_config(self):
        """Load models configuration"""
        try:
            from . import models_config

            self._cache["models"] = models_config.MODEL_REGISTRY
        except ImportError as e:
            raise ConfigurationError(f"Failed to load models config: {e}")

    def _load_ai_config(self):
        """Load AI configuration"""
        try:
            from . import ai_config

            self._cache["ai"] = {
                "models": ai_config.AI_MODELS,
                "permissions": ai_config.AI_PERMISSIONS,
                "trust_levels": ai_config.TRUST_LEVELS,
            }
        except ImportError as e:
            raise ConfigurationError(f"Failed to load AI config: {e}")

    def _load_compliance_config(self):
        """Load compliance configuration"""
        try:
            from . import immutable_compliance_gateway

            self._cache["compliance"] = {
                "rules": immutable_compliance_gateway.COMPLIANCE_RULES,
                "data_sources": immutable_compliance_gateway.APPROVED_DATA_SOURCES,
                "protection_constants": immutable_compliance_gateway.PROTECTION_CONSTANTS,
            }
        except ImportError as e:
            raise ConfigurationError(f"Failed to load compliance config: {e}")

    def _apply_environment_overrides(self):
        """Apply environment-specific overrides"""
        override_file = self.config_dir / f"environments/{self.env}.yaml"
        if override_file.exists():
            with open(override_file, "r") as f:
                overrides = yaml.safe_load(f)
                # Deep merge overrides into cache
                self._deep_merge(self._cache, overrides)

    def _validate_all_configs(self):
        """Validate all loaded configurations"""
        # Check required API keys
        if not self._cache.get("api", {}).get("polygon", {}).get("api_key"):
            raise ConfigurationError("Polygon API key not configured")

        if not self._cache.get("api", {}).get("fred", {}).get("api_key"):
            raise ConfigurationError("FRED API key not configured")

        # Validate model registry
        if not self._cache.get("models"):
            raise ConfigurationError("No models configured")

        # Validate compliance rules
        if not self._cache.get("compliance", {}).get("rules"):
            raise ConfigurationError("No compliance rules configured")

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation path
        Examples: 'api.polygon.rate_limit', 'models.random_forest.hyperparameters'
        """
        keys = path.split(".")
        value = self._cache

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_all(self, prefix: str) -> Dict[str, Any]:
        """Get all config values under a prefix"""
        return self.get(prefix, {})

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge override into base"""
        for key, value in list(override.items()):
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def validate_against_compliance(self, config_path: str, value: Any) -> bool:
        """Validate a config value against compliance rules"""
        compliance_rules = self.get("compliance.rules", {})

        # Check if this config path has compliance rules
        for rule_path, rule in list(compliance_rules.items()):
            if config_path.startswith(rule_path):
                # Apply compliance validation
                return self._validate_value(value, rule)

        return True

    def _validate_value(self, value: Any, rule: Dict) -> bool:
        """Validate value against a compliance rule"""
        if "allowed_values" in rule and value not in rule["allowed_values"]:
            return False

        if "min_value" in rule and value < rule["min_value"]:
            return False

        if "max_value" in rule and value > rule["max_value"]:
            return False

        if "pattern" in rule:
            import re

            if not re.match(rule["pattern"], str(value)):
                return False

        return True


# Singleton instance
_config_loader = None


def get_config_loader(env: str = None) -> ConfigLoader:
    """Get singleton config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(env)
    return _config_loader


def get_config(path: str, default: Any = None) -> Any:
    """Convenience function to get config value"""
    return get_config_loader().get(path, default)
