"""
mlTrainer - API Provider Manager
===============================

Purpose: Centralized API provider management system that handles configuration,
authentication, and provider selection for all AI models and data sources.
Ensures no hardcoded API keys and provides seamless provider switching.

Features:
- Centralized configuration management
- Dynamic provider selection with fallbacks
- API key management from environment variables
- Rate limiting and health monitoring
- Compliance-verified data source management
"""

import os
import yaml
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class APIProviderConfig:
    """Configuration for an API provider"""
    name: str
    service_type: str
    api_key_env: str
    base_url: str
    api_key: Optional[str] = None
    models: Optional[Dict] = None
    endpoints: Optional[Dict] = None
    limits: Optional[Dict] = None
    capabilities: Optional[List] = None
    compliance: Optional[Dict] = None
    enabled: bool = True

@dataclass
class ProviderStatus:
    """Status tracking for an API provider"""
    is_healthy: bool = True
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    response_time_ms: Optional[float] = None

class APIProviderManager:
    """Centralized API provider management with configuration cascading"""
    
    def __init__(self, config_path: str = "config/api_providers.yaml"):
        self.config_path = config_path
        self.config = None
        self.providers = {}
        self.provider_status = {}
        self._lock = Lock()
        
        # Load configuration
        self._load_configuration()
        self._initialize_providers()
        
        logger.info("APIProviderManager initialized with cascading configuration")
    
    def _load_configuration(self) -> None:
        """Load API provider configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Loaded API provider configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback configuration if file cannot be loaded"""
        return {
            "ai_providers": {
                "active_provider": "anthropic",
                "providers": {
                    "anthropic": {
                        "name": "Anthropic Claude",
                        "service_type": "ai_chat",
                        "api_key_env": "ANTHROPIC_API_KEY",
                        "base_url": "https://api.anthropic.com",
                        "models": {
                            "default": "claude-sonnet-4-20250514"
                        }
                    }
                }
            },
            "data_providers": {
                "market_data": {
                    "active_provider": "polygon",
                    "providers": {
                        "polygon": {
                            "name": "Polygon.io",
                            "service_type": "market_data",
                            "api_key_env": "POLYGON_API_KEY",
                            "base_url": "https://api.polygon.io"
                        }
                    }
                },
                "economic_data": {
                    "active_provider": "fred",
                    "providers": {
                        "fred": {
                            "name": "Federal Reserve Economic Data",
                            "service_type": "economic_data", 
                            "api_key_env": "FRED_API_KEY",
                            "base_url": "https://api.stlouisfed.org/fred"
                        }
                    }
                }
            }
        }
    
    def _initialize_providers(self) -> None:
        """Initialize all configured providers with environment variables"""
        with self._lock:
            # Initialize AI providers
            ai_providers = self.config.get("ai_providers", {}).get("providers", {})
            for provider_id, provider_config in ai_providers.items():
                self._initialize_provider(provider_id, provider_config)
            
            # Initialize data providers
            data_providers = self.config.get("data_providers", {})
            for category, category_config in data_providers.items():
                if category == "alternative_data":
                    continue  # Skip alternative data per user request
                    
                providers = category_config.get("providers", {})
                for provider_id, provider_config in providers.items():
                    self._initialize_provider(f"{category}_{provider_id}", provider_config)
    
    def _initialize_provider(self, provider_id: str, config: Dict[str, Any]) -> None:
        """Initialize a single provider with API key from environment"""
        try:
            # Get API key from environment
            api_key_env = config.get("api_key_env")
            api_key = os.environ.get(api_key_env) if api_key_env else None
            
            # Create provider config
            provider = APIProviderConfig(
                name=config.get("name", provider_id),
                service_type=config.get("service_type", "unknown"),
                api_key_env=api_key_env,
                base_url=config.get("base_url", ""),
                api_key=api_key,
                models=config.get("models"),
                endpoints=config.get("endpoints"),
                limits=config.get("limits"),
                capabilities=config.get("capabilities", []),
                compliance=config.get("compliance", {}),
                enabled=config.get("enabled", True) and api_key is not None
            )
            
            self.providers[provider_id] = provider
            self.provider_status[provider_id] = ProviderStatus()
            
            if provider.enabled:
                logger.info(f"Initialized provider: {provider.name} ({provider_id})")
            else:
                logger.warning(f"Provider disabled (no API key): {provider.name} ({provider_id})")
                
        except Exception as e:
            logger.error(f"Failed to initialize provider {provider_id}: {e}")
    
    def get_active_ai_provider(self) -> Optional[APIProviderConfig]:
        """Get the currently active AI provider"""
        active_id = self.config.get("ai_providers", {}).get("active_provider", "anthropic")
        provider = self.providers.get(active_id)
        
        if provider and provider.enabled:
            return provider
        
        # Try fallback
        fallback_id = self.config.get("selection_rules", {}).get("ai_provider_selection", {}).get("fallback")
        if fallback_id and fallback_id in self.providers:
            fallback_provider = self.providers[fallback_id]
            if fallback_provider.enabled:
                logger.warning(f"Using fallback AI provider: {fallback_provider.name}")
                return fallback_provider
        
        logger.error("No active AI provider available")
        return None
    
    def get_active_data_provider(self, data_type: str) -> Optional[APIProviderConfig]:
        """Get the active provider for a specific data type (market_data, economic_data)"""
        data_config = self.config.get("data_providers", {}).get(data_type, {})
        active_id = data_config.get("active_provider")
        
        if not active_id:
            return None
        
        provider_key = f"{data_type}_{active_id}"
        provider = self.providers.get(provider_key)
        
        if provider and provider.enabled:
            return provider
        
        # Try fallback providers
        fallback_id = self.config.get("selection_rules", {}).get("data_provider_selection", {}).get(data_type, {}).get("fallback")
        if fallback_id:
            fallback_key = f"{data_type}_{fallback_id}"
            if fallback_key in self.providers:
                fallback_provider = self.providers[fallback_key]
                if fallback_provider.enabled:
                    logger.warning(f"Using fallback data provider: {fallback_provider.name}")
                    return fallback_provider
        
        logger.error(f"No active {data_type} provider available")
        return None
    
    def get_provider_by_id(self, provider_id: str) -> Optional[APIProviderConfig]:
        """Get a specific provider by ID"""
        return self.providers.get(provider_id)
    
    def list_available_providers(self, service_type: Optional[str] = None) -> List[Tuple[str, APIProviderConfig]]:
        """List all available providers, optionally filtered by service type"""
        providers = []
        for provider_id, provider in self.providers.items():
            # Show all configured providers regardless of enabled status (API key availability)
            if service_type is None or provider.service_type == service_type:
                providers.append((provider_id, provider))
        return providers
    
    def check_provider_health(self, provider_id: str) -> bool:
        """Check if a provider is healthy and responding"""
        provider = self.providers.get(provider_id)
        if not provider or not provider.enabled:
            return False
        
        try:
            # Simple health check - attempt to connect to base URL
            response = requests.get(
                provider.base_url, 
                timeout=10,
                headers={"User-Agent": "mlTrainer-HealthCheck/1.0"}
            )
            
            is_healthy = response.status_code < 500
            response_time = response.elapsed.total_seconds() * 1000
            
            # Update provider status
            with self._lock:
                status = self.provider_status[provider_id]
                status.is_healthy = is_healthy
                status.last_check = datetime.now()
                status.response_time_ms = response_time
                
                if is_healthy:
                    status.consecutive_failures = 0
                else:
                    status.consecutive_failures += 1
            
            return is_healthy
            
        except Exception as e:
            logger.warning(f"Health check failed for {provider.name}: {e}")
            
            with self._lock:
                status = self.provider_status[provider_id]
                status.is_healthy = False
                status.last_check = datetime.now()
                status.consecutive_failures += 1
            
            return False
    
    def get_provider_status(self, provider_id: str) -> Optional[ProviderStatus]:
        """Get the current status of a provider"""
        return self.provider_status.get(provider_id)
    
    def switch_ai_provider(self, new_provider_id: str) -> bool:
        """Switch to a different AI provider"""
        provider = self.providers.get(new_provider_id)
        if not provider or not provider.enabled:
            logger.error(f"Cannot switch to provider {new_provider_id}: not available")
            return False
        
        # Update configuration
        self.config["ai_providers"]["active_provider"] = new_provider_id
        
        # Save configuration
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Switched AI provider to: {provider.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration after provider switch: {e}")
            return False
    
    def switch_data_provider(self, data_type: str, new_provider_id: str) -> bool:
        """Switch to a different data provider for a specific data type"""
        provider_key = f"{data_type}_{new_provider_id}"
        provider = self.providers.get(provider_key)
        
        if not provider or not provider.enabled:
            logger.error(f"Cannot switch to data provider {new_provider_id}: not available")
            return False
        
        # Update configuration
        if data_type in self.config.get("data_providers", {}):
            self.config["data_providers"][data_type]["active_provider"] = new_provider_id
            
            # Save configuration
            try:
                with open(self.config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
                logger.info(f"Switched {data_type} provider to: {provider.name}")
                return True
            except Exception as e:
                logger.error(f"Failed to save configuration after provider switch: {e}")
                return False
        
        return False
    
    def get_compliance_verified_providers(self) -> List[Tuple[str, APIProviderConfig]]:
        """Get all providers that are compliance verified"""
        verified_providers = []
        for provider_id, provider in self.providers.items():
            if provider.enabled and provider.compliance.get("verified_source", False):
                verified_providers.append((provider_id, provider))
        return verified_providers
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that all configured API keys are available"""
        validation_results = {}
        
        for provider_id, provider in self.providers.items():
            has_key = provider.api_key is not None and len(provider.api_key.strip()) > 0
            validation_results[provider_id] = has_key
            
            if not has_key and provider.api_key_env:
                logger.warning(f"Missing API key for {provider.name}: {provider.api_key_env}")
        
        return validation_results
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current provider configuration"""
        ai_provider = self.get_active_ai_provider()
        market_provider = self.get_active_data_provider("market_data")
        economic_provider = self.get_active_data_provider("economic_data")
        
        return {
            "active_providers": {
                "ai": ai_provider.name if ai_provider else "None",
                "market_data": market_provider.name if market_provider else "None",
                "economic_data": economic_provider.name if economic_provider else "None"
            },
            "total_providers": len(self.providers),
            "enabled_providers": len([p for p in self.providers.values() if p.enabled]),
            "api_key_validation": self.validate_api_keys(),
            "compliance_verified": len(self.get_compliance_verified_providers())
        }

# Global instance for system-wide access
_api_manager = None

def get_api_manager() -> APIProviderManager:
    """Get the global API provider manager instance"""
    global _api_manager
    if _api_manager is None:
        _api_manager = APIProviderManager()
    return _api_manager

def get_active_ai_provider() -> Optional[APIProviderConfig]:
    """Convenience function to get active AI provider"""
    return get_api_manager().get_active_ai_provider()

def get_active_data_provider(data_type: str) -> Optional[APIProviderConfig]:
    """Convenience function to get active data provider"""
    return get_api_manager().get_active_data_provider(data_type)

def validate_all_api_keys() -> Dict[str, bool]:
    """Convenience function to validate all API keys"""
    return get_api_manager().validate_api_keys()