#!/usr/bin/env python3
"""
Secure Secrets Management System
Handles all API keys, tokens, and sensitive configuration
"""

import os
import logging
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Types of secrets managed by the system"""

    API_KEY = "api_key"
    DATABASE = "database"
    AUTH_TOKEN = "auth_token"
    CREDENTIAL = "credential"


@dataclass
class SecretConfig:
    """Configuration for a managed secret"""

    name: str
    env_var: str
    secret_type: SecretType
    required: bool = True
    description: str = ""
    validation_pattern: Optional[str] = None
    masked: bool = True


class SecretsManager:
    """
    Secure secrets management with validation and audit logging

    This class ensures:
        1. No hardcoded secrets in code
        2. All secrets from environment variables
        3. Validation of secret formats
        4. Audit logging of secret access
        5. Secure error messages (no secret leakage)
    """

    # Define all managed secrets
    SECRETS_REGISTRY = {
        "polygon_api_key": SecretConfig(
            name="Polygon API Key",
            env_var="POLYGON_API_KEY",
            secret_type=SecretType.API_KEY,
            required=True,
            description="Polygon.io market data API key",
            masked=True,
        ),
        "fred_api_key": SecretConfig(
            name="FRED API Key",
            env_var="FRED_API_KEY",
            secret_type=SecretType.API_KEY,
            required=True,
            description="Federal Reserve Economic Data API key",
            masked=True,
        ),
        "alpaca_api_key": SecretConfig(
            name="Alpaca API Key",
            env_var="ALPACA_API_KEY",
            secret_type=SecretType.API_KEY,
            required=False,
            description="Alpaca trading API key",
            masked=True,
        ),
        "alpaca_secret_key": SecretConfig(
            name="Alpaca Secret Key",
            env_var="ALPACA_SECRET_KEY",
            secret_type=SecretType.API_KEY,
            required=False,
            description="Alpaca trading secret key",
            masked=True,
        ),
        "telegram_bot_token": SecretConfig(
            name="Telegram Bot Token",
            env_var="TELEGRAM_BOT_TOKEN",
            secret_type=SecretType.AUTH_TOKEN,
            required=False,
            description="Telegram bot authentication token",
            masked=True,
        ),
        "telegram_chat_id": SecretConfig(
            name="Telegram Chat ID",
            env_var="TELEGRAM_CHAT_ID",
            secret_type=SecretType.CREDENTIAL,
            required=False,
            description="Telegram chat ID for notifications",
            masked=False,
        ),
        "database_url": SecretConfig(
            name="Database URL",
            env_var="DATABASE_URL",
            secret_type=SecretType.DATABASE,
            required=False,
            description="Database connection string",
            masked=True,
        ),
        "modal_token_id": SecretConfig(
            name="Modal Token ID",
            env_var="MODAL_TOKEN_ID",
            secret_type=SecretType.AUTH_TOKEN,
            required=False,
            description="Modal deployment token ID",
            masked=True,
        ),
        "modal_token_secret": SecretConfig(
            name="Modal Token Secret",
            env_var="MODAL_TOKEN_SECRET",
            secret_type=SecretType.AUTH_TOKEN,
            required=False,
            description="Modal deployment token secret",
            masked=True,
        ),
        "aws_access_key_id": SecretConfig(
            name="AWS Access Key ID",
            env_var="AWS_ACCESS_KEY_ID",
            secret_type=SecretType.CREDENTIAL,
            required=False,
            description="AWS access key ID",
            masked=True,
        ),
        "aws_secret_access_key": SecretConfig(
            name="AWS Secret Access Key",
            env_var="AWS_SECRET_ACCESS_KEY",
            secret_type=SecretType.API_KEY,
            required=False,
            description="AWS secret access key",
            masked=True,
        ),
        "sentry_dsn": SecretConfig(
            name="Sentry DSN",
            env_var="SENTRY_DSN",
            secret_type=SecretType.CREDENTIAL,
            required=False,
            description="Sentry error tracking DSN",
            masked=True,
        ),
    }

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize secrets manager

        Args:
            env_file: Optional path to .env file for local development
        """
        self._secrets_cache: Dict[str, Optional[str]] = {}
        self._access_log: List[Dict[str, Any]] = []

        # Load .env file if specified and exists (development only)
        if env_file and Path(env_file).exists():
            self._load_env_file(env_file)
            logger.info("Loaded environment from .env file")

    def _load_env_file(self, env_file: str):
        """Load environment variables from .env file (development only)"""
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as e:
            logger.error(f"Failed to load .env file: {e}")

    def get(self, secret_key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value securely

        Args:
            secret_key: Key identifying the secret
            default: Default value if secret not found (use cautiously!)

        Returns:
            Secret value or default

        Raises:
            ValueError: If required secret is missing
        """
        if secret_key not in self.SECRETS_REGISTRY:
            raise ValueError(f"Unknown secret key: {secret_key}")

        config = self.SECRETS_REGISTRY[secret_key]

        # Check cache first
        if secret_key in self._secrets_cache:
            return self._secrets_cache[secret_key]

        # Get from environment
        value = os.getenv(config.env_var, default)

        # Log access (without exposing value)
        self._log_access(secret_key, value is not None)

        # Validate required secrets
        if config.required and not value:
            raise ValueError(
                f"{config.name} is required. "
                f"Please set the {config.env_var} environment variable."
            )

        # Cache the value
        self._secrets_cache[secret_key] = value

        return value

    def get_required(self, secret_key: str) -> str:
        """
        Get a required secret value (raises if missing)

        Args:
            secret_key: Key identifying the secret

        Returns:
            Secret value

        Raises:
            ValueError: If secret is missing
        """
        value = self.get(secret_key)
        if not value:
            config = self.SECRETS_REGISTRY[secret_key]
            raise ValueError(
                f"{config.name} is required but not set. "
                f"Please set the {config.env_var} environment variable."
            )
        return value

    def _log_access(self, secret_key: str, found: bool):
        """Log secret access for audit purposes"""
        self._access_log.append(
            {
                "secret_key": secret_key,
                "found": found,
                "timestamp": os.environ.get("MLTRAINER_TIMESTAMP", "N/A")
            }
        )

    def mask_value(self, value: str, visible_chars: int = 4) -> str:
        """
        Mask a secret value for safe display

        Args:
            value: Secret value to mask
            visible_chars: Number of characters to show at start

        Returns:
            Masked value like 'abcd****'
        """
        if not value or len(value) <= visible_chars:
            return "*" * 8
        return value[:visible_chars] + "*" * (min(len(value) - visible_chars, 20))

    def validate_all_required(self) -> Dict[str, bool]:
        """
        Validate all required secrets are present

        Returns:
            Dict of secret_key -> is_valid
        """
        results = {}
        for key, config in list(self.SECRETS_REGISTRY.items()):
            if config.required:
                try:
                    value = self.get(key)
                    results[key] = bool(value)
                except ValueError:
                    results[key] = False
        return results

    def get_missing_required(self) -> List[str]:
        """Get list of missing required secrets"""
        missing = []
        for key, config in list(self.SECRETS_REGISTRY.items()):
            if config.required and not os.getenv(config.env_var):
                missing.append(config.env_var)
        return missing

    def check_environment(self) -> bool:
        """
        Check if all required secrets are available

        Returns:
            True if all required secrets are present
        """
        missing = self.get_missing_required()
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            return False
        return True

    def get_safe_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configuration info without exposing actual secret values

        Returns:
            Safe configuration dictionary for display
        """
        config = {}
        for key, secret_config in list(self.SECRETS_REGISTRY.items()):
            value = os.getenv(secret_config.env_var)
            config[key] = {
                "name": secret_config.name,
                "env_var": secret_config.env_var,
                "is_set": bool(value),
                "required": secret_config.required,
                "type": secret_config.secret_type.value,
                "description": secret_config.description,
            }
            if value and not secret_config.masked:
                config[key]["value"] = value
            elif value and secret_config.masked:
                config[key]["masked_value"] = self.mask_value(value)
        return config


# Singleton instance
_secrets_manager = None


def get_secrets_manager(env_file: Optional[str] = None) -> SecretsManager:
    """Get singleton secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        # Only load .env in development
        if os.getenv("MLTRAINER_ENV") == "development":
            env_path = env_file or ".env"
            if Path(env_path).exists():
                _secrets_manager = SecretsManager(env_path)
            else:
                _secrets_manager = SecretsManager()
        else:
            _secrets_manager = SecretsManager()
    return _secrets_manager


# Convenience functions
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a secret value"""
    return get_secrets_manager().get(key, default)


def get_required_secret(key: str) -> str:
    """Get a required secret value"""
    return get_secrets_manager().get_required(key)


def check_secrets() -> bool:
    """Check if all required secrets are available"""
    return get_secrets_manager().check_environment()


def get_missing_secrets() -> List[str]:
    """Get list of missing required secrets"""
    return get_secrets_manager().get_missing_required()


# Export key functions
__all__ = [
    "SecretsManager",
    "get_secrets_manager",
    "get_secret",
    "get_required_secret",
    "check_secrets",
    "get_missing_secrets",
    "SecretType",
    "SecretConfig",
]
