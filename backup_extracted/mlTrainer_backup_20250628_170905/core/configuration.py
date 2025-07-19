
import json
import os
import logging

logger = logging.getLogger(__name__)

API_CONFIG_PATH = "ai_config.json"

def load_allowed_apis() -> set:
    """
    Load allowed API sources from ai_config.json.
    This is the single source of truth for authorized data sources.
    """
    if not os.path.exists(API_CONFIG_PATH):
        logger.error(f"❌ API config file not found: {API_CONFIG_PATH}")
        raise FileNotFoundError("ai_config.json is required for system operation")

    try:
        with open(API_CONFIG_PATH, "r") as f:
            config = json.load(f)
        # Normalize keys to lowercase for consistent verification
        allowed = set(k.lower() for k in config.keys())
        logger.info(f"✅ Loaded allowed APIs: {allowed}")
        return allowed
    except Exception as e:
        logger.error(f"❌ Failed to load API config: {e}")
        raise e

def get_api_config() -> dict:
    """
    Get the full API configuration.
    """
    if not os.path.exists(API_CONFIG_PATH):
        raise FileNotFoundError("ai_config.json not found")
    
    with open(API_CONFIG_PATH, "r") as f:
        return json.load(f)

def get_api_key(api_name: str) -> str:
    """
    Get API key for a specific service from environment or config.
    """
    config = get_api_config()
    api_config = config.get(api_name.lower(), {})
    api_key = api_config.get("api_key")
    
    if not api_key:
        raise ValueError(f"No API key found for {api_name}")
    
    # Handle ENV: prefix - read from environment variables
    if api_key.startswith("ENV:"):
        env_var_name = api_key[4:]  # Remove "ENV:" prefix
        env_value = os.getenv(env_var_name)
        if not env_value:
            raise ValueError(f"Environment variable '{env_var_name}' not found for {api_name}")
        return env_value
    
    return api_key
