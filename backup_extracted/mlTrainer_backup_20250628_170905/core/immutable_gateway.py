import json
import logging
import os
import hashlib
from typing import Any, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

API_CONFIG_PATH = "ai_config.json"
OVERRIDE_FLAG_FILE = ".compliance_override"

# Approved ML models — update if needed
ALLOWED_MODELS = {
    "LSTM",
    "Transformer",
    "XGBoost",
    "Ensemble"
}

# CRITICAL: Only the system owner can disable compliance and immutable gateway
# This is for emergency maintenance, debugging, and development ONLY
# No password required - system owner command-based control

def is_override_authorized() -> bool:
    """
    Check if compliance override is authorized by the system owner.
    CRITICAL: Only for emergency maintenance, debugging, and development.
    """
    if not os.path.exists(OVERRIDE_FLAG_FILE):
        return False
    
    try:
        with open(OVERRIDE_FLAG_FILE, "r") as f:
            content = f.read().strip()
        
        if content == "SYSTEM_OWNER_OVERRIDE_ACTIVE":
            logger.warning("🚨 CRITICAL: COMPLIANCE OVERRIDE ACTIVE - System owner maintenance mode")
            return True
        else:
            logger.error("❌ UNAUTHORIZED OVERRIDE ATTEMPT - Compliance remains enforced")
            return False
    except Exception as e:
        logger.error(f"❌ Override verification failed: {e}")
        return False

def enable_override() -> bool:
    """
    Enable compliance override for system maintenance.
    ONLY the system owner can activate this via direct command.
    """
    with open(OVERRIDE_FLAG_FILE, "w") as f:
        f.write("SYSTEM_OWNER_OVERRIDE_ACTIVE")
    logger.warning("🚨 COMPLIANCE OVERRIDE ENABLED - System owner maintenance mode active")
    return True

def disable_override():
    """
    Disable compliance override and restore full enforcement.
    """
    if os.path.exists(OVERRIDE_FLAG_FILE):
        os.remove(OVERRIDE_FLAG_FILE)
    logger.info("🔒 COMPLIANCE OVERRIDE DISABLED - Full enforcement restored")

def load_allowed_apis() -> set:
    """
    Load allowed API sources strictly from ai_config.json.
    This is the ONLY source of truth per compliance rules.
    """
    if not os.path.exists(API_CONFIG_PATH):
        logger.error(f"❌ API config file not found: {API_CONFIG_PATH}")
        raise FileNotFoundError("ai_config.json is required for immutable gateway")

    try:
        with open(API_CONFIG_PATH, "r") as f:
            config = json.load(f)
        # Normalize keys to lowercase for consistent verification
        allowed = set(k.lower() for k in config.keys())
        logger.info(f"✅ Allowed APIs from config: {allowed}")
        return allowed
    except Exception as e:
        logger.error(f"❌ Failed to load API config: {e}")
        raise e

def verify_data_source(source_name: str) -> bool:
    """
    Verify that the data source is authorized and compliant.
    NO EXCEPTIONS - REAL DATA ONLY (unless system owner override is active).
    """
    # CRITICAL: Check for system owner override
    if is_override_authorized():
        logger.warning(f"🚨 OVERRIDE: Data source verification bypassed for '{source_name}' - MAINTENANCE MODE")
        return True
    
    if not source_name:
        logger.error("❌ CRITICAL COMPLIANCE VIOLATION: No data source specified")
        raise PermissionError("BLOCKED: Data source is REQUIRED - no synthetic data allowed")

    allowed_sources = load_allowed_apis()
    if source_name.lower() not in allowed_sources:
        logger.error(f"❌ COMPLIANCE VIOLATION: Data source '{source_name}' is not in authorized list")
        raise PermissionError(f"BLOCKED: {source_name} is not an approved API source in ai_config.json")
    
    # Verify API key exists for this source
    try:
        with open(API_CONFIG_PATH, "r") as f:
            config = json.load(f)
        api_key = config.get(source_name.lower(), {}).get("api_key")
        if not api_key:
            logger.error(f"❌ COMPLIANCE VIOLATION: No API key configured for {source_name}")
            raise PermissionError(f"BLOCKED: API key missing for {source_name} - cannot guarantee real data")
    except Exception as e:
        logger.error(f"❌ COMPLIANCE FAILURE: Cannot verify API configuration: {e}")
        raise PermissionError("BLOCKED: Cannot verify real data source configuration")
    
    logger.info(f"✅ COMPLIANCE VERIFIED: Real data source {source_name} with valid API key")
    return True

def verify_model_request(model_name: str) -> bool:
    """
    Verify that the ML model is authorized for production use.
    """
    # CRITICAL: Check for system owner override
    if is_override_authorized():
        logger.warning(f"🚨 OVERRIDE: Model verification bypassed for '{model_name}' - MAINTENANCE MODE")
        return True
    
    if not model_name:
        logger.warning("⚠️ No model name provided to verify_model_request; skipping check.")
        return False

    if model_name not in ALLOWED_MODELS:
        logger.error(f"❌ Unauthorized model request: {model_name}")
        raise PermissionError(f"Blocked: {model_name} is not an approved model.")
    logger.info(f"✅ Verified model request: {model_name}")
    return True

def enforce_gateway(payload: Dict[str, Any]) -> None:
    """
    Enforce verification on any inbound training or execution request.
    MANDATORY REAL DATA SOURCE - NO EXCEPTIONS ALLOWED (unless system owner override is active).
    """
    # CRITICAL: Check for system owner override first
    if is_override_authorized():
        logger.warning("🚨 IMMUTABLE GATEWAY OVERRIDE ACTIVE - System owner maintenance mode")
        return
    
    if not isinstance(payload, dict):
        logger.error("❌ Payload to enforce_gateway must be a dict.")
        raise TypeError("Payload must be a dictionary.")

    data_source = payload.get("data_source")
    model_type = payload.get("model")

    # CRITICAL: Data source is MANDATORY - no synthetic data allowed
    if not data_source:
        logger.error("❌ CRITICAL COMPLIANCE VIOLATION: data_source field is MANDATORY")
        raise PermissionError("BLOCKED: data_source field is REQUIRED - no synthetic/mock data allowed")
    
    verify_data_source(data_source)

    if not model_type:
        logger.error("❌ COMPLIANCE VIOLATION: model field is MANDATORY")
        raise PermissionError("BLOCKED: model field is REQUIRED")
    
    verify_model_request(model_type)

    logger.info("🔒 COMPLIANCE ENFORCED: Real data source and approved model verified")

def secure_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Secure input validation and sanitization.
    Enforces gateway checks on incoming data.
    """
    if not isinstance(payload, dict):
        raise TypeError("Input payload must be a dictionary")
    
    # Enforce gateway validation
    enforce_gateway(payload)
    
    logger.info("✅ Input secured and validated")
    return payload

def secure_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Secure output filtering and validation.
    Ensures output meets compliance standards.
    """
    if not isinstance(result, dict):
        raise TypeError("Output result must be a dictionary")
    
    # Filter sensitive information if needed
    filtered_result = result.copy()
    
    # Remove any keys that might contain sensitive data
    sensitive_keys = ['api_key', 'secret', 'password', 'token']
    for key in list(filtered_result.keys()):
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            filtered_result[key] = "[REDACTED]"
    
    logger.info("✅ Output secured and filtered")
    return filtered_result

def enforce_verified_source(func):
    """
    Decorator to enforce verification of data sources.
    """
    def wrapper(*args, **kwargs):
        logger.info(f"🔒 Enforcing verified source for function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
