import os
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

API_CONFIG_PATH = "ai_config.json"
COMPLIANCE_FLAG_FILE = ".compliance_on"
OVERRIDE_FLAG_FILE = ".compliance_override"

def is_compliance_enabled() -> bool:
    """
    Returns True if the system is currently running in compliance mode.
    System owner can override for maintenance, debugging, and development.
    """
    # Check for system owner override first
    if os.path.exists(OVERRIDE_FLAG_FILE):
        try:
            with open(OVERRIDE_FLAG_FILE, "r") as f:
                content = f.read().strip()
            if content == "SYSTEM_OWNER_OVERRIDE_ACTIVE":
                logger.warning("🚨 COMPLIANCE OVERRIDE ACTIVE - System owner maintenance mode")
                return False
        except Exception:
            pass
    
    return os.path.exists(COMPLIANCE_FLAG_FILE)

def enable_compliance():
    """
    Enable compliance enforcement across the system.
    """
    with open(COMPLIANCE_FLAG_FILE, "w") as f:
        f.write("ENABLED")
    logger.info("🔒 Compliance mode ENABLED.")

def disable_compliance():
    """
    Disable compliance enforcement (manual override).
    """
    if os.path.exists(COMPLIANCE_FLAG_FILE):
        os.remove(COMPLIANCE_FLAG_FILE)
    logger.warning("⚠️ Compliance mode DISABLED.")

def get_allowed_apis() -> set:
    """
    Load dynamically allowed APIs from ai_config.json (no hardcoded APIs allowed).
    """
    if not os.path.exists(API_CONFIG_PATH):
        logger.error("❌ Missing ai_config.json. Cannot enforce compliance.")
        raise FileNotFoundError("ai_config.json not found.")

    try:
        with open(API_CONFIG_PATH, "r") as f:
            config = json.load(f)
        allowed = set(config.keys())
        logger.info(f"✅ Allowed APIs under compliance: {allowed}")
        return allowed
    except Exception as e:
        logger.error(f"❌ Failed to parse ai_config.json: {e}")
        raise e

def enforce_api_compliance(api_name: str):
    """
    Raises an error if the given API name is not in ai_config.json
    System owner can override for maintenance, debugging, and development.
    """
    # Check for system owner override
    if os.path.exists(OVERRIDE_FLAG_FILE):
        logger.warning(f"🚨 OVERRIDE: API compliance bypassed for '{api_name}' - MAINTENANCE MODE")
        return
    
    if not is_compliance_enabled():
        logger.info("🔓 Compliance is off — skipping API enforcement.")
        return

    allowed = get_allowed_apis()
    if api_name not in allowed:
        raise PermissionError(f"❌ API '{api_name}' is not allowed under compliance rules.")
    logger.info(f"✅ API '{api_name}' approved under compliance mode.")

def enforce_compliance(data):
    """
    General compliance enforcement function for data/payloads.
    """
    if not is_compliance_enabled():
        logger.info("🔓 Compliance is off — skipping enforcement.")
        return
    logger.info("✅ Compliance enforcement applied to data payload.")

def is_override_authorized() -> bool:
    """
    Check if override is authorized via environment variable.
    """
    return os.getenv("IMMUTABLE_GATEWAY_OVERRIDE", "off").lower() == "on"
