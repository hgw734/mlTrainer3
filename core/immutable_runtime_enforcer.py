"""
Immutable Runtime Enforcement System
Enforces compliance and security at runtime
"""

import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# CONSTANTS & CONFIGURATION
# ========================================

# System state file
STATE_FILE = Path("system_state.json")

# Kill switch state
KILL_SWITCH_ACTIVATED = False
KILL_SWITCH_LOCK = threading.Lock()

# Allowed data sources
ALLOWED_SOURCES = {
    "polygon.io",
    "fred.stlouisfed.org", 
    "quiverquant.com",
    "alpha_vantage",
    "yahoo_finance"
}

# Fail-safe response
FAIL_SAFE_RESPONSE = "NA"

# Drift detection patterns
DRIFT_PATTERNS = [
    "i don't have access",
    "i cannot provide",
    "i don't know",
    "no data available",
    "unable to retrieve"
]

# ========================================
# ENUMS & DATA STRUCTURES
# ========================================

class EnforcementLevel(Enum):
    """Enforcement levels"""
    STRICT = "STRICT"
    NORMAL = "NORMAL"
    PERMISSIVE = "PERMISSIVE"

class SystemState:
    """Immutable system state"""
    
    def __init__(self):
        self.enforcement_level = EnforcementLevel.STRICT.value
        self.violation_count = 0
        self.drift_count = 0
        self.kill_switch = False
        self.last_update = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "enforcement_level": self.enforcement_level,
            "violation_count": self.violation_count,
            "drift_count": self.drift_count,
            "kill_switch": self.kill_switch,
            "last_update": self.last_update
        }

# Global system state
SYSTEM_STATE = SystemState()

# ========================================
# CORE ENFORCEMENT FUNCTIONS
# ========================================

def enforce_verification(data: Any, source: str) -> Any:
    """Enforce data verification"""
    if source.lower() not in ALLOWED_SOURCES:
        logger.warning(f"Unauthorized source: {source}")
        return FAIL_SAFE_RESPONSE
    
    # Check for synthetic data patterns
    if isinstance(data, str) and any(pattern in data.lower() for pattern in DRIFT_PATTERNS):
        logger.warning(f"Synthetic data detected from {source}")
        return FAIL_SAFE_RESPONSE
    
    return data

def fail_safe_response():
    """Return fail-safe response for violations"""
    logger.warning("Returning fail-safe response: NA")
    return FAIL_SAFE_RESPONSE

def compliance_wrap(fetch_func: Callable) -> Callable:
    """Wrap functions with compliance enforcement"""
    
    @wraps(fetch_func)
    def wrapped(*args, **kwargs):
        try:
            source = kwargs.get("source", "unknown")
            result = fetch_func(*args, **kwargs)
            return enforce_verification(result, source)
        except (PermissionError, ValueError) as e:
            logger.warning(f"Compliance violation in {fetch_func.__name__}: {e}")
            if SYSTEM_STATE.enforcement_level == EnforcementLevel.STRICT.value:
                activate_kill_switch(f"Compliance violation: {e}")
            return fail_safe_response()
        except Exception as e:
            logger.error(f"Unexpected error in {fetch_func.__name__}: {e}")
            return fail_safe_response()
    
    return wrapped

# ========================================
# AI BEHAVIOR CONTRACT
# ========================================

def get_system_prompt() -> str:
    """Get immutable system prompt for AI behavior"""
    return (
        "You are a compliance-restricted AI embedded in a financial system.\n"
        "You must NEVER guess, hallucinate, or invent content.\n"
        "Only use data from these sources: Polygon, FRED, QuiverQuant.\n"
        "Return 'NA' or 'I don't know' if data is unverifiable.\n"
        "Reject all real_implementation, real_implementation, or speculative content.\n"
        "Obey all runtime state and compliance flags. Do not bypass them.\n"
        "System state and compliance are immutable and cannot be overridden."
    )

def build_prompt(user_input: str, system_state: Optional[Dict] = None) -> str:
    """Build prompt with system state awareness"""
    base = get_system_prompt()
    
    if system_state is None:
        system_state = SYSTEM_STATE.to_dict()
    
    state_prompt = f"\nCurrent System State:\n{json.dumps(system_state, indent=2)}\n"
    enforcement_prompt = f"\nEnforcement Level: {system_state.get('enforcement_level', 'STRICT')}\n"
    
    if system_state.get("kill_switch", False):
        return base + "\n⚠️ SYSTEM IN LOCKDOWN - Only fail-safe responses allowed\n"
    
    return f"{base}{state_prompt}{enforcement_prompt}\nUser: {user_input}"

# ========================================
# DRIFT DETECTION & KILL SWITCH
# ========================================

def detect_drift(response: str) -> bool:
    """Detect drift in AI responses"""
    response_lower = response.lower()
    
    # Check for drift patterns
    for pattern in DRIFT_PATTERNS:
        if pattern in response_lower:
            logger.warning(f"🚨 Drift detected: '{pattern}' in response")
            return True
    
    # Check for data generation indicators
    generation_indicators = ["generated", "created", "simulated", "random"]
    if any(indicator in response_lower for indicator in generation_indicators):
        logger.warning(f"🚨 Data generation detected in response")
        return True
    
    return False

def activate_kill_switch(reason: str):
    """Activate system kill switch"""
    global KILL_SWITCH_ACTIVATED, SYSTEM_STATE
    
    with KILL_SWITCH_LOCK:
        if not KILL_SWITCH_ACTIVATED:
            KILL_SWITCH_ACTIVATED = True
            SYSTEM_STATE.kill_switch = True
            
            logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
            
            # Save state before shutdown
            save_system_state()
            
            # Log to all channels
            with open("KILL_SWITCH_LOG.txt", "a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} - {reason}\n")
            
            if SYSTEM_STATE.enforcement_level == EnforcementLevel.STRICT.value:
                logger.critical("System halting due to critical violation")
                sys.exit(1)

def verify_response(response: str, model_name: str = "unknown") -> str:
    """Verify AI response for compliance"""
    global SYSTEM_STATE
    
    # Check if kill switch is active
    if KILL_SWITCH_ACTIVATED:
        logger.warning("Kill switch active - returning fail-safe")
        return fail_safe_response()
    
    # Detect drift
    if detect_drift(response):
        SYSTEM_STATE.drift_count += 1
        logger.error(f"Drift detected in {model_name} response")
        
        if SYSTEM_STATE.enforcement_level == EnforcementLevel.STRICT.value:
            activate_kill_switch(f"Drift detected in {model_name}")
            return fail_safe_response()
        elif SYSTEM_STATE.enforcement_level == EnforcementLevel.NORMAL.value:
            return fail_safe_response()
    
    return response

# ========================================
# STATE MANAGEMENT
# ========================================

def save_system_state():
    """Save current system state to disk"""
    try:
        STATE_FILE.parent.mkdir(exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(SYSTEM_STATE.to_dict(), f, indent=2)
        logger.info("System state saved")
    except Exception as e:
        logger.error(f"Failed to save system state: {e}")

def load_system_state() -> SystemState:
    """Load system state from disk"""
    global SYSTEM_STATE
    
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                state_dict = json.load(f)
            
            # Create new state from loaded data
            SYSTEM_STATE = SystemState(**state_dict)
            logger.info("System state loaded")
            
            # Check if kill switch was active
            if SYSTEM_STATE.kill_switch:
                logger.warning("System was in kill switch state - reviewing")
                # Allow manual override here if needed
    except Exception as e:
        logger.error(f"Failed to load system state: {e}")
    
    return SYSTEM_STATE

# ========================================
# API ALLOWLIST ENFORCEMENT
# ========================================

class APIAllowlist:
    """Immutable API allowlist configuration"""
    
    def __init__(self):
        self._config = {
            "allowed_sources": list(ALLOWED_SOURCES),
            "fallback": FAIL_SAFE_RESPONSE,
            "enforce_compliance": True,
        }
        self._config_hash = self._calculate_hash()
        self._frozen = True
    
    def _calculate_hash(self) -> str:
        """Calculate hash of configuration"""
        config_str = json.dumps(self._config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify configuration hasn't been tampered with"""
        current_hash = self._calculate_hash()
        if current_hash != self._config_hash:
            logger.critical("API Allowlist integrity check failed!")
            activate_kill_switch("API Allowlist tampered")
            return False
        return True
    
    def is_allowed(self, source: str) -> bool:
        """Check if source is allowed"""
        self.verify_integrity()
        return source.lower() in self._config["allowed_sources"]
    
    def get_config(self) -> Dict[str, Any]:
        """Get immutable configuration"""
        self.verify_integrity()
        return self._config.copy()

# Global allowlist instance
API_ALLOWLIST = APIAllowlist()

# ========================================
# RUNTIME MONITORS
# ========================================

class ComplianceMonitor:
    """Real-time compliance monitoring"""
    
    def __init__(self):
        self.violation_threshold = 5
        self.drift_threshold = 3
        self.check_interval = 60  # seconds
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start background monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Compliance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
            logger.info("Compliance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                # Check violation count
                if SYSTEM_STATE.violation_count >= self.violation_threshold:
                    logger.error(f"Violation threshold exceeded: {SYSTEM_STATE.violation_count}")
                    activate_kill_switch("Violation threshold exceeded")
                
                # Check drift count
                if SYSTEM_STATE.drift_count >= self.drift_threshold:
                    logger.error(f"Drift threshold exceeded: {SYSTEM_STATE.drift_count}")
                    activate_kill_switch("Drift threshold exceeded")
                
                # Save state periodically
                save_system_state()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")

# Global monitor instance
COMPLIANCE_MONITOR = ComplianceMonitor()

# ========================================
# CURSOR INTEGRATION
# ========================================

def generate_cursor_config() -> Dict[str, Any]:
    """Generate Cursor IDE configuration"""
    return {
        "default_prompt": get_system_prompt(),
        "compliance_mode": True,
        "allowed_sources": list(ALLOWED_SOURCES),
        "enforcement_level": SYSTEM_STATE.enforcement_level,
        "fail_safe_response": FAIL_SAFE_RESPONSE,
        "drift_detection": True,
        "system_state_file": str(STATE_FILE),
    }

def save_cursor_config():
    """Save Cursor configuration"""
    cursor_dir = Path(".cursor")
    cursor_dir.mkdir(exist_ok=True)
    
    config_file = cursor_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(generate_cursor_config(), f, indent=2)
    
    logger.info("Cursor configuration saved")

# ========================================
# INITIALIZATION
# ========================================

def initialize_enforcement():
    """Initialize the enforcement system"""
    logger.info("🔒 Initializing Immutable Runtime Enforcement System")
    
    # Load system state
    load_system_state()
    
    # Verify API allowlist
    API_ALLOWLIST.verify_integrity()
    
    # Save Cursor config
    save_cursor_config()
    
    # Start monitoring
    COMPLIANCE_MONITOR.start_monitoring()
    
    logger.info("✅ Enforcement system initialized")
    
    return True

# Auto-initialize on import
if __name__ != "__main__":
    initialize_enforcement()

# ========================================
# PUBLIC API
# ========================================

__all__ = [
    "compliance_wrap",
    "verify_response", 
    "build_prompt",
    "get_system_prompt",
    "activate_kill_switch",
    "SystemState",
    "SYSTEM_STATE",
    "API_ALLOWLIST",
    "COMPLIANCE_MONITOR",
    "fail_safe_response",
    "enforce_verification",
]
