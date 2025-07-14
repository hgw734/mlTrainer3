# Core module initialization
# Immutable compliance components are loaded here

# Import immutable components
try:
    from .immutable_rules_kernel import IMMUTABLE_RULES
    from .runtime_enforcement_hooks import ENFORCEMENT_HOOKS
    from .mandatory_execution_validator import EXECUTION_VALIDATOR
    from .consequence_enforcement_system import CONSEQUENCE_ENFORCER
    
    # Verify system integrity on import
    if not IMMUTABLE_RULES.verify_integrity():
        raise RuntimeError("CRITICAL: Immutable rules integrity check failed")
    
    # Export components
    __all__ = [
        'IMMUTABLE_RULES',
        'ENFORCEMENT_HOOKS', 
        'EXECUTION_VALIDATOR',
        'CONSEQUENCE_ENFORCER'
    ]
    
    # Mark as initialized
    IMMUTABLE_COMPLIANCE_ACTIVE = True
    
    # Install enforcement hooks after all modules are loaded
    try:
        ENFORCEMENT_HOOKS.install_hooks()
    except Exception as e:
        print(f"⚠️  Warning: Failed to install enforcement hooks: {e}")
    
except ImportError as e:
    # If components aren't available, system should not start
    print(f"⚠️  Warning: Immutable compliance components not fully loaded: {e}")
    IMMUTABLE_COMPLIANCE_ACTIVE = False
    __all__ = []

# Import existing components
from .governance_kernel import *
from .immutable_runtime_enforcer import *
from .production_efficiency_manager import *
from .trial_feedback_manager import *
from .unified_executor import *
from .async_execution_engine import *
from .audit_log import *
from .autonomous_loop import *
from .compliance_mode import *
from .crypto_signing import *
from .dynamic_executor import *
from .enhanced_background_manager import *
from .enhanced_memory import *
from .governance_enforcement import *