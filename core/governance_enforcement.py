"""
Runtime Governance Enforcement
Enforces governance rules at runtime with comprehensive monitoring
"""

import functools
import inspect
import logging
import sys
import threading
import warnings
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional

from core.audit_log import audit_action
from core.crypto_signing import get_secure_approval
from config.immutable_compliance_gateway import ComplianceGateway

logger = logging.getLogger(__name__)


class GovernanceWarning(Warning):
    """Warning category for governance violations"""
    pass


class PermissionContext:
    """Context for temporarily approved actions"""
    
    def __init__(self, approved_actions: List[str]):
        self.approved_actions = approved_actions
        self.original_enforcement = None
    
    def __enter__(self):
        # Store original enforcement state
        self.original_enforcement = get_runtime_enforcer().enforcement_active
        # Temporarily disable enforcement for approved actions
        get_runtime_enforcer().enforcement_active = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original enforcement state
        if self.original_enforcement is not None:
            get_runtime_enforcer().enforcement_active = self.original_enforcement


class RuntimeGovernanceEnforcer:
    """
    Runtime governance enforcement with comprehensive monitoring
    """

    def __init__(self):
        self.enforcement_active = False
        self.audit_log = None
        self.compliance_gateway = ComplianceGateway()
        self.secure_approval = get_secure_approval()
        self.governance_rules = None

        # Monitoring state
        self._monitored_modules = set()
        self._violation_count = 0
        self._enforcement_stats = {
            "approved_actions": 0,
            "blocked_actions": 0,
            "violations_detected": 0,
        }

        logger.info("Runtime Governance Enforcer initialized")

    def activate_enforcement(self):
        """Activate runtime governance enforcement"""
        try:
            # Import audit log
            from core.audit_log import get_audit_log
            self.audit_log = get_audit_log()

            # Import governance rules
            from config.governance_rules import GovernanceRules
            self.governance_rules = GovernanceRules()

            # Set up module monitoring
            self._setup_module_monitoring()

            # Set up warning filters
            self._setup_warning_filters()

            # Start monitoring thread
            self._start_monitoring_thread()

            # Activate enforcement
            self.enforcement_active = True

            # Audit activation
            audit_action("governance_enforcement_activated", "system")

            logger.info("Runtime governance enforcement activated")

        except Exception as e:
            logger.error(f"Failed to activate governance enforcement: {e}")
            raise

    def _setup_module_monitoring(self):
        """Set up monitoring for critical modules"""
        # Monitor built-in import function
        original_import = __builtins__["__import__"]

        @functools.wraps(original_import)
        def governed_import(name, *args, **kwargs):
            # Import the module
            module = original_import(name, *args, **kwargs)

            # Check if we should monitor this module
            if self._should_monitor_module(name):
                self._inject_governance_into_module(module)

            return module

        # Replace the import function
        __builtins__["__import__"] = governed_import

    def _should_monitor_module(self, module_name: str) -> bool:
        """Determine if a module should be monitored"""
        # Monitor critical modules
        critical_modules = {
            "os", "sys", "subprocess", "builtins",
            "importlib", "types", "inspect"
        }

        # Monitor custom modules
        custom_modules = {
            "mltrainer_models", "mltrainer_financial_models",
            "polygon_connector", "fred_connector"
        }

        return module_name in critical_modules or module_name in custom_modules

    def _inject_governance_into_module(self, module):
        """Inject governance checks into a module"""
        if module in self._monitored_modules:
            return

        self._monitored_modules.add(module)

        # Wrap functions and classes
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith("_"):
                self._wrap_function_with_governance(obj)
            elif inspect.isclass(obj):
                self._wrap_class_with_governance(obj)

    def _wrap_function_with_governance(self, func: Callable) -> Callable:
        """Wrap a function with governance checks"""

        @functools.wraps(func)
        def governed_wrapper(*args, **kwargs):
            # Check permissions
            action_details = {
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_count": len(kwargs),
            }

            # Validate action
            if not self._validate_action("function_call", action_details):
                self._enforcement_stats["blocked_actions"] += 1
                raise PermissionError(f"Governance blocked call to {func.__name__}")

            # Audit the call
            audit_action("function_called", "runtime_enforcer", function=func.__name__, module=func.__module__)

            # Execute function
            result = func(*args, **kwargs)

            self._enforcement_stats["approved_actions"] += 1

            return result

        governed_wrapper._governed = True
        return governed_wrapper

    def _wrap_class_with_governance(self, cls):
        """Wrap class methods with governance"""

        for name, method in inspect.getmembers(cls, inspect.ismethod):
            if not name.startswith("_") and not hasattr(method, "_governed"):
                wrapped = self._wrap_function_with_governance(method)
                setattr(cls, name, wrapped)

    def _validate_action(self, action_type: str, details: Dict[str, Any]) -> bool:
        """Validate an action against governance rules"""

        # Check with governance rules
        valid, reason = self.governance_rules.validate_action(action_type, details)

        if not valid:
            # Log violation
            self._violation_count += 1
            self._enforcement_stats["violations_detected"] += 1

            audit_action(
                "governance_violation", "runtime_enforcer", action_type=action_type, reason=reason, details=details
            )

            # Emit warning
            warnings.warn(f"Governance violation: {reason}", GovernanceWarning)

        return valid

    def _setup_warning_filters(self):
        """Set up warning filters for governance violations"""

        # Always show governance warnings
        warnings.filterwarnings("always", category=GovernanceWarning)

        # Convert synthetic data warnings to errors
        warnings.filterwarnings("error", message=".*synthetic data.*")

    def _start_monitoring_thread(self):
        """Start background monitoring thread"""

        def monitor():
            while self.enforcement_active:
                # Periodic integrity checks
                self._check_system_integrity()

                # Sleep for monitoring interval
                threading.Event().wait(60)  # Check every minute

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def _check_system_integrity(self):
        """Perform periodic system integrity checks"""

        # Verify audit log integrity
        is_valid, issues = self.audit_log.verify_integrity()

        if not is_valid:
            audit_action("integrity_check_failed", "runtime_enforcer", issues=issues)

        # Check for bypass attempts
        self._detect_bypass_attempts()

    def _detect_bypass_attempts(self):
        """Detect attempts to bypass governance"""

        # Check if built-ins have been modified
        suspicious_modules = []

        for module_name in ["builtins", "sys", "os"]:
            module = sys.modules.get(module_name)
            if module:
                # Check for unexpected modifications
                for attr in ["open", "exec", "eval", "importlib.import_module"]:
                    if hasattr(module, attr):
                        func = getattr(module, attr)
                        if not hasattr(func, "_governed") and module_name == "builtins":
                            suspicious_modules.append(f"{module_name}.{attr}")

        if suspicious_modules:
            audit_action("bypass_attempt_detected", "runtime_enforcer", suspicious_functions=suspicious_modules)

    def get_enforcement_report(self) -> Dict[str, Any]:
        """Get enforcement statistics and status"""

        return {
            "enforcement_active": self.enforcement_active,
            "monitored_modules": list(self._monitored_modules),
            "violation_count": self._violation_count,
            "statistics": self._enforcement_stats,
            "audit_log_stats": self.audit_log.get_statistics(),
            "last_check": datetime.now().isoformat(),
        }

    def request_emergency_override(self, reason: str, requested_by: str) -> Optional[str]:
        """Request emergency override for governance rules"""

        # This should be heavily restricted and audited
        override_request = {
            "type": "emergency_override",
            "reason": reason,
            "requested_by": requested_by,
            "timestamp": datetime.now().isoformat(),
        }

        # Request approval
        request_id = self.secure_approval.request_approval(override_request, requested_by)

        # Audit the request
        audit_action("emergency_override_requested", requested_by, request_id=request_id, reason=reason)

        return request_id


class GovernanceContextManager:
    """Context manager for temporarily approved actions"""

    def __init__(self, approved_actions: List[str], approver: str):
        self.approved_actions = approved_actions
        self.approver = approver
        self.context = None
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()

        # Audit the context entry
        audit_action("governance_context_entered", self.approver, approved_actions=self.approved_actions)

        # Create permission context
        self.context = PermissionContext(self.approved_actions)
        return self.context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration
        duration = (datetime.now() - self.start_time).total_seconds()

        # Audit the context exit
        audit_action(
            "governance_context_exited",
            self.approver,
            duration_seconds=duration,
            exception=str(exc_val) if exc_val else None,
        )

        # Exit permission context
        if self.context:
            return self.context.__exit__(exc_type, exc_val, exc_tb)


# Global enforcer instance
_enforcer = None


def get_runtime_enforcer() -> RuntimeGovernanceEnforcer:
    """Get global runtime enforcer instance"""
    global _enforcer
    if _enforcer is None:
        _enforcer = RuntimeGovernanceEnforcer()
    return _enforcer


# Convenience functions
def activate_runtime_enforcement():
    """Activate runtime governance enforcement"""
    enforcer = get_runtime_enforcer()
    enforcer.activate_enforcement()


def get_enforcement_report() -> Dict[str, Any]:
    """Get enforcement report"""
    return get_runtime_enforcer().get_enforcement_report()


def with_approved_actions(actions: List[str], approver: str):
    """Context manager for approved actions"""
    return GovernanceContextManager(actions, approver)
