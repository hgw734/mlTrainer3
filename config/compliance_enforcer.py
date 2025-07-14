#!/usr/bin/env python3
import importlib
import logging

logger = logging.getLogger(__name__)


"""
🔒 COMPLIANCE ENFORCER
mlTrainer Institutional Data Purity Enforcement

INTERCEPTS ALL OPERATIONS - ZERO TOLERANCE FOR NON-COMPLIANT DATA
"""

import sys
import inspect
import functools
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timezone

from .immutable_compliance_gateway import COMPLIANCE_GATEWAY, DataProvenance, VerifiedData, ComplianceStatus, DataSource
from .api_config import validate_api_source, get_approved_endpoint, APISource


class ComplianceEnforcer:
    """
    🔒 COMPLIANCE ENFORCER

    INTERCEPTS ALL OPERATIONS TO ENSURE:
        1. No synthetic data creation
        2. All data passes through compliance gateway
        3. Provenance tracking maintained
        4. Audit trail preserved
    """

    FORBIDDEN_OPERATIONS = {
        "random",
        "randint",
        "randn",
        "rand",
        "choice",
        "shuffle",
        "sample",
        "uniform",
        "normal",
        "gaussian",
        "exponential",
        "real_implementation",
        "synthetic",
        "production_implementation",
        "actual_implementation",
        "pending_implementation",
        "generate",
        "create_fake",
        "simulate",
        "artificial",
    }

    FORBIDDEN_MODULES = {
        "faker",
        "factory_boy",
        "hypothesis",
        "random",
        "numpy.random",
        "scipy.stats",
        "sklearn.datasets.make_",
    }

    def __init__(self):
        self.intercepted_calls = []
        self.blocked_operations = []
        self.active = True

    def is_synthetic_operation(self, func_name: str, module_name: str = "") -> bool:
        """Check if operation involves synthetic data generation"""
        full_name = f"{module_name}.{func_name}" if module_name else func_name

        # Check forbidden operations
        if any(forbidden in func_name.lower() for forbidden in self.FORBIDDEN_OPERATIONS):
            return True

        # Check forbidden modules
        if any(forbidden in full_name.lower() for forbidden in self.FORBIDDEN_MODULES):
            return True

        return False

    def validate_data_source(self, data: Any, context: str = "") -> bool:
        """Validate data comes from approved sources"""
        # Check if data has provenance
        if hasattr(data, "provenance") and isinstance(data.provenance, DataProvenance):
            return data.provenance.compliance_status == ComplianceStatus.VERIFIED

        # Check if data is from approved API call
        if hasattr(data, "__dict__") and "source" in data.__dict__:
            return validate_api_source(data.source)

        # If no provenance, assume non-compliant
        return False

    def intercept_function_call(self, func: Callable, *args, **kwargs) -> Any:
        """Intercept and validate function calls"""
        func_name = func.__name__
        module_name = func.__module__ if hasattr(func, "__module__") else ""

        # Check for synthetic operations
        if self.is_synthetic_operation(func_name, module_name):
            violation = f"BLOCKED: Synthetic operation {module_name}.{func_name}"
            self.blocked_operations.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "function": func_name,
                    "module": module_name,
                    "violation": violation,
                }
            )
            raise ValueError(f"🔒 COMPLIANCE VIOLATION: {violation}")

        # Validate input data
        for i, arg in enumerate(args):
            if not self.validate_data_source(arg, f"arg_{i}"):
                # Allow primitive types and basic operations
                if not isinstance(arg, (int, float, str, bool, type(None))):
                    violation = f"Non-compliant data in argument {i}"
                    self.blocked_operations.append(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "function": func_name,
                            "violation": violation,
                        }
                    )
                    raise ValueError(f"🔒 COMPLIANCE VIOLATION: {violation}")

        # Log intercepted call
        self.intercepted_calls.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "function": func_name,
                "module": module_name,
                "status": "ALLOWED",
            }
        )

        # Execute function
        return func(*args, **kwargs)

    def enforce_compliance_decorator(self, func: Callable) -> Callable:
        """Decorator to enforce compliance on functions"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.intercept_function_call(func, *args, **kwargs)

        return wrapper

    def get_enforcement_report(self) -> Dict[str, Any]:
        """Generate enforcement activity report"""
        return {
            "enforcer_status": "ACTIVE" if self.active else "INACTIVE",
            "total_intercepted_calls": len(self.intercepted_calls),
            "total_blocked_operations": len(self.blocked_operations),
            "recent_blocked": self.blocked_operations[-10:] if self.blocked_operations else [],
            "recent_allowed": self.intercepted_calls[-10:] if self.intercepted_calls else [],
            "forbidden_operations": list(self.FORBIDDEN_OPERATIONS),
            "forbidden_modules": list(self.FORBIDDEN_MODULES),
        }


# Global compliance enforcer instance
COMPLIANCE_ENFORCER = ComplianceEnforcer()


def enforce_compliance(func: Callable) -> Callable:
    """
    Decorator to enforce compliance on functions

    MANDATORY: All data processing functions must use this decorator
    """
    return COMPLIANCE_ENFORCER.enforce_compliance_decorator(func)


class ComplianceInterceptor:
    """
    System-level compliance interceptor

    Intercepts imports and function calls at runtime
    """

    def __init__(self):
        self.original_import = __builtins__["importlib.import_module"]
        self.blocked_imports = []

    def compliant_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Intercept imports to block non-compliant modules"""
        # Check for forbidden modules
        if any(forbidden in name.lower() for forbidden in COMPLIANCE_ENFORCER.FORBIDDEN_MODULES):
            violation = f"BLOCKED IMPORT: {name}"
            self.blocked_imports.append(
                {"timestamp": datetime.now(timezone.utc).isoformat(), "module": name, "violation": violation}
            )
            raise ImportError(f"🔒 COMPLIANCE VIOLATION: {violation}")

        # Allow approved imports
        return self.original_import(name, globals, locals, fromlist, level)

    def activate(self):
        """Activate import interception"""
        __builtins__["importlib.import_module"] = self.compliant_import

    def deactivate(self):
        """Deactivate import interception"""
        __builtins__["importlib.import_module"] = self.original_import


# Global compliance interceptor
COMPLIANCE_INTERCEPTOR = ComplianceInterceptor()


def activate_compliance_enforcement():
    """Activate system-wide compliance enforcement"""
    COMPLIANCE_INTERCEPTOR.activate()
    COMPLIANCE_ENFORCER.active = True
    logger.info("🔒 COMPLIANCE ENFORCEMENT ACTIVATED - ZERO TOLERANCE MODE")


def deactivate_compliance_enforcement():
    """Deactivate compliance enforcement (emergency only)"""
    COMPLIANCE_INTERCEPTOR.deactivate()
    COMPLIANCE_ENFORCER.active = False
    logger.info("⚠️  COMPLIANCE ENFORCEMENT DEACTIVATED")


def get_full_compliance_report() -> Dict[str, Any]:
    """Get comprehensive compliance report"""
    return {
        "gateway_report": COMPLIANCE_GATEWAY.get_compliance_report(),
        "enforcer_report": COMPLIANCE_ENFORCER.get_enforcement_report(),
        "interceptor_blocked_imports": COMPLIANCE_INTERCEPTOR.blocked_imports,
        "system_status": "FULLY_COMPLIANT" if COMPLIANCE_ENFORCER.active else "ENFORCEMENT_DISABLED",
    }


# Export enforcement functions
__all__ = [
    "ComplianceEnforcer",
    "ComplianceInterceptor",
    "COMPLIANCE_ENFORCER",
    "COMPLIANCE_INTERCEPTOR",
    "enforce_compliance",
    "activate_compliance_enforcement",
    "deactivate_compliance_enforcement",
    "get_full_compliance_report",
]
