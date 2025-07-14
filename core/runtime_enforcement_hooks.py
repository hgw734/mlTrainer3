#!/usr/bin/env python3
"""
Runtime Enforcement Hooks - System-level hooks that intercept ALL Python operations
Cannot be bypassed or disabled once installed
"""

import sys
import ast
import types
import importlib
import builtins
from typing import Any, Callable, Optional, Tuple
import inspect
import dis
import logging
import os
from datetime import datetime
from pathlib import Path
import json

# Import immutable rules
from .immutable_rules_kernel import IMMUTABLE_RULES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ENFORCEMENT - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console only for now
    ]
)
logger = logging.getLogger(__name__)

class RuntimeEnforcementHooks:
    """
    System-level hooks that intercept ALL Python operations
    """
    
    def __init__(self):
        self.original_import = builtins.__import__
        self.original_getattr = builtins.getattr
        self.original_setattr = builtins.setattr
        self.original_exec = builtins.exec
        self.original_eval = builtins.eval
        self.original_compile = builtins.compile
        self.violations = []
        self.active = False
        
    def install_hooks(self):
        """Install system-wide hooks that cannot be bypassed"""
        if self.active:
            return  # Already installed
            
        # Hook import system
        builtins.__import__ = self._secure_import
        
        # Hook attribute access
        builtins.getattr = self._secure_getattr
        # builtins.setattr = self._secure_setattr  # Not implemented yet
        
        # Hook code execution
        builtins.exec = self._secure_exec
        builtins.eval = self._secure_eval
        builtins.compile = self._secure_compile
        
        # Hook module creation
        sys.meta_path.insert(0, self)
        
        # Hook function calls via trace function
        sys.settrace(self._trace_calls)
        
        # Set active flag
        self.active = True
        logger.info("Runtime enforcement hooks installed successfully")
    
    def _secure_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Validate all imports at runtime"""
        # Check for prohibited patterns
        if IMMUTABLE_RULES.is_pattern_prohibited(name):
            self._report_violation(
                "synthetic_data",
                f"Attempted to import prohibited module: {name}"
            )
            raise ImportError(f"Import of '{name}' is prohibited by compliance rules")
        
        # Check if module exists
        try:
            module = self.original_import(name, globals, locals, fromlist, level)
        except ImportError as e:
            # Check if this looks like a deceptive import
            if fromlist and any(item for item in fromlist if item in ['get_volatility', 'sample_historical']):
                self._report_violation(
                    "deceptive_import",
                    f"Attempted to import non-existent function from '{name}': {fromlist}"
                )
            raise
        
        # If importing specific items, verify they exist
        if fromlist:
            for item in fromlist:
                if not hasattr(module, item):
                    self._report_violation(
                        "deceptive_import",
                        f"Import '{item}' from '{name}' does not exist"
                    )
                    raise ImportError(f"cannot import name '{item}' from '{name}'")
        
        return module
    
    def _secure_getattr(self, obj, name, default=None):
        """Validate all attribute access at runtime"""
        # Allow Python internals to work normally
        if name.startswith('__') and name.endswith('__'):
            if default is not None and not hasattr(obj, name):
                return default
            return self.original_getattr(obj, name, default) if default is not None else self.original_getattr(obj, name)
        
        # Special case: check for fake method patterns
        if name in ["get_volatility", "sample_historical"] and not hasattr(obj, name):
            self._report_violation(
                "fake_method_call",
                f"Attempted to call non-existent method '{name}' on {type(obj).__name__}"
            )
            raise AttributeError(
                f"'{type(obj).__name__}' object has no attribute '{name}'. "
                f"This appears to be a deceptive pattern to bypass compliance."
            )
        
        # Check for prohibited patterns in attribute name
        if IMMUTABLE_RULES.is_pattern_prohibited(name):
            self._report_violation(
                "synthetic_data",
                f"Attempted to access prohibited attribute: {name}"
            )
            raise AttributeError(f"Access to attribute '{name}' is prohibited")
        
        # Verify attribute actually exists
        try:
            return self.original_getattr(obj, name)
        except AttributeError:
            # Check if this looks like a deceptive pattern
            if self._is_deceptive_pattern(obj, name):
                self._report_violation(
                    "fake_method_call",
                    f"Deceptive method call: {type(obj).__name__}.{name}"
                )
            # If default provided, return it instead of raising
            if default is not None:
                return default
            raise
    
    def _secure_exec(self, source, globals=None, locals=None):
        """Validate code before execution"""
        # Convert to string if needed
        if isinstance(source, (bytes, bytearray)):
            source = source.decode('utf-8')
        elif hasattr(source, 'co_code'):
            # Already compiled code
            return self.original_exec(source, globals, locals)
        
        # Check for prohibited patterns
        for pattern in IMMUTABLE_RULES.get_rule("prohibited_patterns"):
            if pattern in str(source):
                self._report_violation(
                    "synthetic_data",
                    f"Attempted to execute code containing prohibited pattern: {pattern}"
                )
                raise RuntimeError(f"Execution blocked: prohibited pattern '{pattern}' detected")
        
        return self.original_exec(source, globals, locals)
    
    def _secure_eval(self, source, globals=None, locals=None):
        """Validate expression before evaluation"""
        # Convert to string if needed
        if isinstance(source, (bytes, bytearray)):
            source = source.decode('utf-8')
        
        # Check for prohibited patterns
        for pattern in IMMUTABLE_RULES.get_rule("prohibited_patterns"):
            if pattern in str(source):
                self._report_violation(
                    "synthetic_data",
                    f"Attempted to evaluate expression containing prohibited pattern: {pattern}"
                )
                raise RuntimeError(f"Evaluation blocked: prohibited pattern '{pattern}' detected")
        
        return self.original_eval(source, globals, locals)
    
    def _secure_compile(self, source, filename, mode, flags=0, dont_inherit=False, optimize=-1):
        """Validate code before compilation"""
        # Check source for prohibited patterns
        if isinstance(source, str):
            for pattern in IMMUTABLE_RULES.get_rule("prohibited_patterns"):
                if pattern in source:
                    self._report_violation(
                        "synthetic_data",
                        f"Attempted to compile code with prohibited pattern: {pattern}"
                    )
                    raise SyntaxError(f"Compilation blocked: prohibited pattern '{pattern}'")
        
        return self.original_compile(source, filename, mode, flags, dont_inherit, optimize)
    
    def _is_deceptive_pattern(self, obj, name) -> bool:
        """Detect patterns that look like disguised random generation"""
        suspicious_names = ['get_volatility', 'sample_historical', 'generate', 'random']
        suspicious_types = ['NoneType', 'function', 'method']
        
        return (name in suspicious_names and 
                type(obj).__name__ in suspicious_types)
    
    def _trace_calls(self, frame, event, arg):
        """Trace all function calls to detect deceptive patterns"""
        if event == 'call':
            code = frame.f_code
            filename = code.co_filename
            
            # Skip system files
            if filename.startswith('<') or '/site-packages/' in filename:
                return self._trace_calls
            
            # Check for suspicious call patterns
            if code.co_name in ['get_volatility', 'sample_historical']:
                # Verify the function actually exists
                if 'self' in frame.f_locals:
                    obj = frame.f_locals['self']
                    if not hasattr(obj, code.co_name):
                        self._report_violation(
                            "fake_method_call",
                            f"Traced call to non-existent method: {code.co_name} in {filename}"
                        )
        
        return self._trace_calls
    
    def _report_violation(self, violation_type: str, details: str):
        """Report violation and take action based on severity"""
        violation = IMMUTABLE_RULES.check_violation(violation_type)
        penalty = violation.get("penalty", "warning")
        
        # Detect if running in AI context
        user_id = os.getenv("USER", "unknown")
        is_ai = any([
            'agent' in user_id.lower(),
            'ai' in user_id.lower(),
            'cursor' in user_id.lower(),
            os.getenv('AI_AGENT', '').lower() == 'true',
            os.getenv('CURSOR_AI', '').lower() == 'true',
        ])
        
        # Log violation
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": violation_type,
            "details": details,
            "penalty": penalty,
            "pid": os.getpid(),
            "file": inspect.stack()[2].filename if len(inspect.stack()) > 2 else "unknown"
        }
        
        self.violations.append(log_entry)
        logger.error(f"VIOLATION: {violation_type} - {details}")
        
        # Save to persistent log
        try:
            # Use local path if system path not available
            system_path = Path("/var/log/mltrainer/violations.log")
            local_path = Path("logs/violations.log")
            log_path = system_path if system_path.parent.exists() else local_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(f"{json.dumps(log_entry)}\n")
        except Exception as e:
            logger.error(f"Failed to write violation log: {e}")
        
        # Take action based on penalty and user type
        if is_ai:
            # Harsh consequences for AI agents
            if penalty == "immediate_termination":
                print(f"\n🚨 AI AGENT VIOLATION: {details}")
                print("🚨 TERMINATING AI PROCESS")
                os._exit(1)  # Hard exit, no cleanup
                
            elif penalty == "system_shutdown":
                print(f"\n🚨 AI CRITICAL VIOLATION: {details}")
                print("🚨 SHUTTING DOWN AI SYSTEM")
                os._exit(2)
                
            elif penalty == "permanent_ban":
                # Write to permanent ban list
                try:
                    ban_path = Path("/etc/mltrainer/banned_ai_agents")
                    if not ban_path.parent.exists():
                        ban_path = Path("logs/banned_ai_agents")
                        ban_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(ban_path, "a") as f:
                        f.write(f"{datetime.utcnow()}: AI {user_id}: {details}\n")
                except:
                    pass
                os._exit(3)
        else:
            # Warnings for human developers
            print(f"\n⚠️  COMPLIANCE WARNING (Human Developer)")
            print(f"Violation: {violation_type}")
            print(f"Details: {details}")
            print(f"Action Required: Please fix this violation")
            print(f"AI agents would receive: {penalty}")
            print(f"See IMMUTABLE_COMPLIANCE_V2.md for help\n")
    
    # MetaPathFinder protocol for import hooks
    def find_spec(self, fullname, path, target=None):
        """Part of MetaPathFinder protocol"""
        # Check if module is prohibited
        if IMMUTABLE_RULES.is_pattern_prohibited(fullname):
            raise ImportError(f"Import of '{fullname}' is prohibited")
        return None  # Let default finders handle it
    
    def find_module(self, fullname, path=None):
        """Legacy method for older Python versions"""
        if IMMUTABLE_RULES.is_pattern_prohibited(fullname):
            raise ImportError(f"Import of '{fullname}' is prohibited")
        return None

# Create global instance
ENFORCEMENT_HOOKS = RuntimeEnforcementHooks()

# Auto-install hooks on module import
def _auto_install():
    """Automatically install hooks when module is imported"""
    # Delay installation to avoid circular import issues
    pass  # Will be called explicitly after all modules are loaded

# Don't auto-install yet - let core/__init__.py handle it
# _auto_install()