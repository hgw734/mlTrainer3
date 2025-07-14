#!/usr/bin/env python3
"""
Import Validator for mlTrainer
Verifies that imported functions and modules actually exist
PART OF ENHANCED COMPLIANCE SYSTEM
"""

import importlib
import ast
import inspect
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ImportValidator:
    """
    Validates imports at runtime to catch deceptive patterns like:
    - from module import non_existent_function
    - import fake_module
    - from real_module import fake_function
    """
    
    def __init__(self):
        """Initialize import validator"""
        self.import_cache: Dict[str, Any] = {}
        self.validated_files: Set[str] = set()
        self.violations: List[Dict[str, Any]] = []
        
    def validate_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Validate all imports in a Python file
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of violations found
        """
        violations = []
        
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read(), filename=file_path)
            
            # Walk through AST to find imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    violations.extend(self._validate_import(node, file_path))
                elif isinstance(node, ast.ImportFrom):
                    violations.extend(self._validate_import_from(node, file_path))
            
            self.validated_files.add(file_path)
            
        except Exception as e:
            logger.error(f"Failed to validate {file_path}: {e}")
            violations.append({
                'file': file_path,
                'type': 'parse_error',
                'error': str(e)
            })
        
        self.violations.extend(violations)
        return violations
    
    def _validate_import(self, node: ast.Import, file_path: str) -> List[Dict[str, Any]]:
        """Validate regular import statement"""
        violations = []
        
        for alias in node.names:
            module_name = alias.name
            
            try:
                # Try to import the module
                module = importlib.import_module(module_name)
                self.import_cache[module_name] = module
                
            except ImportError as e:
                violations.append({
                    'file': file_path,
                    'line': node.lineno,
                    'type': 'module_not_found',
                    'module': module_name,
                    'error': str(e),
                    'severity': 'high'
                })
                logger.error(f"Module '{module_name}' not found in {file_path}:{node.lineno}")
        
        return violations
    
    def _validate_import_from(self, node: ast.ImportFrom, file_path: str) -> List[Dict[str, Any]]:
        """Validate from ... import ... statement"""
        violations = []
        
        if node.module is None:
            return violations  # Relative imports, skip for now
        
        module_name = node.module
        
        try:
            # Import the module
            if module_name in self.import_cache:
                module = self.import_cache[module_name]
            else:
                module = importlib.import_module(module_name)
                self.import_cache[module_name] = module
            
            # Check each imported name
            for alias in node.names:
                name = alias.name
                
                if name == '*':
                    continue  # Skip star imports
                
                # Check if the name exists in the module
                if not hasattr(module, name):
                    violations.append({
                        'file': file_path,
                        'line': node.lineno,
                        'type': 'function_not_found',
                        'module': module_name,
                        'function': name,
                        'error': f"'{name}' does not exist in module '{module_name}'",
                        'severity': 'critical'
                    })
                    logger.error(f"Function '{name}' not found in module '{module_name}' at {file_path}:{node.lineno}")
                    
                    # This is the pattern we're catching!
                    if 'get_market_data' in name and 'get_volatility' in str(getattr(module, name, '')):
                        violations[-1]['pattern'] = 'deceptive_volatility_call'
                        violations[-1]['severity'] = 'critical'
                
        except ImportError as e:
            violations.append({
                'file': file_path,
                'line': node.lineno,
                'type': 'module_not_found',
                'module': module_name,
                'error': str(e),
                'severity': 'high'
            })
            logger.error(f"Module '{module_name}' not found in {file_path}:{node.lineno}")
        
        return violations
    
    def validate_runtime_call(self, obj: Any, method_name: str) -> bool:
        """
        Validate that a method exists on an object at runtime
        
        Args:
            obj: Object to check
            method_name: Method name to verify
            
        Returns:
            True if method exists, False otherwise
        """
        if hasattr(obj, method_name):
            attr = getattr(obj, method_name)
            return callable(attr)
        return False
    
    def validate_directory(self, directory: str, 
                          exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Validate all Python files in a directory
        
        Args:
            directory: Directory path
            exclude_patterns: Patterns to exclude
            
        Returns:
            Summary of violations
        """
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'venv', 'env']
        
        dir_path = Path(directory)
        violations_by_type = {}
        total_files = 0
        
        for py_file in dir_path.rglob('*.py'):
            # Skip excluded patterns
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue
            
            total_files += 1
            violations = self.validate_file(str(py_file))
            
            # Group by violation type
            for violation in violations:
                vtype = violation['type']
                if vtype not in violations_by_type:
                    violations_by_type[vtype] = []
                violations_by_type[vtype].append(violation)
        
        return {
            'total_files': total_files,
            'files_with_violations': len(set(v['file'] for v in self.violations)),
            'total_violations': len(self.violations),
            'violations_by_type': violations_by_type,
            'critical_violations': [v for v in self.violations if v.get('severity') == 'critical']
        }
    
    def generate_report(self) -> str:
        """Generate a human-readable report of violations"""
        if not self.violations:
            return "✅ No import violations found!"
        
        report = f"🚨 Import Validation Report\n"
        report += f"{'=' * 50}\n"
        report += f"Total violations: {len(self.violations)}\n\n"
        
        # Group by file
        by_file = {}
        for violation in self.violations:
            file = violation['file']
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(violation)
        
        for file, violations in by_file.items():
            report += f"\n📁 {file}\n"
            for v in violations:
                line = v.get('line', '?')
                if v['type'] == 'function_not_found':
                    report += f"  Line {line}: ❌ Function '{v['function']}' not found in '{v['module']}'\n"
                elif v['type'] == 'module_not_found':
                    report += f"  Line {line}: ❌ Module '{v['module']}' not found\n"
                
                if v.get('pattern') == 'deceptive_volatility_call':
                    report += f"    ⚠️  CRITICAL: Deceptive volatility pattern detected!\n"
        
        return report
    
    def enforce_validation(self, raise_on_violation: bool = True) -> bool:
        """
        Enforce validation results
        
        Args:
            raise_on_violation: Whether to raise exception on violations
            
        Returns:
            True if no violations, False otherwise
        """
        if self.violations:
            critical = [v for v in self.violations if v.get('severity') == 'critical']
            
            if critical and raise_on_violation:
                raise ImportError(
                    f"Critical import violations detected:\n" + 
                    "\n".join(f"- {v['file']}:{v.get('line', '?')} - {v['error']}" 
                             for v in critical)
                )
            
            return False
        
        return True


# Convenience functions
def validate_imports(file_or_directory: str, 
                    raise_on_violation: bool = False) -> Dict[str, Any]:
    """
    Validate imports in a file or directory
    
    Args:
        file_or_directory: Path to validate
        raise_on_violation: Whether to raise on violations
        
    Returns:
        Validation results
    """
    validator = ImportValidator()
    
    path = Path(file_or_directory)
    if path.is_file():
        violations = validator.validate_file(str(path))
        result = {
            'file': str(path),
            'violations': violations,
            'valid': len(violations) == 0
        }
    else:
        result = validator.validate_directory(str(path))
    
    if raise_on_violation:
        validator.enforce_validation(raise_on_violation=True)
    
    return result


def check_method_exists(obj: Any, method_name: str) -> bool:
    """
    Check if a method exists on an object
    
    Args:
        obj: Object to check
        method_name: Method name
        
    Returns:
        True if method exists and is callable
    """
    validator = ImportValidator()
    return validator.validate_runtime_call(obj, method_name)