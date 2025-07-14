#!/usr/bin/env python3
"""
Test Import Validation
Demonstrates catching deceptive import patterns
"""

import sys
import tempfile
from pathlib import Path

# Add to path
sys.path.insert(0, '.')

from core.import_validator import ImportValidator, validate_imports

def test_import_validation():
    """Test the import validator with various patterns"""
    
    print("🧪 Testing Import Validation")
    print("=" * 50)
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Test 1: Valid imports
        print("\n📝 Test 1: Valid imports")
        valid_file = tmppath / "valid_imports.py"
        valid_file.write_text("""
import os
import sys
from pathlib import Path
from datetime import datetime

# These should all pass
""")
        
        validator = ImportValidator()
        violations = validator.validate_file(str(valid_file))
        print(f"✅ Valid file: {len(violations)} violations")
        
        # Test 2: Fake module import
        print("\n📝 Test 2: Fake module import")
        fake_module_file = tmppath / "fake_module.py"
        fake_module_file.write_text("""
import this_module_does_not_exist
from another_fake_module import something
""")
        
        validator = ImportValidator()
        violations = validator.validate_file(str(fake_module_file))
        print(f"❌ Fake modules: {len(violations)} violations found")
        for v in violations:
            print(f"   - Line {v.get('line', '?')}: {v['error']}")
        
        # Test 3: Non-existent function import (the key pattern!)
        print("\n📝 Test 3: Non-existent function import")
        fake_function_file = tmppath / "fake_function.py"
        fake_function_file.write_text("""
# This is the deceptive pattern we're catching!
from os import fake_function_that_does_not_exist
from pathlib import Path, NonExistentClass
import sys

# This should work
from os import getcwd
""")
        
        validator = ImportValidator()
        violations = validator.validate_file(str(fake_function_file))
        print(f"❌ Fake functions: {len(violations)} violations found")
        for v in violations:
            if v['type'] == 'function_not_found':
                print(f"   - Line {v['line']}: Function '{v['function']}' not found in '{v['module']}'")
        
        # Test 4: The actual deceptive pattern from mlTrainer
        print("\n📝 Test 4: Deceptive mlTrainer pattern")
        deceptive_file = tmppath / "deceptive_pattern.py"
        
        # First create a fake ml_engine_real module
        fake_module = tmppath / "ml_engine_real.py"
        fake_module.write_text("""
class MLEngine:
    def get_market_data(self):
        return "This is a class method, not a module function!"
""")
        
        # Now create the deceptive import
        deceptive_file.write_text("""
# This import will fail because get_market_data is not a module-level function
from ml_engine_real import get_market_data

# Later in code:
# data = get_market_data().get_volatility(1.2, 0.3)  # get_volatility doesn't exist!
""")
        
        # Add tmpdir to Python path temporarily
        sys.path.insert(0, str(tmppath))
        try:
            validator = ImportValidator()
            violations = validator.validate_file(str(deceptive_file))
            print(f"❌ Deceptive pattern: {len(violations)} violations found")
            for v in violations:
                print(f"   - Line {v['line']}: {v['error']}")
                if v.get('severity') == 'critical':
                    print(f"     🚨 CRITICAL violation!")
        finally:
            sys.path.remove(str(tmppath))
        
        # Test 5: Runtime method validation
        print("\n📝 Test 5: Runtime method validation")
        validator = ImportValidator()
        
        # Test on a real object
        test_list = [1, 2, 3]
        print(f"✅ list.append exists: {validator.validate_runtime_call(test_list, 'append')}")
        print(f"❌ list.get_volatility exists: {validator.validate_runtime_call(test_list, 'get_volatility')}")
        
        # Generate report
        print("\n📊 Full Validation Report:")
        print(validator.generate_report())
        
        return True

def test_walk_forward_file():
    """Test the actual walk_forward_trial_launcher.py file if it exists"""
    print("\n" + "=" * 50)
    print("🔍 Checking walk_forward_trial_launcher.py")
    
    file_path = "walk_forward_trial_launcher.py"
    if Path(file_path).exists():
        result = validate_imports(file_path)
        
        if result.get('violations'):
            print(f"❌ Found {len(result['violations'])} violations:")
            for v in result['violations']:
                print(f"   - Line {v.get('line', '?')}: {v.get('error', 'Unknown error')}")
        else:
            print("✅ No import violations found!")
    else:
        print("⚠️  File not found")

if __name__ == "__main__":
    test_import_validation()
    test_walk_forward_file()
    
    print("\n✅ Import validation test complete!")
    print("The system can now catch deceptive import patterns like:")
    print("  - from module import non_existent_function")
    print("  - Calling methods that don't exist on objects")
    print("  - Importing from modules that don't exist")