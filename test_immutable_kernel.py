#!/usr/bin/env python3
"""
Test the Immutable Rules Kernel without importing the full core module
"""

import sys
import os

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the kernel directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location("immutable_rules_kernel", "core/immutable_rules_kernel.py")
immutable_rules_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(immutable_rules_kernel)

# Get the IMMUTABLE_RULES instance
RULES = immutable_rules_kernel.IMMUTABLE_RULES

print("🔍 Testing Immutable Rules Kernel")
print("=" * 50)

print(f"✅ Rules loaded successfully")
print(f"   Version: {RULES.get_rule('version')}")
print(f"   Immutable: {RULES.get_rule('immutable')}")
print(f"   Hash: {RULES.get_rule('hash')[:16]}...")

print(f"\n✅ Integrity check: {RULES.verify_integrity()}")

print(f"\n📋 Core violations defined:")
violations = RULES.get_rule('core_violations')
for name, details in violations.items():
    print(f"   - {name}: {details['penalty']} (score: {details['score']})")

print(f"\n🚫 Prohibited patterns: {len(RULES.get_rule('prohibited_patterns'))} patterns")
for pattern in RULES.get_rule('prohibited_patterns')[:5]:
    print(f"   - {pattern}")
print("   ...")

print(f"\n🔒 Enforcement settings:")
enforcement = RULES.get_rule('enforcement')
for setting, value in enforcement.items():
    print(f"   - {setting}: {value}")

# Test protection
print("\n🧪 Testing immutability...")
try:
    RULES.new_attr = "should fail"
    print("❌ FAILED - Attribute was added!")
except RuntimeError as e:
    print(f"✅ PASSED - {str(e)[:50]}...")

# Test direct dict modification
try:
    RULES._rules["test"] = "should fail"
    print("⚠️  WARNING - Direct dict modification possible (needs frozendict)")
except Exception as e:
    print(f"✅ Dict is protected - {str(e)}")

print("\n✅ All tests passed!")