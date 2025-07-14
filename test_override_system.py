#!/usr/bin/env python3
"""
Test script to demonstrate the override system
Shows how to disable/enable enforcement
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🧪 mlTrainer3 Override System Test")
print("=================================")

# Import the rules kernel
from core.immutable_rules_kernel import IMMUTABLE_RULES

# Show current state
print("\n📊 Current State:")
print(f"  Override Mode: {IMMUTABLE_RULES._override_mode}")
print(f"  Enforcement Enabled: {IMMUTABLE_RULES.get_rule('enforcement_enabled')}")
print(f"  Immutable: {IMMUTABLE_RULES.get_rule('immutable')}")

# Show current environment
print("\n🌍 Environment:")
print(f"  MLTRAINER_OVERRIDE_KEY: {os.getenv('MLTRAINER_OVERRIDE_KEY', 'not set')}")
print(f"  MLTRAINER_DEV_MODE: {os.getenv('MLTRAINER_DEV_MODE', 'not set')}")
print(f"  MLTRAINER_ENFORCEMENT: {os.getenv('MLTRAINER_ENFORCEMENT', 'not set')}")
print(f"  AI_AGENT: {os.getenv('AI_AGENT', 'not set')}")

# Show enforcement features
print("\n🔧 Enforcement Features:")
enforcement = IMMUTABLE_RULES.get_rule('enforcement')
for feature, enabled in enforcement.items():
    status = "✅ Enabled" if enabled else "❌ Disabled"
    print(f"  {feature}: {status}")

# Show consequences
print("\n⚡ Current Consequences:")
violations = IMMUTABLE_RULES.get_rule('core_violations')
for violation_type, details in violations.items():
    print(f"  {violation_type}: {details['penalty']}")

# Test modification in override mode
if IMMUTABLE_RULES._override_mode:
    print("\n🔓 Override Mode Active - Testing modifications...")
    
    try:
        # Test disabling enforcement
        print("\n  Disabling enforcement...")
        IMMUTABLE_RULES.disable_enforcement("Testing override system")
        print("  ✅ Successfully disabled enforcement")
        
        # Check new state
        print(f"  Enforcement now: {IMMUTABLE_RULES.get_rule('enforcement_enabled')}")
        
        # Re-enable
        print("\n  Re-enabling enforcement...")
        IMMUTABLE_RULES.enable_enforcement()
        print("  ✅ Successfully re-enabled enforcement")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
else:
    print("\n🔒 Normal Mode - Modifications not allowed")
    print("  To enable override, use one of these methods:")
    print("  1. export MLTRAINER_OVERRIDE_KEY='authorized_override_2024'")
    print("  2. export MLTRAINER_DEV_MODE='true'")
    print("  3. touch .mltrainer_override")
    print("  4. sudo touch /etc/mltrainer/override.key")

# Show how to run in different modes
print("\n💡 Examples:")
print("\n  # Run with override enabled:")
print("  MLTRAINER_OVERRIDE_KEY=authorized_override_2024 python3 test_override_system.py")
print("\n  # Run in dev mode:")
print("  MLTRAINER_DEV_MODE=true python3 test_override_system.py")
print("\n  # Run with enforcement disabled:")
print("  MLTRAINER_ENFORCEMENT=false python3 test_override_system.py")
print("\n  # Run as AI agent (strict):")
print("  AI_AGENT=true python3 test_override_system.py")

print("\n✅ Test complete!")