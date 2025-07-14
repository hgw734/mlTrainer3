#!/usr/bin/env python3
"""
Simple demonstration of override system
"""

import os
import sys
import importlib.util

# Load the immutable rules kernel directly
spec = importlib.util.spec_from_file_location("immutable_rules_kernel", "core/immutable_rules_kernel.py")
immutable_rules_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(immutable_rules_kernel)

RULES = immutable_rules_kernel.IMMUTABLE_RULES

print("🧪 mlTrainer3 Override System Demo")
print("==================================")

# Show current state
print("\n📊 Current State:")
print(f"  Override Mode: {RULES._override_mode}")
print(f"  Enforcement: {RULES.get_rule('enforcement_enabled')}")

# Show environment
print("\n🌍 Environment Variables:")
for var in ['MLTRAINER_OVERRIDE_KEY', 'MLTRAINER_DEV_MODE', 'MLTRAINER_ENFORCEMENT', 'AI_AGENT']:
    print(f"  {var}: {os.getenv(var, 'not set')}")

# Show consequences
print("\n⚡ Current Consequences (for AI agents):")
for vtype, details in RULES.get_rule('core_violations').items():
    print(f"  {vtype}: {details['penalty']}")

print("\n💡 To enable override mode, run:")
print("  MLTRAINER_OVERRIDE_KEY=authorized_override_2024 python3 test_override_simple.py")
print("\nOr to disable enforcement:")
print("  MLTRAINER_ENFORCEMENT=false python3 test_override_simple.py")