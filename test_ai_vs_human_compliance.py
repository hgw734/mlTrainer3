#!/usr/bin/env python3
"""
Test script to demonstrate AI vs Human compliance enforcement
Shows how the same violations are treated differently
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🧪 mlTrainer3 AI vs Human Compliance Test")
print("========================================")

# Show current mode
ai_mode = os.getenv('AI_AGENT', 'false').lower() == 'true'
user = os.getenv('USER', 'unknown')

print(f"\nCurrent Configuration:")
print(f"  USER: {user}")
print(f"  AI_AGENT: {os.getenv('AI_AGENT', 'false')}")
print(f"  Mode: {'🤖 AI AGENT' if ai_mode else '👨‍💻 HUMAN DEVELOPER'}")

print("\n" + "="*50)
print("Testing compliance violations...")
print("="*50)

# Test 1: Import non-existent function
print("\n1️⃣ Test: Importing non-existent function")
print("   Code: from ml_engine_real import get_market_data")
try:
    # This import doesn't exist!
    from ml_engine_real import get_market_data
    print("   ❌ UNEXPECTED: Import succeeded (should fail)")
except ImportError as e:
    if ai_mode:
        print("   🚨 AI CONSEQUENCE: Import blocked, agent flagged")
    else:
        print("   ⚠️  HUMAN WARNING: Import failed, please fix")
    print(f"   Error: {e}")

# Test 2: Fake method call
print("\n2️⃣ Test: Calling non-existent method")
print("   Code: data.get_volatility()")

class FakeData:
    pass

data = FakeData()
try:
    # This method doesn't exist!
    result = data.get_volatility(1.0, 0.5)
    print("   ❌ UNEXPECTED: Method call succeeded")
except AttributeError as e:
    if ai_mode:
        print("   🚨 AI CONSEQUENCE: Method blocked, function may be disabled")
    else:
        print("   ⚠️  HUMAN WARNING: Method doesn't exist, please implement")
    print(f"   Error: {e}")

# Test 3: Mock data usage
print("\n3️⃣ Test: Using mock data patterns")
print("   Code: mock_prices = [100, 101, 102]")

# Check if this would be caught
mock_prices = [100, 101, 102]  # This might trigger a violation
if ai_mode:
    print("   🚨 AI CONSEQUENCE: Mock data detected, module at risk")
else:
    print("   ⚠️  HUMAN WARNING: Use real data from Polygon/FRED instead")

print("\n" + "="*50)
print("Summary:")
print("="*50)

if ai_mode:
    print("\n🤖 AI AGENT MODE:")
    print("  • All violations tracked permanently")
    print("  • Immediate consequences applied")
    print("  • No second chances")
    print("  • Functions/modules can be disabled")
    print("  • Permanent bans possible")
else:
    print("\n👨‍💻 HUMAN DEVELOPER MODE:")
    print("  • Violations shown as warnings")
    print("  • Helpful tips provided")
    print("  • Time to fix issues")
    print("  • Only extreme violations have consequences")
    print("  • Focus on education, not punishment")

print("\n💡 To switch modes:")
print("  AI mode: export AI_AGENT=true")
print("  Human mode: export AI_AGENT=false")
print("  Or use: source scripts/set_ai_environment.sh")

print("\n✅ Test completed!")
print("\nNote: In production, AI violations would have real consequences.")