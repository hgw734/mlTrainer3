#!/usr/bin/env python3
"""
Simple test to show AI vs Human compliance difference
"""

import os

print("🧪 mlTrainer3 AI vs Human Mode Test")
print("===================================")

# Check current mode
ai_mode = os.getenv('AI_AGENT', 'false').lower() == 'true'
user = os.getenv('USER', 'unknown')

print(f"\nCurrent Mode: {'🤖 AI AGENT' if ai_mode else '👨‍💻 HUMAN DEVELOPER'}")
print(f"User: {user}")
print(f"AI_AGENT env: {os.getenv('AI_AGENT', 'not set')}")

print("\n📋 Compliance Rules:")

if ai_mode:
    print("\n🤖 AI AGENT RULES:")
    print("  ❌ NO warnings - immediate consequences")
    print("  ❌ Functions can be disabled")
    print("  ❌ Modules can be blocked")
    print("  ❌ Processes can be terminated")
    print("  ❌ Permanent bans possible")
else:
    print("\n👨‍💻 HUMAN DEVELOPER RULES:")
    print("  ⚠️  Warnings for violations")
    print("  💡 Helpful tips provided")
    print("  ⏰ Time to fix issues")
    print("  📚 Documentation links")
    print("  🛡️  Only extreme violations enforced")

print("\n💡 To test different modes:")
print("  Human mode: python3 test_simple_ai_human.py")
print("  AI mode: AI_AGENT=true python3 test_simple_ai_human.py")

print("\n✅ Test complete!")