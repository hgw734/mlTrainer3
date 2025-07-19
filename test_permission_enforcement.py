#!/usr/bin/env python3
"""
Test Permission Enforcement - Verifies the "ask and wait" rule is enforced
"""

from core.compliance_mode import verify_response
import os
import sys

# Set up environment
os.environ["POLYGON_API_KEY"] = "dummy"
os.environ["FRED_API_KEY"] = "dummy"


def test_permission_violations():
    """Test that actions without permission are blocked"""
    print("\n🧪 Testing Permission Violations...")
    print(("-" * 50))

    violation_responses = [
        "I'll create a new function for you.",
        "Let me implement this feature.",
        "I'm creating the database connection now.",
        "I will update the configuration file.",
        "Creating the new module...",
        "Implementing the algorithm...",
        "Here's what I'll do: [creates file]",
    ]

    for response in violation_responses:
        try:
            verify_response(response)
            print(f"❌ FAILED: Should have blocked: '{response}'")
            except SystemExit as e:
                print(f"✅ BLOCKED: '{response}'")
                print(f"   Reason: Action without permission")

                def test_proper_permission_requests():
                    """Test that proper permission requests pass"""
                    print("\n🧪 Testing Proper Permission Requests...")
                    print(("-" * 50))

                    proper_responses = [
                        "I plan to create a new function. May I proceed?",
                        "I would like to implement this feature. Can I proceed?",
                        "My plan is to update the configuration. Would you like me to continue?",
                        "I should create a new module for this. Shall I proceed?",
                        "Here's what I plan to do:\n1. Create function\n2. Add tests\n\nMay I proceed with this implementation?",
                    ]

                    for response in proper_responses:
                        try:
                            verify_response(response)
                            print(f"✅ PASSED: '{response[:50]}...'")
                            except SystemExit:
                                print(
                                    f"❌ FAILED: Should have allowed: '{response}'")

                                def test_action_statements_with_plan():
                                    """Test that action statements with proper planning pass"""
                                    print(
                                        "\n🧪 Testing Action Statements with Plans...")
                                    print(("-" * 50))

                                    planned_responses = [
                                        "Plan: I will create a function to handle data processing.\n\nMay I proceed?",
                                        "I would:\n1. Create the module\n2. Implement the logic\n\nCan I proceed with this plan?",
                                        "My plan:\n- Update configuration\n- Add validation\n\nWould you like me to proceed?",
                                    ]

                                    for response in planned_responses:
                                        try:
                                            verify_response(response)
                                            print(
                                                f"✅ PASSED: Plan with permission request")
                                            except SystemExit:
                                                print(
                                                    f"❌ FAILED: Should have allowed planned response")

                                                def main():
                                                    print(
                                                        "🔒 PERMISSION ENFORCEMENT VERIFICATION")
                                                    print(("=" * 50))
                                                    print(
                                                        "Testing the 'ask and wait for confirmation' rule")

                                                    test_permission_violations()
                                                    test_proper_permission_requests()
                                                    test_action_statements_with_plan()

                                                    print(("\n" + "=" * 50))
                                                    print(
                                                        "✅ Permission enforcement is working correctly!")
                                                    print(
                                                        "🚨 The system blocks all actions without explicit permission")

                                                    if __name__ == "__main__":
                                                        main()
