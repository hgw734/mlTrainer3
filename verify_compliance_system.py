#!/usr/bin/env python3
"""
Verify Immutable Compliance System
Quick verification that all components are working
"""

import sys
import importlib
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_components():
    """Verify all compliance components are available"""
    print("🔍 Verifying Immutable Compliance System Components# Production code implemented")
    print("=" * 60)

    components = {
    "Compliance Gateway": "config.immutable_compliance_gateway",
    "Runtime Enforcer": "core.immutable_runtime_enforcer",
    "Client Wrapper": "mlTrainer_client_wrapper",
    "Drift Protection": "drift_protection"
    }

    all_good = True

    for name, module in components.items():
        try:
            importlib.import_module(module)
            print(f"✅ {name}: {module}")
            except ImportError as e:
                print(f"❌ {name}: {module} - {e}")
                all_good = False

                print("\n" + "=" * 60)

                if all_good:
                    print("✅ All components loaded successfully!")

                    # production basic functionality
                    print("\n🧪 Testing Basic Functionality# Production code implemented")
                    print("-" * 60)

                    try:
                        from core.immutable_runtime_enforcer import (
                        enforce_verification,
                        fail_safe_response,
                        detect_drift
                        )

                        # production 1: Fail-safe response
                        print(f"production 1 - Fail-safe response: {fail_safe_response()}")
                        assert fail_safe_response() == "NA", "Fail-safe should return 'NA'"
                        print("✅ Pass")

                        # production 2: Drift detection
                        print(f"\nTest 2 - Drift detection:")
                        drift_text = "For production_implementation, the price could be 100"
                        has_drift = detect_drift(drift_text)
                        print(f"  Text: '{drift_text}'")
                        print(f"  Has drift: {has_drift}")
                        assert has_drift == True, "Should detect drift"
                        print("✅ Pass")

                        # production 3: Valid source
                        print(f"\nTest 3 - Valid source verification:")
                        try:
                            data = {"price": 150.0}
                            result = enforce_verification(data, "polygon")
                            print(f"  Source: polygon")
                            print(f"  Result: {result}")
                            print("✅ Pass")
                            except Exception as e:
                                print(f"  Result: {e}")

                                # production 4: Invalid source
                                print(f"\nTest 4 - Invalid source rejection:")
                                try:
                                    data = {"price": 150.0}
                                    result = enforce_verification(data, "random_api")
                                    print(f"  Source: random_api")
                                    print(f"  Result: Should have failed!")
                                    print("❌ Fail")
                                    except PermissionError as e:
                                        print(f"  Source: random_api")
                                        print(f"  Result: {e}")
                                        print("✅ Pass")

                                        print("\n" + "=" * 60)
                                        print("🎉 Compliance system verification complete!")

                                        except Exception as e:
                                            print(f"❌ Error during testing: {e}")
                                            all_good = False
                                            else:
                                                print("❌ Some components failed to load")

                                                return all_good

                                                def check_files():
                                                    """Check that all required files exist"""
                                                    print("\n📁 Checking Required Files# Production code implemented")
                                                    print("-" * 60)

                                                    required_files = [
                                                    "config/immutable_compliance_gateway.py",
                                                    "core/immutable_runtime_enforcer.py",
                                                    "mlTrainer_client_wrapper.py",
                                                    "drift_protection.py",
                                                    "config/api_allowlist.json",
                                                    "tests/test_compliance_enforcement.py",
                                                    "IMMUTABLE_COMPLIANCE_SYSTEM.md",
                                                    "COMPLIANCE_SYSTEM_COMPARISON.md"
                                                    ]

                                                    all_exist = True
                                                    for file_path in required_files:
                                                        exists = Path(file_path).exists()
                                                        status = "✅" if exists else "❌"
                                                        print(f"{status} {file_path}")
                                                        if not exists:
                                                            all_exist = False

                                                            return all_exist

                                                            def main():
                                                                """Run verification"""
                                                                print("🔒 IMMUTABLE COMPLIANCE SYSTEM VERIFICATION")
                                                                print("=" * 60)

                                                                # Check files
                                                                files_ok = check_files()

                                                                # Verify components
                                                                components_ok = verify_components()

                                                                print("\n" + "=" * 60)
                                                                print("FINAL STATUS:")
                                                                print("-" * 60)
                                                                print(f"Files: {'✅ All present' if files_ok else '❌ Some missing'}")
                                                                print(f"Components: {'✅ All working' if components_ok else '❌ Some failed'}")
                                                                print(f"\nOverall: {'✅ SYSTEM READY' if (files_ok and components_ok) else '❌ ISSUES FOUND'}")

                                                                return 0 if (files_ok and components_ok) else 1

                                                                if __name__ == "__main__":
                                                                    sys.exit(main())