#!/usr/bin/env python3
"""
Simplified Compliance Check for Environment Setup
"""

import os
import sys


def check_basic_compliance():
    """Check basic compliance without requiring all model implementations"""
    print("🔒 Basic Compliance Check")
    print(("-" * 30))

    # Check core compliance files exist
    required_files = [
        "core/compliance_mode.py",
        "config/compliance_enforcer.py",
        "verify_compliance_enforcement.py"]

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} - missing")
                return False

                # Check environment variables
                required_env_vars = ["POLYGON_API_KEY", "FRED_API_KEY"]

                for var in required_env_vars:
                    if os.getenv(var):
                        print(f"✅ {var} is set")
                        else:
                            print(f"❌ {var} is not set")
                            return False

                            print("✅ Basic compliance check passed")
                            return True

                            if __name__ == "__main__":
                                success = check_basic_compliance()
                                if success:
                                    print(
                                        "\n🎉 Environment is ready for development!")
                                    else:
                                        print(
                                            "\n⚠️ Please fix the issues above before proceeding.")
                                        sys.exit(1)
