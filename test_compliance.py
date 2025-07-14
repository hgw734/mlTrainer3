#!/usr/bin/env python3
"""
Test the compliance gateway to ensure it properly detects missing model implementations
"""

import sys

try:
    # This should trigger the compliance check
    from config.immutable_compliance_gateway import ComplianceGateway

    print("🔍 Testing Compliance Gateway Model Verification...")

    # Try to initialize the gateway
    gateway = ComplianceGateway()

    print("✅ Compliance Gateway initialized successfully!")
    print("✅ All model implementations verified!")

    except ValueError as e:
        print(f"❌ Compliance violation detected: {e}")
        print("This is expected if models are missing implementations.")
        sys.exit(1)

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
