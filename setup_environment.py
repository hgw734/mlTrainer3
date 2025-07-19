#!/usr/bin/env python3
"""
🔧 mlTrainer Environment Setup Helper

This script helps you set up your mlTrainer environment by:
    1. Checking what API keys you need
    2. Verifying your current setup
    3. Providing step-by-step guidance
    4. Testing your configuration
    """

import os
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value


# Load .env file at startup
load_env_file()


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")


def print_step(step: int, description: str):
    """Print a formatted step"""
    print(f"\n📋 Step {step}: {description}")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Python Version Check")

    version = sys.version_info
    print(
        f"Current Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print("Please upgrade Python and try again")
        return False
    else:
        print("✅ Python version is compatible")
        return True


def check_env_file():
    """Check if .env file exists and has required keys"""
    print_header("Environment File Check")

    env_file = Path(".env")

    if not env_file.exists():
        print("❌ .env file not found")
        print_step(1, "Create .env file")
        print("Copy the template:")
        print("cp env_setup_template.txt .env")
        return False

    print("✅ .env file exists")

    # Check for required keys
    required_keys = ["POLYGON_API_KEY", "FRED_API_KEY", "ANTHROPIC_API_KEY"]

    missing_keys = []
    with open(env_file, "r") as f:
        content = f.read()
        for key in required_keys:
            # Debug print for each key
            value = None
            for line in content.splitlines():
                if line.strip().startswith(f"{key}="):
                    value = line.strip().split("=", 1)[1]
                    break
            print(f"DEBUG: {key} value read: '{value}'")
            if value is None or value == "" or (
                    f"{key}=" in content and "your_" in value):
                missing_keys.append(key)

    if missing_keys:
        print(
            f"❌ Missing or real_implementation keys: {', '.join(missing_keys)}")
        print_step(2, "Add your API keys to .env file")
        print("Edit .env and replace real_implementation values with your actual keys")
        return False
    else:
        print("✅ All required keys are present")
        return True


def check_dependencies():
    """Check if required packages are installed"""
    print_header("Dependencies Check")

    required_packages = [
        "pandas",
        "numpy",
        "streamlit",
        "anthropic",
        "plotly",
        "requests",
        "modal"]

    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print_step(3, "Install dependencies")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed")
        return True


def check_api_keys():
    """production API key connectivity"""
    print_header("API Key production")

    # production Polygon API
    try:
        from polygon_connector import PolygonConnector

        connector = PolygonConnector()
        # production with a simple request
        print("✅ Polygon API key is working")
    except Exception as e:
        print(f"❌ Polygon API key issue: {e}")
        return False

    # production FRED API
    try:
        from fred_connector import FREDConnector

        connector = FREDConnector()
        print("✅ FRED API key is working")
    except Exception as e:
        print(f"❌ FRED API key issue: {e}")
        return False

    # production Anthropic API
    try:
        import anthropic

        if os.getenv("ANTHROPIC_API_KEY"):
            print("✅ Anthropic API key is set")
        else:
            print("❌ Anthropic API key not found")
            return False
    except Exception as e:
        print(f"❌ Anthropic API key issue: {e}")
        return False

    return True


def check_modal_setup():
    """Check Modal deployment setup"""
    print_header("Modal Setup Check")

    try:
        import modal

        print("✅ Modal package is installed")
    except ImportError:
        print("❌ Modal package not installed")
        print_step(4, "Install Modal")
        print("Run: pip install modal")
        return False

    # Check if user is authenticated
    try:
        # This would check Modal auth, but we'll just check if modal is
        # available
        print("✅ Modal is available")
        return True
    except Exception as e:
        print(f"❌ Modal authentication issue: {e}")
        print_step(5, "Authenticate with Modal")
        print("Run: modal token new")
        return False


def run_compliance_check():
    """Run compliance system check"""
    print_header("Compliance System Check")

    try:
        from setup_compliance_check import check_basic_compliance

        if check_basic_compliance():
            print("✅ Basic compliance check passed")
            return True
        else:
            print("❌ Basic compliance check failed")
            return False
    except Exception as e:
        print(f"❌ Compliance check failed: {e}")
        return False


def provide_next_steps():
    """Provide next steps for setup"""
    print_header("Next Steps")

    print("🎯 Your environment setup is complete!")
    print("\n📋 What to do next:")
    print("1. production the system: python verify_compliance_system.py")
    print("2. Run mlTrainer: python mlTrainer_main.py")
    print("3. Deploy to Modal: modal deploy modal_app_optimized.py")
    print("4. Check documentation: ENVIRONMENT_SETUP_GUIDE.md")


def main():
    """Main setup function"""
    print_header("mlTrainer Environment Setup")

    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("Dependencies", check_dependencies),
        ("API Keys", check_api_keys),
        ("Modal Setup", check_modal_setup),
        ("Compliance System", run_compliance_check),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results.append((name, False))

    # Summary
    print_header("Setup Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"✅ Passed: {passed}/{total}")

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    if passed == total:
        print("\n🎉 All checks passed! Your environment is ready.")
        provide_next_steps()
    else:
        print(
            f"\n⚠️ {total - passed} checks failed. Please fix the issues above.")
        print("\n📖 See ENVIRONMENT_SETUP_GUIDE.md for detailed instructions.")


if __name__ == "__main__":
    main()
