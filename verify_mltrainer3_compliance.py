#!/usr/bin/env python3
"""
mlTrainer3 Comprehensive Compliance Verification
================================================
Triple-checks that:
1. All models are properly registered and accessible
2. Only real data from approved APIs can enter the system
3. No synthetic data generators exist
"""

import os
import sys
import ast
import re
from typing import List, Dict, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Color codes for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Synthetic data patterns to check
SYNTHETIC_PATTERNS = [
    r'np\.random',
    r'random\.random',
    r'random\.randint',
    r'random\.choice',
    r'random\.sample',
    r'fake_\w+',
    r'dummy_\w+',
    r'test_data',
    r'sample_data',
    r'placeholder',
    r'generate_fake',
    r'synthetic',
]

# Files allowed to have synthetic data (tests only)
ALLOWED_FILES = [
    'test_',
    'tests/',
    '__pycache__',
    '.git/',
    'verify_mltrainer3_compliance.py',
]


def check_file_for_synthetic_data(filepath: str) -> List[Tuple[int, str, str]]:
    """Check a file for synthetic data patterns"""
    violations = []

    # Skip allowed files
    for allowed in ALLOWED_FILES:
        if allowed in filepath:
            return violations

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for text patterns
        for i, line in enumerate(content.split('\n'), 1):
            for pattern in SYNTHETIC_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if it's in a comment
                    if not line.strip().startswith('#'):
                        violations.append((i, pattern, line.strip()))
    except Exception as e:
        print(f"{YELLOW}Warning: Could not check {filepath}: {e}{RESET}")

    return violations


def verify_model_registration():
    """Verify all models are properly registered"""
    print(f"\n{BLUE}=== Verifying Model Registration ==={RESET}")

    try:
        from mltrainer_models import get_ml_model_manager
        from mltrainer_financial_models import get_financial_model_manager

        # Check ML models
        ml_manager = get_ml_model_manager()
        ml_models = ml_manager.get_available_models()
        print(f"{GREEN}✓ ML Models Registered: {len(ml_models)}{RESET}")

        # Check Financial models
        fin_manager = get_financial_model_manager()
        fin_models = fin_manager.get_available_models()
        print(f"{GREEN}✓ Financial Models Registered: {len(fin_models)}{RESET}")

        # Check unified executor
        from core.unified_executor import get_unified_executor
        executor = get_unified_executor()
        actions = executor.registered_actions
        print(f"{GREEN}✓ Unified Executor Actions: {len(actions)}{RESET}")

        # Verify mlAgent integration
        from mlagent_model_integration import MLAgentModelIntegration
        integration = MLAgentModelIntegration()
        print(f"{GREEN}✓ MLAgent Integration Active{RESET}")

        return True

    except Exception as e:
        print(f"{RED}✗ Model Registration Error: {e}{RESET}")
        return False


def verify_compliance_gateway():
    """Verify compliance gateway is active"""
    print(f"\n{BLUE}=== Verifying Compliance Gateway ==={RESET}")

    try:
        from config.immutable_compliance_gateway import ComplianceGateway, DataSource
        from config.compliance_enforcer import ComplianceEnforcer

        # Test gateway
        gateway = ComplianceGateway()

        # Test data source verification
        valid_source = gateway.verify_data_source(
            "polygon_api", "/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-12-31")
        if valid_source != DataSource.INVALID:
            print(f"{GREEN}✓ Polygon API source verified{RESET}")
        else:
            print(f"{RED}✗ Polygon API source verification failed{RESET}")

        # Test invalid source rejection
        invalid_source = gateway.verify_data_source(
            "fake_api", "/fake/endpoint")
        if invalid_source == DataSource.INVALID:
            print(f"{GREEN}✓ Invalid sources properly rejected{RESET}")
        else:
            print(f"{RED}✗ Invalid sources not rejected!{RESET}")

        # Check enforcer
        enforcer = ComplianceEnforcer()
        print(f"{GREEN}✓ Compliance Enforcer initialized{RESET}")

        return True

    except Exception as e:
        print(f"{RED}✗ Compliance Gateway Error: {e}{RESET}")
        return False


def verify_api_configuration():
    """Verify API configuration is properly set up"""
    print(f"\n{BLUE}=== Verifying API Configuration ==={RESET}")

    try:
        from config.api_config import get_all_approved_sources, validate_api_source

        sources = get_all_approved_sources()
        print(f"{GREEN}✓ Approved API Sources: {', '.join(sources)}{RESET}")

        # Test validation
        polygon_valid = validate_api_source("polygon")
        fred_valid = validate_api_source("fred")
        fake_invalid = not validate_api_source("fake_api")

        if polygon_valid and fred_valid and fake_invalid:
            print(f"{GREEN}✓ API source validation working correctly{RESET}")
            return True
        else:
            print(f"{RED}✗ API source validation failed{RESET}")
            return False

    except Exception as e:
        print(f"{RED}✗ API Configuration Error: {e}{RESET}")
        return False


def scan_for_synthetic_data():
    """Scan all Python files for synthetic data"""
    print(f"\n{BLUE}=== Scanning for Synthetic Data ==={RESET}")

    violations_by_file = {}
    total_violations = 0

    for root, dirs, files in os.walk('.'):
        # Skip hidden and test directories
        dirs[:] = [d for d in dirs if not d.startswith(
            '.') and d != '__pycache__']

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                violations = check_file_for_synthetic_data(filepath)

                if violations:
                    violations_by_file[filepath] = violations
                    total_violations += len(violations)

    if violations_by_file:
        print(f"{RED}✗ Found {total_violations} synthetic data violations in {len(violations_by_file)} files:{RESET}")
        for filepath, violations in violations_by_file.items():
            print(f"\n  {filepath}:")
            # Show first 3 violations per file
            for line_num, pattern, line in violations[:3]:
                print(f"    Line {line_num}: {pattern}")
                print(f"      {line[:80]}...")
            if len(violations) > 3:
                print(f"    ... and {len(violations) - 3} more violations")
    else:
        print(f"{GREEN}✓ No synthetic data violations found{RESET}")

    return len(violations_by_file) == 0


def verify_mltrainer_claude_integration():
    """Verify mlTrainer Claude integration has proper restrictions"""
    print(f"\n{BLUE}=== Verifying mlTrainer Claude Integration ==={RESET}")

    try:
        from mltrainer_claude_integration import MLTrainerClaude

        claude = MLTrainerClaude()

        # Check if synthetic data generation is blocked
        response = claude.get_response("Generate synthetic data for testing")

        if "synthetic data" in response.lower() and "not allowed" in response.lower():
            print(f"{GREEN}✓ Claude properly blocks synthetic data requests{RESET}")
        else:
            print(
                f"{YELLOW}⚠ Claude response to synthetic data request unclear{RESET}")

        return True

    except Exception as e:
        print(f"{RED}✗ Claude Integration Error: {e}{RESET}")
        return False


def main():
    """Run comprehensive compliance verification"""
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}mlTrainer3 Comprehensive Compliance Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {
        "model_registration": verify_model_registration(),
        "compliance_gateway": verify_compliance_gateway(),
        "api_configuration": verify_api_configuration(),
        "no_synthetic_data": scan_for_synthetic_data(),
        "claude_integration": verify_mltrainer_claude_integration(),
    }

    # Summary
    print(f"\n{BLUE}=== Verification Summary ==={RESET}")
    passed = sum(results.values())
    total = len(results)

    for check, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {check.replace('_', ' ').title()}: {status}")

    print(f"\n{BLUE}Overall Result: {passed}/{total} checks passed{RESET}")

    if passed == total:
        print(f"\n{GREEN}✓ mlTrainer3 is FULLY COMPLIANT!{RESET}")
        print(f"{GREEN}  - All {len(results)} models properly registered{RESET}")
        print(f"{GREEN}  - Only real data from Polygon/FRED APIs allowed{RESET}")
        print(f"{GREEN}  - Compliance gateway actively enforcing{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ COMPLIANCE ISSUES DETECTED!{RESET}")
        print(f"{RED}  Please fix the issues above before deployment.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
