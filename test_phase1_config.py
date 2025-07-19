#!/usr/bin/env python3
"""
Phase 1 Tests: Configuration System
Ensure no hardcoded values and proper config loading
"""
import os
import re
import pytest
from pathlib import Path


def test_no_hardcoded_api_keys():
    """Ensure no hardcoded API keys in codebase"""
    # Patterns that indicate hardcoded keys
    hardcoded_patterns = [
        r'api_key\s*=\s*["\'][a-zA-Z0-9]{10,}["\']',  # Direct assignment
        # Default values
        r'os\.getenv\(["\'][^"\']+["\']\s*,\s*["\'][a-zA-Z0-9]{10,}["\']',
        r'API_KEY\s*=\s*["\'][a-zA-Z0-9]{10,}["\']',
        r'token\s*=\s*["\'][a-zA-Z0-9]{10,}["\']',
    ]

    # Files to check
    code_files = []
    for ext in [".py", ".yaml", ".yml", ".json"]:
        code_files.extend(Path(".").rglob(f"*{ext}"))

        violations = []
        for file_path in code_files:
            # Skip test files and virtual environments
            if "test_" in str(file_path) or "venv" in str(
                    file_path) or "__pycache__" in str(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    for pattern in hardcoded_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            violations.append(
                                {"file": str(file_path), "pattern": pattern, "matches": matches})
                            except Exception:
                                # Skip files that can't be read
                                pass

                            assert len(
                                violations) == 0, f"Found hardcoded values: {violations}"

                            def test_config_loader_initialization():
                                """Test that config loader initializes properly"""
                                # Set required environment variables for
                                # testing
                                os.environ["POLYGON_API_KEY"] = "test_polygon_key"
                                os.environ["FRED_API_KEY"] = "test_fred_key"

                                try:
                                    from config.config_loader import get_config_loader

                                    loader = get_config_loader()

                                    # Should not raise any exceptions
                                    assert loader is not None
                                    assert loader.env in [
                                        "development", "staging", "production"]
                                    except Exception as e:
                                        pytest.fail(
                                            f"Config loader failed to initialize: {e}")

                                        def test_config_requires_env_vars():
                                            """Test that missing env vars raise proper errors"""
                                            # Remove env vars
                                            os.environ.pop(
                                                "POLYGON_API_KEY", None)
                                            os.environ.pop(
                                                "FRED_API_KEY", None)

                                            from config.config_loader import ConfigLoader, ConfigurationError

                                            with pytest.raises(ConfigurationError) as exc_info:
                                                ConfigLoader()

                                                assert "environment variable required" in str(
                                                    exc_info.value)

                                                def test_config_get_values():
                                                    """Test getting config values by path"""
                                                    os.environ["POLYGON_API_KEY"] = "test_polygon_key"
                                                    os.environ["FRED_API_KEY"] = "test_fred_key"

                                                    from config.config_loader import get_config

                                                    # Test getting nested
                                                    # values
                                                    rate_limit = get_config(
                                                        "api.polygon.rate_limit")
                                                    assert rate_limit == 100  # Default value

                                                    # Test getting with default
                                                    missing = get_config(
                                                        "api.polygon.missing_key", "default_value")
                                                    assert missing == "default_value"

                                                    def test_config_compliance_validation():
                                                        """Test that config values are validated against compliance"""
                                                        os.environ["POLYGON_API_KEY"] = "test_polygon_key"
                                                        os.environ["FRED_API_KEY"] = "test_fred_key"

                                                        from config.config_loader import get_config_loader

                                                        loader = get_config_loader()

                                                        # Test validation method
                                                        # Should accept valid
                                                        # values
                                                        assert loader.validate_against_compliance(
                                                            "models.random_forest.n_estimators", 100)

                                                        # Should reject based on compliance rules
                                                        # (This assumes compliance rules exist for the path)
                                                        # The actual test
                                                        # depends on your
                                                        # compliance
                                                        # configuration

                                                        def test_all_configs_accessible():
                                                            """Test that all major config sections are accessible"""
                                                            os.environ["POLYGON_API_KEY"] = "test_polygon_key"
                                                            os.environ["FRED_API_KEY"] = "test_fred_key"

                                                            from config.config_loader import get_config

                                                            # These should all
                                                            # return non-None
                                                            # values
                                                            assert get_config(
                                                                "api") is not None
                                                            assert get_config(
                                                                "models") is not None
                                                            assert get_config(
                                                                "ai") is not None
                                                            assert get_config(
                                                                "compliance") is not None

                                                            if __name__ == "__main__":
                                                                # Run tests
                                                                print(
                                                                    "🧪 Testing Phase 1: Configuration System")
                                                                print(
                                                                    ("=" * 50))

                                                                # Set test
                                                                # environment
                                                                os.environ["MLTRAINER_ENV"] = "development"

                                                                tests = [
                                                                    test_no_hardcoded_api_keys,
                                                                    test_config_loader_initialization,
                                                                    test_config_requires_env_vars,
                                                                    test_config_get_values,
                                                                    test_config_compliance_validation,
                                                                    test_all_configs_accessible,
                                                                ]

                                                                passed = 0
                                                                failed = 0

                                                                for test in tests:
                                                                    try:
                                                                        test()
                                                                        print(
                                                                            f"✅ {test.__name__}")
                                                                        passed += 1
                                                                        except AssertionError as e:
                                                                            print(
                                                                                f"❌ {test.__name__}: {e}")
                                                                            failed += 1
                                                                            except Exception as e:
                                                                                print(
                                                                                    f"💥 {test.__name__}: {e}")
                                                                                failed += 1

                                                                                print(
                                                                                    f"\n📊 Results: {passed} passed, {failed} failed")

                                                                                if failed > 0:
                                                                                    exit(
                                                                                        1)
