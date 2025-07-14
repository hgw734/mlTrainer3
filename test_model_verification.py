#!/usr/bin/env python3
"""
Direct test of model implementation verification logic
"""

import importlib
import os

# Set dummy API keys
os.environ["POLYGON_API_KEY"] = "dummy"
os.environ["FRED_API_KEY"] = "dummy"

print("🔍 Testing Model Implementation Verification...")

# Load model registry
try:
    # We'll parse the models_config.py file to extract model info without importing numpy
    with open("config/models_config.py", "r") as f:
        content = f.read()

        # Count custom models by looking for library="custom"
        custom_count = content.count('library="custom"')
        print(f"📊 Found {custom_count} models configured with library='custom'")

        # Try to import custom modules
        missing_modules = []
        found_modules = []

        for module_name in [
        "indicators",
        "patterns",
        "volume",
        "risk",
        "volatility",
        "systems",
        "fractal",
        "complexity",
        "nonlinear",
        "position_sizing",
        "automl",
        "adversarial",
        "detectors",
        "momentum",
        ]:
            try:
                module = importlib.import_module(f"custom.{module_name}")
                found_modules.append(module_name)
                except ImportError as e:
                    missing_modules.append((module_name, str(e)))

                    print(f"\n✅ Found {len(found_modules)} custom implementation modules:")
                    for module in found_modules:
                        print(f"   - custom.{module}")

                        if missing_modules:
                            print(f"\n❌ Missing {len(missing_modules)} implementation modules:")
                            for module, error in missing_modules:
                                print(f"   - custom.{module}: {error}")

                                # Check specific model classes
                                print("\n🔍 Checking specific model implementations...")

                                test_cases = [
                                ("custom.indicators", "EMA"),
                                ("custom.patterns", "BreakoutDetection"),
                                ("custom.volume", "OBV"),
                                ("custom.risk", "VAR"),
                                ("custom.volatility", "RegimeSwitchingVolatility"),
                                ("custom.systems", "TurtleTrading"),
                                ("custom.fractal", "HurstExponent"),
                                ("custom.complexity", "ApproximateEntropy"),
                                ("custom.nonlinear", "RecurrenceAnalysis"),
                                ("custom.position_sizing", "KellyCriterion"),
                                ]

                                implementations_found = 0
                                implementations_missing = 0

                                for module_path, class_name in test_cases:
                                    try:
                                        module = importlib.import_module(module_path)
                                        if hasattr(module, class_name):
                                            print(f"✅ Found: {module_path}.{class_name}")
                                            implementations_found += 1
                                            else:
                                                print(f"❌ Missing: {module_path}.{class_name}")
                                                implementations_missing += 1
                                                except ImportError:
                                                    print(f"❌ Module not found: {module_path}")
                                                    implementations_missing += 1

                                                    print(f"\n📊 Summary:")
                                                    print(f"   - Implementation modules found: {len(found_modules)}")
                                                    print(f"   - Implementation modules missing: {len(missing_modules)}")
                                                    print(f"   - Model classes found: {implementations_found}")
                                                    print(f"   - Model classes missing: {implementations_missing}")

                                                    if missing_modules or implementations_missing > 0:
                                                        print("\n⚠️  COMPLIANCE WARNING: Missing model implementations detected!")
                                                        print("The compliance system should prevent the trading system from starting.")
                                                        else:
                                                            print("\n✅ All checked implementations are present!")
                                                            print("Note: This is not a complete check - the actual compliance system does more.")

                                                            except Exception as e:
                                                                print(f"❌ Error during verification: {e}")
                                                                import traceback

                                                                traceback.print_exc()
