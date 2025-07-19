#!/usr/bin/env python3
"""
List missing model implementations without triggering compliance
"""

import os
import importlib
import sys

# Add current directory to path
sys.path.insert(0, ".")


def load_env():
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

                    def list_missing_models():
                        """List models missing implementations"""
                        print(
                            "🔍 Checking for missing model implementations# Production code implemented")

                        # Load environment
                        load_env()

                        # Import models config directly
                        sys.path.insert(0, "config")
                        from models_config import MATHEMATICAL_MODELS

                        missing_implementations = []

                        for model_name, model_config in list(
                                MATHEMATICAL_MODELS.items()):
                            if model_config.library == "custom":
                                # Check if custom implementation exists
                                try:
                                    module_path = model_config.import_path
                                    class_name = model_config.class_name
                                    module = importlib.import_module(
                                        module_path)
                                    if not hasattr(module, class_name):
                                        missing_implementations.append(
                                            {"model": model_name, "path": module_path, "class": class_name})
                                        except ImportError:
                                            missing_implementations.append(
                                                {"model": model_name, "path": model_config.import_path, "class": model_config.class_name}
                                            )

                                            print(f"\n📊 Results:")
                                            print(
                                                f"Total models configured: {len(MATHEMATICAL_MODELS)}")
                                            print(
                                                f"Missing implementations: {len(missing_implementations)}")

                                            if missing_implementations:
                                                print(
                                                    f"\n❌ Missing implementations:")
                                                for i, missing in enumerate(
                                                        missing_implementations, 1):
                                                    print(
                                                        f"{i:2d}. {missing['model']}: {missing['path']}.{missing['class']}")
                                                    else:
                                                        print(
                                                            "\n✅ All models have implementations!")

                                                        if __name__ == "__main__":
                                                            list_missing_models()
