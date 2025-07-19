#!/usr/bin/env python3
"""
Script to fix indentation errors in Python files
Fixes the common pattern of _deterministic_* methods being incorrectly nested
"""

import os
import re
import glob


def fix_file_indentation(filepath):
    """Fix indentation errors in a Python file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

            # Fix the common pattern of _deterministic_* methods being nested inside other methods
            # This pattern appears in many files where methods are incorrectly
            # indented

            # Pattern 1: Fix methods that are indented too much (nested inside
            # other methods)
            lines = content.split("\n")
            fixed_lines = []
            in_class = False
            class_indent = 0

            for i, line in enumerate(lines):
                stripped = line.strip()

                # Detect class definition
                if stripped.startswith("class ") and ":" in stripped:
                    in_class = True
                    class_indent = len(line) - len(line.lstrip())
                    fixed_lines.append(line)
                    continue

                # Detect method definitions inside classes
                if in_class and stripped.startswith("def _deterministic_"):
                    # This method should be at class level, not nested
                    # Calculate proper indentation (class level + 4 spaces)
                    proper_indent = " " * (class_indent + 4)
                    fixed_lines.append(proper_indent + stripped)
                    continue

                # Detect enum values that should be at class level
                if in_class and re.match(r"^[A-Z_]+ = ", stripped):
                    proper_indent = " " * (class_indent + 4)
                    fixed_lines.append(proper_indent + stripped)
                    continue

                # Detect dataclass fields that should be at class level
                if in_class and re.match(r"^[a-z_]+:", stripped):
                    proper_indent = " " * (class_indent + 4)
                    fixed_lines.append(proper_indent + stripped)
                    continue

                # Handle other lines normally
                fixed_lines.append(line)

                # Write the fixed content back
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("\n".join(fixed_lines))

                    print(f"✅ Fixed indentation in {filepath}")
                    return True

                    except Exception as e:
                        print(f"❌ Error fixing {filepath}: {e}")
                        return False

                        def main():
                            """Fix all files with indentation errors"""

                            # List of files that failed to format (from the
                            # error output)
                            files_to_fix = [
                                "app.py",
                                "compliance_status_summary.py",
                                "config/immutable_compliance_gateway.py",
                                "core/governance_kernel.py",
                                "custom/adversarial.py",
                                "custom/alternative_data.py",
                                "custom/automl.py",
                                "custom/adaptive.py",
                                "custom/binomial.py",
                                "custom/complexity.py",
                                "custom/detectors.py",
                                "custom/elliott_wave.py",
                                "custom/financial_models.py",
                                "custom/ensemble.py",
                                "custom/indicators.py",
                                "custom/fractal.py",
                                "custom/macro.py",
                                "custom/interest_rate.py",
                                "custom/momentum_models.py",
                                "custom/momentum.py",
                                "custom/meta_learning.py",
                                "custom/microstructure.py",
                                "custom/optimization.py",
                                "custom/nonlinear.py",
                                "custom/pairs.py",
                                "custom/regime_detection.py",
                                "custom/patterns.py",
                                "custom/regime_ensemble.py",
                                "custom/position_sizing.py",
                                "custom/risk.py",
                                "custom/risk_management.py",
                                "custom/rl.py",
                                "custom/stress.py",
                                "custom/time_series.py",
                                "custom/volatility.py",
                                "custom/systems.py",
                                "custom/volume.py",
                                "example_governed_agent.py",
                                "drift_protection.py",
                                "scripts/comprehensive_audit_v2.py",
                                "scripts/production_audit_final.py",
                            ]

                            print("🔧 Fixing indentation errors in Python files...")
                            print(("=" * 60))

                            fixed_count = 0
                            total_count = len(files_to_fix)

                            for filepath in files_to_fix:
                                if os.path.exists(filepath):
                                    if fix_file_indentation(filepath):
                                        fixed_count += 1
                                        else:
                                            print(
                                                f"⚠️  File not found: {filepath}")

                                            print(("=" * 60))
                                            print(
                                                f"✅ Fixed {fixed_count}/{total_count} files")

                                            # Now try to format with black
                                            print(
                                                "\n🎨 Running Black formatter...")
                                            import subprocess

                                            try:
                                                result = subprocess.run(
                                                    ["python3", "-m", "black", ".", "--line-length", "120"], capture_output=True, text=True)
                                                if result.returncode == 0:
                                                    print(
                                                        "✅ Black formatting completed successfully!")
                                                    else:
                                                        print(
                                                            "⚠️  Some files still have issues:")
                                                        print((result.stderr))
                                                        except Exception as e:
                                                            print(
                                                                f"❌ Error running Black: {e}")

                                                            if __name__ == "__main__":
                                                                main()
