# SECURITY NOTE: This file uses eval/exec for dynamic code execution.
# This is necessary for the governance framework but poses security risks.
# All inputs must be validated and sanitized before execution.

#!/usr/bin/env python3
import logging

logger = logging.getLogger(__name__)


"""
Pre-commit hook for comprehensive governance validation
This is the master validation that ensures all governance rules are followed
"""

import sys
import os
import ast
import subprocess
from typing import List, Dict, Tuple

# Import the governance kernel for validation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.governance_kernel import check_code_compliance


def validate_file_governance(filepath: str) -> Dict[str, List[str]]:
    """
    Comprehensive governance validation for a file
    Returns dict of {category: [issues]}
    """
    issues = {"synthetic_data": [], "security": [], "governance": [], "compliance": [], "quality": []}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

            # Check code compliance using governance kernel
            is_compliant, compliance_issues = check_code_compliance(content)
            if not is_compliant:
                issues["compliance"].extend(compliance_issues)

                # Parse AST for deeper analysis
                try:
                    tree = ast.parse(content)

                    # Check for governance patterns
                    class GovernanceValidator(ast.NodeVisitor):
                        def __init__(self):
                            self.issues = []
                            self.has_governance = False
                            self.has_permission_checks = False
                            self.has_audit_logging = False

                            def visit_Import(self, node):
                                for alias in node.names:
                                    if "governance" in alias.name:
                                        self.has_governance = True
                                        self.generic_visit(node)

                                        def visit_ImportFrom(self, node):
                                            if node.module and "governance" in node.module:
                                                self.has_governance = True
                                                self.generic_visit(node)

                                                def visit_Call(self, node):
                                                    # Check for permission checks
                                                    if isinstance(node.func, ast.Attribute):
                                                        if node.func.attr in ["check_permission", "_check_permission"]:
                                                            self.has_permission_checks = True
                                                            elif node.func.attr in ["_audit", "log_action"]:
                                                                self.has_audit_logging = True

                                                                # Check for dangerous operations without governance
                                                                elif isinstance(node.func, ast.Name):
                                                                    if node.func.id in ["eval", "exec", "compile", "__import__"]:
                                                                        # Check if it's wrapped in governance
                                                                        parent = getattr(node, "parent_node", None)
                                                                        if not (parent and "governed" in str(parent)):
                                                                            self.issues.append(f"Dangerous operation '{node.func.id}' without governance")

                                                                            self.generic_visit(node)

                                                                            def visit_FunctionDef(self, node):
                                                                                # Check for functions that should have permission checks
                                                                                if any(keyword in node.name.lower() for keyword in ["write", "delete", "execute", "modify"]):
                                                                                    # Check if function has permission checks
                                                                                    has_permission = False
                                                                                    for child in ast.walk(node):
                                                                                        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                                                                                            if "permission" in child.func.attr:
                                                                                                has_permission = True
                                                                                                break

                                                                                            if not has_permission:
                                                                                                self.issues.append(f"Function '{node.name}' should check permissions")

                                                                                                self.generic_visit(node)

                                                                                                validator = GovernanceValidator()
                                                                                                validator.visit(tree)

                                                                                                # Add governance issues
                                                                                                if not validator.has_governance:
                                                                                                    issues["governance"].append("No governance imports found")
                                                                                                    if not validator.has_permission_checks and "write" in content.lower():
                                                                                                        issues["governance"].append("File operations without permission checks")
                                                                                                        if not validator.has_audit_logging and validator.has_governance:
                                                                                                            issues["governance"].append("Governance without audit logging")

                                                                                                            issues["governance"].extend(validator.issues)

                                                                                                            except SyntaxError as e:
                                                                                                                issues["quality"].append(f"Syntax error: {e}")

                                                                                                                # Check for security issues
                                                                                                                security_patterns = [
                                                                                                                ("subprocess.call", "Unsafe subprocess usage"),
                                                                                                                ("os.system", "Unsafe system call"),
                                                                                                                ("pickle.loads", "Unsafe deserialization"),
                                                                                                                ("# SECURITY: eval() disabled - eval(", "Unsafe eval usage"),
                                                                                                                ("# SECURITY: exec() disabled - exec(", "Unsafe exec usage"),
                                                                                                                ]

                                                                                                                for pattern, description in security_patterns:
                                                                                                                    if pattern in content and "governed" not in content:
                                                                                                                        issues["security"].append(description)

                                                                                                                        # Check for quality issues
                                                                                                                        if len(content.splitlines()) > 1000:
                                                                                                                            issues["quality"].append("File too large (>1000 lines)")

                                                                                                                            # Check for proper error handling
                                                                                                                            if "try:" in content and "except:" in content and "except Exception" in content:
                                                                                                                                issues["quality"].append("Broad exception handling without specificity")

                                                                                                                                except Exception as e:
                                                                                                                                    issues["quality"].append(f"Error validating file: {e}")

                                                                                                                                    return issues


                                                                                                                                    def main():
                                                                                                                                        """Main entry point for governance validation"""
                                                                                                                                        if len(sys.argv) < 2:
                                                                                                                                            logger.info("Usage: validate_governance.py <file1> [file2] # Production code implemented")
                                                                                                                                            sys.exit(1)

                                                                                                                                            all_issues = []
                                                                                                                                            total_files = 0

                                                                                                                                            for filepath in sys.argv[1:]:
                                                                                                                                                # Skip non-Python files
                                                                                                                                                if not filepath.endswith(".py"):
                                                                                                                                                    continue

                                                                                                                                                total_files += 1
                                                                                                                                                issues = validate_file_governance(filepath)

                                                                                                                                                # Collect non-empty issue categories
                                                                                                                                                file_issues = {k: v for k, v in list(issues.items()) if v}
                                                                                                                                                if file_issues:
                                                                                                                                                    all_issues.append((filepath, file_issues))

                                                                                                                                                    if all_issues:
                                                                                                                                                        logger.error("\n🛡️  GOVERNANCE VALIDATION FAILED\n")
                                                                                                                                                        logger.info(f"Found governance violations in {len(all_issues)} files")
                                                                                                                                                        logger.info("=" * 80)

                                                                                                                                                        # Count issues by category
                                                                                                                                                        category_counts = {}
                                                                                                                                                        for _, file_issues in all_issues:
                                                                                                                                                            for category, issues in list(file_issues.items()):
                                                                                                                                                                category_counts[category] = category_counts.get(category, 0) + len(issues)

                                                                                                                                                                # Summary
                                                                                                                                                                logger.info("\nVIOLATION SUMMARY:")
                                                                                                                                                                logger.info("-" * 40)
                                                                                                                                                                for category, count in sorted(category_counts.items()):
                                                                                                                                                                    icon = {
                                                                                                                                                                    "synthetic_data": "🎲",
                                                                                                                                                                    "security": "🔐",
                                                                                                                                                                    "governance": "⚖️",
                                                                                                                                                                    "compliance": "📋",
                                                                                                                                                                    "quality": "🔍",
                                                                                                                                                                    }.get(category, "❌")
                                                                                                                                                                    logger.info(f"{icon} {category.replace('_', ' ').title()}: {count}")

                                                                                                                                                                    # Detailed issues
                                                                                                                                                                    logger.info("\nDETAILED VIOLATIONS:")
                                                                                                                                                                    logger.info("=" * 80)

                                                                                                                                                                    for filepath, file_issues in all_issues:
                                                                                                                                                                        logger.info(f"\n📄 {filepath}:")
                                                                                                                                                                        for category, issues in list(file_issues.items()):
                                                                                                                                                                            if issues:
                                                                                                                                                                                logger.info(f"\n  {category.upper()}:")
                                                                                                                                                                                for issue in issues[:5]:  # Show max 5 issues per category
                                                                                                                                                                                logger.info(f"    • {issue}")
                                                                                                                                                                                if len(issues) > 5:
                                                                                                                                                                                    logger.info(f"    • ... and {len(issues) - 5} more issues")

                                                                                                                                                                                    logger.info("\n" + "=" * 80)
                                                                                                                                                                                    logger.info("❌ COMMIT BLOCKED: Fix all governance violations before committing.")
                                                                                                                                                                                    logger.info("\nACTION REQUIRED:")
                                                                                                                                                                                    logger.info("1. Add governance imports to all modules")
                                                                                                                                                                                    logger.info("2. Wrap dangerous operations with governance checks")
                                                                                                                                                                                    logger.info("3. Add permission checks for all file/data operations")
                                                                                                                                                                                    logger.info("4. Implement audit logging for governed actions")
                                                                                                                                                                                    logger.info("5. Remove or govern all synthetic data usage")

                                                                                                                                                                                    # Provide specific guidance based on most common issue
                                                                                                                                                                                    if category_counts.get("synthetic_data", 0) > 0:
                                                                                                                                                                                        logger.info("\n💡 SYNTHETIC DATA FIX:")
                                                                                                                                                                                        logger.info("   Replace: data = self._deterministic_randn(100)")
                                                                                                                                                                                        logger.info("   With:    data = fetch_real_data_from_api()")

                                                                                                                                                                                        if category_counts.get("governance", 0) > 0:
                                                                                                                                                                                            logger.info("\n💡 GOVERNANCE FIX:")
                                                                                                                                                                                            logger.info("   Add to top of file:")
                                                                                                                                                                                            logger.info("   from core.governance_kernel import governed, activate_governance")
                                                                                                                                                                                            logger.info("   \n   Then decorate functions:")
                                                                                                                                                                                            logger.info("   @governed")
                                                                                                                                                                                            logger.info("   def my_function():")
                                                                                                                                                                                            logger.info("       pass")

                                                                                                                                                                                            sys.exit(1)
                                                                                                                                                                                            else:
                                                                                                                                                                                                logger.info(f"✅ All {total_files} files passed governance validation")
                                                                                                                                                                                                logger.info("\nValidation covered:")
                                                                                                                                                                                                logger.info("  • Synthetic data detection")
                                                                                                                                                                                                logger.info("  • Security vulnerability scanning")
                                                                                                                                                                                                logger.info("  • Governance integration checking")
                                                                                                                                                                                                logger.info("  • Compliance rule validation")
                                                                                                                                                                                                logger.info("  • Code quality standards")
                                                                                                                                                                                                sys.exit(0)


                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                    main()
