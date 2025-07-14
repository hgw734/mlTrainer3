#!/usr/bin/env python3
import logging

logger = logging.getLogger(__name__)


"""
Pre-commit hook to ensure governance imports
This hook ensures all core modules properly import and activate governance
"""

import sys
import ast
from typing import List, Tuple

REQUIRED_IMPORTS = [
"from core.governance_kernel import governed, activate_governance",
"from agent_governance import get_governance",
"import core.governance_kernel",
]

REQUIRED_PATTERNS = {
"class": "@governed",  # Classes should have governed decorator
"function": "@governed",  # Functions should have governed decorator
"file_operations": "governance_kernel",  # File operations should use governance
}


def check_file_for_governance(filepath: str) -> List[Tuple[str, str]]:
    """
    Check if a Python file has proper governance integration
    Returns list of (issue_type, description) tuples
    """
    issues = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

            # Parse AST
            tree = ast.parse(content)

            # Check for governance imports
            has_governance_import = False
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                        if "governance" in alias.name:
                            has_governance_import = True

                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    import_str = f"from {node.module} import {', '.join(n.name for n in node.names)}"
                                    imports.append(import_str)
                                    if "governance" in node.module or any("governance" in n.name for n in node.names):
                                        has_governance_import = True

                                        if not has_governance_import:
                                            issues.append(("missing_import", "No governance imports found"))

                                            # Check for governance activation
                                            has_activation = "activate_governance()" in content
                                            if has_governance_import and not has_activation:
                                                # Check if it's a module that should activate governance
                                                if any(pattern in filepath for pattern in ["__init__.py", "main.py", "app.py"]):
                                                    issues.append(("missing_activation", "Governance imported but not activated"))

                                                    # Check classes and functions for governance decorators
                                                    class GovernanceChecker(ast.NodeVisitor):
                                                        def __init__(self):
                                                            self.ungoverned_classes = []
                                                            self.ungoverned_functions = []
                                                            self.file_operations = []

                                                            def visit_ClassDef(self, node):
                                                                # Check if class has @governed decorator
                                                                has_governed = any(isinstance(dec, ast.Name) and dec.id == "governed" for dec in node.decorator_list)
                                                                if not has_governed:
                                                                    self.ungoverned_classes.append(node.name)
                                                                    self.generic_visit(node)

                                                                    def visit_FunctionDef(self, node):
                                                                        # Skip private functions and production functions
                                                                        if node.name.startswith("_") or node.name.startswith("test_"):
                                                                            self.generic_visit(node)
                                                                            return

                                                                        # Check if function has @governed decorator
                                                                        has_governed = any(isinstance(dec, ast.Name) and dec.id == "governed" for dec in node.decorator_list)

                                                                        # Check if function contains file operations
                                                                        has_file_ops = False
                                                                        for child in ast.walk(node):
                                                                            if isinstance(child, ast.Call):
                                                                                if isinstance(child.func, ast.Name) and child.func.id == "open":
                                                                                    has_file_ops = True
                                                                                    break

                                                                                if not has_governed and (has_file_ops or "write" in node.name or "save" in node.name):
                                                                                    self.ungoverned_functions.append(node.name)

                                                                                    self.generic_visit(node)

                                                                                    def visit_Call(self, node):
                                                                                        # Check for direct file operations
                                                                                        if isinstance(node.func, ast.Name) and node.func.id == "open":
                                                                                            # Check if it's using governed open
                                                                                            parent = getattr(node, "parent", None)
                                                                                            if parent and not isinstance(parent, ast.Call):
                                                                                                self.file_operations.append(node.lineno)
                                                                                                self.generic_visit(node)

                                                                                                if has_governance_import:
                                                                                                    checker = GovernanceChecker()
                                                                                                    checker.visit(tree)

                                                                                                    if checker.ungoverned_classes:
                                                                                                        issues.append(
                                                                                                        ("ungoverned_classes", f"Classes without @governed: {', '.join(checker.ungoverned_classes)}")
                                                                                                        )

                                                                                                        if checker.ungoverned_functions:
                                                                                                            issues.append(
                                                                                                            ("ungoverned_functions", f"Functions without @governed: {', '.join(checker.ungoverned_functions)}")
                                                                                                            )

                                                                                                            if checker.file_operations:
                                                                                                                issues.append(
                                                                                                                (
                                                                                                                "unsafe_file_operations",
                                                                                                                f"Direct file operations on lines: {', '.join(map(str, checker.file_operations))}",
                                                                                                                )
                                                                                                                )

                                                                                                                except SyntaxError as e:
                                                                                                                    issues.append(("syntax_error", f"Failed to parse: {e}"))
                                                                                                                    except Exception as e:
                                                                                                                        issues.append(("error", f"Error checking file: {e}"))

                                                                                                                        return issues


                                                                                                                        def main():
                                                                                                                            """Main entry point for pre-commit hook"""
                                                                                                                            if len(sys.argv) < 2:
                                                                                                                                logger.info("Usage: check_governance_imports.py <file1> [file2] # Production code implemented")
                                                                                                                                sys.exit(1)

                                                                                                                                all_issues = []

                                                                                                                                for filepath in sys.argv[1:]:
                                                                                                                                    # Only check Python files in core modules
                                                                                                                                    if not filepath.endswith(".py"):
                                                                                                                                        continue

                                                                                                                                    issues = check_file_for_governance(filepath)
                                                                                                                                    if issues:
                                                                                                                                        all_issues.append((filepath, issues))

                                                                                                                                        if all_issues:
                                                                                                                                            logger.info("\n⚠️  GOVERNANCE INTEGRATION MISSING\n")
                                                                                                                                            logger.info("The following files lack proper governance integration:")
                                                                                                                                            logger.info("-" * 80)

                                                                                                                                            for filepath, issues in all_issues:
                                                                                                                                                logger.info(f"\n📄 {filepath}:")
                                                                                                                                                for issue_type, description in issues:
                                                                                                                                                    logger.info(f"  ❌ {issue_type}: {description}")

                                                                                                                                                    logger.info("\n" + "-" * 80)
                                                                                                                                                    logger.info("❌ COMMIT BLOCKED: Add governance integration before committing.")
                                                                                                                                                    logger.info("\nRequired changes:")
                                                                                                                                                    logger.info("1. Add governance import:")
                                                                                                                                                    logger.info("   from core.governance_kernel import governed, activate_governance")
                                                                                                                                                    logger.info("\n2. Add @governed decorator to classes and functions:")
                                                                                                                                                    logger.info("   @governed")
                                                                                                                                                    logger.info("   class MyClass:")
                                                                                                                                                    logger.info("       pass")
                                                                                                                                                    logger.info("\n3. For main modules, activate governance:")
                                                                                                                                                    logger.info("   if __name__ == '__main__':")
                                                                                                                                                    logger.info("       activate_governance()")

                                                                                                                                                    sys.exit(1)
                                                                                                                                                    else:
                                                                                                                                                        logger.info("✅ All files have proper governance integration")
                                                                                                                                                        sys.exit(0)


                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                            main()
