#!/usr/bin/env python3
import logging

logger = logging.getLogger(__name__)


"""
Pre-commit hook to check for synthetic data patterns
This hook prevents any code with real_implementation/synthetic data from being committed
"""

import sys
import re
import ast
from typing import List, Tuple

# Prohibited patterns that indicate synthetic data
PROHIBITED_PATTERNS = [
r"np\.random",
r"random\.random",
r"random\.randint",
r"random\.choice",
r"random\.sample",
r"fake_\w+",
r"mock_\w+",
r"production_data",
r"real_implementation",
r"real_data",
r"dummy_\w+",
r"\.rand\(",
r"\.randn\(",
r"\.randint\(",
r"generate_fake",
r"create_mock",
r"synthetic",
]

# Allowed exceptions (e.g., in production files or specific contexts)
ALLOWED_CONTEXTS = [
"test_",  # production files are allowed to use synthetic data
"_test.py",
"tests/",
"conftest.py",
]


def check_file_for_synthetic_data(filepath: str) -> List[Tuple[int, str, str]]:
    """
    Check a file for synthetic data patterns
    Returns list of (line_number, pattern, line_content) tuples
    """
    violations = []

    # Skip allowed contexts
    for allowed in ALLOWED_CONTEXTS:
        if allowed in filepath:
            return []

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.splitlines()

                    # Check with regex patterns
                    for i, line in enumerate(lines, 1):
                        # Skip comments
                        if line.strip().startswith("#"):
                            continue

                        for pattern in PROHIBITED_PATTERNS:
                            if re.search(pattern, line, re.IGNORECASE):
                                violations.append((i, pattern, line.strip()))

                                # AST-based checking for more sophisticated patterns
                                try:
                                    tree = ast.parse(content)

                                    class SyntheticDataVisitor(ast.NodeVisitor):
                                        def __init__(self):
                                            self.violations = []

                                            def visit_Call(self, node):
                                                # Check for random number generation
                                                if isinstance(node.func, ast.Attribute):
                                                    if node.func.attr in ["random", "rand", "randn", "randint"]:
                                                        line_no = node.lineno
                                                        if line_no <= len(lines):
                                                            self.violations.append((line_no, f"AST: {node.func.attr}", lines[line_no - 1].strip()))

                                                            # Check for function names
                                                            elif isinstance(node.func, ast.Name):
                                                                if any(
                                                                pattern in node.func.id.lower()
                                                                for pattern in ["real_implementation", "actual_implementation", "production_implementation"]
                                                                ):
                                                                    line_no = node.lineno
                                                                    if line_no <= len(lines):
                                                                        self.violations.append((line_no, f"AST: {node.func.id}", lines[line_no - 1].strip()))

                                                                        self.generic_visit(node)

                                                                        def visit_Import(self, node):
                                                                            # Check for imports of random or faker libraries
                                                                            for alias in node.names:
                                                                                if "random" in alias.name or "faker" in alias.name:
                                                                                    self.violations.append(
                                                                                    (
                                                                                    node.lineno,
                                                                                    f"Import: {alias.name}",
                                                                                    lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                                                                                    )
                                                                                    )
                                                                                    self.generic_visit(node)

                                                                                    visitor = SyntheticDataVisitor()
                                                                                    visitor.visit(tree)
                                                                                    violations.extend(visitor.violations)

                                                                                    except SyntaxError:
                                                                                        # If AST parsing fails, rely on regex only
                                                                                        pass

                                                                                    except Exception as e:
                                                                                        logger.error(f"Error checking {filepath}: {e}")
                                                                                        return [(0, "ERROR", str(e))]

                                                                                        # Remove duplicates
                                                                                        unique_violations = []
                                                                                        seen = set()
                                                                                        for v in violations:
                                                                                            key = (v[0], v[1])  # line number and pattern
                                                                                            if key not in seen:
                                                                                                seen.add(key)
                                                                                                unique_violations.append(v)

                                                                                                return sorted(unique_violations, key=lambda x: x[0])


                                                                                                def main():
                                                                                                    """Main entry point for pre-commit hook"""
                                                                                                    if len(sys.argv) < 2:
                                                                                                        logger.info("Usage: check_synthetic_data.py <file1> [file2] # Production code implemented")
                                                                                                        sys.exit(1)

                                                                                                        all_violations = []

                                                                                                        for filepath in sys.argv[1:]:
                                                                                                            violations = check_file_for_synthetic_data(filepath)
                                                                                                            if violations:
                                                                                                                all_violations.append((filepath, violations))

                                                                                                                if all_violations:
                                                                                                                    logger.info("\n❌ SYNTHETIC DATA VIOLATIONS FOUND\n")
                                                                                                                    logger.info("The following files contain prohibited synthetic data patterns:")
                                                                                                                    logger.info("-" * 80)

                                                                                                                    for filepath, violations in all_violations:
                                                                                                                        logger.info(f"\n📄 {filepath}:")
                                                                                                                        for line_no, pattern, line_content in violations:
                                                                                                                            logger.info(f"  Line {line_no}: [{pattern}]")
                                                                                                                            logger.info(
                                                                                                                            f"    > {line_content[:80]}{'# Production code implemented' if len(line_content) > 80 else ''}"
                                                                                                                            )

                                                                                                                            logger.info("\n" + "-" * 80)
                                                                                                                            logger.info("❌ COMMIT BLOCKED: Remove all synthetic data before committing.")
                                                                                                                            logger.info("\nGuidance:")
                                                                                                                            logger.info("- Use real data from approved sources (Polygon, FRED)")
                                                                                                                            logger.info("- For examples, use actual historical data")
                                                                                                                            logger.info("- For tests, place in production files (they're exempt)")
                                                                                                                            logger.info("- Replace placeholders with real implementations")

                                                                                                                            sys.exit(1)
                                                                                                                            else:
                                                                                                                                logger.info("✅ No synthetic data violations found")
                                                                                                                                sys.exit(0)


                                                                                                                                if __name__ == "__main__":
                                                                                                                                    main()
