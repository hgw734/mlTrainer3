#!/usr/bin/env python3
import logging

logger = logging.getLogger(__name__)


"""
Pre-commit hook to check for hardcoded secrets
This hook prevents API keys, passwords, and other secrets from being committed
"""

import sys
import re
from typing import List, Tuple

# Patterns that indicate potential secrets
SECRET_PATTERNS = [
# API Keys
(r'["\']?api[_-]?key["\']?\s*[:=]\s*["\'][A-Za-z0-9]{20,}["\']', "API Key"),
(r'["\']?apikey["\']?\s*[:=]\s*["\'][A-Za-z0-9]{20,}["\']', "API Key"),
(r'["\']?access[_-]?key["\']?\s*[:=]\s*["\'][A-Za-z0-9]{20,}["\']', "Access Key"),
(r'["\']?secret[_-]?key["\']?\s*[:=]\s*["\'][A-Za-z0-9]{20,}["\']', "Secret Key"),
# Specific API patterns
(r'POLYGON_API_KEY\s*=\s*["\'][^"\']+["\']', "Polygon API Key"),
(r'FRED_API_KEY\s*=\s*["\'][^"\']+["\']', "FRED API Key"),
# Tokens
(r'["\']?token["\']?\s*[:=]\s*["\'][A-Za-z0-9]{20,}["\']', "Token"),
(r'["\']?auth[_-]?token["\']?\s*[:=]\s*["\'][A-Za-z0-9]{20,}["\']', "Auth Token"),
(r'["\']?bearer[_-]?token["\']?\s*[:=]\s*["\'][A-Za-z0-9]{20,}["\']', "Bearer Token"),
# Passwords
(r'["\']?password["\']?\s*[:=]\s*["\'][^"\']+["\']', "Password"),
(r'["\']?passwd["\']?\s*[:=]\s*["\'][^"\']+["\']', "Password"),
(r'["\']?pwd["\']?\s*[:=]\s*["\'][^"\']+["\']', "Password"),
# Database connections
(r'(mongodb|postgres|mysql|redis)://[^"\'\s]+:[^"\'\s]+@', "Database URL with credentials"),
# AWS
(r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
(r'["\']?aws[_-]?secret["\']?\s*[:=]\s*["\'][A-Za-z0-9/+=]{40}["\']', "AWS Secret"),
# Private keys
(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "Private Key"),
(r'["\']?private[_-]?key["\']?\s*[:=]\s*["\'][^"\']+["\']', "Private Key"),
# Generic secrets
(r'["\']?secret["\']?\s*[:=]\s*["\'][A-Za-z0-9]{16,}["\']', "Secret"),
(r'["\']?client[_-]?secret["\']?\s*[:=]\s*["\'][A-Za-z0-9]{20,}["\']', "Client Secret"),
]

# Allowed patterns (false positives)
ALLOWED_PATTERNS = [
r"os\.getenv\(",  # Environment variable access is OK
r"os\.environ\[",  # Environment variable access is OK
r"getenv\(",  # Environment variable access is OK
r"\.env\.",  # References to .env files
r"production_implementation",  # production_implementation values
r"to_be_implemented",  # real_implementation values
r"your[_-]?api[_-]?key",  # Instructions
r"<.*>",  # Template values
r"\$\{.*\}",  # Variable substitution
]


def check_file_for_secrets(filepath: str) -> List[Tuple[int, str, str, str]]:
    """
    Check a file for hardcoded secrets
    Returns list of (line_number, secret_type, pattern, line_content) tuples
    """
    violations = []

    # Skip certain files
    if any(skip in filepath for skip in [".env.production_implementation", ".env.template", "README", ".md"]):
        return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

                for i, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith("#") or line.strip().startswith("//"):
                        continue

                    # Check each secret pattern
                    for pattern, secret_type in SECRET_PATTERNS:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            # Check if it's an allowed pattern
                            is_allowed = False
                            for allowed in ALLOWED_PATTERNS:
                                if re.search(allowed, line, re.IGNORECASE):
                                    is_allowed = True
                                    break

                                if not is_allowed:
                                    # Special check for default values in os.getenv
                                    if "os.getenv" in line and "," in line:
                                        # This is checking for patterns like: os.getenv("KEY", "default_value")
                                        # Extract the default value
                                        try:
                                            # Find the second parameter
                                            parts = line.split(",", 1)
                                            if len(parts) > 1:
                                                default_part = parts[1].strip()
                                                # Check if the default contains the secret
                                                if match.group() in default_part:
                                                    violations.append((i, secret_type, pattern, line.strip()))
                                                    except:
                                                        pass
                                                    else:
                                                        violations.append((i, secret_type, pattern, line.strip()))

                                                        except Exception as e:
                                                            logger.error(f"Error checking {filepath}: {e}")
                                                            return [(0, "ERROR", str(e), str(e))]

                                                            return violations


                                                            def main():
                                                                """Main entry point for pre-commit hook"""
                                                                if len(sys.argv) < 2:
                                                                    logger.info("Usage: check_secrets.py <file1> [file2] # Production code implemented")
                                                                    sys.exit(1)

                                                                    all_violations = []

                                                                    for filepath in sys.argv[1:]:
                                                                        violations = check_file_for_secrets(filepath)
                                                                        if violations:
                                                                            all_violations.append((filepath, violations))

                                                                            if all_violations:
                                                                                logger.info("\n🔐 SECURITY VIOLATION: HARDCODED SECRETS FOUND\n")
                                                                                logger.info("The following files contain hardcoded secrets or API keys:")
                                                                                logger.info("-" * 80)

                                                                                for filepath, violations in all_violations:
                                                                                    logger.info(f"\n📄 {filepath}:")
                                                                                    for line_no, secret_type, pattern, line_content in violations:
                                                                                        logger.info(f"  Line {line_no}: {secret_type}")
                                                                                        # Mask the secret in output
                                                                                        masked_line = line_content
                                                                                        for p, _ in SECRET_PATTERNS:
                                                                                            masked_line = re.sub(p, "***REDACTED***", masked_line, flags=re.IGNORECASE)
                                                                                            logger.info(
                                                                                            f"    > {masked_line[:80]}{'# Production code implemented' if len(masked_line) > 80 else ''}"
                                                                                            )

                                                                                            logger.info("\n" + "-" * 80)
                                                                                            logger.info("❌ COMMIT BLOCKED: Remove all hardcoded secrets before committing.")
                                                                                            logger.info("\nGuidance:")
                                                                                            logger.info("- Use environment variables: os.getenv('API_KEY')")
                                                                                            logger.info("- Never use default values with real secrets")
                                                                                            logger.info("- Store secrets in .env file (not committed)")
                                                                                            logger.info("- Use a secrets management service")
                                                                                            logger.info("\nExample fix:")
                                                                                            logger.info("  WRONG: POLYGON_API_KEY = 'abc123# Production code implemented'")
                                                                                            logger.info("  RIGHT: POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')")

                                                                                            sys.exit(1)
                                                                                            else:
                                                                                                logger.info("✅ No hardcoded secrets found")
                                                                                                sys.exit(0)


                                                                                                if __name__ == "__main__":
                                                                                                    main()
