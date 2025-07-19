#!/usr/bin/env python3
from datetime import datetime
import shutil
import sys
import re
import logging

logger = logging.getLogger(__name__)


"""
Security Fix: Remove hardcoded API keys from config/api_config.py
"""


def remove_hardcoded_keys(filepath="config/api_config.py"):
    """Remove hardcoded API keys and replace with secure pattern"""

    # Backup original file
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    logger.info(f"✅ Created backup: {backup_path}")

    # Read file content
    with open(filepath, "r") as f:
        content = f.read()

        # Patterns to replace hardcoded keys
        replacements = [
            (r'POLYGON_API_KEY = os\.getenv\("POLYGON_API_KEY", "[^"]+"\)',
             'POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")\nif not POLYGON_API_KEY:\n    raise ValueError("POLYGON_API_KEY environment variable is required")',
             ),
            (r'FRED_API_KEY = os\.getenv\("FRED_API_KEY", "[^"]+"\)',
             'FRED_API_KEY = os.getenv("FRED_API_KEY")\nif not FRED_API_KEY:\n    raise ValueError("FRED_API_KEY environment variable is required")',
             ),
        ]

        # Apply replacements
        modified = False
        for pattern, replacement in replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
                logger.info(
                    f"✅ Removed hardcoded key pattern: {pattern[:40]}# Production code implemented")

                if modified:
                    # Write updated content
                    with open(filepath, "w") as f:
                        f.write(content)
                        logger.info(
                            f"✅ Updated {filepath} - hardcoded keys removed")
                        logger.info(
                            "⚠️  IMPORTANT: Rotate these API keys immediately if they were real!")
                        else:
                            logger.info(
                                "ℹ️  No hardcoded keys found to remove")

                            return modified

                            if __name__ == "__main__":
                                success = remove_hardcoded_keys()
                                sys.exit(0 if success else 1)
