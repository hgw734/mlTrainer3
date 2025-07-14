#!/usr/bin/env python3
"""
Production Audit Final Script
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionAuditor:
    """Final production audit for mlTrainer"""

                                            def __init__(self):
        self.violations = []
        self.warnings = []
        self.files_processed = 0

    def run_final_audit(self):
        """Run final production audit"""
        logger.info("🔍 Final Production Audit")
        logger.info("=" * 50)
        
        # Get all Python files
        python_files = self._get_python_files()

                                                    for file_path in python_files:
                                                        self._audit_file(file_path)

                                                        # Generate report
                                                        self._generate_report()

        return len(self.violations) == 0

    def _get_python_files(self) -> List[str]:
        """Get all Python files to audit"""
                                                                        python_files = []

        # Directories to scan
        scan_dirs = [
            'core',
            'custom',
            'config',
            'backend',
            'scripts',
            'hooks',
            'utils'
        ]
        
        # Add root directory files
        for file in os.listdir('.'):
            if file.endswith('.py') and not file.startswith('test_'):
                python_files.append(file)
        
        # Add subdirectory files
        for dir_name in scan_dirs:
            if os.path.exists(dir_name):
                for root, dirs, files in os.walk(dir_name):
                                                                            for file in files:
                        if file.endswith('.py') and not file.startswith('test_'):
                            python_files.append(os.path.join(root, file))
        
        return python_files

    def _audit_file(self, file_path: str):
        """Audit a single file"""
        try:
            with open(file_path, 'r') as f:
                                                                                                content = f.read()

            # Check for syntax errors
                                                                                                try:
                ast.parse(content)
                                                                                                    except SyntaxError as e:
                self.violations.append(f"{file_path}: Syntax error - {e}")
                                                                                                        return

            # Check for security issues
            self._check_security_issues(file_path, content)
            
            # Check for governance compliance
            self._check_governance_compliance(file_path, content)
            
            self.files_processed += 1

                                                                                                    except Exception as e:
            self.warnings.append(f"{file_path}: Error reading file - {e}")

    def _check_security_issues(self, file_path: str, content: str):
        """Check for security issues"""
        # Check for dangerous patterns
        dangerous_patterns = [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, content):
                self.violations.append(f"{file_path}: Dangerous pattern found - {pattern}")

    def _check_governance_compliance(self, file_path: str, content: str):
        """Check for governance compliance"""
        # Check for governance imports
        governance_imports = ['governance_kernel', 'compliance_mode', 'agent_rules']
        has_governance = any(import_name in content for import_name in governance_imports)
        
        if not has_governance and 'test_' not in file_path:
            self.warnings.append(f"{file_path}: No governance imports found")

                                                                                                                                                                        def _generate_report(self):
                                                                                                                                                                            """Generate audit report"""
        print(f"\n📊 Final Production Audit Report")
        print("=" * 50)
        
        if self.violations:
            print(f"\n❌ Violations Found ({len(self.violations)}):")
            for violation in self.violations:
                print(f"  - {violation}")
                                                                                                                                                                                            else:
            print(f"\n✅ No violations found!")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:5]:  # Show first 5
                print(f"  - {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
        
        print(f"\n📈 Statistics:")
        print(f"  Files processed: {self.files_processed}")
        print(f"  Total violations: {len(self.violations)}")
        print(f"  Total warnings: {len(self.warnings)}")

                                                                                                                                                                                                                    def main():
    """Main function"""
                                                                                                                                                                                                                        auditor = ProductionAuditor()
    success = auditor.run_final_audit()
    
    if success:
        print("\n🎉 Production audit passed!")
        return 0
    else:
        print("\n❌ Production audit failed!")
        return 1

                                                                                                                                                                                                                                if __name__ == "__main__":
    exit(main()) 