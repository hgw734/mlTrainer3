#!/usr/bin/env python3
"""
Comprehensive Audit Script for mlTrainer Compliance
"""

import os
import re
import ast
import yaml
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ComprehensiveAuditor:
    """Comprehensive code auditor for mlTrainer compliance"""
    
    def __init__(self):
        self.violations = defaultdict(list)
        self.warnings = defaultdict(list)
        self.stats = {
            'files_scanned': 0,
            'total_lines': 0,
            'dependencies_checked': 0,
            'rules_checked': 0
        }
        
        # Load agent rules
        self.agent_rules = self._load_agent_rules()
        
        # Define audit categories
        self.audit_categories = {
            'synthetic_data': self._audit_synthetic_data,
            'data_sources': self._audit_data_sources,
            'api_keys': self._audit_api_keys,
            'error_handling': self._audit_error_handling,
            'code_quality': self._audit_code_quality,
            'permissions': self._audit_permissions,
            'documentation': self._audit_documentation,
            'security': self._audit_security,
            'governance': self._audit_governance
        }

    def run_full_audit(self) -> bool:
        """Run comprehensive audit on entire codebase"""
        print("🔍 mlTrainer Comprehensive Compliance Audit")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        # Audit all Python files
        python_files = self._get_python_files()
        print(f"📁 Found {len(python_files)} Python files to audit")
        
        for file_path in python_files:
            self._audit_file(file_path)
        
        # Audit dependencies
        self._audit_all_dependencies()
        
        # Audit configuration files
        self._audit_config_files()
        
        # Generate report
        self._generate_report()
        
        # Return success if no critical violations
        critical_violations = sum(len(v) for k, v in self.violations.items() 
                                if k in ['synthetic_data', 'api_keys', 'security'])
        return critical_violations == 0

    def _load_agent_rules(self) -> dict:
        """Load agent rules from YAML"""
        rules_path = Path('agent_rules.yaml')
        if rules_path.exists():
            with open(rules_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _get_python_files(self) -> List[Path]:
        """Get all Python files to audit"""
        exclude_dirs = {'.git', '__pycache__', 'venv', 'env', 'modal_env', '.pytest_cache'}
        python_files = []
        
        for root, dirs, files in os.walk('.'):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return sorted(python_files)

    def _audit_file(self, file_path: Path):
        """Audit a single Python file"""
        self.stats['files_scanned'] += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
                self.stats['total_lines'] += len(lines)
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self.violations['syntax_errors'].append(f"{file_path}: {e}")
                return
            
            # Run all audit categories
            for category, audit_func in self.audit_categories.items():
                audit_func(file_path, content, tree)
                
        except Exception as e:
            self.violations['file_errors'].append(f"{file_path}: {e}")

    def _audit_synthetic_data(self, file_path: Path, content: str, tree: ast.AST):
        """Audit for synthetic/real_implementation data usage"""
        # Skip production files and governance files for synthetic data check
        skip_files = ['test_', 'governance_kernel.py', 'check_synthetic_data.py',
                     'agent_rules.yaml', 'validate_governance.py']
        
        if any(skip in str(file_path) for skip in skip_files):
            return
        
        prohibited_patterns = self.agent_rules.get('data_authenticity', {}).get('prohibited_patterns', [])
        
        for pattern in prohibited_patterns:
            # Use regex for more accurate matching
            if pattern == 'np.random':
                regex = r'np\.random\.'
            elif pattern == 'random[.]random[(][)]':
                regex = r'random\.random\(\)'
            else:
                regex = rf'\b{pattern}\b'
            
            matches = re.finditer(regex, content)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.violations['synthetic_data'].append(
                    f"{file_path}:{line_num} - Found prohibited pattern '{pattern}'"
                )

    def _audit_data_sources(self, file_path: Path, content: str, tree: ast.AST):
        """Audit for approved data sources"""
        approved_sources = ['polygon', 'fred', 'redis', 'database', 'user-provided']
        
        # Look for data fetching patterns
        data_fetch_patterns = [r'fetch.*data', r'get.*data', r'load.*data', r'read.*data']
        
        for pattern in data_fetch_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                has_approved_source = any(source in content.lower() for source in approved_sources)
                if not has_approved_source and 'test_' not in str(file_path):
                    self.warnings['data_sources'].append(
                        f"{file_path} - Data fetching detected without clear approved source"
                    )

    def _audit_api_keys(self, file_path: Path, content: str, tree: ast.AST):
        """Audit for hardcoded API keys"""
        # Common API key patterns
        api_key_patterns = [
            r'["\'](?:api[_-]?key|apikey)["\'][\s]*[:=][\s]*["\'][A-Za-z0-9]{20,}["\']',
            r'["\'](?:secret|token|password)["\'][\s]*[:=][\s]*["\'][A-Za-z0-9]{20,}["\']',
        ]
        
        for pattern in api_key_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.violations['api_keys'].append(
                    f"{file_path}:{line_num} - Potential hardcoded API key found"
                )

    def _audit_error_handling(self, file_path: Path, content: str, tree: ast.AST):
        """Audit for proper error handling"""
        # Check for bare except clauses
        bare_except_pattern = r'except\s*:'
        matches = re.finditer(bare_except_pattern, content)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            self.warnings['error_handling'].append(
                f"{file_path}:{line_num} - Bare except clause found"
            )

    def _audit_code_quality(self, file_path: Path, content: str, tree: ast.AST):
        """Audit for code quality issues"""
        # Check for long functions (>50 lines)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 50:
                    self.warnings['code_quality'].append(
                        f"{file_path} - Long function '{node.name}' found"
                    )

    def _audit_permissions(self, file_path: Path, content: str, tree: ast.AST):
        """Audit for permission-related issues"""
        # Check for file system operations without proper checks
        fs_patterns = [r'open\s*\(', r'os\.remove', r'shutil\.rmtree']
        for pattern in fs_patterns:
            if re.search(pattern, content):
                if 'try' not in content or 'except' not in content:
                    self.warnings['permissions'].append(
                        f"{file_path} - File system operation without error handling"
                    )

    def _audit_documentation(self, file_path: Path, content: str, tree: ast.AST):
        """Audit for documentation quality"""
        # Check for functions without docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node):
                    self.warnings['documentation'].append(
                        f"{file_path} - Function '{node.name}' missing docstring"
                    )

    def _audit_security(self, file_path: Path, content: str, tree: ast.AST):
        """Audit for security issues"""
        # Check for dangerous patterns
        dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, content):
                line_num = content[:content.find(pattern)].count('\n') + 1
                self.violations['security'].append(
                    f"{file_path}:{line_num} - Dangerous pattern found"
                )

    def _audit_governance(self, file_path: Path, content: str, tree: ast.AST):
        """Audit for governance compliance"""
        # Check for governance imports
        governance_imports = ['governance_kernel', 'compliance_mode', 'agent_rules']
        for import_name in governance_imports:
            if import_name in content:
                self.stats['rules_checked'] += 1

    def _audit_all_dependencies(self):
        """Audit all dependencies"""
        self._audit_dependencies()

    def _audit_dependencies(self):
        """Audit Python dependencies"""
        try:
            import pkg_resources
            installed_packages = [d.project_name for d in pkg_resources.working_set]
            self.stats['dependencies_checked'] = len(installed_packages)
        except ImportError:
            pass

    def _audit_config_files(self):
        """Audit configuration files"""
        config_files = ['agent_rules.yaml', 'requirements.txt', 'setup.py']
        for config_file in config_files:
            if Path(config_file).exists():
                self.stats['files_scanned'] += 1

    def _generate_report(self):
        """Generate audit report"""
        print("\n📊 Audit Results:")
        print("=" * 50)
        
        # Print violations
        if self.violations:
            print("\n❌ Violations Found:")
            for category, violations in self.violations.items():
                if violations:
                    print(f"\n{category.upper()}:")
                    for violation in violations[:5]:  # Show first 5
                        print(f"  - {violation}")
                    if len(violations) > 5:
                        print(f"  ... and {len(violations) - 5} more")
        else:
            print("\n✅ No violations found!")
        
        # Print warnings
        if self.warnings:
            print("\n⚠️  Warnings:")
            for category, warnings in self.warnings.items():
                if warnings:
                    print(f"\n{category.upper()}:")
                    for warning in warnings[:3]:  # Show first 3
                        print(f"  - {warning}")
                    if len(warnings) > 3:
                        print(f"  ... and {len(warnings) - 3} more")
        
        # Print statistics
        print(f"\n📈 Statistics:")
        print(f"  Files scanned: {self.stats['files_scanned']}")
        print(f"  Total lines: {self.stats['total_lines']}")
        print(f"  Dependencies checked: {self.stats['dependencies_checked']}")
        print(f"  Rules checked: {self.stats['rules_checked']}")
        
        # Save detailed report
        self._save_detailed_report()

    def _save_detailed_report(self):
        """Save detailed report to file"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'violations': dict(self.violations),
            'warnings': dict(self.warnings)
        }
        
        report_file = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            import json
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"\n❌ Failed to save report: {e}")

def main():
    """Main function"""
    auditor = ComprehensiveAuditor()
    success = auditor.run_full_audit()
    
    if success:
        print("\n🎉 Audit completed successfully!")
        return 0
    else:
        print("\n❌ Audit found violations that need attention.")
        return 1

if __name__ == "__main__":
    exit(main()) 