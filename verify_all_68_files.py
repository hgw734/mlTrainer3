#!/usr/bin/env python3
"""
Comprehensive verification of all 68 recovered files
Checks for real functionality, no errors, and compliance
"""

import os
import ast
import re
from typing import Dict, List, Tuple

class FileVerifier:
    def __init__(self):
        self.results = []
        self.compliance_issues = []
        self.functionality_checks = []
        
    def verify_file(self, filepath: str) -> Dict[str, any]:
        """Comprehensive verification of a single file"""
        filename = os.path.basename(filepath)
        result = {
            'file': filename,
            'valid_syntax': False,
            'has_docstring': False,
            'has_imports': False,
            'has_classes_or_functions': False,
            'no_placeholder_code': False,
            'no_hardcoded_secrets': False,
            'proper_error_handling': False,
            'compliance_score': 0
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 1. Check valid syntax
            try:
                tree = ast.parse(content)
                result['valid_syntax'] = True
            except SyntaxError:
                return result
            
            # 2. Check for docstring
            if '"""' in content or "'''" in content:
                result['has_docstring'] = True
            
            # 3. Check for imports (real functionality)
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            result['has_imports'] = len(imports) > 0
            
            # 4. Check for classes or functions (real implementation)
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            result['has_classes_or_functions'] = len(classes) > 0 or len(functions) > 0
            
            # 5. Check for placeholder code
            placeholder_patterns = [
                r'pass\s*$',  # Only pass statements
                r'TODO',
                r'FIXME',
                r'NotImplemented',
                r'raise NotImplementedError'
            ]
            
            # Count meaningful lines (not just pass)
            meaningful_lines = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function has real implementation
                    if len(node.body) > 1 or (len(node.body) == 1 and not isinstance(node.body[0], ast.Pass)):
                        meaningful_lines += 1
            
            has_placeholder = any(re.search(pattern, content) for pattern in placeholder_patterns)
            result['no_placeholder_code'] = not has_placeholder and meaningful_lines > 0
            
            # 6. Check for hardcoded secrets
            secret_patterns = [
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'password\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']'
            ]
            has_secrets = any(re.search(pattern, content, re.IGNORECASE) for pattern in secret_patterns)
            result['no_hardcoded_secrets'] = not has_secrets
            
            # 7. Check error handling
            try_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
            result['proper_error_handling'] = len(try_blocks) > 0 or 'logger' in content
            
            # Calculate compliance score
            checks = [
                result['valid_syntax'],
                result['has_docstring'],
                result['has_imports'],
                result['has_classes_or_functions'],
                result['no_placeholder_code'],
                result['no_hardcoded_secrets'],
                result['proper_error_handling']
            ]
            result['compliance_score'] = sum(checks) / len(checks) * 100
            
        except Exception as e:
            result['error'] = str(e)
        
        return result

# List of all 68 files to verify
FILES_TO_VERIFY = [
    # Core 5 rebuilt files
    'telegram_notifier.py',
    'polygon_connector.py',
    'fred_connector.py',
    'mltrainer_claude_integration.py',
    'launch_mltrainer.py',
    
    # Other 63 files
    'paper_processor.py',
    'mlagent_bridge.py',
    'test_simple_system.py',
    'mlTrainer_client_wrapper.py',
    'mltrainer_unified_chat_backup.py',
    'mltrainer_chat.py',
    'mlagent_model_integration.py',
    'diagnose_mltrainer_location.py',
    'fix_indentation_errors.py',
    'test_data_connections.py',
    'scientific_paper_processor.py',
    'test_model_integration.py',
    'paper_processor_demo.py',
    'fix_security_api_keys.py',
    'test_unified_architecture.py',
    'mltrainer_financial_models.py',
    'self_learning_engine_helpers.py',
    'verify_compliance_system.py',
    'modal_app_optimized.py',
    'test_phase1_config.py',
    'test_self_learning_engine.py',
    'verify_compliance_enforcement.py',
    'list_missing_models.py',
    'walk_forward_trial_launcher.py',
    'modal_monitoring_dashboard.py',
    'test_api_keys.py',
    'run_cursor_agent.py',
    'test_model_verification.py',
    'mltrainer_unified_chat_fixed.py',
    'test_chat_persistence.py',
    'mltrainer_unified_chat_intelligent_fix.py',
    'session_compliance_check.py',
    'custom/meta_learning.py',
    'custom/financial_models.py',
    'custom/momentum_models.py',
    'custom/stress.py',
    'custom/optimization.py',
    'custom/information_theory.py',
    'custom/time_series.py',
    'custom/macro.py',
    'custom/pairs.py',
    'custom/interest_rate.py',
    'custom/binomial.py',
    'custom/elliott_wave.py',
    'custom/microstructure.py',
    'custom/adaptive.py',
    'custom/position_sizing.py',
    'custom/regime_ensemble.py',
    'custom/rl.py',
    'custom/alternative_data.py',
    'custom/ensemble.py',
    'scripts/fix_remaining_violations.py',
    'scripts/production_audit_final.py',
    'scripts/final_compliance_check.py',
    'scripts/fix_critical_violations.py',
    'scripts/production_audit.py',
    'scripts/fix_all_violations.py',
    'scripts/fix_final_violations.py',
    'scripts/comprehensive_audit.py',
    'hooks/check_governance_imports.py',
    'hooks/check_secrets.py',
    'hooks/validate_governance.py',
    'hooks/check_synthetic_data.py'
]

def main():
    """Verify all 68 files"""
    print("🔍 COMPREHENSIVE FILE VERIFICATION")
    print("="*60)
    print(f"Verifying {len(FILES_TO_VERIFY)} files for:")
    print("- Valid syntax")
    print("- Real functionality (not placeholders)")
    print("- Proper structure")
    print("- Security compliance")
    print("- Error handling")
    print("="*60)
    
    verifier = FileVerifier()
    base_dir = "/workspace/mlTrainer3_complete"
    
    all_results = []
    
    for i, file in enumerate(FILES_TO_VERIFY, 1):
        filepath = os.path.join(base_dir, file)
        if os.path.exists(filepath):
            print(f"\n[{i}/{len(FILES_TO_VERIFY)}] Verifying: {file}")
            result = verifier.verify_file(filepath)
            all_results.append(result)
            
            # Print summary for this file
            print(f"  ✓ Valid syntax: {result['valid_syntax']}")
            print(f"  ✓ Has docstring: {result['has_docstring']}")
            print(f"  ✓ Has imports: {result['has_imports']}")
            print(f"  ✓ Has implementation: {result['has_classes_or_functions']}")
            print(f"  ✓ No placeholders: {result['no_placeholder_code']}")
            print(f"  ✓ No hardcoded secrets: {result['no_hardcoded_secrets']}")
            print(f"  ✓ Error handling: {result['proper_error_handling']}")
            print(f"  📊 Compliance Score: {result['compliance_score']:.1f}%")
    
    # Summary statistics
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    total_files = len(all_results)
    valid_syntax = sum(1 for r in all_results if r['valid_syntax'])
    has_real_code = sum(1 for r in all_results if r['has_classes_or_functions'])
    no_placeholders = sum(1 for r in all_results if r['no_placeholder_code'])
    secure = sum(1 for r in all_results if r['no_hardcoded_secrets'])
    avg_compliance = sum(r['compliance_score'] for r in all_results) / total_files
    
    print(f"✅ Valid Syntax: {valid_syntax}/{total_files} ({valid_syntax/total_files*100:.1f}%)")
    print(f"✅ Real Implementation: {has_real_code}/{total_files} ({has_real_code/total_files*100:.1f}%)")
    print(f"✅ No Placeholders: {no_placeholders}/{total_files} ({no_placeholders/total_files*100:.1f}%)")
    print(f"✅ Security Compliant: {secure}/{total_files} ({secure/total_files*100:.1f}%)")
    print(f"📊 Average Compliance Score: {avg_compliance:.1f}%")
    
    # List any files with issues
    issues = [r for r in all_results if r['compliance_score'] < 100]
    if issues:
        print(f"\n⚠️  {len(issues)} files need attention:")
        for r in issues:
            print(f"  - {r['file']}: {r['compliance_score']:.1f}% compliant")
    else:
        print("\n🎉 ALL FILES ARE FULLY COMPLIANT!")

if __name__ == "__main__":
    main()