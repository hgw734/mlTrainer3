#!/usr/bin/env python3
"""
SECURITY AUDIT: Analyze all failed files for placeholder code and security risks
"""

import os
import re

# List of failed files from the previous run
FAILED_FILES = [
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
    'mltrainer_claude_integration.py',
    'verify_compliance_system.py',
    'modal_app_optimized.py',
    'telegram_notifier.py',
    'test_phase1_config.py',
    'launch_mltrainer.py',
    'fred_connector.py',
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
    'polygon_connector.py',
    'mltrainer_unified_chat_intelligent_fix.py',
    'session_compliance_check.py',
    # Custom models
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
    # Scripts
    'scripts/fix_remaining_violations.py',
    'scripts/production_audit_final.py',
    'scripts/final_compliance_check.py',
    'scripts/fix_critical_violations.py',
    'scripts/production_audit.py',
    'scripts/fix_all_violations.py',
    'scripts/fix_final_violations.py',
    'scripts/comprehensive_audit.py',
    # Hooks
    'hooks/check_governance_imports.py',
    'hooks/check_secrets.py',
    'hooks/validate_governance.py',
    'hooks/check_synthetic_data.py'
]

def analyze_file(filepath):
    """Analyze a file for security issues and placeholder code"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = []
        
        # Check for TODO/FIXME comments
        todos = re.findall(r'#\s*(TODO|FIXME|XXX|HACK|BUG).*', content, re.IGNORECASE)
        if todos:
            issues.append(f"Found {len(todos)} TODO/FIXME markers")
        
        # Check for placeholder implementations
        if 'raise NotImplementedError' in content:
            count = content.count('raise NotImplementedError')
            issues.append(f"NotImplementedError: {count} occurrences")
        
        # Check for pass statements in functions
        pass_in_func = re.findall(r'def\s+\w+.*:\s*\n\s*pass', content)
        if pass_in_func:
            issues.append(f"Empty functions with 'pass': {len(pass_in_func)}")
        
        # Check for hardcoded credentials
        hardcoded = re.findall(r'(password|secret|key|token)\s*=\s*["\'].*["\']', content, re.IGNORECASE)
        if hardcoded:
            issues.append(f"CRITICAL: Hardcoded credentials found: {len(hardcoded)}")
        
        # Check for dummy returns
        dummy_returns = re.findall(r'return\s+(None|0|""|\'\'|\[\]|\{\})\s*#.*dummy', content, re.IGNORECASE)
        if dummy_returns:
            issues.append(f"Dummy return values: {len(dummy_returns)}")
        
        # Get the actual error line
        lines = content.split('\n')
        error_context = None
        
        # Try to find the syntax error
        for i, line in enumerate(lines):
            if '# TODO:' in line or 'pass  # Implement' in line:
                error_context = f"Line {i+1}: {line.strip()}"
                break
        
        return issues, error_context
        
    except Exception as e:
        return [f"Error reading file: {str(e)}"], None

def main():
    base_dir = "/workspace/mlTrainer3_complete"
    
    print("🔍 SECURITY AUDIT: Analyzing Failed Files")
    print("=" * 80)
    
    critical_files = []
    high_risk_files = []
    medium_risk_files = []
    
    for file in FAILED_FILES:
        filepath = os.path.join(base_dir, file)
        if os.path.exists(filepath):
            issues, error_context = analyze_file(filepath)
            
            # Categorize by risk
            risk_level = "LOW"
            if any('CRITICAL' in issue for issue in issues):
                risk_level = "CRITICAL"
                critical_files.append((file, issues))
            elif any('NotImplementedError' in issue for issue in issues):
                risk_level = "HIGH"
                high_risk_files.append((file, issues))
            elif issues:
                risk_level = "MEDIUM"
                medium_risk_files.append((file, issues))
            
            print(f"\n[{risk_level}] {file}")
            for issue in issues:
                print(f"  ⚠️  {issue}")
            if error_context:
                print(f"  📍 {error_context}")
    
    # Summary
    print("\n" + "=" * 80)
    print("🚨 SECURITY BREACH SUMMARY")
    print("=" * 80)
    print(f"CRITICAL RISK FILES: {len(critical_files)}")
    print(f"HIGH RISK FILES: {len(high_risk_files)}")
    print(f"MEDIUM RISK FILES: {len(medium_risk_files)}")
    
    if critical_files:
        print("\n🔴 CRITICAL SECURITY ISSUES:")
        for file, issues in critical_files[:5]:
            print(f"  {file}: {', '.join(issues)}")

if __name__ == "__main__":
    main()