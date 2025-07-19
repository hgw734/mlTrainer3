#!/usr/bin/env python3
"""
Detailed analysis of each failed file
"""

import os
import ast
import re

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

def analyze_file_detailed(filepath):
    """Get detailed analysis of what's wrong with a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Try to parse and get specific error
        error_details = {}
        try:
            ast.parse(content)
            return {"status": "OK", "issues": []}
        except SyntaxError as e:
            error_details = {
                "error_type": e.msg,
                "line": e.lineno,
                "text": e.text.strip() if e.text else "N/A",
                "offset": e.offset
            }
        
        # Analyze issues
        issues = []
        
        # Check maximum indentation
        max_indent = 0
        excessive_indent_lines = []
        for i, line in enumerate(lines):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > max_indent:
                    max_indent = indent
                if indent > 40:  # Excessive indentation
                    excessive_indent_lines.append((i+1, indent, line.strip()[:50]))
        
        if max_indent > 40:
            issues.append(f"EXTREME INDENTATION: Up to {max_indent} spaces (normal max: 12-16)")
            issues.append(f"Lines with 40+ spaces: {len(excessive_indent_lines)}")
        
        # Check for split statements
        split_statements = 0
        for i in range(len(lines)-1):
            if lines[i].strip() and lines[i+1].strip():
                # Check if continuation makes sense
                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                next_indent = len(lines[i+1]) - len(lines[i+1].lstrip())
                if next_indent > curr_indent + 20:
                    split_statements += 1
        
        if split_statements > 0:
            issues.append(f"SPLIT STATEMENTS: {split_statements} statements split across lines")
        
        # Check for comment corruption
        comment_breaks = 0
        for i, line in enumerate(lines):
            if '#' in line and i > 0:
                # Check if comment is breaking code
                prev_line = lines[i-1].strip()
                if prev_line and not prev_line.endswith((':',',','(','{','[')):
                    if len(line) - len(line.lstrip()) > 40:
                        comment_breaks += 1
        
        if comment_breaks > 0:
            issues.append(f"COMMENT CORRUPTION: {comment_breaks} comments breaking code flow")
        
        # Check for specific patterns
        if 'TODO:' in content:
            todo_count = content.count('TODO:')
            issues.append(f"TODO MARKERS: {todo_count} unimplemented sections")
        
        if 'raise NotImplementedError' in content:
            not_impl = content.count('raise NotImplementedError')
            issues.append(f"NOT IMPLEMENTED: {not_impl} methods not implemented")
        
        # Get sample of the problematic line
        if error_details.get('line'):
            line_num = error_details['line']
            if 0 < line_num <= len(lines):
                sample = lines[line_num-1]
                indent = len(sample) - len(sample.lstrip())
                issues.append(f"ERROR LINE {line_num}: {indent} space indent")
                
        return {
            "status": "FAILED",
            "error": error_details,
            "issues": issues,
            "max_indentation": max_indent,
            "excessive_indent_count": len(excessive_indent_lines)
        }
        
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "issues": ["Could not read file"]
        }

def main():
    base_dir = "/workspace/mlTrainer3_complete"
    
    print("# DETAILED ANALYSIS OF ALL 68 FAILED FILES\n")
    print("=" * 100)
    
    # Categorize files
    categories = {
        "AUTHENTICATION & API": [],
        "FINANCIAL MODELS": [],
        "TESTING": [],
        "SCRIPTS & AUTOMATION": [],
        "HOOKS & SECURITY": [],
        "UI & CHAT": [],
        "INFRASTRUCTURE": []
    }
    
    # Analyze each file
    for file in FAILED_FILES:
        filepath = os.path.join(base_dir, file)
        if os.path.exists(filepath):
            analysis = analyze_file_detailed(filepath)
            
            # Categorize
            if any(x in file for x in ['auth', 'api', 'connector', 'telegram']):
                category = "AUTHENTICATION & API"
            elif 'custom/' in file:
                category = "FINANCIAL MODELS"
            elif 'test' in file:
                category = "TESTING"
            elif 'scripts/' in file:
                category = "SCRIPTS & AUTOMATION"
            elif 'hooks/' in file:
                category = "HOOKS & SECURITY"
            elif any(x in file for x in ['chat', 'ui']):
                category = "UI & CHAT"
            else:
                category = "INFRASTRUCTURE"
            
            categories[category].append((file, analysis))
    
    # Print detailed report
    for category, files in categories.items():
        if files:
            print(f"\n## {category} ({len(files)} files)")
            print("-" * 100)
            
            for file, analysis in files:
                print(f"\n### {file}")
                
                if analysis['status'] == 'FAILED':
                    error = analysis['error']
                    print(f"**Error**: {error.get('error_type', 'Unknown')} at line {error.get('line', '?')}")
                    
                    if analysis['issues']:
                        print("**Issues Found**:")
                        for issue in analysis['issues']:
                            print(f"  - {issue}")
                    
                    if error.get('text'):
                        print(f"**Problem Code**: `{error['text'][:80]}{'...' if len(error.get('text', '')) > 80 else ''}`")
                    
                    if analysis.get('max_indentation', 0) > 40:
                        print(f"**Indentation**: Maximum {analysis['max_indentation']} spaces (should be ≤16)")
                
                elif analysis['status'] == 'ERROR':
                    print(f"**Error**: {analysis['error']}")

if __name__ == "__main__":
    main()