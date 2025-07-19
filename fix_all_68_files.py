#!/usr/bin/env python3
"""
Emergency Recovery: Fix all 68 corrupted files
This will intelligently reconstruct the code by removing extreme indentation
"""

import os
import re
import ast
from pathlib import Path

class EmergencyCodeReconstructor:
    def __init__(self):
        self.fixed = 0
        self.failed = 0
        
    def reconstruct_code(self, content):
        """Reconstruct code from corrupted indentation"""
        lines = content.split('\n')
        reconstructed = []
        
        # Track context for intelligent reconstruction
        current_class = None
        current_function = None
        in_try_block = False
        in_if_block = False
        expected_indent = 0
        bracket_stack = []
        
        for i, line in enumerate(lines):
            original_line = line
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                reconstructed.append('')
                continue
            
            # Handle multiline strings
            if '"""' in stripped or "'''" in stripped:
                # Multiline strings should maintain some formatting
                reconstructed.append(' ' * expected_indent + stripped)
                continue
            
            # Detect class definitions
            if stripped.startswith('class '):
                current_class = stripped
                current_function = None
                expected_indent = 0
                reconstructed.append(stripped)
                expected_indent = 4
                continue
            
            # Detect function definitions
            if stripped.startswith('def ') or stripped.startswith('async def '):
                current_function = stripped
                # If we're in a class, indent the function
                if current_class:
                    reconstructed.append(' ' * 4 + stripped)
                    expected_indent = 8
                else:
                    reconstructed.append(stripped)
                    expected_indent = 4
                continue
            
            # Handle decorators
            if stripped.startswith('@'):
                # Decorators go at the same level as the function they decorate
                if current_class:
                    reconstructed.append(' ' * 4 + stripped)
                else:
                    reconstructed.append(stripped)
                continue
            
            # Handle imports (always at top level)
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or
                re.match(r'^(import|from)\s+', stripped)):
                reconstructed.append(stripped)
                expected_indent = 0
                continue
            
            # Handle docstrings
            if (stripped.startswith('"""') or stripped.startswith("'''") or
                stripped.startswith('"') and len(stripped) > 20):
                reconstructed.append(' ' * expected_indent + stripped)
                continue
            
            # Handle control flow
            if any(stripped.startswith(kw) for kw in ['if ', 'elif ', 'else:', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:', 'match ']):
                # Dedent for else/elif/except/finally
                if any(stripped.startswith(kw) for kw in ['else:', 'elif ', 'except', 'finally:']):
                    expected_indent = max(0, expected_indent - 4)
                
                reconstructed.append(' ' * expected_indent + stripped)
                
                # Indent for the block content
                if stripped.endswith(':'):
                    expected_indent += 4
                    if stripped.startswith('try:'):
                        in_try_block = True
                    elif any(stripped.startswith(kw) for kw in ['if ', 'elif ']):
                        in_if_block = True
                continue
            
            # Handle returns, breaks, continues
            if any(stripped.startswith(kw) for kw in ['return', 'break', 'continue', 'pass', 'raise', 'yield']):
                reconstructed.append(' ' * expected_indent + stripped)
                # These often end a block
                if i + 1 < len(lines) and lines[i + 1].strip():
                    next_line = lines[i + 1].strip()
                    # Check if next line is a dedent
                    if (any(next_line.startswith(kw) for kw in ['def ', 'class ', '@', 'if ', 'for ', 'while ']) or
                        (current_function and not any(next_line.startswith(kw) for kw in ['except', 'finally', 'elif', 'else']))):
                        expected_indent = 4 if current_class else 0
                continue
            
            # Handle brackets/parentheses
            open_brackets = stripped.count('(') + stripped.count('[') + stripped.count('{')
            close_brackets = stripped.count(')') + stripped.count(']') + stripped.count('}')
            
            # If we have unclosed brackets, the next line continues this one
            if open_brackets > close_brackets:
                bracket_stack.append(expected_indent)
                reconstructed.append(' ' * expected_indent + stripped)
                # Don't change indent for continuation
                continue
            elif close_brackets > open_brackets and bracket_stack:
                # We're closing brackets from previous lines
                bracket_stack.pop()
                reconstructed.append(' ' * expected_indent + stripped)
                continue
            
            # Handle assignments and expressions
            if ('=' in stripped and not any(op in stripped for op in ['==', '!=', '<=', '>='])) or stripped.startswith('self.'):
                reconstructed.append(' ' * expected_indent + stripped)
                continue
            
            # Handle function calls and other statements
            reconstructed.append(' ' * expected_indent + stripped)
            
            # Check if we need to dedent based on context
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not any(next_line.startswith(kw) for kw in ['#', 'except', 'finally', 'elif', 'else']):
                    # If the next line starts a new top-level construct
                    if any(next_line.startswith(kw) for kw in ['def ', 'class ', '@']) and expected_indent > 4:
                        expected_indent = 4 if current_class else 0
        
        return '\n'.join(reconstructed)
    
    def fix_file(self, filepath):
        """Fix a single file"""
        try:
            print(f"\n{'='*60}")
            print(f"Fixing: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip if already valid
            try:
                ast.parse(content)
                print("✅ Already valid, skipping")
                return True
            except:
                pass
            
            # Reconstruct the code
            fixed_content = self.reconstruct_code(content)
            
            # Validate the fix
            try:
                ast.parse(fixed_content)
                # Save the fixed file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print("✅ Successfully fixed and saved!")
                self.fixed += 1
                return True
            except SyntaxError as e:
                print(f"❌ Still has errors after fix: Line {e.lineno}: {e.msg}")
                # Save it anyway for manual review
                with open(filepath + '.attempted_fix', 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"💾 Saved attempted fix to: {filepath}.attempted_fix")
                self.failed += 1
                return False
                
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            self.failed += 1
            return False


# List of all 68 files to fix
FILES_TO_FIX = [
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


def main():
    """Main recovery function"""
    print("🚨 EMERGENCY RECOVERY SYSTEM")
    print("Attempting to fix all 68 corrupted files...")
    
    base_dir = "/workspace/mlTrainer3_complete"
    reconstructor = EmergencyCodeReconstructor()
    
    # Process each file
    for file in FILES_TO_FIX:
        filepath = os.path.join(base_dir, file)
        if os.path.exists(filepath):
            reconstructor.fix_file(filepath)
        else:
            print(f"\n⚠️  File not found: {file}")
    
    # Summary
    print("\n" + "="*60)
    print("RECOVERY SUMMARY")
    print("="*60)
    print(f"✅ Successfully fixed: {reconstructor.fixed} files")
    print(f"❌ Failed to fix: {reconstructor.failed} files")
    print(f"📊 Success rate: {reconstructor.fixed / len(FILES_TO_FIX) * 100:.1f}%")
    
    if reconstructor.failed > 0:
        print("\n⚠️  Some files need manual review.")
        print("Check files ending with '.attempted_fix' for partial fixes.")
    
    print("\n✅ Recovery process complete!")


if __name__ == "__main__":
    main()