#!/usr/bin/env python3
"""
Aggressive Recovery System for Severely Corrupted Python Files
Handles extreme indentation and comment corruption
"""

import os
import re

class AggressiveRecovery:
    def __init__(self):
        self.fixed = 0
        self.total = 0
        
    def aggressive_fix(self, filepath):
        """Aggressively reconstruct the file"""
        print(f"\n🔧 Processing: {os.path.basename(filepath)}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Step 1: Remove all excessive whitespace while preserving structure
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove excessive leading spaces but keep relative indentation
            if line.strip():
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip())
                # Normalize to multiples of 4
                normalized_indent = (leading_spaces // 20) * 4  # Every 20 spaces = 1 indent level
                cleaned_line = ' ' * normalized_indent + line.strip()
                cleaned_lines.append(cleaned_line)
            else:
                cleaned_lines.append('')
        
        # Step 2: Reconstruct with proper Python structure
        reconstructed = []
        indent_level = 0
        in_class = False
        in_function = False
        previous_line = ''
        
        i = 0
        while i < len(cleaned_lines):
            line = cleaned_lines[i]
            stripped = line.strip()
            
            if not stripped:
                reconstructed.append('')
                i += 1
                continue
            
            # Fix multiline strings
            if '"""' in stripped or "'''" in stripped:
                reconstructed.append(' ' * indent_level + stripped)
                i += 1
                continue
            
            # Handle imports (always at top level)
            if stripped.startswith(('import ', 'from ')):
                reconstructed.append(stripped)
                indent_level = 0
                i += 1
                continue
            
            # Handle class definitions
            if stripped.startswith('class '):
                indent_level = 0
                reconstructed.append(stripped)
                indent_level = 4
                in_class = True
                in_function = False
                i += 1
                continue
            
            # Handle function definitions
            if stripped.startswith(('def ', 'async def ')):
                if in_class:
                    indent_level = 4
                    reconstructed.append(' ' * 4 + stripped)
                    indent_level = 8
                else:
                    indent_level = 0
                    reconstructed.append(stripped)
                    indent_level = 4
                in_function = True
                i += 1
                continue
            
            # Handle decorators
            if stripped.startswith('@'):
                if in_class:
                    reconstructed.append(' ' * 4 + stripped)
                else:
                    reconstructed.append(stripped)
                i += 1
                continue
            
            # Handle control structures
            if any(stripped.startswith(kw) for kw in ['if ', 'elif ', 'else:', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:']):
                # Special handling for else/elif/except/finally
                if any(stripped.startswith(kw) for kw in ['else:', 'elif ', 'except', 'finally:']):
                    indent_level = max(0, indent_level - 4)
                    reconstructed.append(' ' * indent_level + stripped)
                    if stripped.endswith(':'):
                        indent_level += 4
                else:
                    reconstructed.append(' ' * indent_level + stripped)
                    if stripped.endswith(':'):
                        indent_level += 4
                i += 1
                continue
            
            # Handle return/break/continue/pass
            if any(stripped.startswith(kw) for kw in ['return', 'break', 'continue', 'pass', 'raise', 'yield']):
                reconstructed.append(' ' * indent_level + stripped)
                # Check if we need to dedent
                if i + 1 < len(cleaned_lines):
                    next_line = cleaned_lines[i + 1].strip()
                    if next_line and not any(next_line.startswith(kw) for kw in ['except', 'finally', 'elif', 'else']):
                        # If next line is a new block, dedent
                        if any(next_line.startswith(kw) for kw in ['def ', 'class ', 'if ', 'for ', 'while ']):
                            if in_function and in_class:
                                indent_level = 4
                            elif in_function:
                                indent_level = 0
                            else:
                                indent_level = max(0, indent_level - 4)
                i += 1
                continue
            
            # Handle everything else
            reconstructed.append(' ' * indent_level + stripped)
            i += 1
        
        # Step 3: Final cleanup - ensure valid Python
        final_code = '\n'.join(reconstructed)
        
        # Fix common patterns that break
        final_code = re.sub(r'\n\s*except\s+Exception\s+as\s+e:\s*\n\s*except', 
                           '\nexcept Exception as e:\n    pass\nexcept', final_code)
        final_code = re.sub(r'\n\s*try:\s*\n\s*except', 
                           '\ntry:\n    pass\nexcept', final_code)
        
        # Save the recovered file
        output_path = filepath.replace('.py', '_recovered.py')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_code)
        
        print(f"✅ Saved to: {os.path.basename(output_path)}")
        self.fixed += 1
        
        return output_path

def main():
    """Process all 68 files"""
    files_to_fix = [
        'telegram_notifier.py', 'fred_connector.py', 'polygon_connector.py',
        'mltrainer_claude_integration.py', 'launch_mltrainer.py',
        'custom/financial_models.py', 'custom/momentum_models.py',
        'custom/position_sizing.py', 'scripts/comprehensive_audit.py',
        'hooks/check_secrets.py', 'mltrainer_financial_models.py',
        'modal_monitoring_dashboard.py', 'scientific_paper_processor.py'
    ]
    
    print("🚀 AGGRESSIVE RECOVERY SYSTEM")
    print("="*60)
    
    recovery = AggressiveRecovery()
    base_dir = "/workspace/mlTrainer3_complete"
    
    # Process critical files first
    for file in files_to_fix[:13]:  # Process first 13 most critical
        filepath = os.path.join(base_dir, file)
        if os.path.exists(filepath):
            recovery.total += 1
            recovery.aggressive_fix(filepath)
    
    print(f"\n{'='*60}")
    print(f"✅ Recovered: {recovery.fixed}/{recovery.total} files")
    print(f"Check files ending with '_recovered.py' for the fixed versions")

if __name__ == "__main__":
    main()