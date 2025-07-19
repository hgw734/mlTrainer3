#!/usr/bin/env python3
"""
Fix ALL Python files in mlTrainer3
"""

import os
import ast
import sys
from pathlib import Path

class CompleteFixer:
    def __init__(self):
        self.total = 0
        self.fixed = 0
        self.already_valid = 0
        self.failed = 0
        self.errors = []
        
    def fix_file(self, filepath):
        """Fix a single Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip empty files
            if not content.strip():
                return "empty"
                
            # Check if already valid
            try:
                ast.parse(content)
                self.already_valid += 1
                return "valid"
            except SyntaxError as e:
                # Fix the file
                fixed_content = self.fix_syntax(content, e)
                
                # Test the fix
                try:
                    ast.parse(fixed_content)
                    # Save the fixed file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    self.fixed += 1
                    return "fixed"
                except Exception as e2:
                    self.failed += 1
                    self.errors.append((filepath, str(e)))
                    return "failed"
                    
        except Exception as e:
            self.failed += 1
            self.errors.append((filepath, str(e)))
            return "error"
    
    def fix_syntax(self, content, error):
        """Fix syntax errors in content"""
        lines = content.split('\n')
        
        # Fix based on error type
        if "invalid syntax" in str(error):
            return self.fix_invalid_syntax(lines, error)
        elif "unexpected indent" in str(error) or "unindent" in str(error):
            return self.fix_indentation(lines)
        else:
            # Default: fix indentation
            return self.fix_indentation(lines)
    
    def fix_invalid_syntax(self, lines, error):
        """Fix invalid syntax errors"""
        # First try to fix the specific line
        if hasattr(error, 'lineno') and 0 < error.lineno <= len(lines):
            line = lines[error.lineno - 1]
            stripped = line.strip()
            
            # Common fixes
            if any(stripped.startswith(kw) for kw in ['if ', 'for ', 'while ', 'def ', 'class ', 'with ', 'try']):
                if not stripped.endswith(':'):
                    lines[error.lineno - 1] = line.rstrip() + ':'
            
            # Fix unclosed parentheses
            open_count = line.count('(') - line.count(')')
            if open_count > 0:
                lines[error.lineno - 1] += ')' * open_count
        
        # Then fix indentation
        return self.fix_indentation(lines)
    
    def fix_indentation(self, lines):
        """Fix indentation in Python code"""
        fixed_lines = []
        indent_level = 0
        prev_line = ""
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Empty lines
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Comments - preserve at current indent
            if stripped.startswith('#'):
                fixed_lines.append(' ' * indent_level + stripped)
                continue
            
            # Decorators
            if stripped.startswith('@'):
                fixed_lines.append(' ' * indent_level + stripped)
                continue
            
            # Handle dedent keywords
            if stripped in ['else:', 'elif:', 'except:', 'finally:']:
                indent_level = max(0, indent_level - 4)
                fixed_lines.append(' ' * indent_level + stripped)
                indent_level += 4
                continue
            elif any(stripped.startswith(kw + ' ') for kw in ['elif', 'except', 'finally']):
                indent_level = max(0, indent_level - 4)
                fixed_lines.append(' ' * indent_level + stripped)
                indent_level += 4
                continue
            
            # Regular lines
            fixed_lines.append(' ' * indent_level + stripped)
            
            # Adjust indentation for next line
            if stripped.endswith(':') and not stripped.startswith('#'):
                indent_level += 4
            elif stripped in ['pass', 'break', 'continue'] or \
                 any(stripped.startswith(kw) for kw in ['return', 'raise', 'yield']):
                # Check next line to see if we should dedent
                if i + 1 < len(lines):
                    next_stripped = lines[i + 1].strip()
                    if next_stripped and not next_stripped.startswith('#'):
                        if any(next_stripped.startswith(kw) for kw in ['def ', 'class ', '@']):
                            indent_level = 0
                        elif any(next_stripped.startswith(kw) for kw in ['if ', 'for ', 'while ', 'with ']):
                            if indent_level > 4:
                                indent_level -= 4
            
            # Reset indent for top-level definitions
            if stripped.startswith(('def ', 'class ')) and indent_level > 4:
                # This might be a top-level definition
                if i == 0 or (i > 0 and not lines[i-1].strip()):
                    indent_level = 0
                    fixed_lines[-1] = stripped
                    if stripped.endswith(':'):
                        indent_level = 4
            
            prev_line = stripped
        
        return '\n'.join(fixed_lines)
    
    def process_directory(self, directory):
        """Process all Python files in directory"""
        for root, dirs, files in os.walk(directory):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    self.total += 1
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, directory)
                    
                    result = self.fix_file(filepath)
                    
                    # Progress output
                    if result == "valid":
                        print(f"✅ {rel_path}")
                    elif result == "fixed":
                        print(f"🔧 {rel_path} - FIXED")
                    elif result == "failed":
                        print(f"❌ {rel_path} - FAILED")
                    elif result == "empty":
                        print(f"⚪ {rel_path} - Empty")
                    else:
                        print(f"⚠️  {rel_path} - Error")


def main():
    """Main function"""
    directory = "/workspace/mlTrainer3_complete"
    
    print("🚀 Fixing ALL Python files in mlTrainer3")
    print("=" * 60)
    
    fixer = CompleteFixer()
    fixer.process_directory(directory)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total Python files: {fixer.total}")
    print(f"Already valid: {fixer.already_valid}")
    print(f"Fixed: {fixer.fixed}")
    print(f"Failed: {fixer.failed}")
    
    if fixer.errors:
        print("\n❌ Failed files:")
        for filepath, error in fixer.errors[:10]:
            rel_path = os.path.relpath(filepath, directory)
            print(f"  - {rel_path}: {error}")
        if len(fixer.errors) > 10:
            print(f"  ... and {len(fixer.errors) - 10} more")
    
    # Test critical files
    print("\n🧪 Testing critical files...")
    critical_files = [
        'mltrainer_unified_chat.py',
        'app.py',
        'launch_mltrainer.py',
        'mltrainer_models.py'
    ]
    
    for file in critical_files:
        filepath = os.path.join(directory, file)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    ast.parse(f.read())
                print(f"✅ {file}")
            except SyntaxError as e:
                print(f"❌ {file} - Line {e.lineno}: {e.msg}")
    
    print("\n✅ Processing complete!")
    print(f"📁 Fixed files are in: {directory}")

if __name__ == "__main__":
    main()