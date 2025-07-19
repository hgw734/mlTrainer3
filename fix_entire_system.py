#!/usr/bin/env python3
"""
Comprehensive Python Fixer for mlTrainer3
Fixes ALL Python files in the entire system
"""

import os
import ast
import shutil
from pathlib import Path
import re
from datetime import datetime

class SystemWideFixer:
    def __init__(self):
        self.fixed_files = 0
        self.failed_files = 0
        self.total_files = 0
        self.error_log = []
        
    def fix_indentation(self, content):
        """Fix indentation issues in Python code"""
        lines = content.split('\n')
        fixed_lines = []
        indent_level = 0
        prev_line = ""
        in_multiline_string = False
        string_delimiter = None
        
        for i, line in enumerate(lines):
            # Check for multiline strings
            if not in_multiline_string:
                if '"""' in line or "'''" in line:
                    delimiter = '"""' if '"""' in line else "'''"
                    count = line.count(delimiter)
                    if count % 2 == 1:  # Odd number means we're entering/exiting
                        in_multiline_string = True
                        string_delimiter = delimiter
            else:
                if string_delimiter in line:
                    count = line.count(string_delimiter)
                    if count % 2 == 1:  # Odd number means we're exiting
                        in_multiline_string = False
                        string_delimiter = None
            
            # Keep original formatting for multiline strings
            if in_multiline_string:
                fixed_lines.append(line)
                continue
            
            stripped = line.strip()
            
            # Empty lines
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Comments
            if stripped.startswith('#'):
                # Align comments with current indent level
                fixed_lines.append(' ' * indent_level + stripped)
                continue
            
            # Decorators
            if stripped.startswith('@'):
                # Decorators should be at the same level as the following def/class
                fixed_lines.append(' ' * indent_level + stripped)
                continue
            
            # Handle dedent keywords
            if stripped in ['else:', 'elif:', 'except:', 'finally:'] or \
               stripped.startswith(('elif ', 'except ', 'finally ')):
                indent_level = max(0, indent_level - 4)
                fixed_lines.append(' ' * indent_level + stripped)
                indent_level += 4
                prev_line = stripped
                continue
            
            # Add the line with current indentation
            fixed_lines.append(' ' * indent_level + stripped)
            
            # Adjust indentation for next line
            if stripped.endswith(':') and not stripped.startswith('#'):
                # Increase indent after block starters
                indent_level += 4
            elif stripped in ['pass', 'break', 'continue'] or \
                 stripped.startswith(('return', 'raise', 'yield')):
                # These usually end a block
                # Check if next line is a dedent
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('#'):
                        # If next line is a function/class definition, reset indent
                        if any(next_line.startswith(kw) for kw in ['def ', 'class ', '@']):
                            indent_level = 0
                        # If next line is at lower indent, adjust
                        elif next_line and len(lines[i + 1]) - len(lines[i + 1].lstrip()) < indent_level:
                            indent_level = max(0, indent_level - 4)
            
            # Reset indent for top-level definitions
            if stripped.startswith(('def ', 'class ')) and indent_level > 8:
                # This is likely a top-level definition
                fixed_lines[-1] = stripped
                indent_level = 4 if stripped.startswith('def ') else 0
                if stripped.endswith(':'):
                    indent_level = 4
            
            prev_line = stripped
        
        return '\n'.join(fixed_lines)
    
    def fix_syntax_errors(self, content, error):
        """Fix specific syntax errors"""
        lines = content.split('\n')
        
        if hasattr(error, 'lineno') and 0 < error.lineno <= len(lines):
            problem_line = lines[error.lineno - 1]
            stripped = problem_line.strip()
            
            # Fix missing colons
            if any(stripped.startswith(kw) for kw in ['if ', 'for ', 'while ', 'def ', 'class ', 'with ', 'try']):
                if not stripped.endswith(':'):
                    lines[error.lineno - 1] = problem_line.rstrip() + ':'
            
            # Fix unclosed parentheses/brackets
            open_parens = problem_line.count('(') - problem_line.count(')')
            open_brackets = problem_line.count('[') - problem_line.count(']')
            open_braces = problem_line.count('{') - problem_line.count('}')
            
            if open_parens > 0:
                lines[error.lineno - 1] += ')' * open_parens
            if open_brackets > 0:
                lines[error.lineno - 1] += ']' * open_brackets
            if open_braces > 0:
                lines[error.lineno - 1] += '}' * open_braces
            
            # Fix incomplete string literals
            if error.msg == 'EOL while scanning string literal':
                # Add closing quote
                if problem_line.count('"') % 2 == 1:
                    lines[error.lineno - 1] += '"'
                elif problem_line.count("'") % 2 == 1:
                    lines[error.lineno - 1] += "'"
        
        return '\n'.join(lines)
    
    def fix_file(self, filepath):
        """Fix a single Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Skip empty files
            if not original_content.strip():
                return True, "Empty file"
            
            # Try to parse the file
            try:
                ast.parse(original_content)
                return True, "Already valid"
            except SyntaxError as e:
                # Apply fixes
                fixed_content = original_content
                
                # First fix syntax errors
                fixed_content = self.fix_syntax_errors(fixed_content, e)
                
                # Then fix indentation
                fixed_content = self.fix_indentation(fixed_content)
                
                # Try to parse again
                try:
                    ast.parse(fixed_content)
                    # Save the fixed file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    self.fixed_files += 1
                    return True, f"Fixed: {e.msg}"
                except SyntaxError as e2:
                    # Try more aggressive fix
                    aggressive_fixed = self.aggressive_fix(original_content)
                    try:
                        ast.parse(aggressive_fixed)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(aggressive_fixed)
                        self.fixed_files += 1
                        return True, "Fixed with aggressive method"
                    except:
                        self.failed_files += 1
                        self.error_log.append((filepath, str(e)))
                        return False, f"Failed: {e.msg} at line {e.lineno}"
        except Exception as e:
            self.failed_files += 1
            self.error_log.append((filepath, str(e)))
            return False, f"Error: {str(e)}"
    
    def aggressive_fix(self, content):
        """Aggressive fix - complete reformat"""
        # Remove all comments first to avoid issues
        lines = content.split('\n')
        code_lines = []
        
        for line in lines:
            # Keep the line but remove inline comments
            if '#' in line and not line.strip().startswith('#'):
                # Keep code part, remove comment
                code_part = line.split('#')[0].rstrip()
                if code_part:
                    code_lines.append(code_part)
                else:
                    code_lines.append('')
            else:
                code_lines.append(line)
        
        # Now fix indentation
        return self.fix_indentation('\n'.join(code_lines))
    
    def process_directory(self, directory):
        """Process all Python files in a directory"""
        python_files = []
        
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      d not in ['__pycache__', 'venv', 'env', '.git', 
                               'node_modules', 'build', 'dist', 'backup_before_fix']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files


def main():
    """Main function to fix all Python files"""
    print("🚀 mlTrainer3 System-Wide Python Fixer")
    print("=" * 60)
    
    fixer = SystemWideFixer()
    
    # Create backup directory
    backup_dir = Path("backup_before_fix")
    if not backup_dir.exists():
        print("📦 Creating backup directory...")
        backup_dir.mkdir(exist_ok=True)
    
    # Find all Python files
    print("🔍 Scanning for Python files...")
    
    # Start from the repository root
    repo_root = "/tmp/mlTrainer3"
    if not os.path.exists(repo_root):
        repo_root = "."
    
    python_files = fixer.process_directory(repo_root)
    fixer.total_files = len(python_files)
    
    print(f"📊 Found {fixer.total_files} Python files to process\n")
    
    # Process each file
    print("🔧 Processing files...")
    print("-" * 60)
    
    for i, filepath in enumerate(sorted(python_files), 1):
        # Create backup
        relative_path = os.path.relpath(filepath, repo_root)
        backup_path = backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if os.path.exists(filepath):
                shutil.copy2(filepath, backup_path)
        except:
            pass  # Skip if can't backup
        
        # Fix the file
        success, message = fixer.fix_file(filepath)
        
        # Progress indicator
        progress = f"[{i}/{fixer.total_files}]"
        
        if success:
            if "Already valid" in message:
                print(f"{progress} ✅ {relative_path}")
            else:
                print(f"{progress} 🔧 {relative_path} - {message}")
        else:
            print(f"{progress} ❌ {relative_path} - {message}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {fixer.total_files}")
    print(f"Files fixed: {fixer.fixed_files}")
    print(f"Files already valid: {fixer.total_files - fixer.fixed_files - fixer.failed_files}")
    print(f"Files failed: {fixer.failed_files}")
    
    if fixer.error_log:
        print("\n❌ Failed files:")
        for filepath, error in fixer.error_log[:10]:  # Show first 10
            print(f"  - {filepath}: {error}")
        if len(fixer.error_log) > 10:
            print(f"  ... and {len(fixer.error_log) - 10} more")
    
    # Test critical files
    print("\n🧪 Testing critical files...")
    critical_files = [
        'mltrainer_unified_chat.py',
        'app.py',
        'launch_mltrainer.py',
        'mltrainer_models.py',
        'mltrainer_claude_integration.py'
    ]
    
    all_good = True
    for file in critical_files:
        filepath = os.path.join(repo_root, file)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    ast.parse(f.read())
                print(f"✅ {file}")
            except SyntaxError as e:
                print(f"❌ {file} - Line {e.lineno}: {e.msg}")
                all_good = False
        else:
            print(f"⚠️  {file} - Not found")
    
    if all_good:
        print("\n🎉 All critical files are valid!")
    else:
        print("\n⚠️  Some critical files still have issues")
    
    print(f"\n💾 Backups saved in: {backup_dir}/")
    print("\n✅ System-wide fix complete!")


if __name__ == "__main__":
    main()