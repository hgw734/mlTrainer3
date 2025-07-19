#!/usr/bin/env python3
"""
Process all mlTrainer3 files and copy fixed versions to workspace
"""

import os
import shutil
import ast
from pathlib import Path

# Import the fixer class
import sys
sys.path.append('/workspace')
from fix_entire_system import SystemWideFixer

def copy_fixed_files():
    """Copy all fixed files from /tmp/mlTrainer3 to /workspace/mlTrainer3_fixed"""
    
    source_dir = "/tmp/mlTrainer3"
    target_dir = "/workspace/mlTrainer3_fixed"
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"📁 Copying from {source_dir} to {target_dir}")
    
    # Copy all files
    for root, dirs, files in os.walk(source_dir):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        # Calculate relative path
        rel_path = os.path.relpath(root, source_dir)
        target_root = os.path.join(target_dir, rel_path)
        
        # Create directories
        os.makedirs(target_root, exist_ok=True)
        
        # Copy files
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_root, file)
            
            try:
                shutil.copy2(source_file, target_file)
                print(f"✅ Copied: {rel_path}/{file}")
            except Exception as e:
                print(f"❌ Failed to copy {file}: {e}")
    
    print("\n✅ All files copied to workspace!")
    
    # Create a summary of what was fixed
    create_fix_summary(target_dir)

def create_fix_summary(target_dir):
    """Create a summary of all Python files and their status"""
    
    summary_file = os.path.join(target_dir, "FIX_SUMMARY.md")
    
    with open(summary_file, 'w') as f:
        f.write("# mlTrainer3 Fix Summary\n\n")
        f.write("## Overview\n\n")
        f.write("This directory contains the fixed version of all mlTrainer3 files.\n")
        f.write("All Python files have been processed to fix syntax and indentation errors.\n\n")
        
        f.write("## File Status\n\n")
        
        # Check all Python files
        valid_count = 0
        total_count = 0
        
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    total_count += 1
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, target_dir)
                    
                    try:
                        with open(filepath, 'r') as pf:
                            ast.parse(pf.read())
                        f.write(f"- ✅ `{rel_path}`\n")
                        valid_count += 1
                    except SyntaxError as e:
                        f.write(f"- ❌ `{rel_path}` - Line {e.lineno}: {e.msg}\n")
        
        f.write(f"\n## Summary\n\n")
        f.write(f"- Total Python files: {total_count}\n")
        f.write(f"- Valid files: {valid_count}\n")
        f.write(f"- Files with errors: {total_count - valid_count}\n")
        f.write(f"\n## Next Steps\n\n")
        f.write("1. Review the fixed files\n")
        f.write("2. Copy the files you need back to your main repository\n")
        f.write("3. Test the application\n")
        f.write("4. Push to GitHub to trigger deployment\n")
    
    print(f"\n📄 Summary written to: {summary_file}")

def main():
    """Main function"""
    print("🚀 Processing and copying all mlTrainer3 files\n")
    
    # First, run the fixer on the /tmp/mlTrainer3 directory
    print("Step 1: Fixing all Python files...")
    print("=" * 60)
    
    # Run the fixer
    os.system("cd /tmp/mlTrainer3 && python3 /workspace/fix_entire_system.py")
    
    print("\n" + "=" * 60)
    print("Step 2: Copying fixed files to workspace...")
    print("=" * 60)
    
    # Copy all files
    copy_fixed_files()
    
    print("\n✅ Complete! Check /workspace/mlTrainer3_fixed for all fixed files.")

if __name__ == "__main__":
    main()