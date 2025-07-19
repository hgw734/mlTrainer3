#!/usr/bin/env python3
"""
Fix critical mlTrainer3 files
"""

import os
import ast
import shutil

# List of critical files to fix
CRITICAL_FILES = [
    'mltrainer_unified_chat.py',
    'app.py',
    'launch_mltrainer.py',
    'mltrainer_models.py',
    'mltrainer_claude_integration.py',
    'mltrainer_chat.py',
    'config/models_config.py',
    'config/api_config.py',
    'core/unified_executor.py',
    'core/enhanced_background_manager.py',
    'goal_system.py',
    'mlagent_bridge.py',
    'recommendation_tracker.py',
    'virtual_portfolio_manager.py',
    'polygon_connector.py',
    'fred_connector.py',
    'telegram_notifier.py'
]

def fix_indentation(content):
    """Simple indentation fixer"""
    lines = content.split('\n')
    fixed = []
    indent = 0
    
    for line in lines:
        s = line.strip()
        
        if not s:
            fixed.append('')
            continue
            
        # Dedent keywords
        if s in ['else:', 'elif:', 'except:', 'finally:'] or s.startswith(('elif ', 'except ')):
            indent = max(0, indent - 4)
        
        # Add line
        fixed.append(' ' * indent + s)
        
        # Adjust indent
        if s.endswith(':') and not s.startswith('#'):
            indent += 4
        elif s in ['pass', 'break', 'continue'] or s.startswith('return'):
            indent = max(0, indent - 4)
        
        # Reset for top-level
        if s.startswith(('def ', 'class ', '@')) and indent > 8:
            indent = 0
            fixed[-1] = s
    
    return '\n'.join(fixed)

def process_file(source_path, target_path):
    """Process a single file"""
    try:
        # Read file
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if valid
        try:
            ast.parse(content)
            # Just copy if valid
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)
            return True, "Already valid"
        except SyntaxError as e:
            # Fix it
            fixed = fix_indentation(content)
            
            # Test fix
            try:
                ast.parse(fixed)
                # Save fixed version
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(fixed)
                return True, f"Fixed: {e.msg}"
            except:
                # Copy anyway for manual fix
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(source_path, target_path)
                return False, f"Needs manual fix: {e.msg}"
                
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    source_dir = "/tmp/mlTrainer3"
    target_dir = "/workspace/mlTrainer3_critical_fixed"
    
    print("🚀 Fixing critical mlTrainer3 files")
    print("=" * 60)
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Process each critical file
    success_count = 0
    
    for file in CRITICAL_FILES:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, file)
        
        if os.path.exists(source_path):
            success, message = process_file(source_path, target_path)
            
            if success:
                if "Already valid" in message:
                    print(f"✅ {file} - {message}")
                else:
                    print(f"🔧 {file} - {message}")
                success_count += 1
            else:
                print(f"❌ {file} - {message}")
        else:
            print(f"⚠️  {file} - Not found")
    
    # Also copy essential non-Python files
    essential_files = [
        'requirements.txt',
        'README.md',
        '.env.example',
        'Dockerfile',
        '.github/workflows/unified-ci-cd.yml'
    ]
    
    print("\n📄 Copying essential non-Python files...")
    
    for file in essential_files:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, file)
        
        if os.path.exists(source_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)
            print(f"✅ {file}")
    
    print(f"\n✅ Complete! Fixed {success_count}/{len(CRITICAL_FILES)} critical files")
    print(f"📁 Files saved to: {target_dir}")

if __name__ == "__main__":
    main()