
import os
import sys
import json
import importlib.util
from datetime import datetime


def log_error(error_type, message, details=None):
    """Log errors with timestamp and details"""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] ❌ {error_type}: {message}")
    if details:
        print(f"    Details: {details}")


def check_python_environment():
    """Check Python version and environment"""
    print("🐍 Python Environment Analysis:")
    print(f"✅ Python Version: {sys.version}")
    print(f"✅ Python Executable: {sys.executable}")
    print(f"✅ Python Path: {sys.path[0]}")

    # Check if we're in the right environment
    if "nix" in sys.executable.lower():
        log_error(
            "Environment",
            "Python is running from Nix store, may conflict with modules config")
    else:
        print("✅ Python environment looks compatible with modules config")


def check_critical_imports():
    """Check if critical modules can be imported"""
    print("\n📦 Critical Import Analysis:")

    critical_modules = [
        'flask', 'streamlit', 'pandas', 'numpy', 'tensorflow',
        'sklearn', 'requests', 'anthropic', 'json', 'os'
    ]

    failed_imports = []

    for module in critical_modules:
        try:
            spec = importlib.util.find_spec(module)
            if spec is None:
                failed_imports.append(module)
                log_error("Import", f"Module '{module}' not found")
            else:
                print(f"✅ {module}: Available")
        except Exception as e:
            failed_imports.append(module)
            log_error("Import", f"Error checking '{module}'", str(e))

    return failed_imports


def check_file_structure():
    """Check for missing critical files and circular imports"""
    print("\n📁 File Structure Analysis:")

    # Check critical files
    critical_files = [
        'main.py', 'ai_config.json', 'requirements.txt',
        'core/system_router.py', 'core/mlTrainer_engine.py',
        'ml/ml_manager.py', 'monitoring/health_monitor.py'
    ]

    missing_files = []
    for file_path in critical_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            log_error("File", f"Missing critical file: {file_path}")
        else:
            print(f"✅ {file_path}: Found")

    # Check for duplicate folders (common cause of import issues)
    print("\n🔍 Checking for duplicate folders:")
    folders = {}
    for item in os.listdir('.'):
        if os.path.isdir(item) and not item.startswith(
                '.') and item != 'backup_before_cleanup':
            folder_name = item.lower()
            if folder_name in folders:
                log_error(
                    "Duplicate",
                    f"Duplicate folder detected: {item} and {folders[folder_name]}")
            else:
                folders[folder_name] = item
                print(f"✅ {item}: Unique")

    return missing_files


def check_configuration_files():
    """Check configuration files for issues"""
    print("\n⚙️ Configuration Analysis:")

    # Check ai_config.json
    if os.path.exists('ai_config.json'):
        try:
            with open('ai_config.json', 'r') as f:
                config = json.load(f)

            required_keys = ['polygon', 'fred', 'claude']
            missing_keys = []

            for key in required_keys:
                if key not in config:
                    missing_keys.append(key)
                elif not config[key].get('api_key'):
                    missing_keys.append(f"{key}.api_key")

            if missing_keys:
                log_error("Config", f"Missing API keys: {missing_keys}")
            else:
                print("✅ ai_config.json: All API keys present")

        except json.JSONDecodeError as e:
            log_error("Config", "ai_config.json is not valid JSON", str(e))
        except Exception as e:
            log_error("Config", "Error reading ai_config.json", str(e))
    else:
        log_error("Config", "ai_config.json not found")


def check_port_conflicts():
    """Check for port conflicts"""
    print("\n🔌 Port Analysis:")

    try:
        import socket

        # Check common ports
        test_ports = [5000, 8501, 8080, 3000]

        for port in test_ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                result = s.connect_ex(('0.0.0.0', port))
                if result == 0:
                    log_error("Port", f"Port {port} is already in use")
                else:
                    print(f"✅ Port {port}: Available")

    except Exception as e:
        log_error("Port", "Error checking ports", str(e))


def check_runtime_errors():
    """Check for potential runtime errors in main files"""
    print("\n🔧 Runtime Error Analysis:")

    # Check main.py syntax
    try:
        with open('main.py', 'r') as f:
            content = f.read()

        # Try to compile the code
        compile(content, 'main.py', 'exec')
        print("✅ main.py: Syntax OK")

    except SyntaxError as e:
        log_error("Syntax", f"Syntax error in main.py line {e.lineno}", str(e))
    except Exception as e:
        log_error("Runtime", "Error checking main.py", str(e))


def check_workflow_config():
    """Check workflow configuration"""
    print("\n🔄 Workflow Analysis:")

    if os.path.exists('.replit'):
        try:
            with open('.replit', 'r') as f:
                content = f.read()

            print("✅ .replit file contents:")
            print(content)

            # Check for conflicting configurations
            if 'nix' in content.lower() and 'modules' in content.lower():
                log_error(
                    "Config",
                    "Potential conflict: Both Nix and modules configuration detected")

        except Exception as e:
            log_error("Config", "Error reading .replit", str(e))


def run_comprehensive_analysis():
    """Run all error analysis checks"""
    print("🔍 COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 50)

    check_python_environment()
    failed_imports = check_critical_imports()
    missing_files = check_file_structure()
    check_configuration_files()
    check_port_conflicts()
    check_runtime_errors()
    check_workflow_config()

    print("\n" + "=" * 50)
    print("📋 SUMMARY OF ISSUES:")

    if failed_imports:
        print(f"❌ Failed imports: {len(failed_imports)} modules")
        print("   Solution: Check requirements.txt and run package installation")

    if missing_files:
        print(f"❌ Missing files: {len(missing_files)} critical files")
        print("   Solution: Restore missing files from backup or recreate them")

    print("\n🔧 RECOMMENDED ACTIONS:")
    print("1. Fix the workflow configuration (create new Python-based workflow)")
    print("2. Install missing packages if any")
    print("3. Resolve any configuration conflicts")
    print("4. Test the application startup")


if __name__ == "__main__":
    run_comprehensive_analysis()
