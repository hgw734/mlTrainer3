
import os
import sys
import json
import importlib.util


def validate_startup():
    """Validate system before startup"""
    print("🔍 Validating system startup...")

    issues = []

    # 1. Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python version too old: {sys.version}")

    # 2. Check critical files
    critical_files = ['main.py', 'ai_config.json']
    for file in critical_files:
        if not os.path.exists(file):
            issues.append(f"Missing critical file: {file}")

    # 3. Check critical imports
    critical_imports = ['flask', 'pandas', 'numpy']
    for module in critical_imports:
        if importlib.util.find_spec(module) is None:
            issues.append(f"Missing critical module: {module}")

    # 4. Check configuration
    if os.path.exists('ai_config.json'):
        try:
            with open('ai_config.json', 'r') as f:
                config = json.load(f)

            if not config.get('polygon', {}).get('api_key'):
                issues.append("Missing Polygon API key")
            if not config.get('fred', {}).get('api_key'):
                issues.append("Missing FRED API key")

        except Exception as e:
            issues.append(f"Invalid ai_config.json: {e}")

    # 5. Check for port conflicts (skip if we're already running)
    try:
        import socket
        # Only check if we're not already running our app
        if not os.environ.get('FLASK_PORT'):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                result = s.connect_ex(('0.0.0.0', 5000))
                if result == 0:
                    issues.append(
                        "Port 5000 already in use by external process")
    except BaseException:
        pass

    if issues:
        print("❌ Startup validation failed:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    else:
        print("✅ Startup validation passed")
        return True


if __name__ == "__main__":
    validate_startup()
