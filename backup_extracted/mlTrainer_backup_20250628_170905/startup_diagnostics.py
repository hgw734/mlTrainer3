
import os
import json
import importlib.util
import platform
from datetime import datetime
from collections import Counter

REQUIRED_FILES = ['ai_config.json', 'fmt_memory.json']
REQUIRED_FOLDERS = ['core', 'ml', 'monitoring', 'data_sources']
REQUIRED_PACKAGES = ['streamlit', 'tensorflow', 'pandas', 'numpy']

# Exclude any folder starting with these prefixes or exact matches
EXCLUDE_FOLDERS_PREFIX = ['backup', '__pycache__', '.git', '.replit']
EXCLUDE_FOLDERS_EXACT = {
    'backup_before_cleanup',
    'venv',
    'node_modules',
    '.pytest_cache'}

CRITICAL_FILES = {
    "core": [
        "compliance_mode.py",
        "immutable_gateway.py",
        "error_monitor.py",
        "startup_validator.py",
        "fmt_engine.py",
        "fmt_config.py",
        "compliance_enforcer.py"
    ],
    "ml": [
        "ml_manager.py",
        "ml_bridge.py",
        "model_registry.py",
        "walkforward.py",
        "trainer_lstm.py",
        "trainer_transformer.py",
        "trainer_meta.py",
        "strategy_selector.py",
        "regime_classifier.py"
    ],
    "monitoring": [
        "health_monitor.py",
        "error_monitor.py",
        "training_logger.py",
        "system_status.py"
    ],
    "data_sources": [
        "polygon_handler.py",
        "fred_handler.py",
        "quiver_handler.py",
        "data_fetcher.py"
    ]
}


def should_exclude_folder(folder_name):
    """Check if a folder should be excluded from scanning."""
    # Check exact matches first
    if folder_name in EXCLUDE_FOLDERS_EXACT:
        return True

    # Check prefix matches
    return any(folder_name.startswith(prefix)
               for prefix in EXCLUDE_FOLDERS_PREFIX)


def check_python_version():
    version = platform.python_version()
    try:
        major, minor, patch = map(int, version.split('.'))
        if (major, minor, patch) >= (3, 8, 0):
            return f"✅ Python {version} OK"
        else:
            return f"❌ Python version too low: {version}"
    except Exception:
        return f"❌ Unable to parse Python version: {version}"


def check_required_files():
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(f)]
    if missing:
        return f"❌ Missing files: {missing}"
    return f"✅ Required files: {', '.join(REQUIRED_FILES)}"


def check_required_folders():
    missing = [f for f in REQUIRED_FOLDERS if not os.path.isdir(f)]
    if missing:
        return f"❌ Missing folders: {missing}"
    return f"✅ Required folders: {', '.join(REQUIRED_FOLDERS)}"


def check_critical_files_in_folders():
    issues = []
    for folder, files in CRITICAL_FILES.items():
        if not os.path.isdir(folder):
            issues.append(f"❌ Folder missing: {folder}")
            continue
        existing_files = set(os.listdir(folder))
        missing_files = [f for f in files if f not in existing_files]
        if missing_files:
            issues.append(f"❌ Missing files in {folder}: {missing_files}")
    if issues:
        return "\n".join(issues)
    return "✅ All critical files found in core folders"


def check_python_packages():
    missing = []
    for package in REQUIRED_PACKAGES:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    if missing:
        return f"❌ Missing Python packages: {missing}"
    return "✅ All required Python packages installed"


def list_installed_packages():
    try:
        import pkg_resources
        installed = sorted([str(d).split()[0]
                           for d in pkg_resources.working_set])
        return f"📦 Installed packages: {installed}"
    except ImportError:
        return "⚠️ pkg_resources not available, cannot list packages."


def check_api_keys():
    try:
        with open("ai_config.json", "r") as f:
            config = json.load(f)
        required_keys = ['polygon', 'fred', 'claude']
        missing_keys = []
        for key in required_keys:
            if key not in config or not config[key].get("api_key"):
                missing_keys.append(key)
        if missing_keys:
            return f"❌ Missing API key(s) for: {missing_keys}"
        return "✅ API keys found in ai_config.json"
    except Exception as e:
        return f"❌ Failed to load ai_config.json: {e}"


def check_compliance_mode():
    try:
        from core.compliance_mode import is_compliance_enabled
        if is_compliance_enabled():
            return "✅ Compliance mode loaded and enabled"
        else:
            return "❌ Compliance mode loaded but NOT enabled"
    except Exception as e:
        return f"❌ Failed to load compliance system: {e}"


def check_memory_file():
    if not os.path.exists("fmt_memory.json"):
        return "❌ fmt_memory.json not found"
    try:
        with open("fmt_memory.json", "r") as f:
            json.load(f)
        return "✅ fmt_memory.json is present and readable"
    except Exception as e:
        return f"❌ fmt_memory.json unreadable: {e}"


def check_duplicate_modules():
    all_dirs = []
    excluded_count = 0

    for root, dirs, _ in os.walk("."):
        # Filter out excluded directories using the enhanced exclusion logic
        original_count = len(dirs)
        dirs[:] = [d for d in dirs if not should_exclude_folder(d)]
        excluded_count += original_count - len(dirs)

        for d in dirs:
            all_dirs.append(d)

    counts = Counter(all_dirs)
    dups = [folder for folder, count in counts.items() if count > 1]

    result_lines = []
    if dups:
        result_lines.append(f"❌ Duplicate module folders found: {dups}")
    else:
        result_lines.append("✅ No duplicate module folders")

    result_lines.append(
        f"📁 Scanned {len(all_dirs)} folders, excluded {excluded_count} backup/cache folders")

    return "\n".join(result_lines)


def run_all_checks():
    print(
        f"🛠️ Running FMT2 Startup Diagnostics @ {datetime.utcnow().isoformat()}")
    print(check_python_version())
    print(check_required_files())
    print(check_required_folders())
    print(check_critical_files_in_folders())
    print(check_python_packages())
    print(list_installed_packages())
    print(check_api_keys())
    print(check_compliance_mode())
    print(check_memory_file())
    print(check_duplicate_modules())


if __name__ == "__main__":
    run_all_checks()
