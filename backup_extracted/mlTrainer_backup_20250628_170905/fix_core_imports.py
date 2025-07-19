import os

ROOT_DIR = "."  # Start from the project root
CORE_MODULES = [
    "compliance_mode",
    "immutable_gateway",
    "fmt_engine",
    "mlTrainer_client_wrapper",
    "routes",
    "system_router"]
updated_files = []


def is_valid_py_file(filepath):
    return (
        filepath.endswith(".py")
        and "__pycache__" not in filepath
        and "/." not in filepath
    )


for subdir, _, files in os.walk(ROOT_DIR):
    for file in files:
        if not is_valid_py_file(file):
            continue

        filepath = os.path.join(subdir, file)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError):
            continue  # Skip unreadable or binary files

        original = content
        for module in CORE_MODULES:
            # Replace both `from xyz import` and `import xyz`
            content = content.replace(
                f"from {module} import",
                f"from core.{module} import")
            content = content.replace(
                f"import {module}", f"import core.{module}")

        if content != original:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            updated_files.append(filepath)

# Report
if updated_files:
    print("✅ Updated the following files to use core.* routing:")
    for f in updated_files:
        print(f" - {f}")
else:
    print("✅ No changes needed. All imports already properly routed through core.")
