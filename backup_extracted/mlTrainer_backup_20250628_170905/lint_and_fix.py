
import os
import ast
import subprocess
import autopep8
import shutil
from datetime import datetime

LOG_FOLDER = "lint_logs"


def log(message, log_lines):
    print(message)
    log_lines.append(message)


def find_python_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.py'):
                yield os.path.join(dirpath, file)


def check_ast_errors(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return None
    except (SyntaxError, IndentationError) as e:
        return f"{type(e).__name__} in {filepath} at line {e.lineno}: {e.msg}"
    except Exception as e:
        return f"Parsing error in {filepath}: {e}"


def autopep8_fix(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()
        fixed = autopep8.fix_code(original, options={'aggressive': 2})
        if original != fixed:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed)
            return True
    except Exception:
        return False
    return False


def autoflake_fix(filepath):
    try:
        subprocess.run(['autoflake',
                        '--in-place',
                        '--remove-unused-variables',
                        '--remove-all-unused-imports',
                        filepath],
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


def flake8_check(filepath):
    try:
        result = subprocess.run(['flake8', filepath],
                                capture_output=True, text=True)
        if result.stdout.strip():
            return result.stdout.strip()
        return None
    except Exception as e:
        return f"flake8 failed on {filepath}: {e}"


def mypy_check(filepath):
    try:
        result = subprocess.run(
            ['mypy', filepath], capture_output=True, text=True)
        if result.stdout.strip() and "Success" not in result.stdout:
            return result.stdout.strip()
        return None
    except Exception as e:
        return f"mypy failed on {filepath}: {e}"


def write_log_file(log_lines):
    os.makedirs(LOG_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOG_FOLDER, f"log_{timestamp}.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        for line in log_lines:
            f.write(line + "\n")
    print(f"\n📄 Log written to: {log_file}")


def run_diagnostics(root="."):
    total_files = 0
    fixed_pep8 = 0
    removed_imports = 0
    syntax_errors = []
    flake_warnings = []
    mypy_warnings = []

    log_lines = []

    for filepath in find_python_files(root):
        total_files += 1
        log(f"\n🔍 Checking: {filepath}", log_lines)

        err = check_ast_errors(filepath)
        if err:
            syntax_errors.append(err)
            log(f"❌ {err}", log_lines)
        else:
            if autopep8_fix(filepath):
                fixed_pep8 += 1
                log("✅ autopep8 fixed", log_lines)

            if autoflake_fix(filepath):
                removed_imports += 1
                log("✅ autoflake cleaned", log_lines)

            flake_output = flake8_check(filepath)
            if flake_output:
                flake_warnings.append((filepath, flake_output))
                log("⚠️ flake8 issues:\n" + flake_output, log_lines)

            mypy_output = mypy_check(filepath)
            if mypy_output:
                mypy_warnings.append((filepath, mypy_output))
                log("⚠️ mypy issues:\n" + mypy_output, log_lines)
            else:
                log("✅ mypy OK", log_lines)

    log("\n📊 Summary Report", log_lines)
    log(f"  - Files checked: {total_files}", log_lines)
    log(f"  - autopep8 fixes: {fixed_pep8}", log_lines)
    log(f"  - Unused imports removed: {removed_imports}", log_lines)
    log(f"  - Syntax/Indentation errors: {len(syntax_errors)}", log_lines)
    log(f"  - flake8 warnings: {len(flake_warnings)}", log_lines)
    log(f"  - mypy issues: {len(mypy_warnings)}", log_lines)

    if syntax_errors:
        log("\n🚫 Syntax Issues:", log_lines)
        for e in syntax_errors:
            log("  - " + e, log_lines)

    if flake_warnings:
        log("\n⚠️ flake8 Style Warnings:", log_lines)
        for path, msg in flake_warnings:
            log(f"\n{path}:\n{msg}", log_lines)

    if mypy_warnings:
        log("\n🔍 mypy Type Warnings:", log_lines)
        for path, msg in mypy_warnings:
            log(f"\n{path}:\n{msg}", log_lines)

    write_log_file(log_lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fix and lint Python codebase with logging")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Folder to scan (default: .)")
    args = parser.parse_args()

    if not shutil.which("flake8") or not shutil.which("mypy"):
        print("⚠️ Please install flake8 and mypy via pip")
    else:
        run_diagnostics(args.path)
