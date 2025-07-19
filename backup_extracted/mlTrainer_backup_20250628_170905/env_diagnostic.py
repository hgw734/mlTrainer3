import os
import subprocess
from datetime import datetime
import requests

LOG_FILE = "env_diagnostic_log.txt"


def log(msg):
    timestamp = datetime.now().isoformat()
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def check_and_fix_packages():
    log("🔍 Checking package dependencies...")
    try:
        result = subprocess.run(
            ['pip', 'check'], capture_output=True, text=True, timeout=10)
        if result.stdout.strip():
            log("❌ Package issues detected:")
            log(result.stdout.strip())
            for line in result.stdout.strip().splitlines():
                pkg = line.split()[0]
                log(f"🔄 Reinstalling: {pkg}")
                subprocess.run(
                    ['pip', 'install', '--force-reinstall', pkg], timeout=30)
        else:
            log("✅ Packages OK")
    except subprocess.TimeoutExpired:
        log("❌ Timeout during package check or reinstall")


def verify_core_files():
    for file in ["main.py", ".replit", "requirements.txt"]:
        if not os.path.exists(file):
            log(f"❌ Missing: {file}")
            if file == ".replit":
                with open(".replit", "w") as f:
                    f.write('run = "python3 main.py"\nlanguage = "python3"\n')
                log("✅ .replit recreated")
            elif file == "requirements.txt":
                with open("requirements.txt", "w") as f:
                    f.write("flask\nanthropic\nrequests\n")
                log("✅ requirements.txt created")
            elif file == "main.py":
                with open("main.py", "w") as f:
                    f.write('print("⚠ Placeholder main.py — replace soon")\n')
                log("✅ main.py placeholder added")
        else:
            log(f"✅ Found: {file}")


def check_env_vars():
    for var in ["POLYGON_API_KEY", "FRED_API_KEY", "ANTHROPIC_API_KEY"]:
        log(f"{'✅' if var in os.environ else '❌'} {var}")


def check_internet():
    try:
        r = requests.get("https://httpbin.org/get", timeout=5)
        if r.status_code == 200:
            log("✅ Internet connection OK")
        else:
            log(f"⚠️ Unexpected HTTP status: {r.status_code}")
    except Exception as e:
        log(f"❌ Internet failed: {e}")


def check_anthropic_patch():
    try:
        import anthropic
        anthropic.Anthropic()
        log("✅ Anthropic SDK loaded successfully")
    except TypeError as e:
        if "proxies" in str(e):
            log("⚠ Bad proxy arg passed — fix constructor")
        else:
            log(f"❌ Anthropic error: {e}")
    except Exception as e:
        log(f"❌ Anthropic init failed: {e}")


def print_summary():
    log("\n✅ DONE — check env_diagnostic_log.txt")


if __name__ == "__main__":
    log("🚀 Running Safe Environment Diagnostic")
    check_and_fix_packages()
    verify_core_files()
    check_env_vars()
    check_internet()
    check_anthropic_patch()
    print_summary()
