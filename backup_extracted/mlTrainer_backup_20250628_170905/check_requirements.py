import pkg_resources
import sys
import os

REQUIREMENTS_FILE = "requirements.txt"


def read_requirements(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()
            and not line.startswith("#")]


def check_requirements(requirements):
    installed_packages = {
        pkg.key: pkg.version for pkg in pkg_resources.working_set}
    errors = []

    for req in requirements:
        try:
            req_parsed = pkg_resources.Requirement.parse(req)
            name = req_parsed.project_name.lower()
            spec = str(req_parsed.specifier)

            if name not in installed_packages:
                errors.append(f"❌ MISSING: {req}")
            elif spec and not pkg_resources.require(req):
                installed_version = installed_packages[name]
                errors.append(
                    f"⚠️ VERSION MISMATCH: {name} {installed_version} (required {spec})")
            else:
                print(f"✅ OK: {req}")
        except Exception as e:
            errors.append(f"⚠️ COULD NOT PARSE: {req} → {e}")

    return errors


def main():
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"❌ File not found: {REQUIREMENTS_FILE}")
        sys.exit(1)

    print(f"📦 Checking packages listed in {REQUIREMENTS_FILE}...\n")
    reqs = read_requirements(REQUIREMENTS_FILE)
    issues = check_requirements(reqs)

    if issues:
        print("\n--- Issues Found ---")
        for issue in issues:
            print(issue)
        print("\n🔧 You can fix issues by running:")
        print("    pip install -r requirements.txt")
    else:
        print("\n✅ All packages are correctly installed with matching versions.")

# Optional auto-fix (uncomment to enable):
#     subprocess.run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])


if __name__ == "__main__":
    main()
