import os

OUTPUT_FILE = "system_structure.txt"
EXCLUDE_DIRS = {"__pycache__", ".git", ".replit", "venv", "backup"}


def list_files(base_path, prefix=""):
    lines = []
    try:
        for name in sorted(os.listdir(base_path)):
            if name in EXCLUDE_DIRS:
                continue
            full_path = os.path.join(base_path, name)
            if os.path.isdir(full_path):
                lines.append(f"{prefix}[DIR]  {name}/")
                lines.extend(list_files(full_path, prefix + "    "))
            else:
                lines.append(f"{prefix}[FILE] {name}")
    except Exception as e:
        lines.append(f"{prefix}[ERROR] Could not access {base_path}: {e}")
    return lines


if __name__ == "__main__":
    structure = list_files(".")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(structure))
    print(f"✅ System structure saved to: {OUTPUT_FILE}")
