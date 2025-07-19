
import os
import hashlib
from collections import defaultdict

# Folders to completely ignore during duplicate search
IGNORE_FOLDERS = {
    "backup_before_cleanup",
    "__pycache__",
    ".git",
    ".replit",
    "venv",
    "node_modules"}


def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except BaseException:
        return None


def find_duplicate_files_and_folders():
    """Find duplicate files and folders in root directory, excluding backup folders."""

    # Track folder names
    folder_names = defaultdict(list)

    # Track file names and their hashes
    file_names = defaultdict(list)
    file_hashes = defaultdict(list)

    print("🔍 Scanning for duplicates in root directory (excluding backup_before_cleanup)...\n")

    # Scan root directory only, excluding ignored folders
    root_items = []
    for item in os.listdir('.'):
        if item.startswith('.') or item in IGNORE_FOLDERS:
            continue
        root_items.append(item)

        if os.path.isdir(item):
            folder_names[item.lower()].append(item)
        elif os.path.isfile(item):
            file_names[item.lower()].append(item)
            file_hash = get_file_hash(item)
            if file_hash:
                file_hashes[file_hash].append(item)

    # Report duplicate folder names
    print("📁 DUPLICATE FOLDER NAMES:")
    found_folder_dups = False
    for folder_key, folders in folder_names.items():
        if len(folders) > 1:
            print(f"  • {folder_key}: {folders}")
            found_folder_dups = True

    if not found_folder_dups:
        print("  ✅ No duplicate folder names found")

    print("\n📄 DUPLICATE FILE NAMES:")
    found_file_name_dups = False
    for file_key, files in file_names.items():
        if len(files) > 1:
            print(f"  • {file_key}: {files}")
            found_file_name_dups = True

    if not found_file_name_dups:
        print("  ✅ No duplicate file names found")

    print("\n🔗 DUPLICATE FILE CONTENTS (same hash):")
    found_content_dups = False
    for file_hash, files in file_hashes.items():
        if len(files) > 1:
            print(f"  • Same content: {files}")
            found_content_dups = True

    if not found_content_dups:
        print("  ✅ No duplicate file contents found")

    print("\n" + "="*50)
    print("SUMMARY OF ACTUAL DUPLICATES IN ROOT (excluding backup):")

    # List all actual duplicate items found
    actual_duplicates = []

    # Add folder name duplicates
    for folder_key, folders in folder_names.items():
        if len(folders) > 1:
            # Keep first, mark others as duplicates
            actual_duplicates.extend(folders[1:])

    # Add file name duplicates
    for file_key, files in file_names.items():
        if len(files) > 1:
            # Keep first, mark others as duplicates
            actual_duplicates.extend(files[1:])

    if actual_duplicates:
        print(f"❌ Found {len(actual_duplicates)} duplicate items:")
        for item in sorted(actual_duplicates):
            print(f"  • {item}")

        print(f"\n💡 To clean up these duplicates, you could run:")
        print(f"   rm -rf {' '.join(sorted(actual_duplicates))}")
        print(f"   (⚠️  Review the list carefully before executing!)")
    else:
        print("✅ No duplicate items found in root directory")

    print(f"\n📊 Total items scanned: {len(root_items)}")
    print(f"📁 Folders: {len([f for f in root_items if os.path.isdir(f)])}")
    print(f"📄 Files: {len([f for f in root_items if os.path.isfile(f)])}")


if __name__ == "__main__":
    find_duplicate_files_and_folders()
