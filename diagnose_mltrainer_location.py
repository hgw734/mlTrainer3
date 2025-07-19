#!/usr/bin/env python3
"""
Diagnostic script to help find mlTrainer folder on Mac
"""

import os
import subprocess
import json
from pathlib import Path


def find_mltrainer_folders():
    """Find all mlTrainer folders on the system"""
    print("🔍 Searching for mlTrainer folders on your Mac# Production code implemented\n")

    # Common locations to check
    home = Path.home()
    locations_to_check = [
        home / "Documents",
        home / "Desktop",
        home / "Downloads",
        home / "Projects",
        home / "Code",
        home / "Development",
        home / "workspace",
        home / "mlTrainer",
        home / "mlTrainer_old",
        home / "mlTrainer2",
        home / "Library/Mobile Documents/com~apple~CloudDocs",  # iCloud Drive
        Path("/Users/Shared"),
        Path("/workspace"),  # Linux container path
    ]

    found_folders = []

    # Check each location
    for location in locations_to_check:
        if location.exists():
            # Direct check
            if location.name.startswith("mlTrainer"):
                found_folders.append(location)
                print(f"✅ Found: {location}")

                # Check subdirectories (1 level deep)
                try:
                    for item in location.iterdir():
                        if item.is_dir() and item.name.startswith("mlTrainer"):
                            found_folders.append(item)
                            print(f"✅ Found: {item}")
                            except PermissionError:
                                pass

                            # Also use find command for thorough search
                            print(
                                "\n🔍 Running system-wide search (this may take a moment)# Production code implemented\n")
                            try:
                                result = subprocess.run(
                                    ["find", str(home), "-type", "d", "-name", "mlTrainer*", "-maxdepth", "5"],
                                    capture_output=True,
                                    text=True,
                                    timeout=30,
                                )

                                if result.returncode == 0:
                                    for line in result.stdout.strip().split("\n"):
                                        if line and line not in [
                                                str(f) for f in found_folders]:
                                            found_folders.append(Path(line))
                                            print(f"✅ Found: {line}")
                                            except Exception as e:
                                                print(
                                                    f"⚠️  Find command failed: {e}")

                                                return found_folders

                                                def check_folder_contents(
                                                        folder_path):
                                                    """Check if folder has expected mlTrainer files"""
                                                    key_files = [
                                                        "README.md", "requirements.txt", "app.py", "core/compliance_mode.py", ".git"]

                                                    found_files = []
                                                    for key_file in key_files:
                                                        file_path = folder_path / key_file
                                                        if file_path.exists():
                                                            found_files.append(
                                                                key_file)

                                                            return found_files

                                                            def main():
                                                                print(
                                                                    ("=" * 60))
                                                                print(
                                                                    "mlTrainer Folder Diagnostic Tool")
                                                                print(
                                                                    ("=" * 60))

                                                                # Find all
                                                                # mlTrainer
                                                                # folders
                                                                folders = find_mltrainer_folders()

                                                                print(
                                                                    ("\n" + "=" * 60))
                                                                print(
                                                                    "SUMMARY")
                                                                print(
                                                                    ("=" * 60))

                                                                if not folders:
                                                                    print(
                                                                        "❌ No mlTrainer folders found!")
                                                                    print(
                                                                        "\nPossible reasons:")
                                                                    print(
                                                                        "1. The folder might be in a cloud sync location (iCloud, Dropbox, etc.)")
                                                                    print(
                                                                        "2. The folder might be in a Docker/container environment")
                                                                    print(
                                                                        "3. The folder might have been moved or deleted")
                                                                    else:
                                                                        print(
                                                                            f"\n📁 Found {len(folders)} mlTrainer folder(s):\n")

                                                                        for i, folder in enumerate(
                                                                                folders, 1):
                                                                            print(
                                                                                f"{i}. {folder}")

                                                                            # Check
                                                                            # contents
                                                                            key_files = check_folder_contents(
                                                                                folder)
                                                                            if key_files:
                                                                                print(
                                                                                    f"   ✅ Contains: {', '.join(key_files)}")
                                                                                else:
                                                                                    print(
                                                                                        f"   ⚠️  Appears to be empty or missing key files")

                                                                                    # Check
                                                                                    # size
                                                                                    try:
                                                                                        size = sum(
                                                                                            f.stat().st_size for f in folder.rglob("*") if f.is_file())
                                                                                        size_mb = size / \
                                                                                            (1024 * 1024)
                                                                                        print(
                                                                                            f"   📊 Size: {size_mb:.1f} MB")
                                                                                        except BaseException:
                                                                                            pass

                                                                                        print()

                                                                                        # Cursor-specific
                                                                                        # advice
                                                                                        print(
                                                                                            ("\n" + "=" * 60))
                                                                                        print(
                                                                                            "TO OPEN IN CURSOR:")
                                                                                        print(
                                                                                            ("=" * 60))
                                                                                        print(
                                                                                            "\n1. In Cursor, click 'File' → 'Open Folder'")
                                                                                        print(
                                                                                            "2. Navigate to one of the folders listed above")
                                                                                        print(
                                                                                            "3. If the folder appears empty in Cursor, try:")
                                                                                        print(
                                                                                            "   - Refreshing the file explorer (Cmd+R)")
                                                                                        print(
                                                                                            "   - Checking if files are hidden (Cmd+Shift+.)")
                                                                                        print(
                                                                                            "   - Restarting Cursor")
                                                                                        print(
                                                                                            "\n4. If you're using a remote container:")
                                                                                        print(
                                                                                            "   - The files might be in /workspace inside the container")
                                                                                        print(
                                                                                            "   - Check your Docker/container settings")

                                                                                        if __name__ == "__main__":
                                                                                            main()
