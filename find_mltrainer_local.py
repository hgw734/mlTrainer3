#!/usr/bin/env python3
"""
Find mlTrainer Project on Local System
This script helps locate the mlTrainer project directory on your local machine
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
import json


def find_mltrainer_directories():
    """Search for mlTrainer directories on the system"""
    found_dirs = []

    # Common project locations
    home = Path.home()
    common_paths = [
        home / "Documents",
        home / "Projects",
        home / "projects",
        home / "Code",
        home / "code",
        home / "Development",
        home / "dev",
        home / "workspace",
        home / "repos",
        home / "github",
        home / "Desktop",
        home / "Downloads",
    ]

    # Add Windows-specific paths
    if platform.system() == "Windows":
        common_paths.extend([Path("C:/") /
                             "Projects", Path("C:/") /
                             "Code", Path("C:/") /
                             "Users" /
                             os.environ.get("USERNAME", "") /
                             "source" /
                             "repos", ])

        # Add Mac-specific paths
        if platform.system() == "Darwin":
            common_paths.extend(
                [
                    home / "Developer",
                    Path("/") / "Users" / "Shared" / "Projects",
                ]
            )

            print("🔍 Searching for mlTrainer directories# Production code implemented")
            print(f"System: {platform.system()}")
            print(f"Home directory: {home}")
            print(("-" * 50))

            # Search in common locations first (faster)
            for base_path in common_paths:
                if base_path.exists():
                    # Look for mlTrainer directory
                    mltrainer_path = base_path / "mlTrainer"
                    if mltrainer_path.exists() and mltrainer_path.is_dir():
                        # Verify it's the right project by checking for key
                        # files
                        if is_mltrainer_project(mltrainer_path):
                            found_dirs.append(mltrainer_path)

                        # Also check one level deeper
                        try:
                            for subdir in base_path.iterdir():
                                if subdir.is_dir() and subdir.name == "mlTrainer":
                                    if is_mltrainer_project(subdir):
                                        found_dirs.append(subdir)
                        except PermissionError:
                            pass

            # Check git config for recent clones
            git_dirs = find_via_git()
            found_dirs.extend(git_dirs)

            # Remove duplicates
            found_dirs = list(set(found_dirs))

            return found_dirs


def is_mltrainer_project(path):
    """Verify if a directory is the mlTrainer project"""
    key_files = [
        "config/models_config.py",
        "config/immutable_compliance_gateway.py",
        "requirements.txt",
        ".git",
        "custom",
    ]

    matches = 0
    for key_file in key_files:
        if (path / key_file).exists():
            matches += 1

    return matches >= 3  # At least 3 key files/dirs should exist


def find_via_git():
    """Find mlTrainer via git global config"""
    found_dirs = []

    try:
        # Get list of all git repositories
        if platform.system() == "Windows":
            cmd = ["wsl",
                   "find",
                   str(Path.home()),
                   "-name",
                   ".git",
                   "-type",
                   "d",
                   "2>/dev/null"]
        else:
            cmd = ["find", str(Path.home()), "-name", ".git", "-type", "d"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            git_dirs = result.stdout.strip().split("\n")

            for git_dir in git_dirs:
                if git_dir:
                    repo_dir = Path(git_dir).parent
                    if repo_dir.name == "mlTrainer" and is_mltrainer_project(
                            repo_dir):
                        found_dirs.append(repo_dir)
    except Exception:
        pass

    return found_dirs


def check_cursor_recent_projects():
    """Check Cursor's recent projects"""
    cursor_configs = []

    # Cursor config locations
    if platform.system() == "Windows":
        cursor_paths = [
            Path.home() /
            "AppData" /
            "Roaming" /
            "Cursor" /
            "User" /
            "globalStorage",
            Path.home() /
            "AppData" /
            "Local" /
            "Cursor",
        ]
    elif platform.system() == "Darwin":  # macOS
        cursor_paths = [
            Path.home() / "Library" / "Application Support" / "Cursor",
            Path.home() / ".cursor",
        ]
    else:  # Linux
        cursor_paths = [
            Path.home() / ".config" / "Cursor",
            Path.home() / ".cursor",
        ]

    for cursor_path in cursor_paths:
        if cursor_path.exists():
            cursor_configs.append(cursor_path)

    return cursor_configs


def display_results(found_dirs):
    """Display the search results"""
    print(("\n" + "=" * 50))
    print("🎯 SEARCH RESULTS")
    print(("=" * 50))

    if not found_dirs:
        print("❌ No mlTrainer directories found!")
        print("\nPossible reasons:")
        print("1. The project hasn't been cloned yet")
        print("2. It's in a non-standard location")
        print("3. Permission issues preventing search")
        print("\nTry cloning it:")
        print("git clone https://github.com/hgw734/mlTrainer.git")
    else:
        print(f"✅ Found {len(found_dirs)} mlTrainer installation(s):\n")

        for i, dir_path in enumerate(found_dirs, 1):
            print(f"{i}. {dir_path}")

            # Check git status
            try:
                os.chdir(dir_path)
                result = subprocess.run(
                    ["git", "status", "--porcelain"], capture_output=True, text=True)
                if result.returncode == 0:
                    if result.stdout.strip():
                        print(f"   ⚠️  Has uncommitted changes")
                    else:
                        print(f"   ✅ Clean working directory")

                    # Get last commit
                    result = subprocess.run(
                        ["git", "log", "-1", "--oneline"], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"   📝 Last commit: {result.stdout.strip()}")
            except Exception:
                pass

        print()

        # Show Cursor config locations
        cursor_configs = check_cursor_recent_projects()
        if cursor_configs:
            print("\n📁 Cursor configuration found at:")
            for config in cursor_configs:
                print(f"   {config}")

            print(("\n" + "=" * 50))
            print("💡 TO OPEN IN CURSOR:")
            print(("=" * 50))

            if found_dirs:
                print("Option 1 - Command Line:")
                print(f"cursor {found_dirs[0]}")
                print("\nOption 2 - Cursor App:")
                print("1. Open Cursor")
                print("2. File → Open Folder")
                print(f"3. Navigate to: {found_dirs[0]}")
                print("\nOption 3 - Drag and Drop:")
                print(
                    f"Drag the folder {found_dirs[0]} onto the Cursor app icon")
            else:
                print("First, clone the repository:")
                print("git clone https://github.com/hgw734/mlTrainer.git")
                print("\nThen open in Cursor:")
                print("cursor mlTrainer")


def main():
    """Main function"""
    print("🔍 mlTrainer Local File Finder")
    print(("=" * 50))

    # Search for directories
    found_dirs = find_mltrainer_directories()

    # Display results
    display_results(found_dirs)

    # Save results to file for reference
    results = {
        "search_time": str(Path.cwd()),
        "system": platform.system(),
        "found_directories": [str(d) for d in found_dirs],
        "cursor_command": f"cursor {found_dirs[0]}" if found_dirs else None,
    }

    with open("mltrainer_locations.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {Path.cwd() / 'mltrainer_locations.json'}")


if __name__ == "__main__":
    main()
