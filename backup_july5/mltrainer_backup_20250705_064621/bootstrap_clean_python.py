#!/usr/bin/env python3
"""
Bootstrap Clean Python Environment
=================================
Downloads and sets up a clean Python environment free from Nix contamination.
"""

import os
import sys
import subprocess
import urllib.request
import tarfile
import shutil

def download_portable_python():
    """Download portable Python distribution"""
    url = "https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"
    target_dir = "/tmp/clean_python_install"
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Download
    archive_path = os.path.join(target_dir, "python.tar.gz")
    print(f"Downloading clean Python to {archive_path}...")
    urllib.request.urlretrieve(url, archive_path)
    
    # Extract
    print("Extracting clean Python...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(target_dir)
    
    # Find Python executable
    python_path = None
    for root, dirs, files in os.walk(target_dir):
        if 'python' in files or 'python3' in files:
            if 'python3' in files:
                python_path = os.path.join(root, 'python3')
                break
            elif 'python' in files:
                python_path = os.path.join(root, 'python')
                break
    
    if python_path and os.path.exists(python_path):
        print(f"Clean Python found at: {python_path}")
        
        # Test the clean Python
        try:
            result = subprocess.run([python_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            print(f"Clean Python version: {result.stdout.strip()}")
            
            # Try installing pip
            try:
                subprocess.run([python_path, '-m', 'ensurepip', '--upgrade'], 
                             capture_output=True, timeout=30)
                print("Successfully installed pip in clean Python")
                
                # Try installing basic ML packages
                packages = ['numpy', 'pandas', 'scikit-learn']
                for package in packages:
                    print(f"Installing {package}...")
                    result = subprocess.run([python_path, '-m', 'pip', 'install', package], 
                                          capture_output=True, text=True, timeout=120)
                    if result.returncode == 0:
                        print(f"✓ {package} installed successfully")
                    else:
                        print(f"✗ {package} failed: {result.stderr}")
                        
                return python_path
                
            except subprocess.TimeoutExpired:
                print("Timeout installing packages")
                return None
                
        except subprocess.TimeoutExpired:
            print("Timeout testing clean Python")
            return None
    
    print("No clean Python executable found")
    return None

def create_clean_wrapper():
    """Create wrapper script for clean Python"""
    clean_python = download_portable_python()
    
    if clean_python:
        wrapper_path = "/tmp/clean_python"
        with open(wrapper_path, 'w') as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"export PATH={os.path.dirname(clean_python)}:$PATH\n")
            f.write(f"exec {clean_python} \"$@\"\n")
        
        os.chmod(wrapper_path, 0o755)
        print(f"Clean Python wrapper created at: {wrapper_path}")
        return wrapper_path
    
    return None

if __name__ == "__main__":
    print("🧹 BOOTSTRAPPING CLEAN PYTHON ENVIRONMENT")
    print("=" * 50)
    
    # Show current contaminated environment
    print(f"Current contaminated Python: {sys.executable}")
    
    # Attempt to create clean environment
    clean_wrapper = create_clean_wrapper()
    
    if clean_wrapper:
        print("✅ SUCCESS: Clean Python environment created!")
        print(f"Use: {clean_wrapper}")
    else:
        print("❌ FAILED: Could not create clean Python environment")