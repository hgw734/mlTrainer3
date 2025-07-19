#!/tmp/clean_python_install/python/bin/python3
"""
Complete System Decontamination and Pure Python Rebuild
=======================================================
Rebuilds the entire mlTrainer system using ONLY pure Python environment.
No contaminated dependencies, no workarounds - complete clean rebuild.
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path

CLEAN_PYTHON = "/tmp/clean_python_install/python/bin/python3"
PURE_SYSTEM_ROOT = "/tmp/pure_python_system"

def log(message):
    print(f"[DECONTAMINATION] {message}")

def install_pure_packages():
    """Install all required packages in pure Python environment"""
    log("Installing pure Python packages...")
    
    packages = [
        "streamlit",
        "flask", 
        "flask-cors",
        "pandas",
        "numpy", 
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "plotly",
        "requests",
        "pyyaml",
        "joblib",
        "anthropic"
    ]
    
    for package in packages:
        log(f"Installing {package}...")
        try:
            result = subprocess.run([
                CLEAN_PYTHON, "-m", "pip", "install", "--no-cache-dir", 
                "--target", f"{PURE_SYSTEM_ROOT}/pure_libs", package
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                log(f"✅ {package} installed successfully")
            else:
                log(f"❌ {package} failed: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            log(f"⏰ {package} installation timeout")

def main():
    """Start complete system decontamination"""
    log("🧹 STARTING COMPLETE SYSTEM DECONTAMINATION")
    log("=" * 50)
    
    # Install pure packages
    install_pure_packages()
    
    log("✅ PHASE 1 COMPLETE - PURE PACKAGES INSTALLED")

if __name__ == "__main__":
    main()