#!/usr/bin/env python3
"""Comprehensive environment test for the trading system."""

import sys
import importlib
import subprocess
from typing import Dict, List, Tuple

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and get its version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        return True, version
    except ImportError:
        return False, 'Not installed'

def test_environment() -> Dict[str, Dict[str, str]]:
    """Test the Python environment and installed packages."""
    results = {
        'core': {},
        'data_science': {},
        'visualization': {},
        'trading': {},
        'ml_dl': {},
        'utils': {}
    }
    
    # Core packages
    core_packages = [
        ('python', 'sys', f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
        ('numpy', 'numpy', None),
        ('pandas', 'pandas', None),
        ('scipy', 'scipy', None),
        ('requests', 'requests', None),
    ]
    
    # Data science packages
    ds_packages = [
        ('scikit-learn', 'sklearn', None),
        ('statsmodels', 'statsmodels', None),
        ('matplotlib', 'matplotlib', None),
        ('seaborn', 'seaborn', None),
        ('plotly', 'plotly', None),
    ]
    
    # Trading-specific packages (to check)
    trading_packages = [
        ('ta', 'ta', None),
        ('ta-lib', 'talib', None),
        ('yfinance', 'yfinance', None),
        ('ccxt', 'ccxt', None),
        ('alpaca-trade-api', 'alpaca_trade_api', None),
        ('ib_insync', 'ib_insync', None),
        ('polygon-api-client', 'polygon', None),
        ('fredapi', 'fredapi', None),
    ]
    
    # ML/DL packages (to check)
    ml_packages = [
        ('tensorflow', 'tensorflow', None),
        ('torch', 'torch', None),
        ('xgboost', 'xgboost', None),
        ('lightgbm', 'lightgbm', None),
        ('arch', 'arch', None),
        ('prophet', 'prophet', None),
        ('pmdarima', 'pmdarima', None),
    ]
    
    # Utility packages
    util_packages = [
        ('websocket-client', 'websocket', None),
        ('asyncio', 'asyncio', None),
        ('aiohttp', 'aiohttp', None),
        ('sqlalchemy', 'sqlalchemy', None),
        ('redis', 'redis', None),
        ('anthropic', 'anthropic', None),
    ]
    
    # Check core packages
    print("=== ENVIRONMENT TEST RESULTS ===\n")
    print("Python Version:", sys.version)
    print("\n--- Core Packages ---")
    for pkg, imp, ver in core_packages:
        if ver:
            results['core'][pkg] = ver
            print(f"✓ {pkg}: {ver}")
        else:
            installed, version = check_package(pkg, imp)
            results['core'][pkg] = version
            status = "✓" if installed else "✗"
            print(f"{status} {pkg}: {version}")
    
    # Check data science packages
    print("\n--- Data Science Packages ---")
    for pkg, imp, _ in ds_packages:
        installed, version = check_package(pkg, imp)
        results['data_science'][pkg] = version
        status = "✓" if installed else "✗"
        print(f"{status} {pkg}: {version}")
    
    # Check trading packages
    print("\n--- Trading Packages ---")
    for pkg, imp, _ in trading_packages:
        installed, version = check_package(pkg, imp)
        results['trading'][pkg] = version
        status = "✓" if installed else "✗"
        print(f"{status} {pkg}: {version}")
    
    # Check ML/DL packages
    print("\n--- ML/DL Packages ---")
    for pkg, imp, _ in ml_packages:
        installed, version = check_package(pkg, imp)
        results['ml_dl'][pkg] = version
        status = "✓" if installed else "✗"
        print(f"{status} {pkg}: {version}")
    
    # Check utility packages
    print("\n--- Utility Packages ---")
    for pkg, imp, _ in util_packages:
        installed, version = check_package(pkg, imp)
        results['utils'][pkg] = version
        status = "✓" if installed else "✗"
        print(f"{status} {pkg}: {version}")
    
    return results

def suggest_installations(results: Dict[str, Dict[str, str]]) -> List[str]:
    """Suggest packages to install based on test results."""
    suggestions = []
    
    # Essential trading packages
    essential_trading = ['ta', 'yfinance', 'ccxt', 'fredapi']
    missing_trading = [pkg for pkg in essential_trading 
                      if results['trading'].get(pkg) == 'Not installed']
    
    # Essential ML packages
    essential_ml = ['xgboost', 'lightgbm', 'arch', 'prophet', 'pmdarima']
    missing_ml = [pkg for pkg in essential_ml 
                 if results['ml_dl'].get(pkg) == 'Not installed']
    
    # Essential utilities
    essential_utils = ['websocket-client', 'aiohttp', 'sqlalchemy']
    missing_utils = [pkg for pkg in essential_utils 
                    if results['utils'].get(pkg) == 'Not installed']
    
    if missing_trading:
        suggestions.append(f"pip install {' '.join(missing_trading)}")
    if missing_ml:
        suggestions.append(f"pip install {' '.join(missing_ml)}")
    if missing_utils:
        suggestions.append(f"pip install {' '.join(missing_utils)}")
    
    return suggestions

def main():
    """Run the environment test."""
    results = test_environment()
    
    print("\n=== INSTALLATION SUGGESTIONS ===")
    suggestions = suggest_installations(results)
    if suggestions:
        print("\nTo install missing packages, run:")
        for suggestion in suggestions:
            print(f"  {suggestion}")
    else:
        print("\nAll essential packages are installed!")
    
    # Count installed vs missing
    total_checked = sum(len(cat) for cat in results.values())
    total_installed = sum(1 for cat in results.values() 
                         for status in cat.values() 
                         if status != 'Not installed')
    
    print(f"\n=== SUMMARY ===")
    print(f"Total packages checked: {total_checked}")
    print(f"Installed: {total_installed}")
    print(f"Missing: {total_checked - total_installed}")
    
    # Create requirements file for missing packages
    all_missing = []
    for category, packages in results.items():
        for pkg, status in packages.items():
            if status == 'Not installed':
                all_missing.append(pkg)
    
    if all_missing:
        with open('requirements_missing.txt', 'w') as f:
            f.write('\n'.join(all_missing))
        print(f"\nMissing packages saved to: requirements_missing.txt")

if __name__ == "__main__":
    main()