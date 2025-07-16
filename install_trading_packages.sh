#!/bin/bash

# Trading Package Installation Script
# This script installs additional packages needed for the trading system

echo "=== Trading Package Installation Script ==="
echo "This will install additional packages for the mlTrainer trading system"
echo ""

# Ensure conda environment is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Activating conda environment..."
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda activate base
fi

echo "Python: $(which python)"
echo "Pip: $(which pip)"
echo ""

# Function to install packages with error handling
install_packages() {
    local package_group=$1
    shift
    local packages=("$@")
    
    echo "Installing $package_group packages..."
    for pkg in "${packages[@]}"; do
        echo "  - Installing $pkg..."
        pip install "$pkg" || echo "  ! Failed to install $pkg"
    done
    echo ""
}

# 1. Technical Analysis
echo "=== Phase 1: Technical Analysis ==="
install_packages "Technical Analysis" \
    "ta" \
    "yfinance" \
    "pandas-ta"

# 2. Market Data APIs
echo "=== Phase 2: Market Data APIs ==="
install_packages "Market Data APIs" \
    "ccxt" \
    "alpaca-trade-api" \
    "polygon-api-client" \
    "fredapi"

# 3. Time Series & ML
echo "=== Phase 3: Time Series & ML ==="
install_packages "Time Series & ML" \
    "pmdarima" \
    "arch" \
    "prophet" \
    "xgboost" \
    "lightgbm"

# 4. Async & Real-time
echo "=== Phase 4: Async & Real-time ==="
install_packages "Async & Real-time" \
    "websocket-client" \
    "aiohttp" \
    "asyncio-mqtt"

# 5. Database & Storage
echo "=== Phase 5: Database & Storage ==="
install_packages "Database & Storage" \
    "sqlalchemy" \
    "redis" \
    "pymongo" \
    "influxdb-client"

# 6. Utilities
echo "=== Phase 6: Utilities ==="
install_packages "Utilities" \
    "python-dotenv" \
    "pydantic" \
    "loguru" \
    "backtrader" \
    "vectorbt"

# Optional: Deep Learning (commented out due to size)
# echo "=== Phase 7: Deep Learning (Optional) ==="
# install_packages "Deep Learning" \
#     "tensorflow" \
#     "torch" \
#     "transformers"

echo "=== Installation Complete ==="
echo ""
echo "Testing imports..."
python -c "
import sys
print(f'Python: {sys.version}')
print('')

# Test core imports
try:
    import ta
    print('✓ Technical Analysis (ta) imported successfully')
except ImportError:
    print('✗ Technical Analysis (ta) import failed')

try:
    import yfinance
    print('✓ yfinance imported successfully')
except ImportError:
    print('✗ yfinance import failed')

try:
    import ccxt
    print('✓ ccxt imported successfully')
except ImportError:
    print('✗ ccxt import failed')

try:
    import xgboost
    print('✓ xgboost imported successfully')
except ImportError:
    print('✗ xgboost import failed')

try:
    import websocket
    print('✓ websocket-client imported successfully')
except ImportError:
    print('✗ websocket-client import failed')

try:
    import sqlalchemy
    print('✓ sqlalchemy imported successfully')
except ImportError:
    print('✗ sqlalchemy import failed')
"

echo ""
echo "To test all packages, run: python test_environment_complete.py"
echo "Done!"