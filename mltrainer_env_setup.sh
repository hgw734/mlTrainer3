#!/bin/bash

# ============================
# mlTrainer Environment Setup
# Ubuntu 25.04 (Python 3.13 + 3.11)
# ============================

echo "📦 Setting up mlTrainer..."

# ----------------------------
# STEP 1: Create main env (Python 3.13)
# ----------------------------
echo "🧠 Creating Python 3.13 environment..."
python3.13 -m venv venv313
source venv313/bin/activate
pip install --upgrade pip setuptools wheel

cat > requirements_py313.txt <<REQEOF
# ================================
# mlTrainer - Python 3.13 Safe Set
# ================================

# Core Framework
streamlit
flask
flask-cors
pandas
numpy>=1.26.0,<2.0
requests
python-dotenv

# ML / DL (3.13 ready)
scikit-learn
xgboost
lightgbm
catboost
optuna
hyperopt
transformers
sentence-transformers
openai
anthropic

# Time Series & Stats
statsmodels
scipy
pmdarima

# Financial Data & Tools
yfinance
fredapi
pandas-ta
pypfopt
arch
polygon
quiverquant
ccxt

# Visualization
plotly
dash
matplotlib
seaborn
streamlit-option-menu
streamlit-aggrid
streamlit-plotly-events

# Infra & Async
websockets
aiohttp
redis
celery
flower

# Utilities
python-dateutil
pytz
joblib
featuretools
tqdm
pydantic
uvicorn
psutil
Pillow
pyyaml
websocket-client

# Dev Tools
autoflake
autopep8
flake8
mypy
werkzeug

# Web Framework
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
jinja2==3.1.2
aiofiles==23.2.1
REQEOF

pip install -r requirements_py313.txt
deactivate

echo "✅ Python 3.13 environment setup complete."
echo "🚀 Setup complete. Use venv313 for main environment."
