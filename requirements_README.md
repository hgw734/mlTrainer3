# mlTrainer Requirements Guide

## 📦 Which Requirements File to Use?

mlTrainer provides three different requirements files for different use cases:

### 1. **requirements_minimal.txt** (Quick Start) 🚀
```bash
pip install -r requirements_minimal.txt
```
- **Size**: ~200 MB
- **Install time**: 2-5 minutes
- **Use when**: You want to start trading quickly
- **Includes**: Essential packages only (pandas, numpy, yfinance, AI APIs, streamlit)
- **Models available**: Basic trading strategies

### 2. **requirements.txt** (Standard) ⭐
```bash
pip install -r requirements.txt
```
- **Size**: ~1-2 GB
- **Install time**: 10-20 minutes
- **Use when**: You want most features without heavy dependencies
- **Includes**: Core ML, financial analysis, time series, web interface
- **Models available**: ~50-60 models

### 3. **requirements_comprehensive.txt** (Full Suite) 💪
```bash
pip install -r requirements_comprehensive.txt
```
- **Size**: ~5-10 GB
- **Install time**: 30-60 minutes
- **Use when**: You need all 125+ models and advanced features
- **Includes**: Everything - deep learning, NLP, signal processing, etc.
- **Models available**: All 125+ models

## 🎯 Decision Tree

```
Are you just testing mlTrainer?
├─ Yes → Use requirements_minimal.txt
└─ No
   │
   Need deep learning or advanced models?
   ├─ No → Use requirements.txt (recommended)
   └─ Yes
      │
      Have 10GB+ disk space and good internet?
      ├─ Yes → Use requirements_comprehensive.txt
      └─ No → Use requirements.txt + install specific packages
```

## 📋 Package Categories

### Always Included (All Files)
- pandas, numpy - Data manipulation
- yfinance, fredapi - Market data
- anthropic/openai - AI integration
- streamlit - Web interface
- plotly - Visualization

### Standard Additions (requirements.txt)
- scikit-learn, xgboost, lightgbm - ML models
- ta-lib, pandas-ta - Technical analysis
- statsmodels, prophet - Time series
- backtrader - Backtesting
- fastapi - API server

### Comprehensive Additions (requirements_comprehensive.txt)
- tensorflow, torch - Deep learning
- transformers - Large language models
- gymnasium, stable-baselines3 - Reinforcement learning
- quantlib, zipline - Advanced finance
- apache-airflow - Data pipelines
- 50+ specialized libraries

## ⚠️ Installation Notes

### System Dependencies
Some packages require system libraries:
```bash
# For ta-lib
# Ubuntu/Debian:
sudo apt-get install libta-lib0-dev

# macOS:
brew install ta-lib

# Windows: Download from ta-lib.org
```

### Common Issues & Solutions

1. **ta-lib installation fails**
   - Install system library first (see above)
   - Or remove from requirements and use pandas-ta only

2. **Out of memory during install**
   ```bash
   # Install in chunks
   pip install pandas numpy scipy
   pip install scikit-learn xgboost lightgbm
   pip install -r requirements.txt --no-deps
   ```

3. **Conflicts with existing packages**
   ```bash
   # Use virtual environment
   python -m venv mltrainer_env
   source mltrainer_env/bin/activate  # Linux/Mac
   # or
   mltrainer_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

4. **Slow installation**
   ```bash
   # Use faster mirror
   pip install -r requirements.txt -i https://pypi.douban.com/simple/
   ```

## 🚀 Quick Start Commands

### Minimal Setup (5 minutes)
```bash
git clone https://github.com/hgw734/mlTrainer.git
cd mlTrainer
python -m venv venv
source venv/bin/activate
pip install -r requirements_minimal.txt
cp .env.example .env
# Edit .env with your API keys
python mlTrainer_main.py
```

### Standard Setup (Recommended)
```bash
git clone https://github.com/hgw734/mlTrainer.git
cd mlTrainer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python verify_compliance_system.py
python mlTrainer_main.py
```

### Development Setup (Full)
```bash
git clone https://github.com/hgw734/mlTrainer.git
cd mlTrainer
python -m venv venv
source venv/bin/activate
pip install -r requirements_comprehensive.txt
pip install -r requirements-dev.txt  # If exists
pre-commit install
cp .env.example .env
# Edit .env with your API keys
pytest
python mlTrainer_main.py
```

## 📊 Disk Space Requirements

| File | Download | Installed | Virtual Env Total |
|------|----------|-----------|-------------------|
| minimal | ~100 MB | ~200 MB | ~300 MB |
| standard | ~500 MB | ~1.5 GB | ~2 GB |
| comprehensive | ~2 GB | ~8 GB | ~10 GB |

## 🔄 Upgrading

To upgrade from minimal to standard:
```bash
pip install -r requirements.txt
```

To upgrade to comprehensive:
```bash
pip install -r requirements_comprehensive.txt
```

To upgrade specific packages:
```bash
pip install --upgrade pandas numpy yfinance
```

## 🧹 Cleanup

Remove unused packages:
```bash
pip freeze > current.txt
comm -23 <(sort current.txt) <(sort requirements.txt) | xargs pip uninstall -y
```

## 💡 Pro Tips

1. **Start with minimal**, upgrade as needed
2. **Use virtual environments** to avoid conflicts
3. **Install heavy packages separately** (tensorflow, torch)
4. **Cache downloads** for faster reinstalls:
   ```bash
   pip download -r requirements.txt -d ./pip_cache
   pip install --find-links ./pip_cache -r requirements.txt
   ```

5. **Production deployment**: Use Docker for consistency
   ```bash
   docker build -t mltrainer .
   docker run -p 8501:8501 mltrainer
   ```