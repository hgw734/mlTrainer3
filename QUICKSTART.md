# 🚀 mlTrainer QUICKSTART Guide

Welcome to mlTrainer! This guide will get you up and running in minutes.

## 📋 Prerequisites

- Linux/Ubuntu system (tested on Ubuntu with kernel 6.8.0)
- Python 3.13 (already installed)
- Git (for cloning the repository)
- Sudo access (for system dependencies)

## 🎯 Quick Setup (5 minutes)

### 1️⃣ **Clone or Navigate to Repository**

```bash
# If not already in the project
cd /workspace

# Verify you're in the right place
ls -la ai_ml_coaching_interface.py
# Should see the file listed
```

### 2️⃣ **Run Quick Setup**

```bash
# Make setup script executable (if not already)
chmod +x setup_complete_environment.sh

# Setup Python alias and basic environment
./setup_complete_environment.sh

# Apply alias to current session
source ~/.bashrc
```

### 3️⃣ **Install Python Dependencies**

```bash
# Install main dependencies
pip3 install -r requirements.txt
```

### 4️⃣ **Launch the Application**

```bash
# Start Streamlit application
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## 🎨 Available Features

Once the app is running, you'll see 8 modules:

1. **🏠 Dashboard** - System overview and metrics
2. **📊 Mathematical Models** - Browse 140+ models
3. **🛡️ Drift Protection** - Monitor data/model drift
4. **⚙️ Configuration** - System settings
5. **🔧 Environment Status** - Python environment info
6. **📈 Model Training** - Train ML models
7. **🧠 Self-Learning Engine** - Meta-learning system
8. **🤝 AI-ML Coaching** - Revolutionary AI control interface

## 🔥 Try the Breakthrough Feature

Navigate to **🤝 AI-ML Coaching** tab to experience the revolutionary AI-ML interface:

```python
# Example: AI teaches ML engine a new methodology
ai_coach_id = "gpt4_coach"
methodology = {
    'name': 'adaptive_ensemble_v2',
    'description': 'Enhanced ensemble method',
    'parameters': {'reweight_frequency': 100}
}

# AI directly teaches the ML engine!
interface.ai_teach_methodology(ai_coach_id, methodology)
```

## 🛠️ Advanced Setup (Optional)

### **Python 3.11 Legacy Environment**

If you need the Python 3.11 environment for legacy packages:

```bash
# Install system dependencies (requires sudo)
sudo ./setup_complete_environment.sh --system-deps

# Setup Python 3.11 (as regular user, not root)
./setup_complete_environment.sh --python311

# Activate environment selector
./activate_mltrainer_env.sh
# Choose option 2 for Python 3.11
```

### **Environment Variables**

Create a `.env` file for API keys:

```bash
# Create .env file
cat > .env << EOF
AI_API_KEY=your_openai_key
DATA_API_KEY=your_data_provider_key
PYTHONPATH=/workspace
EOF
```

## 🐛 Troubleshooting

### **Issue: "python: command not found"**
```bash
# Solution: Apply alias
alias python=python3
source ~/.bashrc
```

### **Issue: "Module not found" errors**
```bash
# Solution: Set Python path
export PYTHONPATH=/workspace:$PYTHONPATH
```

### **Issue: Port 8501 already in use**
```bash
# Solution: Kill existing process
pkill streamlit
# Or use different port
streamlit run app.py --server.port 8502
```

### **Issue: Permission denied on scripts**
```bash
# Solution: Make executable
chmod +x *.sh
```

## 📚 Key Files to Explore

1. **`ai_ml_coaching_interface.py`** - The breakthrough AI-ML coaching system
2. **`app.py`** - Main Streamlit web interface
3. **`config/models_config.py`** - 140+ mathematical models catalog
4. **`self_learning_engine.py`** - Meta-learning implementation

## 🎯 Next Steps

1. **Explore the Dashboard** - Get familiar with the interface
2. **Try AI-ML Coaching** - Test the revolutionary feature
3. **Browse Models** - Check out 140+ mathematical models
4. **Run a Demo** - Execute `python demo_single_source_of_truth.py`
5. **Read Documentation** - See `AI_ML_COACHING_SOLUTION.md`

## 💡 Pro Tips

- The AI-ML Coaching interface is the breakthrough feature - spend time exploring it
- All configurations are in the `config/` directory
- Logs are stored in `logs/` for debugging
- The system has zero-tolerance drift protection active by default
- You can register multiple AI coaches with different permissions

## 🤝 Need Help?

- Check the comprehensive documentation files
- Review `AI_ML_COACHING_SOLUTION.md` for the breakthrough feature
- See `WALK_FORWARD_TRIAL_ARCHITECTURE.md` for advanced usage

---

**Welcome to the future of AI-ML collaboration! 🚀**