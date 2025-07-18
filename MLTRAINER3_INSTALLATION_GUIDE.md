# mlTrainer3 Complete Installation Guide

## 🚀 Quick Start - Deploy to Modal (Recommended)

### Prerequisites
- Python installed on any computer
- Modal account (free at https://modal.com)

### Step 1: Install Modal
```bash
pip install modal
modal token new
```

### Step 2: Deploy mlTrainer3 from GitHub
```bash
# Option A: Download and run the deployment script
curl -O https://raw.githubusercontent.com/hgw734/mlTrainer3/main/deploy_to_modal.py
python deploy_to_modal.py

# Option B: Deploy directly (if you have the repo)
git clone https://github.com/hgw734/mlTrainer3.git
cd mlTrainer3
modal deploy modal_github_deploy.py
```

### Step 3: Set Up API Keys in Modal

1. Go to https://modal.com/secrets
2. Create a new secret named `mltrainer3-secrets`
3. Add your API keys:
```json
{
  "POLYGON_API_KEY": "your-polygon-api-key",
  "FRED_API_KEY": "your-fred-api-key",
  "ANTHROPIC_API_KEY": "your-anthropic-api-key"
}
```

### Step 4: Access mlTrainer3

Your mlTrainer3 system will be available at:
```
https://YOUR-MODAL-USERNAME--mltrainer3.modal.run
```

## 📱 Mobile Access (iPhone/iPad)

1. Open Safari on your iPhone
2. Go to your mlTrainer3 URL
3. Tap the Share button
4. Select "Add to Home Screen"
5. Name it "mlTrainer3"

Now you have app-like access to mlTrainer3!

## 💻 Local Installation (Alternative)

### Prerequisites
- Python 3.8+
- Git

### Step 1: Clone mlTrainer3
```bash
git clone https://github.com/hgw734/mlTrainer3.git
cd mlTrainer3
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements_unified.txt
```

### Step 4: Set Up Environment Variables
Create a `.env` file:
```bash
POLYGON_API_KEY=your-polygon-api-key
FRED_API_KEY=your-fred-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

### Step 5: Launch mlTrainer3
```bash
python launch_mltrainer.py
```

Access at: http://localhost:8501

## 🔧 Configuration

### API Keys Required

1. **Polygon.io** (Market Data)
   - Sign up at https://polygon.io
   - Free tier available
   - Provides stock prices, volumes, etc.

2. **FRED** (Economic Data)
   - Sign up at https://fred.stlouisfed.org/docs/api/api_key.html
   - Free API key
   - Provides economic indicators

3. **Anthropic** (Claude AI)
   - Sign up at https://console.anthropic.com
   - Paid API (usage-based)
   - Powers the mlTrainer chat

### Database Setup (Optional)

If you want to use a real database instead of file storage:

```sql
-- Run the schema
psql -U your_user -d your_database -f database/recommendation_schema.sql
```

## 📊 Using mlTrainer3

### Main Features

1. **Chat with mlTrainer (Claude)**
   - Ask about trading strategies
   - Request market analysis
   - Get recommendations

2. **Trading Recommendations**
   - Automatic scanning every 15 minutes
   - 7-12 day momentum trades
   - 50-70 day position trades

3. **Virtual Portfolio**
   - Paper trading with $100k
   - Automatic position tracking
   - Performance metrics (Sharpe, Sortino)

4. **Background Trials**
   - Walk-forward analysis
   - Model optimization
   - Continuous learning

### First Time Setup

1. **Scan for Opportunities**
   - Click "🔍 Scan S&P 500"
   - Wait for recommendations
   - Review top picks

2. **Start Paper Trading**
   - Recommendations auto-trade top 5
   - Monitor virtual portfolio
   - Track performance

3. **Set Trading Goals**
   - Use chat to set objectives
   - mlTrainer will optimize for your goals

## 🔄 Updating mlTrainer3

### For Modal Deployment
```bash
# Just redeploy - Modal pulls latest from GitHub
modal deploy modal_github_deploy.py --force
```

### For Local Installation
```bash
git pull origin main
pip install -r requirements_unified.txt --upgrade
```

## 🛠️ Troubleshooting

### Modal Issues

**Problem**: Deployment fails
```bash
# Check Modal authentication
modal token set

# Check logs
modal logs mltrainer3
```

**Problem**: App not loading
- Check API keys in Modal secrets
- Verify secret name is `mltrainer3-secrets`

### Local Issues

**Problem**: Import errors
```bash
# Ensure virtual environment is activated
which python  # Should show venv path

# Reinstall requirements
pip install -r requirements_unified.txt --force-reinstall
```

**Problem**: API connection errors
- Verify `.env` file exists
- Check API key validity
- Ensure internet connection

## 📈 Performance Optimization

### Modal Deployment
- Keep warm: 1 instance always ready
- Auto-scales with traffic
- Persistent storage for data

### Scheduled Jobs
- Recommendation scan: Every 15 minutes
- Portfolio update: Every 5 minutes
- Data is preserved between runs

## 🔐 Security Best Practices

1. **Never commit API keys**
   - Use environment variables
   - Use Modal secrets

2. **Secure your Modal account**
   - Enable 2FA
   - Use strong passwords

3. **Monitor usage**
   - Check API usage regularly
   - Set up billing alerts

## 📞 Getting Help

1. **Documentation**: Check this guide first
2. **Issues**: https://github.com/hgw734/mlTrainer3/issues
3. **Modal Support**: https://modal.com/docs

## 🎯 Next Steps

1. **Explore the Chat**: Ask mlTrainer about momentum strategies
2. **Review Recommendations**: Check the quality of signals
3. **Monitor Performance**: Watch the virtual portfolio
4. **Optimize**: Use mlTrainer to improve strategies

Welcome to mlTrainer3! 🚀