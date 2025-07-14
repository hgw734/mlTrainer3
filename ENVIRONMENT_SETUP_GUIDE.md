# 🚀 mlTrainer Environment Setup Guide

This guide will help you set up your environment for mlTrainer, including API keys, Modal deployment, and local development.

## 📋 Prerequisites

1. **Python 3.8+** installed on your system
2. **Git** for version control
3. **API keys** for various services (see below)
4. **Modal account** for cloud deployment

## 🔑 Required API Keys

### 1. Financial Data APIs (Required)

#### Polygon.io (Free tier available)
- **Purpose**: Real-time market data
- **Get key**: https://polygon.io/dashboard/api-keys
- **Free tier**: 5 requests/minute, 5 years of data
- **Environment variable**: `POLYGON_API_KEY`

#### FRED (Free)
- **Purpose**: Economic data from Federal Reserve
- **Get key**: https://fred.stlouisfed.org/docs/api/api_key.html
- **Free tier**: 120 requests/minute
- **Environment variable**: `FRED_API_KEY`

### 2. AI Model APIs (Required)

#### OpenAI (Paid)
- **Purpose**: GPT models for analysis
- **Get key**: https://platform.openai.com/api-keys
- **Cost**: ~$0.01-0.03 per 1K tokens
- **Environment variable**: `OPENAI_API_KEY`

#### Anthropic (Paid)
- **Purpose**: Claude models (currently configured as default)
- **Get key**: https://console.anthropic.com/settings/keys
- **Cost**: ~$0.003 per 1K tokens
- **Environment variable**: `ANTHROPIC_API_KEY`

### 3. Optional APIs

#### Alpaca Trading (Optional)
- **Purpose**: Execute trades
- **Get key**: https://alpaca.markets/
- **Environment variables**: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`

#### Telegram (Optional)
- **Purpose**: Notifications
- **Get key**: Create bot via @BotFather
- **Environment variables**: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

## 🛠️ Environment Setup

### Step 1: Create Environment File

```bash
# Copy the template
cp env_setup_template.txt .env

# Edit with your actual keys
nano .env
# or
code .env
```

### Step 2: Fill in Your API Keys

Edit your `.env` file and replace the placeholder values:

```bash
# Required - Financial Data
POLYGON_API_KEY=your_actual_polygon_key_here
FRED_API_KEY=your_actual_fred_key_here

# Required - AI Models
OPENAI_API_KEY=your_actual_openai_key_here
ANTHROPIC_API_KEY=your_actual_anthropic_key_here

# Optional - Trading
ALPACA_API_KEY=your_actual_alpaca_key_here
ALPACA_SECRET_KEY=your_actual_alpaca_secret_here

# Compliance settings (keep as-is)
COMPLIANCE_MODE=strict
ENABLE_KILL_SWITCH=true
MAX_VIOLATIONS=5
```

### Step 3: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows

# Install packages
pip install -r requirements.txt
```

### Step 4: Verify Setup

```bash
# Test environment
python verify_compliance_system.py

# Test API connections
python test_data_connections.py

# Test AI models
python test_model_integration.py
```

## ☁️ Modal Deployment Setup

### Step 1: Install Modal CLI

```bash
pip install modal
```

### Step 2: Authenticate with Modal

```bash
modal token new
```

Follow the prompts to authenticate with your Modal account.

### Step 3: Set Up Modal Secrets

Create secrets in Modal dashboard or via CLI:

```bash
# Create secrets for deployment
modal secret create mltrainer-secrets \
  POLYGON_API_KEY="your_key" \
  FRED_API_KEY="your_key" \
  OPENAI_API_KEY="your_key" \
  ANTHROPIC_API_KEY="your_key" \
  COMPLIANCE_MODE="strict" \
  ENABLE_KILL_SWITCH="true"
```

### Step 4: Deploy to Modal

```bash
# Deploy the optimized version
modal deploy modal_app_optimized.py

# Or deploy the basic version
modal deploy modal_app.py
```

## 🔧 Local Development

### Quick Start

```bash
# Start the main interface
python mlTrainer_main.py

# Or use the chat interface
python mltrainer_chat.py

# Or run autonomous trading
python launch_mltrainer.py
```

### Development Mode

```bash
# Set development environment
export MLTRAINER_ENV=development

# Run with debug logging
export LOG_LEVEL=DEBUG

# Start development server
python app.py
```

## 🧪 Testing

### Run All Tests

```bash
# Run compliance tests
pytest tests/test_compliance_enforcement.py

# Run integration tests
python test_full_integration.py

# Run production audit
python scripts/production_audit_final.py
```

### Test Individual Components

```bash
# Test API connections
python test_data_connections.py

# Test model integration
python test_model_integration.py

# Test compliance system
python test_compliance.py
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   # Check if keys are loaded
   python -c "import os; print('POLYGON_API_KEY' in os.environ)"
   ```

2. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

3. **Modal Authentication Issues**
   ```bash
   # Re-authenticate
   modal token new
   ```

4. **Compliance Violations**
   ```bash
   # Check compliance status
   python compliance_status_summary.py
   ```

### Environment Verification

```bash
# Verify all required secrets
python -c "
from config.secrets_manager import get_secrets_manager
sm = get_secrets_manager()
print('Missing:', sm.get_missing_required())
print('All good!' if sm.check_environment() else 'Issues found')
"
```

## 📊 Cost Estimation

### Monthly Costs (Typical Usage)

| Service | Cost/Month | Usage |
|---------|------------|-------|
| Polygon.io | $0 (Free tier) | 5 req/min |
| FRED | $0 (Free) | 120 req/min |
| OpenAI | $30-300 | 1M-10M tokens |
| Anthropic | $40-400 | 1M-10M tokens |
| Modal | $50-200 | Cloud hosting |
| **Total** | **$120-900** | **Typical range** |

### Budget Optimization

- Use Claude 3.5 Sonnet (best value for $50/month)
- Start with free tiers for financial data
- Use Modal's free tier for development
- Monitor usage with built-in tracking

## 🔒 Security Notes

1. **Never commit `.env` file** - it's in `.gitignore`
2. **Rotate API keys regularly**
3. **Use environment variables** - never hardcode secrets
4. **Enable compliance mode** - prevents violations
5. **Monitor audit logs** - track all operations

## 📞 Support

- **Documentation**: Check `docs/` folder
- **Issues**: Create GitHub issue
- **Compliance**: Check `COMPLIANCE_ENFORCEMENT_SYSTEM.md`
- **Deployment**: Check `MODAL_DEPLOYMENT_GUIDE.md`

---

**Next Steps**: Once environment is set up, proceed to Modal web service deployment! 