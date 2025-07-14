# mlTrainer - Advanced ML Trading System with Mobile Interface

[![CI/CD Pipeline](https://github.com/hgw734/mlTrainer/actions/workflows/unified-ci-cd.yml/badge.svg)](https://github.com/hgw734/mlTrainer/actions/workflows/unified-ci-cd.yml)
[![Compliance Status](https://img.shields.io/badge/Compliance-Enforced-green.svg)](./IMMUTABLE_COMPLIANCE_SYSTEM.md)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-July%2013%2C%202025-blue.svg)](#)

## 🚀 Production-Ready Financial Trading System

mlTrainer is an AI-powered financial trading system with comprehensive compliance enforcement, ensuring safe and regulated operation in production environments.

### ✅ Key Features:
- **Immutable Compliance System**: 7-layer security preventing regulatory violations
- **Real Data Integration**: Polygon and FRED APIs for authentic market data
- **AI-Powered Analysis**: GPT-4 and Claude integration with behavior controls
- **Emergency Kill Switch**: Automatic shutdown on compliance violations
- **Complete Audit Trail**: Cryptographically signed decision logging

### 🔒 Compliance First
This system includes a mathematically enforced compliance framework that makes it impossible to:
- Use synthetic or fake data
- Expose API credentials
- Allow uncontrolled AI behavior
- Violate financial regulations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Redis (for caching)
- PostgreSQL (optional, for persistence)
- API Keys for:
  - Polygon.io
  - FRED (Federal Reserve Economic Data)
  - QuiverQuant (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlTrainer.git
cd mlTrainer
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Run compliance verification:
```bash
python verify_compliance_system.py
```

6. Start the system:
```bash
python app.py
```

## 🏗️ System Architecture

### Core Components

1. **Immutable Compliance Gateway** (`config/immutable_compliance_gateway.py`)
   - Runtime hard barriers
   - Data provenance tracking
   - Cryptographic verification

2. **Runtime Enforcer** (`core/immutable_runtime_enforcer.py`)
   - Real-time monitoring
   - Kill switch mechanism
   - System state management

3. **AI Client Wrapper** (`mlTrainer_client_wrapper.py`)
   - Permanent prompt injection
   - Response verification
   - Streaming compliance

4. **Drift Protection** (`drift_protection.py`)
   - Statistical monitoring
   - Performance tracking
   - Anomaly detection

### Enforcement Layers

```
User Request → Client Wrapper → Runtime Enforcer → Compliance Gateway → Data APIs
     ↓              ↓                ↓                    ↓              ↓
   BLOCK          BLOCK            BLOCK               BLOCK         VERIFIED
```

## 📋 Compliance Rules

### Absolute Prohibitions
- ❌ NO synthetic/random/generated data
- ❌ NO placeholders or examples
- ❌ NO hypothetical scenarios
- ❌ NO unauthorized data sources
- ❌ NO bypassing compliance checks

### Required Behaviors
- ✅ ONLY verified API data
- ✅ ALWAYS return "NA" when uncertain
- ✅ ALWAYS track data provenance
- ✅ ALWAYS verify freshness
- ✅ ALWAYS maintain audit trail

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# API Keys (Required)
POLYGON_API_KEY=your_polygon_key_here
FRED_API_KEY=your_fred_key_here
QUIVERQUANT_API_KEY=your_quiverquant_key_here  # Optional

# Redis Configuration
REDIS_URL=redis://localhost:6379

# System Configuration
ENFORCEMENT_LEVEL=strict  # strict, normal, monitoring
MAX_DATA_AGE_SECONDS=3600
KILL_SWITCH_ENABLED=true

# Model Configuration
OPENAI_API_KEY=your_openai_key_here  # If using OpenAI
ANTHROPIC_API_KEY=your_anthropic_key_here  # If using Claude
```

### API Configuration

Modify `config/api_config.py` to add new data sources (must be approved).

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/ -v

# Compliance tests only
pytest tests/test_compliance_enforcement.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## 📊 Monitoring

### Log Files
- `runtime_enforcement.log` - Real-time enforcement events
- `compliance_audit.log` - Compliance tracking
- `drift_protection.log` - Drift detection events
- `KILL_SWITCH_LOG.txt` - Critical shutdowns

### Metrics Dashboard
Access the monitoring dashboard at `http://localhost:8000/monitoring`

## 🚨 Emergency Procedures

### Kill Switch Activation
If the system activates the kill switch:

1. Check `KILL_SWITCH_LOG.txt` for the reason
2. Review `logs/system_state.json` for context
3. Fix the underlying issue
4. Reset with authorization:
   ```python
   python scripts/reset_system_state.py --authorize "Your Name"
   ```

## 🛠️ Development

### Adding New Features

1. All data fetching must use the compliance wrapper:
```python
from core.immutable_runtime_enforcer import compliance_wrap

@compliance_wrap
def fetch_data(symbol, source="polygon"):
    # Your implementation
    return data
```

2. All AI interactions must use the client wrapper:
```python
from mlTrainer_client_wrapper import mltrainer

response = mltrainer.query("Your prompt here")
```

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Run `black` and `isort` before committing

## 📚 Documentation

- [Immutable Compliance System](IMMUTABLE_COMPLIANCE_SYSTEM.md)
- [System Comparison](COMPLIANCE_SYSTEM_COMPARISON.md)
- [Compliance Achievement Report](COMPLIANCE_ACHIEVEMENT_REPORT.md)
- [API Documentation](docs/api.md)
- [Architecture Guide](docs/architecture.md)

## 🤝 Contributing

This is a proprietary system. Contributing guidelines:

1. All code must pass compliance audit
2. No synthetic data in any form
3. Comprehensive tests required
4. Security review mandatory

## 📄 License

Proprietary - All Rights Reserved

## 🔐 Security

- Report security issues to: security@mltrainer.ai
- Do NOT create public issues for security vulnerabilities
- See [SECURITY.md](SECURITY.md) for details

## 📞 Support

- Documentation: [docs.mltrainer.ai](https://docs.mltrainer.ai)
- Issues: Use GitHub Issues
- Enterprise Support: enterprise@mltrainer.ai

---

**⚠️ WARNING**: This system has immutable compliance enforcement. Any attempt to bypass compliance will result in automatic system shutdown.