# mlTrainer Unified System

A comprehensive AI-powered trading and machine learning system with 140+ models, real-time data integration, compliance framework, and autonomous execution capabilities.

## 🚀 Features

### Core Capabilities
- **140+ Machine Learning Models**: Linear, tree-based, neural networks, SVM, clustering, and specialized financial models
- **Real-Time Data Integration**: Polygon.io for market data, FRED for economic indicators
- **Compliance Framework**: Built-in anti-drift protection and model governance
- **Autonomous Execution**: Self-directed goal achievement with mlTrainer ↔ ML Agent communication
- **Background Processing**: Parallel model training and execution with real async support
- **Production-Ready**: Docker, Kubernetes, CI/CD, monitoring, and authentication

### Model Categories
- **Linear Models**: Ridge, Lasso, ElasticNet, Bayesian Ridge, etc.
- **Tree-Based**: Random Forest, XGBoost, LightGBM, CatBoost
- **Neural Networks**: MLP, CNN, RNN, LSTM, Transformer architectures
- **Financial Models**: Black-Scholes, portfolio optimization, risk management, Monte Carlo
- **Time Series**: ARIMA, GARCH, state-space models, regime detection
- **Specialized**: Reinforcement learning, ensemble methods, custom strategies

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  FastAPI Backend │────▶│   PostgreSQL    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                          │
         ▼                       ▼                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Claude AI API  │     │  Async Executor  │     │   Redis Cache   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                          │
         ▼                       ▼                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ mlAgent Bridge  │     │  Model Managers  │     │   Prometheus    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 📋 Prerequisites

- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- API Keys:
  - Anthropic Claude API
  - Polygon.io API
  - FRED API
- PostgreSQL (optional, SQLite for development)
- Redis (optional, for caching)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/mltrainer-unified.git
cd mltrainer-unified

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_unified.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_anthropic_key
POLYGON_API_KEY=your_polygon_key
FRED_API_KEY=your_fred_key
DATABASE_URL=sqlite:///mltrainer.db
REDIS_URL=redis://localhost:6379
JWT_SECRET_KEY=your_secret_key_change_in_production
```

### 3. Run Development Server

```bash
# Start FastAPI backend
uvicorn backend.unified_api:app --reload --port 8000

# In another terminal, start Streamlit UI
streamlit run mltrainer_unified_chat.py
```

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Build and start all services
docker-compose -f docker-compose.unified.yml up -d

# View logs
docker-compose -f docker-compose.unified.yml logs -f

# Stop services
docker-compose -f docker-compose.unified.yml down
```

### Production Build

```bash
# Build production image
docker build -f Dockerfile.unified -t mltrainer/unified:latest .

# Run with environment variables
docker run -d \
  -p 8501:8501 \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e POLYGON_API_KEY=$POLYGON_API_KEY \
  -e FRED_API_KEY=$FRED_API_KEY \
  mltrainer/unified:latest
```

## ☸️ Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace mltrainer

# Apply configurations
kubectl apply -f k8s/mltrainer-deployment.yaml

# Check status
kubectl get pods -n mltrainer
kubectl get services -n mltrainer
```

## 🔧 Configuration

### Model Configuration
Models are configured in `config/models_config.py`. Each model includes:
- Required parameters
- Data requirements
- Compliance rules
- Performance thresholds

### Compliance Rules
Edit `backend/compliance_engine.py` to modify:
- Approved data sources
- Model complexity limits
- Anti-drift thresholds
- Audit requirements

## 📊 API Documentation

### Authentication

```bash
# Register new user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "email": "user@example.com", "password": "password"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'
```

### Model Training

```bash
# Train a model (requires authentication)
curl -X POST http://localhost:8000/api/models/train \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "random_forest",
    "symbol": "AAPL",
    "parameters": {"n_estimators": 100}
  }'
```

### Autonomous Sessions

```bash
# Start autonomous session
curl -X POST http://localhost:8000/api/autonomous/start \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Optimize portfolio for maximum Sharpe ratio",
    "context": {"symbols": ["AAPL", "GOOGL", "MSFT"]}
  }'
```

## 📈 Monitoring

### Prometheus Metrics
- Access metrics: http://localhost:8000/metrics
- Grafana dashboards: http://localhost:3000 (admin/admin)

### Key Metrics
- `mltrainer_api_requests_total`: API request counts
- `mltrainer_model_training_duration_seconds`: Model training times
- `mltrainer_background_queue_size`: Background job queue size
- `mltrainer_compliance_violations_total`: Compliance violations

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/unit/test_models.py::test_random_forest
```

## 🔒 Security

### Authentication
- JWT-based authentication with refresh tokens
- User roles and permissions
- Session management

### API Security
- Rate limiting
- CORS configuration
- Input validation
- SQL injection protection

### Data Security
- Encrypted connections
- Secure credential storage
- Audit logging

## 🚀 Production Checklist

- [ ] Set strong JWT secret key
- [ ] Configure proper CORS origins
- [ ] Enable HTTPS/TLS
- [ ] Set up database backups
- [ ] Configure monitoring alerts
- [ ] Review rate limits
- [ ] Update API keys
- [ ] Enable audit logging
- [ ] Configure firewall rules
- [ ] Set up log aggregation

## 📚 Advanced Features

### Dynamic Action Generation
The system can generate new actions at runtime based on natural language descriptions:

```python
from core.dynamic_executor import get_dynamic_action_generator

generator = get_dynamic_action_generator()
action = generator.generate_action(
    "Calculate 20-day SMA for stock prices",
    {"indicator": "sma", "period": 20}
)
```

### Trial Feedback Learning
The system learns from execution results to improve future performance:

```python
from core.trial_feedback_manager import get_trial_feedback_manager

feedback_manager = get_trial_feedback_manager()
recommendations = feedback_manager.get_recommendations("train_random_forest")
```

### Memory Management
Advanced memory system with importance scoring and context retrieval:

```python
from core.enhanced_memory import get_memory_manager

memory = get_memory_manager()
relevant_context = memory.get_relevant_context("portfolio optimization")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Anthropic for Claude AI
- Polygon.io for market data
- Federal Reserve Economic Data (FRED)
- The open-source ML community

## 📞 Support

- Documentation: [docs/](./docs/)
- Issues: [GitHub Issues](https://github.com/yourusername/mltrainer-unified/issues)
- Email: support@mltrainer.ai

---

Built with ❤️ by the mlTrainer Team