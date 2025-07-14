# 🚀 mlTrainer Modal Deployment Guide

This guide will help you deploy mlTrainer to Modal with custom domain, optimizations, and monitoring.

## 📋 Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install with `pip install modal`
3. **Modal Token**: Run `modal token new` and authenticate
4. **Domain Access**: Admin access to windfuhr.net DNS settings

## 🔧 Quick Start

### 1. Set Up Secrets

```bash
# Create secrets in Modal dashboard
modal secret create mltrainer-secrets \
  ANTHROPIC_API_KEY="your-key" \
  COHERE_API_KEY="your-key" \
  JWT_SECRET="your-secret" \
  MODAL_WORKSPACE="your-workspace-name"
```

### 2. Deploy Base Application

```bash
# Deploy the optimized version
modal deploy modal_app_optimized.py

# Note your endpoints:
# https://[workspace]--mltrainer-app.modal.run
# https://[workspace]--mltrainer-api.modal.run
# https://[workspace]--mltrainer-health.modal.run
```

### 3. Deploy Monitoring Dashboard

```bash
# Deploy monitoring separately
modal deploy modal_monitoring_dashboard.py

# Access at:
# https://[workspace]--mltrainer-monitor.modal.run
```

### 4. Configure Custom Domain

Follow the steps in `docs/CUSTOM_DOMAIN_SETUP.md`:

1. Add CNAME records to windfuhr.net
2. Configure domains in Modal dashboard
3. Wait for SSL certificates
4. Update app configuration

### 5. Verify Deployment

```bash
# Check health endpoint
curl https://monitor.mltrainer.windfuhr.net/health

# Test API
curl -X POST https://api.mltrainer.windfuhr.net/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "model": "random_forest"}'

# Visit main app
open https://mltrainer.windfuhr.net
```

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    windfuhr.net                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────┐│
│  │   Streamlit    │  │     API      │  │  Monitor   ││
│  │   (Main App)   │  │  (FastAPI)   │  │(Dashboard) ││
│  └────────┬────────┘  └──────┬───────┘  └─────┬──────┘│
│           │                  │                 │        │
│  ┌────────┴──────────────────┴─────────────────┴──────┐│
│  │                    Modal Cloud                      ││
│  ├─────────────────────────────────────────────────────┤│
│  │  • Dynamic Scaling  • GPU Training  • Redis Cache  ││
│  │  • Persistent Storage  • Scheduled Jobs            ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## ⚡ Key Features

### Optimizations
- **Dynamic Scaling**: 5 containers during market hours, 1 off-hours
- **Redis Caching**: 5-minute TTL for predictions
- **Model Preloading**: Common models loaded on startup
- **Connection Pooling**: Reused database connections
- **GPU Scheduling**: T4 GPUs for training jobs

### Monitoring
- **Real-time Metrics**: API latency, container count, memory usage
- **Cost Tracking**: Daily/weekly/monthly with projections
- **Log Viewer**: Filtered logs with search
- **Alert System**: Email/SMS/Slack notifications
- **Health Checks**: Component status monitoring

### Security
- **Secrets Management**: Environment-based configuration
- **JWT Authentication**: Secure API endpoints
- **CORS Protection**: Restricted origins
- **SSL/TLS**: Automatic certificates via Let's Encrypt

## 🔍 Troubleshooting

### Container Issues
```bash
# View logs
modal logs -f mltrainer

# Check container status
modal app list

# Force restart
modal deploy --force modal_app_optimized.py
```

### DNS/Domain Issues
```bash
# Verify DNS propagation
nslookup mltrainer.windfuhr.net

# Check Modal domain status
modal app describe mltrainer
```

### Performance Issues
1. Check monitoring dashboard: https://monitor.mltrainer.windfuhr.net
2. Review container scaling settings
3. Check Redis cache hit rates
4. Monitor GPU utilization during training

## 📈 Scaling Guidelines

| Metric | Threshold | Action |
|--------|-----------|--------|
| API Latency | >200ms | Increase container count |
| Memory Usage | >85% | Upgrade memory allocation |
| Cache Hit Rate | <70% | Increase cache TTL |
| Error Rate | >5% | Check logs, add retries |
| Monthly Cost | >$150 | Review optimization suggestions |

## 🛠️ Maintenance

### Daily
- Monitor dashboard for alerts
- Check cost projections

### Weekly
- Review performance metrics
- Update model registry
- Clean old logs

### Monthly
- Analyze cost breakdown
- Optimize container scaling
- Update dependencies

## 📞 Support

- **Modal Documentation**: https://modal.com/docs
- **mlTrainer Issues**: Create issue in GitHub repo
- **Emergency**: Check monitoring dashboard alerts

---

**Pro Tip**: Use the monitoring dashboard's cost optimization suggestions to reduce monthly bills by up to 40%!