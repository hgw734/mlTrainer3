# mlTrainer Deployment Runbook

## Pre-Deployment Checklist

- [ ] All tests passing in CI/CD pipeline
- [ ] Security scan completed (no critical vulnerabilities)
- [ ] Performance benchmarks meet SLA requirements
- [ ] Database migrations reviewed and tested
- [ ] Rollback plan documented and tested
- [ ] Stakeholders notified of deployment window
- [ ] On-call engineer assigned
- [ ] Monitoring dashboards loaded

## Deployment Environments

| Environment | Purpose | URL | Deploy Branch |
|-------------|---------|-----|---------------|
| Development | Developer testing | dev.mltrainer.internal | `develop` |
| Staging | Pre-production validation | staging.mltrainer.windfuhr.net | `staging` |
| Production | Live system | mltrainer.windfuhr.net | `main` |

## 1. Local Development Deployment

```bash
# Clone repository
git clone https://github.com/mltrainer/mltrainer.git
cd mltrainer

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys

# Run migrations
python manage.py migrate

# Start development server
python manage.py runserver

# Or use Docker
docker-compose up -d
```

## 2. Staging Deployment

### 2.1 Pre-deployment Steps

```bash
# Ensure on correct branch
git checkout staging
git pull origin staging

# Run tests
pytest tests/ -v --cov=mltrainer

# Build Docker images
docker build -t mltrainer/api:staging-${BUILD_NUMBER} .
docker build -t mltrainer/ml-engine:staging-${BUILD_NUMBER} ./ml-engine

# Push to registry
docker push mltrainer/api:staging-${BUILD_NUMBER}
docker push mltrainer/ml-engine:staging-${BUILD_NUMBER}
```

### 2.2 Database Migration

```bash
# Backup current database
pg_dump -h staging-db.mltrainer.internal -U mltrainer -d mltrainer > backup-$(date +%Y%m%d-%H%M%S).sql

# Run migrations
kubectl exec -it deployment/api-staging -- python manage.py migrate --check
kubectl exec -it deployment/api-staging -- python manage.py migrate
```

### 2.3 Deploy to Kubernetes

```bash
# Update image tags
kubectl set image deployment/api-staging api=mltrainer/api:staging-${BUILD_NUMBER}
kubectl set image deployment/ml-engine-staging ml-engine=mltrainer/ml-engine:staging-${BUILD_NUMBER}

# Monitor rollout
kubectl rollout status deployment/api-staging
kubectl rollout status deployment/ml-engine-staging

# Verify pods are healthy
kubectl get pods -l env=staging
```

## 3. Production Deployment (Modal)

### 3.1 Pre-deployment Verification

```bash
# Ensure staging is stable for 24 hours
./scripts/check_staging_stability.sh

# Verify no ongoing incidents
curl https://status.mltrainer.windfuhr.net/api/incidents

# Check resource availability
modal status
```

### 3.2 Blue-Green Deployment

```bash
# Deploy to Modal (Blue environment)
cd /workspace
source modal_env/bin/activate

# Deploy new version alongside current
modal deploy modal_app_optimized.py --tag blue-${VERSION}

# Verify health
curl https://blue.mltrainer.windfuhr.net/health

# Run smoke tests
python tests/smoke_tests.py --env blue

# Switch traffic (0% -> 10% -> 50% -> 100%)
modal deploy modal_app_optimized.py --tag production --canary 10
sleep 300  # Monitor for 5 minutes

modal deploy modal_app_optimized.py --tag production --canary 50
sleep 600  # Monitor for 10 minutes

modal deploy modal_app_optimized.py --tag production --canary 100
```

### 3.3 Post-deployment Verification

```bash
# Health checks
curl https://mltrainer.windfuhr.net/health
curl https://api.mltrainer.windfuhr.net/health

# Run integration tests
pytest tests/integration/ -v --env production

# Check key metrics
python scripts/verify_metrics.py --env production

# Monitor error rates
python scripts/check_error_rates.py --duration 30m
```

## 4. Rollback Procedures

### 4.1 Immediate Rollback (< 5 minutes)

```bash
# Modal rollback
modal deploy modal_app_optimized.py --rollback

# Verify rollback
curl https://mltrainer.windfuhr.net/version
```

### 4.2 Database Rollback

```sql
-- Restore from backup
psql -h prod-db.mltrainer.internal -U mltrainer -d mltrainer_restore < backup-20240710-143000.sql

-- Swap databases
ALTER DATABASE mltrainer RENAME TO mltrainer_failed;
ALTER DATABASE mltrainer_restore RENAME TO mltrainer;
```

## 5. Monitoring During Deployment

### 5.1 Key Metrics to Watch

```yaml
Critical Metrics:
  - API Response Time: < 200ms p99
  - Error Rate: < 0.1%
  - CPU Usage: < 80%
  - Memory Usage: < 85%
  - Active Connections: < 5000

Business Metrics:
  - Predictions per Second: > 1000
  - Model Accuracy: > 0.85
  - Cache Hit Rate: > 70%
```

### 5.2 Monitoring Commands

```bash
# Real-time logs
modal logs -f mltrainer

# Metrics dashboard
open https://monitor.mltrainer.windfuhr.net

# Error tracking
open https://sentry.mltrainer.internal

# Performance monitoring
open https://grafana.mltrainer.internal/d/mltrainer-prod
```

## 6. Common Issues & Solutions

### Issue: High Memory Usage
```bash
# Check memory usage
kubectl top pods -l app=ml-engine

# Restart pods with memory issues
kubectl delete pod ml-engine-xxxxx

# Scale horizontally if needed
kubectl scale deployment/ml-engine --replicas=5
```

### Issue: Slow Predictions
```bash
# Check cache status
redis-cli -h cache.mltrainer.internal INFO stats

# Warm cache if needed
python scripts/warm_cache.py --models all

# Check model loading
curl https://api.mltrainer.windfuhr.net/models/status
```

### Issue: Database Connection Errors
```bash
# Check connection pool
psql -h prod-db.mltrainer.internal -U mltrainer -c "SELECT count(*) FROM pg_stat_activity;"

# Reset connections if needed
kubectl exec -it deployment/api -- python manage.py reset_db_connections
```

## 7. Post-Deployment Tasks

- [ ] Update status page
- [ ] Send deployment notification
- [ ] Update documentation
- [ ] Close deployment ticket
- [ ] Schedule retrospective
- [ ] Monitor for 24 hours
- [ ] Update runbook with learnings

## 8. Emergency Contacts

| Role | Name | Contact | Escalation |
|------|------|---------|------------|
| DevOps Lead | John Doe | john@mltrainer.com | Primary |
| ML Lead | Jane Smith | jane@mltrainer.com | Secondary |
| CTO | Bob Johnson | bob@mltrainer.com | Executive |
| Modal Support | - | support@modal.com | Vendor |

## 9. Deployment Schedule

| Day | Time (EST) | Type | Duration |
|-----|------------|------|----------|
| Tuesday | 10:00 PM | Staging | 30 min |
| Thursday | 10:00 PM | Production | 60 min |
| Sunday | 6:00 AM | Maintenance | 120 min |

## 10. Compliance & Audit

```bash
# Generate deployment report
python scripts/generate_deployment_report.py \
  --version ${VERSION} \
  --deployer ${USER} \
  --ticket JIRA-1234 \
  > reports/deployment-$(date +%Y%m%d).json

# Log deployment for audit
echo "$(date): Deployed version ${VERSION} by ${USER}" >> /var/log/mltrainer/deployments.log

# Verify compliance
python scripts/compliance_check.py --deployment ${VERSION}
```

## Appendix: Scripts

### Health Check Script
```python
#!/usr/bin/env python3
import requests
import sys

endpoints = [
    'https://mltrainer.windfuhr.net/health',
    'https://api.mltrainer.windfuhr.net/health',
    'https://monitor.mltrainer.windfuhr.net/health'
]

for endpoint in endpoints:
    try:
        resp = requests.get(endpoint, timeout=5)
        if resp.status_code != 200:
            print(f"FAIL: {endpoint} returned {resp.status_code}")
            sys.exit(1)
        print(f"OK: {endpoint}")
    except Exception as e:
        print(f"FAIL: {endpoint} - {e}")
        sys.exit(1)

print("All health checks passed!")
```