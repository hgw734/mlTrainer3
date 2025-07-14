# mlTrainer API Documentation

## Overview

The mlTrainer API provides programmatic access to all ML models, predictions, and system capabilities.

### Base URLs

- **Production**: `https://api.mltrainer.windfuhr.net`
- **Staging**: `https://staging-api.mltrainer.windfuhr.net`
- **Development**: `http://localhost:8000`

### Authentication

All API requests require JWT authentication:

```bash
Authorization: Bearer <jwt_token>
```

## Endpoints

### 1. Predictions

#### `POST /api/v1/predict`

Generate predictions for a given symbol.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "model": "random_forest",
  "timeframe": "1d",
  "features": {
    "volume": true,
    "technical_indicators": ["RSI", "MACD", "BB"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "symbol": "AAPL",
    "model": "random_forest",
    "value": 175.50,
    "confidence": 0.85,
    "direction": "up",
    "timestamp": "2024-07-10T15:30:00Z"
  },
  "metadata": {
    "model_version": "2.1.0",
    "features_used": 45,
    "computation_time_ms": 123
  }
}
```

#### `POST /api/v1/predict/batch`

Batch predictions for multiple symbols.

**Request Body:**
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "model": "ensemble",
  "async": true
}
```

**Response:**
```json
{
  "status": "accepted",
  "job_id": "pred_batch_12345",
  "estimated_completion": "2024-07-10T15:35:00Z",
  "webhook_url": "https://api.mltrainer.windfuhr.net/jobs/pred_batch_12345"
}
```

### 2. Model Management

#### `GET /api/v1/models`

List all available models.

**Query Parameters:**
- `category`: Filter by category (ml, deep_learning, financial, etc.)
- `status`: Filter by status (active, training, deprecated)
- `page`: Page number (default: 1)
- `limit`: Results per page (default: 20)

**Response:**
```json
{
  "models": [
    {
      "id": "random_forest_v2",
      "name": "Random Forest",
      "category": "machine_learning",
      "status": "active",
      "accuracy": 0.89,
      "last_trained": "2024-07-09T12:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "total_pages": 7,
    "total_items": 140
  }
}
```

#### `POST /api/v1/models/train`

Train a new model or retrain existing.

**Request Body:**
```json
{
  "model_type": "xgboost",
  "dataset": "market_data_2024",
  "parameters": {
    "n_estimators": 1000,
    "max_depth": 10,
    "learning_rate": 0.01
  },
  "validation_split": 0.2,
  "gpu_enabled": true
}
```

### 3. Market Data

#### `GET /api/v1/market/quote/{symbol}`

Get real-time quote for a symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "price": 175.25,
  "change": 2.50,
  "change_percent": 1.45,
  "volume": 45678900,
  "timestamp": "2024-07-10T15:30:45Z"
}
```

#### `GET /api/v1/market/history/{symbol}`

Get historical data.

**Query Parameters:**
- `period`: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
- `interval`: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

### 4. System Health

#### `GET /api/v1/health`

System health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-07-10T15:30:00Z",
  "services": {
    "database": "healthy",
    "cache": "healthy",
    "ml_engine": "healthy",
    "market_data": "healthy"
  },
  "metrics": {
    "uptime_seconds": 864000,
    "active_models": 140,
    "requests_per_minute": 150
  }
}
```

## Rate Limiting

- **Free tier**: 100 requests/hour
- **Pro tier**: 1,000 requests/hour
- **Enterprise**: Unlimited

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1625097600
```

## Error Responses

Standard error format:
```json
{
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "Symbol 'XYZ' not found",
    "details": {
      "provided_symbol": "XYZ",
      "valid_symbols_example": ["AAPL", "GOOGL", "MSFT"]
    }
  },
  "request_id": "req_12345",
  "timestamp": "2024-07-10T15:30:00Z"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `MODEL_ERROR` | 500 | Model computation error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## WebSocket API

### Real-time Predictions

```javascript
const ws = new WebSocket('wss://api.mltrainer.windfuhr.net/ws');

ws.send(JSON.stringify({
  action: 'subscribe',
  channels: ['predictions', 'market_data'],
  symbols: ['AAPL', 'GOOGL']
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Real-time update:', data);
};
```

## SDK Examples

### Python
```python
from mltrainer import Client

client = Client(api_key='your_api_key')

# Single prediction
result = client.predict('AAPL', model='ensemble')

# Batch prediction
results = client.predict_batch(['AAPL', 'GOOGL', 'MSFT'])

# Train model
job = client.train_model('xgboost', dataset='market_2024')
```

### JavaScript/TypeScript
```typescript
import { MLTrainerClient } from '@mltrainer/sdk';

const client = new MLTrainerClient({ apiKey: 'your_api_key' });

// Async/await pattern
const prediction = await client.predict('AAPL');

// Promise pattern
client.predictBatch(['AAPL', 'GOOGL'])
  .then(results => console.log(results))
  .catch(error => console.error(error));
```

## Webhooks

Configure webhooks for async operations:

```json
{
  "url": "https://your-app.com/webhooks/mltrainer",
  "events": ["prediction.completed", "training.finished"],
  "secret": "your_webhook_secret"
}
```

## Changelog

### v2.1.0 (2024-07-10)
- Added batch prediction endpoint
- Improved model training API
- WebSocket support for real-time updates

### v2.0.0 (2024-06-01)
- Complete API redesign
- Added 140+ model support
- GPU training capabilities

### v1.0.0 (2024-01-01)
- Initial release