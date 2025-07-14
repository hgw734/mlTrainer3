# 🚀 mlTrainer Step-by-Step Build Plan

## Overview
Build the system incrementally, testing each layer before moving to the next. Focus on the real architecture: AI → Natural Language → MLAgent Bridge → ML Engine.

---

## Phase 1: Foundation & Critical Fixes (Day 1)

### Step 1.1: Fix Security & Dependencies
```bash
# 1. Remove hardcoded API keys
sed -i 's/os.getenv("POLYGON_API_KEY", ".*")/os.getenv("POLYGON_API_KEY")/' config/api_config.py
sed -i 's/os.getenv("FRED_API_KEY", ".*")/os.getenv("FRED_API_KEY")/' config/api_config.py

# 2. Install missing dependencies
pip install pandas numpy scikit-learn pyjwt prometheus-client scipy

# 3. Create .env file
cp .env.example .env
# Add your API keys to .env
```

### Step 1.2: Fix Import Errors
```python
# backend/compliance_engine.py - Add missing function
def get_compliance_gateway():
    """Return singleton compliance gateway instance"""
    global _compliance_instance
    if _compliance_instance is None:
        _compliance_instance = ComplianceEngine()
    return _compliance_instance
```

### 🧪 Functional Test 1.1: Basic System Health
```python
# test_phase1_foundation.py
import sys
import importlib

def test_critical_imports():
    """Test all critical imports work"""
    critical_modules = [
        'pandas',
        'numpy',
        'sklearn',
        'jwt',
        'prometheus_client',
        'core.unified_executor',
        'backend.compliance_engine',
        'mlagent_bridge'
    ]
    
    failed = []
    for module in critical_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            failed.append((module, str(e)))
            print(f"❌ {module}: {e}")
    
    assert len(failed) == 0, f"Failed imports: {failed}"
    print("\n✅ All critical imports successful!")

def test_api_configuration():
    """Test API keys are properly configured"""
    import os
    from config.api_config import POLYGON_API_KEY, FRED_API_KEY
    
    # Should raise error if not set
    assert POLYGON_API_KEY, "POLYGON_API_KEY not configured"
    assert FRED_API_KEY, "FRED_API_KEY not configured"
    print("✅ API keys configured")

if __name__ == "__main__":
    test_critical_imports()
    test_api_configuration()
```

**Run**: `python test_phase1_foundation.py`

---

## Phase 2: MLAgent Bridge - The Core Conduit (Day 2)

### Step 2.1: Enhance MLAgent Bridge Pattern Recognition
```python
# Extend mlagent_bridge.py with more patterns
def _initialize_extended_patterns(self):
    """Add more sophisticated pattern matching"""
    return {
        'data_source': [
            re.compile(r'(?:use|fetch|get)\s+data\s+from\s+(\w+)', re.IGNORECASE),
            re.compile(r'data[_\s]?source[:\s]+(\w+)', re.IGNORECASE),
        ],
        'feature_engineering': [
            re.compile(r'features?[:\s]+\[(.*?)\]', re.IGNORECASE),
            re.compile(r'add\s+(\w+)\s+as\s+feature', re.IGNORECASE),
        ],
        'validation_strategy': [
            re.compile(r'validation[:\s]+(\w+)', re.IGNORECASE),
            re.compile(r'use\s+(\w+)\s+validation', re.IGNORECASE),
        ]
    }
```

### Step 2.2: Create Mock ML Engine for Testing
```python
# mock_ml_engine.py
class MockMLEngine:
    """Mock ML engine for testing the conduit"""
    
    def __init__(self):
        self.active_model = None
        self.training_data = None
        self.is_training = False
        
    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parsed command from MLAgent Bridge"""
        action = command.get('action', 'unknown')
        
        if action == 'train_model':
            return self._mock_train_model(command['params'])
        elif action == 'fetch_data':
            return self._mock_fetch_data(command['params'])
        elif action == 'evaluate':
            return self._mock_evaluate(command['params'])
        else:
            return {'status': 'error', 'message': f'Unknown action: {action}'}
    
    def _mock_train_model(self, params):
        """Simulate model training"""
        self.is_training = True
        import time
        time.sleep(0.5)  # Simulate training time
        self.is_training = False
        
        return {
            'status': 'success',
            'type': 'training_complete',
            'model': params.get('model', 'lstm'),
            'accuracy': 0.85 + (hash(str(params)) % 10) / 100,
            'loss': 0.15 - (hash(str(params)) % 5) / 100,
            'epochs': params.get('epochs', 50)
        }
```

### 🧪 Functional Test 2.1: Conduit Pattern Testing
```python
# test_phase2_conduit.py
from mlagent_bridge import MLAgentBridge
from mock_ml_engine import MockMLEngine
import json

def test_natural_language_parsing():
    """Test AI natural language → structured command conversion"""
    bridge = MLAgentBridge()
    
    test_cases = [
        {
            'input': "Let's train a random forest model on AAPL with 80% train ratio",
            'expected': {
                'symbol': 'AAPL',
                'model': 'random_forest',
                'train_ratio': 0.8
            }
        },
        {
            'input': "Use LSTM with 60 day lookback, train for 100 epochs",
            'expected': {
                'model': 'lstm',
                'lookback': 60,
                'epochs': 100
            }
        }
    ]
    
    for i, test in enumerate(test_cases):
        parsed = bridge.parse_mltrainer_response(test['input'])
        params = parsed['extracted_params']
        
        # Check each expected value
        for key, expected_value in test['expected'].items():
            if key in ['symbol', 'model']:
                assert params.get(key) == expected_value, \
                    f"Test {i}: Expected {key}={expected_value}, got {params.get(key)}"
            else:
                assert params.get('parameters', {}).get(key) == expected_value, \
                    f"Test {i}: Expected {key}={expected_value}"
        
        print(f"✅ Test case {i}: Correctly parsed '{test['input'][:50]}...'")
    
    print("\n✅ Natural language parsing working correctly!")

def test_ml_feedback_formatting():
    """Test ML engine data → natural language conversion"""
    bridge = MLAgentBridge()
    
    ml_feedback = {
        'type': 'model_performance',
        'model': 'random_forest',
        'accuracy': 0.87,
        'precision': 0.85,
        'sharpe': 1.2
    }
    
    question = bridge.format_ml_feedback_as_question(ml_feedback)
    
    # Should contain key information
    assert 'random_forest' in question
    assert '87' in question  # accuracy
    assert 'continue, retrain, or switch' in question
    
    print(f"✅ ML feedback formatted as: {question[:100]}...")
    print("✅ ML → Natural language conversion working!")

def test_end_to_end_conduit():
    """Test complete flow: AI text → Bridge → ML → Bridge → AI text"""
    bridge = MLAgentBridge()
    ml_engine = MockMLEngine()
    
    # Step 1: AI provides instruction
    ai_response = "Train an LSTM model on TSLA with 70% train ratio for 50 epochs"
    
    # Step 2: Bridge parses to command
    parsed = bridge.parse_mltrainer_response(ai_response)
    trial_config = bridge.create_trial_config(parsed)
    
    assert trial_config['symbol'] == 'TSLA'
    assert trial_config['model'] == 'lstm'
    print("✅ Step 1-2: AI → Bridge parsing successful")
    
    # Step 3: Execute in ML engine
    ml_command = {
        'action': 'train_model',
        'params': trial_config
    }
    ml_result = ml_engine.execute_command(ml_command)
    
    assert ml_result['status'] == 'success'
    print("✅ Step 3: ML engine execution successful")
    
    # Step 4: Convert result back to natural language
    feedback_question = bridge.format_ml_feedback_as_question(ml_result)
    
    assert 'LSTM' in feedback_question or 'lstm' in feedback_question
    assert 'accuracy' in feedback_question.lower()
    print("✅ Step 4: ML → Natural language successful")
    
    print(f"\n✅ Complete conduit flow tested!")
    print(f"   AI said: '{ai_response}'")
    print(f"   ML executed: {trial_config['model']} on {trial_config['symbol']}")
    print(f"   Result: {ml_result['accuracy']:.2%} accuracy")
    print(f"   Question back: '{feedback_question[:80]}...'")

if __name__ == "__main__":
    test_natural_language_parsing()
    print("\n" + "="*50 + "\n")
    test_ml_feedback_formatting()
    print("\n" + "="*50 + "\n")
    test_end_to_end_conduit()
```

**Run**: `python test_phase2_conduit.py`

---

## Phase 3: Real ML Engine Integration (Day 3-4)

### Step 3.1: Implement Real Model Training
```python
# ml_engine_real.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RealMLEngine:
    """Real ML engine with actual model training"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            # Add more as needed
        }
        self.trained_models = {}
        self.scalers = {}
        
    def train_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Actually train a model with real data"""
        try:
            # Get data (mock for now, real data connection in Phase 4)
            X, y = self._prepare_mock_data(params['symbol'])
            
            # Split data
            train_ratio = params.get('parameters', {}).get('train_ratio', 0.8)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_ratio, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model_name = params.get('model', 'random_forest')
            model_class = self.models.get(model_name, RandomForestClassifier)
            model = model_class()
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Store model and scaler
            model_id = f"{params['symbol']}_{model_name}_{params['id']}"
            self.trained_models[model_id] = model
            self.scalers[model_id] = scaler
            
            return {
                'status': 'success',
                'type': 'training_complete',
                'model_id': model_id,
                'train_accuracy': float(train_score),
                'test_accuracy': float(test_score),
                'overfitting_risk': float(train_score - test_score)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'type': 'training_failed',
                'error': str(e)
            }
    
    def _prepare_mock_data(self, symbol: str):
        """Generate mock financial data for testing"""
        # In production, this would fetch real data
        n_samples = 1000
        n_features = 10
        
        # Create synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Create target (1 for price up, 0 for price down)
        # Add some pattern so model can learn
        y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.5) > 0
        y = y.astype(int)
        
        return X, y
```

### 🧪 Functional Test 3.1: Real Model Training
```python
# test_phase3_ml_engine.py
from ml_engine_real import RealMLEngine
from mlagent_bridge import MLAgentBridge
import json

def test_real_model_training():
    """Test actual scikit-learn model training"""
    engine = RealMLEngine()
    
    params = {
        'id': 'test_001',
        'symbol': 'AAPL',
        'model': 'random_forest',
        'parameters': {
            'train_ratio': 0.8
        }
    }
    
    result = engine.train_model(params)
    
    assert result['status'] == 'success'
    assert 'train_accuracy' in result
    assert 0 <= result['train_accuracy'] <= 1
    assert 'test_accuracy' in result
    assert 'model_id' in result
    
    print(f"✅ Model trained successfully!")
    print(f"   Train accuracy: {result['train_accuracy']:.2%}")
    print(f"   Test accuracy: {result['test_accuracy']:.2%}")
    print(f"   Overfitting risk: {result['overfitting_risk']:.2%}")
    
    # Verify model is stored
    assert result['model_id'] in engine.trained_models
    print(f"✅ Model stored with ID: {result['model_id']}")

def test_multiple_models():
    """Test training different model types"""
    engine = RealMLEngine()
    bridge = MLAgentBridge()
    
    test_instructions = [
        "Train random forest on AAPL",
        "Train logistic regression on TSLA"
    ]
    
    for instruction in test_instructions:
        # Parse instruction
        parsed = bridge.parse_mltrainer_response(instruction)
        config = bridge.create_trial_config(parsed)
        
        # Train model
        result = engine.train_model(config)
        
        assert result['status'] == 'success'
        print(f"✅ Trained {config['model']} on {config['symbol']}")
    
    print(f"\n✅ Multiple models trained successfully!")
    print(f"   Total models in memory: {len(engine.trained_models)}")

if __name__ == "__main__":
    test_real_model_training()
    print("\n" + "="*50 + "\n")
    test_multiple_models()
```

**Run**: `python test_phase3_ml_engine.py`

---

## Phase 4: Data Integration (Day 5-6)

### Step 4.1: Implement Real Data Fetchers
```python
# data_fetcher_real.py
import pandas as pd
from polygon_connector import PolygonConnector
from fred_connector import FREDConnector

class DataFetcherReal:
    """Real data fetching with fallback to mock data"""
    
    def __init__(self):
        self.polygon = PolygonConnector()
        self.fred = FREDConnector()
        self.use_mock = False  # Set True if APIs unavailable
        
    def fetch_stock_data(self, symbol: str, days: int = 100):
        """Fetch real stock data with mock fallback"""
        try:
            if not self.use_mock:
                # Try real data first
                df = self.polygon.get_stock_data(symbol, days)
                if df is not None and not df.empty:
                    return self._prepare_features(df)
        except Exception as e:
            print(f"⚠️ Real data fetch failed: {e}, using mock data")
        
        # Fallback to mock
        return self._generate_mock_stock_data(symbol, days)
    
    def _prepare_features(self, df: pd.DataFrame):
        """Convert raw data to ML features"""
        features = pd.DataFrame()
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Technical indicators
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        features['rsi'] = self._calculate_rsi(df['close'])
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Target: Next day direction
        features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return features.dropna()
```

### 🧪 Functional Test 4.1: Data Pipeline
```python
# test_phase4_data.py
def test_data_pipeline():
    """Test complete data fetching and preparation"""
    fetcher = DataFetcherReal()
    
    # Test with mock data (safer for testing)
    fetcher.use_mock = True
    
    features = fetcher.fetch_stock_data('AAPL', days=100)
    
    assert len(features) > 50  # Some rows lost to rolling calculations
    assert 'returns' in features.columns
    assert 'target' in features.columns
    
    print(f"✅ Data pipeline working!")
    print(f"   Features shape: {features.shape}")
    print(f"   Columns: {list(features.columns)}")
```

---

## Phase 5: Complete Integration (Day 7-8)

### Step 5.1: Wire Everything Together
```python
# mltrainer_system_integrated.py
class MLTrainerSystem:
    """Complete integrated system"""
    
    def __init__(self):
        self.bridge = MLAgentBridge()
        self.ml_engine = RealMLEngine()
        self.data_fetcher = DataFetcherReal()
        self.compliance = ComplianceEngine()
        
    def process_ai_instruction(self, instruction: str):
        """Complete flow from AI instruction to result"""
        # 1. Parse instruction
        parsed = self.bridge.parse_mltrainer_response(instruction)
        
        # 2. Check compliance
        if not self.compliance.check_data_source(parsed.get('data_source')):
            return {'error': 'Non-compliant data source'}
        
        # 3. Create trial config
        config = self.bridge.create_trial_config(parsed)
        
        # 4. Fetch data
        data = self.data_fetcher.fetch_stock_data(
            config['symbol'], 
            config.get('lookback', 100)
        )
        
        # 5. Train model
        result = self.ml_engine.train_model(config, data)
        
        # 6. Format result as natural language
        feedback = self.bridge.format_ml_feedback_as_question(result)
        
        return {
            'original_instruction': instruction,
            'parsed_config': config,
            'ml_result': result,
            'feedback_question': feedback
        }
```

### 🧪 Functional Test 5.1: End-to-End System Test
```python
# test_phase5_integration.py
def test_complete_system():
    """Test the complete integrated system"""
    system = MLTrainerSystem()
    
    # Simulate AI giving instructions
    ai_instructions = [
        "Train a random forest model on AAPL with 80% training data",
        "Use LSTM on TSLA with 60 day lookback for momentum trading",
        "Analyze MSFT with logistic regression, focus on 30-day patterns"
    ]
    
    for instruction in ai_instructions:
        print(f"\n🤖 AI says: '{instruction}'")
        
        result = system.process_ai_instruction(instruction)
        
        assert 'ml_result' in result
        assert result['ml_result']['status'] == 'success'
        
        print(f"✅ Processed successfully!")
        print(f"   Model: {result['parsed_config']['model']}")
        print(f"   Symbol: {result['parsed_config']['symbol']}")
        print(f"   Accuracy: {result['ml_result'].get('test_accuracy', 'N/A')}")
        print(f"📝 Question back: '{result['feedback_question'][:100]}...'")
```

---

## Phase 6: Production Readiness (Day 9-10)

### Step 6.1: Add Monitoring & Logging
```python
# monitoring.py
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Metrics
model_training_counter = Counter('ml_model_training_total', 'Total model trainings')
model_accuracy_gauge = Gauge('ml_model_accuracy', 'Model accuracy', ['model_type', 'symbol'])
parsing_duration = Histogram('parsing_duration_seconds', 'Time to parse AI instructions')
```

### Step 6.2: Dockerize
```dockerfile
# Dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "mltrainer_unified_chat.py"]
```

### 🧪 Functional Test 6.1: Production Readiness
```python
# test_phase6_production.py
def test_system_under_load():
    """Test system can handle multiple concurrent requests"""
    import concurrent.futures
    system = MLTrainerSystem()
    
    instructions = [
        f"Train model on {symbol}" 
        for symbol in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(system.process_ai_instruction, inst) 
                  for inst in instructions]
        
        results = [f.result() for f in futures]
    
    assert all(r['ml_result']['status'] == 'success' for r in results)
    print(f"✅ System handled {len(results)} concurrent requests!")
```

---

## Validation Checkpoints

### Checkpoint 1: Foundation (After Phase 1)
- [ ] All dependencies installed
- [ ] No import errors
- [ ] API keys configured
- [ ] Basic tests pass

### Checkpoint 2: Conduit Working (After Phase 2)
- [ ] Natural language → Commands parsing works
- [ ] ML results → Natural language works
- [ ] End-to-end conduit flow tested

### Checkpoint 3: Real ML (After Phase 3)
- [ ] Actual models training
- [ ] Models stored in memory
- [ ] Accuracy metrics realistic

### Checkpoint 4: Data Integration (After Phase 4)
- [ ] Can fetch real or mock data
- [ ] Feature engineering works
- [ ] Data pipeline stable

### Checkpoint 5: Complete System (After Phase 5)
- [ ] Full flow from AI text to ML results works
- [ ] Compliance checks active
- [ ] System responds appropriately

### Checkpoint 6: Production Ready (After Phase 6)
- [ ] Monitoring active
- [ ] Docker container builds
- [ ] Can handle concurrent requests
- [ ] All tests green

---

## Success Criteria

1. **Functional**: AI can give natural language instructions that result in actual model training
2. **Reliable**: System handles errors gracefully, has fallbacks
3. **Transparent**: The "conduit" pattern is working as designed
4. **Testable**: Each component can be tested independently
5. **Scalable**: Can handle multiple models/requests

## Total Timeline: 10 Days

With functional tests at each phase ensuring we build on solid foundations.