# ML/AI Integration: 9/10 → 10/10 Upgrade Plan

## Current State (9/10)
✅ Self-learning engine with helpers pattern
✅ Multi-model architecture supporting 140+ model types
✅ Walk-forward trial system for backtesting
✅ Real ML/financial integration (not just toy examples)
✅ Paper processing pipeline for research integration

## Missing for 10/10

### 1. Model Versioning System

**File: `core/model_versioning.py`**
```python
"""
Model Versioning System
=======================
Tracks model versions, lineage, and experiments
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import dill  # Better than pickle for complex objects

@dataclass
class ModelVersion:
    model_id: str
    version: str
    parent_version: Optional[str]
    created_at: datetime
    created_by: str
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    data_fingerprint: str
    code_fingerprint: str
    tags: List[str]
    status: str  # 'training', 'ready', 'deployed', 'deprecated'
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

class ModelVersionControl:
    """
    Git-like version control for ML models
    """
    
    def __init__(self, storage_backend='s3'):
        self.storage_backend = storage_backend
        self.versions: Dict[str, List[ModelVersion]] = {}
        self.deployments: Dict[str, str] = {}  # model_id -> deployed_version
        
    def create_version(self, 
                      model_id: str,
                      model_object: Any,
                      training_config: Dict[str, Any],
                      performance_metrics: Dict[str, float],
                      data_info: Dict[str, Any],
                      created_by: str = 'system') -> str:
        """Create a new model version"""
        
        # Generate version hash
        version_data = {
            'model_id': model_id,
            'timestamp': datetime.now().isoformat(),
            'config': training_config,
            'metrics': performance_metrics
        }
        version_hash = hashlib.sha256(
            json.dumps(version_data, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        # Find parent version
        parent_version = None
        if model_id in self.versions and self.versions[model_id]:
            parent_version = self.versions[model_id][-1].version
        
        # Create version object
        version = ModelVersion(
            model_id=model_id,
            version=version_hash,
            parent_version=parent_version,
            created_at=datetime.now(),
            created_by=created_by,
            training_config=training_config,
            performance_metrics=performance_metrics,
            data_fingerprint=self._compute_data_fingerprint(data_info),
            code_fingerprint=self._compute_code_fingerprint(model_object),
            tags=[],
            status='ready'
        )
        
        # Store version
        if model_id not in self.versions:
            self.versions[model_id] = []
        self.versions[model_id].append(version)
        
        # Save model object
        self._save_model_object(model_id, version_hash, model_object)
        
        return version_hash
    
    def get_version(self, model_id: str, version: str = 'latest') -> Optional[ModelVersion]:
        """Get a specific model version"""
        if model_id not in self.versions:
            return None
            
        if version == 'latest':
            return self.versions[model_id][-1] if self.versions[model_id] else None
        
        for v in self.versions[model_id]:
            if v.version == version:
                return v
        return None
    
    def load_model(self, model_id: str, version: str = 'latest') -> Any:
        """Load a specific model version"""
        model_version = self.get_version(model_id, version)
        if not model_version:
            raise ValueError(f"Version {version} not found for model {model_id}")
            
        return self._load_model_object(model_id, model_version.version)
    
    def compare_versions(self, model_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        v1 = self.get_version(model_id, version1)
        v2 = self.get_version(model_id, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        return {
            'config_diff': self._diff_configs(v1.training_config, v2.training_config),
            'metrics_diff': self._diff_metrics(v1.performance_metrics, v2.performance_metrics),
            'data_changed': v1.data_fingerprint != v2.data_fingerprint,
            'code_changed': v1.code_fingerprint != v2.code_fingerprint,
            'time_diff': (v2.created_at - v1.created_at).total_seconds()
        }
    
    def rollback(self, model_id: str, target_version: str) -> str:
        """Rollback to a previous version"""
        # Create new version pointing to old model
        old_version = self.get_version(model_id, target_version)
        if not old_version:
            raise ValueError(f"Target version {target_version} not found")
            
        # Load old model
        old_model = self.load_model(model_id, target_version)
        
        # Create new version as rollback
        new_version = self.create_version(
            model_id=model_id,
            model_object=old_model,
            training_config=old_version.training_config,
            performance_metrics=old_version.performance_metrics,
            data_info={'fingerprint': old_version.data_fingerprint},
            created_by='rollback'
        )
        
        # Tag it
        self.tag_version(model_id, new_version, f'rollback_from_{target_version}')
        
        return new_version
    
    def tag_version(self, model_id: str, version: str, tag: str):
        """Add a tag to a version"""
        model_version = self.get_version(model_id, version)
        if model_version and tag not in model_version.tags:
            model_version.tags.append(tag)
    
    def _compute_data_fingerprint(self, data_info: Dict[str, Any]) -> str:
        """Compute fingerprint of training data"""
        return hashlib.md5(
            json.dumps(data_info, sort_keys=True).encode()
        ).hexdigest()
    
    def _compute_code_fingerprint(self, model_object: Any) -> str:
        """Compute fingerprint of model code"""
        # Serialize model structure
        model_str = str(type(model_object)) + str(model_object.__dict__)
        return hashlib.md5(model_str.encode()).hexdigest()
    
    def _save_model_object(self, model_id: str, version: str, model_object: Any):
        """Save model object to storage"""
        # In production, save to S3/GCS/Azure
        path = f"models/{model_id}/{version}.pkl"
        with open(path, 'wb') as f:
            dill.dump(model_object, f)
    
    def _load_model_object(self, model_id: str, version: str) -> Any:
        """Load model object from storage"""
        path = f"models/{model_id}/{version}.pkl"
        with open(path, 'rb') as f:
            return dill.load(f)
    
    def _diff_configs(self, config1: Dict, config2: Dict) -> Dict[str, Any]:
        """Compare configurations"""
        diff = {}
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            if key not in config1:
                diff[key] = {'added': config2[key]}
            elif key not in config2:
                diff[key] = {'removed': config1[key]}
            elif config1[key] != config2[key]:
                diff[key] = {'old': config1[key], 'new': config2[key]}
                
        return diff
    
    def _diff_metrics(self, metrics1: Dict, metrics2: Dict) -> Dict[str, Any]:
        """Compare metrics with improvement indicators"""
        diff = {}
        
        for key in set(metrics1.keys()) | set(metrics2.keys()):
            if key in metrics1 and key in metrics2:
                change = metrics2[key] - metrics1[key]
                pct_change = (change / metrics1[key] * 100) if metrics1[key] != 0 else 0
                diff[key] = {
                    'old': metrics1[key],
                    'new': metrics2[key],
                    'change': change,
                    'pct_change': pct_change,
                    'improved': change > 0 if 'loss' not in key else change < 0
                }
                
        return diff
```

### 2. A/B Testing Framework

**File: `core/ab_testing.py`**
```python
"""
A/B Testing Framework for Models
=================================
Statistical testing and gradual rollout
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class ABTestConfig:
    test_name: str
    model_a: str  # Control model
    model_b: str  # Treatment model
    traffic_split: float = 0.5  # Percentage to model B
    min_samples: int = 1000
    confidence_level: float = 0.95
    metrics: List[str] = None

@dataclass
class ABTestResult:
    metric: str
    model_a_mean: float
    model_b_mean: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    winner: Optional[str]
    sample_size_a: int
    sample_size_b: int

class ABTestingFramework:
    """
    Manages A/B tests for model comparison
    """
    
    def __init__(self):
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, List[ABTestResult]] = {}
        self.test_data: Dict[str, Dict] = {}
        
    def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test"""
        test_id = f"{config.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_tests[test_id] = config
        self.test_data[test_id] = {
            'model_a_results': [],
            'model_b_results': [],
            'model_a_predictions': [],
            'model_b_predictions': []
        }
        
        return test_id
    
    async def route_request(self, test_id: str) -> str:
        """Route request to appropriate model based on test config"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
            
        config = self.active_tests[test_id]
        
        # Random routing based on traffic split
        if np.random.random() < config.traffic_split:
            return config.model_b
        else:
            return config.model_a
    
    def record_result(self, test_id: str, model: str, 
                     prediction: float, actual: float, 
                     metadata: Dict = None):
        """Record test result"""
        if test_id not in self.test_data:
            return
            
        data = self.test_data[test_id]
        
        result = {
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual),
            'squared_error': (prediction - actual) ** 2,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        if model == self.active_tests[test_id].model_a:
            data['model_a_results'].append(result)
            data['model_a_predictions'].append(prediction)
        else:
            data['model_b_results'].append(result)
            data['model_b_predictions'].append(prediction)
    
    def analyze_test(self, test_id: str) -> List[ABTestResult]:
        """Analyze A/B test results"""
        if test_id not in self.test_data:
            raise ValueError(f"Test {test_id} not found")
            
        config = self.active_tests[test_id]
        data = self.test_data[test_id]
        
        results = []
        
        # Check if we have enough samples
        n_a = len(data['model_a_results'])
        n_b = len(data['model_b_results'])
        
        if n_a < config.min_samples or n_b < config.min_samples:
            return []  # Not enough data yet
        
        # Analyze each metric
        metrics = config.metrics or ['error', 'squared_error']
        
        for metric in metrics:
            # Extract metric values
            values_a = [r[metric] for r in data['model_a_results']]
            values_b = [r[metric] for r in data['model_b_results']]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            # Calculate means
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            
            # Calculate confidence interval
            pooled_std = np.sqrt(
                ((n_a - 1) * np.var(values_a) + (n_b - 1) * np.var(values_b)) 
                / (n_a + n_b - 2)
            )
            
            se = pooled_std * np.sqrt(1/n_a + 1/n_b)
            t_critical = stats.t.ppf((1 + config.confidence_level) / 2, n_a + n_b - 2)
            
            ci_lower = (mean_b - mean_a) - t_critical * se
            ci_upper = (mean_b - mean_a) + t_critical * se
            
            # Determine significance and winner
            is_significant = p_value < (1 - config.confidence_level)
            winner = None
            
            if is_significant:
                # For error metrics, lower is better
                if metric in ['error', 'squared_error', 'loss']:
                    winner = config.model_b if mean_b < mean_a else config.model_a
                else:
                    winner = config.model_b if mean_b > mean_a else config.model_a
            
            result = ABTestResult(
                metric=metric,
                model_a_mean=mean_a,
                model_b_mean=mean_b,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                is_significant=is_significant,
                winner=winner,
                sample_size_a=n_a,
                sample_size_b=n_b
            )
            
            results.append(result)
        
        self.test_results[test_id] = results
        return results
    
    def get_recommendation(self, test_id: str) -> Dict[str, Any]:
        """Get recommendation based on test results"""
        results = self.test_results.get(test_id, [])
        
        if not results:
            return {
                'status': 'insufficient_data',
                'recommendation': 'Continue collecting data'
            }
        
        # Count wins for each model
        model_a_wins = sum(1 for r in results if r.winner == self.active_tests[test_id].model_a)
        model_b_wins = sum(1 for r in results if r.winner == self.active_tests[test_id].model_b)
        
        # Calculate average improvement
        improvements = []
        for r in results:
            if r.is_significant:
                pct_change = ((r.model_b_mean - r.model_a_mean) / r.model_a_mean) * 100
                improvements.append(pct_change)
        
        avg_improvement = np.mean(improvements) if improvements else 0
        
        # Make recommendation
        if model_b_wins > model_a_wins:
            recommendation = {
                'status': 'model_b_wins',
                'recommendation': f'Deploy {self.active_tests[test_id].model_b}',
                'confidence': 'high' if all(r.is_significant for r in results) else 'medium',
                'avg_improvement': avg_improvement
            }
        elif model_a_wins > model_b_wins:
            recommendation = {
                'status': 'model_a_wins',
                'recommendation': f'Keep {self.active_tests[test_id].model_a}',
                'confidence': 'high' if all(r.is_significant for r in results) else 'medium',
                'avg_improvement': 0
            }
        else:
            recommendation = {
                'status': 'no_clear_winner',
                'recommendation': 'No significant difference detected',
                'confidence': 'low',
                'avg_improvement': 0
            }
        
        return recommendation
    
    def gradual_rollout(self, test_id: str, target_split: float, 
                       duration_hours: int = 24) -> asyncio.Task:
        """Gradually increase traffic to treatment model"""
        async def _rollout():
            config = self.active_tests[test_id]
            initial_split = config.traffic_split
            steps = 10
            
            for i in range(steps + 1):
                # Calculate current split
                progress = i / steps
                current_split = initial_split + (target_split - initial_split) * progress
                
                # Update config
                config.traffic_split = current_split
                
                # Wait before next step
                if i < steps:
                    await asyncio.sleep(duration_hours * 3600 / steps)
        
        return asyncio.create_task(_rollout())
```

### 3. Feature Store

**File: `core/feature_store.py`**
```python
"""
Feature Store
=============
Centralized feature management and serving
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import asyncio
from dataclasses import dataclass

@dataclass
class FeatureDefinition:
    name: str
    description: str
    data_type: str
    source: str
    computation: str  # SQL or Python code
    refresh_frequency: timedelta
    tags: List[str]
    owner: str
    created_at: datetime
    version: int

@dataclass
class FeatureSet:
    name: str
    features: List[str]
    entity_key: str  # e.g., 'symbol', 'user_id'
    description: str

class FeatureStore:
    """
    Centralized feature store for ML
    """
    
    def __init__(self, backend='postgres'):
        self.backend = backend
        self.features: Dict[str, FeatureDefinition] = {}
        self.feature_sets: Dict[str, FeatureSet] = {}
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self._refresh_tasks: Dict[str, asyncio.Task] = {}
        
    def register_feature(self, feature: FeatureDefinition):
        """Register a new feature"""
        feature_id = f"{feature.name}_v{feature.version}"
        self.features[feature_id] = feature
        
        # Start refresh task if needed
        if feature.refresh_frequency:
            self._start_refresh_task(feature_id)
    
    def create_feature_set(self, name: str, features: List[str], 
                          entity_key: str, description: str = ""):
        """Create a feature set"""
        self.feature_sets[name] = FeatureSet(
            name=name,
            features=features,
            entity_key=entity_key,
            description=description
        )
    
    async def get_features(self, 
                          feature_set: str, 
                          entities: List[str],
                          timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """Get features for entities"""
        if feature_set not in self.feature_sets:
            raise ValueError(f"Feature set {feature_set} not found")
            
        fs = self.feature_sets[feature_set]
        
        # Collect all features
        feature_dfs = []
        
        for feature_name in fs.features:
            # Get latest version
            feature_id = self._get_latest_feature_id(feature_name)
            
            if feature_id in self.feature_cache:
                df = self.feature_cache[feature_id]
            else:
                df = await self._compute_feature(feature_id, entities, timestamp)
                
            feature_dfs.append(df)
        
        # Join all features
        result = feature_dfs[0]
        for df in feature_dfs[1:]:
            result = result.merge(df, on=fs.entity_key, how='left')
            
        # Filter to requested entities
        result = result[result[fs.entity_key].isin(entities)]
        
        return result
    
    async def get_training_data(self,
                               feature_set: str,
                               start_date: datetime,
                               end_date: datetime,
                               label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Get historical training data"""
        # Get all entities for time period
        entities = await self._get_entities_for_period(start_date, end_date)
        
        # Get features
        features_df = await self.get_features(feature_set, entities, end_date)
        
        # Get labels
        labels = await self._get_labels(entities, label_column, start_date, end_date)
        
        # Align features and labels
        features_df = features_df.merge(labels, on='entity_id', how='inner')
        
        X = features_df.drop(columns=[label_column])
        y = features_df[label_column]
        
        return X, y
    
    def compute_feature_importance(self, 
                                 feature_set: str,
                                 model: Any) -> Dict[str, float]:
        """Compute feature importance from model"""
        fs = self.feature_sets[feature_set]
        
        # Get importance scores
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return {}
        
        # Map to feature names
        importance_dict = {}
        for i, feature in enumerate(fs.features):
            importance_dict[feature] = float(importances[i])
            
        # Sort by importance
        return dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
    
    async def _compute_feature(self, 
                             feature_id: str,
                             entities: List[str],
                             timestamp: Optional[datetime]) -> pd.DataFrame:
        """Compute feature values"""
        feature = self.features[feature_id]
        
        if feature.source == 'sql':
            # Execute SQL query
            query = feature.computation.format(
                entities=','.join(f"'{e}'" for e in entities),
                timestamp=timestamp or datetime.now()
            )
            df = await self._execute_query(query)
            
        elif feature.source == 'python':
            # Execute Python function
            func = self._compile_feature_function(feature.computation)
            df = await func(entities, timestamp)
            
        else:
            raise ValueError(f"Unknown source: {feature.source}")
            
        # Cache result
        self.feature_cache[feature_id] = df
        
        return df
    
    def _compile_feature_function(self, code: str):
        """Compile feature computation code"""
        # Create safe execution environment
        safe_globals = {
            'pd': pd,
            'np': np,
            'datetime': datetime
        }
        
        # Compile code
        exec(code, safe_globals)
        
        # Return the compute_feature function
        return safe_globals.get('compute_feature')
    
    def _get_latest_feature_id(self, feature_name: str) -> str:
        """Get latest version of a feature"""
        matching = [
            fid for fid in self.features.keys() 
            if fid.startswith(f"{feature_name}_v")
        ]
        
        if not matching:
            raise ValueError(f"Feature {feature_name} not found")
            
        # Sort by version number
        return sorted(matching, key=lambda x: int(x.split('_v')[1]))[-1]
    
    def _start_refresh_task(self, feature_id: str):
        """Start background refresh task"""
        async def refresh_loop():
            feature = self.features[feature_id]
            
            while True:
                try:
                    # Recompute feature for all entities
                    entities = await self._get_all_entities()
                    await self._compute_feature(feature_id, entities, None)
                    
                except Exception as e:
                    print(f"Error refreshing {feature_id}: {e}")
                    
                await asyncio.sleep(feature.refresh_frequency.total_seconds())
        
        self._refresh_tasks[feature_id] = asyncio.create_task(refresh_loop())
    
    async def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query"""
        # Implementation depends on backend
        # This is a placeholder
        return pd.DataFrame()
    
    async def _get_entities_for_period(self, 
                                     start: datetime, 
                                     end: datetime) -> List[str]:
        """Get all entities for time period"""
        # Placeholder implementation
        return ['AAPL', 'GOOGL', 'MSFT']
    
    async def _get_all_entities(self) -> List[str]:
        """Get all entities"""
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    async def _get_labels(self, 
                        entities: List[str],
                        label_column: str,
                        start: datetime,
                        end: datetime) -> pd.DataFrame:
        """Get label data"""
        # Placeholder - would query from database
        return pd.DataFrame({
            'entity_id': entities,
            label_column: np.random.randn(len(entities))
        })

# Example feature definitions
example_features = [
    FeatureDefinition(
        name='price_sma_20',
        description='20-day simple moving average of price',
        data_type='float',
        source='sql',
        computation='''
        SELECT 
            symbol as entity_id,
            AVG(close) OVER (
                PARTITION BY symbol 
                ORDER BY date 
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) as price_sma_20
        FROM prices
        WHERE symbol IN ({entities})
        AND date <= '{timestamp}'
        ''',
        refresh_frequency=timedelta(hours=1),
        tags=['technical', 'price'],
        owner='ml_team',
        created_at=datetime.now(),
        version=1
    ),
    
    FeatureDefinition(
        name='volume_zscore',
        description='Z-score of volume relative to 30-day average',
        data_type='float',
        source='python',
        computation='''
async def compute_feature(entities, timestamp):
    # Get volume data
    volumes = await get_volume_data(entities, timestamp, lookback=30)
    
    # Compute z-scores
    result = []
    for entity in entities:
        entity_volumes = volumes[volumes.symbol == entity]['volume'].values
        if len(entity_volumes) >= 30:
            mean = entity_volumes[:-1].mean()
            std = entity_volumes[:-1].std()
            current = entity_volumes[-1]
            zscore = (current - mean) / std if std > 0 else 0
        else:
            zscore = 0
            
        result.append({
            'entity_id': entity,
            'volume_zscore': zscore
        })
    
    return pd.DataFrame(result)
        ''',
        refresh_frequency=timedelta(minutes=15),
        tags=['technical', 'volume'],
        owner='ml_team',
        created_at=datetime.now(),
        version=1
    )
]
```

### 4. Model Registry

**File: `core/model_registry.py`**
```python
"""
Model Registry
==============
Central registry for all models with metadata
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from enum import Enum

class ModelStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

@dataclass
class ModelMetadata:
    model_id: str
    model_name: str
    version: str
    stage: ModelStage
    description: str
    algorithm: str
    framework: str  # sklearn, tensorflow, pytorch, etc.
    created_by: str
    created_at: datetime
    updated_at: datetime
    
    # Training info
    training_dataset: str
    training_params: Dict[str, Any]
    training_metrics: Dict[str, float]
    
    # Model info
    input_schema: Dict[str, str]  # feature_name -> type
    output_schema: Dict[str, str]
    model_size_mb: float
    
    # Deployment info
    serving_endpoint: Optional[str] = None
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage
    parent_model: Optional[str] = None
    derived_models: List[str] = field(default_factory=list)
    
    # Tags and properties
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

class ModelRegistry:
    """
    Central registry for model management
    """
    
    def __init__(self, backend='postgres'):
        self.backend = backend
        self.models: Dict[str, ModelMetadata] = {}
        self.model_versions: Dict[str, List[str]] = {}  # model_name -> versions
        
    def register_model(self, metadata: ModelMetadata) -> str:
        """Register a new model"""
        # Generate unique ID
        model_key = f"{metadata.model_name}:{metadata.version}"
        
        # Store metadata
        self.models[model_key] = metadata
        
        # Update version tracking
        if metadata.model_name not in self.model_versions:
            self.model_versions[metadata.model_name] = []
        self.model_versions[metadata.model_name].append(metadata.version)
        
        # Update timestamps
        metadata.created_at = datetime.now()
        metadata.updated_at = datetime.now()
        
        return model_key
    
    def get_model(self, model_name: str, version: str = None) -> Optional[ModelMetadata]:
        """Get model by name and version"""
        if version:
            model_key = f"{model_name}:{version}"
            return self.models.get(model_key)
        else:
            # Get latest version
            if model_name not in self.model_versions:
                return None
            latest_version = sorted(self.model_versions[model_name])[-1]
            model_key = f"{model_name}:{latest_version}"
            return self.models.get(model_key)
    
    def promote_model(self, model_name: str, version: str, 
                     target_stage: ModelStage) -> bool:
        """Promote model to new stage"""
        model = self.get_model(model_name, version)
        if not model:
            return False
            
        # Demote current production model if promoting to production
        if target_stage == ModelStage.PRODUCTION:
            for key, m in self.models.items():
                if (m.model_name == model_name and 
                    m.stage == ModelStage.PRODUCTION):
                    m.stage = ModelStage.STAGING
                    m.updated_at = datetime.now()
        
        # Promote model
        model.stage = target_stage
        model.updated_at = datetime.now()
        
        return True
    
    def get_models_by_stage(self, stage: ModelStage) -> List[ModelMetadata]:
        """Get all models in a specific stage"""
        return [
            model for model in self.models.values()
            if model.stage == stage
        ]
    
    def search_models(self, 
                     tags: List[str] = None,
                     algorithm: str = None,
                     min_metric: Dict[str, float] = None) -> List[ModelMetadata]:
        """Search models by criteria"""
        results = list(self.models.values())
        
        # Filter by tags
        if tags:
            results = [
                m for m in results
                if any(tag in m.tags for tag in tags)
            ]
        
        # Filter by algorithm
        if algorithm:
            results = [m for m in results if m.algorithm == algorithm]
        
        # Filter by minimum metrics
        if min_metric:
            for metric, min_value in min_metric.items():
                results = [
                    m for m in results
                    if m.training_metrics.get(metric, 0) >= min_value
                ]
        
        return results
    
    def get_model_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get model lineage graph"""
        model = self.get_model(model_name, version)
        if not model:
            return {}
            
        lineage = {
            'model': f"{model_name}:{version}",
            'parent': model.parent_model,
            'children': model.derived_models,
            'ancestors': [],
            'descendants': []
        }
        
        # Trace ancestors
        current = model.parent_model
        while current:
            lineage['ancestors'].append(current)
            parent_model = self.models.get(current)
            current = parent_model.parent_model if parent_model else None
        
        # Trace descendants
        to_check = model.derived_models.copy()
        while to_check:
            child_key = to_check.pop(0)
            lineage['descendants'].append(child_key)
            child_model = self.models.get(child_key)
            if child_model:
                to_check.extend(child_model.derived_models)
        
        return lineage
    
    def compare_models(self, 
                      model1: str, 
                      model2: str) -> Dict[str, Any]:
        """Compare two models"""
        m1 = self.models.get(model1)
        m2 = self.models.get(model2)
        
        if not m1 or not m2:
            return {}
        
        comparison = {
            'models': [model1, model2],
            'metrics_comparison': {},
            'params_diff': {},
            'features_diff': {
                'only_in_model1': [],
                'only_in_model2': [],
                'common': []
            }
        }
        
        # Compare metrics
        all_metrics = set(m1.training_metrics.keys()) | set(m2.training_metrics.keys())
        for metric in all_metrics:
            val1 = m1.training_metrics.get(metric)
            val2 = m2.training_metrics.get(metric)
            if val1 is not None and val2 is not None:
                comparison['metrics_comparison'][metric] = {
                    'model1': val1,
                    'model2': val2,
                    'diff': val2 - val1,
                    'pct_change': ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }
        
        # Compare parameters
        all_params = set(m1.training_params.keys()) | set(m2.training_params.keys())
        for param in all_params:
            val1 = m1.training_params.get(param)
            val2 = m2.training_params.get(param)
            if val1 != val2:
                comparison['params_diff'][param] = {
                    'model1': val1,
                    'model2': val2
                }
        
        # Compare features
        features1 = set(m1.input_schema.keys())
        features2 = set(m2.input_schema.keys())
        comparison['features_diff']['only_in_model1'] = list(features1 - features2)
        comparison['features_diff']['only_in_model2'] = list(features2 - features1)
        comparison['features_diff']['common'] = list(features1 & features2)
        
        return comparison
    
    def export_registry(self) -> str:
        """Export registry as JSON"""
        export_data = {
            'models': {},
            'export_timestamp': datetime.now().isoformat()
        }
        
        for key, model in self.models.items():
            export_data['models'][key] = {
                'model_id': model.model_id,
                'model_name': model.model_name,
                'version': model.version,
                'stage': model.stage.value,
                'description': model.description,
                'algorithm': model.algorithm,
                'framework': model.framework,
                'created_by': model.created_by,
                'created_at': model.created_at.isoformat(),
                'updated_at': model.updated_at.isoformat(),
                'training_metrics': model.training_metrics,
                'tags': model.tags
            }
        
        return json.dumps(export_data, indent=2)
```

### 5. Automated Retraining Pipeline

**File: `core/auto_retraining.py`**
```python
"""
Automated Model Retraining Pipeline
====================================
Monitors model performance and triggers retraining
"""

import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

@dataclass
class RetrainingConfig:
    model_name: str
    check_frequency: timedelta
    
    # Trigger conditions
    performance_threshold: float  # Retrain if performance drops below
    data_drift_threshold: float  # Retrain if data drift exceeds
    time_threshold: timedelta  # Retrain after this time regardless
    
    # Retraining settings
    training_config: Dict[str, Any]
    validation_split: float = 0.2
    early_stopping: bool = True
    
    # Notification settings
    notify_on_trigger: bool = True
    notify_on_completion: bool = True
    auto_deploy: bool = False

class AutoRetrainingPipeline:
    """
    Automated retraining based on performance monitoring
    """
    
    def __init__(self):
        self.configs: Dict[str, RetrainingConfig] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.retraining_history: Dict[str, List[Dict]] = {}
        
    def register_model(self, config: RetrainingConfig):
        """Register model for automated retraining"""
        self.configs[config.model_name] = config
        
        # Start monitoring
        task = asyncio.create_task(
            self._monitor_model(config.model_name)
        )
        self.monitoring_tasks[config.model_name] = task
    
    async def _monitor_model(self, model_name: str):
        """Monitor model and trigger retraining when needed"""
        config = self.configs[model_name]
        
        while model_name in self.configs:
            try:
                # Check if retraining needed
                should_retrain, reason = await self._check_retraining_conditions(
                    model_name
                )
                
                if should_retrain:
                    await self._trigger_retraining(model_name, reason)
                    
            except Exception as e:
                print(f"Error monitoring {model_name}: {e}")
                
            # Wait before next check
            await asyncio.sleep(config.check_frequency.total_seconds())
    
    async def _check_retraining_conditions(self, 
                                         model_name: str) -> Tuple[bool, str]:
        """Check if model needs retraining"""
        config = self.configs[model_name]
        
        # Check performance
        current_performance = await self._get_model_performance(model_name)
        if current_performance < config.performance_threshold:
            return True, f"Performance dropped to {current_performance:.3f}"
        
        # Check data drift
        drift_score = await self._calculate_data_drift(model_name)
        if drift_score > config.data_drift_threshold:
            return True, f"Data drift score: {drift_score:.3f}"
        
        # Check time since last training
        last_training = await self._get_last_training_time(model_name)
        if datetime.now() - last_training > config.time_threshold:
            return True, "Time threshold exceeded"
        
        return False, ""
    
    async def _trigger_retraining(self, model_name: str, reason: str):
        """Trigger model retraining"""
        config = self.configs[model_name]
        
        print(f"Triggering retraining for {model_name}: {reason}")
        
        # Record trigger
        trigger_record = {
            'timestamp': datetime.now(),
            'reason': reason,
            'status': 'triggered'
        }
        
        if model_name not in self.retraining_history:
            self.retraining_history[model_name] = []
        self.retraining_history[model_name].append(trigger_record)
        
        # Notify if configured
        if config.notify_on_trigger:
            await self._send_notification(
                f"Retraining triggered for {model_name}: {reason}"
            )
        
        # Start retraining
        try:
            new_model = await self._retrain_model(model_name)
            
            # Validate new model
            validation_passed = await self._validate_model(new_model, model_name)
            
            if validation_passed:
                # Deploy if configured
                if config.auto_deploy:
                    await self._deploy_model(new_model, model_name)
                    trigger_record['status'] = 'deployed'
                else:
                    trigger_record['status'] = 'validated'
                    
                # Notify completion
                if config.notify_on_completion:
                    await self._send_notification(
                        f"Retraining completed for {model_name}"
                    )
            else:
                trigger_record['status'] = 'validation_failed'
                await self._send_notification(
                    f"Retraining failed validation for {model_name}"
                )
                
        except Exception as e:
            trigger_record['status'] = 'failed'
            trigger_record['error'] = str(e)
            await self._send_notification(
                f"Retraining failed for {model_name}: {e}"
            )
    
    async def _retrain_model(self, model_name: str) -> Any:
        """Retrain the model"""
        config = self.configs[model_name]
        
        # Get latest training data
        X_train, y_train = await self._get_training_data(model_name)
        
        # Split validation set
        split_idx = int(len(X_train) * (1 - config.validation_split))
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]
        X_train, y_train = X_train[:split_idx], y_train[:split_idx]
        
        # Load model architecture
        from mltrainer_models import get_ml_model_manager
        model_manager = get_ml_model_manager()
        
        # Train model
        model = model_manager.create_model(
            model_name, 
            **config.training_config
        )
        
        # Training with early stopping
        if config.early_stopping:
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                early_stopping=True
            )
        else:
            model.fit(X_train, y_train)
        
        return model
    
    async def _validate_model(self, model: Any, model_name: str) -> bool:
        """Validate retrained model"""
        config = self.configs[model_name]
        
        # Get test data
        X_test, y_test = await self._get_test_data(model_name)
        
        # Evaluate model
        predictions = model.predict(X_test)
        performance = self._calculate_performance(predictions, y_test)
        
        # Check if performance meets threshold
        return performance >= config.performance_threshold
    
    async def _deploy_model(self, model: Any, model_name: str):
        """Deploy the retrained model"""
        # Save model
        from core.model_versioning import ModelVersionControl
        version_control = ModelVersionControl()
        
        version = version_control.create_version(
            model_id=model_name,
            model_object=model,
            training_config=self.configs[model_name].training_config,
            performance_metrics={'auto_retrained': True},
            data_info={'timestamp': datetime.now()},
            created_by='auto_retraining'
        )
        
        # Update production pointer
        from core.model_registry import ModelRegistry, ModelStage
        registry = ModelRegistry()
        registry.promote_model(model_name, version, ModelStage.PRODUCTION)
    
    async def _get_model_performance(self, model_name: str) -> float:
        """Get current model performance"""
        # Placeholder - would calculate from recent predictions
        return np.random.uniform(0.7, 0.95)
    
    async def _calculate_data_drift(self, model_name: str) -> float:
        """Calculate data drift score"""
        # Placeholder - would use statistical tests
        return np.random.uniform(0, 0.5)
    
    async def _get_last_training_time(self, model_name: str) -> datetime:
        """Get last training timestamp"""
        # Placeholder
        return datetime.now() - timedelta(days=np.random.randint(1, 30))
    
    async def _get_training_data(self, model_name: str) -> Tuple[Any, Any]:
        """Get latest training data"""
        # Placeholder
        n_samples = 10000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        return X, y
    
    async def _get_test_data(self, model_name: str) -> Tuple[Any, Any]:
        """Get test data for validation"""
        # Placeholder
        n_samples = 2000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        return X, y
    
    def _calculate_performance(self, predictions: Any, actuals: Any) -> float:
        """Calculate model performance metric"""
        # Placeholder - would use appropriate metric
        return np.random.uniform(0.7, 0.95)
    
    async def _send_notification(self, message: str):
        """Send notification"""
        print(f"[NOTIFICATION] {message}")
        # In production, would send to Slack/email/etc
```

## Implementation Priority

1. **Week 1**: Model versioning system
2. **Week 2**: Feature store
3. **Week 3**: Model registry
4. **Week 4**: A/B testing framework
5. **Week 5**: Automated retraining

## Success Metrics

- All models versioned with full lineage tracking
- Feature store serving 100% of model features
- A/B testing on all model deployments
- Automated retraining reducing manual intervention by 90%
- Model registry providing full model governance