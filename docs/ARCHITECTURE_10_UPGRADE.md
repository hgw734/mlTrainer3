# Architecture Excellence: 9/10 → 10/10 Upgrade Plan

## Current State (9/10)
✅ Sophisticated modular design with proper separation of concerns
✅ Multiple abstraction layers: engines, bridges, executors, managers
✅ Async patterns throughout (async_execution_engine.py)
✅ Plugin architecture for models (140+ configurations)
✅ Event-driven components with autonomous loops

## Missing for 10/10

### 1. Architecture Decision Records (ADRs)

Create formal ADRs documenting key decisions:

**File: `docs/adr/001-plugin-architecture.md`**
```markdown
# ADR-001: Plugin Architecture for ML Models

## Status
Accepted

## Context
We need to support 140+ different ML models with varying interfaces.

## Decision
Implement a plugin-based architecture with:
- Model registry pattern
- Unified interface through adapters
- Dynamic loading based on configuration

## Consequences
- Easy to add new models
- Consistent interface for all models
- Some performance overhead for adaptation layer
```

**File: `docs/adr/002-async-execution.md`**
```markdown
# ADR-002: Async Execution Engine

## Status
Accepted

## Context
ML training and financial calculations are CPU-intensive and need parallelization.

## Decision
Implement dual-pool async execution:
- ThreadPoolExecutor for I/O-bound tasks
- ProcessPoolExecutor for CPU-bound tasks
- Asyncio orchestration layer

## Consequences
- Maximum CPU utilization
- Complex error handling
- Need careful resource management
```

### 2. Dependency Injection Framework

**File: `core/dependency_injection.py`**
```python
"""
Dependency Injection Container
==============================
Manages component lifecycle and dependencies
"""

from typing import Dict, Any, Type, Callable
import inspect
from functools import wraps

class DIContainer:
    """
    Dependency injection container with lifecycle management
    """
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._scopes = ['singleton', 'transient', 'scoped']
        
    def register_singleton(self, interface: Type, implementation: Type = None):
        """Register a singleton service"""
        def decorator(cls):
            self._services[interface] = {
                'implementation': cls,
                'scope': 'singleton'
            }
            return cls
            
        if implementation:
            self._services[interface] = {
                'implementation': implementation,
                'scope': 'singleton'
            }
            return implementation
        return decorator
    
    def register_transient(self, interface: Type, implementation: Type = None):
        """Register a transient service (new instance each time)"""
        def decorator(cls):
            self._services[interface] = {
                'implementation': cls,
                'scope': 'transient'
            }
            return cls
            
        if implementation:
            self._services[interface] = {
                'implementation': implementation,
                'scope': 'transient'
            }
            return implementation
        return decorator
    
    def register_factory(self, interface: Type, factory: Callable):
        """Register a factory function"""
        self._factories[interface] = factory
    
    def resolve(self, interface: Type) -> Any:
        """Resolve a service by interface"""
        if interface in self._factories:
            return self._factories[interface]()
            
        if interface not in self._services:
            raise ValueError(f"Service {interface} not registered")
            
        service_info = self._services[interface]
        
        if service_info['scope'] == 'singleton':
            if interface not in self._singletons:
                self._singletons[interface] = self._create_instance(
                    service_info['implementation']
                )
            return self._singletons[interface]
        
        elif service_info['scope'] == 'transient':
            return self._create_instance(service_info['implementation'])
    
    def _create_instance(self, cls: Type) -> Any:
        """Create instance with dependency injection"""
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            if param.annotation != param.empty:
                # Try to resolve the dependency
                try:
                    kwargs[param_name] = self.resolve(param.annotation)
                except ValueError:
                    if param.default == param.empty:
                        raise
                        
        return cls(**kwargs)
    
    def inject(self, func: Callable) -> Callable:
        """Decorator for dependency injection"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            
            for param_name, param in sig.parameters.items():
                if param_name not in kwargs and param.annotation != param.empty:
                    try:
                        kwargs[param_name] = self.resolve(param.annotation)
                    except ValueError:
                        pass
                        
            return func(*args, **kwargs)
        return wrapper

# Global container
container = DIContainer()

# Register core services
from self_learning_engine import SelfLearningEngine
from mltrainer_financial_models import MLTrainerFinancialModels
from core.async_execution_engine import AsyncExecutionEngine

container.register_singleton(SelfLearningEngine)
container.register_singleton(MLTrainerFinancialModels)
container.register_singleton(AsyncExecutionEngine)
```

### 3. Service Discovery & Registry

**File: `core/service_registry.py`**
```python
"""
Service Registry Pattern
========================
Dynamic service discovery and health checking
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import aiohttp

@dataclass
class ServiceInstance:
    name: str
    host: str
    port: int
    version: str
    health_endpoint: str = "/health"
    metadata: Dict = None
    last_health_check: Optional[datetime] = None
    healthy: bool = True

class ServiceRegistry:
    """
    Service registry with health checking
    """
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self._health_check_interval = 30  # seconds
        self._health_check_task = None
        
    async def register(self, service: ServiceInstance):
        """Register a service instance"""
        if service.name not in self.services:
            self.services[service.name] = []
            
        self.services[service.name].append(service)
        
        # Start health checking if not running
        if not self._health_check_task:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )
    
    async def discover(self, service_name: str) -> Optional[ServiceInstance]:
        """Discover a healthy service instance"""
        if service_name not in self.services:
            return None
            
        healthy_instances = [
            s for s in self.services[service_name] 
            if s.healthy
        ]
        
        if not healthy_instances:
            return None
            
        # Simple round-robin selection
        return healthy_instances[0]
    
    async def _health_check_loop(self):
        """Continuously check service health"""
        while True:
            for service_list in self.services.values():
                for service in service_list:
                    await self._check_health(service)
                    
            await asyncio.sleep(self._health_check_interval)
    
    async def _check_health(self, service: ServiceInstance):
        """Check health of a single service"""
        url = f"http://{service.host}:{service.port}{service.health_endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    service.healthy = response.status == 200
                    service.last_health_check = datetime.now()
        except Exception:
            service.healthy = False
            service.last_health_check = datetime.now()
```

### 4. Architecture Diagrams

**File: `docs/architecture/system-overview.py`**
```python
"""
Generate architecture diagrams using Python
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

def create_system_architecture_diagram():
    """Create high-level system architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define components
    components = {
        'UI Layer': {
            'position': (2, 8),
            'size': (10, 1.5),
            'color': '#3498db',
            'services': ['Streamlit UI', 'API Gateway', 'WebSocket']
        },
        'API Layer': {
            'position': (2, 6),
            'size': (10, 1.5),
            'color': '#2ecc71',
            'services': ['FastAPI', 'GraphQL', 'REST']
        },
        'Business Logic': {
            'position': (1, 3.5),
            'size': (5, 2),
            'color': '#e74c3c',
            'services': ['ML Engine', 'Financial Models', 'Compliance']
        },
        'Async Engine': {
            'position': (7, 3.5),
            'size': (5, 2),
            'color': '#f39c12',
            'services': ['Thread Pool', 'Process Pool', 'Task Queue']
        },
        'Data Layer': {
            'position': (1, 1),
            'size': (5, 2),
            'color': '#9b59b6',
            'services': ['PostgreSQL', 'Redis', 'S3']
        },
        'External Services': {
            'position': (7, 1),
            'size': (5, 2),
            'color': '#1abc9c',
            'services': ['Polygon', 'FRED', 'Claude']
        }
    }
    
    # Draw components
    for name, comp in components.items():
        # Main box
        box = FancyBboxPatch(
            comp['position'], 
            comp['size'][0], 
            comp['size'][1],
            boxstyle="round,pad=0.1",
            facecolor=comp['color'],
            edgecolor='black',
            alpha=0.7
        )
        ax.add_patch(box)
        
        # Title
        ax.text(
            comp['position'][0] + comp['size'][0]/2,
            comp['position'][1] + comp['size'][1] - 0.3,
            name,
            fontsize=12,
            fontweight='bold',
            ha='center'
        )
        
        # Services
        service_text = ' | '.join(comp['services'])
        ax.text(
            comp['position'][0] + comp['size'][0]/2,
            comp['position'][1] + 0.3,
            service_text,
            fontsize=9,
            ha='center'
        )
    
    # Draw connections
    connections = [
        ('UI Layer', 'API Layer'),
        ('API Layer', 'Business Logic'),
        ('API Layer', 'Async Engine'),
        ('Business Logic', 'Data Layer'),
        ('Async Engine', 'External Services'),
        ('Business Logic', 'External Services')
    ]
    
    # Connection arrows
    for start, end in connections:
        start_comp = components[start]
        end_comp = components[end]
        
        start_x = start_comp['position'][0] + start_comp['size'][0]/2
        start_y = start_comp['position'][1]
        
        end_x = end_comp['position'][0] + end_comp['size'][0]/2
        end_y = end_comp['position'][1] + end_comp['size'][1]
        
        ax.arrow(
            start_x, start_y,
            end_x - start_x, end_y - start_y,
            head_width=0.2, 
            head_length=0.1,
            fc='gray', 
            ec='gray',
            alpha=0.5
        )
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('mlTrainer System Architecture', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/architecture/system-architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate diagram
create_system_architecture_diagram()
```

### 5. SOLID Principles Enhancement

**File: `core/solid_principles.py`**
```python
"""
SOLID Principles Implementation Examples
========================================
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Any

# Single Responsibility Principle
class ModelTrainer:
    """Only responsible for training models"""
    def train(self, model_id: str, data: Any):
        pass

class ModelValidator:
    """Only responsible for validating models"""
    def validate(self, model_id: str, data: Any):
        pass

class ModelPersister:
    """Only responsible for saving/loading models"""
    def save(self, model_id: str, model: Any):
        pass
    
    def load(self, model_id: str) -> Any:
        pass

# Open/Closed Principle
class ModelStrategy(ABC):
    """Base strategy - open for extension, closed for modification"""
    @abstractmethod
    def execute(self, data: Any) -> Any:
        pass

class RandomForestStrategy(ModelStrategy):
    def execute(self, data: Any) -> Any:
        return "RandomForest result"

class XGBoostStrategy(ModelStrategy):
    def execute(self, data: Any) -> Any:
        return "XGBoost result"

# Liskov Substitution Principle
class Model(ABC):
    @abstractmethod
    def predict(self, features: List[float]) -> float:
        pass

class LinearModel(Model):
    def predict(self, features: List[float]) -> float:
        # Can be substituted anywhere Model is used
        return sum(features) / len(features)

# Interface Segregation Principle
class Trainable(Protocol):
    def train(self, data: Any) -> None: ...

class Predictable(Protocol):
    def predict(self, features: Any) -> Any: ...

class Serializable(Protocol):
    def serialize(self) -> Dict[str, Any]: ...
    def deserialize(self, data: Dict[str, Any]) -> None: ...

# Client only depends on interfaces it uses
class TrainingPipeline:
    def run(self, model: Trainable, data: Any):
        model.train(data)

# Dependency Inversion Principle
class DataSource(ABC):
    @abstractmethod
    def fetch_data(self) -> Any:
        pass

class DatabaseSource(DataSource):
    def fetch_data(self) -> Any:
        return "DB data"

class APISource(DataSource):
    def fetch_data(self) -> Any:
        return "API data"

class MLPipeline:
    def __init__(self, data_source: DataSource):
        # Depends on abstraction, not concrete implementation
        self.data_source = data_source
    
    def run(self):
        data = self.data_source.fetch_data()
        # Process data
```

## Implementation Priority

1. **Week 1**: ADRs and documentation
2. **Week 2**: Dependency injection framework
3. **Week 3**: Service registry and discovery
4. **Week 4**: Architecture diagrams and SOLID refactoring

## Success Metrics

- All major architectural decisions documented in ADRs
- 100% of core services using dependency injection
- Service discovery enabled for all microservices
- Complete architecture diagrams generated
- SOLID principles applied throughout codebase