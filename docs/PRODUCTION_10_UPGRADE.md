# Production Readiness: 8.5/10 → 10/10 Upgrade Plan

## Current State (8.5/10)
✅ Complete observability: monitoring, logging, metrics export
✅ Rate limiting for external APIs
✅ Caching strategies at multiple levels
✅ Health checks and compliance engines
✅ Database abstraction with proper migrations

## Missing for 10/10

### 1. Chaos Engineering Tests

**File: `chaos/chaos_engine.py`**
```python
"""
Chaos Engineering Framework
===========================
Test system resilience by injecting failures
"""

import asyncio
import random
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import psutil
import os

class ChaosType(Enum):
    LATENCY = "latency"
    ERROR = "error"
    RESOURCE = "resource"
    NETWORK = "network"
    CRASH = "crash"
    DATA_CORRUPTION = "data_corruption"

@dataclass
class ChaosExperiment:
    name: str
    chaos_type: ChaosType
    target_service: str
    duration: timedelta
    intensity: float  # 0.0 to 1.0
    parameters: Dict[str, Any]
    
@dataclass
class ChaosResult:
    experiment: ChaosExperiment
    start_time: datetime
    end_time: datetime
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    incidents: List[Dict[str, Any]]
    recovery_time: Optional[timedelta]
    passed: bool

class ChaosEngine:
    """
    Inject controlled chaos to test resilience
    """
    
    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.results: List[ChaosResult] = []
        self.chaos_middleware: Dict[str, Callable] = {}
        self._original_functions: Dict[str, Callable] = {}
        
    def register_experiment(self, experiment: ChaosExperiment):
        """Register a chaos experiment"""
        self.experiments[experiment.name] = experiment
        
    async def run_experiment(self, experiment_name: str) -> ChaosResult:
        """Run a chaos experiment"""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found")
            
        print(f"Starting chaos experiment: {experiment.name}")
        
        # Collect baseline metrics
        metrics_before = await self._collect_metrics(experiment.target_service)
        
        # Start monitoring
        incidents = []
        monitor_task = asyncio.create_task(
            self._monitor_service(experiment.target_service, incidents)
        )
        
        # Inject chaos
        start_time = datetime.now()
        await self._inject_chaos(experiment)
        
        # Run for duration
        await asyncio.sleep(experiment.duration.total_seconds())
        
        # Remove chaos
        await self._remove_chaos(experiment)
        end_time = datetime.now()
        
        # Wait for recovery
        recovery_start = datetime.now()
        recovered = await self._wait_for_recovery(experiment.target_service)
        recovery_time = datetime.now() - recovery_start if recovered else None
        
        # Stop monitoring
        monitor_task.cancel()
        
        # Collect post-chaos metrics
        metrics_after = await self._collect_metrics(experiment.target_service)
        
        # Analyze results
        passed = self._analyze_results(
            metrics_before, 
            metrics_after, 
            incidents,
            recovery_time
        )
        
        result = ChaosResult(
            experiment=experiment,
            start_time=start_time,
            end_time=end_time,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            incidents=incidents,
            recovery_time=recovery_time,
            passed=passed
        )
        
        self.results.append(result)
        return result
        
    async def _inject_chaos(self, experiment: ChaosExperiment):
        """Inject chaos based on type"""
        if experiment.chaos_type == ChaosType.LATENCY:
            await self._inject_latency(experiment)
        elif experiment.chaos_type == ChaosType.ERROR:
            await self._inject_errors(experiment)
        elif experiment.chaos_type == ChaosType.RESOURCE:
            await self._inject_resource_pressure(experiment)
        elif experiment.chaos_type == ChaosType.NETWORK:
            await self._inject_network_issues(experiment)
        elif experiment.chaos_type == ChaosType.CRASH:
            await self._inject_crash(experiment)
        elif experiment.chaos_type == ChaosType.DATA_CORRUPTION:
            await self._inject_data_corruption(experiment)
            
    async def _inject_latency(self, experiment: ChaosExperiment):
        """Add latency to service calls"""
        target_module = experiment.parameters.get('module', 'backend.unified_api')
        target_function = experiment.parameters.get('function', 'predict')
        latency_ms = experiment.parameters.get('latency_ms', 1000)
        
        # Create wrapper that adds latency
        def latency_wrapper(original_func):
            async def wrapper(*args, **kwargs):
                if random.random() < experiment.intensity:
                    await asyncio.sleep(latency_ms / 1000.0)
                return await original_func(*args, **kwargs)
            return wrapper
        
        # Monkey patch the function
        module = __import__(target_module, fromlist=[target_function])
        original = getattr(module, target_function)
        self._original_functions[f"{target_module}.{target_function}"] = original
        setattr(module, target_function, latency_wrapper(original))
        
    async def _inject_errors(self, experiment: ChaosExperiment):
        """Make services return errors"""
        target_module = experiment.parameters.get('module')
        target_function = experiment.parameters.get('function')
        error_type = experiment.parameters.get('error_type', Exception)
        error_message = experiment.parameters.get('error_message', 'Chaos injection')
        
        def error_wrapper(original_func):
            async def wrapper(*args, **kwargs):
                if random.random() < experiment.intensity:
                    raise error_type(error_message)
                return await original_func(*args, **kwargs)
            return wrapper
        
        module = __import__(target_module, fromlist=[target_function])
        original = getattr(module, target_function)
        self._original_functions[f"{target_module}.{target_function}"] = original
        setattr(module, target_function, error_wrapper(original))
        
    async def _inject_resource_pressure(self, experiment: ChaosExperiment):
        """Consume CPU/memory resources"""
        resource_type = experiment.parameters.get('resource', 'cpu')
        
        if resource_type == 'cpu':
            # CPU pressure
            cores = psutil.cpu_count()
            tasks = []
            for _ in range(int(cores * experiment.intensity)):
                tasks.append(asyncio.create_task(self._cpu_stress()))
            
            # Run for duration
            await asyncio.sleep(experiment.duration.total_seconds())
            
            # Cancel tasks
            for task in tasks:
                task.cancel()
                
        elif resource_type == 'memory':
            # Memory pressure
            target_mb = experiment.parameters.get('memory_mb', 1000)
            data = []
            
            # Allocate memory
            for _ in range(int(target_mb * experiment.intensity)):
                data.append(bytearray(1024 * 1024))  # 1MB chunks
                
            # Hold for duration
            await asyncio.sleep(experiment.duration.total_seconds())
            
            # Release
            data.clear()
            
    async def _inject_network_issues(self, experiment: ChaosExperiment):
        """Simulate network problems"""
        issue_type = experiment.parameters.get('issue', 'packet_loss')
        
        if issue_type == 'packet_loss':
            # Simulate packet loss by making requests fail randomly
            pass
        elif issue_type == 'partition':
            # Simulate network partition
            pass
            
    async def _inject_crash(self, experiment: ChaosExperiment):
        """Crash a service"""
        service_name = experiment.target_service
        
        # In production, would actually restart service
        # For testing, we'll simulate by making it unresponsive
        print(f"Simulating crash of {service_name}")
        
    async def _inject_data_corruption(self, experiment: ChaosExperiment):
        """Corrupt data to test validation"""
        corruption_type = experiment.parameters.get('type', 'bit_flip')
        
        # Would modify data in cache/database
        print(f"Injecting {corruption_type} data corruption")
        
    async def _remove_chaos(self, experiment: ChaosExperiment):
        """Remove injected chaos"""
        # Restore original functions
        for key, original in self._original_functions.items():
            module_name, function_name = key.rsplit('.', 1)
            module = __import__(module_name, fromlist=[function_name])
            setattr(module, function_name, original)
            
        self._original_functions.clear()
        
    async def _cpu_stress(self):
        """CPU stress function"""
        while True:
            # Busy loop
            sum(i * i for i in range(10000))
            await asyncio.sleep(0.001)  # Yield briefly
            
    async def _collect_metrics(self, service: str) -> Dict[str, float]:
        """Collect service metrics"""
        # In production, would query Prometheus
        return {
            'latency_p99': random.uniform(10, 100),
            'error_rate': random.uniform(0, 0.05),
            'throughput': random.uniform(100, 1000),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent
        }
        
    async def _monitor_service(self, service: str, incidents: List[Dict]):
        """Monitor service during chaos"""
        while True:
            try:
                # Check service health
                health = await self._check_service_health(service)
                
                if not health['healthy']:
                    incidents.append({
                        'timestamp': datetime.now().isoformat(),
                        'service': service,
                        'issue': health['issue']
                    })
                    
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
                
    async def _check_service_health(self, service: str) -> Dict[str, Any]:
        """Check if service is healthy"""
        # Would make actual health check
        return {
            'healthy': random.random() > 0.1,
            'issue': 'timeout' if random.random() > 0.5 else 'error'
        }
        
    async def _wait_for_recovery(self, service: str, 
                               timeout: int = 300) -> bool:
        """Wait for service to recover"""
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < timeout:
            health = await self._check_service_health(service)
            if health['healthy']:
                return True
            await asyncio.sleep(1)
            
        return False
        
    def _analyze_results(self, 
                        metrics_before: Dict[str, float],
                        metrics_after: Dict[str, float],
                        incidents: List[Dict],
                        recovery_time: Optional[timedelta]) -> bool:
        """Analyze if experiment passed"""
        # Check if metrics degraded significantly
        latency_increase = (
            metrics_after['latency_p99'] / metrics_before['latency_p99']
        )
        error_increase = (
            metrics_after['error_rate'] - metrics_before['error_rate']
        )
        
        # Pass criteria
        passed = True
        
        if latency_increase > 2.0:  # Latency doubled
            passed = False
            
        if error_increase > 0.1:  # Error rate increased by 10%
            passed = False
            
        if recovery_time and recovery_time.total_seconds() > 300:  # 5 min
            passed = False
            
        if len(incidents) > 10:  # Too many incidents
            passed = False
            
        return passed
        
    def generate_report(self) -> str:
        """Generate chaos engineering report"""
        report = ["# Chaos Engineering Report\n"]
        report.append(f"Generated: {datetime.now().isoformat()}\n")
        
        # Summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        report.append(f"## Summary\n")
        report.append(f"- Total experiments: {total}\n")
        report.append(f"- Passed: {passed} ({passed/total*100:.1f}%)\n")
        report.append(f"- Failed: {total-passed}\n\n")
        
        # Details
        report.append("## Experiment Details\n")
        for result in self.results:
            report.append(f"\n### {result.experiment.name}\n")
            report.append(f"- Type: {result.experiment.chaos_type.value}\n")
            report.append(f"- Target: {result.experiment.target_service}\n")
            report.append(f"- Duration: {result.experiment.duration}\n")
            report.append(f"- Result: {'PASSED' if result.passed else 'FAILED'}\n")
            
            if result.recovery_time:
                report.append(f"- Recovery time: {result.recovery_time}\n")
                
            if result.incidents:
                report.append(f"- Incidents: {len(result.incidents)}\n")
                
        return ''.join(report)

# Example experiments
chaos_experiments = [
    ChaosExperiment(
        name="api_latency_test",
        chaos_type=ChaosType.LATENCY,
        target_service="mltrainer-api",
        duration=timedelta(minutes=5),
        intensity=0.5,
        parameters={
            'module': 'backend.unified_api',
            'function': 'predict',
            'latency_ms': 2000
        }
    ),
    
    ChaosExperiment(
        name="database_error_test",
        chaos_type=ChaosType.ERROR,
        target_service="postgres",
        duration=timedelta(minutes=3),
        intensity=0.3,
        parameters={
            'module': 'backend.database',
            'function': 'execute_query',
            'error_type': ConnectionError,
            'error_message': 'Database connection lost'
        }
    ),
    
    ChaosExperiment(
        name="high_cpu_test",
        chaos_type=ChaosType.RESOURCE,
        target_service="mltrainer-api",
        duration=timedelta(minutes=10),
        intensity=0.8,
        parameters={
            'resource': 'cpu'
        }
    )
]
```

### 2. Circuit Breakers

**File: `core/circuit_breaker.py`**
```python
"""
Circuit Breaker Pattern
========================
Prevent cascading failures
"""

from enum import Enum
from typing import Callable, Optional, Any, Dict
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(seconds=60)
    expected_exception: type = Exception
    success_threshold: int = 2

@dataclass
class CircuitStats:
    failures: int = 0
    successes: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._half_open_attempts = 0
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker"""
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker"""
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self._half_open_attempts = 0
            else:
                raise CircuitOpenError(
                    f"Circuit breaker {self.name} is OPEN"
                )
        
        # Try to execute function
        try:
            result = await self._execute(func, *args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
            
    async def _execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the function"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
            
    def _on_success(self):
        """Handle successful call"""
        self.stats.total_calls += 1
        self.stats.total_successes += 1
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.config.success_threshold:
                self._reset()
                
    def _on_failure(self):
        """Handle failed call"""
        self.stats.total_calls += 1
        self.stats.total_failures += 1
        self.stats.failures += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self._trip()
        elif (self.stats.consecutive_failures >= self.config.failure_threshold):
            self._trip()
            
    def _trip(self):
        """Trip the circuit breaker"""
        self.state = CircuitState.OPEN
        logger.warning(f"Circuit breaker {self.name} tripped to OPEN")
        
    def _reset(self):
        """Reset the circuit breaker"""
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        logger.info(f"Circuit breaker {self.name} reset to CLOSED")
        
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset"""
        return (
            self.stats.last_failure_time and
            datetime.now() - self.stats.last_failure_time > 
            self.config.recovery_timeout
        )
        
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'stats': {
                'total_calls': self.stats.total_calls,
                'total_failures': self.stats.total_failures,
                'total_successes': self.stats.total_successes,
                'consecutive_failures': self.stats.consecutive_failures,
                'failure_rate': (
                    self.stats.total_failures / self.stats.total_calls
                    if self.stats.total_calls > 0 else 0
                )
            }
        }

class CircuitOpenError(Exception):
    """Raised when circuit is open"""
    pass

class CircuitBreakerRegistry:
    """
    Registry for all circuit breakers
    """
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        
    def get_or_create(self, 
                     name: str, 
                     config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
        
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get state of all circuit breakers"""
        return {
            name: breaker.get_state() 
            for name, breaker in self.breakers.items()
        }
        
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker._reset()

# Global registry
circuit_registry = CircuitBreakerRegistry()

# Decorator helper
def circuit_breaker(name: str, **kwargs):
    """Decorator to apply circuit breaker"""
    config = CircuitBreakerConfig(**kwargs)
    breaker = circuit_registry.get_or_create(name, config)
    return breaker

# Usage examples
@circuit_breaker("external_api", failure_threshold=3, recovery_timeout=timedelta(seconds=30))
async def call_external_api(endpoint: str) -> Dict:
    """Example function with circuit breaker"""
    # Make API call
    pass

@circuit_breaker("database", failure_threshold=5, expected_exception=ConnectionError)
async def query_database(query: str) -> Any:
    """Example database query with circuit breaker"""
    # Execute query
    pass
```

### 3. Blue-Green Deployment

**File: `deployment/blue_green.py`**
```python
"""
Blue-Green Deployment System
============================
Zero-downtime deployments
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio
from enum import Enum

class Environment(Enum):
    BLUE = "blue"
    GREEN = "green"

@dataclass
class DeploymentConfig:
    service_name: str
    version: str
    environment: Environment
    health_check_url: str
    warmup_time: int = 60  # seconds
    canary_percentage: int = 0  # 0 for full switch, >0 for canary

class BlueGreenDeployment:
    """
    Manages blue-green deployments
    """
    
    def __init__(self):
        self.active_environment = Environment.BLUE
        self.environments: Dict[Environment, Dict] = {
            Environment.BLUE: {'version': 'v1.0.0', 'healthy': True},
            Environment.GREEN: {'version': None, 'healthy': False}
        }
        self.deployment_history: List[Dict] = []
        
    async def deploy(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy new version to inactive environment"""
        
        # Determine target environment
        target_env = (
            Environment.GREEN if self.active_environment == Environment.BLUE 
            else Environment.BLUE
        )
        
        deployment = {
            'id': f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'service': config.service_name,
            'version': config.version,
            'target_env': target_env,
            'started_at': datetime.now(),
            'status': 'in_progress'
        }
        
        try:
            # Step 1: Deploy to inactive environment
            await self._deploy_to_environment(config, target_env)
            
            # Step 2: Health check
            healthy = await self._health_check(config.health_check_url)
            if not healthy:
                raise Exception("Health check failed")
                
            # Step 3: Warm up
            await self._warmup(config.warmup_time)
            
            # Step 4: Switch traffic
            if config.canary_percentage > 0:
                await self._canary_deployment(
                    target_env, 
                    config.canary_percentage
                )
            else:
                await self._switch_traffic(target_env)
                
            # Step 5: Verify
            await asyncio.sleep(30)  # Monitor for 30 seconds
            
            final_healthy = await self._health_check(config.health_check_url)
            if not final_healthy:
                # Rollback
                await self._switch_traffic(self.active_environment)
                raise Exception("Post-deployment health check failed")
                
            # Success
            self.active_environment = target_env
            deployment['status'] = 'success'
            deployment['completed_at'] = datetime.now()
            
        except Exception as e:
            deployment['status'] = 'failed'
            deployment['error'] = str(e)
            deployment['completed_at'] = datetime.now()
            
        self.deployment_history.append(deployment)
        return deployment
        
    async def _deploy_to_environment(self, 
                                   config: DeploymentConfig, 
                                   env: Environment):
        """Deploy to specific environment"""
        print(f"Deploying {config.version} to {env.value}")
        
        # In production, would:
        # 1. Update kubernetes deployment
        # 2. Update docker containers
        # 3. Update load balancer config
        
        self.environments[env] = {
            'version': config.version,
            'healthy': False
        }
        
    async def _health_check(self, url: str, retries: int = 5) -> bool:
        """Check service health"""
        for i in range(retries):
            try:
                # In production, make actual HTTP request
                # Simulating here
                await asyncio.sleep(1)
                return True
            except Exception:
                if i < retries - 1:
                    await asyncio.sleep(5)
                else:
                    return False
        return False
        
    async def _warmup(self, duration: int):
        """Warm up the new deployment"""
        print(f"Warming up for {duration} seconds")
        
        # In production, would:
        # 1. Send synthetic traffic
        # 2. Pre-load caches
        # 3. Initialize connection pools
        
        await asyncio.sleep(duration)
        
    async def _switch_traffic(self, target_env: Environment):
        """Switch all traffic to target environment"""
        print(f"Switching traffic to {target_env.value}")
        
        # In production, update load balancer
        # For now, just update state
        self.active_environment = target_env
        
    async def _canary_deployment(self, 
                               target_env: Environment, 
                               percentage: int):
        """Gradual traffic shift"""
        print(f"Starting canary deployment: {percentage}% to {target_env.value}")
        
        steps = 5
        for i in range(1, steps + 1):
            current_percentage = (percentage / steps) * i
            print(f"Shifting {current_percentage}% traffic")
            
            # In production, update load balancer weights
            await asyncio.sleep(60)  # 1 minute between steps
            
            # Check health
            healthy = await self._health_check("")
            if not healthy:
                raise Exception(f"Canary failed at {current_percentage}%")
                
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        return {
            'active_environment': self.active_environment.value,
            'environments': self.environments,
            'last_deployment': (
                self.deployment_history[-1] 
                if self.deployment_history else None
            )
        }
        
    async def rollback(self) -> bool:
        """Rollback to previous environment"""
        previous_env = (
            Environment.BLUE if self.active_environment == Environment.GREEN
            else Environment.GREEN
        )
        
        if self.environments[previous_env]['healthy']:
            await self._switch_traffic(previous_env)
            return True
        return False

# Kubernetes integration
class K8sBlueGreenDeployer:
    """
    Kubernetes-specific blue-green deployment
    """
    
    def __init__(self, namespace: str = "mltrainer"):
        self.namespace = namespace
        
    async def deploy(self, config: DeploymentConfig):
        """Deploy using Kubernetes"""
        # Would use kubernetes-client library
        
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {config.service_name}-{config.environment.value}
  namespace: {self.namespace}
  labels:
    app: {config.service_name}
    version: {config.version}
    environment: {config.environment.value}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {config.service_name}
      environment: {config.environment.value}
  template:
    metadata:
      labels:
        app: {config.service_name}
        version: {config.version}
        environment: {config.environment.value}
    spec:
      containers:
      - name: {config.service_name}
        image: mltrainer/{config.service_name}:{config.version}
        ports:
        - containerPort: 8000
"""
        
        # Apply deployment
        # kubectl apply -f deployment.yaml
        
    async def switch_service(self, 
                           service_name: str, 
                           target_env: Environment):
        """Update service selector"""
        service_yaml = f"""
apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  namespace: {self.namespace}
spec:
  selector:
    app: {service_name}
    environment: {target_env.value}
  ports:
  - port: 80
    targetPort: 8000
"""
        
        # Apply service update
        # kubectl apply -f service.yaml
```

### 4. Distributed Tracing

**File: `core/distributed_tracing.py`**
```python
"""
Distributed Tracing System
==========================
Track requests across services
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextvars import ContextVar
import json

# Context variable for trace context
trace_context: ContextVar[Optional['TraceContext']] = ContextVar('trace_context', default=None)

@dataclass
class TraceContext:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None

class DistributedTracer:
    """
    Distributed tracing implementation
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.spans: List[Span] = []
        self.span_exporters: List[Callable] = []
        
    def start_span(self, 
                  operation_name: str,
                  parent_context: Optional[TraceContext] = None) -> 'SpanContext':
        """Start a new span"""
        
        # Get or create trace context
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            ctx = trace_context.get()
            if ctx:
                trace_id = ctx.trace_id
                parent_span_id = ctx.span_id
            else:
                trace_id = str(uuid.uuid4())
                parent_span_id = None
                
        span_id = str(uuid.uuid4())
        
        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=time.time()
        )
        
        # Create context
        ctx = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
        
        return SpanContext(span, self)
        
    def finish_span(self, span: Span):
        """Finish a span and export it"""
        span.end_time = time.time()
        span.duration = span.end_time - span.start_time
        
        self.spans.append(span)
        
        # Export to collectors
        for exporter in self.span_exporters:
            asyncio.create_task(exporter(span))
            
    def add_exporter(self, exporter: Callable):
        """Add span exporter"""
        self.span_exporters.append(exporter)
        
    def inject_context(self, carrier: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into carrier (e.g., HTTP headers)"""
        ctx = trace_context.get()
        if ctx:
            carrier['X-Trace-ID'] = ctx.trace_id
            carrier['X-Span-ID'] = ctx.span_id
            if ctx.parent_span_id:
                carrier['X-Parent-Span-ID'] = ctx.parent_span_id
            
            # Add baggage
            for key, value in ctx.baggage.items():
                carrier[f'X-Baggage-{key}'] = value
                
        return carrier
        
    def extract_context(self, carrier: Dict[str, str]) -> Optional[TraceContext]:
        """Extract trace context from carrier"""
        trace_id = carrier.get('X-Trace-ID')
        span_id = carrier.get('X-Span-ID')
        
        if trace_id and span_id:
            ctx = TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=carrier.get('X-Parent-Span-ID')
            )
            
            # Extract baggage
            for key, value in carrier.items():
                if key.startswith('X-Baggage-'):
                    baggage_key = key.replace('X-Baggage-', '')
                    ctx.baggage[baggage_key] = value
                    
            return ctx
        return None

class SpanContext:
    """
    Context manager for spans
    """
    
    def __init__(self, span: Span, tracer: DistributedTracer):
        self.span = span
        self.tracer = tracer
        self._token = None
        
    def __enter__(self):
        # Set context
        ctx = TraceContext(
            trace_id=self.span.trace_id,
            span_id=self.span.span_id,
            parent_span_id=self.span.parent_span_id
        )
        self._token = trace_context.set(ctx)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Handle errors
        if exc_type:
            self.span.status = "error"
            self.span.error = str(exc_val)
            self.add_log("error", str(exc_val))
            
        # Finish span
        self.tracer.finish_span(self.span)
        
        # Reset context
        if self._token:
            trace_context.reset(self._token)
            
    def add_tag(self, key: str, value: Any):
        """Add tag to span"""
        self.span.tags[key] = value
        
    def add_log(self, event: str, message: str):
        """Add log to span"""
        self.span.logs.append({
            'timestamp': time.time(),
            'event': event,
            'message': message
        })

# Decorators for tracing
def trace(operation_name: str = None):
    """Decorator to trace a function"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            with tracer.start_span(name) as span:
                span.add_tag('function', func.__name__)
                span.add_tag('args', str(args))
                span.add_tag('kwargs', str(kwargs))
                
                result = await func(*args, **kwargs)
                
                span.add_tag('result_type', type(result).__name__)
                return result
                
        def sync_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            with tracer.start_span(name) as span:
                span.add_tag('function', func.__name__)
                
                result = func(*args, **kwargs)
                
                span.add_tag('result_type', type(result).__name__)
                return result
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

# Middleware for FastAPI
class TracingMiddleware:
    """FastAPI middleware for distributed tracing"""
    
    def __init__(self, app, tracer: DistributedTracer):
        self.app = app
        self.tracer = tracer
        
    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            # Extract context from headers
            headers = dict(scope['headers'])
            ctx = self.tracer.extract_context(headers)
            
            # Start span
            with self.tracer.start_span(
                f"{scope['method']} {scope['path']}", 
                ctx
            ) as span:
                span.add_tag('http.method', scope['method'])
                span.add_tag('http.path', scope['path'])
                span.add_tag('http.url', str(scope.get('raw_path', b''), 'utf-8'))
                
                # Process request
                await self.app(scope, receive, send)
                
                span.add_tag('http.status_code', scope.get('status', 200))
        else:
            await self.app(scope, receive, send)

# Exporters
async def jaeger_exporter(span: Span):
    """Export span to Jaeger"""
    # Convert to Jaeger format
    jaeger_span = {
        'traceID': span.trace_id,
        'spanID': span.span_id,
        'parentSpanID': span.parent_span_id or '',
        'operationName': span.operation_name,
        'startTime': int(span.start_time * 1_000_000),  # microseconds
        'duration': int(span.duration * 1_000_000) if span.duration else 0,
        'tags': [
            {'key': k, 'value': str(v)} 
            for k, v in span.tags.items()
        ],
        'logs': span.logs,
        'process': {
            'serviceName': span.service_name,
            'tags': []
        }
    }
    
    # Send to Jaeger
    # await http_client.post(jaeger_url, json=jaeger_span)

async def prometheus_exporter(span: Span):
    """Export span metrics to Prometheus"""
    # Update metrics
    from backend.metrics_exporter import api_request_duration
    
    api_request_duration.labels(
        method=span.tags.get('http.method', 'unknown'),
        endpoint=span.tags.get('http.path', 'unknown')
    ).observe(span.duration)

# Global tracer instance
tracer = DistributedTracer('mltrainer')
tracer.add_exporter(jaeger_exporter)
tracer.add_exporter(prometheus_exporter)
```

### 5. SLA Monitoring

**File: `monitoring/sla_monitor.py`**
```python
"""
SLA Monitoring System
=====================
Track and alert on SLA violations
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

class SLAMetric(Enum):
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"

@dataclass
class SLATarget:
    metric: SLAMetric
    target_value: float
    measurement_window: timedelta
    comparison: str  # 'gte', 'lte', 'eq'
    
@dataclass
class SLA:
    name: str
    description: str
    service: str
    targets: List[SLATarget]
    alert_channels: List[str] = field(default_factory=list)
    
@dataclass
class SLAViolation:
    sla: SLA
    target: SLATarget
    actual_value: float
    expected_value: float
    timestamp: datetime
    duration: Optional[timedelta] = None
    resolved: bool = False

class SLAMonitor:
    """
    Monitor services for SLA compliance
    """
    
    def __init__(self):
        self.slas: Dict[str, SLA] = {}
        self.violations: List[SLAViolation] = []
        self.active_violations: Dict[str, SLAViolation] = {}
        self.metrics_store: Dict[str, List[Dict]] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        
    def register_sla(self, sla: SLA):
        """Register an SLA to monitor"""
        self.slas[sla.name] = sla
        
        # Start monitoring
        task = asyncio.create_task(self._monitor_sla(sla.name))
        self._monitoring_tasks[sla.name] = task
        
    async def _monitor_sla(self, sla_name: str):
        """Monitor a specific SLA"""
        sla = self.slas[sla_name]
        
        while sla_name in self.slas:
            try:
                # Check each target
                for target in sla.targets:
                    value = await self._get_metric_value(
                        sla.service, 
                        target.metric,
                        target.measurement_window
                    )
                    
                    # Check if violated
                    is_violated = self._check_violation(value, target)
                    
                    if is_violated:
                        await self._handle_violation(sla, target, value)
                    else:
                        await self._handle_recovery(sla, target)
                        
            except Exception as e:
                print(f"Error monitoring SLA {sla_name}: {e}")
                
            # Check every minute
            await asyncio.sleep(60)
            
    async def _get_metric_value(self, 
                              service: str,
                              metric: SLAMetric,
                              window: timedelta) -> float:
        """Get metric value for time window"""
        
        # In production, query from Prometheus/metrics store
        # For now, simulate
        
        if metric == SLAMetric.AVAILABILITY:
            # Calculate uptime percentage
            return 99.5  # Example
            
        elif metric == SLAMetric.LATENCY:
            # Get P99 latency
            return 150.0  # milliseconds
            
        elif metric == SLAMetric.ERROR_RATE:
            # Calculate error percentage
            return 0.5  # percentage
            
        elif metric == SLAMetric.THROUGHPUT:
            # Get requests per second
            return 1000.0  # RPS
            
    def _check_violation(self, value: float, target: SLATarget) -> bool:
        """Check if target is violated"""
        if target.comparison == 'gte':
            return value < target.target_value
        elif target.comparison == 'lte':
            return value > target.target_value
        elif target.comparison == 'eq':
            return value != target.target_value
        return False
        
    async def _handle_violation(self, 
                              sla: SLA, 
                              target: SLATarget,
                              actual_value: float):
        """Handle SLA violation"""
        
        violation_key = f"{sla.name}_{target.metric.value}"
        
        if violation_key not in self.active_violations:
            # New violation
            violation = SLAViolation(
                sla=sla,
                target=target,
                actual_value=actual_value,
                expected_value=target.target_value,
                timestamp=datetime.now()
            )
            
            self.active_violations[violation_key] = violation
            self.violations.append(violation)
            
            # Send alerts
            await self._send_alerts(violation)
            
    async def _handle_recovery(self, sla: SLA, target: SLATarget):
        """Handle recovery from violation"""
        
        violation_key = f"{sla.name}_{target.metric.value}"
        
        if violation_key in self.active_violations:
            violation = self.active_violations[violation_key]
            violation.resolved = True
            violation.duration = datetime.now() - violation.timestamp
            
            del self.active_violations[violation_key]
            
            # Send recovery notification
            await self._send_recovery_alert(violation)
            
    async def _send_alerts(self, violation: SLAViolation):
        """Send violation alerts"""
        
        message = f"""
SLA Violation Detected!

SLA: {violation.sla.name}
Service: {violation.sla.service}
Metric: {violation.target.metric.value}
Expected: {violation.target.comparison} {violation.expected_value}
Actual: {violation.actual_value}
Time: {violation.timestamp}
"""
        
        for channel in violation.sla.alert_channels:
            await self._send_to_channel(channel, message)
            
    async def _send_recovery_alert(self, violation: SLAViolation):
        """Send recovery notification"""
        
        message = f"""
SLA Violation Resolved!

SLA: {violation.sla.name}
Service: {violation.sla.service}
Metric: {violation.target.metric.value}
Duration: {violation.duration}
Resolved at: {datetime.now()}
"""
        
        for channel in violation.sla.alert_channels:
            await self._send_to_channel(channel, message)
            
    async def _send_to_channel(self, channel: str, message: str):
        """Send message to alert channel"""
        
        if channel == 'slack':
            # Send to Slack
            pass
        elif channel == 'email':
            # Send email
            pass
        elif channel == 'pagerduty':
            # Create PagerDuty incident
            pass
            
        print(f"[{channel}] {message}")
        
    def get_sla_status(self, sla_name: str) -> Dict[str, Any]:
        """Get current SLA status"""
        
        if sla_name not in self.slas:
            return {}
            
        sla = self.slas[sla_name]
        
        # Calculate compliance
        total_time = timedelta(days=30)  # Last 30 days
        violation_time = timedelta()
        
        for violation in self.violations:
            if violation.sla.name == sla_name:
                if violation.duration:
                    violation_time += violation.duration
                    
        compliance_pct = (
            (total_time - violation_time) / total_time * 100
        )
        
        return {
            'sla_name': sla_name,
            'service': sla.service,
            'compliance_percentage': compliance_pct,
            'active_violations': [
                v for k, v in self.active_violations.items()
                if v.sla.name == sla_name
            ],
            'total_violations': len([
                v for v in self.violations
                if v.sla.name == sla_name
            ])
        }
        
    def generate_sla_report(self) -> str:
        """Generate SLA compliance report"""
        
        report = ["# SLA Compliance Report\n"]
        report.append(f"Generated: {datetime.now()}\n")
        
        for sla_name, sla in self.slas.items():
            status = self.get_sla_status(sla_name)
            
            report.append(f"\n## {sla_name}\n")
            report.append(f"Service: {sla.service}\n")
            report.append(f"Compliance: {status['compliance_percentage']:.2f}%\n")
            report.append(f"Total Violations: {status['total_violations']}\n")
            
            if status['active_violations']:
                report.append("\n### Active Violations:\n")
                for v in status['active_violations']:
                    report.append(
                        f"- {v.target.metric.value}: {v.actual_value} "
                        f"(expected {v.target.comparison} {v.expected_value})\n"
                    )
                    
        return ''.join(report)

# Example SLAs
example_slas = [
    SLA(
        name="API_Availability_SLA",
        description="API must be available 99.9% of the time",
        service="mltrainer-api",
        targets=[
            SLATarget(
                metric=SLAMetric.AVAILABILITY,
                target_value=99.9,
                measurement_window=timedelta(days=30),
                comparison='gte'
            )
        ],
        alert_channels=['slack', 'pagerduty']
    ),
    
    SLA(
        name="API_Latency_SLA",
        description="API P99 latency must be under 200ms",
        service="mltrainer-api",
        targets=[
            SLATarget(
                metric=SLAMetric.LATENCY,
                target_value=200.0,
                measurement_window=timedelta(hours=1),
                comparison='lte'
            )
        ],
        alert_channels=['slack']
    ),
    
    SLA(
        name="ML_Model_Performance_SLA",
        description="Model inference must handle 1000 RPS",
        service="ml-inference",
        targets=[
            SLATarget(
                metric=SLAMetric.THROUGHPUT,
                target_value=1000.0,
                measurement_window=timedelta(minutes=5),
                comparison='gte'
            ),
            SLATarget(
                metric=SLAMetric.ERROR_RATE,
                target_value=1.0,
                measurement_window=timedelta(hours=1),
                comparison='lte'
            )
        ],
        alert_channels=['slack', 'email']
    )
]
```

## Implementation Priority

1. **Week 1**: Circuit breakers for all external calls
2. **Week 2**: Distributed tracing infrastructure
3. **Week 3**: Blue-green deployment setup
4. **Week 4**: Chaos engineering framework
5. **Week 5**: SLA monitoring and alerting

## Success Metrics

- Circuit breakers prevent 100% of cascading failures
- Full request tracing across all services
- Zero-downtime deployments achieved
- Chaos tests pass with <5% performance degradation
- 99.9% SLA compliance maintained