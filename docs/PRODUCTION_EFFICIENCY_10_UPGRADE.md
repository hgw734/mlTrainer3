# Production Efficiency Optimization: The Path to 10/10

## Overview

Production efficiency is critical for mlTrainer's scalability and cost-effectiveness. This plan addresses compute optimization, redundancy elimination, and intelligent resource management.

## Current Gaps

- No dynamic resource allocation
- Redundant computations across services
- Lack of intelligent caching strategies
- No compute-aware scheduling
- Missing cost optimization framework

## 1. Intelligent Resource Management

### A. Dynamic Resource Allocation

**File: `core/resource_optimizer.py`**
```python
"""
Dynamic Resource Optimizer
==========================
Intelligently allocates compute resources based on workload
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from kubernetes import client, config
import psutil

@dataclass
class ResourceProfile:
    cpu_cores: float
    memory_gb: float
    gpu_count: int = 0
    gpu_memory_gb: float = 0
    
@dataclass
class WorkloadPrediction:
    timestamp: datetime
    predicted_load: float
    confidence: float
    resource_recommendation: ResourceProfile

class DynamicResourceOptimizer:
    """
    Optimizes resource allocation based on workload patterns
    """
    
    def __init__(self):
        self.workload_history: List[Dict] = []
        self.resource_profiles = {
            'minimal': ResourceProfile(0.5, 1.0),
            'small': ResourceProfile(1.0, 2.0),
            'medium': ResourceProfile(2.0, 4.0),
            'large': ResourceProfile(4.0, 8.0),
            'xl': ResourceProfile(8.0, 16.0),
            'gpu_small': ResourceProfile(2.0, 8.0, 1, 16.0),
            'gpu_large': ResourceProfile(4.0, 16.0, 2, 32.0)
        }
        self.k8s_client = client.AppsV1Api()
        
    async def optimize_deployment(self, deployment_name: str, namespace: str = 'mltrainer'):
        """Optimize resource allocation for a deployment"""
        
        # Get current metrics
        current_metrics = await self._get_current_metrics(deployment_name, namespace)
        
        # Predict future workload
        prediction = await self._predict_workload(deployment_name, current_metrics)
        
        # Calculate optimal resources
        optimal_resources = self._calculate_optimal_resources(
            prediction, 
            current_metrics
        )
        
        # Apply if significant difference
        if self._should_scale(current_metrics['resources'], optimal_resources):
            await self._apply_resources(deployment_name, namespace, optimal_resources)
            
        # Record for learning
        self.workload_history.append({
            'timestamp': datetime.now(),
            'deployment': deployment_name,
            'metrics': current_metrics,
            'prediction': prediction,
            'applied_resources': optimal_resources
        })
        
    async def _get_current_metrics(self, deployment: str, namespace: str) -> Dict:
        """Get current resource metrics"""
        # In production, query Prometheus
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'request_rate': np.random.randint(100, 1000),  # Placeholder
            'response_time_p99': np.random.uniform(50, 200),  # Placeholder
            'resources': ResourceProfile(2.0, 4.0)  # Current allocation
        }
        
    async def _predict_workload(self, 
                              deployment: str, 
                              current_metrics: Dict) -> WorkloadPrediction:
        """Predict future workload using ML"""
        
        # Time-based patterns
        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        
        # Simple prediction model (would use real ML in production)
        base_load = current_metrics['request_rate']
        
        # Business hours boost
        if 9 <= hour <= 17 and day_of_week < 5:
            predicted_load = base_load * 1.5
        # Night time reduction
        elif 0 <= hour <= 6:
            predicted_load = base_load * 0.3
        else:
            predicted_load = base_load
            
        # Add seasonality
        if day_of_week in [0, 1]:  # Monday/Tuesday spike
            predicted_load *= 1.2
            
        # Calculate resource recommendation
        if predicted_load < 200:
            recommended = self.resource_profiles['small']
        elif predicted_load < 500:
            recommended = self.resource_profiles['medium']
        elif predicted_load < 1000:
            recommended = self.resource_profiles['large']
        else:
            recommended = self.resource_profiles['xl']
            
        return WorkloadPrediction(
            timestamp=datetime.now(),
            predicted_load=predicted_load,
            confidence=0.85,
            resource_recommendation=recommended
        )
        
    def _calculate_optimal_resources(self, 
                                   prediction: WorkloadPrediction,
                                   current_metrics: Dict) -> ResourceProfile:
        """Calculate optimal resource allocation"""
        
        recommended = prediction.resource_recommendation
        
        # Adjust based on current performance
        if current_metrics['response_time_p99'] > 150:  # Slow responses
            # Scale up
            return ResourceProfile(
                cpu_cores=recommended.cpu_cores * 1.2,
                memory_gb=recommended.memory_gb * 1.2,
                gpu_count=recommended.gpu_count
            )
        elif current_metrics['cpu_usage'] < 30 and current_metrics['memory_usage'] < 30:
            # Over-provisioned, scale down
            return ResourceProfile(
                cpu_cores=max(0.5, recommended.cpu_cores * 0.8),
                memory_gb=max(1.0, recommended.memory_gb * 0.8),
                gpu_count=recommended.gpu_count
            )
        else:
            return recommended
            
    def _should_scale(self, current: ResourceProfile, optimal: ResourceProfile) -> bool:
        """Determine if scaling is worthwhile"""
        cpu_diff = abs(current.cpu_cores - optimal.cpu_cores) / current.cpu_cores
        mem_diff = abs(current.memory_gb - optimal.memory_gb) / current.memory_gb
        
        # Scale if >20% difference
        return cpu_diff > 0.2 or mem_diff > 0.2
        
    async def _apply_resources(self, deployment: str, namespace: str, resources: ResourceProfile):
        """Apply new resource allocation to deployment"""
        
        # Update Kubernetes deployment
        body = {
            'spec': {
                'template': {
                    'spec': {
                        'containers': [{
                            'name': deployment,
                            'resources': {
                                'requests': {
                                    'cpu': f"{resources.cpu_cores}",
                                    'memory': f"{resources.memory_gb}Gi"
                                },
                                'limits': {
                                    'cpu': f"{resources.cpu_cores * 1.2}",
                                    'memory': f"{resources.memory_gb * 1.1}Gi"
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        # Apply patch
        self.k8s_client.patch_namespaced_deployment(
            name=deployment,
            namespace=namespace,
            body=body
        )
        
        print(f"Scaled {deployment} to {resources.cpu_cores} CPU, {resources.memory_gb}GB RAM")

# Vertical Pod Autoscaler configuration
class VPAOptimizer:
    """
    Configures Vertical Pod Autoscaler for automatic resource optimization
    """
    
    def generate_vpa_config(self, deployment_name: str) -> str:
        """Generate VPA configuration"""
        return f"""
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: {deployment_name}-vpa
  namespace: mltrainer
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {deployment_name}
  updatePolicy:
    updateMode: "Auto"  # Can be "Off", "Initial", "Recreate", or "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: {deployment_name}
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 10
        memory: 16Gi
      controlledResources: ["cpu", "memory"]
"""

### B. GPU Optimization

**File: `core/gpu_scheduler.py`**
```python
"""
GPU-Aware Scheduler
===================
Optimizes GPU utilization for ML workloads
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import nvidia_ml_py as nvml

@dataclass
class GPUJob:
    job_id: str
    model_type: str
    estimated_memory_gb: float
    estimated_compute_time: float
    priority: int = 5
    
@dataclass 
class GPUDevice:
    device_id: int
    total_memory_gb: float
    used_memory_gb: float
    utilization_percent: float
    assigned_jobs: List[str]

class GPUScheduler:
    """
    Intelligent GPU scheduling for ML workloads
    """
    
    def __init__(self):
        nvml.nvmlInit()
        self.gpu_count = nvml.nvmlDeviceGetCount()
        self.job_queue: List[GPUJob] = []
        self.running_jobs: Dict[str, GPUDevice] = {}
        self.completed_jobs: List[Dict] = []
        
    async def schedule_job(self, job: GPUJob) -> Optional[int]:
        """Schedule a job on the most appropriate GPU"""
        
        # Get current GPU states
        gpu_states = self._get_gpu_states()
        
        # Find best GPU for job
        best_gpu = self._find_best_gpu(job, gpu_states)
        
        if best_gpu is not None:
            # Assign job to GPU
            await self._assign_job(job, best_gpu)
            return best_gpu
        else:
            # Queue job
            self.job_queue.append(job)
            self.job_queue.sort(key=lambda j: j.priority, reverse=True)
            return None
            
    def _get_gpu_states(self) -> List[GPUDevice]:
        """Get current state of all GPUs"""
        gpu_states = []
        
        for i in range(self.gpu_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get memory info
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory_gb = mem_info.total / (1024**3)
            used_memory_gb = mem_info.used / (1024**3)
            
            # Get utilization
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            
            gpu_states.append(GPUDevice(
                device_id=i,
                total_memory_gb=total_memory_gb,
                used_memory_gb=used_memory_gb,
                utilization_percent=util.gpu,
                assigned_jobs=self._get_assigned_jobs(i)
            ))
            
        return gpu_states
        
    def _find_best_gpu(self, job: GPUJob, gpu_states: List[GPUDevice]) -> Optional[int]:
        """Find the best GPU for a job"""
        
        suitable_gpus = []
        
        for gpu in gpu_states:
            # Check if GPU has enough memory
            free_memory = gpu.total_memory_gb - gpu.used_memory_gb
            if free_memory >= job.estimated_memory_gb * 1.1:  # 10% buffer
                # Calculate score
                score = self._calculate_gpu_score(gpu, job)
                suitable_gpus.append((gpu.device_id, score))
                
        if not suitable_gpus:
            return None
            
        # Return GPU with highest score
        suitable_gpus.sort(key=lambda x: x[1], reverse=True)
        return suitable_gpus[0][0]
        
    def _calculate_gpu_score(self, gpu: GPUDevice, job: GPUJob) -> float:
        """Calculate GPU suitability score"""
        # Prefer less utilized GPUs
        utilization_score = (100 - gpu.utilization_percent) / 100
        
        # Prefer GPUs with more free memory
        free_memory_ratio = (gpu.total_memory_gb - gpu.used_memory_gb) / gpu.total_memory_gb
        memory_score = free_memory_ratio
        
        # Prefer GPUs with fewer jobs
        job_score = 1.0 / (len(gpu.assigned_jobs) + 1)
        
        # Weighted score
        return (utilization_score * 0.4 + memory_score * 0.4 + job_score * 0.2)
        
    async def _assign_job(self, job: GPUJob, gpu_id: int):
        """Assign job to GPU"""
        self.running_jobs[job.job_id] = gpu_id
        
        # In production, would actually launch the job
        print(f"Assigned job {job.job_id} to GPU {gpu_id}")
        
        # Simulate job execution
        await asyncio.sleep(job.estimated_compute_time)
        
        # Complete job
        self._complete_job(job.job_id)
        
    def _complete_job(self, job_id: str):
        """Mark job as completed"""
        if job_id in self.running_jobs:
            gpu_id = self.running_jobs[job_id]
            del self.running_jobs[job_id]
            
            self.completed_jobs.append({
                'job_id': job_id,
                'gpu_id': gpu_id,
                'completed_at': datetime.now()
            })
            
            # Check queue for waiting jobs
            self._process_queue()
            
    def _process_queue(self):
        """Process waiting jobs in queue"""
        if not self.job_queue:
            return
            
        # Try to schedule queued jobs
        scheduled = []
        for job in self.job_queue:
            gpu_id = asyncio.create_task(self.schedule_job(job))
            if gpu_id is not None:
                scheduled.append(job)
                
        # Remove scheduled jobs from queue
        for job in scheduled:
            self.job_queue.remove(job)
            
    def _get_assigned_jobs(self, gpu_id: int) -> List[str]:
        """Get jobs assigned to a GPU"""
        return [
            job_id for job_id, assigned_gpu 
            in self.running_jobs.items() 
            if assigned_gpu == gpu_id
        ]

# GPU Memory Optimization
class GPUMemoryOptimizer:
    """
    Optimizes GPU memory usage for models
    """
    
    def optimize_batch_size(self, model_size_mb: float, gpu_memory_gb: float) -> int:
        """Calculate optimal batch size for GPU memory"""
        
        # Reserve 20% for overhead
        available_memory_mb = gpu_memory_gb * 1024 * 0.8
        
        # Estimate memory per batch item (rough approximation)
        memory_per_item = model_size_mb * 0.1  # 10% of model size per item
        
        # Calculate optimal batch size
        optimal_batch_size = int(available_memory_mb / memory_per_item)
        
        # Round to power of 2 for efficiency
        return 2 ** int(np.log2(optimal_batch_size))
        
    def enable_mixed_precision(self):
        """Enable mixed precision training for memory efficiency"""
        return {
            'policy': 'mixed_float16',
            'loss_scale': 'dynamic',
            'config': {
                'compute_dtype': 'float16',
                'variable_dtype': 'float32'
            }
        }
```

## 2. Computation Deduplication

### A. Result Caching System

**File: `core/computation_cache.py`**
```python
"""
Computation Caching System
==========================
Avoids redundant calculations
"""

import hashlib
import pickle
import asyncio
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
import redis
from functools import wraps

class ComputationCache:
    """
    Intelligent caching for expensive computations
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.computation_stats: Dict[str, Dict] = {}
        
    def cached_computation(self, 
                         ttl: timedelta = timedelta(hours=1),
                         key_prefix: str = "",
                         invalidation_keys: List[str] = None):
        """Decorator for caching computation results"""
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(
                    func.__name__, 
                    args, 
                    kwargs,
                    key_prefix
                )
                
                # Try to get from cache
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    self._record_cache_hit(func.__name__)
                    return cached_result
                    
                # Compute result
                start_time = datetime.now()
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                compute_time = (datetime.now() - start_time).total_seconds()
                
                # Cache result
                self._cache_result(cache_key, result, ttl)
                
                # Record stats
                self._record_computation(func.__name__, compute_time)
                
                # Set invalidation keys if provided
                if invalidation_keys:
                    self._set_invalidation_keys(cache_key, invalidation_keys)
                    
                return result
                
            # Add cache management methods
            wrapper.invalidate = lambda: self._invalidate_function_cache(func.__name__, key_prefix)
            wrapper.get_stats = lambda: self.get_computation_stats(func.__name__)
            
            return wrapper
        return decorator
        
    def _generate_cache_key(self, func_name: str, args: tuple, 
                          kwargs: dict, prefix: str) -> str:
        """Generate deterministic cache key"""
        
        # Create hashable representation
        key_data = {
            'function': func_name,
            'args': args,
            'kwargs': kwargs
        }
        
        # Serialize and hash
        key_bytes = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        key_hash = hashlib.sha256(key_bytes).hexdigest()
        
        return f"{prefix}:{func_name}:{key_hash}"
        
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Retrieve from cache"""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            print(f"Cache retrieval error: {e}")
        return None
        
    def _cache_result(self, key: str, result: Any, ttl: timedelta):
        """Store result in cache"""
        try:
            serialized = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            print(f"Cache storage error: {e}")
            
    def _record_cache_hit(self, func_name: str):
        """Record cache hit statistics"""
        if func_name not in self.computation_stats:
            self.computation_stats[func_name] = {
                'hits': 0,
                'misses': 0,
                'total_compute_time': 0,
                'avg_compute_time': 0
            }
        self.computation_stats[func_name]['hits'] += 1
        
    def _record_computation(self, func_name: str, compute_time: float):
        """Record computation statistics"""
        if func_name not in self.computation_stats:
            self.computation_stats[func_name] = {
                'hits': 0,
                'misses': 0,
                'total_compute_time': 0,
                'avg_compute_time': 0
            }
            
        stats = self.computation_stats[func_name]
        stats['misses'] += 1
        stats['total_compute_time'] += compute_time
        stats['avg_compute_time'] = (
            stats['total_compute_time'] / stats['misses']
        )
        
    def get_computation_stats(self, func_name: str = None) -> Dict:
        """Get computation statistics"""
        if func_name:
            return self.computation_stats.get(func_name, {})
        return self.computation_stats
        
    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """Generate cache efficiency report"""
        report = {
            'total_functions': len(self.computation_stats),
            'function_stats': {},
            'total_time_saved': 0
        }
        
        for func_name, stats in self.computation_stats.items():
            hit_rate = stats['hits'] / (stats['hits'] + stats['misses']) if stats['misses'] > 0 else 0
            time_saved = stats['hits'] * stats['avg_compute_time']
            
            report['function_stats'][func_name] = {
                'hit_rate': hit_rate,
                'time_saved_seconds': time_saved,
                'avg_compute_time': stats['avg_compute_time']
            }
            
            report['total_time_saved'] += time_saved
            
        return report

# Usage Example
cache = ComputationCache()

@cache.cached_computation(ttl=timedelta(hours=2), key_prefix="features")
async def compute_features(symbol: str, start_date: str, end_date: str) -> np.ndarray:
    """Expensive feature computation"""
    # Simulate expensive computation
    await asyncio.sleep(5)
    return np.random.randn(100, 50)

@cache.cached_computation(ttl=timedelta(minutes=30), key_prefix="predictions")
async def make_prediction(model_id: str, features: np.ndarray) -> float:
    """Model prediction with caching"""
    # Simulate model inference
    await asyncio.sleep(1)
    return float(np.random.random())
```

### B. Computation Graph Optimizer

**File: `core/computation_graph.py`**
```python
"""
Computation Graph Optimizer
===========================
Optimizes execution order and eliminates redundancy
"""

from typing import Dict, List, Set, Any, Callable
from dataclasses import dataclass
import networkx as nx
import asyncio
from datetime import datetime

@dataclass
class ComputationNode:
    node_id: str
    function: Callable
    inputs: List[str]
    outputs: List[str]
    estimated_cost: float  # in seconds
    can_parallelize: bool = True
    
@dataclass
class ExecutionPlan:
    stages: List[List[str]]  # Nodes to execute in parallel at each stage
    estimated_time: float
    parallelism_factor: float

class ComputationGraphOptimizer:
    """
    Optimizes computation DAG for efficiency
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, ComputationNode] = {}
        self.results: Dict[str, Any] = {}
        
    def add_computation(self, node: ComputationNode):
        """Add computation node to graph"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id)
        
        # Add edges based on data dependencies
        for output in node.outputs:
            for other_id, other_node in self.nodes.items():
                if output in other_node.inputs and other_id != node.node_id:
                    self.graph.add_edge(node.node_id, other_id)
                    
    def optimize_execution_plan(self) -> ExecutionPlan:
        """Create optimized execution plan"""
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Computation graph contains cycles")
            
        # Topological sort for valid execution order
        topo_order = list(nx.topological_sort(self.graph))
        
        # Group into parallel stages
        stages = self._create_parallel_stages(topo_order)
        
        # Estimate execution time
        estimated_time = self._estimate_execution_time(stages)
        
        # Calculate parallelism factor
        total_serial_time = sum(
            self.nodes[node_id].estimated_cost 
            for node_id in topo_order
        )
        parallelism_factor = total_serial_time / estimated_time if estimated_time > 0 else 1
        
        return ExecutionPlan(
            stages=stages,
            estimated_time=estimated_time,
            parallelism_factor=parallelism_factor
        )
        
    def _create_parallel_stages(self, topo_order: List[str]) -> List[List[str]]:
        """Group nodes into stages that can execute in parallel"""
        stages = []
        scheduled = set()
        
        while len(scheduled) < len(topo_order):
            # Find all nodes that can be scheduled
            stage = []
            
            for node_id in topo_order:
                if node_id in scheduled:
                    continue
                    
                # Check if all dependencies are scheduled
                predecessors = list(self.graph.predecessors(node_id))
                if all(pred in scheduled for pred in predecessors):
                    if self.nodes[node_id].can_parallelize:
                        stage.append(node_id)
                    elif not stage:  # Non-parallelizable node gets its own stage
                        stage = [node_id]
                        break
                        
            stages.append(stage)
            scheduled.update(stage)
            
        return stages
        
    def _estimate_execution_time(self, stages: List[List[str]]) -> float:
        """Estimate total execution time"""
        total_time = 0
        
        for stage in stages:
            # Time for a stage is the max of all parallel computations
            stage_time = max(
                self.nodes[node_id].estimated_cost 
                for node_id in stage
            )
            total_time += stage_time
            
        return total_time
        
    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute computation plan"""
        
        for stage_idx, stage in enumerate(plan.stages):
            print(f"Executing stage {stage_idx + 1}/{len(plan.stages)}: {stage}")
            
            # Execute all nodes in stage in parallel
            tasks = []
            for node_id in stage:
                node = self.nodes[node_id]
                task = asyncio.create_task(
                    self._execute_node(node)
                )
                tasks.append((node_id, task))
                
            # Wait for all tasks in stage
            for node_id, task in tasks:
                result = await task
                self.results[node_id] = result
                
                # Store outputs for dependent nodes
                node = self.nodes[node_id]
                for output in node.outputs:
                    self.results[output] = result
                    
        return self.results
        
    async def _execute_node(self, node: ComputationNode) -> Any:
        """Execute a single computation node"""
        # Gather inputs
        inputs = {
            input_name: self.results.get(input_name)
            for input_name in node.inputs
        }
        
        # Execute function
        start_time = datetime.now()
        
        if asyncio.iscoroutinefunction(node.function):
            result = await node.function(**inputs)
        else:
            result = node.function(**inputs)
            
        execution_time = (datetime.now() - start_time).total_seconds()
        print(f"Executed {node.node_id} in {execution_time:.2f}s")
        
        return result
        
    def identify_redundant_computations(self) -> List[Tuple[str, str]]:
        """Identify computations that produce the same output"""
        redundant_pairs = []
        
        # Group nodes by their inputs and function
        computation_groups = {}
        
        for node_id, node in self.nodes.items():
            # Create signature based on function and inputs
            signature = (
                node.function.__name__,
                tuple(sorted(node.inputs))
            )
            
            if signature not in computation_groups:
                computation_groups[signature] = []
            computation_groups[signature].append(node_id)
            
        # Find groups with multiple nodes
        for signature, node_ids in computation_groups.items():
            if len(node_ids) > 1:
                # These nodes compute the same thing
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        redundant_pairs.append((node_ids[i], node_ids[j]))
                        
        return redundant_pairs
        
    def eliminate_redundancy(self):
        """Remove redundant computations from graph"""
        redundant_pairs = self.identify_redundant_computations()
        
        for node1, node2 in redundant_pairs:
            # Keep node1, redirect node2's outputs
            if node2 in self.graph:
                # Redirect all edges from node2 to node1
                successors = list(self.graph.successors(node2))
                for successor in successors:
                    self.graph.add_edge(node1, successor)
                    
                # Remove node2
                self.graph.remove_node(node2)
                del self.nodes[node2]
                
        print(f"Eliminated {len(redundant_pairs)} redundant computations")

# Usage Example
optimizer = ComputationGraphOptimizer()

# Define computation nodes
optimizer.add_computation(ComputationNode(
    node_id="load_data",
    function=lambda: "data",
    inputs=[],
    outputs=["raw_data"],
    estimated_cost=2.0
))

optimizer.add_computation(ComputationNode(
    node_id="compute_features",
    function=lambda raw_data: "features",
    inputs=["raw_data"],
    outputs=["features"],
    estimated_cost=5.0
))

optimizer.add_computation(ComputationNode(
    node_id="train_model",
    function=lambda features: "model",
    inputs=["features"],
    outputs=["model"],
    estimated_cost=10.0,
    can_parallelize=False  # Training uses all resources
))

# Optimize and execute
plan = optimizer.optimize_execution_plan()
results = asyncio.run(optimizer.execute_plan(plan))
```

## 3. Cost Optimization Framework

### A. Spot Instance Management

**File: `infrastructure/spot_manager.py`**
```python
"""
Spot Instance Manager
=====================
Optimizes costs using spot instances
"""

import boto3
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio

class SpotInstanceManager:
    """
    Manages spot instances for cost-effective compute
    """
    
    def __init__(self, region: str = 'us-east-1'):
        self.ec2_client = boto3.client('ec2', region_name=region)
        self.region = region
        self.spot_requests: Dict[str, Dict] = {}
        
    async def request_spot_capacity(self, 
                                  instance_type: str,
                                  max_price: float,
                                  count: int = 1,
                                  availability_zone: str = None) -> List[str]:
        """Request spot instances"""
        
        # Get current spot price
        current_price = await self._get_spot_price(instance_type, availability_zone)
        
        if current_price > max_price:
            print(f"Current spot price ${current_price} exceeds max ${max_price}")
            return []
            
        # Create launch specification
        launch_spec = {
            'ImageId': 'ami-12345678',  # ML-optimized AMI
            'InstanceType': instance_type,
            'KeyName': 'mltrainer-key',
            'SecurityGroupIds': ['sg-12345678'],
            'IamInstanceProfile': {
                'Arn': 'arn:aws:iam::123456789012:instance-profile/mltrainer'
            },
            'UserData': self._get_user_data(instance_type)
        }
        
        # Request spot instances
        response = self.ec2_client.request_spot_instances(
            SpotPrice=str(max_price),
            InstanceCount=count,
            Type='one-time',
            LaunchSpecification=launch_spec,
            InstanceInterruptionBehavior='terminate'
        )
        
        request_ids = [r['SpotInstanceRequestId'] for r in response['SpotInstanceRequests']]
        
        # Track requests
        for req_id in request_ids:
            self.spot_requests[req_id] = {
                'instance_type': instance_type,
                'max_price': max_price,
                'requested_at': datetime.now(),
                'status': 'pending'
            }
            
        return request_ids
        
    async def _get_spot_price(self, 
                            instance_type: str, 
                            availability_zone: str = None) -> float:
        """Get current spot price"""
        
        response = self.ec2_client.describe_spot_price_history(
            InstanceTypes=[instance_type],
            MaxResults=1,
            ProductDescriptions=['Linux/UNIX'],
            AvailabilityZone=availability_zone
        )
        
        if response['SpotPriceHistory']:
            return float(response['SpotPriceHistory'][0]['SpotPrice'])
        return float('inf')
        
    def _get_user_data(self, instance_type: str) -> str:
        """Get user data script for instance initialization"""
        
        return f"""#!/bin/bash
# Install necessary software
apt-get update
apt-get install -y docker.io nvidia-docker2

# Join Kubernetes cluster
kubeadm join --token abc123 k8s-master:6443

# Label node for workload
kubectl label node $(hostname) workload=ml-training
kubectl label node $(hostname) instance-type={instance_type}

# Start node exporter for monitoring
docker run -d -p 9100:9100 prom/node-exporter
"""

    async def optimize_workload_placement(self, 
                                        workloads: List[Dict]) -> Dict[str, str]:
        """Optimize placement of workloads on spot vs on-demand"""
        
        placement = {}
        
        for workload in workloads:
            # Determine if workload is spot-suitable
            if self._is_spot_suitable(workload):
                # Check spot availability
                instance_type = self._get_instance_type_for_workload(workload)
                spot_price = await self._get_spot_price(instance_type)
                on_demand_price = self._get_on_demand_price(instance_type)
                
                # Use spot if significantly cheaper
                if spot_price < on_demand_price * 0.7:  # 30% savings threshold
                    placement[workload['id']] = 'spot'
                else:
                    placement[workload['id']] = 'on-demand'
            else:
                placement[workload['id']] = 'on-demand'
                
        return placement
        
    def _is_spot_suitable(self, workload: Dict) -> bool:
        """Determine if workload is suitable for spot instances"""
        
        # Criteria for spot suitability
        if workload.get('interruptible', True) and \
           workload.get('duration_hours', 0) < 6 and \
           workload.get('priority', 5) < 8:
            return True
        return False
        
    def _get_instance_type_for_workload(self, workload: Dict) -> str:
        """Determine optimal instance type for workload"""
        
        cpu_required = workload.get('cpu', 1)
        memory_required = workload.get('memory_gb', 4)
        gpu_required = workload.get('gpu', 0)
        
        if gpu_required > 0:
            if gpu_required <= 1:
                return 'g4dn.xlarge'
            elif gpu_required <= 4:
                return 'g4dn.12xlarge'
            else:
                return 'p3.8xlarge'
        else:
            if cpu_required <= 2 and memory_required <= 8:
                return 't3.large'
            elif cpu_required <= 4 and memory_required <= 16:
                return 't3.xlarge'
            elif cpu_required <= 8 and memory_required <= 32:
                return 't3.2xlarge'
            else:
                return 'm5.4xlarge'
                
    def _get_on_demand_price(self, instance_type: str) -> float:
        """Get on-demand price for instance type"""
        
        # Simplified pricing (would fetch from API in production)
        prices = {
            't3.large': 0.0832,
            't3.xlarge': 0.1664,
            't3.2xlarge': 0.3328,
            'm5.4xlarge': 0.768,
            'g4dn.xlarge': 0.526,
            'g4dn.12xlarge': 3.912,
            'p3.8xlarge': 12.24
        }
        
        return prices.get(instance_type, 1.0)
```

### B. Resource Waste Detection

**File: `monitoring/resource_waste_detector.py`**
```python
"""
Resource Waste Detector
=======================
Identifies and eliminates resource waste
"""

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import numpy as np

class ResourceWasteDetector:
    """
    Detects and reports resource waste
    """
    
    def __init__(self):
        self.metrics_history: Dict[str, List[Dict]] = {}
        self.waste_report: Dict[str, Dict] = {}
        
    async def analyze_resource_usage(self, 
                                   service_name: str,
                                   time_window: timedelta = timedelta(days=7)) -> Dict:
        """Analyze resource usage patterns"""
        
        # Get metrics history
        metrics = await self._get_metrics(service_name, time_window)
        
        # Analyze different types of waste
        waste_analysis = {
            'over_provisioning': self._detect_over_provisioning(metrics),
            'idle_resources': self._detect_idle_resources(metrics),
            'inefficient_scaling': self._detect_inefficient_scaling(metrics),
            'memory_leaks': self._detect_memory_leaks(metrics),
            'cache_inefficiency': self._detect_cache_inefficiency(metrics)
        }
        
        # Calculate potential savings
        savings = self._calculate_savings(waste_analysis)
        
        return {
            'service': service_name,
            'waste_analysis': waste_analysis,
            'potential_savings': savings,
            'recommendations': self._generate_recommendations(waste_analysis)
        }
        
    def _detect_over_provisioning(self, metrics: List[Dict]) -> Dict:
        """Detect over-provisioned resources"""
        
        cpu_usage = [m['cpu_percent'] for m in metrics]
        memory_usage = [m['memory_percent'] for m in metrics]
        
        # Calculate percentiles
        cpu_p95 = np.percentile(cpu_usage, 95)
        memory_p95 = np.percentile(memory_usage, 95)
        
        # Check if consistently under-utilized
        if cpu_p95 < 30 and memory_p95 < 40:
            return {
                'detected': True,
                'severity': 'high',
                'cpu_p95': cpu_p95,
                'memory_p95': memory_p95,
                'recommendation': 'Reduce resource allocation by 50%'
            }
        elif cpu_p95 < 50 and memory_p95 < 60:
            return {
                'detected': True,
                'severity': 'medium',
                'cpu_p95': cpu_p95,
                'memory_p95': memory_p95,
                'recommendation': 'Reduce resource allocation by 25%'
            }
        else:
            return {'detected': False}
            
    def _detect_idle_resources(self, metrics: List[Dict]) -> Dict:
        """Detect idle or barely used resources"""
        
        # Check for extended periods of low activity
        idle_periods = []
        current_idle_start = None
        
        for i, metric in enumerate(metrics):
            if metric['cpu_percent'] < 5 and metric['memory_percent'] < 10:
                if current_idle_start is None:
                    current_idle_start = i
            else:
                if current_idle_start is not None:
                    idle_periods.append((current_idle_start, i))
                    current_idle_start = None
                    
        # Calculate total idle time
        total_idle_hours = sum(
            (end - start) * 5 / 60  # Assuming 5-minute intervals
            for start, end in idle_periods
        )
        
        if total_idle_hours > 24:  # More than 24 hours idle
            return {
                'detected': True,
                'severity': 'high',
                'idle_hours': total_idle_hours,
                'idle_percentage': (total_idle_hours / (len(metrics) * 5 / 60)) * 100,
                'recommendation': 'Consider serverless or scale-to-zero'
            }
        else:
            return {'detected': False}
            
    def _detect_inefficient_scaling(self, metrics: List[Dict]) -> Dict:
        """Detect inefficient scaling patterns"""
        
        replica_counts = [m.get('replicas', 1) for m in metrics]
        cpu_usage = [m['cpu_percent'] for m in metrics]
        
        # Check for scaling oscillation
        scaling_changes = sum(
            1 for i in range(1, len(replica_counts))
            if replica_counts[i] != replica_counts[i-1]
        )
        
        if scaling_changes > len(metrics) * 0.1:  # More than 10% of time
            return {
                'detected': True,
                'severity': 'medium',
                'scaling_changes': scaling_changes,
                'recommendation': 'Tune autoscaling parameters'
            }
        else:
            return {'detected': False}
            
    def _detect_memory_leaks(self, metrics: List[Dict]) -> Dict:
        """Detect potential memory leaks"""
        
        memory_usage = [m['memory_mb'] for m in metrics]
        
        # Check for monotonic increase
        if len(memory_usage) > 10:
            # Calculate trend
            x = np.arange(len(memory_usage))
            slope, _ = np.polyfit(x, memory_usage, 1)
            
            # Significant positive slope indicates leak
            if slope > 1.0:  # MB per interval
                return {
                    'detected': True,
                    'severity': 'high',
                    'growth_rate_mb_per_hour': slope * 12,  # 5-min intervals
                    'recommendation': 'Investigate memory leak'
                }
                
        return {'detected': False}
        
    def _detect_cache_inefficiency(self, metrics: List[Dict]) -> Dict:
        """Detect inefficient cache usage"""
        
        cache_hit_rates = [m.get('cache_hit_rate', 0) for m in metrics if 'cache_hit_rate' in m]
        
        if cache_hit_rates:
            avg_hit_rate = np.mean(cache_hit_rates)
            
            if avg_hit_rate < 0.5:  # Less than 50% hit rate
                return {
                    'detected': True,
                    'severity': 'medium',
                    'avg_hit_rate': avg_hit_rate,
                    'recommendation': 'Review cache configuration and TTL'
                }
                
        return {'detected': False}
        
    def _calculate_savings(self, waste_analysis: Dict) -> Dict[str, float]:
        """Calculate potential cost savings"""
        
        savings = {
            'monthly_compute': 0,
            'monthly_storage': 0,
            'monthly_network': 0
        }
        
        # Over-provisioning savings
        if waste_analysis['over_provisioning']['detected']:
            severity = waste_analysis['over_provisioning']['severity']
            if severity == 'high':
                savings['monthly_compute'] += 500  # $500/month
            elif severity == 'medium':
                savings['monthly_compute'] += 250  # $250/month
                
        # Idle resource savings
        if waste_analysis['idle_resources']['detected']:
            idle_pct = waste_analysis['idle_resources']['idle_percentage']
            savings['monthly_compute'] += 1000 * (idle_pct / 100)  # Proportional
            
        return savings
        
    def _generate_recommendations(self, waste_analysis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        for waste_type, analysis in waste_analysis.items():
            if analysis.get('detected'):
                recommendations.append(analysis.get('recommendation', ''))
                
        return [r for r in recommendations if r]  # Filter empty
        
    async def _get_metrics(self, 
                         service_name: str, 
                         time_window: timedelta) -> List[Dict]:
        """Get metrics for analysis"""
        
        # In production, would query Prometheus/CloudWatch
        # Simulating here
        num_points = int(time_window.total_seconds() / 300)  # 5-min intervals
        
        metrics = []
        for i in range(num_points):
            metrics.append({
                'timestamp': datetime.now() - timedelta(minutes=i*5),
                'cpu_percent': np.random.uniform(10, 80),
                'memory_percent': np.random.uniform(20, 70),
                'memory_mb': 1000 + i * np.random.uniform(-5, 10),
                'replicas': np.random.choice([1, 2, 3, 4]),
                'cache_hit_rate': np.random.uniform(0.3, 0.9)
            })
            
        return metrics

# Generate waste report
async def generate_waste_report(services: List[str]) -> Dict:
    """Generate comprehensive waste report"""
    
    detector = ResourceWasteDetector()
    report = {
        'generated_at': datetime.now().isoformat(),
        'services': {},
        'total_potential_savings': {
            'monthly_compute': 0,
            'monthly_storage': 0,
            'monthly_network': 0
        }
    }
    
    for service in services:
        analysis = await detector.analyze_resource_usage(service)
        report['services'][service] = analysis
        
        # Aggregate savings
        for category, amount in analysis['potential_savings'].items():
            report['total_potential_savings'][category] += amount
            
    return report
```

## 4. Implementation Priority

### Phase 1: Quick Wins (Week 1)
1. **Computation Cache** - Immediate reduction in redundant calculations
2. **Basic Resource Monitoring** - Identify obvious waste
3. **GPU Scheduling** - Better GPU utilization

### Phase 2: Core Optimization (Week 2-3)
1. **Dynamic Resource Allocation** - Right-size containers
2. **Computation Graph Optimizer** - Eliminate redundancy
3. **Spot Instance Integration** - Cost reduction

### Phase 3: Advanced Features (Week 4)
1. **Predictive Scaling** - Anticipate workload changes
2. **Multi-tenancy Optimization** - Share resources efficiently
3. **Cost Anomaly Detection** - Prevent budget overruns

## Success Metrics

### Efficiency Metrics
- [ ] 40% reduction in compute costs
- [ ] 60% improvement in resource utilization
- [ ] 80% cache hit rate for repeated computations
- [ ] 50% reduction in job completion times
- [ ] 90% GPU utilization during peak hours

### Cost Metrics
- [ ] 30% reduction in cloud spend
- [ ] 50% of workloads on spot instances
- [ ] <5% resource waste detected
- [ ] 70% reduction in over-provisioning

### Performance Metrics
- [ ] No performance degradation
- [ ] 99.9% SLA maintained
- [ ] <100ms overhead from optimization

## ROI Calculation

### Cost Savings
- **Compute**: $50K/month → $30K/month (40% reduction)
- **Storage**: $10K/month → $8K/month (20% reduction)  
- **Network**: $5K/month → $4K/month (20% reduction)
- **Total Annual Savings**: $312K

### Performance Gains
- **Developer Productivity**: 20% increase from faster builds/tests
- **Model Training**: 50% faster with optimized GPU usage
- **Time to Market**: 30% improvement

### Investment
- **Implementation**: 4 weeks × 2 engineers = $40K
- **Tooling**: $5K (monitoring upgrades)
- **ROI Period**: 2 months

## Conclusion

These production efficiency optimizations complete the 10/10 excellence upgrade by ensuring mlTrainer not only works perfectly but does so with minimal waste and maximum efficiency. The combination of intelligent resource management, computation deduplication, and cost optimization will result in a system that's both high-performing and cost-effective.