"""
Async Execution Engine for Parallel ML Model Execution
Handles concurrent execution of CPU-bound and I/O-bound tasks
"""

import asyncio
import logging
import multiprocessing
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ExecutionTask:
    """Represents a single execution task"""

    task_id: str
    action: str
    params: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class AsyncExecutionEngine:
    """
    Manages parallel execution of ML models and financial calculations.
    Uses both asyncio for I/O-bound tasks and multiprocessing for CPU-bound tasks.
    """

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.active_tasks: Dict[str, ExecutionTask] = {}
        self._task_counter = 0

        logger.info(f"Async Execution Engine initialized with {self.max_workers} workers")

    async def execute_parallel(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple actions in parallel"""
        # Create execution tasks
        tasks = self._create_execution_tasks(actions)

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(tasks)

        # Execute tasks respecting dependencies
        results = await self._execute_with_dependencies(tasks, dependency_graph)

        return {
            "total_tasks": len(tasks),
            "successful": sum(1 for t in list(tasks.values()) if t.status == "completed"),
            "failed": sum(1 for t in list(tasks.values()) if t.status == "failed"),
            "results": results,
        }

    def _create_execution_tasks(self, actions: List[Dict[str, Any]]) -> Dict[str, ExecutionTask]:
        """Create execution tasks from actions"""
        tasks = {}

        for action in actions:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}"

            task = ExecutionTask(
                task_id=task_id,
                action=action.get("action", ""),
                params=action.get("params", {}),
                dependencies=action.get("dependencies", []),
            )

            tasks[task_id] = task
            self.active_tasks[task_id] = task

        return tasks

    def _build_dependency_graph(self, tasks: Dict[str, ExecutionTask]) -> Dict[str, List[str]]:
        """Build dependency graph for task execution"""
        graph = {task_id: [] for task_id in tasks}

        for task_id, task in list(tasks.items()):
            for dep in task.dependencies:
                if dep in graph:
                    graph[dep].append(task_id)

        return graph

    async def _execute_with_dependencies(
        self, tasks: Dict[str, ExecutionTask], dependency_graph: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Execute tasks respecting dependencies"""
        completed = set()
        results = {}

        # Find tasks with no dependencies
        ready_tasks = [task_id for task_id, task in list(tasks.items()) if not task.dependencies]

        while ready_tasks or len(completed) < len(tasks):
            if not ready_tasks:
                # Check if we're stuck (circular dependencies)
                pending = set(tasks.keys()) - completed
                if pending:
                    logger.error(f"Circular dependency detected. Pending tasks: {pending}")
                    break
                continue

            # Execute ready tasks in parallel
            batch_results = await self._execute_batch({task_id: tasks[task_id] for task_id in ready_tasks})

            # Update results and find newly ready tasks
            for task_id, result in list(batch_results.items()):
                results[task_id] = result
                completed.add(task_id)

                # Check if any dependent tasks are now ready
                for dependent_id in dependency_graph.get(task_id, []):
                    dependent_task = tasks.get(dependent_id)
                    if dependent_task and all(dep in completed for dep in dependent_task.dependencies):
                        ready_tasks.append(dependent_id)

            # Remove executed tasks from ready list
            ready_tasks = [tid for tid in ready_tasks if tid not in completed]

        return results

    async def _execute_batch(self, batch: Dict[str, ExecutionTask]) -> Dict[str, Any]:
        """Execute a batch of tasks in parallel"""
        coroutines = []

        for task_id, task in list(batch.items()):
            if self._is_cpu_bound(task.action):
                # Use process pool for CPU-bound tasks
                coro = self._execute_cpu_bound(task)
            else:
                # Use thread pool for I/O-bound tasks
                coro = self._execute_io_bound(task)

            coroutines.append((task_id, coro))

        # Execute all coroutines in parallel
        results = {}
        for task_id, coro in coroutines:
            try:
                task = batch[task_id]
                task.status = "running"
                task.started_at = datetime.now().isoformat()

                result = await coro

                task.status = "completed"
                task.result = result
                task.completed_at = datetime.now().isoformat()

                results[task_id] = {
                    "status": "success",
                    "result": result,
                    "duration": self._calculate_duration(task.started_at, task.completed_at),
                }

            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                task.completed_at = datetime.now().isoformat()

                results[task_id] = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}

                logger.error(f"Task {task_id} failed: {e}")

        return results

    def _is_cpu_bound(self, action: str) -> bool:
        """Determine if an action is CPU-bound"""
        cpu_bound_actions = [
            "train_",
            "optimize_",
            "calculate_",
            "backtest_",
            "monte_carlo",
            "clustering",
            "neural_network",
        ]

        return any(pattern in action.lower() for pattern in cpu_bound_actions)

    async def _execute_cpu_bound(self, task: ExecutionTask) -> Any:
        """Execute CPU-bound task in process pool"""
        loop = asyncio.get_event_loop()

        # Import here to avoid pickling issues
        from core.unified_executor import get_unified_executor

        # Create a partial function that can be pickled
        if task.action.startswith("train_"):
            model_id = task.action.replace("train_", "")
            func = _train_model_process
            args = (model_id, task.params)
        elif task.action.startswith("calculate_"):
            model_id = task.action.replace("calculate_", "")
            func = _calculate_financial_process
            args = (model_id, task.params)
        else:
            # Default execution
            func = _execute_action_process
            args = (task.action, task.params)

        # Run in process pool
        result = await loop.run_in_executor(self.process_executor, func, *args)

        return result

    async def _execute_io_bound(self, task: ExecutionTask) -> Any:
        """Execute I/O-bound task in thread pool"""
        loop = asyncio.get_event_loop()

        # Import here to avoid circular imports
        from core.unified_executor import get_unified_executor

        executor = get_unified_executor()

        # Run in thread pool
        if task.action in executor.registered_actions:
            func = executor.registered_actions[task.action]["function"]
            result = await loop.run_in_executor(self.thread_executor, func, **task.params)
        else:
            result = await loop.run_in_executor(self.thread_executor, self._default_execution, task.action, task.params)

        return result

    def _default_execution(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default execution for unknown actions"""
        return {
            "action": action,
            "params": params,
            "message": "Action executed (default handler)",
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_duration(self, start: str, end: str) -> float:
        """Calculate duration in seconds"""
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        return (end_dt - start_dt).total_seconds()

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if task.status == "running":
                task.status = "cancelled"
                task.completed_at = datetime.now().isoformat()
                return True
        return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task.task_id,
                "action": task.action,
                "status": task.status,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error": task.error,
            }
        return None

    def cleanup(self):
        """Cleanup executors"""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


# Process pool functions (must be at module level for pickling)
def _train_model_process(model_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Train model in separate process"""
    try:
        # Import here to avoid circular imports
        import sys
        import os

        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from mltrainer_models import get_ml_model_manager

        manager = get_ml_model_manager()
        result = manager.train_model(model_id, **params)

        return {"model_id": model_id, "metrics": result.performance_metrics, "training_time": result.training_time}
    except Exception as e:
        return {"error": str(e), "model_id": model_id}


def _calculate_financial_process(model_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate financial model in separate process"""
    try:
        # Import here to avoid circular imports
        import sys
        import os

        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from mltrainer_financial_models import get_financial_model_manager

        manager = get_financial_model_manager()
        result = manager.run_model(model_id, **params)

        return {
            "model_id": model_id,
            "execution_time": result.execution_time,
            "results": {
                "option_price": result.option_price,
                "portfolio_weights": result.portfolio_weights,
                "risk_metrics": result.risk_metrics,
            },
        }
    except Exception as e:
        return {"error": str(e), "model_id": model_id}


def _execute_action_process(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute generic action in separate process"""
    return {"action": action, "params": params, "executed_in": "process_pool", "timestamp": datetime.now().isoformat()}


# Singleton instance
_execution_engine = None


def get_async_execution_engine() -> AsyncExecutionEngine:
    """Get the async execution engine instance"""
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = AsyncExecutionEngine()
    return _execution_engine
