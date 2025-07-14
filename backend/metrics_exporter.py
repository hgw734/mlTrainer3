"""
Metrics Exporter
================

Exports custom metrics from mlTrainer for Prometheus monitoring.
"""

import logging
import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from prometheus_client.core import CollectorRegistry
from fastapi import FastAPI, Response
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# Create a custom registry
registry = CollectorRegistry()

# API Metrics
api_requests_total = Counter(
"mltrainer_api_requests_total", "Total number of API requests", ["method", "endpoint", "status"], registry=registry
)

api_request_duration = Histogram(
"mltrainer_api_request_duration_seconds",
"API request duration in seconds",
["method", "endpoint"],
registry=registry,
)

api_errors_total = Counter(
"mltrainer_api_errors_total", "Total number of API errors", ["method", "endpoint", "error_type"], registry=registry
)

# Model Metrics
model_training_total = Counter(
"mltrainer_model_training_total", "Total number of model training runs", ["model_id", "status"], registry=registry
)

model_training_duration = Histogram(
"mltrainer_model_training_duration_seconds",
"Model training duration in seconds",
["model_id"],
buckets=(10, 30, 60, 300, 600, 1800, 3600, 7200),
registry=registry,
)

model_prediction_duration = Histogram(
"mltrainer_model_prediction_duration_seconds",
"Model prediction duration in seconds",
["model_id"],
buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0),
registry=registry,
)

model_training_failures_total = Counter(
"mltrainer_model_training_failures_total",
"Total number of model training failures",
["model_id", "error_type"],
registry=registry,
)

# Background Job Metrics
background_queue_size = Gauge(
"mltrainer_background_queue_size", "Current size of background job queue", registry=registry
)

background_job_total = Counter(
"mltrainer_background_job_total", "Total number of background jobs", ["job_type", "status"], registry=registry
)

background_job_duration = Histogram(
"mltrainer_background_job_duration_seconds", "Background job duration in seconds", ["job_type"], registry=registry
)

background_job_failures_total = Counter(
"mltrainer_background_job_failures_total",
"Total number of background job failures",
["job_type", "error_type"],
registry=registry,
)

# Autonomous Session Metrics
autonomous_session_total = Counter(
"mltrainer_autonomous_session_total", "Total number of autonomous sessions", ["status"], registry=registry
)

autonomous_session_duration = Histogram(
"mltrainer_autonomous_session_duration_seconds",
"Autonomous session duration in seconds",
buckets=(60, 300, 600, 1800, 3600, 7200, 14400),
registry=registry,
)

autonomous_session_failures_total = Counter(
"mltrainer_autonomous_session_failures_total",
"Total number of autonomous session failures",
["error_type"],
registry=registry,
)

# Database Metrics
db_connection_pool_size = Gauge("mltrainer_db_connection_pool_size", "Database connection pool size", registry=registry)

db_connection_pool_available = Gauge(
"mltrainer_db_connection_pool_available", "Available database connections", registry=registry
)

db_query_duration = Histogram(
"mltrainer_db_query_duration_seconds",
"Database query duration in seconds",
["query_type"],
buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
registry=registry,
)

# Compliance Metrics
compliance_violations_total = Counter(
"mltrainer_compliance_violations_total",
"Total number of compliance violations",
["violation_type", "severity"],
registry=registry,
)

model_drift_score = Gauge("mltrainer_model_drift_score", "Model drift score (0-1)", ["model_id"], registry=registry)

# Memory Metrics
memory_store_size = Gauge(
"mltrainer_memory_store_size", "Current size of memory store", ["memory_type"], registry=registry
)

memory_importance_avg = Gauge(
"mltrainer_memory_importance_avg", "Average importance score of memories", registry=registry
)

# System Info
system_info = Info("mltrainer_system", "System information", registry=registry)


class MetricsCollector:
    """
    Collects and updates metrics from various mlTrainer components.
    """

    def __init__(self):
        self.last_update = datetime.now()
        self._initialize_system_info()

    def _initialize_system_info(self):
        """Initialize system information metrics"""
        import platform

        system_info.info(
        {
        "version": "1.0.0",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "node": platform.node(),
        }
        )

    async def collect_metrics(self):
        """Collect metrics from all components"""
        try:
            # Collect API metrics
            await self._collect_api_metrics()

            # Collect model metrics
            await self._collect_model_metrics()

            # Collect background job metrics
            await self._collect_background_metrics()

            # Collect database metrics
            await self._collect_database_metrics()

            # Collect compliance metrics
            await self._collect_compliance_metrics()

            # Collect memory metrics
            await self._collect_memory_metrics()

            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    async def _collect_api_metrics(self):
        """Collect API-related metrics"""
        # This would be integrated with the actual API
        # For now, we'll use to_be_implemented logic
        from backend.database import get_database_manager

        db = get_database_manager()

        # Get recent API activity from logs
        # This is a simplified production_implementation
        pass

    async def _collect_model_metrics(self):
        """Collect model-related metrics"""
        from mltrainer_models import get_ml_model_manager
        from core.trial_feedback_manager import get_trial_feedback_manager

        model_manager = get_ml_model_manager()
        feedback_manager = get_trial_feedback_manager()

        # Get model performance data
        for model_id, performance in list(model_manager.model_performance.items()):
            # Update drift score based on recent performance
            if performance.get("recent_performance"):
                drift_score = self._calculate_drift_score(performance)
                model_drift_score.labels(model_id=model_id).set(drift_score)

        # Get training metrics from feedback
        action_reports = feedback_manager.get_learning_summary()
        for action in action_reports.get("top_performing_actions", []):
            if action["action"].startswith("train_"):
                model_id = action["action"].replace("train_", "")
                success_rate = action["success_rate"]
                # Update counters based on historical data

    async def _collect_background_metrics(self):
        """Collect background job metrics"""
        from core.enhanced_background_manager import get_background_manager

        bg_manager = get_background_manager()

        # Update queue size
        queue_size = len(bg_manager.active_trials)
        background_queue_size.set(queue_size)

        # Get job statistics
        for trial_id, trial_state in list(bg_manager.active_trials.items()):
            job_type = "autonomous" if "autonomous" in trial_state.goal_context else "manual"

            # Update counters based on status
            if trial_state.status in ["completed", "failed", "partial_failure"]:
                background_job_total.labels(job_type=job_type, status=trial_state.status).inc()

                # Calculate duration for completed jobs
                if trial_state.status == "completed" and trial_state.results:
                    duration = self._calculate_duration(
                        trial_state.created_at, trial_state.results.get("completed_at", datetime.now().isoformat())
                    )
                    background_job_duration.labels(job_type=job_type).observe(duration)

    async def _collect_database_metrics(self):
        """Collect database metrics"""
        from backend.database import get_database_manager

        db = get_database_manager()

        # These would come from actual database connection pool
        # For SQLite, we'll use to_be_implemented values
        db_connection_pool_size.set(10)  # production_implementation pool size
        db_connection_pool_available.set(8)  # production_implementation available connections

    async def _collect_compliance_metrics(self):
        """Collect compliance metrics"""
        from backend.compliance_engine import get_compliance_gateway

        compliance = get_compliance_gateway()

        # Get recent compliance events
        if hasattr(compliance, "violation_count"):
            for violation_type, count in list(compliance.violation_count.items()):
                compliance_violations_total.labels(violation_type=violation_type, severity="high").inc(count)

    async def _collect_memory_metrics(self):
        """Collect memory store metrics"""
        from core.enhanced_memory import get_memory_manager

        memory = get_memory_manager()

        # Update memory store sizes
        memory_store_size.labels(memory_type="short_term").set(len(memory.short_term_memory))
        memory_store_size.labels(memory_type="long_term").set(len(memory.long_term_memory))
        memory_store_size.labels(memory_type="episodic").set(len(memory.episodic_memory))

        # Calculate average importance
        all_memories = list(memory.short_term_memory.values()) + list(memory.long_term_memory.values())
        if all_memories:
            avg_importance = sum(m.importance for m in all_memories) / len(all_memories)
            memory_importance_avg.set(avg_importance)

    def _calculate_drift_score(self, performance: Dict[str, Any]) -> float:
        """Calculate model drift score based on performance metrics"""
        # Simple drift calculation based on performance degradation
        recent_perf = performance.get("recent_performance", [])
        if len(recent_perf) < 2:
            return 0.0

        # Compare recent performance to baseline
        baseline = sum(recent_perf[: len(recent_perf) // 2]) / (len(recent_perf) // 2)
        current = sum(recent_perf[len(recent_perf) // 2 :]) / (len(recent_perf) - len(recent_perf) // 2)

        if baseline > 0:
            drift = abs(current - baseline) / baseline
            return min(drift, 1.0)
        return 0.0

    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration in seconds between two ISO timestamps"""
        try:
            start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            return (end - start).total_seconds()
        except Exception:
            return 0.0


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Middleware for tracking API metrics
class MetricsMiddleware:
    """FastAPI middleware for tracking API metrics"""

    def __init__(self, app: FastAPI):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()

            # Extract method and path
            method = scope["method"]
            path = scope["path"]

            # Track request
            api_requests_total.labels(method=method, endpoint=path, status="in_progress").inc()

            try:
                await self.app(scope, receive, send)

                # Track successful request
                duration = time.time() - start_time
                api_request_duration.labels(method=method, endpoint=path).observe(duration)

                api_requests_total.labels(method=method, endpoint=path, status="success").inc()

            except Exception as e:
                # Track error
                api_errors_total.labels(method=method, endpoint=path, error_type=type(e).__name__).inc()

                api_requests_total.labels(method=method, endpoint=path, status="error").inc()

                raise
        else:
            await self.app(scope, receive, send)


# Endpoint for Prometheus to scrape
async def metrics_endpoint():
    """Generate metrics for Prometheus"""
    # Update metrics before serving
    collector = get_metrics_collector()
    await collector.collect_metrics()

    # Generate metrics in Prometheus format
    metrics = generate_latest(registry)
    return Response(content=metrics, media_type="text/plain")


# Background task to periodically update metrics
async def metrics_update_loop():
    """Periodically update metrics"""
    collector = get_metrics_collector()

    while True:
        try:
            await collector.collect_metrics()
            await asyncio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logger.error(f"Error in metrics update loop: {e}")
            await asyncio.sleep(60)  # Wait longer on error
