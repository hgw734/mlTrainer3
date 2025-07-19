"""
mlTrainer - System Monitoring
=============================

Purpose: Comprehensive system monitoring including health checks, performance
metrics, resource utilization, and alert generation. Provides real-time
system status and automated health assessments.

Features:
- Real-time health monitoring
- Performance metrics collection
- Resource utilization tracking
- Automated alert generation
- System diagnostics and reporting
"""

import logging
import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
import os
from collections import deque
import json

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Comprehensive system monitoring and health checking"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.health_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.alert_thresholds = self._load_alert_thresholds()
        self.monitoring_active = True
        
        # Component status tracking
        self.component_status = {
            "data_sources": False,
            "ml_pipeline": False,
            "compliance_engine": False,
            "notification_system": False,
            "backend_api": False
        }
        
        # Performance baseline
        self.performance_baseline = {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "disk_threshold": 90.0,
            "response_time_threshold": 5.0
        }
        
        logger.info("SystemMonitor initialized")
    
    def _load_alert_thresholds(self) -> Dict[str, Any]:
        """Load monitoring alert thresholds"""
        return {
            "cpu_warning": 70.0,
            "cpu_critical": 85.0,
            "memory_warning": 75.0,
            "memory_critical": 90.0,
            "disk_warning": 80.0,
            "disk_critical": 95.0,
            "response_time_warning": 3.0,
            "response_time_critical": 10.0,
            "api_failure_threshold": 3,
            "component_failure_threshold": 2
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "healthy": True,
            "overall_score": 100.0,
            "components": {},
            "metrics": {},
            "alerts": []
        }
        
        try:
            # Check system resources
            resource_health = self._check_system_resources()
            health_status["components"]["system_resources"] = resource_health
            
            # Check component health
            component_health = self._check_component_health()
            health_status["components"]["system_components"] = component_health
            
            # Check API connectivity
            api_health = self._check_api_connectivity()
            health_status["components"]["api_connectivity"] = api_health
            
            # Calculate overall health score
            component_scores = []
            for component, status in health_status["components"].items():
                if isinstance(status, dict) and "score" in status:
                    component_scores.append(status["score"])
                    if not status.get("healthy", True):
                        health_status["healthy"] = False
            
            if component_scores:
                health_status["overall_score"] = round(sum(component_scores) / len(component_scores), 1)
            
            # Generate alerts for issues
            health_status["alerts"] = self._generate_health_alerts(health_status)
            
            # Store in history
            self.health_history.append(health_status.copy())
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["healthy"] = False
            health_status["overall_score"] = 0
            health_status["error"] = str(e)
        
        return health_status
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization"""
        resource_status = {
            "healthy": True,
            "score": 100.0,
            "metrics": {},
            "warnings": []
        }
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            resource_status["metrics"]["cpu_percent"] = cpu_percent
            
            if cpu_percent > self.alert_thresholds["cpu_critical"]:
                resource_status["warnings"].append(f"Critical CPU usage: {cpu_percent:.1f}%")
                resource_status["healthy"] = False
                resource_status["score"] -= 30
            elif cpu_percent > self.alert_thresholds["cpu_warning"]:
                resource_status["warnings"].append(f"High CPU usage: {cpu_percent:.1f}%")
                resource_status["score"] -= 15
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            resource_status["metrics"]["memory_percent"] = memory_percent
            resource_status["metrics"]["memory_available_gb"] = round(memory.available / (1024**3), 2)
            
            if memory_percent > self.alert_thresholds["memory_critical"]:
                resource_status["warnings"].append(f"Critical memory usage: {memory_percent:.1f}%")
                resource_status["healthy"] = False
                resource_status["score"] -= 30
            elif memory_percent > self.alert_thresholds["memory_warning"]:
                resource_status["warnings"].append(f"High memory usage: {memory_percent:.1f}%")
                resource_status["score"] -= 15
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            resource_status["metrics"]["disk_percent"] = disk_percent
            resource_status["metrics"]["disk_free_gb"] = round(disk.free / (1024**3), 2)
            
            if disk_percent > self.alert_thresholds["disk_critical"]:
                resource_status["warnings"].append(f"Critical disk usage: {disk_percent:.1f}%")
                resource_status["healthy"] = False
                resource_status["score"] -= 30
            elif disk_percent > self.alert_thresholds["disk_warning"]:
                resource_status["warnings"].append(f"High disk usage: {disk_percent:.1f}%")
                resource_status["score"] -= 15
            
            # Network connections
            connections = len(psutil.net_connections())
            resource_status["metrics"]["network_connections"] = connections
            
            # Process count
            process_count = len(psutil.pids())
            resource_status["metrics"]["process_count"] = process_count
            
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            resource_status["healthy"] = False
            resource_status["score"] = 0
            resource_status["error"] = str(e)
        
        return resource_status
    
    def _check_component_health(self) -> Dict[str, Any]:
        """Check health of system components"""
        component_status = {
            "healthy": True,
            "score": 100.0,
            "components": {},
            "failures": []
        }
        
        # Check backend API
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                api_data = response.json()
                component_status["components"]["backend_api"] = {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds(),
                    "services": api_data.get("services", {})
                }
            else:
                component_status["components"]["backend_api"] = {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
                component_status["failures"].append("backend_api")
        except Exception as e:
            component_status["components"]["backend_api"] = {
                "status": "unreachable",
                "error": str(e)
            }
            component_status["failures"].append("backend_api")
        
        # Check database connectivity (if applicable)
        component_status["components"]["database"] = {
            "status": "not_configured",
            "note": "Using in-memory storage"
        }
        
        # Check file system access
        try:
            test_file = "test_write.tmp"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            component_status["components"]["filesystem"] = {"status": "healthy"}
        except Exception as e:
            component_status["components"]["filesystem"] = {
                "status": "error",
                "error": str(e)
            }
            component_status["failures"].append("filesystem")
        
        # Calculate component health score
        total_components = len(component_status["components"])
        failed_components = len(component_status["failures"])
        
        if failed_components > 0:
            component_status["healthy"] = False
            component_status["score"] -= (failed_components / total_components) * 100
        
        return component_status
    
    def _check_api_connectivity(self) -> Dict[str, Any]:
        """Check external API connectivity"""
        api_status = {
            "healthy": True,
            "score": 100.0,
            "apis": {},
            "failures": []
        }
        
        # API endpoints to test
        api_tests = {
            "polygon": {
                "url": "https://api.polygon.io/v1/marketstatus/now",
                "params": {"apikey": os.getenv("POLYGON_API_KEY", "test")}
            },
            "fred": {
                "url": "https://api.stlouisfed.org/fred/series",
                "params": {
                    "series_id": "VIXCLS",
                    "api_key": os.getenv("FRED_API_KEY", "test"),
                    "file_type": "json",
                    "limit": "1"
                }
            }
        }
        
        for api_name, test_config in api_tests.items():
            try:
                start_time = time.time()
                response = requests.get(
                    test_config["url"],
                    params=test_config["params"],
                    timeout=10
                )
                response_time = time.time() - start_time
                
                api_status["apis"][api_name] = {
                    "status": "reachable" if response.status_code < 500 else "error",
                    "response_time": round(response_time, 3),
                    "status_code": response.status_code
                }
                
                if response.status_code >= 500:
                    api_status["failures"].append(api_name)
                
            except Exception as e:
                api_status["apis"][api_name] = {
                    "status": "unreachable",
                    "error": str(e)
                }
                api_status["failures"].append(api_name)
        
        # Calculate API health score
        total_apis = len(api_tests)
        failed_apis = len(api_status["failures"])
        
        if failed_apis > 0:
            if failed_apis == total_apis:
                api_status["healthy"] = False
            api_status["score"] -= (failed_apis / total_apis) * 50  # Less critical than system components
        
        return api_status
    
    def _generate_health_alerts(self, health_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on health status"""
        alerts = []
        
        # Check overall health
        if not health_status["healthy"]:
            alerts.append({
                "type": "system_health",
                "severity": "critical",
                "message": f"System health compromised - Score: {health_status['overall_score']:.1f}%",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check component alerts
        for component_name, component_data in health_status.get("components", {}).items():
            if isinstance(component_data, dict):
                warnings = component_data.get("warnings", [])
                for warning in warnings:
                    alerts.append({
                        "type": "resource_warning",
                        "component": component_name,
                        "severity": "warning",
                        "message": warning,
                        "timestamp": datetime.now().isoformat()
                    })
                
                failures = component_data.get("failures", [])
                for failure in failures:
                    alerts.append({
                        "type": "component_failure",
                        "component": failure,
                        "severity": "critical",
                        "message": f"Component {failure} is not functioning properly",
                        "timestamp": datetime.now().isoformat()
                    })
        
        return alerts
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime": self._get_uptime(),
            "system": {},
            "application": {}
        }
        
        try:
            # System metrics
            metrics["system"] = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_io": dict(psutil.disk_io_counters()._asdict()) if psutil.disk_io_counters() else {},
                "network_io": dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {},
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
            
            # Application metrics
            current_process = psutil.Process()
            metrics["application"] = {
                "memory_usage_mb": round(current_process.memory_info().rss / (1024 * 1024), 2),
                "cpu_percent": current_process.cpu_percent(),
                "num_threads": current_process.num_threads(),
                "num_fds": current_process.num_fds() if hasattr(current_process, 'num_fds') else 0,
                "create_time": datetime.fromtimestamp(current_process.create_time()).isoformat()
            }
            
            # Store in history
            self.performance_history.append(metrics.copy())
            
        except Exception as e:
            logger.error(f"Performance metrics collection failed: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _get_uptime(self) -> Dict[str, Any]:
        """Get system and application uptime"""
        now = datetime.now()
        app_uptime = now - self.start_time
        
        return {
            "application_uptime_seconds": app_uptime.total_seconds(),
            "application_uptime_human": self._format_timedelta(app_uptime),
            "system_uptime_seconds": time.time() - psutil.boot_time(),
            "started_at": self.start_time.isoformat()
        }
    
    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta to human readable string"""
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds or not parts:
            parts.append(f"{seconds}s")
        
        return " ".join(parts)
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        logger.info("Running system health check")
        
        health_result = self.get_system_health()
        performance_result = self.get_performance_metrics()
        
        comprehensive_check = {
            "check_time": datetime.now().isoformat(),
            "health": health_result,
            "performance": performance_result,
            "recommendations": []
        }
        
        # Generate recommendations
        if not health_result["healthy"]:
            comprehensive_check["recommendations"].append("System health issues detected - review component status")
        
        if performance_result["system"]["cpu_percent"] > 80:
            comprehensive_check["recommendations"].append("High CPU usage - consider scaling or optimization")
        
        if performance_result["system"]["memory_percent"] > 85:
            comprehensive_check["recommendations"].append("High memory usage - check for memory leaks")
        
        return comprehensive_check
    
    def get_monitoring_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get monitoring summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent health checks
        recent_health = []
        for health_check in self.health_history:
            try:
                check_time = datetime.fromisoformat(health_check["timestamp"])
                if check_time > cutoff_time:
                    recent_health.append(health_check)
            except:
                continue
        
        # Filter recent performance metrics
        recent_performance = []
        for perf_metric in self.performance_history:
            try:
                metric_time = datetime.fromisoformat(perf_metric["timestamp"])
                if metric_time > cutoff_time:
                    recent_performance.append(perf_metric)
            except:
                continue
        
        summary = {
            "period_hours": hours,
            "health_checks": len(recent_health),
            "performance_snapshots": len(recent_performance),
            "avg_health_score": 0,
            "uptime_percentage": 100.0,
            "alert_count": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        if recent_health:
            # Calculate average health score
            health_scores = [h["overall_score"] for h in recent_health if "overall_score" in h]
            if health_scores:
                summary["avg_health_score"] = round(sum(health_scores) / len(health_scores), 1)
            
            # Count alerts
            total_alerts = sum(len(h.get("alerts", [])) for h in recent_health)
            summary["alert_count"] = total_alerts
            
            # Calculate uptime (based on healthy status)
            healthy_checks = sum(1 for h in recent_health if h.get("healthy", False))
            summary["uptime_percentage"] = round((healthy_checks / len(recent_health)) * 100, 1)
        
        if recent_performance:
            # Calculate average resource usage
            cpu_values = [p["system"]["cpu_percent"] for p in recent_performance if "system" in p]
            memory_values = [p["system"]["memory_percent"] for p in recent_performance if "system" in p]
            
            if cpu_values:
                summary["avg_cpu_percent"] = round(sum(cpu_values) / len(cpu_values), 1)
            if memory_values:
                summary["avg_memory_percent"] = round(sum(memory_values) / len(memory_values), 1)
        
        return summary
    
    def clear_monitoring_history(self, older_than_hours: int = 168) -> Dict[str, int]:
        """Clear old monitoring data"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        # Clear health history
        initial_health_count = len(self.health_history)
        filtered_health = deque(maxlen=1000)
        
        for health_check in self.health_history:
            try:
                check_time = datetime.fromisoformat(health_check["timestamp"])
                if check_time > cutoff_time:
                    filtered_health.append(health_check)
            except:
                filtered_health.append(health_check)  # Keep if timestamp invalid
        
        self.health_history = filtered_health
        
        # Clear performance history
        initial_perf_count = len(self.performance_history)
        filtered_performance = deque(maxlen=1000)
        
        for perf_metric in self.performance_history:
            try:
                metric_time = datetime.fromisoformat(perf_metric["timestamp"])
                if metric_time > cutoff_time:
                    filtered_performance.append(perf_metric)
            except:
                filtered_performance.append(perf_metric)  # Keep if timestamp invalid
        
        self.performance_history = filtered_performance
        
        removed_counts = {
            "health_checks_removed": initial_health_count - len(self.health_history),
            "performance_metrics_removed": initial_perf_count - len(self.performance_history)
        }
        
        logger.info(f"Cleared monitoring history: {removed_counts}")
        return removed_counts
