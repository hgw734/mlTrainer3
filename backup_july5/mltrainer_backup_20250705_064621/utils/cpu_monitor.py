"""
CPU Performance Monitor for mlTrainer
=====================================

Purpose: Monitor CPU utilization and validate that 6-CPU allocation is being
used effectively during ML training and trials. Tracks resource usage
patterns and identifies potential performance bottlenecks.
"""

import psutil
import threading
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CPUMonitor:
    """Monitor CPU usage patterns for mlTrainer workloads"""
    
    def __init__(self, config_path: str = "config/system_resources.json"):
        """Initialize CPU monitor with system configuration"""
        self.config_path = Path(config_path)
        self.monitoring = False
        self.monitor_thread = None
        self.usage_history = []
        self.max_history_size = 1000
        
        # Load CPU configuration
        self.load_cpu_config()
        
        # Current monitoring data
        self.current_usage = {
            "total_cpu_percent": 0.0,
            "per_cpu_percent": [],
            "ml_workload_estimate": 0.0,
            "system_overhead": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"CPUMonitor initialized - tracking {self.total_cpus} CPUs")
    
    def load_cpu_config(self):
        """Load CPU configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    cpu_config = config.get("cpu_configuration", {})
                    
                    self.total_cpus = cpu_config.get("total_cpus", psutil.cpu_count())
                    self.ml_cpus = cpu_config.get("allocated_for_ml_trials", 6)
                    self.system_cpus = cpu_config.get("reserved_for_system", 2)
            else:
                # Default configuration
                self.total_cpus = psutil.cpu_count()
                self.ml_cpus = 6
                self.system_cpus = 2
                
        except Exception as e:
            logger.error(f"Error loading CPU config: {e}")
            self.total_cpus = psutil.cpu_count()
            self.ml_cpus = 6
            self.system_cpus = 2
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous CPU monitoring"""
        if self.monitoring:
            logger.warning("CPU monitoring already active")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"CPU monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop CPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("CPU monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get current CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
                total_percent = psutil.cpu_percent(interval=0.1)
                
                # Update current usage
                self.current_usage = {
                    "total_cpu_percent": total_percent,
                    "per_cpu_percent": cpu_percent,
                    "ml_workload_estimate": self._estimate_ml_workload(cpu_percent),
                    "system_overhead": self._estimate_system_overhead(cpu_percent),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add to history
                self.usage_history.append(self.current_usage.copy())
                
                # Trim history if too long
                if len(self.usage_history) > self.max_history_size:
                    self.usage_history = self.usage_history[-self.max_history_size:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in CPU monitoring loop: {e}")
                time.sleep(interval)
    
    def _estimate_ml_workload(self, cpu_percent: List[float]) -> float:
        """Estimate ML workload based on CPU usage patterns"""
        if not cpu_percent:
            return 0.0
        
        # Assume ML workloads use multiple CPUs heavily
        # Look for sustained high usage across multiple cores
        high_usage_cores = sum(1 for usage in cpu_percent if usage > 50.0)
        
        if high_usage_cores >= 4:  # 4+ cores heavily used
            return min(100.0, sum(cpu_percent) / len(cpu_percent))
        else:
            return max(0.0, (high_usage_cores / 6.0) * 100.0)
    
    def _estimate_system_overhead(self, cpu_percent: List[float]) -> float:
        """Estimate system overhead CPU usage"""
        if not cpu_percent:
            return 0.0
        
        # System overhead typically shows as lower, distributed usage
        low_usage_cores = [usage for usage in cpu_percent if usage < 30.0]
        
        if low_usage_cores:
            return sum(low_usage_cores) / len(low_usage_cores)
        else:
            return 0.0
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current CPU usage snapshot"""
        return self.current_usage.copy()
    
    def get_usage_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get CPU usage summary for the last N minutes"""
        if not self.usage_history:
            return {"error": "No usage history available"}
        
        # Filter recent history
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        recent_usage = []
        
        for usage in self.usage_history:
            try:
                usage_time = datetime.fromisoformat(usage["timestamp"]).timestamp()
                if usage_time >= cutoff_time:
                    recent_usage.append(usage)
            except:
                continue
        
        if not recent_usage:
            return {"error": f"No usage data in last {minutes} minutes"}
        
        # Calculate statistics
        total_cpu_values = [u["total_cpu_percent"] for u in recent_usage]
        ml_workload_values = [u["ml_workload_estimate"] for u in recent_usage]
        system_overhead_values = [u["system_overhead"] for u in recent_usage]
        
        return {
            "period_minutes": minutes,
            "sample_count": len(recent_usage),
            "total_cpu": {
                "avg": sum(total_cpu_values) / len(total_cpu_values),
                "min": min(total_cpu_values),
                "max": max(total_cpu_values)
            },
            "ml_workload": {
                "avg": sum(ml_workload_values) / len(ml_workload_values),
                "min": min(ml_workload_values),
                "max": max(ml_workload_values)
            },
            "system_overhead": {
                "avg": sum(system_overhead_values) / len(system_overhead_values),
                "min": min(system_overhead_values),
                "max": max(system_overhead_values)
            },
            "cpu_efficiency": {
                "target_ml_cpus": self.ml_cpus,
                "estimated_ml_utilization": sum(ml_workload_values) / len(ml_workload_values),
                "utilization_ratio": (sum(ml_workload_values) / len(ml_workload_values)) / (self.ml_cpus * 100 / self.total_cpus)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def validate_cpu_allocation(self) -> Dict[str, Any]:
        """Validate that CPU allocation is working as expected"""
        summary = self.get_usage_summary(minutes=2)
        
        if "error" in summary:
            return {
                "valid": False,
                "error": summary["error"],
                "recommendation": "Start CPU monitoring and run ML training to validate allocation"
            }
        
        # Validation criteria
        ml_utilization = summary["ml_workload"]["avg"]
        cpu_efficiency = summary["cpu_efficiency"]["utilization_ratio"]
        
        validation_result = {
            "valid": True,
            "cpu_allocation": {
                "total_cpus": self.total_cpus,
                "ml_allocated": self.ml_cpus,
                "system_reserved": self.system_cpus
            },
            "performance_metrics": {
                "avg_total_cpu": summary["total_cpu"]["avg"],
                "avg_ml_workload": ml_utilization,
                "cpu_efficiency_ratio": cpu_efficiency
            },
            "validation_status": "passed",
            "recommendations": []
        }
        
        # Check for issues
        if ml_utilization < 10.0:
            validation_result["recommendations"].append(
                "Low ML workload detected - consider running ML training to validate CPU allocation"
            )
        
        if cpu_efficiency > 1.5:
            validation_result["recommendations"].append(
                "High CPU efficiency - ML workload may be using more CPUs than allocated"
            )
        
        if summary["total_cpu"]["max"] > 90.0:
            validation_result["recommendations"].append(
                "High peak CPU usage detected - monitor for system bottlenecks"
            )
        
        return validation_result

# Global CPU monitor instance
_cpu_monitor = None

def get_cpu_monitor() -> CPUMonitor:
    """Get global CPU monitor instance"""
    global _cpu_monitor
    if _cpu_monitor is None:
        _cpu_monitor = CPUMonitor()
    return _cpu_monitor

def start_cpu_monitoring(interval: float = 1.0):
    """Start global CPU monitoring"""
    monitor = get_cpu_monitor()
    monitor.start_monitoring(interval)
    return monitor

def get_cpu_usage_summary(minutes: int = 5) -> Dict[str, Any]:
    """Get CPU usage summary from global monitor"""
    monitor = get_cpu_monitor()
    return monitor.get_usage_summary(minutes)

def validate_cpu_configuration() -> Dict[str, Any]:
    """Validate CPU configuration from global monitor"""
    monitor = get_cpu_monitor()
    return monitor.validate_cpu_allocation()