
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class Alert:
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    context: Dict[str, Any]

class AdvancedAlertSystem:
    """Real-time performance monitoring and alerting system"""
    
    def __init__(self):
        self.alert_rules = {}
        self.alert_history = []
        self.callbacks = []
        
    def add_alert_rule(self, name: str, metric_key: str, threshold: float, 
                      comparison: str, level: AlertLevel, callback: Callable = None):
        """Add a new alert rule"""
        self.alert_rules[name] = {
            "metric_key": metric_key,
            "threshold": threshold,
            "comparison": comparison,  # "greater", "less", "equal"
            "level": level,
            "callback": callback,
            "enabled": True,
            "last_triggered": None
        }
        logger.info(f"📋 Alert rule added: {name}")
    
    def check_performance_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Check all metrics against alert rules"""
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule["enabled"]:
                continue
                
            metric_value = self._get_nested_metric(metrics, rule["metric_key"])
            if metric_value is None:
                continue
                
            should_alert = self._evaluate_condition(
                metric_value, rule["threshold"], rule["comparison"]
            )
            
            if should_alert:
                alert = Alert(
                    level=rule["level"],
                    message=f"Alert: {rule_name} - {rule['metric_key']} is {metric_value:.4f} (threshold: {rule['threshold']:.4f})",
                    metric_name=rule["metric_key"],
                    current_value=metric_value,
                    threshold=rule["threshold"],
                    timestamp=datetime.now(),
                    context={"rule_name": rule_name, "metrics": metrics}
                )
                
                triggered_alerts.append(alert)
                self.alert_history.append(alert)
                rule["last_triggered"] = datetime.now()
                
                # Execute callback if provided
                if rule["callback"]:
                    try:
                        rule["callback"](alert)
                    except Exception as e:
                        logger.error(f"❌ Alert callback failed: {e}")
        
        return triggered_alerts
    
    def _get_nested_metric(self, metrics: Dict, key: str) -> float:
        """Get nested metric value using dot notation"""
        try:
            keys = key.split('.')
            value = metrics
            for k in keys:
                value = value[k]
            return float(value)
        except (KeyError, TypeError, ValueError):
            return None
    
    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate alert condition"""
        if comparison == "greater":
            return value > threshold
        elif comparison == "less":
            return value < threshold
        elif comparison == "equal":
            return abs(value - threshold) < 1e-6
        return False
    
    def setup_default_trading_alerts(self):
        """Setup standard trading performance alerts"""
        # Performance alerts
        self.add_alert_rule(
            "Low Sharpe Ratio",
            "sharpe",
            0.5,
            "less",
            AlertLevel.WARNING
        )
        
        self.add_alert_rule(
            "High Drawdown",
            "drawdown",
            0.15,
            "greater",
            AlertLevel.CRITICAL
        )
        
        self.add_alert_rule(
            "Low Accuracy",
            "accuracy",
            0.45,
            "less",
            AlertLevel.WARNING
        )
        
        self.add_alert_rule(
            "Extreme Loss",
            "total_return",
            -1000,
            "less",
            AlertLevel.EMERGENCY
        )
        
        # Model performance alerts
        self.add_alert_rule(
            "High Model Error",
            "rmse",
            10.0,
            "greater",
            AlertLevel.WARNING
        )
        
        logger.info("✅ Default trading alerts configured")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert system status"""
        recent_alerts = [a for a in self.alert_history if 
                        (datetime.now() - a.timestamp).days < 1]
        
        return {
            "total_rules": len(self.alert_rules),
            "active_rules": sum(1 for r in self.alert_rules.values() if r["enabled"]),
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL]),
            "last_alert": self.alert_history[-1] if self.alert_history else None
        }

# Initialize global alert system
alert_system = AdvancedAlertSystem()
alert_system.setup_default_trading_alerts()
