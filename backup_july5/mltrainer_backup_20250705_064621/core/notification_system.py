"""
mlTrainer - Notification System
==============================

Purpose: Manages the 7-type notification system for real-time trading alerts.
Handles alert generation, prioritization, and delivery for critical market
events and system status changes.

Alert Types:
1. Regime Change Detected
2. Entry Signal Strength Spiked  
3. Exit Signal Triggered
4. Stop-Loss Hit
5. Target Reached
6. Confidence Drop Warning
7. Portfolio Deviation from Optimal Path
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import threading
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

class AlertType(Enum):
    REGIME_CHANGE = "regime_change"
    ENTRY_SIGNAL = "entry_signal"
    EXIT_SIGNAL = "exit_signal"
    STOP_LOSS = "stop_loss"
    TARGET_REACHED = "target_reached"
    CONFIDENCE_DROP = "confidence_drop"
    PORTFOLIO_DEVIATION = "portfolio_deviation"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"

class NotificationSystem:
    """Manages real-time alerts and notifications"""
    
    def __init__(self):
        self.active_alerts = deque(maxlen=1000)  # Keep last 1000 alerts
        self.alert_history = deque(maxlen=5000)  # Keep history
        self.subscribers = {}
        self.alert_thresholds = self._load_alert_thresholds()
        self.is_active_flag = True
        
        # Threading for alert processing
        self.alert_lock = threading.Lock()
        
        # Alert statistics
        self.alert_stats = {
            "total_generated": 0,
            "by_type": {},
            "by_severity": {},
            "last_24h": 0
        }
        
        logger.info("NotificationSystem initialized")
    
    def _load_alert_thresholds(self) -> Dict:
        """Load alert thresholds configuration"""
        return {
            "regime_change": {
                "score_change_threshold": 20,
                "confidence_threshold": 70
            },
            "entry_signal": {
                "score_threshold": 80,
                "confidence_threshold": 75
            },
            "exit_signal": {
                "score_threshold": 20,
                "confidence_threshold": 70
            },
            "stop_loss": {
                "loss_threshold": -0.05,  # 5% loss
                "immediate_alert": True
            },
            "target_reached": {
                "target_percentage": 0.90,  # 90% of target
                "immediate_alert": True
            },
            "confidence_drop": {
                "drop_threshold": 20,  # 20 point drop
                "minimum_confidence": 50
            },
            "portfolio_deviation": {
                "deviation_threshold": 0.10,  # 10% deviation
                "check_frequency": 300  # 5 minutes
            }
        }
    
    def is_active(self) -> bool:
        """Check if notification system is active"""
        return self.is_active_flag
    
    def generate_alert(self, alert_type: AlertType, title: str, message: str,
                      severity: AlertSeverity = AlertSeverity.INFO,
                      metadata: Dict = None, priority: int = 5) -> str:
        """Generate a new alert"""
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_alerts)}"
        
        alert = {
            "id": alert_id,
            "alert_type": alert_type.value,
            "title": title,
            "message": message,
            "severity": severity.value,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False,
            "source": "mlTrainer_system",
            "metadata": metadata or {}
        }
        
        with self.alert_lock:
            self.active_alerts.append(alert)
            self.alert_history.append(alert.copy())
            
            # Update statistics
            self.alert_stats["total_generated"] += 1
            self.alert_stats["by_type"][alert_type.value] = \
                self.alert_stats["by_type"].get(alert_type.value, 0) + 1
            self.alert_stats["by_severity"][severity.value] = \
                self.alert_stats["by_severity"].get(severity.value, 0) + 1
        
        logger.info(f"Generated alert: {alert_type.value} - {title}")
        
        # Trigger immediate processing for critical alerts
        if severity == AlertSeverity.CRITICAL:
            self._process_critical_alert(alert)
        
        return alert_id
    
    def _process_critical_alert(self, alert: Dict):
        """Process critical alerts immediately"""
        try:
            # Log critical alert
            logger.critical(f"CRITICAL ALERT: {alert['title']} - {alert['message']}")
            
            # Add to priority queue (implement if needed)
            # Could trigger immediate notifications, emails, etc.
            
        except Exception as e:
            logger.error(f"Failed to process critical alert: {e}")
    
    def check_regime_change(self, current_regime: Dict, previous_regime: Dict = None) -> Optional[str]:
        """Check for regime change and generate alert if needed"""
        if not previous_regime:
            return None
        
        thresholds = self.alert_thresholds["regime_change"]
        
        # Check score change
        score_change = abs(current_regime.get("regime_score", 50) - 
                          previous_regime.get("regime_score", 50))
        
        # Check regime type change
        regime_type_changed = (current_regime.get("regime_type") != 
                             previous_regime.get("regime_type"))
        
        if score_change >= thresholds["score_change_threshold"] or regime_type_changed:
            title = "Market Regime Transition Detected"
            message = f"Regime changed from {previous_regime.get('regime_type', 'unknown')} " \
                     f"to {current_regime.get('regime_type', 'unknown')}. " \
                     f"Score change: {score_change:.1f} points."
            
            severity = AlertSeverity.WARNING if score_change < 30 else AlertSeverity.CRITICAL
            
            return self.generate_alert(
                AlertType.REGIME_CHANGE,
                title,
                message,
                severity,
                {
                    "previous_regime": previous_regime,
                    "current_regime": current_regime,
                    "score_change": score_change
                },
                priority=8
            )
        
        return None
    
    def check_entry_signal(self, recommendation: Dict) -> Optional[str]:
        """Check for strong entry signals"""
        thresholds = self.alert_thresholds["entry_signal"]
        
        score = recommendation.get("score", 0)
        confidence = recommendation.get("confidence", 0)
        ticker = recommendation.get("ticker", "UNKNOWN")
        
        if (score >= thresholds["score_threshold"] and 
            confidence >= thresholds["confidence_threshold"]):
            
            title = f"Strong Entry Signal - {ticker}"
            message = f"{ticker} showing strong momentum signal with " \
                     f"score {score} and confidence {confidence}%"
            
            return self.generate_alert(
                AlertType.ENTRY_SIGNAL,
                title,
                message,
                AlertSeverity.INFO,
                {"recommendation": recommendation},
                priority=6
            )
        
        return None
    
    def check_exit_signal(self, ticker: str, current_data: Dict, entry_data: Dict) -> Optional[str]:
        """Check for exit signals on holdings"""
        # This would implement exit logic based on technical indicators
        # For now, implement a simple example
        
        current_price = current_data.get("price", 0)
        entry_price = entry_data.get("entry_price", 0)
        
        if entry_price > 0:
            return_pct = (current_price - entry_price) / entry_price
            
            # Example exit conditions
            if return_pct <= -0.10:  # 10% loss
                title = f"Exit Signal Triggered - {ticker}"
                message = f"{ticker} has declined {return_pct*100:.1f}% from entry. Consider exit."
                
                return self.generate_alert(
                    AlertType.EXIT_SIGNAL,
                    title,
                    message,
                    AlertSeverity.WARNING,
                    {
                        "ticker": ticker,
                        "return_pct": return_pct,
                        "current_price": current_price,
                        "entry_price": entry_price
                    },
                    priority=7
                )
        
        return None
    
    def check_stop_loss(self, ticker: str, current_price: float, entry_price: float, 
                       stop_loss_price: float) -> Optional[str]:
        """Check for stop-loss triggers"""
        if current_price <= stop_loss_price:
            loss_pct = (current_price - entry_price) / entry_price * 100
            
            title = f"Stop-Loss Hit - {ticker}"
            message = f"{ticker} hit stop-loss at ${current_price:.2f}. " \
                     f"Loss: {loss_pct:.1f}%"
            
            return self.generate_alert(
                AlertType.STOP_LOSS,
                title,
                message,
                AlertSeverity.CRITICAL,
                {
                    "ticker": ticker,
                    "current_price": current_price,
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss_price,
                    "loss_pct": loss_pct
                },
                priority=10
            )
        
        return None
    
    def check_target_reached(self, ticker: str, current_price: float, 
                           target_price: float, entry_price: float) -> Optional[str]:
        """Check for target achievement"""
        target_threshold = self.alert_thresholds["target_reached"]["target_percentage"]
        progress = (current_price - entry_price) / (target_price - entry_price)
        
        if progress >= target_threshold:
            gain_pct = (current_price - entry_price) / entry_price * 100
            
            title = f"Target Reached - {ticker}"
            message = f"{ticker} reached {progress*100:.0f}% of target. " \
                     f"Current gain: {gain_pct:.1f}%"
            
            return self.generate_alert(
                AlertType.TARGET_REACHED,
                title,
                message,
                AlertSeverity.SUCCESS,
                {
                    "ticker": ticker,
                    "current_price": current_price,
                    "target_price": target_price,
                    "entry_price": entry_price,
                    "progress": progress,
                    "gain_pct": gain_pct
                },
                priority=6
            )
        
        return None
    
    def check_confidence_drop(self, model_name: str, current_confidence: float, 
                            previous_confidence: float) -> Optional[str]:
        """Check for significant confidence drops"""
        thresholds = self.alert_thresholds["confidence_drop"]
        
        confidence_drop = previous_confidence - current_confidence
        
        if (confidence_drop >= thresholds["drop_threshold"] and 
            current_confidence < thresholds["minimum_confidence"]):
            
            title = "Model Confidence Drop Warning"
            message = f"{model_name} confidence dropped {confidence_drop:.1f} points " \
                     f"to {current_confidence:.1f}%"
            
            return self.generate_alert(
                AlertType.CONFIDENCE_DROP,
                title,
                message,
                AlertSeverity.WARNING,
                {
                    "model_name": model_name,
                    "current_confidence": current_confidence,
                    "previous_confidence": previous_confidence,
                    "confidence_drop": confidence_drop
                },
                priority=7
            )
        
        return None
    
    def check_portfolio_deviation(self, current_allocation: Dict, 
                                target_allocation: Dict) -> Optional[str]:
        """Check for portfolio allocation deviation"""
        threshold = self.alert_thresholds["portfolio_deviation"]["deviation_threshold"]
        
        max_deviation = 0
        deviating_assets = []
        
        for asset, target_weight in target_allocation.items():
            current_weight = current_allocation.get(asset, 0)
            deviation = abs(current_weight - target_weight)
            
            if deviation > threshold:
                max_deviation = max(max_deviation, deviation)
                deviating_assets.append({
                    "asset": asset,
                    "target": target_weight,
                    "current": current_weight,
                    "deviation": deviation
                })
        
        if deviating_assets:
            title = "Portfolio Deviation from Optimal Path"
            message = f"Portfolio allocation deviates by up to {max_deviation*100:.1f}% " \
                     f"from target for {len(deviating_assets)} assets"
            
            return self.generate_alert(
                AlertType.PORTFOLIO_DEVIATION,
                title,
                message,
                AlertSeverity.WARNING,
                {
                    "max_deviation": max_deviation,
                    "deviating_assets": deviating_assets,
                    "threshold": threshold
                },
                priority=5
            )
        
        return None
    
    def get_active_alerts(self, limit: int = 50) -> List[Dict]:
        """Get current active alerts"""
        with self.alert_lock:
            # Convert deque to list and apply limit
            alerts = list(self.active_alerts)
            return alerts[-limit:] if limit else alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        with self.alert_lock:
            for alert in self.active_alerts:
                if alert["id"] == alert_id:
                    alert["acknowledged"] = True
                    alert["acknowledged_at"] = datetime.now().isoformat()
                    logger.info(f"Alert acknowledged: {alert_id}")
                    return True
        
        return False
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert (remove from active list)"""
        with self.alert_lock:
            for i, alert in enumerate(self.active_alerts):
                if alert["id"] == alert_id:
                    dismissed_alert = self.active_alerts[i]
                    dismissed_alert["dismissed"] = True
                    dismissed_alert["dismissed_at"] = datetime.now().isoformat()
                    
                    # Move to history only
                    del self.active_alerts[i]
                    logger.info(f"Alert dismissed: {alert_id}")
                    return True
        
        return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics"""
        # Count alerts in last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_count = 0
        
        for alert in self.alert_history:
            try:
                alert_time = datetime.fromisoformat(alert["timestamp"])
                if alert_time > cutoff_time:
                    recent_count += 1
            except:
                continue
        
        self.alert_stats["last_24h"] = recent_count
        
        return {
            "total_active": len(self.active_alerts),
            "total_generated": self.alert_stats["total_generated"],
            "last_24h": recent_count,
            "by_type": self.alert_stats["by_type"].copy(),
            "by_severity": self.alert_stats["by_severity"].copy(),
            "system_active": self.is_active_flag,
            "thresholds": self.alert_thresholds
        }
    
    def send_notification(self, notification_type: str, message: str, metadata: Dict = None):
        """Send general notification (not alert-specific)"""
        # This could be extended to integrate with external notification services
        logger.info(f"Notification [{notification_type}]: {message}")
        
        # Store as info-level alert
        self.generate_alert(
            AlertType.ENTRY_SIGNAL,  # Default type for general notifications
            f"System Notification",
            message,
            AlertSeverity.INFO,
            metadata,
            priority=3
        )
    
    def clear_old_alerts(self, hours: int = 168) -> int:  # Default 1 week
        """Clear alerts older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.alert_lock:
            initial_count = len(self.active_alerts)
            
            # Filter out old alerts
            filtered_alerts = deque(maxlen=1000)
            for alert in self.active_alerts:
                try:
                    alert_time = datetime.fromisoformat(alert["timestamp"])
                    if alert_time > cutoff_time:
                        filtered_alerts.append(alert)
                except:
                    # Keep alerts with invalid timestamps
                    filtered_alerts.append(alert)
            
            self.active_alerts = filtered_alerts
            removed_count = initial_count - len(self.active_alerts)
            
            logger.info(f"Cleared {removed_count} old alerts")
            return removed_count

