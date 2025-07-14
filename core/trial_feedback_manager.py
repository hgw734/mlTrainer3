"""
Trial Feedback Manager for Learning from Execution Results
"""

import json
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TrialFeedback:
    """Represents feedback from a trial execution"""
    trial_id: str
    action_type: str
    parameters: Dict[str, Any]
    outcome: str  # 'success', 'partial_success', 'failure'
    performance_metrics: Dict[str, float]
    execution_time: float
    error_details: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ActionPerformance:
    """Tracks performance of specific actions"""
    action_id: str
    success_rate: float
    avg_execution_time: float
    total_executions: int
    parameter_performance: Dict[str, float]
    best_parameters: Dict[str, Any]
    recent_trend: str  # 'improving', 'stable', 'declining'

class TrialFeedbackManager:
    """Manages trial feedback and learns from execution results."""
    
    def __init__(self, feedback_file: str = "trial_feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_history: List[TrialFeedback] = []
        self.action_performance: Dict[str, ActionPerformance] = {}
        self.parameter_optimization: Dict[str, Dict] = defaultdict(dict)
        self.learning_insights: List[Dict[str, Any]] = []
        
        # Load existing feedback
        self._load_feedback()

    def _load_feedback(self):
        """Load feedback history from file"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r") as f:
                    data = json.load(f)
                
                # Reconstruct feedback objects
                for fb_data in data.get("feedback_history", []):
                    self.feedback_history.append(TrialFeedback(**fb_data))
                
                # Reconstruct performance data
                for action_id, perf_data in data.get("action_performance", {}).items():
                    self.action_performance[action_id] = ActionPerformance(**perf_data)
                
                self.parameter_optimization = data.get("parameter_optimization", {})
                self.learning_insights = data.get("learning_insights", [])
                
                logger.info(f"Loaded {len(self.feedback_history)} feedback entries")
            except Exception as e:
                logger.error(f"Failed to load feedback: {e}")

    def add_feedback(self, feedback: TrialFeedback):
        """Add new feedback from trial execution"""
        self.feedback_history.append(feedback)
        
        # Update action performance
        self._update_action_performance(feedback)
        
        # Optimize parameters based on feedback
        self._optimize_parameters(feedback)
        
        # Generate insights
        insights = self._generate_insights(feedback)
        if insights:
            self.learning_insights.extend(insights)
        
        # Save updated feedback
        self._save_feedback()
        
        logger.info(f"Added feedback for trial {feedback.trial_id}")

    def _update_action_performance(self, feedback: TrialFeedback):
        """Update performance metrics for action"""
        action_id = feedback.action_type
        
        if action_id not in self.action_performance:
            self.action_performance[action_id] = ActionPerformance(
                action_id=action_id,
                success_rate=0.0,
                avg_execution_time=0.0,
                total_executions=0,
                parameter_performance={},
                best_parameters={},
                recent_trend="stable",
            )
        
        perf = self.action_performance[action_id]
        
        # Update metrics
        success_value = 1.0 if feedback.outcome == "success" else 0.5 if feedback.outcome == "partial_success" else 0.0
        perf.success_rate = (perf.success_rate * perf.total_executions + success_value) / (perf.total_executions + 1)
        perf.avg_execution_time = (perf.avg_execution_time * perf.total_executions + feedback.execution_time) / (perf.total_executions + 1)
        perf.total_executions += 1
        
        # Update parameter performance
        param_key = json.dumps(feedback.parameters, sort_keys=True)
        if param_key not in perf.parameter_performance:
            perf.parameter_performance[param_key] = 0.0
        
        perf.parameter_performance[param_key] = (
            perf.parameter_performance[param_key] * 0.8 + success_value * 0.2  # Exponential moving average
        )
        
        # Update best parameters
        if success_value > 0.8 and feedback.execution_time < perf.avg_execution_time * 1.2:
            perf.best_parameters = feedback.parameters
        
        # Calculate trend
        perf.recent_trend = self._calculate_trend(action_id)

    def _calculate_trend(self, action_id: str) -> str:
        """Calculate recent performance trend"""
        recent_feedback = [fb for fb in self.feedback_history[-20:] if fb.action_type == action_id]
        
        if len(recent_feedback) < 5:
            return "stable"
        
        # Calculate success rate for first half and second half
        mid_point = len(recent_feedback) // 2
        first_half_success = sum(1 for fb in recent_feedback[:mid_point] if fb.outcome == "success") / mid_point
        
        second_half_success = sum(1 for fb in recent_feedback[mid_point:] if fb.outcome == "success") / (len(recent_feedback) - mid_point)
        
        if second_half_success > first_half_success * 1.1:
            return "improving"
        elif second_half_success < first_half_success * 0.9:
            return "declining"
        else:
            return "stable"

    def _optimize_parameters(self, feedback: TrialFeedback):
        """Optimize parameters based on feedback"""
        action_id = feedback.action_type
        
        if action_id not in self.parameter_optimization:
            self.parameter_optimization[action_id] = {
                "parameter_ranges": {},
                "optimal_values": {},
                "sensitivity": {}
            }
        
        opt = self.parameter_optimization[action_id]
        
        # Analyze parameter impact
        for param_name, param_value in feedback.parameters.items():
            if param_name not in opt["parameter_ranges"]:
                opt["parameter_ranges"][param_name] = {
                    "min": param_value,
                    "max": param_value,
                    "values": []
                }
            
            # Update ranges
            opt["parameter_ranges"][param_name]["min"] = min(opt["parameter_ranges"][param_name]["min"], param_value)
            opt["parameter_ranges"][param_name]["max"] = max(opt["parameter_ranges"][param_name]["max"], param_value)
            opt["parameter_ranges"][param_name]["values"].append({
                "value": param_value,
                "outcome": feedback.outcome,
                "performance": feedback.performance_metrics.get("primary_metric", 0.0),
            })
            
            # Calculate optimal value (simple approach - can be enhanced)
            if len(opt["parameter_ranges"][param_name]["values"]) >= 5:
                values_perf = [(v["value"], v["performance"]) for v in opt["parameter_ranges"][param_name]["values"]]
                values_perf.sort(key=lambda x: x[1], reverse=True)
                opt["optimal_values"][param_name] = values_perf[0][0]  # Best performing value
                
                # Calculate sensitivity
                if len(set(v[0] for v in values_perf)) > 1:
                    performances = [v[1] for v in values_perf]
                    opt["sensitivity"][param_name] = (
                        np.std(performances) / np.mean(performances) if np.mean(performances) > 0 else 0
                    )

    def _generate_insights(self, feedback: TrialFeedback) -> List[Dict[str, Any]]:
        """Generate insights from feedback"""
        insights = []
        
        # Check for repeated failures
        recent_failures = [
            fb for fb in self.feedback_history[-10:]
            if fb.action_type == feedback.action_type and fb.outcome == "failure"
        ]
        
        if len(recent_failures) >= 3:
            insights.append({
                "type": "repeated_failure",
                "action": feedback.action_type,
                "message": f"Action {feedback.action_type} has failed {len(recent_failures)} times recently",
                "recommendation": "Consider adjusting parameters or using alternative approach",
                "timestamp": datetime.now().isoformat(),
            })
        
        # Check for performance degradation
        if self.action_performance.get(feedback.action_type):
            perf = self.action_performance[feedback.action_type]
            if perf.recent_trend == "declining" and perf.total_executions > 10:
                insights.append({
                    "type": "performance_degradation",
                    "action": feedback.action_type,
                    "message": f"Performance of {feedback.action_type} is declining",
                    "current_success_rate": perf.success_rate,
                    "recommendation": "Review recent changes and consider reverting",
                    "timestamp": datetime.now().isoformat(),
                })
        
        # Check for parameter optimization opportunities
        if feedback.action_type in self.parameter_optimization:
            opt = self.parameter_optimization[feedback.action_type]
            sensitive_params = [
                param for param, sensitivity in opt.get("sensitivity", {}).items()
                if sensitivity > 0.3
            ]
            
            if sensitive_params:
                insights.append({
                    "type": "parameter_sensitivity",
                    "action": feedback.action_type,
                    "message": f"Parameters {sensitive_params} significantly impact performance",
                    "optimal_values": {param: opt["optimal_values"].get(param) for param in sensitive_params},
                    "timestamp": datetime.now().isoformat(),
                })
        
        return insights

    def get_recommendations(self, action_type: str) -> Dict[str, Any]:
        """Get recommendations for action execution"""
        recommendations = {
            "action_type": action_type,
            "recommended_parameters": {},
            "warnings": [],
            "insights": []
        }
        
        # Get performance data
        if action_type in self.action_performance:
            perf = self.action_performance[action_type]
            
            # Recommend best parameters
            if perf.best_parameters:
                recommendations["recommended_parameters"] = perf.best_parameters
            
            # Add warnings
            if perf.success_rate < 0.5:
                recommendations["warnings"].append(f"Low success rate ({perf.success_rate:.2%}) for {action_type}")
            
            if perf.recent_trend == "declining":
                recommendations["warnings"].append(f"Performance is declining for {action_type}")
            
            # Get optimal parameters
            if action_type in self.parameter_optimization:
                opt = self.parameter_optimization[action_type]
                recommendations["recommended_parameters"].update(opt.get("optimal_values", {}))
            
            # Add relevant insights
            relevant_insights = [
                insight for insight in self.learning_insights[-10:]
                if insight.get("action") == action_type
            ]
            recommendations["insights"] = relevant_insights
        
        return recommendations

    def get_action_report(self, action_type: str) -> Dict[str, Any]:
        """Generate detailed report for an action"""
        if action_type not in self.action_performance:
            return {"error": f"No data for action {action_type}"}
        
        perf = self.action_performance[action_type]
        
        # Get feedback history
        action_feedback = [fb for fb in self.feedback_history if fb.action_type == action_type]
        
        # Calculate statistics
        success_count = sum(1 for fb in action_feedback if fb.outcome == "success")
        failure_count = sum(1 for fb in action_feedback if fb.outcome == "failure")
        
        # Parameter analysis
        param_stats = {}
        if action_type in self.parameter_optimization:
            opt = self.parameter_optimization[action_type]
            for param, data in opt["parameter_ranges"].items():
                param_stats[param] = {
                    "range": [data["min"], data["max"]],
                    "optimal": opt["optimal_values"].get(param),
                    "sensitivity": opt["sensitivity"].get(param, 0.0),
                }
        
        return {
            "action_type": action_type,
            "performance": {
                "success_rate": perf.success_rate,
                "avg_execution_time": perf.avg_execution_time,
                "total_executions": perf.total_executions,
                "recent_trend": perf.recent_trend,
            },
            "statistics": {
                "successes": success_count,
                "failures": failure_count,
                "partial_successes": perf.total_executions - success_count - failure_count,
            },
            "parameters": param_stats,
            "best_parameters": perf.best_parameters,
            "recent_feedback": [
                {
                    "trial_id": fb.trial_id,
                    "outcome": fb.outcome,
                    "execution_time": fb.execution_time,
                    "timestamp": fb.timestamp,
                }
                for fb in action_feedback[-5:]
            ],
        }

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning insights"""
        # Group insights by type
        insights_by_type = defaultdict(list)
        for insight in self.learning_insights:
            insights_by_type[insight["type"]].append(insight)
        
        # Get top performing actions
        top_actions = sorted(
            self.action_performance.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )[:5]
        
        # Get actions needing attention
        problematic_actions = [
            (action_id, perf)
            for action_id, perf in self.action_performance.items()
            if perf.success_rate < 0.5 or perf.recent_trend == "declining"
        ]
        
        return {
            "total_trials": len(self.feedback_history),
            "total_insights": len(self.learning_insights),
            "insights_by_type": dict(insights_by_type),
            "top_performing_actions": [
                {
                    "action": action_id,
                    "success_rate": perf.success_rate,
                    "executions": perf.total_executions
                }
                for action_id, perf in top_actions
            ],
            "actions_needing_attention": [
                {
                    "action": action_id,
                    "success_rate": perf.success_rate,
                    "trend": perf.recent_trend,
                    "issue": "low_success" if perf.success_rate < 0.5 else "declining_performance",
                }
                for action_id, perf in problematic_actions
            ],
            "parameter_optimization_available": list(self.parameter_optimization.keys()),
        }

    def _save_feedback(self):
        """Save feedback data to file"""
        try:
            data = {
                "feedback_history": [
                    {
                        "trial_id": fb.trial_id,
                        "action_type": fb.action_type,
                        "parameters": fb.parameters,
                        "outcome": fb.outcome,
                        "performance_metrics": fb.performance_metrics,
                        "execution_time": fb.execution_time,
                        "error_details": fb.error_details,
                        "context": fb.context,
                        "timestamp": fb.timestamp,
                    }
                    for fb in self.feedback_history
                ],
                "action_performance": {
                    action_id: {
                        "action_id": perf.action_id,
                        "success_rate": perf.success_rate,
                        "avg_execution_time": perf.avg_execution_time,
                        "total_executions": perf.total_executions,
                        "parameter_performance": perf.parameter_performance,
                        "best_parameters": perf.best_parameters,
                        "recent_trend": perf.recent_trend,
                    }
                    for action_id, perf in self.action_performance.items()
                },
                "parameter_optimization": self.parameter_optimization,
                "learning_insights": self.learning_insights,
            }
            
            with open(self.feedback_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")

# Singleton instance
_feedback_manager = None

def get_trial_feedback_manager() -> TrialFeedbackManager:
    """Get the trial feedback manager instance"""
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = TrialFeedbackManager()
    return _feedback_manager 