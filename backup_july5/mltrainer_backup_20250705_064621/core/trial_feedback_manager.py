"""
Trial Feedback Manager - Real-time Progress Communication
=======================================================

Purpose: Provides real-time feedback to mlTrainer during dynamic code generation
and trial execution, preventing timeouts and keeping the AI informed of progress.

Features:
- Real-time status updates during code generation
- Progress tracking for long-running operations
- Timeout prevention through active communication
- Structured feedback for mlTrainer understanding
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class TrialFeedbackManager:
    """
    Manages real-time feedback during trial execution and code generation
    """
    
    def __init__(self):
        self.active_operations = {}
        self.feedback_callbacks = []
        self.status_updates = []
        self.operation_lock = threading.Lock()
        
        logger.info("TrialFeedbackManager initialized - ready for progress tracking")
    
    def start_operation(self, operation_id: str, operation_type: str, description: str) -> None:
        """
        Start tracking a new operation with progress feedback
        """
        with self.operation_lock:
            self.active_operations[operation_id] = {
                "type": operation_type,
                "description": description,
                "start_time": datetime.now(),
                "status": "starting",
                "progress": 0,
                "steps": [],
                "feedback_sent": []
            }
        
        self._send_feedback(operation_id, "operation_started", {
            "message": f"Starting {operation_type}: {description}",
            "estimated_duration": self._estimate_duration(operation_type),
            "status": "in_progress"
        })
    
    def update_progress(self, operation_id: str, step: str, progress: int, details: Optional[str] = None):
        """
        Update progress of an ongoing operation
        """
        if operation_id not in self.active_operations:
            return
        
        with self.operation_lock:
            operation = self.active_operations[operation_id]
            operation["status"] = "in_progress"
            operation["progress"] = progress
            operation["steps"].append({
                "step": step,
                "timestamp": datetime.now(),
                "details": details
            })
        
        # Send progress feedback to prevent timeout
        self._send_feedback(operation_id, "progress_update", {
            "step": step,
            "progress": progress,
            "details": details,
            "message": f"Progress: {progress}% - {step}"
        })
    
    def complete_operation(self, operation_id: str, success: bool, result: Dict[str, Any]):
        """
        Mark operation as complete with results
        """
        if operation_id not in self.active_operations:
            return
        
        with self.operation_lock:
            operation = self.active_operations[operation_id]
            operation["status"] = "completed" if success else "failed"
            operation["progress"] = 100
            operation["end_time"] = datetime.now()
            operation["result"] = result
        
        self._send_feedback(operation_id, "operation_completed", {
            "success": success,
            "result": result,
            "duration": self._calculate_duration(operation_id),
            "message": f"Operation {'completed successfully' if success else 'failed'}"
        })
    
    def send_code_generation_feedback(self, operation_id: str, action: str, stage: str):
        """
        Send specific feedback during dynamic code generation
        """
        stage_messages = {
            "analyzing_action": f"Analyzing new action '{action}' - determining execution pattern",
            "categorizing": f"Categorizing '{action}' for appropriate API template",
            "generating_method": f"Generating execution method for '{action}'",
            "creating_api_calls": f"Creating API call patterns for '{action}'",
            "implementing_fallback": f"Implementing intelligent fallback for '{action}'",
            "testing_method": f"Testing generated method for '{action}'",
            "integrating": f"Integrating new '{action}' capability into executor"
        }
        
        message = stage_messages.get(stage, f"Processing {stage} for {action}")
        
        self.update_progress(operation_id, stage, self._get_stage_progress(stage), message)
    
    def send_mltrainer_keepalive(self, operation_id: str, context: str):
        """
        Send keepalive message to mlTrainer to prevent timeout
        """
        self._send_feedback(operation_id, "keepalive", {
            "message": f"Working on {context} - please wait",
            "timestamp": datetime.now().isoformat(),
            "status": "actively_processing"
        })
    
    def _send_feedback(self, operation_id: str, feedback_type: str, data: Dict[str, Any]):
        """
        Send feedback through registered callbacks
        """
        feedback = {
            "operation_id": operation_id,
            "type": feedback_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Store feedback
        self.status_updates.append(feedback)
        
        # Limit status updates to last 100
        if len(self.status_updates) > 100:
            self.status_updates = self.status_updates[-100:]
        
        # Send to callbacks
        for callback in self.feedback_callbacks:
            try:
                callback(feedback)
            except Exception as e:
                logger.error(f"Feedback callback failed: {e}")
    
    def _estimate_duration(self, operation_type: str) -> str:
        """
        Estimate operation duration for user expectations
        """
        duration_estimates = {
            "dynamic_code_generation": "30-60 seconds",
            "api_endpoint_creation": "20-40 seconds", 
            "method_integration": "10-20 seconds",
            "trial_execution": "15-45 seconds",
            "data_processing": "10-30 seconds"
        }
        return duration_estimates.get(operation_type, "30-60 seconds")
    
    def _get_stage_progress(self, stage: str) -> int:
        """
        Get progress percentage for different stages
        """
        stage_progress = {
            "analyzing_action": 10,
            "categorizing": 20,
            "generating_method": 40,
            "creating_api_calls": 60,
            "implementing_fallback": 80,
            "testing_method": 90,
            "integrating": 95
        }
        return stage_progress.get(stage, 50)
    
    def _calculate_duration(self, operation_id: str) -> float:
        """
        Calculate operation duration in seconds
        """
        operation = self.active_operations.get(operation_id)
        if operation and "end_time" in operation:
            delta = operation["end_time"] - operation["start_time"]
            return delta.total_seconds()
        return 0.0
    
    def register_feedback_callback(self, callback: Callable):
        """
        Register a callback for receiving feedback updates
        """
        self.feedback_callbacks.append(callback)
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of an operation
        """
        return self.active_operations.get(operation_id)
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent feedback messages
        """
        return self.status_updates[-limit:] if self.status_updates else []

# Global feedback manager instance
_feedback_manager = None

def get_feedback_manager():
    """Get global feedback manager instance"""
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = TrialFeedbackManager()
    return _feedback_manager

class ProgressTracker:
    """
    Context manager for tracking operation progress
    """
    
    def __init__(self, operation_type: str, description: str):
        self.operation_id = f"{operation_type}_{int(time.time())}"
        self.operation_type = operation_type
        self.description = description
        self.feedback_manager = get_feedback_manager()
    
    def __enter__(self):
        self.feedback_manager.start_operation(
            self.operation_id, 
            self.operation_type, 
            self.description
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        result = {"success": success}
        if exc_type:
            result["error"] = str(exc_val)
        
        self.feedback_manager.complete_operation(
            self.operation_id, 
            success, 
            result
        )
    
    def update(self, step: str, progress: int, details: str = None):
        """Update progress within context"""
        self.feedback_manager.update_progress(
            self.operation_id, 
            step, 
            progress, 
            details
        )
    
    def code_generation_update(self, action: str, stage: str):
        """Specific update for code generation"""
        self.feedback_manager.send_code_generation_feedback(
            self.operation_id, 
            action, 
            stage
        )
    
    def keepalive(self, context: str):
        """Send keepalive to prevent timeout"""
        self.feedback_manager.send_mltrainer_keepalive(
            self.operation_id, 
            context
        )