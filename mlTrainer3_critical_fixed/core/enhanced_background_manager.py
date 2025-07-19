"""
Enhanced Background Trial Manager
Manages autonomous trial execution with compliance enforcement
"""

import asyncio
import json
import logging
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os

from core.unified_executor import get_unified_executor
from config.immutable_compliance_gateway import ComplianceGateway
from goal_system import GoalSystem

logger = logging.getLogger(__name__)


@dataclass
class TrialState:
    """Tracks the state of a background trial"""

    trial_id: str
    total_steps: int
    goal_context: Dict[str, Any]
    status: str = "pending"
    steps_completed: int = 0
    current_action: Optional[str] = None
    results: List[Dict] = field(default_factory=list)
    compliance_checks: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())


class EnhancedBackgroundTrialManager:
    """
    Enhanced background trial manager with compliance enforcement

    Features:
        - Runs trials autonomously
        - Enforces compliance at each step
        - Supports mlTrainer ↔ ML Agent loops
        - Provides real-time progress tracking
    """

    def __init__(self, ai_client=None):
        self.executor = get_unified_executor()
        self.ai_client = ai_client
        self.goal_system = GoalSystem()
        self.compliance_gateway = ComplianceGateway()

        # Trial tracking
        self.active_trials: Dict[str, TrialState] = {}
        self.trial_history: List[Dict] = []
        self._lock = threading.Lock()

        # Async event loop
        self._loop = None
        self._thread = None

        logger.info("Enhanced Background Trial Manager initialized")

    def start_trial(self, mltrainer_response: str, auto_approve: bool = False) -> str:
        """Start a new background trial from mlTrainer response"""
        # Parse executable actions
        parsed = self.executor.parse_mltrainer_response(mltrainer_response)

        if not parsed["executable"]:
            return None

        # Generate trial ID
        trial_id = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create trial state
        trial_state = TrialState(
            trial_id=trial_id,
            total_steps=len(parsed["actions"]) + (1 if parsed["model_request"]["model_id"] else 0),
            goal_context=self.goal_system.get_current_goal(),
        )

        # Compliance pre-check
        compliance_result = self._pre_trial_compliance_check(parsed, trial_state)
        trial_state.compliance_checks.append(compliance_result)

        if not compliance_result["passed"]:
            trial_state.status = "blocked_compliance"
            trial_state.error = compliance_result["reason"]
            self._save_trial_state(trial_state)
            return None

        # Register trial
        with self._lock:
            self.active_trials[trial_id] = trial_state

        # Start execution
        if auto_approve:
            self._start_async_execution(trial_id, parsed)
        else:
            trial_state.status = "pending_approval"
            self._save_trial_state(trial_state)

        logger.info(f"Started trial {trial_id} with {trial_state.total_steps} steps")
        return trial_id

    def approve_trial(self, trial_id: str) -> bool:
        """Approve a pending trial for execution"""
        with self._lock:
            trial_state = self.active_trials.get(trial_id)

            if not trial_state or trial_state.status != "pending_approval":
                return False

            # Re-parse the original response (would be stored in real implementation)
            # For now, we'll mark it as approved and ready
            trial_state.status = "approved"
            trial_state.last_update = datetime.now().isoformat()
            self._save_trial_state(trial_state)

            # Would trigger execution here
            logger.info(f"Trial {trial_id} approved for execution")
            return True

    def execute_trial_step(self, trial_id: str, action: str, params: Dict) -> Dict[str, Any]:
        """Execute a single trial step with compliance"""
        with self._lock:
            trial_state = self.active_trials.get(trial_id)

            if not trial_state:
                return {"success": False, "error": "Trial not found"}

            # Update trial state
            trial_state.current_action = action
            trial_state.status = "executing"
            trial_state.last_update = datetime.now().isoformat()

            # Compliance check for this step
            compliance_result = self._step_compliance_check(action, params, trial_state)
            trial_state.compliance_checks.append(compliance_result)

            if not compliance_result["passed"]:
                trial_state.status = "blocked_compliance"
                trial_state.error = compliance_result["reason"]
                self._save_trial_state(trial_state)
                return {"success": False, "error": compliance_result["reason"]}

            try:
                # Route to appropriate handler
                if action.startswith("train_"):
                    model_id = action.replace("train_", "")
                    result = self.executor.execute_ml_model_training(model_id, **params)
                elif action.startswith("calculate_"):
                    model_id = action.replace("calculate_", "")
                    result = self.executor.execute_financial_model(model_id, **params)
                elif action in self.executor.registered_actions:
                    result = self.executor.registered_actions[action]["function"](**params)
                else:
                    result = {"success": False, "error": f"Unknown action: {action}"}

                # Record result
                step_result = {
                    "action": action,
                    "params": params,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                    "compliance_status": "approved",
                }
                trial_state.results.append(step_result)
                trial_state.steps_completed += 1

                # Update status
                if trial_state.steps_completed >= trial_state.total_steps:
                    trial_state.status = "completed"
                else:
                    trial_state.status = "in_progress"

                self._save_trial_state(trial_state)

                # If using AI client, get next steps
                if self.ai_client and result["success"]:
                    self._trigger_ai_feedback(trial_id, step_result)

                return result

            except Exception as e:
                logger.error(f"Trial step execution failed: {e}")
                trial_state.status = "failed"
                trial_state.error = str(e)
                self._save_trial_state(trial_state)
                return {"success": False, "error": str(e)}

    def get_trial_status(self, trial_id: str) -> Dict[str, Any]:
        """Get current status of a trial"""
        with self._lock:
            trial_state = self.active_trials.get(trial_id)

            if not trial_state:
                # Check history
                for trial in self.trial_history:
                    if trial["trial_id"] == trial_id:
                        return trial
                return {"error": "Trial not found"}

            return {
                "trial_id": trial_state.trial_id,
                "status": trial_state.status,
                "progress": {
                    "completed": trial_state.steps_completed,
                    "total": trial_state.total_steps,
                    "percentage": (
                        (trial_state.steps_completed / trial_state.total_steps * 100)
                        if trial_state.total_steps > 0 else 0
                    ),
                },
                "current_action": trial_state.current_action,
                "created_at": trial_state.created_at,
                "last_update": trial_state.last_update,
                "compliance_checks": len(trial_state.compliance_checks),
                "results_count": len(trial_state.results),
                "error": trial_state.error,
            }

    def get_all_trials(self) -> List[Dict[str, Any]]:
        """Get status of all trials"""
        all_trials = []

        # Active trials
        with self._lock:
            for trial_id in self.active_trials:
                all_trials.append(self.get_trial_status(trial_id))

        # Historical trials
        all_trials.extend(self.trial_history)

        # Sort by creation time
        all_trials.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return all_trials

    def cancel_trial(self, trial_id: str) -> bool:
        """Cancel an active trial"""
        with self._lock:
            trial_state = self.active_trials.get(trial_id)

            if not trial_state or trial_state.status in ["completed", "failed", "cancelled"]:
                return False

            trial_state.status = "cancelled"
            trial_state.last_update = datetime.now().isoformat()
            self._save_trial_state(trial_state)

            logger.info(f"Trial {trial_id} cancelled")
            return True

    def _pre_trial_compliance_check(self, parsed: Dict, trial_state: TrialState) -> Dict[str, Any]:
        """Pre-trial compliance verification"""
        try:
            # Check for prohibited actions
            prohibited_actions = ["delete", "remove", "drop", "truncate"]
            for action in parsed["actions"]:
                if any(term in action.lower() for term in prohibited_actions):
                    return {
                        "passed": False,
                        "reason": f"Prohibited action detected: {action}",
                        "timestamp": datetime.now().isoformat(),
                    }

            # Check data sources
            if "data_sources" in parsed:
                for source in parsed["data_sources"]:
                    if not self.compliance_gateway.verify_data_source(source, ""):
                        return {
                            "passed": False,
                            "reason": f"Unauthorized data source: {source}",
                            "timestamp": datetime.now().isoformat(),
                        }

            return {
                "passed": True,
                "reason": "Pre-trial compliance check passed",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "passed": False,
                "reason": f"Compliance check failed: {e}",
                "timestamp": datetime.now().isoformat(),
            }

    def _step_compliance_check(self, action: str, params: Dict, trial_state: TrialState) -> Dict[str, Any]:
        """Step-level compliance verification"""
        try:
            # Check for dangerous parameters
            dangerous_params = ["password", "secret", "key", "token"]
            for param_name in params:
                if any(term in param_name.lower() for term in dangerous_params):
                    return {
                        "passed": False,
                        "reason": f"Dangerous parameter detected: {param_name}",
                        "timestamp": datetime.now().isoformat(),
                    }

            # Check action safety
            if action.startswith("system.") or action.startswith("os."):
                return {
                    "passed": False,
                    "reason": f"System-level action prohibited: {action}",
                    "timestamp": datetime.now().isoformat(),
                }

            return {
                "passed": True,
                "reason": "Step compliance check passed",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "passed": False,
                "reason": f"Step compliance check failed: {e}",
                "timestamp": datetime.now().isoformat(),
            }

    def _start_async_execution(self, trial_id: str, parsed: Dict):
        """Start asynchronous trial execution"""
        if not self._loop:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self._thread.start()

        # Schedule trial execution
        asyncio.run_coroutine_threadsafe(
            self._execute_trial_async(trial_id, parsed), self._loop
        )

    async def _execute_trial_async(self, trial_id: str, parsed: Dict):
        """Execute trial asynchronously"""
        try:
            for action in parsed["actions"]:
                # Execute each action
                result = self.execute_trial_step(trial_id, action, {})
                
                if not result["success"]:
                    logger.error(f"Trial {trial_id} failed at action {action}")
                    break

                # Brief pause between actions
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Async trial execution failed: {e}")

    def _run_async_loop(self):
        """Run the async event loop in a separate thread"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _trigger_ai_feedback(self, trial_id: str, step_result: Dict):
        """Trigger AI feedback for trial step"""
        if not self.ai_client:
            return

        try:
            # Prepare feedback context
            context = {
                "trial_id": trial_id,
                "step_result": step_result,
                "goal": self.goal_system.get_current_goal(),
            }

            # Get AI feedback
            feedback = self.ai_client.get_feedback(context)
            
            # Process feedback (would integrate with trial execution)
            logger.info(f"AI feedback for trial {trial_id}: {feedback}")

        except Exception as e:
            logger.error(f"Failed to get AI feedback: {e}")

    def _save_trial_state(self, trial_state: TrialState):
        """Save trial state to persistent storage"""
        try:
            # Convert to serializable format
            state_dict = {
                "trial_id": trial_state.trial_id,
                "status": trial_state.status,
                "steps_completed": trial_state.steps_completed,
                "total_steps": trial_state.total_steps,
                "current_action": trial_state.current_action,
                "results": trial_state.results,
                "compliance_checks": trial_state.compliance_checks,
                "error": trial_state.error,
                "created_at": trial_state.created_at,
                "last_update": trial_state.last_update,
            }

            # Save to file (in real implementation, would use database)
            filename = f"trials/{trial_state.trial_id}.json"
            os.makedirs("trials", exist_ok=True)
            
            with open(filename, "w") as f:
                json.dump(state_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save trial state: {e}")

    def _archive_trial(self, trial_id: str):
        """Archive completed trial to history"""
        with self._lock:
            trial_state = self.active_trials.get(trial_id)
            if trial_state:
                # Convert to dict for archiving
                trial_dict = {
                    "trial_id": trial_state.trial_id,
                    "status": trial_state.status,
                    "steps_completed": trial_state.steps_completed,
                    "total_steps": trial_state.total_steps,
                    "results": trial_state.results,
                    "compliance_checks": trial_state.compliance_checks,
                    "error": trial_state.error,
                    "created_at": trial_state.created_at,
                    "last_update": trial_state.last_update,
                }
                
                self.trial_history.append(trial_dict)
                del self.active_trials[trial_id]

    def _get_approved_symbols(self) -> List[str]:
        """Get list of approved trading symbols"""
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


# Global instance
_enhanced_manager = None


def get_enhanced_background_manager(ai_client=None) -> EnhancedBackgroundTrialManager:
    """Get the enhanced background manager instance"""
    global _enhanced_manager
    if _enhanced_manager is None:
        _enhanced_manager = EnhancedBackgroundTrialManager(ai_client)
    return _enhanced_manager
