"""
Background Trial Manager - Autonomous ML Trial Execution
======================================================

Purpose: Manages autonomous communication between mlTrainer AI and ML system
during trial execution, keeping the chat interface clean while allowing
real-time feedback and parameter adjustments in the background.
"""

import asyncio
import logging
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import time
import requests

logger = logging.getLogger(__name__)

@dataclass
class TrialStep:
    step_id: str
    action: str
    status: str  # pending, running, completed, failed
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    mltrainer_analysis: Optional[str] = None
    next_suggested_action: Optional[str] = None

@dataclass
class TrialSession:
    session_id: str
    objective: str
    start_time: datetime
    status: str  # active, paused, completed, failed
    steps: List[TrialStep]
    current_step: int
    user_notifications: List[str]
    final_results: Optional[Dict[str, Any]] = None

class BackgroundTrialManager:
    """
    Manages autonomous trial execution with background mlTrainer communication
    """
    
    def __init__(self, ml_executor, ai_client):
        self.ml_executor = ml_executor
        self.ai_client = ai_client
        self.active_trials = {}
        self.trial_lock = threading.Lock()
        self.update_callbacks = []
        self.background_thread = None
        self.running = False
        
        logger.info("BackgroundTrialManager initialized")
    
    def register_update_callback(self, callback: Callable):
        """Register callback for trial updates (for UI notifications)"""
        self.update_callbacks.append(callback)
    
    def start_background_trial(self, 
                              objective: str, 
                              initial_action: str,
                              user_id: str = "default") -> str:
        """
        Start a background trial with autonomous mlTrainer communication
        
        Args:
            objective: High-level trial objective (e.g., "Find 7-day momentum stocks")
            initial_action: First ML action to execute
            user_id: User identifier for session management
            
        Returns:
            Trial session ID
        """
        session_id = f"trial_{int(time.time())}"
        
        trial_session = TrialSession(
            session_id=session_id,
            objective=objective,
            start_time=datetime.now(),
            status="active",
            steps=[],
            current_step=0,
            user_notifications=[f"🚀 Background trial started: {objective}"]
        )
        
        with self.trial_lock:
            self.active_trials[session_id] = trial_session
        
        # Start background execution thread
        if not self.running:
            self.running = True
            self.background_thread = threading.Thread(
                target=self._background_execution_loop,
                daemon=True
            )
            self.background_thread.start()
        
        # Queue initial action
        self._queue_trial_step(session_id, initial_action, {
            "objective": objective,
            "initial_request": True
        })
        
        self._notify_updates(f"Trial {session_id} started in background")
        
        return session_id
    
    def _queue_trial_step(self, session_id: str, action: str, input_data: Dict[str, Any]):
        """Queue a new step for trial execution"""
        step = TrialStep(
            step_id=f"{session_id}_step_{len(self.active_trials[session_id].steps)}",
            action=action,
            status="pending",
            timestamp=datetime.now(),
            input_data=input_data
        )
        
        with self.trial_lock:
            self.active_trials[session_id].steps.append(step)
    
    def _background_execution_loop(self):
        """Main background loop for trial execution"""
        while self.running:
            try:
                # Process all active trials
                active_session_ids = list(self.active_trials.keys())
                
                for session_id in active_session_ids:
                    if session_id in self.active_trials:
                        self._process_trial_session(session_id)
                
                # Sleep between iterations
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Background execution error: {e}")
                time.sleep(5)
    
    def _process_trial_session(self, session_id: str):
        """Process next pending step in a trial session"""
        with self.trial_lock:
            session = self.active_trials.get(session_id)
            if not session or session.status != "active":
                return
            
            # Validate data quality before executing any ML trial steps
            if self._should_validate_data_quality(session):
                quality_check = self._validate_trial_data_quality(session)
                if not quality_check["approved"]:
                    self._handle_data_quality_failure(session_id, quality_check)
                    return
        
        # Find next pending step
        pending_steps = [s for s in session.steps if s.status == "pending"]
        if not pending_steps:
            return
        
        current_step = pending_steps[0]
        
        # Mark as running
        current_step.status = "running"
        self._notify_updates(f"Executing: {current_step.action}")
        
        try:
            # Execute ML action
            if current_step.action == "momentum_screening":
                result = self.ml_executor.execute_momentum_screening()
            elif current_step.action == "regime_detection":
                result = self.ml_executor.execute_regime_detection()
            else:
                result = {"success": False, "error": f"Unknown action: {current_step.action}"}
            
            current_step.output_data = result
            
            if result.get("success"):
                # Get mlTrainer analysis of results
                analysis_response = self._get_mltrainer_analysis(
                    session.objective,
                    current_step.action,
                    result
                )
                
                current_step.mltrainer_analysis = analysis_response["analysis"]
                current_step.next_suggested_action = analysis_response.get("next_action")
                current_step.status = "completed"
                
                # Queue next action if suggested
                if analysis_response.get("next_action"):
                    self._queue_trial_step(
                        session_id,
                        analysis_response["next_action"],
                        {"previous_results": result}
                    )
                
                # Notify user of progress
                session.user_notifications.append(
                    f"✅ {current_step.action} completed - {analysis_response['summary']}"
                )
                
            else:
                current_step.status = "failed"
                session.user_notifications.append(
                    f"❌ {current_step.action} failed: {result.get('error', 'Unknown error')}"
                )
        
        except Exception as e:
            current_step.status = "failed"
            current_step.output_data = {"error": str(e)}
            logger.error(f"Step execution failed: {e}")
        
        # Check if trial should continue or conclude
        self._evaluate_trial_completion(session_id)
    
    def _get_mltrainer_analysis(self, objective: str, action: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get mlTrainer's analysis of ML results and next action suggestion"""
        
        analysis_prompt = f"""
AUTONOMOUS TRIAL ANALYSIS - BACKGROUND MODE - FULL COMMAND AUTHORITY

**IMPORTANT**: You are in autonomous background trial mode. You have FULL COMMAND AUTHORITY over the ML system.
Any action you suggest will be AUTOMATICALLY EXECUTED without user approval.

Objective: {objective}
Completed Action: {action}
Results: {json.dumps(results, indent=2)}

Provide concise analysis and commanding next action:

1. ANALYSIS: What do these results tell us? (2-3 sentences max)
2. NEXT_ACTION: Command the next specific action to execute:
   - momentum_screening (if need more data)
   - regime_detection (if need market context) 
   - deep_analysis (if ready to analyze top candidates)
   - parameter_adjustment (if need to tune models)
   - model_selection (if need different models)
   - conclude_trial (if objective achieved)

3. SUMMARY: One-sentence progress update for user notification

**Your commands will be executed immediately. Use imperative language like "execute", "run", "initiate".**

Respond in JSON format:
{
  "analysis": "Brief analysis text",
  "next_action": "specific_action_name", 
  "summary": "Progress summary for user"
}
"""
        
        try:
            response = self.ai_client.chat_completion([
                {"role": "user", "content": analysis_prompt}
            ])
            
            # In background trial mode, mlTrainer has full command authority
            # Parse the response using trial_mode=True to enable all command patterns
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback if JSON parsing fails
            return {
                "analysis": "Analysis completed successfully",
                "next_action": None,
                "summary": "Step completed"
            }
            
        except Exception as e:
            logger.error(f"mlTrainer analysis failed: {e}")
            return {
                "analysis": f"Analysis error: {e}",
                "next_action": None,
                "summary": "Analysis completed with errors"
            }
    
    def _evaluate_trial_completion(self, session_id: str):
        """Evaluate if trial should be concluded"""
        session = self.active_trials[session_id]
        
        # Check completion conditions
        completed_steps = [s for s in session.steps if s.status == "completed"]
        failed_steps = [s for s in session.steps if s.status == "failed"]
        pending_steps = [s for s in session.steps if s.status == "pending"]
        
        # Conclude if no pending steps and enough completed
        if not pending_steps and len(completed_steps) >= 1:
            session.status = "completed"
            session.final_results = self._compile_final_results(session)
            session.user_notifications.append(
                f"🎯 Trial completed! {len(completed_steps)} successful steps."
            )
            self._notify_updates(f"Trial {session_id} completed successfully")
        
        # Fail if too many failures
        elif len(failed_steps) >= 3:
            session.status = "failed"
            session.user_notifications.append(
                f"❌ Trial failed after {len(failed_steps)} failed attempts"
            )
    
    def _compile_final_results(self, session: TrialSession) -> Dict[str, Any]:
        """Compile final results from completed trial"""
        successful_steps = [s for s in session.steps if s.status == "completed"]
        
        return {
            "objective": session.objective,
            "total_steps": len(session.steps),
            "successful_steps": len(successful_steps),
            "duration_minutes": (datetime.now() - session.start_time).total_seconds() / 60,
            "key_findings": [s.mltrainer_analysis for s in successful_steps if s.mltrainer_analysis],
            "final_recommendations": successful_steps[-1].mltrainer_analysis if successful_steps else None
        }
    
    def _notify_updates(self, message: str):
        """Notify registered callbacks of trial updates"""
        for callback in self.update_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Update callback failed: {e}")
    
    def get_trial_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a trial"""
        with self.trial_lock:
            session = self.active_trials.get(session_id)
            if not session:
                return None
            
            return {
                "session_id": session_id,
                "objective": session.objective,
                "status": session.status,
                "total_steps": len(session.steps),
                "completed_steps": len([s for s in session.steps if s.status == "completed"]),
                "current_step": session.steps[-1].action if session.steps else None,
                "notifications": session.user_notifications,
                "final_results": session.final_results
            }
    
    def get_trial_notifications(self, session_id: str) -> List[str]:
        """Get new notifications for a trial (and clear them)"""
        with self.trial_lock:
            session = self.active_trials.get(session_id)
            if not session:
                return []
            
            notifications = session.user_notifications.copy()
            session.user_notifications.clear()
            return notifications
    
    def pause_trial(self, session_id: str):
        """Pause an active trial"""
        with self.trial_lock:
            session = self.active_trials.get(session_id)
            if session and session.status == "active":
                session.status = "paused"
                logger.info(f"Trial {session_id} paused")
    
    def resume_trial(self, session_id: str):
        """Resume a paused trial"""
        with self.trial_lock:
            session = self.active_trials.get(session_id)
            if session and session.status == "paused":
                session.status = "active"
                logger.info(f"Trial {session_id} resumed")
    
    def stop_background_execution(self):
        """Stop background execution thread"""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
        logger.info("Background trial execution stopped")
    
    def _should_validate_data_quality(self, session: TrialSession) -> bool:
        """Determine if data quality validation is needed for this session"""
        # Validate before first ML step and periodically during execution
        ml_steps = [s for s in session.steps if "model" in s.action.lower() or "momentum" in s.action.lower()]
        return len(ml_steps) == 0 or len(session.steps) % 5 == 0  # Every 5th step
    
    def _validate_trial_data_quality(self, session: TrialSession) -> Dict[str, Any]:
        """Validate data quality for trial execution"""
        try:
            # Extract symbols from session objective or use defaults
            symbols = self._extract_symbols_from_objective(session.objective)
            if not symbols:
                symbols = ['SPY']  # Default to SPY for validation
            
            # Call data quality validation API
            response = requests.post(
                "http://localhost:8000/api/data-quality/trial-validation",
                json={"symbols": symbols},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "approved": result.get("trial_approved", False),
                    "symbols": symbols,
                    "validation_results": result.get("validation_results", {}),
                    "quality_summary": result.get("quality_summary", {}),
                    "reason": result.get("reason", "")
                }
            else:
                return {
                    "approved": False,
                    "error": f"API error: {response.status_code}",
                    "symbols": symbols
                }
                
        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            return {
                "approved": False,
                "error": str(e),
                "symbols": symbols if 'symbols' in locals() else ['AAPL', 'MSFT', 'GOOGL']  # S&P 500 only
            }
    
    def _extract_symbols_from_objective(self, objective: str) -> List[str]:
        """Extract stock symbols from trial objective"""
        # Simple extraction - look for common stock symbol patterns
        import re
        
        # Look for 3-4 letter uppercase sequences that might be symbols
        potential_symbols = re.findall(r'\b[A-Z]{2,5}\b', objective)
        
        # Filter to common symbols and remove common words
        excluded_words = {'ML', 'AI', 'API', 'CPU', 'GPU', 'RAM', 'USD', 'THE', 'AND', 'FOR', 'WITH'}
        symbols = [s for s in potential_symbols if s not in excluded_words]
        
        # If no symbols found, return empty list to use defaults
        return symbols[:10]  # Limit to 10 symbols max
    
    def _handle_data_quality_failure(self, session_id: str, quality_check: Dict[str, Any]):
        """Handle data quality validation failure"""
        with self.trial_lock:
            session = self.active_trials.get(session_id)
            if not session:
                return
            
            # Pause the trial due to data quality issues
            session.status = "paused"
            
            # Create notification for user
            error_msg = quality_check.get("error", "Unknown data quality issue")
            reason = quality_check.get("reason", "Data quality validation failed")
            
            notification = f"⚠️ Trial paused: {reason}. Error: {error_msg}"
            session.user_notifications.append(notification)
            
            # Log the issue
            logger.warning(f"Trial {session_id} paused due to data quality issues: {quality_check}")
            
            # Add a step to record the quality failure
            quality_step = TrialStep(
                step_id=f"{session_id}_quality_check_{len(session.steps)}",
                action="data_quality_validation",
                status="failed",
                timestamp=datetime.now(),
                input_data={"validation_request": quality_check.get("symbols", [])},
                output_data=quality_check,
                mltrainer_analysis=f"Data quality validation failed: {reason}. Trial paused pending quality improvement.",
                next_suggested_action="wait_for_data_quality_improvement"
            )
            
            session.steps.append(quality_step)
            
            self._notify_updates(f"Trial {session_id} paused - data quality issues detected")

# Global instance for background trial management
_background_trial_manager = None

def get_background_trial_manager(ml_executor=None, ai_client=None):
    """Get or create the global background trial manager"""
    global _background_trial_manager
    if _background_trial_manager is None and ml_executor and ai_client:
        _background_trial_manager = BackgroundTrialManager(ml_executor, ai_client)
    return _background_trial_manager