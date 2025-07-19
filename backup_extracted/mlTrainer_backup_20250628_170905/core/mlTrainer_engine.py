import traceback
import json
import os
import time
import gc
from datetime import datetime
from core.mlTrainer_client_wrapper import fmt_agent as mlTrainerClient
from ml.ml_trainer import ml_trainer
# mlTrainer operates as external AI agent - bypasses compliance
from core.immutable_gateway import secure_input, secure_output
from monitoring.health_monitor import log_status
from monitoring.error_monitor import log_error

MEMORY_FILE = "mltrainer_memory.json"
MAX_MESSAGES = 50  # Reduced for efficiency

class mlTrainerEngine:
    def __init__(self):
        self.trial_history = []
        self.model_client = mlTrainerClient
        self._memory_cache = None
        self._last_cache_time = 0
        
    def _load_memory(self):
        """Lazy load with caching"""
        current_time = time.time()
        if self._memory_cache is None or (current_time - self._last_cache_time) > 300:  # 5min cache
            if os.path.exists(MEMORY_FILE):
                try:
                    with open(MEMORY_FILE, "r") as f:
                        self._memory_cache = json.load(f)[-MAX_MESSAGES:]  # Only keep recent
                        self._last_cache_time = current_time
                except Exception:
                    self._memory_cache = []
            else:
                self._memory_cache = []
        self.trial_history = self._memory_cache

    def _save_memory(self):
        """Efficient memory saving with size limits"""
        try:
            # Only save essential data
            trimmed = []
            for trial in self.trial_history[-MAX_MESSAGES:]:
                compact_trial = {
                    "timestamp": trial.get("timestamp"),
                    "user_prompt": trial.get("user_prompt", "")[:200],  # Truncate
                    "trial_config": {k: v for k, v in trial.get("trial_config", {}).items() 
                                   if k in ["symbol", "model", "score"]},  # Essential only
                    "result_summary": trial.get("ml_result", {}).get("metrics", {})
                }
                trimmed.append(compact_trial)
            
            with open(MEMORY_FILE, "w") as f:
                json.dump(trimmed, f, separators=(',', ':'))  # Compact JSON
        except Exception:
            pass

    def start_trial(self, user_prompt: str, trial_config: dict = None, chat_context: list = None) -> dict:
        try:
            # Agent orchestration with exit conditions
            max_iterations = 10
            current_iteration = 0

            # mlTrainer bypasses compliance - operates as external AI agent
            user_prompt = secure_input(user_prompt)
            trial_config = trial_config or {}

            # Smart symbol detection
            words = user_prompt.upper().split()
            symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "NFLX", "SPY", "QQQ"]
            detected_symbol = None

            for word in words:
                if word in symbols:
                    detected_symbol = word
                    break

            # Force ML analysis for almost everything - only skip for pure greetings
            pure_greeting_keywords = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
            is_pure_greeting = len(user_prompt.strip().lower().split()) <= 2 and any(greeting in user_prompt.lower() for greeting in pure_greeting_keywords)

            should_run_ml = detected_symbol is not None or not is_pure_greeting

            if should_run_ml:
                # AUTONOMOUS TRIAL CONFIGURATION - mlTrainer decides parameters
                trial_config["symbol"] = detected_symbol or "AAPL"
                trial_config["model"] = trial_config.get("model", "LSTM")
                trial_config["start_date"] = trial_config.get("start_date", "2023-01-01") 
                trial_config["end_date"] = trial_config.get("end_date", "2024-01-01")
                trial_config["train_ratio"] = float(trial_config.get("train_ratio", 0.8))
                trial_config["paper_mode"] = True  # Always enable paper trading

                log_status("🚀 mlTrainer AUTO-INITIATING ML Trial", context=trial_config)

                try:
                    # Create communication callback for interactive ML trials
                    def ml_communication_callback(question_data):
                        """Handle real-time questions from ML engine"""
                        log_status("🔥 REAL-TIME ML COMMUNICATION ACTIVE", context=question_data)

                        # Store communication log for later analysis
                        if not hasattr(self, '_communication_log'):
                            self._communication_log = []
                        
                        self._communication_log.append({
                            "timestamp": datetime.now().isoformat(),
                            "question": question_data,
                            "type": "ml_to_trainer"
                        })

                        # Agent communication with Claude
                        try:
                            communication_prompt = f"""
ML ENGINE COMMUNICATION:
{json.dumps(question_data, indent=2)}

As mlTrainer, analyze this ML engine question and provide your decision:

AVAILABLE ACTIONS:
- continue: Proceed with current parameters
- retrain: Adjust parameters and retrain model
- switch_model: Change to different model (specify: XGBoost, LSTM, Transformer, Ensemble)
- optimize: Optimize hyperparameters
- stop: Halt trial due to issues

Respond with your decision and reasoning in this format:
ACTION: [your_action]
REASON: [your_reasoning]
PARAMETERS: [any parameter changes if needed]
"""

                            claude_response = self.model_client.send_message([{
                                "role": "user",
                                "content": communication_prompt
                            }])

                            response_text = claude_response.get("response", "ACTION: continue\nREASON: Default continue")
                            
                            # Log mlTrainer's response
                            self._communication_log.append({
                                "timestamp": datetime.now().isoformat(),
                                "response": response_text,
                                "type": "trainer_to_ml"
                            })

                            # Parse Claude's structured response
                            action = "continue"
                            reason = "Default continue"
                            parameters = {}
                            
                            for line in response_text.split('\n'):
                                if line.startswith('ACTION:'):
                                    action = line.replace('ACTION:', '').strip().lower()
                                elif line.startswith('REASON:'):
                                    reason = line.replace('REASON:', '').strip()
                                elif line.startswith('PARAMETERS:'):
                                    param_str = line.replace('PARAMETERS:', '').strip()
                                    try:
                                        parameters = json.loads(param_str) if param_str and param_str != '[]' else {}
                                    except:
                                        parameters = {}

                            # Return structured response for ML engine
                            response = {
                                "action": action,
                                "reason": reason,
                                "parameters": parameters,
                                "mltrainer_response": response_text
                            }

                            # Handle specific actions
                            if action == "switch_model":
                                # Extract model name from response
                                model_keywords = ["xgboost", "lstm", "transformer", "ensemble"]
                                for keyword in model_keywords:
                                    if keyword in response_text.lower():
                                        response["new_model"] = keyword.capitalize()
                                        break

                            log_status("✅ mlTrainer decision communicated", context=response)
                            return response

                        except Exception as e:
                            log_error("ML communication callback failed", details=str(e))
                            return {"action": "continue", "error": str(e), "reason": "Communication error - proceeding with defaults"}

                    # Run ML trial with communication callback
                    ml_result = ml_trainer.run_trial(trial_config, ml_communication_callback)

                    # Get mlTrainer's commentary on results
                    commentary = self.model_client.get_model_feedback(ml_result)

                    # Send to Claude for final response generation
                    final_prompt = f"""
USER QUERY: {user_prompt}

ML TRIAL RESULTS: {json.dumps(ml_result, indent=2, default=str)}

MLTRAINER COMMENTARY: {commentary}

Generate a comprehensive response that includes:
1. Direct answer to user's query
2. ML analysis results
3. Trading insights
4. Recommendations
"""

                    claude_response = self.model_client.send_message([{
                        "role": "user", 
                        "content": final_prompt
                    }])

                    # Store trial in history
                    trial_record = {
                        "timestamp": datetime.now().isoformat(),
                        "user_prompt": user_prompt,
                        "trial_config": trial_config,
                        "ml_result": ml_result,
                        "commentary": commentary
                    }

                    self.trial_history.append(trial_record)
                    self._save_memory()

                    return {
                        "response": claude_response.get("response", "Analysis complete"),
                        "ml_analysis": {
                            "symbol": ml_result.get("symbol", "Unknown"),
                            "model": ml_result.get("model", "Unknown"),
                            "score": ml_result.get("metrics", {}).get("accuracy", 0),
                            "total_return": ml_result.get("paper_trading", {}).get("win_ratio", 0) * 1000
                        },
                        "trial_id": len(self.trial_history)
                    }

                except Exception as e:
                    log_error("ML trial execution failed", details=traceback.format_exc())
                    # Fallback to Claude-only response
                    claude_response = self.model_client.send_message([{
                        "role": "user",
                        "content": f"User query: {user_prompt}\n\nML engine unavailable. Provide helpful trading/financial analysis response."
                    }])

                    return {
                        "response": claude_response.get("response", "I'm analyzing your request..."),
                        "error": f"ML engine error: {str(e)}"
                    }

            else:
                # Pure conversation mode
                messages = [{"role": "user", "content": user_prompt}]
                if chat_context:
                    # Add recent context
                    for msg in chat_context[-5:]:
                        messages.insert(-1, {
                            "role": msg["role"],
                            "content": msg["content"][:500]  # Truncate for context
                        })

                claude_response = self.model_client.send_message(messages)
                return {"response": claude_response.get("response", "Hello! How can I help you with trading analysis?")}

        except Exception as e:
            log_error("mlTrainer engine failed", details=traceback.format_exc())
            return {"error": f"Engine error: {str(e)}"}

    def get_trial_history(self) -> list:
        """Get recent trial history"""
        self._load_memory()
        return self.trial_history[-10:]  # Last 10 trials

    def get_system_status(self) -> dict:
        """Get system status for monitoring"""
        return {
            "trials_completed": len(self.trial_history),
            "status": "active",
            "last_trial": self.trial_history[-1]["timestamp"] if self.trial_history else None
        }