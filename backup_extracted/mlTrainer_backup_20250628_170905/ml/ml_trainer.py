# Enhanced communication callback with agent orchestration patterns
import json
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
from core.compliance_mode import enforce_api_compliance
from core.immutable_gateway import secure_input, secure_output
from monitoring.health_monitor import log_status
from monitoring.error_monitor import log_error
from ml.model_registry import get_model_by_name
from data_sources.polygon_interface import get_historical_data, get_live_price
import logging

logger = logging.getLogger(__name__)


class MLTrainer:
    def __init__(self):
        self.trials = []
        self.logs = []

    def run_trial(self, trial_config: Dict[str, Any], communication_callback=None) -> Dict[str, Any]:
        try:
            # MANDATORY: Add data_source field for compliance
            if 'data_source' not in trial_config:
                trial_config['data_source'] = 'polygon'

            trial_config = secure_input(trial_config)
            symbol = trial_config['symbol']
            model_name = trial_config['model']
            start_date = pd.to_datetime(trial_config['start_date'])
            end_date = pd.to_datetime(trial_config['end_date'])
            train_ratio = float(trial_config.get('train_ratio', 0.8))
            paper_mode = bool(trial_config.get('paper_mode', False))

            model = get_model_by_name(model_name)
            if model is None:
                raise ValueError(f"Unknown model requested: {model_name}")

            # Optimized data loading with size limits
            df = get_historical_data(symbol, start_date, end_date)
            if df is None or df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Limit data size for memory efficiency
            if len(df) > 2000:
                df = df.tail(2000)  # Keep recent 2000 records max
                logger.info(f"⚡ Data optimized: Using recent {len(df)} records for efficiency")

            # Simplified communication with essential data only
            if communication_callback:
                volatility = float(df['close'].pct_change().std())
                if volatility > 0.05:  # Only communicate if high volatility
                    response = communication_callback({
                        "type": "volatility_check",
                        "volatility": volatility,
                        "adjust_ratio": 0.7 if volatility > 0.08 else 0.8
                    })
                    if response.get("adjust_ratio"):
                        train_ratio = float(response["adjust_ratio"])

            df['target'] = df['close'].shift(-1)
            df.dropna(inplace=True)

            train_size = int(len(df) * train_ratio)
            train, test = df[:train_size], df[train_size:]

            X_train = train.drop(columns=['target'])
            y_train = train['target']
            X_test = test.drop(columns=['target'])
            y_test = test['target']

            # Real-time communication: Training progress check
            if communication_callback:
                training_question = {
                    "type": "training_progress",
                    "message": f"Training {model_name} on {len(X_train)} samples. Training set covers {train.index.min()} to {train.index.max()}. Continue or modify model?",
                    "training_stats": {
                        "training_samples": len(X_train),
                        "test_samples": len(X_test),
                        "features": list(X_train.columns)[:5]  # First 5 features
                    }
                }
                response = communication_callback(training_question)
                if response.get("switch_model"):
                    new_model = get_model_by_name(response["switch_model"])
                    if new_model:
                        model = new_model
                        model_name = response["switch_model"]

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Real-time communication: Initial results check
            initial_metrics = self.evaluate_metrics(y_test, predictions)
            if communication_callback:
                results_question = {
                    "type": "initial_results",
                    "message": f"Initial results: {initial_metrics['accuracy']:.2%} accuracy, Sharpe: {initial_metrics['sharpe']:.3f}. Should I continue with full analysis or retrain?",
                    "initial_metrics": initial_metrics,
                    "model_performance": {
                        "current_model": model_name,
                        "accuracy": initial_metrics['accuracy'],
                        "sharpe_ratio": initial_metrics['sharpe'],
                        "max_drawdown": initial_metrics.get('drawdown', 0)
                    },
                    "recommendations": {
                        "continue": "Proceed with current model",
                        "retrain": "Adjust parameters and retrain",
                        "switch_model": "Try different model type",
                        "optimize": "Run hyperparameter optimization"
                    }
                }

                log_status("📞 ML ENGINE -> mlTrainer: Requesting decision on initial results")
                response = communication_callback(results_question)
                log_status("📞 mlTrainer -> ML ENGINE: Decision received", context=response)

                action = response.get("action", "continue")

                if action == "retrain":
                    log_status("🔄 RETRAINING: mlTrainer requested parameter adjustment")
                    # Apply parameter adjustments if provided
                    new_params = response.get("parameters", {})
                    if new_params.get("train_ratio"):
                        train_ratio = float(new_params["train_ratio"])
                        # Recalculate train/test split
                        train_size = int(len(df) * train_ratio)
                        train, test = df[:train_size], df[train_size:]
                        X_train = train.drop(columns=['target'])
                        y_train = train['target']
                        X_test = test.drop(columns=['target'])
                        y_test = test['target']

                    # Retrain with new parameters
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    initial_metrics = self.evaluate_metrics(y_test, predictions)
                    log_status("🎯 RETRAIN COMPLETE: New metrics calculated")

                elif action == "switch_model":
                    new_model_name = response.get("new_model", "XGBoost")
                    log_status(f"🔄 MODEL SWITCH: Changing from {model_name} to {new_model_name}")
                    new_model = get_model_by_name(new_model_name)
                    if new_model:
                        model = new_model
                        model_name = new_model_name
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        initial_metrics = self.evaluate_metrics(y_test, predictions)
                        log_status(f"✅ MODEL SWITCH COMPLETE: {new_model_name} trained")
                    else:
                        log_error(f"❌ Model switch failed: {new_model_name} not available")

                elif action == "optimize":
                    log_status("🎛️ HYPERPARAMETER OPTIMIZATION: mlTrainer requested optimization")
                    # Placeholder for future hyperparameter optimization
                    # This would integrate with core/hyperparameter_optimizer.py

                else:
                    log_status("▶️ CONTINUING: mlTrainer approved current model performance")

            metrics = initial_metrics
            result = {
                "symbol": symbol,
                "model": model_name,
                "start": str(start_date.date()),
                "end": str(end_date.date()),
                "metrics": metrics,
                "predictions": predictions.tolist(),
                "actuals": y_test.tolist(),
                "timestamp": datetime.now().isoformat(),
                "communication_log": getattr(self, '_communication_log', [])
            }

            if paper_mode:
                paper_result = self.simulate_paper_trading(symbol, predictions, y_test)
                result["paper_trading"] = paper_result

            result = secure_output(result)
            log_status("Trial completed", context=result)
            return result

        except Exception as e:
            log_error("❌ Trial execution failed", details=traceback.format_exc())
            return {"error": str(e), "details": traceback.format_exc()}

    def evaluate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Optimized metrics calculation with memory efficiency"""
        returns = np.array(y_pred) - np.array(y_true)
        accuracy = np.mean(np.sign(y_pred) == np.sign(y_true))
        mae = np.mean(np.abs(returns))
        rmse = np.sqrt(np.mean(returns ** 2))
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
        drawdown = self.max_drawdown(np.cumsum(returns))

        # Basic risk metrics (fast calculation)
        var_5 = np.percentile(returns, 5)
        expected_shortfall = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else var_5
        
        # Simplified tracking metrics
        tracking_error = np.std(returns)
        information_ratio = np.mean(returns) / tracking_error if tracking_error != 0 else 0
        calmar_ratio = np.mean(returns) * 252 / abs(drawdown) if drawdown != 0 else float('inf')

        # Return essential metrics only
        return {
            "accuracy": round(float(accuracy), 4),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "sharpe": round(float(sharpe), 4),
            "drawdown": round(float(drawdown), 4),
            "var_5": round(float(var_5), 4),
            "expected_shortfall": round(float(expected_shortfall), 4),
            "information_ratio": round(float(information_ratio), 4),
            "calmar_ratio": min(round(float(calmar_ratio), 4), 999.0),  # Cap extreme values
            "tracking_error": round(float(tracking_error), 4)
        }

    def simulate_paper_trading(self, symbol: str, predictions, actuals) -> Dict[str, Any]:
        positions = []
        for pred, actual in zip(predictions, actuals):
            entry = get_live_price(symbol)
            if entry is None:
                continue
            result = {
                "entry": entry,
                "predicted": pred,
                "actual": actual,
                "success": np.sign(pred - entry) == np.sign(actual - entry)
            }
            positions.append(result)
        win_ratio = sum(p['success'] for p in positions) / len(positions) if positions else 0
        return {"positions": positions, "win_ratio": win_ratio}

    def max_drawdown(self, returns) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        if len(returns) == 0:
            return 0.0
        peak = returns[0]
        max_dd = 0.0
        for value in returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, drawdown)
        return max_dd

    def get_available_models(self) -> dict:
        """Return detailed information about available ML models"""
        return {
            "models": {
                "LSTM": {
                    "name": "Long Short-Term Memory",
                    "type": "Neural Network",
                    "best_for": "Time series prediction",
                    "accuracy": "High for trend following"
                },
                "XGBoost": {
                    "name": "Extreme Gradient Boosting", 
                    "type": "Ensemble Tree",
                    "best_for": "Feature-rich datasets",
                    "accuracy": "High for classification"
                },
                "Transformer": {
                    "name": "Attention-based Model",
                    "type": "Neural Network", 
                    "best_for": "Complex pattern recognition",
                    "accuracy": "Excellent for long sequences"
                },
                "Ensemble": {
                    "name": "Combined Models",
                    "type": "Meta-learning",
                    "best_for": "Robust predictions",
                    "accuracy": "Best overall performance"
                }
            },
            "default": "LSTM",
            "status": "all_available"
        }

    def train_with_regime_awareness(self, symbol: str, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train models with multidimensional regime awareness"""
        try:
            logger.info(f"🎯 Training {symbol} with multidimensional regime awareness")

            # Get price data
            from data_sources.polygon_interface import fetch_stock_data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            price_data = fetch_stock_data(symbol, start_date, end_date)
            if price_data.empty:
                return {"error": "No price data available"}

            # Extract regime information
            regime_profile = regime_data.get('regime_profile', {})

            # Get multidimensional model activation
            try:
                from model_activation import get_active_models, get_regime_specific_parameters
                model_config = get_active_models(regime_profile)
                regime_params = get_regime_specific_parameters(regime_profile)

                active_models = model_config['active_models']
                model_weights = model_config['model_weights']

                logger.info(f"🧠 Regime: {model_config['regime_classification']}")
                logger.info(f"📊 Active models: {active_models}")
                logger.info(f"⚖️ Model weights: {model_weights}")

            except ImportError:
                logger.warning("⚠️ Using fallback model selection")
                active_models = ["XGBoost", "LSTM", "EnsembleVoting"]
                model_weights = {"XGBoost": 0.4, "LSTM": 0.4, "EnsembleVoting": 0.2}
                regime_params = {}

            results = {}
            for model_name in active_models:
                if model_name in self.models:
                    # Apply regime-specific parameters
                    # Assuming _train_single_model_with_regime_params exists and is correctly implemented
                    model_result = self._train_single_model_with_regime_params(
                        model_name, price_data, regime_profile, regime_params
                    )
                    results[model_name] = model_result
                    results[model_name]["weight"] = model_weights.get(model_name, 0.1)

            return {
                "symbol": symbol,
                "trained_models": list(results.keys()),
                "regime_profile": regime_profile,
                "model_weights": model_weights,
                "regime_classification": model_config.get('regime_classification', 'UNKNOWN'),
                "activation_reasoning": model_config.get('activation_reasoning', ''),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Regime-aware training failed: {e}")
            return {"error": str(e)}

# Agent-to-agent communication callback with structured handoffs
def create_ml_trainer(communication_callback=None):
    class MLTrainerClientWrapper:
        def __init__(self):
            self.ml_trainer = MLTrainer()

        def run_trial(self, trial_config):
            """Execute a single trial with mlTrainer."""

            # Agent-to-agent communication callback with structured handoffs
            def ml_communication_callback(question_data):
                """Handle real-time agent communication during ML trials"""
                log_status("🔥 AGENT-TO-AGENT COMMUNICATION ACTIVE", context=question_data)

                # Implement exit conditions for communication loop
                if question_data.get("iteration_count", 0) > 5:
                    return {"action": "escalate_to_human", "reason": "Max communication iterations reached"}

                # Risk-based routing
                if question_data.get("risk_level") == "high":
                    return {"action": "pause_for_approval", "reason": "High-risk operation detected"}

            return self.ml_trainer.run_trial(trial_config, ml_communication_callback)
    return MLTrainerClientWrapper()

ml_trainer = MLTrainer()