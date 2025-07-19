from flask import Blueprint, request
from core.immutable_gateway import immutable_route
from core.mlTrainer_engine import mlTrainerEngine
from monitoring.health_monitor import get_system_health
from monitoring.error_monitor import get_recent_errors

routes = Blueprint('routes', __name__)
mltrainer_engine = mlTrainerEngine()

@routes.route("/api/run_trial", methods=["POST"])
@immutable_route
def run_trial():
    trial_config = request.get_json()
    return mltrainer_engine.start_trial(
        user_prompt=f"Run trial for {trial_config.get('symbol', 'unknown')}",
        trial_config=trial_config
    )

@routes.route("/api/system_health", methods=["GET"])
@immutable_route
def system_health():
    return get_system_health()

@routes.route("/api/errors", methods=["GET"])
@immutable_route
def get_errors():
    return get_recent_errors()

@routes.route("/api/mltrainer/status", methods=["GET"])
@immutable_route
def mltrainer_status():
    return {
        "status": "active",
        "last_result": getattr(mltrainer_engine, 'last_result', None)
    }

@routes.route("/api/mltrainer/logs", methods=["GET"])
@immutable_route
def mltrainer_logs():
    return {
        "logs": getattr(mltrainer_engine, 'trial_history', [])
    }
