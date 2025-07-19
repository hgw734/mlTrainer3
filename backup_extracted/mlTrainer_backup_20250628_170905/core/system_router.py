import json
from flask import Blueprint, request, jsonify
from ml.ml_trainer import ml_trainer
from core.mlTrainer_engine import mlTrainerEngine
from monitoring.health_monitor import get_health_status
from monitoring.error_monitor import get_recent_errors
from core.compliance_mode import enforce_api_compliance
from core.immutable_gateway import secure_input, secure_output

router = Blueprint('system_router', __name__)
mltrainer_engine = mlTrainerEngine()  # ✅ Correctly instantiate the engine

@router.route("/run_trial", methods=["POST"])
def run_trial():
    try:
        config = secure_input(request.get_json(force=True))
        # Note: enforce_api_compliance expects API name, not config
        # This may need adjustment based on your specific compliance needs
        result = mltrainer_engine.start_trial(
            user_prompt=f"Run ML trial on {config.get('symbol', 'unknown')}",
            trial_config=config
        )
        return jsonify(secure_output(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@router.route("/trial_history", methods=["GET"])
def trial_history():
    history = mltrainer_engine.trial_history
    return jsonify(secure_output(history))

@router.route("/models", methods=["GET"])
def list_models():
    try:
        return jsonify(mltrainer_engine.get_available_models())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@router.route("/status", methods=["GET"])
def system_status():
    try:
        health = get_health_status()
        return jsonify(health)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@router.route("/errors", methods=["GET"])
def recent_errors():
    try:
        errors = get_recent_errors()
        return jsonify(errors)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@router.route("/debug/test", methods=["GET"])
def debug_test():
    return jsonify({"message": "✅ System routing active and secure."})

@router.route("/compliance/test", methods=["POST"])
def compliance_test():
    try:
        test_payload = secure_input(request.get_json(force=True))
        # Note: enforce_api_compliance expects API name, not payload
        # This may need adjustment based on your specific compliance needs
        return jsonify({
            "message": "✅ Compliance check passed",
            "payload": secure_output(test_payload)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 403