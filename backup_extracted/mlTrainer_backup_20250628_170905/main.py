import logging
import json
import socket
import sys
import os
from datetime import datetime
from flask import Flask
from core.system_router import router as system_router_blueprint
from core.compliance_mode import is_compliance_enabled, enable_compliance
from core.configuration import load_allowed_apis

# Add startup validation
try:
    from startup_validator import validate_startup
    if not validate_startup():
        print("❌ Startup validation failed. Check startup_validator.py for details.")
        sys.exit(1)
except ImportError:
    print("⚠️ Startup validator not found, proceeding without validation")
except Exception as e:
    print(f"⚠️ Startup validation error: {e}")
    print("Proceeding with caution...")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_port_available(port, host="0.0.0.0"):
    """Check if a port is available to bind"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.connect_ex((host, port)) != 0


def find_open_port(start=5000, end=5100):
    """Find an open port in the specified range"""
    for port in range(start, end + 1):
        if is_port_available(port):
            logger.info(f"Found available port: {port}")
            return port
    raise RuntimeError("❌ No available port found in range")


def resolve_and_store_port(preferred_port=5000):
    """Resolve port and store configuration for other components"""
    try:
        # First try the preferred port
        if is_port_available(preferred_port):
            port = preferred_port
        else:
            # Find alternative port
            port = find_open_port(preferred_port, preferred_port + 100)

        # Store in environment variable
        os.environ["APP_PORT"] = str(port)
        os.environ["FLASK_PORT"] = str(port)

        # Write configuration file for other components
        port_config = {
            "port": port,
            "host": "0.0.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }

        with open("port_config.json", "w") as f:
            json.dump(port_config, f, indent=2)

        logger.info(f"✅ Port resolved and stored: {port}")
        return port

    except Exception as e:
        logger.error(f"Failed to resolve port: {e}")
        # Fall back to default port
        default_port = 5000
        os.environ["APP_PORT"] = str(default_port)
        os.environ["FLASK_PORT"] = str(default_port)
        return default_port


def main():
    logger.info("🚀 mlTrainer System Startup")

    # Enable strict compliance mode
    enable_compliance()

    if is_compliance_enabled():
        logger.info("🔒 Compliance mode is ACTIVE")

    # Load configuration
    logger.info(
        "🎯 mlTrainer system ready - API verification deferred to first use")
    logger.info("🔒 COMPLIANCE ENFORCED: Real data from authorized sources")

    available_apis = load_allowed_apis()
    available_models = ["LSTM", "XGBoost", "Transformer", "Ensemble"]
    logger.info(f"📋 APIs: {len(available_apis)} configured")
    logger.info(f"🤖 Models: {len(available_models)} available")

    # Initialize Flask app
    app = Flask(__name__)
    app.register_blueprint(system_router_blueprint)

    @app.route("/health")
    def health():
        return {"status": "ok", "service": "mlTrainer Flask Backend"}, 200

    @app.route('/')
    def home():
        streamlit_port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
        return f'''
        <script>
        // Redirect to Streamlit interface
        setTimeout(function() {{
            window.location.href = window.location.protocol + '//' + window.location.hostname + ':{streamlit_port}';
        }}, 1500);
        </script>
        <div style="text-align: center; font-family: Georgia, serif; padding: 50px;">
            <h1>🧠 mlTrainer Chat Interface</h1>
            <p>Redirecting to the main chat interface on port {streamlit_port}...</p>
            <p>If you're not redirected automatically, <a href=":{streamlit_port}">click here</a></p>
        </div>
        '''

    # Use unified port from fix_ports_replit.py
    unified_port = int(os.environ.get("FLASK_RUN_PORT", 5000))

    logger.info(f"🌐 Starting Flask web server on http://0.0.0.0:{unified_port}")
    app.run(host="0.0.0.0", port=unified_port, debug=False)


if __name__ == "__main__":
    main()