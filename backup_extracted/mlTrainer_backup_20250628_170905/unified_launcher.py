
import subprocess
import threading
import time
import os
import signal
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_flask():
    """Start Flask backend on port 5000"""
    logger.info("🚀 Starting Flask backend...")

    env = os.environ.copy()
    env.update({
        "FLASK_PORT": "5000",
        "APP_PORT": "5000"
    })

    try:
        subprocess.run([
            "python3", "main.py"
        ], env=env, check=True)
    except Exception as e:
        logger.error(f"❌ Flask failed: {e}")


def start_streamlit():
    """Start Streamlit frontend on port 8501 (mapped to external port 3000)"""
    logger.info("🚀 Starting Streamlit frontend...")

    # Wait for Flask to be ready
    time.sleep(5)

    env = os.environ.copy()
    env.update({
        "STREAMLIT_DISABLE_EMAIL_COLLECTION": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_SERVER_ADDRESS": "0.0.0.0",
        "STREAMLIT_SERVER_PORT": "8501"
    })

    try:
        subprocess.run([
            "streamlit", "run", "interactive_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableWebsocketCompression=false",
            "--server.allowRunOnSave=false",
            "--server.enableXsrfProtection=false",
            "--browser.gatherUsageStats=false"
        ], env=env, check=True)
    except Exception as e:
        logger.error(f"❌ Streamlit failed: {e}")


def cleanup():
    """Clean up processes on exit"""
    logger.info("🧹 Cleaning up processes...")
    subprocess.run(["pkill", "-f", "flask"], capture_output=True)
    subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
    subprocess.run(["fuser", "-k", "5000/tcp"], capture_output=True)
    subprocess.run(["fuser", "-k", "8501/tcp"], capture_output=True)


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("🛑 Shutdown signal received")
    cleanup()
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("🚀 Starting unified mlTrainer system...")

    # Clean up any existing processes
    cleanup()
    time.sleep(2)

    try:
        # Start Flask in background thread
        flask_thread = threading.Thread(target=start_flask, daemon=True)
        flask_thread.start()

        # Wait a bit for Flask to initialize
        logger.info("⏳ Waiting for Flask to initialize...")
        time.sleep(8)

        # Verify Flask is responding
        import requests
        try:
            response = requests.get("http://127.0.0.1:5000/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Flask backend is healthy")
            else:
                logger.warning(
                    f"⚠️ Flask responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Flask health check failed: {e}")
            cleanup()
            sys.exit(1)

        # Start Streamlit in foreground (this will be the main process Replit
        # routes to)
        logger.info("🎯 Starting Streamlit as primary process...")
        start_streamlit()

    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unified launcher failed: {e}")
    finally:
        cleanup()
