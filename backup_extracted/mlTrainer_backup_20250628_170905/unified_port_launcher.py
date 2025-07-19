import os
import sys
import subprocess
import time
import logging
from fix_ports_replit import fix_port_conflicts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_flask_backend(port):
    """Start Flask backend using port from fix_ports_replit"""
    logger.info(f"🚀 Starting Flask backend on port {port}")

    env = os.environ.copy()
    env['FLASK_RUN_PORT'] = str(port)
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'

    flask_process = subprocess.Popen(
        [sys.executable, 'main.py'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(3)

    if flask_process.poll() is None:
        logger.info("✅ Flask backend started successfully")
        return flask_process
    else:
        logger.error("❌ Flask backend failed to start")
        return None

def start_streamlit_frontend(port):
    """Start Streamlit frontend using port from fix_ports_replit"""
    logger.info(f"🚀 Starting Streamlit frontend on port {port}")

    env = os.environ.copy()
    env['STREAMLIT_SERVER_PORT'] = str(port)
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'

    streamlit_cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'interactive_app.py',
        '--server.port', str(port),
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableWebsocketCompression', 'false',
        '--server.allowRunOnSave', 'false',
        '--server.enableXsrfProtection', 'false',
        '--server.maxUploadSize', '200'
    ]

    streamlit_process = subprocess.Popen(
        streamlit_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(5)

    if streamlit_process.poll() is None:
        logger.info("✅ Streamlit frontend started successfully")
        return streamlit_process
    else:
        logger.error("❌ Streamlit frontend failed to start")
        return None

def main():
    """Main launcher using fix_ports_replit for port management"""
    logger.info("🚀 mlTrainer Unified Port Launcher (Using fix_ports_replit)")

    try:
        # Use fix_ports_replit to get separate ports
        ports = fix_port_conflicts()
        flask_port = ports['flask_port']
        streamlit_port = ports['streamlit_port']

        logger.info(f"✅ Flask port: {flask_port}")
        logger.info(f"✅ Streamlit port: {streamlit_port}")

        # Start Flask backend
        flask_process = start_flask_backend(flask_port)
        if not flask_process:
            logger.error("❌ Failed to start Flask backend")
            sys.exit(1)

        # Start Streamlit frontend on separate port
        streamlit_process = start_streamlit_frontend(streamlit_port)
        if not streamlit_process:
            logger.error("❌ Failed to start Streamlit frontend")
            flask_process.terminate()
            sys.exit(1)

        logger.info(f"🌐 Flask Backend: http://0.0.0.0:{flask_port}")
        logger.info(f"🌐 Streamlit Frontend: http://0.0.0.0:{streamlit_port}")
        logger.info("✅ Both services started successfully with separate ports")

        # Keep processes running
        try:
            while True:
                time.sleep(1)

                if flask_process.poll() is not None:
                    logger.error("❌ Flask backend stopped unexpectedly")
                    break

                if streamlit_process.poll() is not None:
                    logger.error("❌ Streamlit frontend stopped unexpectedly") 
                    break

        except KeyboardInterrupt:
            logger.info("🛑 Shutting down services...")

        finally:
            if flask_process and flask_process.poll() is None:
                flask_process.terminate()
            if streamlit_process and streamlit_process.poll() is None:
                streamlit_process.terminate()

    except Exception as e:
        logger.error(f"❌ Launcher failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()