
import os
import socket
import subprocess
import logging
import sys

DEFAULT_PORT = 8501
PORT_RANGE = 100  # Scan 100 ports upward
MAX_SOCKET_RETRIES = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_free_port(start_port=DEFAULT_PORT, range_size=PORT_RANGE):
    """Finds an available port starting from start_port."""
    for port in range(start_port, start_port + range_size):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                logger.info(f"✅ Found available port: {port}")
                return port
            except OSError:
                continue
    logger.error("❌ No available ports found in range.")
    sys.exit(1)


def kill_process_on_port(port):
    """Kills the process using the specified port."""
    try:
        # Use lsof to find process (macOS/Linux) or netstat (Windows)
        if os.name == 'posix':
            result = subprocess.check_output(
                ["lsof", "-ti", f":{port}"]).decode().strip()
            if result:
                for pid in result.split("\n"):
                    logger.warning(
                        f"⚠️ Killing process {pid} using port {port}")
                    subprocess.run(["kill", "-9", pid])
        elif os.name == 'nt':
            result = subprocess.check_output(
                f"netstat -ano | findstr :{port}", shell=True).decode()
            for line in result.splitlines():
                parts = line.split()
                pid = parts[-1]
                logger.warning(f"⚠️ Killing process {pid} using port {port}")
                subprocess.run(["taskkill", "/PID", pid, "/F"], shell=True)
    except Exception as e:
        logger.error(f"❌ Failed to kill process on port {port}: {e}")


def verify_port_is_free(port):
    """Check if the port is actually free by attempting to bind."""
    for _ in range(MAX_SOCKET_RETRIES):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return True
        except OSError:
            logger.warning(f"Port {port} still in use.")
    return False


def resolve_port_conflict(preferred_port=DEFAULT_PORT):
    """Main function to resolve port conflicts and return safe port."""
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

    # Check if preferred port is free
    logger.info(f"🔍 Checking port {preferred_port}")
    if verify_port_is_free(preferred_port):
        os.environ['STREAMLIT_SERVER_PORT'] = str(preferred_port)
        logger.info(f"✅ Using preferred port {preferred_port}")
        return preferred_port

    logger.warning(
        f"⚠️ Port {preferred_port} is in use. Attempting to free it...")
    kill_process_on_port(preferred_port)

    if verify_port_is_free(preferred_port):
        os.environ['STREAMLIT_SERVER_PORT'] = str(preferred_port)
        logger.info(f"✅ Reclaimed preferred port {preferred_port}")
        return preferred_port

    logger.info("🔄 Trying to find a new free port...")
    fallback_port = find_free_port(preferred_port + 1, PORT_RANGE)
    os.environ['STREAMLIT_SERVER_PORT'] = str(fallback_port)
    logger.info(f"✅ Using fallback port {fallback_port}")
    return fallback_port


if __name__ == "__main__":
    resolved_port = resolve_port_conflict()
    print(f"✅ Safe to start your app on port {resolved_port}")
