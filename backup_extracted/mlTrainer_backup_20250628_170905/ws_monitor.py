
# ws_monitor.py

import time
import subprocess
import os
import logging
import websocket
import ssl
import signal
import json
from datetime import datetime

# === CONFIG ===
STREAMLIT_PORT = 8501
WS_PATH = "/_stcore/host-config"
CHECK_INTERVAL = 10  # seconds
STREAMLIT_COMMAND = [
    "streamlit", "run", "interactive_app.py",
    "--server.port", str(STREAMLIT_PORT),
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableWebsocketCompression", "false",
    "--server.allowRunOnSave", "false",
    "--server.enableXsrfProtection", "false"
]

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ws_monitor.log")
    ]
)

# Health logger for timestamped failures
health_logger = logging.getLogger('health')
health_handler = logging.FileHandler("health.log")
health_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
health_logger.addHandler(health_handler)
health_logger.setLevel(logging.INFO)


def test_websocket(host="0.0.0.0", port=8501, path="/_stcore/host-config"):
    """Test WebSocket connection to Streamlit"""
    url = f"ws://{host}:{port}{path}"
    try:
        ws = websocket.create_connection(
            url,
            timeout=5,
            sslopt={"cert_reqs": ssl.CERT_NONE},
            header={"Origin": f"http://{host}:{port}"}
        )
        ws.send("ping")
        response = ws.recv()
        ws.close()
        return True, f"Response: {response}"
    except websocket.WebSocketBadStatusException as e:
        return False, f"HTTP {e.status_code}: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def kill_port(port: int):
    """Kill all processes using the specified port"""
    try:
        # Try fuser first (more reliable)
        subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)

        # Also try lsof as backup
        try:
            output = subprocess.check_output(
                f"lsof -t -i:{port}", shell=True).decode().strip()
            for pid in output.splitlines():
                if pid.strip():
                    logging.warning(f"⚠️ Killing process {pid} on port {port}")
                    os.kill(int(pid), signal.SIGKILL)
        except subprocess.CalledProcessError:
            pass

        # Kill any streamlit processes
        subprocess.run("pkill -f streamlit", shell=True, capture_output=True)

        logging.info(f"🧹 Cleaned up port {port}")

    except Exception as e:
        logging.error(f"❌ Error cleaning port {port}: {e}")


def check_port_available(port: int):
    """Check if port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("0.0.0.0", port)) != 0


def restart_streamlit():
    """Restart Streamlit with proper cleanup"""
    logging.info("🔄 Restarting Streamlit...")
    increment_restart_counter()
    health_logger.warning("Streamlit restart initiated")

    # Kill existing processes
    kill_port(STREAMLIT_PORT)
    time.sleep(2)

    # Set environment variables
    env = os.environ.copy()
    env.update({
        "STREAMLIT_DISABLE_EMAIL_COLLECTION": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_SERVER_ADDRESS": "0.0.0.0",
        "STREAMLIT_SERVER_PORT": str(STREAMLIT_PORT)
    })

    # Start Streamlit
    try:
        process = subprocess.Popen(
            STREAMLIT_COMMAND,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )

        logging.info(f"🚀 Started Streamlit with PID {process.pid}")

        # Wait for startup
        time.sleep(8)

        # Check if process is still alive
        if process.poll() is None:
            logging.info("✅ Streamlit process is running")
        else:
            stdout, stderr = process.communicate()
            logging.error(f"❌ Streamlit died immediately:")
            logging.error(f"STDOUT: {stdout.decode()}")
            logging.error(f"STDERR: {stderr.decode()}")

    except Exception as e:
        logging.error(f"❌ Failed to start Streamlit: {e}")


def log_status(success: bool, result: str):
    """Log status to file for debugging with enhanced metrics"""
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "websocket_ok": success,
        "result": result,
        "port": STREAMLIT_PORT,
        "uptime_seconds": time.time() -
        start_time if 'start_time' in globals() else 0}

    # Log to health.log for timestamped failures
    if not success:
        health_logger.error(f"WebSocket failure: {result}")
    else:
        health_logger.info("WebSocket healthy")

    try:
        with open("websocket_status.json", "w") as f:
            json.dump(status, f, indent=2)

        # Update metrics file
        update_metrics(success)
    except Exception:
        pass  # Don't fail monitor if logging fails


def update_metrics(success: bool):
    """Update metrics tracking"""
    try:
        metrics_file = "ws_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {
                "total_checks": 0,
                "failures": 0,
                "restarts": 0,
                "uptime_start": datetime.utcnow().isoformat()
            }

        metrics["total_checks"] += 1
        if not success:
            metrics["failures"] += 1
        metrics["last_check"] = datetime.utcnow().isoformat()

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

    except Exception as e:
        logging.error(f"Failed to update metrics: {e}")


def increment_restart_counter():
    """Track number of restarts"""
    try:
        metrics_file = "ws_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            metrics["restarts"] += 1
        else:
            metrics = {"restarts": 1}

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass


def monitor_loop():
    """Main monitoring loop"""
    global start_time
    start_time = time.time()

    logging.info("🩺 Starting WebSocket monitor for Streamlit...")
    logging.info(
        f"🔍 Monitoring port {STREAMLIT_PORT} every {CHECK_INTERVAL} seconds")
    health_logger.info("WebSocket monitor started")

    consecutive_failures = 0

    while True:
        try:
            success, result = test_websocket(port=STREAMLIT_PORT, path=WS_PATH)
            log_status(success, result)

            if success:
                logging.info("✅ WebSocket OK")
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                logging.error(
                    f"❌ WebSocket FAILED ({consecutive_failures}x): {result}")

                # Only restart after multiple failures to avoid flapping
                if consecutive_failures >= 2:
                    restart_streamlit()
                    consecutive_failures = 0

        except KeyboardInterrupt:
            logging.info("🛑 Monitor stopped by user")
            break
        except Exception as e:
            logging.error(f"❌ Monitor error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    monitor_loop()
