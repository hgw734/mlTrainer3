# port_socket_repair.py
import os
import subprocess
import time
import logging
import socket
import webbrowser
import psutil
import signal

CONFIG_PATH = ".streamlit/config.toml"
STREAMLIT_FILE = "interactive_app.py"
FLASK_FILE = "main.py"
FLASK_PORT = 8502
STREAMLIT_PORT = 8501

# ✅ Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - PortSocketRepair - %(levelname)s - %(message)s")
logger = logging.getLogger("PortSocketRepair")

def is_port_in_use(port):
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

def kill_processes_on_port(port):
    """Dynamically find and kill processes holding the specified port."""
    logger.info(f"🔍 Searching for processes using port {port}...")
    killed_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            connections = proc.info['connections']
            if connections:
                for conn in connections:
                    if conn.laddr.port == port:
                        logger.info(f"🎯 Found process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                        try:
                            os.kill(proc.info['pid'], signal.SIGTERM)
                            time.sleep(1)
                            # Force kill if still running
                            if psutil.pid_exists(proc.info['pid']):
                                os.kill(proc.info['pid'], signal.SIGKILL)
                            logger.info(f"✅ Killed process {proc.info['pid']}")
                            killed_count += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Could not kill process {proc.info['pid']}: {e}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    if killed_count == 0:
        logger.info(f"ℹ️ No processes found using port {port}")
    return killed_count

def patch_config_toml():
    """Write Streamlit config with safe CORS/XSRF settings."""
    logger.info("🛠️ Patching .streamlit/config.toml...")
    os.makedirs(".streamlit", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        f.write(f"""
[server]
port = {STREAMLIT_PORT}
enableCORS = false
enableXsrfProtection = false
headless = true

[browser]
gatherUsageStats = false
""")
    logger.info("✅ Patched config.toml with safe settings")

def suppress_tf_warnings():
    """Suppress TensorFlow GPU messages."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def launch_flask():
    """Launch Flask backend with absolute path."""
    flask_path = os.path.abspath(FLASK_FILE)
    logger.info(f"🚀 Launching Flask backend on port {FLASK_PORT} from {flask_path}...")
    
    if not os.path.exists(flask_path):
        logger.error(f"❌ Flask file not found: {flask_path}")
        return None
    
    os.environ["FLASK_RUN_PORT"] = str(FLASK_PORT)
    try:
        proc = subprocess.Popen(
            ["python3", flask_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"✅ Flask started with PID: {proc.pid}")
        return proc
    except Exception as e:
        logger.error(f"❌ Failed to start Flask: {e}")
        return None

def launch_streamlit():
    """Launch Streamlit app with absolute path and comprehensive logging."""
    streamlit_path = os.path.abspath(STREAMLIT_FILE)
    logger.info(f"🚀 Launching Streamlit from {streamlit_path} on port {STREAMLIT_PORT}...")
    
    if not os.path.exists(streamlit_path):
        logger.error(f"❌ Streamlit file not found: {streamlit_path}")
        return None
    
    try:
        # Launch Streamlit with detailed logging
        proc = subprocess.Popen(
            ["streamlit", "run", streamlit_path, "--server.port", str(STREAMLIT_PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        logger.info(f"✅ Streamlit started with PID: {proc.pid}")
        
        # Log first few lines of output to verify success
        time.sleep(3)
        if proc.poll() is None:  # Process still running
            logger.info("📊 Streamlit process is running, checking output...")
            # Read some initial output
            try:
                output_lines = []
                for _ in range(10):  # Read up to 10 lines
                    line = proc.stdout.readline()
                    if line:
                        output_lines.append(line.strip())
                        logger.info(f"📝 Streamlit: {line.strip()}")
                    else:
                        break
                
                # Check for success indicators
                output_text = "\n".join(output_lines)
                if f":{STREAMLIT_PORT}" in output_text or "You can now view" in output_text:
                    logger.info("✅ Streamlit successfully started and accessible!")
                else:
                    logger.warning("⚠️ Streamlit output doesn't show expected success messages")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not read Streamlit output: {e}")
        else:
            logger.error(f"❌ Streamlit process terminated early with code: {proc.returncode}")
            if proc.stdout:
                error_output = proc.stdout.read()
                logger.error(f"❌ Streamlit error output: {error_output}")
                
        return proc
        
    except Exception as e:
        logger.error(f"❌ Failed to start Streamlit: {e}")
        return None

def open_browser():
    """Open Streamlit in default browser or iframe."""
    try:
        url = f"http://localhost:{STREAMLIT_PORT}"
        logger.info(f"🌐 Opening Streamlit at {url}...")
        webbrowser.open(url)
    except Exception as e:
        logger.warning(f"⚠️ Could not open browser automatically: {e}")

def main():
    logger.info("🎯 Enhanced Port Socket Repair & Launch System Activated")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"📄 Target files: Flask={os.path.abspath(FLASK_FILE)}, Streamlit={os.path.abspath(STREAMLIT_FILE)}")
    
    suppress_tf_warnings()
    patch_config_toml()

    # Dynamically kill processes on required ports
    for port in [STREAMLIT_PORT, FLASK_PORT]:
        if is_port_in_use(port):
            logger.warning(f"⚠️ Port {port} is in use, attempting cleanup...")
            killed = kill_processes_on_port(port)
            time.sleep(2)  # Allow time for cleanup
            
            # Verify port is now free
            if is_port_in_use(port):
                logger.error(f"❌ Port {port} still in use after cleanup attempt")
            else:
                logger.info(f"✅ Port {port} is now available")

    # Launch backend and frontend with error handling
    logger.info("🚀 Starting service launch sequence...")
    
    flask_proc = launch_flask()
    time.sleep(3)  # Allow Flask to start
    
    streamlit_proc = launch_streamlit()
    time.sleep(2)  # Allow Streamlit to start
    
    # Final verification
    if is_port_in_use(FLASK_PORT):
        logger.info("✅ Flask backend is responding on port 8502")
    else:
        logger.warning("⚠️ Flask backend may not be running properly")
        
    if is_port_in_use(STREAMLIT_PORT):
        logger.info("✅ Streamlit frontend is responding on port 8501")
        logger.info("🌐 mlTrainer Interactive Chat should be available in Preview tab")
        
        # Try to open browser
        open_browser()
        
        # Launch interactive_app.py specifically
        logger.info(f"🎯 Launching {STREAMLIT_FILE} directly...")
        try:
            direct_launch = subprocess.Popen(["python3", os.path.abspath(STREAMLIT_FILE)])
            logger.info(f"✅ Direct launch of {STREAMLIT_FILE} started with PID: {direct_launch.pid}")
        except Exception as e:
            logger.warning(f"⚠️ Could not directly launch {STREAMLIT_FILE}: {e}")
            
    else:
        logger.error("❌ Streamlit frontend is not responding properly")

    logger.info("🎉 Enhanced launcher sequence completed!")
    logger.info("📱 Access your mlTrainer system via the Preview tab or the URLs shown above")

if __name__ == "__main__":
    main()
