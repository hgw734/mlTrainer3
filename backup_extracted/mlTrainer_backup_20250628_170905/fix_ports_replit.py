
import os
import socket
import logging
import subprocess
import time

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "port_diagnostics.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s — %(message)s')
log = logging.getLogger()

# Define separate port ranges to prevent collision
FLASK_PORT_RANGE = range(5000, 5050)  # Flask gets 5000-5049
STREAMLIT_PORT_RANGE = range(8501, 8551)  # Streamlit gets 8501-8550

def cleanup_processes():
    """Kill any existing processes on common ports"""
    log.info("🧼 Cleaning up existing processes...")
    common_ports = [3000, 5000, 8501, 8080]
    for port in common_ports:
        subprocess.run(f"fuser -k {port}/tcp 2>/dev/null || true", shell=True)
    
    # Kill specific processes
    subprocess.run("pkill -f streamlit 2>/dev/null || true", shell=True)
    subprocess.run("pkill -f flask 2>/dev/null || true", shell=True)
    subprocess.run("pkill -f 'python.*main.py' 2>/dev/null || true", shell=True)

def is_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        result = sock.connect_ex(("0.0.0.0", port))
        return result != 0  # True if port is free

def find_flask_port():
    """Find available port for Flask backend"""
    log.info("🔍 Scanning for Flask port...")
    for port in FLASK_PORT_RANGE:
        if is_port_available(port):
            log.info(f"✅ Found Flask port: {port}")
            return port
    
    log.error("❌ No Flask port found in range.")
    raise RuntimeError("No available Flask port found.")

def find_streamlit_port():
    """Find available port for Streamlit frontend"""
    log.info("🔍 Scanning for Streamlit port...")
    for port in STREAMLIT_PORT_RANGE:
        if is_port_available(port):
            log.info(f"✅ Found Streamlit port: {port}")
            return port
    
    log.error("❌ No Streamlit port found in range.")
    raise RuntimeError("No available Streamlit port found.")

def fix_port_conflicts():
    """Main function to resolve all port conflicts with separate ports"""
    log.info("🛠️ Starting Replit port conflict resolution...")
    
    # Step 1: Clean up existing processes
    cleanup_processes()
    time.sleep(2)
    
    # Step 2: Find separate ports for Flask and Streamlit
    flask_port = find_flask_port()
    streamlit_port = find_streamlit_port()
    
    # Step 3: Set Flask environment variables
    flask_env_vars = {
        "FLASK_RUN_PORT": str(flask_port),
        "APP_PORT": str(flask_port),
        "FLASK_PORT": str(flask_port)
    }
    
    for var, value in flask_env_vars.items():
        os.environ[var] = value
        log.info(f"🔧 Set {var}={value}")
    
    # Step 4: Set Streamlit environment variables
    streamlit_env_vars = {
        "STREAMLIT_SERVER_PORT": str(streamlit_port),
        "STREAMLIT_SERVER_ADDRESS": "0.0.0.0",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_SERVER_ENABLE_CORS": "false",
        "STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION": "false",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_GLOBAL_DISABLE_WATCHDOG_WARNING": "true"
    }
    
    for key, value in streamlit_env_vars.items():
        os.environ[key] = value
        log.info(f"🔧 Set {key}={value}")
    
    # Step 5: Create port configuration file for other components
    port_config = {
        "flask_port": flask_port,
        "streamlit_port": streamlit_port,
        "main_interface": streamlit_port,  # Streamlit is the main UI
        "backend_api": flask_port,
        "timestamp": time.time()
    }
    
    with open("port_config.json", "w") as f:
        import json
        json.dump(port_config, f, indent=2)
    
    log.info(f"✅ Flask configured for port {flask_port}")
    log.info(f"✅ Streamlit configured for port {streamlit_port}")
    print(f"[✔] Flask backend: http://0.0.0.0:{flask_port}")
    print(f"[✔] Streamlit frontend: http://0.0.0.0:{streamlit_port}")
    print(f"[✔] Main interface: {streamlit_port}")
    
    return {"flask_port": flask_port, "streamlit_port": streamlit_port}

if __name__ == "__main__":
    try:
        ports = fix_port_conflicts()
        print(f"✅ Port resolution complete:")
        print(f"  - Flask: {ports['flask_port']}")
        print(f"  - Streamlit: {ports['streamlit_port']}")
    except Exception as e:
        log.error(f"❌ Port fix failed: {e}")
        print(f"[✖] Failed to fix port issues: {e}")
        exit(1)
