
import subprocess
import time
import os


def kill_all_streamlit():
    """Kill all Streamlit processes"""
    try:
        subprocess.run(["pkill", "-f", "streamlit"], check=False)
        subprocess.run(["fuser", "-k", "8501/tcp"],
                       check=False, capture_output=True)
        print("✅ Cleaned up old Streamlit processes")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")


def start_streamlit_clean():
    """Start Streamlit with proper configuration"""
    kill_all_streamlit()
    time.sleep(2)

    # Set environment variables
    env = os.environ.copy()
    env.update({
        "STREAMLIT_SERVER_ADDRESS": "0.0.0.0",
        "STREAMLIT_SERVER_PORT": "8501",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_ENABLE_CORS": "false",
        "STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION": "false"
    })

    command = [
        "streamlit", "run", "interactive_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableWebsocketCompression", "false",
        "--server.allowRunOnSave", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ]

    print("🚀 Starting Streamlit with clean configuration...")
    process = subprocess.Popen(command, env=env)

    print(f"✅ Streamlit started with PID {process.pid}")
    print("🌐 Access at: http://localhost:8501")
    print("📊 WebSocket endpoint: ws://localhost:8501/_stcore/host-config")

    return process


if __name__ == "__main__":
    try:
        process = start_streamlit_clean()
        print("Press Ctrl+C to stop...")
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping Streamlit...")
        kill_all_streamlit()
