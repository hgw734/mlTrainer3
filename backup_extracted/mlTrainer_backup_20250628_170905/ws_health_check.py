
import subprocess
import socket
import requests
import json
from datetime import datetime


def check_streamlit_process():
    """Check if Streamlit is running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        streamlit_procs = [line for line in result.stdout.split(
            '\n') if 'streamlit' in line.lower() and 'grep' not in line]
        return len(streamlit_procs) > 0, streamlit_procs
    except Exception as e:
        return False, [f"Error checking processes: {e}"]


def check_port_status(port):
    """Check if port is accessible"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('0.0.0.0', port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_streamlit_health():
    """Check Streamlit health endpoint"""
    try:
        response = requests.get(
            'http://0.0.0.0:3000/_stcore/health', timeout=5)
        return response.status_code == 200, response.status_code
    except Exception as e:
        return False, str(e)


def check_websocket_connection():
    """Test WebSocket connection"""
    try:
        import websocket
        ws = websocket.create_connection(
            "ws://0.0.0.0:3000/_stcore/host-config",
            timeout=5
        )
        ws.send("ping")
        response = ws.recv()
        ws.close()
        return True, response
    except Exception as e:
        return False, str(e)


def main():
    print(f"🔍 WebSocket Health Check - {datetime.now()}")
    print("=" * 60)

    # Check Streamlit process
    is_running, procs = check_streamlit_process()
    print(f"{'✅' if is_running else '❌'} Streamlit Process: {'Running' if is_running else 'Not Found'}")
    if is_running:
        for proc in procs:
            print(f"   {proc}")

    # Check ports
    for port in [3000, 8501, 5000]:
        accessible = check_port_status(port)
        print(f"{'✅' if accessible else '❌'} Port {port}: {'Accessible' if accessible else 'Not Accessible'}")

    # Check Streamlit health
    healthy, status = check_streamlit_health()
    print(f"{'✅' if healthy else '❌'} Streamlit Health: {status}")

    # Check WebSocket
    ws_ok, ws_result = check_websocket_connection()
    print(f"{'✅' if ws_ok else '❌'} WebSocket: {ws_result}")

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "streamlit_running": is_running,
        "ports_accessible": {
            "3000": check_port_status(3000),
            "8501": check_port_status(8501),
            "5000": check_port_status(5000)
        },
        "streamlit_healthy": healthy,
        "websocket_ok": ws_ok,
        "recommendations": []
    }

    if not is_running:
        report["recommendations"].append("Start Streamlit service")
    if not ws_ok:
        report["recommendations"].append("Fix WebSocket configuration")

    with open("health_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Full report saved to health_report.json")


if __name__ == "__main__":
    main()
