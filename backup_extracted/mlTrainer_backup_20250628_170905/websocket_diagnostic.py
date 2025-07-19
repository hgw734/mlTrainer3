
import websocket
import ssl
import socket
from datetime import datetime


def test_websocket(host="0.0.0.0", port=8501, path="/_stcore/host-config"):
    """Test WebSocket connection to Streamlit backend"""
    url = f"ws://{host}:{port}{path}"
    print(f"🔍 Testing WebSocket connection to: {url}")

    try:
        # First check if port is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()

        if result != 0:
            return False, f"Port {port} is not accessible on {host}"

        # Try WebSocket connection
        ws = websocket.create_connection(
            url,
            timeout=10,
            sslopt={"cert_reqs": ssl.CERT_NONE},
            header={"Origin": f"http://{host}:{port}"}
        )

        # Send a ping
        ws.send("ping")
        response = ws.recv()
        ws.close()

        return True, f"Connection successful - Response: {response}"

    except websocket.WebSocketBadStatusException as e:
        return False, f"HTTP Status Error: {e.status_code} - {e}"
    except websocket.WebSocketTimeoutException:
        return False, "Connection timeout - WebSocket handshake failed"
    except ConnectionRefusedError:
        return False, f"Connection refused - No service listening on {host}:{port}"
    except Exception as e:
        return False, f"WebSocket error: {type(e).__name__}: {e}"


def test_multiple_endpoints():
    """Test multiple WebSocket endpoints"""
    endpoints = [
        ("/_stcore/host-config", "Host Config"),
        ("/_stcore/health", "Health Check"),
        ("/stream", "Stream"),
        ("/_stcore/stream", "Core Stream")
    ]

    print("🔍 Testing multiple WebSocket endpoints...")
    results = {}

    for path, name in endpoints:
        success, result = test_websocket(path=path)
        results[name] = {"success": success, "result": result}
        status = "✅" if success else "❌"
        print(f"{status} {name} ({path}): {result}")

    return results


def check_streamlit_process():
    """Check if Streamlit process is running"""
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        processes = result.stdout

        streamlit_processes = [
            line for line in processes.split('\n')
            if 'streamlit' in line.lower() and 'grep' not in line
        ]

        if streamlit_processes:
            print("✅ Streamlit processes found:")
            for proc in streamlit_processes:
                print(f"   {proc}")
        else:
            print("❌ No Streamlit processes found")

        return len(streamlit_processes) > 0

    except Exception as e:
        print(f"⚠️ Could not check processes: {e}")
        return None


def main():
    print(
        f"🛠️ WebSocket Diagnostic Tool - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check if Streamlit is running
    check_streamlit_process()
    print()

    # Test basic WebSocket connection
    success, result = test_websocket()
    if success:
        print("✅ Basic WebSocket test PASSED")
        print(f"   Result: {result}")
    else:
        print("❌ Basic WebSocket test FAILED")
        print(f"   Error: {result}")

    print()

    # Test multiple endpoints
    test_multiple_endpoints()

    print("\n" + "=" * 60)
    print("🔧 Troubleshooting Tips:")
    print("• If port not accessible: Check if Streamlit is running")
    print("• If connection refused: Verify Streamlit is bound to 0.0.0.0")
    print("• If HTTP 502: Proxy/gateway issue - check Replit configuration")
    print("• If timeout: WebSocket upgrade may be blocked")


if __name__ == "__main__":
    main()
