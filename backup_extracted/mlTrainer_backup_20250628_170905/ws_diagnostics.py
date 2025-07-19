import socket
import os


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('0.0.0.0', port))
        return result != 0


def find_open_port(start=3000, end=3100):
    for port in range(start, end):
        if check_port(port):
            return port
    return 3000


port = find_open_port()
os.environ["STREAMLIT_SERVER_PORT"] = str(port)
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

print(f"✅ Streamlit will use port {port}")
