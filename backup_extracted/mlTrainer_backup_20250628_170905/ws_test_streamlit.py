
import streamlit as st
import subprocess
import sys
import os
from websocket_diagnostic import test_websocket, test_multiple_endpoints, check_streamlit_process

st.set_page_config(
    page_title="🔍 WebSocket Diagnostic",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 WebSocket Diagnostic Tool for mlTrainer")

# Sidebar with system info
with st.sidebar:
    st.header("🔧 System Status")

    # Check if websocket-client is installed
    try:
        st.success("✅ websocket-client installed")
    except ImportError:
        st.error("❌ websocket-client not installed")
        if st.button("Install websocket-client"):
            with st.spinner("Installing..."):
                subprocess.run([sys.executable, "-m", "pip",
                               "install", "websocket-client"])
                st.rerun()

    # Environment info
    st.subheader("Environment")
    st.text(f"Python: {sys.version.split()[0]}")
    st.text(f"Working Dir: {os.getcwd()}")

    # Port info
    if os.path.exists("port_config.json"):
        import json
        with open("port_config.json", "r") as f:
            port_config = json.load(f)
        st.json(port_config)

# Main diagnostic area
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 WebSocket Connection Test")

    host = st.text_input("Host", value="0.0.0.0")
    port = st.number_input("Port", value=8501, min_value=1, max_value=65535)
    path = st.text_input("WebSocket Path", value="/_stcore/host-config")

    if st.button("Test WebSocket Connection", type="primary"):
        with st.spinner("Testing connection..."):
            success, result = test_websocket(host=host, port=port, path=path)

        if success:
            st.success("✅ WebSocket connection succeeded!")
            st.code(result, language="text")
        else:
            st.error("❌ WebSocket connection failed")
            st.code(result, language="text")

            # Provide specific troubleshooting
            if "Connection refused" in result:
                st.warning(
                    "🔧 **Troubleshooting**: Port may not be open. Check if Streamlit is running.")
            elif "502" in result:
                st.warning(
                    "🔧 **Troubleshooting**: HTTP 502 indicates a proxy/gateway issue. This is likely a Replit configuration problem.")
            elif "timeout" in result.lower():
                st.warning(
                    "🔧 **Troubleshooting**: Connection timeout. WebSocket upgrade may be blocked.")

with col2:
    st.subheader("🔍 Process Status")

    if st.button("Check Streamlit Processes"):
        with st.spinner("Checking processes..."):
            process_running = check_streamlit_process()

        if process_running:
            st.success("✅ Streamlit processes are running")
        elif process_running is False:
            st.error("❌ No Streamlit processes found")
            st.info(
                "Try running: `streamlit run interactive_app.py --server.port=8501 --server.address=0.0.0.0`")
        else:
            st.warning("⚠️ Could not determine process status")

# Multiple endpoint testing
st.subheader("🔍 Multiple Endpoint Test")

if st.button("Test All WebSocket Endpoints"):
    with st.spinner("Testing multiple endpoints..."):
        results = test_multiple_endpoints()

    # Display results in a nice table
    import pandas as pd

    data = []
    for name, info in results.items():
        data.append({
            "Endpoint": name,
            "Status": "✅ Success" if info["success"] else "❌ Failed",
            "Result": info["result"]
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

# JavaScript browser test
st.subheader("🌐 Browser WebSocket Test")

st.markdown("""
**Use this JavaScript in your browser's DevTools console to test WebSocket from the frontend:**
""")

js_code = f"""
// Test WebSocket connection from browser
const ws = new WebSocket("ws://{st.text_input('Browser Test Host', value='localhost')}:8501/_stcore/host-config");

ws.onopen = () => {{
    console.log("✅ WebSocket connected successfully");
    ws.send("ping");
}};

ws.onmessage = (event) => {{
    console.log("📨 WebSocket message received:", event.data);
}};

ws.onerror = (error) => {{
    console.error("❌ WebSocket error:", error);
}};

ws.onclose = (event) => {{
    console.log("🔌 WebSocket closed:", event.code, event.reason);
}};
"""

st.code(js_code, language="javascript")

# Quick fix buttons
st.subheader("🛠️ Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Kill Streamlit Processes"):
        with st.spinner("Killing processes..."):
            subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
        st.success("Streamlit processes terminated")

with col2:
    if st.button("Clear Port 8501"):
        with st.spinner("Clearing port..."):
            subprocess.run(["fuser", "-k", "8501/tcp"], capture_output=True)
        st.success("Port 8501 cleared")

with col3:
    if st.button("Restart System"):
        st.info("Run this command in terminal: `bash start_system.sh`")

# Error log analysis
if st.expander("📋 Error Log Analysis"):
    log_files = ["streamlit.log", "flask.log"]

    for log_file in log_files:
        if os.path.exists(log_file):
            st.subheader(f"📄 {log_file}")
            with open(log_file, "r") as f:
                log_content = f.read()

            # Show last 20 lines
            lines = log_content.split('\n')
            recent_lines = lines[-20:] if len(lines) > 20 else lines

            st.code('\n'.join(recent_lines), language="text")
        else:
            st.warning(f"📄 {log_file} not found")
