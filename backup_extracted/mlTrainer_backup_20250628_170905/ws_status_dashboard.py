
import streamlit as st
import json
import os
import subprocess

st.set_page_config(page_title="🩺 WebSocket Health Dashboard", layout="wide")

st.title("🩺 WebSocket Health Dashboard")

# Real-time status check
col1, col2, col3 = st.columns(3)

with col1:
    # WebSocket Status
    if os.path.exists("websocket_status.json"):
        with open("websocket_status.json", "r") as f:
            status = json.load(f)

        if status.get("websocket_ok", False):
            st.success("✅ WebSocket Healthy")
        else:
            st.error("❌ WebSocket Failed")
            st.caption(f"Error: {status.get('result', 'Unknown')}")
    else:
        st.warning("⚠️ Status Unknown")

with col2:
    # Metrics
    if os.path.exists("ws_metrics.json"):
        with open("ws_metrics.json", "r") as f:
            metrics = json.load(f)

        st.metric("Total Checks", metrics.get("total_checks", 0))
        st.metric("Failures", metrics.get("failures", 0))
        st.metric("Restarts", metrics.get("restarts", 0))
    else:
        st.info("📊 No metrics available")

with col3:
    # Process Status
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        streamlit_processes = [
            line for line in result.stdout.split('\n')
            if 'streamlit' in line.lower() and 'grep' not in line
        ]

        if streamlit_processes:
            st.success(f"✅ {len(streamlit_processes)} Streamlit process(es)")
        else:
            st.error("❌ No Streamlit processes")
    except Exception:
        st.warning("⚠️ Cannot check processes")

# Recent health logs
st.subheader("📋 Recent Health Events")

if os.path.exists("health.log"):
    with open("health.log", "r") as f:
        lines = f.readlines()

    # Show last 10 lines
    recent_logs = lines[-10:] if len(lines) > 10 else lines

    for log_line in reversed(recent_logs):
        if log_line.strip():
            if "failure" in log_line.lower() or "error" in log_line.lower():
                st.error(log_line.strip())
            elif "restart" in log_line.lower():
                st.warning(log_line.strip())
            else:
                st.info(log_line.strip())
else:
    st.info("No health logs available")

# Quick actions
st.subheader("🔧 Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Test WebSocket Now"):
        from websocket_diagnostic import test_websocket
        success, result = test_websocket()
        if success:
            st.success(f"✅ Test passed: {result}")
        else:
            st.error(f"❌ Test failed: {result}")

with col2:
    if st.button("📊 Refresh Metrics"):
        st.rerun()

with col3:
    if st.button("🧹 Clear Logs"):
        for log_file in [
            "health.log",
            "ws_monitor.log",
                "websocket_status.json"]:
            if os.path.exists(log_file):
                os.remove(log_file)
        st.success("Logs cleared")
        st.rerun()

# Auto-refresh every 5 seconds
time.sleep(5)
st.rerun()
